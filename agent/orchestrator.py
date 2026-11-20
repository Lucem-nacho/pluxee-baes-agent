"""Orquestador del flujo completo del agente — versión EP2 (AgentExecutor).

Qué cambia respecto a EP1 y qué se mantiene intacto:

1. `agent.classifier.clasificar(...)` sigue siendo el primer filtro (el
   código no cambió; solo se amplió `prompts/clasificador.py` para que las
   preguntas sobre el historial propio del estudiante cuenten como
   INFORMATIVA — antes caían en ESCALAR y la memoria de largo plazo era
   inalcanzable, hallazgo de la primera corrida real). Se mantiene la
   decisión de diseño de EP1 ("clasificador antes del RAG"): evita invocar el ciclo completo de tool-calling del
   agente — varias llamadas al LLM, no una sola — para consultas claramente
   fuera de alcance (reclamos, acciones de cuenta).
2. Si ESCALAR -> `memory.store.registrar_escalamiento(...)` directo, igual
   que en EP1 pero ahora persistido en SQLite en vez de perderse en un dict
   (EP2, escritura — ver `memory/store.py`).
3. Si INFORMATIVA -> se arma un `AgentExecutor` de LangChain (EP2, framework
   de agentes) con las 4 tools de `agent/tools.py`. El LLM decide él mismo
   qué herramientas llamar y en qué orden — a diferencia de EP1, donde el
   camino era fijo (recuperar -> comercios -> arbitrar -> generar, siempre
   en ese orden, llamaran o no la pregunta). Esto es lo que permite
   demostrar planificación y toma de decisiones adaptativas (EP2, IE5/IE7):
   el mismo agente puede terminar escalando a mitad de camino si lo que
   encuentra lo amerita (ver `escalar_a_ejecutivo` en `agent/tools.py`),
   sin que eso estuviera decidido de antemano por el clasificador.
4. Toda interacción resuelta (por cualquiera de los dos caminos) se
   registra en la memoria de largo plazo (`memory.store.registrar_interaccion`,
   EP2 IE3/IE4) antes de devolver el resultado a `app.py`.
5. (EP3) Cada llamada a `responder` queda registrada en la base de
   observabilidad (`observabilidad/`), incluidas las que terminan en
   excepción. Es solo medición: no cambia lo que `responder` devuelve ni lo
   que lanza.

`agent/generator.py` y la llamada directa a `rag/arbitration.py` de EP1 YA
NO se usan en el camino INFORMATIVA: su lógica fue absorbida por
`agent/tools.py` (el arbitraje sigue en uso, solo que ahora lo llama la
tool `buscar_normativa` en vez de este módulo directamente) y por el propio
system prompt del agente (`prompts/agente.py`), que reemplaza la llamada
directa a `generar_respuesta()` de EP1. Ambos archivos se mantienen en el
repositorio sin tocar — documentan el diseño de EP1 tal cual se construyó y
probó, no código muerto.

Diferencia de resiliencia frente a EP1, documentada como limitación
conocida: `llm/client.py` (usado por el clasificador) tiene manejo de
reintentos para 503/429 probado con llamadas reales a Gemini. El LLM del
AgentExecutor usa `ChatGoogleGenerativeAI` de `langchain-google-genai`
directamente, con su propio `max_retries` — no pasa por `llm/client.py`, así
que no hereda ese manejo específico de `retryDelay`. Unificar ambos caminos
bajo un solo wrapper de reintentos queda como mejora futura (ver
`docs/arquitectura.md`), igual que el validador de fidelidad en EP1.
"""

# AgentExecutor/create_tool_calling_agent se movieron a `langchain-classic`
# en LangChain 1.x (ya no existen en `langchain.agents`) — verificado al
# correr el código con la librería realmente instalada.
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.classifier import clasificar
from agent.tools import construir_tools
from config import settings
from memory.store import registrar_escalamiento, registrar_interaccion
from observabilidad import store as observabilidad_store
from observabilidad.callbacks import RegistroPasos
from observabilidad.medicion import Medicion
from prompts.agente import SYSTEM_PROMPT_AGENTE

MAX_ITERACIONES_AGENTE = 6  # tope de llamadas a tools por consulta: evita que
# el agente encadene herramientas indefinidamente y agote la cuota diaria de
# Gemini en una sola consulta mal planificada.

# AgentExecutor devuelve estos textos internos (dos variantes según la ruta
# de código de langchain-classic) cuando corta por límite de iteraciones, en
# vez de lanzar una excepción. Observado con Gemini real: ante una pregunta
# que el manual no cubre (ej. bebidas energéticas), el agente reformula la
# búsqueda en bucle hasta agotar las iteraciones.
PREFIJO_CORTE_POR_LIMITE = "Agent stopped due to"

# Misma conducta que la regla 1 del prompt (decirlo explícitamente y sugerir
# derivar), aplicada como red de seguridad cuando el agente no llega solo a
# esa conclusión. Nunca debe verse al estudiante ni guardarse en la memoria el
# texto interno de la librería.
MENSAJE_SIN_RESPUESTA = (
    "No logré confirmar esta información en el manual BAES ni en las demás "
    "fuentes que tengo disponibles, así que prefiero no darte una respuesta "
    "que podría ser incorrecta. Te sugiero consultarlo con un ejecutivo."
)


def responder(
    consulta_usuario: str,
    historial_contexto: str,
    id_usuario: str,
    origen: str = "app",
    *,
    caso_id: str | None = None,
    variante: str | None = None,
    repeticion: int | None = None,
) -> dict:
    """Punto de entrada único que usa `app.py`.

    `id_usuario` identifica la memoria de largo plazo del estudiante (ver
    limitación de autenticación documentada en `memory/store.py`) — NO
    reemplaza `historial_contexto`, que sigue siendo la memoria de corto
    plazo de la sesión actual en el navegador.

    `origen` ("app" por defecto, o "evaluacion") solo marca la ejecución en
    la base de observabilidad, para que los scripts de evaluación no se
    mezclen con el uso real. `caso_id`, `variante` y `repeticion` (opcionales)
    identifican la ejecución dentro de un conjunto de evaluación; el uso
    normal no los pasa.

    Devuelve `{"estado": "INFORMATIVA", "respuesta": ..., "pasos": [...]}`
    o `{"estado": "ESCALAR", ...}` — ambas formas comparten la clave
    `estado` para que `app.py` pueda ramificar sin conocer el detalle de
    cada una, igual que en EP1."""
    if origen not in observabilidad_store.ORIGENES:
        raise ValueError(f"origen inválido: {origen!r} (debe ser uno de {observabilidad_store.ORIGENES})")

    medicion = Medicion(
        origen=origen,
        id_usuario=id_usuario,
        consulta=consulta_usuario,
        caso_id=caso_id,
        variante=variante,
        repeticion=repeticion,
    )
    try:
        resultado = _responder(consulta_usuario, historial_contexto, id_usuario, medicion)
        medicion.registrar_resultado(resultado)
        return resultado
    except Exception as error:
        medicion.registrar_excepcion(error)
        raise
    finally:
        medicion.cerrar()


def _responder(
    consulta_usuario: str, historial_contexto: str, id_usuario: str, medicion: Medicion
) -> dict:
    with medicion.medir("clasificador"):
        clasificacion = clasificar(consulta_usuario, historial_contexto)
    medicion.categoria = clasificacion["categoria"]
    if clasificacion.get("no_interpretable"):
        medicion.tipo_error = "clasificador_no_interpretable"

    if clasificacion["categoria"] == "ESCALAR":
        resultado = registrar_escalamiento(
            consulta_usuario, historial_contexto, clasificacion["justificacion"]
        )
        registrar_interaccion(
            id_usuario, consulta_usuario, "[Derivado a ejecutivo humano]", "ESCALAR"
        )
        return resultado

    return _responder_informativa(consulta_usuario, historial_contexto, id_usuario, medicion)


def _responder_informativa(
    consulta_usuario: str, historial_contexto: str, id_usuario: str, medicion: Medicion
) -> dict:
    registro_escalamiento: dict = {}
    tools = construir_tools(id_usuario, consulta_usuario, historial_contexto, registro_escalamiento)

    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_AGENTE),
            (
                "human",
                "Historial de la conversación:\n{historial_contexto}\n\n"
                "Consulta del estudiante: {consulta_usuario}",
            ),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agente = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agente,
        tools=tools,
        max_iterations=MAX_ITERACIONES_AGENTE,
        return_intermediate_steps=True,
    )

    # El handler va en `config`, no en el constructor del AgentExecutor: los
    # callbacks del constructor no se heredan a las tools ni al modelo.
    medicion.registro_pasos = RegistroPasos()
    with medicion.medir("agente"):
        resultado = executor.invoke(
            {"consulta_usuario": consulta_usuario, "historial_contexto": historial_contexto},
            config={"callbacks": [medicion.registro_pasos]},
        )
    pasos = _resumir_pasos(resultado["intermediate_steps"])

    if registro_escalamiento:
        # Señal estructural (la tool fue llamada), no texto parseado del LLM
        # — ver docstring de agent/tools.py.
        registrar_interaccion(
            id_usuario, consulta_usuario, "[Derivado a ejecutivo humano]", "ESCALAR"
        )
        return {**registro_escalamiento, "estado": "ESCALAR", "pasos": pasos}

    respuesta_final = _texto_de_salida(resultado["output"])
    if not respuesta_final or respuesta_final.startswith(PREFIJO_CORTE_POR_LIMITE):
        medicion.tipo_error = "sin_respuesta" if not respuesta_final else "limite_iteraciones"
        respuesta_final = MENSAJE_SIN_RESPUESTA
    registrar_interaccion(id_usuario, consulta_usuario, respuesta_final, "INFORMATIVA")

    return {"estado": "INFORMATIVA", "respuesta": respuesta_final, "pasos": pasos}


def _texto_de_salida(output) -> str:
    """`resultado["output"]` no siempre es un `str`: con `langchain-google-genai`
    y Gemini, el contenido final del mensaje puede venir como lista de
    bloques (`[{"type": "text", "text": "..."}, ...]`). Se normaliza a texto
    plano antes de mostrarlo o guardarlo en SQLite (que rechaza listas)."""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, list):
        partes = []
        for bloque in output:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict) and bloque.get("type", "text") == "text":
                partes.append(bloque.get("text", ""))
        return "".join(partes).strip()
    return str(output).strip()


def _resumir_pasos(intermediate_steps) -> list[str]:
    """Lista legible de qué tools se llamaron y con qué argumentos — permite
    mostrar en la demo que el agente planifica su propio camino en vez de
    seguir uno fijo (EP2, IE7), sin exponer el texto completo de cada
    observación devuelta por las tools."""
    return [f"{accion.tool}({accion.tool_input})" for accion, _ in intermediate_steps]
