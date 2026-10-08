"""Tools de LangChain sobre las piezas RAG ya construidas en EP1.

Responsabilidad: envolver `rag/retrieval/vector_retriever.py`,
`rag/retrieval/comercios_query.py`, `rag/arbitration.py` y `memory/store.py`
como herramientas que un AgentExecutor de LangChain puede invocar — sin
reescribir la lógica de ninguna, solo dar la interfaz que el framework de
agentes espera (`@tool`: nombre, docstring como descripción que el LLM lee
para decidir cuándo usarla, y un único string de salida).

Dos de las cuatro tools (`escalar_a_ejecutivo` y `consultar_historial`)
necesitan contexto que el LLM NO debe decidir por su cuenta: a qué usuario
pertenece el historial, y cuál es la consulta/conversación real a derivar.
Si fueran parámetros normales del tool, el LLM tendría que "inventar" un
id_usuario o reconstruir el historial de memoria — tanto frágil como un
riesgo real de que una consulta maliciosa (vía inyección de instrucciones
en el propio texto de la consulta) le pida al agente leer el historial de
OTRO id_usuario. Por eso `construir_tools` es una fábrica: recibe el
contexto real de la sesión actual (desde `agent/orchestrator.py`, que a su
vez lo recibe de `app.py`) y devuelve las 4 tools ya atadas a ese contexto
— el LLM solo decide CUÁNDO llamarlas, nunca CON QUÉ usuario o historial.

Registrar la interacción resuelta en la memoria de largo plazo (tabla
`interacciones`) NO es una tool: es contabilidad que debe pasar siempre,
una vez, cuando el agente ya terminó de responder — no una decisión que el
LLM deba tomar a mitad de su razonamiento. Eso lo hace directamente
`agent/orchestrator.py` después de que el AgentExecutor devuelve su
resultado final.

`construir_tools` también recibe `registro_escalamiento` (un dict vacío que
`agent/orchestrator.py` crea antes de invocar al agente). Si el LLM llama a
`escalar_a_ejecutivo`, esa tool llena ese dict con el paquete completo que
`memory.store.registrar_escalamiento` devolvió (id, fecha, justificación).
Así el orquestador puede detectar la derivación DESPUÉS de que el
AgentExecutor termina, sin tener que re-parsear el texto de la respuesta
final del LLM ni adivinar sus intenciones — es una señal estructural, no
una heurística de texto como las de `agent/generator.py` en EP1."""

from langchain_core.tools import BaseTool, tool

from memory.store import consultar_historial_usuario, registrar_escalamiento
from rag.arbitration import resolver_contradicciones
from rag.retrieval.comercios_query import consultar_comercio
from rag.retrieval.vector_retriever import recuperar


def construir_tools(
    id_usuario: str,
    consulta_usuario: str,
    historial_contexto: str,
    registro_escalamiento: dict,
) -> list[BaseTool]:
    """Devuelve las 4 tools del agente, atadas al contexto de la sesión
    actual (`id_usuario`, la consulta que se está resolviendo ahora, y el
    historial completo de la conversación). `registro_escalamiento` debe
    ser un dict vacío provisto por quien llama — ver docstring del módulo."""

    @tool
    def buscar_normativa(consulta: str) -> str:
        """Busca en el manual BAES, la FAQ y las actualizaciones normativas
        por redes sociales. Úsala para cualquier pregunta sobre qué está
        permitido o prohibido comprar con la BAES, restricciones generales,
        o cómo funciona el beneficio. NO la uses para preguntar por un
        comercio específico por nombre — para eso usa buscar_comercio."""
        fragmentos = recuperar(consulta)
        vigente = resolver_contradicciones(fragmentos)["vigente"]
        return _formatear_fragmentos(vigente)

    @tool
    def buscar_comercio(nombre_o_categoria: str) -> str:
        """Busca un comercio específico por nombre (ej. "Jumbo", "Farmacias
        Cruz Verde") o por categoría (ej. "supermercado", "farmacia") en el
        listado de comercios asociados a la BAES. Devuelve sus restricciones
        particulares si existen. Si no encuentra nada, el comercio no está
        en el listado — eso NO significa que esté prohibido, solo que no
        hay una restricción particular registrada para él."""
        filas = consultar_comercio(nombre_o_categoria)
        if not filas:
            return f'Sin resultados para "{nombre_o_categoria}" en el listado de comercios asociados.'
        return "\n".join(
            f"- {f['nombre_comercio']} ({f['categoria']}, {f['comuna']}): "
            f"{f['restricciones']} [actualizado {f['fecha_actualizacion']}]"
            for f in filas
        )

    @tool
    def consultar_historial(limite: int = 5) -> str:
        """Consulta las últimas consultas que este mismo estudiante ya hizo
        en sesiones anteriores (memoria de largo plazo, distinta del
        historial de esta conversación). Úsala cuando la pregunta actual
        parezca relacionarse con algo que el estudiante ya preguntó antes,
        o cuando pregunte explícitamente algo como "¿qué te pregunté la
        última vez?"."""
        historial = consultar_historial_usuario(id_usuario, limite=limite)
        if not historial:
            return "Este estudiante no tiene interacciones registradas en sesiones anteriores."
        return "\n".join(
            f'- [{h["fecha"]}] Preguntó: "{h["consulta_usuario"]}" → Se respondió: "{h["respuesta"]}"'
            for h in historial
        )

    @tool
    def escalar_a_ejecutivo(justificacion: str) -> str:
        """Deriva la conversación completa a un ejecutivo humano, en vez de
        que tú sigas respondiendo. Úsala si, durante tu investigación,
        descubres que la consulta en realidad requiere una acción sobre la
        cuenta del estudiante, involucra un reclamo formal, o las fuentes
        consultadas se contradicen de forma que no puedes resolver con
        confianza — aunque el clasificador inicial haya decidido que era
        informativa. Después de llamar esta herramienta, tu única respuesta
        final debe ser avisarle al estudiante que su consulta fue derivada;
        no intentes responder la pregunta original."""
        resultado = registrar_escalamiento(
            consulta_usuario=consulta_usuario,
            historial_contexto=historial_contexto,
            justificacion=justificacion,
        )
        registro_escalamiento.update(resultado)
        return (
            f"Derivado a ejecutivo humano (registro #{resultado['id_escalamiento']}). "
            "Informa al estudiante que un ejecutivo revisará su conversación a la brevedad."
        )

    return [buscar_normativa, buscar_comercio, consultar_historial, escalar_a_ejecutivo]


def _formatear_fragmentos(fragmentos: list[dict]) -> str:
    if not fragmentos:
        return "Sin resultados en el manual, FAQ o actualizaciones normativas para esta consulta."
    return "\n\n".join(
        f'[Fuente: {f["fuente"]} | Fecha: {f["fecha"]}]\n{f["texto"]}' for f in fragmentos
    )
