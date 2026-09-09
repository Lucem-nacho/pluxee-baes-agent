"""Orquestador del flujo completo del agente.

Cableado entre piezas ya probadas por separado — no contiene lógica de LLM
propia (eso vive en `agent/classifier.py` y `agent/generator.py`):

1. `agent.classifier.clasificar(...)` sobre la consulta + historial.
2. Si ESCALAR -> `agent.escalation.derivar(...)`.
3. Si INFORMATIVA ->
   a. `rag.retrieval.vector_retriever.recuperar(...)` sobre la colección
      vectorizada (manual + FAQ + resolución).
   b. `rag.retrieval.comercios_query.consultar_comercio(...)` — heurística
      simple (no NLP): prueba cada palabra de >=4 letras de la consulta
      contra la tabla de comercios. Es barata y suficiente para este alcance
      porque el dataset es de 15 filas; con un catálogo real de comercios
      esto necesitaría reemplazarse por una extracción de entidades real.
   c. `rag.arbitration.resolver_contradicciones(...)` sobre TODOS los
      fragmentos juntos (vectoriales + comercios) — arbitra por metadata
      fuente/fecha sin importar de dónde vino cada uno, tal como está
      diseñado el módulo.
   d. `agent.generator.generar(...)` con el contexto ya arbitrado.
"""

from agent.classifier import clasificar
from agent.escalation import derivar
from agent.generator import generar
from rag.arbitration import resolver_contradicciones
from rag.retrieval.comercios_query import consultar_comercio
from rag.retrieval.vector_retriever import recuperar

FUENTE_COMERCIOS = "Listado de comercios asociados"
LARGO_MINIMO_PALABRA_COMERCIO = 4


def responder(consulta_usuario: str, historial_contexto: str) -> dict:
    """Punto de entrada único que usa `app.py`. Devuelve la respuesta final
    ({"estado": "INFORMATIVA", "respuesta": ..., "advertencias": [...]})
    o el paquete de derivación de `agent.escalation.derivar`
    ({"estado": "ESCALAR", ...}) — ambas formas comparten la clave `estado`
    para que `app.py` pueda ramificar sin conocer el detalle de cada una."""
    clasificacion = clasificar(consulta_usuario, historial_contexto)

    if clasificacion["categoria"] == "ESCALAR":
        return derivar(consulta_usuario, historial_contexto, clasificacion["justificacion"])

    return _responder_informativa(consulta_usuario)


def _responder_informativa(consulta_usuario: str) -> dict:
    fragmentos = recuperar(consulta_usuario)
    fragmentos += [
        _comercio_a_fragmento(fila) for fila in _buscar_comercios_mencionados(consulta_usuario)
    ]

    contexto = resolver_contradicciones(fragmentos)["vigente"]
    resultado = generar(consulta_usuario, contexto)

    return {
        "estado": "INFORMATIVA",
        "respuesta": resultado["respuesta"],
        "advertencias": resultado["advertencias"],
    }


def _buscar_comercios_mencionados(consulta_usuario: str) -> list[dict]:
    """Heurística de "cableado", no NLP: prueba cada palabra relevante de la
    consulta contra `consultar_comercio` (LIKE sobre nombre/categoría) y
    devuelve la unión sin duplicar por id. Suficiente para el dataset de 15
    comercios; no escala a un catálogo real sin una extracción de entidades."""
    palabras = {
        palabra
        for palabra_cruda in consulta_usuario.split()
        if len(palabra := palabra_cruda.strip(".,;:!?¿¡")) >= LARGO_MINIMO_PALABRA_COMERCIO
    }

    encontrados: dict[int, dict] = {}
    for palabra in palabras:
        for fila in consultar_comercio(palabra):
            encontrados[fila["id"]] = fila
    return list(encontrados.values())


def _comercio_a_fragmento(fila: dict) -> dict:
    return {
        "texto": (
            f"Comercio: {fila['nombre_comercio']} ({fila['categoria']}, {fila['comuna']}). "
            f"Restricciones: {fila['restricciones']}."
        ),
        "fuente": FUENTE_COMERCIOS,
        "fecha": fila["fecha_actualizacion"],
    }
