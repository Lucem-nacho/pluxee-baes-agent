"""Ingesta de Twitter/X de Pluxee (externa, dinámica).

Responsabilidad: leer data/social/tweets_mock.json (simulación de la ingesta
real cada 24h vía API de X — el mock no cambia solo, se edita a mano para
simular nuevos posts, ver `scripts/refresh_social.py`), filtrar solo
contenido de naturaleza normativa (descartar posts promocionales/genéricos)
y devolver fragmentos con metadata (`fuente`, `fecha`) para indexar en la
colección "manual_faq_social".

El filtro es una heurística de palabras clave, no NLP real — mismo criterio
de "cableado barato" que la heurística de comercios en
`agent/orchestrator.py`: suficiente para un dataset mock de pocos tweets,
no para clasificar contenido real de redes sociales a escala.
"""

import json
from pathlib import Path

TWEETS_PATH = Path(__file__).resolve().parents[2] / "data" / "social" / "tweets_mock.json"

FUENTE_TWITTER = "Twitter/X de Pluxee"

PALABRAS_CLAVE_NORMATIVAS = (
    "beneficio",
    "restricción",
    "restriccion",
    "prohibido",
    "no puede",
    "no se puede",
    "actualización",
    "actualizacion",
    "normativa",
    "causal",
    "sanción",
    "sancion",
    "habilitado",
    "focalización",
    "focalizacion",
    "prioridad",
)


def cargar_tweets_normativos(path: str | Path = TWEETS_PATH) -> list[dict]:
    """Devuelve un fragmento por cada tweet clasificado como normativo (los
    promocionales/genéricos se descartan). No pasa por
    `rag/indexing/chunking.py`: un tweet ya es un fragmento atómico."""
    with open(path, encoding="utf-8") as f:
        tweets = json.load(f)

    return [
        {"texto": tweet["texto"], "fuente": FUENTE_TWITTER, "fecha": tweet["fecha"]}
        for tweet in tweets
        if _es_normativo(tweet["texto"])
    ]


def _es_normativo(texto: str) -> bool:
    texto_normalizado = texto.lower()
    return any(palabra in texto_normalizado for palabra in PALABRAS_CLAVE_NORMATIVAS)
