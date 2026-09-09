"""Recuperación semántica sobre las colecciones Chroma.

Responsabilidad: dada una consulta, buscar en "manual_faq_social" y
"normativa_junaeb" (las dos colecciones, no una sola) y devolver los
fragmentos más relevantes de cada una, con su metadata (fuente, fecha)
intacta para que `rag/arbitration.py` pueda operar sobre ella.

"normativa_junaeb" no existe como colección en esta versión — se determinó
redundante con el contenido ya cubierto por el manual (ver limitación
documentada en docs/arquitectura.md), no un vacío pendiente. Se omite en vez
de fallar, para que este módulo no se rompa si esa decisión cambia más adelante.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings
from rag.text_utils import normalizar_para_embedding

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Carga perezosa: el modelo de embeddings solo se instancia la primera vez
# que se llama a recuperar(), no al importar el módulo.
_modelo = None


def _get_modelo() -> SentenceTransformer:
    global _modelo
    if _modelo is None:
        _modelo = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _modelo


def _query_coleccion(cliente: chromadb.ClientAPI, nombre_coleccion: str, embedding: list[float], k: int) -> list[dict]:
    try:
        coleccion = cliente.get_collection(nombre_coleccion)
    except chromadb.errors.NotFoundError:
        return []

    resultado = coleccion.query(query_embeddings=[embedding], n_results=k)
    fragmentos = []
    for doc, meta, distancia in zip(
        resultado["documents"][0], resultado["metadatas"][0], resultado["distances"][0]
    ):
        fragmentos.append({**meta, "texto": doc, "distancia": distancia, "coleccion": nombre_coleccion})
    return fragmentos


def recuperar(consulta_usuario: str, k: int = 4) -> list[dict]:
    """Busca `consulta_usuario` en ambas colecciones Chroma y devuelve la
    lista combinada de fragmentos (sin deduplicar ni arbitrar — eso es
    responsabilidad de `rag/arbitration.py`)."""
    modelo = _get_modelo()
    embedding = modelo.encode([normalizar_para_embedding(consulta_usuario)])[0].tolist()

    cliente = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    fragmentos = []
    fragmentos += _query_coleccion(cliente, settings.COLLECTION_MANUAL_FAQ_SOCIAL, embedding, k)
    fragmentos += _query_coleccion(cliente, settings.COLLECTION_NORMATIVA_JUNAEB, embedding, k)
    return fragmentos
