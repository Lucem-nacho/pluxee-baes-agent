"""Construcción de los índices vectoriales persistentes (Chroma).

Colección "manual_faq_social": manual BAES + resolución + FAQ + tweets
normativos filtrados (`rag/loaders/social_loader.py`, mock de Twitter/X).

La colección "normativa_junaeb" (Ley de Etiquetado / normativa externa) no
se construye todavía: se determinó redundante con el manual (ver
docs/arquitectura.md).

Filtro explícito: se excluye el chunk de la página 2 del manual (el índice
de contenidos) — es una lista de títulos con puntos de relleno
("......... 5"), sin contenido normativo real, que solo agrega ruido al
vector store (ver la observación dejada en rag/indexing/chunking.py).

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, gratis — ver
rag/indexing/chunking.py para el razonamiento del límite de ~256 tokens que
fija CHUNK_SIZE).
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings
from rag.indexing.chunking import chunkear_documento
from rag.loaders import faq_loader, pdf_loader, social_loader
from rag.text_utils import normalizar_para_embedding

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# pagina_manual del índice de contenidos del manual — se excluye, ver docstring.
PAGINA_INDICE_CONTENIDOS = 2


def _con_tipo(fragmentos: list[dict], tipo: str) -> list[dict]:
    return [{**f, "tipo_fragmento": tipo} for f in fragmentos]


def preparar_documentos_manual_faq_social() -> list[dict]:
    """Recolecta y filtra los fragmentos de manual + resolución + FAQ +
    tweets normativos, listos para calcular embeddings. Separado de
    `construir_indices` para poder probar la lógica de recolección/filtrado
    sin cargar el modelo de embeddings ni tocar Chroma."""
    manual_chunks = chunkear_documento(pdf_loader.cargar_manual_baes())
    manual_chunks = [
        c for c in manual_chunks if c.get("pagina_manual") != PAGINA_INDICE_CONTENIDOS
    ]

    resolucion_chunks = chunkear_documento(pdf_loader.cargar_resolucion())
    faq_fragmentos = faq_loader.cargar_faq()
    social_fragmentos = social_loader.cargar_tweets_normativos()

    return (
        _con_tipo(manual_chunks, "manual")
        + _con_tipo(resolucion_chunks, "resolucion")
        + _con_tipo(faq_fragmentos, "faq")
        + _con_tipo(social_fragmentos, "social")
    )


def construir_indices() -> None:
    """Calcula embeddings y persiste la colección 'manual_faq_social' en Chroma.

    Usa upsert (no add) para que correr el script de nuevo tras cambiar un
    PDF o la FAQ no falle por ids duplicados.
    """
    documentos = preparar_documentos_manual_faq_social()

    modelo = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # normalizar_para_embedding() solo afecta lo que se vectoriza — el texto
    # citado/mostrado (`documents=` más abajo) conserva el formato original.
    embeddings = modelo.encode(
        [normalizar_para_embedding(d["texto"]) for d in documentos]
    ).tolist()

    cliente = chromadb.PersistentClient(path=str(Path(settings.CHROMA_PERSIST_DIR)))
    coleccion = cliente.get_or_create_collection(settings.COLLECTION_MANUAL_FAQ_SOCIAL)
    coleccion.upsert(
        ids=[f"{d['tipo_fragmento']}-{i}" for i, d in enumerate(documentos)],
        embeddings=embeddings,
        documents=[d["texto"] for d in documentos],
        metadatas=[{k: v for k, v in d.items() if k != "texto"} for d in documentos],
    )
