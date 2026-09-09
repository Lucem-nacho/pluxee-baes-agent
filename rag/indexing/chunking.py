"""Chunking de texto largo (manual BAES, resolución, normativa Junaeb/Ley de Etiquetado).

FAQ y social NO pasan por aquí: cada entrada ya es un fragmento atómico
(un par pregunta-respuesta, o un post filtrado) — ver `rag/loaders/faq_loader.py`
y `rag/loaders/social_loader.py`.

Tamaño de chunk: el embedding model (`sentence-transformers/all-MiniLM-L6-v2`,
ver rag/indexing/build_vectorstore.py) trunca en ~256 tokens, que en español
administrativo ronda los 1000-1200 caracteres — pasado ese punto el modelo
ignora el resto del texto en silencio. CHUNK_SIZE se fija bajo ese límite
para no perder contenido. CHUNK_OVERLAP conserva contexto entre chunks
consecutivos (una oración o un ítem de tabla que quede partido justo en el
borde del corte).
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunkear_fragmento(fragmento: dict) -> list[dict]:
    """Divide fragmento["texto"] (ej. el texto de una página del manual) en
    uno o más chunks, preservando toda la metadata original (fuente, fecha,
    pagina_pdf, pagina_manual, etc.) y agregando `chunk_index` para poder
    reconstruir el orden dentro de la página."""
    texto = fragmento["texto"]
    metadata_base = {k: v for k, v in fragmento.items() if k != "texto"}

    pedazos = _splitter.split_text(texto)
    return [
        {**metadata_base, "texto": pedazo, "chunk_index": i}
        for i, pedazo in enumerate(pedazos)
    ]


def chunkear_documento(fragmentos: list[dict]) -> list[dict]:
    """Aplica `chunkear_fragmento` a una lista de fragmentos (ej. todas las
    páginas devueltas por `rag/loaders/pdf_loader.py`) y aplana el resultado
    en una sola lista de chunks listos para `rag/indexing/build_vectorstore.py`."""
    chunks = []
    for fragmento in fragmentos:
        chunks.extend(chunkear_fragmento(fragmento))
    return chunks
