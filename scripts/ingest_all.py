"""Setup inicial: construye la colección Chroma "manual_faq_social" (manual +
resolución + FAQ + tweets normativos) vía `rag/indexing/build_vectorstore.py`,
carga la tabla sqlite de comercios vía `rag/retrieval/comercios_query.py`, y
(EP2) crea el archivo sqlite de memoria persistente vía `memory/store.py`.

La colección "normativa_junaeb" no se construye — se determinó redundante
con el manual, ver docs/arquitectura.md.

Correr una vez al preparar el proyecto, y de nuevo si cambian los PDFs/FAQ/
tweets mock. No requiere GOOGLE_API_KEY: todo el trabajo es local
(embeddings con sentence-transformers, sin llamadas a Gemini).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.store import inicializar_db
from rag.indexing.build_vectorstore import construir_indices
from rag.retrieval.comercios_query import cargar_tabla_comercios

if __name__ == "__main__":
    print("Construyendo colección 'manual_faq_social' (manual + resolución + FAQ + social)...")
    construir_indices()
    print("Índice vectorial listo.")

    print("Cargando tabla de comercios (sqlite en memoria)...")
    cargar_tabla_comercios()
    print("Tabla de comercios lista.")

    print("Creando base de memoria persistente (escalamientos + interacciones)...")
    inicializar_db()
    print("Base de memoria lista.")

    print("\nIngesta completa.")
