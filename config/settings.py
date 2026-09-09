"""Configuración centralizada del proyecto (paths, nombres de colecciones, modelo).

Lee variables desde .env (ver .env.example) vía python-dotenv. Todo lo demás
en el proyecto debe importar de aquí en vez de leer os.environ directamente.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Proveedor de LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# gemini-3.6-flash explícito (ver .env.example y docs/arquitectura.md: es el
# modelo que Google recomienda para keys nuevas; gemini-2.5-flash da 404).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Alternativos, no implementados todavía (ver llm/client.py)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Vector store ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/processed/chroma")
COLLECTION_MANUAL_FAQ_SOCIAL = "manual_faq_social"
COLLECTION_NORMATIVA_JUNAEB = "normativa_junaeb"

# --- Fuente estructurada (comercios) ---
COMERCIOS_CSV_PATH = "data/comercios/comercios.csv"
