"""Smoke test manual: confirma que la conexión con el LLM configurado
(por defecto, Gemini) funciona end-to-end, con un prompt genérico fuera
del dominio BAES (todavía no hay pipeline RAG ni prompts de dominio).

No es un test de pytest a propósito: hace una llamada de red real y
requiere una API key configurada, así que no debe correr en una suite
automatizada por defecto.

Uso:
    python scripts/check_llm_connection.py

Requiere GOOGLE_API_KEY en .env (ver .env.example; gratis en
https://aistudio.google.com/apikey, sin activar facturación).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from llm.client import generar_respuesta

PROMPT_DE_PRUEBA = "Respondeme en una sola palabra: 'ok'."


def main() -> None:
    print(f"Proveedor configurado: {settings.LLM_PROVIDER}")
    if settings.LLM_PROVIDER == "gemini":
        print(f"Modelo: {settings.GEMINI_MODEL}")
    print(f"Prompt de prueba: {PROMPT_DE_PRUEBA!r}")

    respuesta = generar_respuesta(PROMPT_DE_PRUEBA)

    print(f"Respuesta del modelo: {respuesta!r}")
    print("Conexión OK.")


if __name__ == "__main__":
    main()
