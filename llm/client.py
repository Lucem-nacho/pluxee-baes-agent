"""Wrapper del proveedor LLM, agnóstico de proveedor.

Default: Google Gemini (gratis vía Google AI Studio, sin facturación activada
en el proyecto de Cloud asociado — ver .env.example). Soporta cambiar a
Anthropic u OpenAI más adelante sin tocar `agent/classifier.py` ni
`agent/orchestrator.py`: ambos solo llaman a `generar_respuesta(prompt)`.

Selección de proveedor vía LLM_PROVIDER en .env ("gemini" | "anthropic" | "openai").
"""

import time

from google import genai
from google.genai import errors

from config import settings

# El free tier de Gemini puede devolver 503 (alta demanda) de forma
# transitoria — observado en pruebas reales: 2 de 5 llamadas fallaron así.
# Reintenta solo errores de SERVIDOR (5xx); un ClientError (API key inválida,
# 400, etc.) va a fallar igual las 3 veces, así que no vale la pena esperar.
MAX_INTENTOS_GEMINI = 3
ESPERA_BASE_SEGUNDOS = 1  # backoff exponencial: 1s, 2s, 4s...

# 429 (RESOURCE_EXHAUSTED) es un caso distinto de ClientError: SÍ es
# recuperable, y la propia API indica cuánto esperar (`retryDelay`, ej.
# "34s"). Probado con gemini-3.6-flash: el free tier limita a 5
# requests/minuto (quotaId "GenerateRequestsPerMinutePerProjectPerModel-
# FreeTier") — se agota fácil en pruebas rápidas seguidas. Se reintenta UNA
# sola vez esperando ese tiempo (acotado a MAX_ESPERA_CUOTA_SEGUNDOS), nunca
# más — a diferencia de los demás ClientError, que no se reintentan.
MAX_ESPERA_CUOTA_SEGUNDOS = 60


def generar_respuesta(prompt: str) -> str:
    """Envía `prompt` ya renderizado al LLM configurado y devuelve el texto
    de la respuesta. Punto único de acoplamiento con el proveedor."""
    if settings.LLM_PROVIDER == "gemini":
        return _generar_gemini(prompt)
    if settings.LLM_PROVIDER == "anthropic":
        raise NotImplementedError(
            "Proveedor Anthropic no implementado todavía — pendiente para "
            "cuando haya créditos disponibles. Cambia LLM_PROVIDER a 'gemini' en .env."
        )
    if settings.LLM_PROVIDER == "openai":
        raise NotImplementedError(
            "Proveedor OpenAI no implementado todavía — pendiente para "
            "cuando haya créditos disponibles. Cambia LLM_PROVIDER a 'gemini' en .env."
        )
    raise ValueError(f"LLM_PROVIDER desconocido: {settings.LLM_PROVIDER!r}")


def _generar_gemini(prompt: str) -> str:
    if not settings.GOOGLE_API_KEY:
        raise RuntimeError(
            "Falta GOOGLE_API_KEY en .env. Consigue una gratis en "
            "https://aistudio.google.com/apikey y complétala en .env "
            "(ver .env.example) — no actives facturación en el proyecto de Cloud asociado."
        )
    client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    ya_reintento_cuota = False
    intento = 0
    while True:
        try:
            respuesta = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
            )
            return respuesta.text
        except errors.ServerError:
            intento += 1
            if intento >= MAX_INTENTOS_GEMINI:
                raise
            time.sleep(ESPERA_BASE_SEGUNDOS * (2 ** (intento - 1)))
        except errors.ClientError as error:
            if error.code == 429 and not ya_reintento_cuota:
                ya_reintento_cuota = True
                time.sleep(_segundos_espera_cuota(error))
                continue
            raise


def _segundos_espera_cuota(error: "errors.ClientError", tope: float = MAX_ESPERA_CUOTA_SEGUNDOS) -> float:
    """Extrae `retryDelay` (ej. "34s") del cuerpo del error 429 y lo acota a
    `tope` segundos, para no bloquear la demo indefinidamente si Google
    alguna vez pide una espera larga. Si no se puede extraer, espera el
    tope como fallback conservador."""
    try:
        detalles = error.details["error"]["details"]
        retry_info = next(d for d in detalles if d.get("@type", "").endswith("RetryInfo"))
        segundos = float(retry_info["retryDelay"].rstrip("s"))
    except (KeyError, StopIteration, ValueError, TypeError, AttributeError):
        segundos = tope
    return min(segundos, tope)
