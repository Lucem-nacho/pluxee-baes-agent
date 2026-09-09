"""Clasificador INFORMATIVA / ESCALAR.

Responsabilidad: renderizar `prompts.clasificador.CLASIFICADOR_PROMPT` con la
consulta y el historial de conversación, llamar al LLM vía `llm.client`, y
parsear la salida JSON esperada ({"categoria": ..., "justificacion": ...}).

Ante cualquier respuesta del LLM que no se pueda interpretar como ese JSON
(clave faltante, categoría fuera de {INFORMATIVA, ESCALAR}, texto no-JSON),
se clasifica como ESCALAR: ante la duda, deriva a un ejecutivo humano en vez
de arriesgar que una consulta mal clasificada reciba una respuesta automática.
"""

import json

from llm.client import generar_respuesta
from prompts.clasificador import CLASIFICADOR_PROMPT

CATEGORIAS_VALIDAS = {"INFORMATIVA", "ESCALAR"}


def clasificar(consulta_usuario: str, historial_contexto: str) -> dict:
    """Devuelve {"categoria": "INFORMATIVA" | "ESCALAR", "justificacion": str}."""
    prompt = CLASIFICADOR_PROMPT.format(
        consulta_usuario=consulta_usuario,
        historial_contexto=historial_contexto,
    )
    respuesta_cruda = generar_respuesta(prompt)

    try:
        return _parsear_respuesta(respuesta_cruda)
    except ValueError as error:
        return {
            "categoria": "ESCALAR",
            "justificacion": f"Respuesta del clasificador no interpretable, se deriva por seguridad ({error}).",
        }


def _parsear_respuesta(respuesta_cruda: str) -> dict:
    texto = respuesta_cruda.strip()

    # Pese a que el prompt pide "sin texto adicional", es común que el LLM
    # envuelva el JSON en un bloque de código markdown (```json ... ```).
    if texto.startswith("```"):
        texto = texto.strip("`").strip()
        if texto.lower().startswith("json"):
            texto = texto[4:].strip()

    datos = json.loads(texto)  # lanza json.JSONDecodeError (subclase de ValueError) si no es JSON

    categoria = datos.get("categoria")
    justificacion = datos.get("justificacion")
    if categoria not in CATEGORIAS_VALIDAS or not justificacion:
        raise ValueError(f"campos inesperados en la respuesta del LLM: {datos!r}")

    return {"categoria": categoria, "justificacion": justificacion}
