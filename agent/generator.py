"""Generador de la respuesta final del pipeline RAG.

Responsabilidad: renderizar `prompts.generador.GENERADOR_PROMPT` con el
contexto recuperado (ya arbitrado) y la consulta, llamar al LLM vía
`llm.client`, y verificar que la respuesta cumpla el formato que el propio
prompt exige: cita de fuente y máximo 4 frases.

Estas dos verificaciones son heurísticas de FORMATO, no de fidelidad de
contenido — no confirman que lo dicho sea correcto ni que no haya
alucinación (eso es responsabilidad de `agent/validator.py`, explícitamente
no implementado en esta versión). Si la respuesta no cumple, se reintenta
UNA vez con un recordatorio explícito de formato (acotado a 1 reintento para
no multiplicar el consumo de cuota del free tier de Gemini). Si sigue sin
cumplir, se devuelve igual — no se bloquea — pero con `advertencias` no
vacías para que quien la consuma decida qué hacer (loguearla, mostrarla con
una nota, etc.).

Falso positivo conocido y aceptado: si el LLM sigue correctamente la regla 1
del prompt ("si el contexto no cubre la pregunta, dilo explícitamente...")
sin necesidad de citar una fuente, esta verificación igual marca la
advertencia de "no cita fuente" — es una respuesta válida marcada de más,
no un error real.
"""

import re

from llm.client import generar_respuesta
from prompts.generador import GENERADOR_PROMPT

MAX_FRASES = 4
PATRON_CITA = "según"  # tal como lo pide el ejemplo de la propia regla 4 del prompt


def generar(consulta_usuario: str, fragmentos_contexto: list[dict]) -> dict:
    """Devuelve {"respuesta": str, "advertencias": list[str]}."""
    prompt = _armar_prompt(consulta_usuario, fragmentos_contexto)

    respuesta = generar_respuesta(prompt)
    advertencias = _validar_formato(respuesta)

    if advertencias:
        prompt_reforzado = (
            prompt
            + "\n\nRecuerda cumplir ESTRICTAMENTE: cita la fuente explícitamente "
            "(ej. \"según el manual BAES...\") y responde en máximo 4 frases."
        )
        respuesta_reintento = generar_respuesta(prompt_reforzado)
        advertencias_reintento = _validar_formato(respuesta_reintento)
        if len(advertencias_reintento) < len(advertencias):
            respuesta, advertencias = respuesta_reintento, advertencias_reintento

    return {"respuesta": respuesta, "advertencias": advertencias}


def _armar_prompt(consulta_usuario: str, fragmentos_contexto: list[dict]) -> str:
    return GENERADOR_PROMPT.format(
        fragmentos_recuperados_con_fuente_y_fecha=_formatear_contexto(fragmentos_contexto),
        consulta_usuario=consulta_usuario,
    )


def _formatear_contexto(fragmentos: list[dict]) -> str:
    if not fragmentos:
        return "(sin fragmentos recuperados para esta consulta)"
    return "\n\n".join(
        f"[Fuente: {f['fuente']} | Fecha: {f['fecha']}]\n{f['texto']}" for f in fragmentos
    )


def _validar_formato(respuesta: str) -> list[str]:
    advertencias = []

    if PATRON_CITA not in respuesta.lower():
        advertencias.append(f"no cita la fuente con el patrón esperado ('{PATRON_CITA} ...')")

    num_frases = len([f for f in re.split(r"[.!?]+", respuesta) if f.strip()])
    if num_frases > MAX_FRASES:
        advertencias.append(f"excede el límite de {MAX_FRASES} frases (tiene {num_frases})")

    return advertencias
