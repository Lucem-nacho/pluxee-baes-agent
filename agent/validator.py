"""Validador de fidelidad de la respuesta generada — NO IMPLEMENTADO EN ESTA VERSIÓN.

Qué haría (diseño previsto):
    Post-procesar la respuesta del LLM generador comparando las entidades y
    afirmaciones normativas que contiene (ej. nombres de comercios, montos,
    causales, códigos de estado) contra el contenido literal de los
    fragmentos recuperados (`fragmentos_recuperados_con_fuente_y_fecha`).
    Si detecta una afirmación no respaldada por el contexto (alucinación),
    bloquear la respuesta y forzar una derivación a ejecutivo humano en vez
    de entregarla al estudiante.

Por qué queda fuera de esta versión:
    Alcance de tiempo de un proyecto individual de 5 semanas. La regla 1 del
    prompt del generador (`prompts/generador.py`) ya instruye al LLM a no
    responder fuera del contexto recuperado, lo que mitiga el riesgo sin
    requerir un segundo paso de verificación. Este módulo queda documentado
    como limitación conocida y mejora futura en `docs/arquitectura.md`, no
    como funcionalidad construida.
"""


def validar_fidelidad(respuesta_generada: str, fragmentos_recuperados: list) -> bool:
    """Placeholder de la interfaz prevista. No implementado."""
    raise NotImplementedError(
        "Validador de fidelidad fuera de alcance de esta versión — ver docstring del módulo."
    )
