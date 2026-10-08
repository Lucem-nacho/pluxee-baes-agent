"""Derivación a ejecutivo humano — versión EP1, SUPERADA por EP2.

`agent/orchestrator.py` ya NO importa este módulo: desde EP2, la derivación
se persiste con `memory.store.registrar_escalamiento` (tabla `escalamientos`
en SQLite) en vez de devolver un dict que se perdía al cerrar la sesión. Se
deja el archivo sin borrar porque documenta el diseño original de EP1 (y
`memory.store.registrar_escalamiento` devuelve un dict con la misma forma,
a propósito, para que `app.py` no tuviera que cambiar su lógica de
renderizado entre versiones) — no es código muerto por descuido, es
historial de diseño.

Responsabilidad original (EP1): cuando el clasificador devuelve ESCALAR,
armar el paquete de derivación (consulta original + historial completo de
la conversación + justificación del clasificador, con marca de tiempo) para
entregarlo al canal de atención humana. No ejecuta ninguna acción sobre la
cuenta del estudiante ni integra ningún sistema de tickets real — solo
empaquetaba el contexto para que un ejecutivo humano actuara.
"""

from datetime import datetime, timezone


def derivar(consulta_usuario: str, historial_contexto: str, justificacion: str) -> dict:
    """Devuelve el paquete de derivación, listo para que `app.py` lo muestre
    o lo entregue al canal de atención humana que corresponda."""
    return {
        "estado": "ESCALAR",
        "consulta_usuario": consulta_usuario,
        "historial_contexto": historial_contexto,
        "justificacion_clasificador": justificacion,
        "fecha_derivacion": datetime.now(timezone.utc).isoformat(),
    }
