"""Derivación a ejecutivo humano.

Responsabilidad: cuando el clasificador devuelve ESCALAR, armar el paquete
de derivación (consulta original + historial completo de la conversación +
justificación del clasificador, con marca de tiempo) para entregarlo al
canal de atención humana. No ejecuta ninguna acción sobre la cuenta del
estudiante ni integra ningún sistema de tickets real (fuera de alcance de
esta versión) — solo empaqueta el contexto para que un ejecutivo humano actúe.
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
