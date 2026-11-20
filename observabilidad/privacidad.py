"""Privacidad del registro de ejecuciones: qué se enmascara y cómo se
anonimiza el identificador del usuario.

Alcance y límites de este módulo:

- El enmascarado es por expresiones regulares y cubre solo RUT, correo y
  teléfono chilenos. NO cubre nombres ni direcciones, ni ningún otro dato
  personal escrito en texto libre, ni un RUT escrito sin guion ni puntos.
- El identificador del usuario queda seudonimizado (hash con sal), no
  anonimizado: el mismo id produce siempre el mismo código.
- Esto aplica SOLO a la base de observabilidad. `memoria.db`
  (`memory/store.py`, la memoria de largo plazo) NO aplica este enmascarado:
  guarda las consultas y el `id_usuario` tal como llegan.
"""

import hashlib
import os
import re
import secrets
from pathlib import Path

from config import settings

_CORREO = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# 12.345.678-9 / 12.345.678k (con puntos, guion opcional) y 12345678-9 (guion).
_RUT = re.compile(
    r"(?<!\d)\d{1,2}(?:\.\d{3}){2}-?[\dkK](?!\w)"
    r"|(?<!\d)\d{7,8}\s?-\s?[\dkK](?!\w)"
)

# Móvil (9 xxxx xxxx), fijo de Santiago (2 xxxx xxxx) y fijo regional
# ((41) xxx xxxx), con o sin +56 y con espacios, puntos o guiones.
_TELEFONO = re.compile(
    r"(?<![\w.])(?:\+\s?56[\s.-]?|56[\s.-]?)?"
    r"(?:9[\s.-]?\d{4}[\s.-]?\d{4}"
    r"|2[\s.-]?\d{4}[\s.-]?\d{4}"
    r"|\(?[3-7]\d\)?[\s.-]?\d{3}[\s.-]?\d{4})(?!\w)"
)

# El orden importa: el correo puede contener dígitos, y un RUT con guion no
# debe quedar a medias para que el patrón de teléfono lo parta.
_PATRONES = ((_CORREO, "[CORREO]"), (_RUT, "[RUT]"), (_TELEFONO, "[TELEFONO]"))


def enmascarar(texto: str | None) -> str | None:
    """Reemplaza RUT, correos y teléfonos por marcadores. `None` pasa tal cual."""
    if texto is None:
        return None
    for patron, marcador in _PATRONES:
        texto = patron.sub(marcador, texto)
    return texto


def anonimizar_id_usuario(id_usuario: str, db_path: str | Path | None = None) -> str:
    """SHA-256 de `sal:id_usuario`, truncado a 12 caracteres hexadecimales.

    Es seudonimización, no anonimato absoluto: el mismo id produce siempre el
    mismo código (permite agrupar ejecuciones de un usuario), pero sin la sal
    no se puede recuperar el id. La sal vive fuera de la base para poder
    compartir el .db sin ella.
    """
    sal = _obtener_sal(db_path)
    return hashlib.sha256(f"{sal}:{id_usuario}".encode("utf-8")).hexdigest()[:12]


def _obtener_sal(db_path: str | Path | None = None) -> str:
    if settings.OBSERVABILIDAD_SAL:
        return settings.OBSERVABILIDAD_SAL

    ruta_db = Path(db_path or settings.OBSERVABILIDAD_DB_PATH)
    ruta = ruta_db.with_name("observabilidad.sal")
    if not ruta.exists():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = ruta.with_name(f"observabilidad.sal.{secrets.token_hex(4)}.tmp")
        temporal.write_text(secrets.token_hex(16), encoding="utf-8")
        try:
            # Enlace atómico: si otro hilo creó la sal primero, os.link falla y
            # se usa la de ese hilo — nunca dos sales distintas en una instalación.
            os.link(temporal, ruta)
        except FileExistsError:
            pass
        finally:
            temporal.unlink(missing_ok=True)
    return ruta.read_text(encoding="utf-8").strip()
