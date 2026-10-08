"""Persistencia del agente: escalamientos (escritura) y memoria de largo plazo.

Responsabilidad: reemplazar los dos puntos del agente que hoy NO persisten
nada entre ejecuciones:

1. `agent/escalation.py` armaba un dict de derivación que `app.py` mostraba y
   luego se perdía — no quedaba ningún registro de qué se derivó ni cuándo.
   Este módulo agrega la tabla `escalamientos`: la primera capacidad de
   ESCRITURA real del agente (EP2, IE1/IE2).
2. La única memoria existente era `st.session_state.mensajes` en `app.py`:
   vive en el navegador y se pierde al cerrar la pestaña (memoria de CORTO
   plazo). Este módulo agrega la tabla `interacciones`: memoria de LARGO
   plazo que persiste entre sesiones, consultable por `id_usuario` (EP2,
   IE3/IE4). Las dos capas son intencionalmente distintas, no una ni otra:
   el historial de sesión sigue viviendo en `app.py` sin cambios.

Mismo patrón de conexión que `rag/retrieval/comercios_query.py` (y la misma
razón): cada función abre su propia conexión sqlite, hace su trabajo y la
cierra — nunca se comparte una conexión global entre llamadas. Streamlit
ejecuta cada interacción del usuario en su propio hilo, y SQLite no permite
reutilizar una conexión creada en un hilo distinto. Acá además la base es en
disco (no ":memory:" como en comercios_query.py) porque el propósito es
justamente que los datos sobrevivan entre ejecuciones del proceso.

Limitación conocida y aceptada: `id_usuario` es un string que el estudiante
ingresa libremente en la interfaz (ver `app.py`), sin autenticación real —
suficiente para demostrar memoria de largo plazo en un contexto académico,
no para producción (cualquiera puede leer el historial de cualquier
`id_usuario` con solo escribirlo). Quedaría documentado como mejora futura
(autenticación real) en `docs/arquitectura.md`, igual que el validador de
fidelidad en EP1.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config import settings


def _conectar(db_path: str | Path = None) -> sqlite3.Connection:
    """Abre una conexión NUEVA al archivo sqlite en disco, creando el
    directorio y las tablas si todavía no existen. Quien llama es
    responsable de cerrarla (usar siempre con `try/finally` o `with`)."""
    ruta = Path(db_path or settings.MEMORIA_DB_PATH)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(ruta)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS escalamientos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            consulta_usuario TEXT NOT NULL,
            historial_contexto TEXT NOT NULL,
            justificacion_clasificador TEXT NOT NULL,
            fecha_derivacion TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS interacciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_usuario TEXT NOT NULL,
            consulta_usuario TEXT NOT NULL,
            respuesta TEXT NOT NULL,
            estado TEXT NOT NULL CHECK (estado IN ('INFORMATIVA', 'ESCALAR')),
            fecha TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_interacciones_usuario ON interacciones (id_usuario, fecha)"
    )
    conn.commit()
    return conn


def inicializar_db(db_path: str | Path = None) -> None:
    """Crea el archivo y las tablas si no existen, sin insertar nada.
    Pensado para `scripts/ingest_all.py`, igual que
    `rag.retrieval.comercios_query.cargar_tabla_comercios` valida el CSV al
    preparar el proyecto — acá valida que el archivo de memoria sea
    escribible antes de que el agente lo necesite en caliente."""
    _conectar(db_path).close()


def registrar_escalamiento(
    consulta_usuario: str,
    historial_contexto: str,
    justificacion: str,
    db_path: str | Path = None,
) -> dict:
    """Persiste la derivación y devuelve el mismo paquete que
    `agent.escalation.derivar` ya devolvía (misma forma, para no romper a
    quien lo consume en `app.py`), con `id_escalamiento` agregado."""
    fecha_derivacion = datetime.now(timezone.utc).isoformat()

    conn = _conectar(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO escalamientos
                (consulta_usuario, historial_contexto, justificacion_clasificador, fecha_derivacion)
            VALUES (?, ?, ?, ?)
            """,
            (consulta_usuario, historial_contexto, justificacion, fecha_derivacion),
        )
        conn.commit()
        id_escalamiento = cursor.lastrowid
    finally:
        conn.close()

    return {
        "estado": "ESCALAR",
        "id_escalamiento": id_escalamiento,
        "consulta_usuario": consulta_usuario,
        "historial_contexto": historial_contexto,
        "justificacion_clasificador": justificacion,
        "fecha_derivacion": fecha_derivacion,
    }


def registrar_interaccion(
    id_usuario: str,
    consulta_usuario: str,
    respuesta: str,
    estado: str,
    db_path: str | Path = None,
) -> int:
    """Guarda una interacción resuelta (INFORMATIVA o ESCALAR) en la memoria
    de largo plazo del usuario. Devuelve el id de la fila insertada."""
    if estado not in ("INFORMATIVA", "ESCALAR"):
        raise ValueError(f"estado inválido: {estado!r} (debe ser INFORMATIVA o ESCALAR)")

    conn = _conectar(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO interacciones (id_usuario, consulta_usuario, respuesta, estado, fecha)
            VALUES (?, ?, ?, ?, ?)
            """,
            (id_usuario, consulta_usuario, respuesta, estado, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def consultar_historial_usuario(
    id_usuario: str,
    limite: int = 5,
    db_path: str | Path = None,
) -> list[dict]:
    """Devuelve hasta `limite` interacciones pasadas de `id_usuario`, más
    recientes primero. Lista vacía si el usuario no tiene historial — no es
    un error, es el caso normal de un usuario nuevo."""
    conn = _conectar(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, consulta_usuario, respuesta, estado, fecha
            FROM interacciones
            WHERE id_usuario = ?
            ORDER BY fecha DESC
            LIMIT ?
            """,
            (id_usuario, limite),
        )
        return [dict(fila) for fila in cursor.fetchall()]
    finally:
        conn.close()
