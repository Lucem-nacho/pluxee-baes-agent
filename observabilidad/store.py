"""Base SQLite de observabilidad: tablas `ejecuciones` y `pasos`.

Mismo patrón de conexión que `memory/store.py` y por la misma razón: cada
función abre su propia conexión, la usa y la cierra, porque Streamlit ejecuta
cada interacción en un hilo distinto y SQLite no permite compartir una
conexión entre hilos. La ruta se lee de `settings` en cada llamada (no al
importar) para poder redirigirla en las pruebas.

Este módulo guarda lo que recibe: el enmascarado de datos personales y la
anonimización del id ocurren antes, en `observabilidad/medicion.py`.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import settings

ORIGENES = ("app", "evaluacion")
ESTADOS = ("INFORMATIVA", "ESCALAR", "ERROR")
TIPOS_ERROR = (
    "ninguno",
    "429",
    "503",
    "limite_iteraciones",
    "sin_respuesta",
    "clasificador_no_interpretable",
    "otro",
)

_COLUMNAS_EJECUCION = (
    "fecha_utc",
    "origen",
    "id_usuario_anonimo",
    "consulta",
    "categoria",
    "estado",
    "latencia_total_ms",
    "latencia_clasificador_ms",
    "latencia_agente_ms",
    "llamadas_llm",
    "tokens_entrada",
    "tokens_salida",
    "iteraciones_agente",
    "tipo_error",
    "modelo",
    "respuesta",
    "caso_id",
    "variante",
    "repeticion",
)
# Columnas agregadas después de la primera versión del esquema: las bases ya
# existentes no las tienen y `_conectar` las agrega (ver `_migrar`).
_COLUMNAS_AGREGADAS = {"caso_id": "TEXT", "variante": "TEXT", "repeticion": "INTEGER"}
_COLUMNAS_PASO = ("orden", "herramienta", "argumentos", "duracion_ms", "exito", "tamano_salida")


def _lista_sql(valores: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in valores)


def _conectar(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Abre una conexión NUEVA, creando el directorio y las tablas si no
    existen. Quien llama es responsable de cerrarla."""
    ruta = Path(db_path or settings.OBSERVABILIDAD_DB_PATH)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(ruta)
    conn.row_factory = sqlite3.Row
    # La integridad referencial es por conexión en SQLite: sin esto, borrar
    # una ejecución dejaría sus pasos huérfanos.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS ejecuciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_utc TEXT NOT NULL,
            origen TEXT NOT NULL CHECK (origen IN ({_lista_sql(ORIGENES)})),
            id_usuario_anonimo TEXT NOT NULL,
            consulta TEXT NOT NULL,
            categoria TEXT CHECK (categoria IS NULL OR categoria IN ('INFORMATIVA', 'ESCALAR')),
            estado TEXT NOT NULL CHECK (estado IN ({_lista_sql(ESTADOS)})),
            latencia_total_ms INTEGER NOT NULL,
            latencia_clasificador_ms INTEGER,
            latencia_agente_ms INTEGER,
            llamadas_llm INTEGER NOT NULL,
            tokens_entrada INTEGER,
            tokens_salida INTEGER,
            iteraciones_agente INTEGER,
            tipo_error TEXT NOT NULL CHECK (tipo_error IN ({_lista_sql(TIPOS_ERROR)})),
            modelo TEXT NOT NULL,
            respuesta TEXT,
            caso_id TEXT,
            variante TEXT,
            repeticion INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pasos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_ejecucion INTEGER NOT NULL REFERENCES ejecuciones (id) ON DELETE CASCADE,
            orden INTEGER NOT NULL,
            herramienta TEXT NOT NULL,
            argumentos TEXT,
            duracion_ms INTEGER,
            exito INTEGER NOT NULL CHECK (exito IN (0, 1)),
            tamano_salida INTEGER
        )
        """
    )
    _migrar(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ejecuciones_fecha ON ejecuciones (fecha_utc)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ejecuciones_caso ON ejecuciones (origen, caso_id)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pasos_ejecucion ON pasos (id_ejecucion)")
    conn.commit()
    return conn


def _migrar(conn: sqlite3.Connection) -> None:
    """Agrega las columnas que le falten a una tabla `ejecuciones` creada con
    una versión anterior del esquema. Es idempotente y no toca los datos."""
    existentes = {fila["name"] for fila in conn.execute("PRAGMA table_info(ejecuciones)")}
    for columna, tipo in _COLUMNAS_AGREGADAS.items():
        if columna in existentes:
            continue
        try:
            conn.execute(f"ALTER TABLE ejecuciones ADD COLUMN {columna} {tipo}")
        except sqlite3.OperationalError as error:
            # Otro hilo la agregó entre el PRAGMA y el ALTER: es lo que queríamos.
            if "duplicate column name" not in str(error).lower():
                raise
    conn.commit()


def fecha_utc_ahora() -> str:
    """Formato fijo (milisegundos, +00:00) para que el orden alfabético de las
    fechas coincida con el cronológico — la purga compara strings."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def registrar_ejecucion(
    ejecucion: dict, pasos: list[dict], db_path: str | Path | None = None
) -> int:
    """Inserta una ejecución y sus pasos en una sola transacción y devuelve el
    id de la ejecución. `fecha_utc`, `caso_id`, `variante` y `repeticion` son
    opcionales (por defecto, ahora y sin caso)."""
    fila = {
        "fecha_utc": fecha_utc_ahora(),
        "caso_id": None,
        "variante": None,
        "repeticion": None,
        **ejecucion,
    }
    faltantes = [c for c in _COLUMNAS_EJECUCION if c not in fila]
    if faltantes:
        raise ValueError(f"faltan columnas de la ejecución: {faltantes}")

    conn = _conectar(db_path)
    try:
        with conn:  # commit al salir, rollback si algo falla a mitad de camino
            cursor = conn.execute(
                f"INSERT INTO ejecuciones ({', '.join(_COLUMNAS_EJECUCION)}) "
                f"VALUES ({', '.join('?' for _ in _COLUMNAS_EJECUCION)})",
                [fila[c] for c in _COLUMNAS_EJECUCION],
            )
            id_ejecucion = cursor.lastrowid
            conn.executemany(
                f"INSERT INTO pasos (id_ejecucion, {', '.join(_COLUMNAS_PASO)}) "
                f"VALUES (?, {', '.join('?' for _ in _COLUMNAS_PASO)})",
                [(id_ejecucion, *(paso[c] for c in _COLUMNAS_PASO)) for paso in pasos],
            )
        return id_ejecucion
    finally:
        conn.close()


def purgar_anteriores_a(
    dias: int,
    db_path: str | Path | None = None,
    ahora: datetime | None = None,
) -> int:
    """Borra las ejecuciones con más de `dias` días (y sus pasos, por la
    clave foránea con ON DELETE CASCADE). Devuelve cuántas ejecuciones borró."""
    if dias < 0:
        raise ValueError("dias no puede ser negativo")
    limite = (ahora or datetime.now(timezone.utc)) - timedelta(days=dias)
    corte = limite.isoformat(timespec="milliseconds")

    conn = _conectar(db_path)
    try:
        with conn:
            return conn.execute("DELETE FROM ejecuciones WHERE fecha_utc < ?", (corte,)).rowcount
    finally:
        conn.close()


def combinaciones_evaluadas(db_path: str | Path | None = None) -> set[tuple[str, str, int]]:
    """Combinaciones (caso_id, variante, repeticion) de origen "evaluacion" que
    ya tienen una ejecución registrada. Sirve para reanudar una evaluación.

    Una ejecución que terminó en ERROR por 429 o 503 NO cuenta como hecha: es
    una falla del proveedor (cuota o disponibilidad), no un resultado del
    agente, y la combinación debe poder reintentarse. Cualquier otro
    desenlace, incluidos otros errores, sí cuenta.
    """
    conn = _conectar(db_path)
    try:
        filas = conn.execute(
            """
            SELECT DISTINCT caso_id, variante, repeticion
            FROM ejecuciones
            WHERE origen = 'evaluacion'
              AND caso_id IS NOT NULL AND variante IS NOT NULL AND repeticion IS NOT NULL
              AND NOT (estado = 'ERROR' AND tipo_error IN ('429', '503'))
            """
        ).fetchall()
        return {(f["caso_id"], f["variante"], f["repeticion"]) for f in filas}
    finally:
        conn.close()
