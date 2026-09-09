"""Consulta estructurada sobre el listado de comercios asociados.

Fuente interna, estructurada — NO se vectoriza. `data/comercios/comercios.csv`
se carga a una tabla sqlite en memoria (stdlib, sin dependencias extra) y se
consulta con SQL parametrizado por nombre/categoría, no por búsqueda semántica.

Responsabilidad: exponer una función de consulta que el orquestador invoque
cuando la pregunta del estudiante sea sobre un comercio o categoría puntual
(ej. "¿puedo comprar en el Jumbo de Providencia?"). Un comercio "No habilitado
para BAES" (ej. Farmacias Cruz Verde) SÍ aparece en el resultado — la fila
completa, restricciones incluidas — porque el comercio existe en el listado;
decidir qué decir al respecto es responsabilidad de quien consume el
resultado (el generador), no de esta consulta.

Ver `rag/loaders/comercios_scraper.py` para el punto de extensión futuro
(reemplazar el CSV simulado por datos reales del buscador de Pluxee).
"""

import csv
import sqlite3
from pathlib import Path

from config import settings

TABLA = "comercios"

# Conexión en memoria, cargada de forma perezosa (mismo patrón que el modelo
# de embeddings en rag/retrieval/vector_retriever.py) — se reconstruye solo
# si se llama explícitamente a cargar_tabla_comercios() de nuevo.
_conn: sqlite3.Connection | None = None


def cargar_tabla_comercios(csv_path: str | Path = settings.COMERCIOS_CSV_PATH) -> None:
    """Carga el CSV a una tabla sqlite en memoria, reemplazando cualquier
    carga previa. Separado de `consultar_comercio` para poder recargar el
    dataset (ej. tras actualizar comercios.csv) sin reiniciar el proceso."""
    global _conn

    conn = sqlite3.connect(":memory:")
    conn.execute(
        f"""
        CREATE TABLE {TABLA} (
            id INTEGER PRIMARY KEY,
            nombre_comercio TEXT NOT NULL,
            categoria TEXT NOT NULL,
            comuna TEXT NOT NULL,
            restricciones TEXT NOT NULL,
            fecha_actualizacion TEXT NOT NULL
        )
        """
    )
    with open(csv_path, encoding="utf-8") as f:
        filas = [
            (
                int(fila["id"]),
                fila["nombre_comercio"],
                fila["categoria"],
                fila["comuna"],
                fila["restricciones"],
                fila["fecha_actualizacion"],
            )
            for fila in csv.DictReader(f)
        ]
    conn.executemany(f"INSERT INTO {TABLA} VALUES (?, ?, ?, ?, ?, ?)", filas)
    conn.commit()
    _conn = conn


def _get_conn() -> sqlite3.Connection:
    if _conn is None:
        cargar_tabla_comercios()
    return _conn


def consultar_comercio(nombre_o_categoria: str) -> list[dict]:
    """Busca comercios cuyo `nombre_comercio` o `categoria` contengan
    `nombre_o_categoria` (case-insensitive vía COLLATE NOCASE — no resuelve
    acentos, ej. "lider" no matchea "Líder", limitación aceptada dado el
    tamaño del dataset). Devuelve la fila completa como dict, o [] si no hay
    coincidencias."""
    conn = _get_conn()
    termino = f"%{nombre_o_categoria}%"
    cursor = conn.execute(
        f"""
        SELECT id, nombre_comercio, categoria, comuna, restricciones, fecha_actualizacion
        FROM {TABLA}
        WHERE nombre_comercio LIKE ? COLLATE NOCASE
           OR categoria LIKE ? COLLATE NOCASE
        """,
        (termino, termino),
    )
    columnas = [d[0] for d in cursor.description]
    return [dict(zip(columnas, fila)) for fila in cursor.fetchall()]
