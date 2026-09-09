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

Cada llamada a `consultar_comercio` abre su propia conexión sqlite en
memoria, la usa y la cierra — NO se comparte una conexión global entre
llamadas. SQLite no permite reutilizar una misma conexión desde un hilo
distinto al que la creó, y Streamlit ejecuta cada interacción del usuario en
su propio hilo (`sqlite3.ProgrammingError: SQLite objects created in a
thread can only be used in that same thread` — encontrado en uso real de la
app). Con 15 filas, recargar el CSV en cada consulta es insignificante en
costo; como beneficio adicional, un cambio en comercios.csv se refleja de
inmediato sin reiniciar el proceso, sin necesidad de recargar nada a mano.

Ver `rag/loaders/comercios_scraper.py` para el punto de extensión futuro
(reemplazar el CSV simulado por datos reales del buscador de Pluxee).
"""

import csv
import sqlite3
from pathlib import Path

from config import settings

TABLA = "comercios"


def _construir_conexion(csv_path: str | Path) -> sqlite3.Connection:
    """Abre una conexión sqlite en memoria NUEVA, crea la tabla y carga el
    CSV completo. Quien la llama es responsable de cerrarla."""
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
    return conn


def cargar_tabla_comercios(csv_path: str | Path = settings.COMERCIOS_CSV_PATH) -> None:
    """Valida que el CSV cargue sin errores (usado por scripts/ingest_all.py
    como chequeo de sanidad al preparar el proyecto). No deja ningún estado
    en memoria para llamadas futuras: `consultar_comercio` vuelve a cargar
    el CSV por su cuenta en cada llamada de todas formas."""
    conn = _construir_conexion(csv_path)
    conn.close()


def consultar_comercio(nombre_o_categoria: str) -> list[dict]:
    """Busca comercios cuyo `nombre_comercio` o `categoria` contengan
    `nombre_o_categoria` (case-insensitive vía COLLATE NOCASE — no resuelve
    acentos, ej. "lider" no matchea "Líder", limitación aceptada dado el
    tamaño del dataset). Devuelve la fila completa como dict, o [] si no hay
    coincidencias."""
    conn = _construir_conexion(settings.COMERCIOS_CSV_PATH)
    try:
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
    finally:
        conn.close()
