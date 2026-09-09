"""Integración con el buscador real de comercios de Pluxee — NO IMPLEMENTADO EN ESTA VERSIÓN.

Qué haría (diseño previsto):
    Reemplazar `data/comercios/comercios.csv` (dataset simulado) por datos
    obtenidos del buscador web público de Pluxee, manteniendo la misma
    estructura de tabla (id, nombre_comercio, categoria, comuna,
    restricciones, fecha_actualizacion) para que `rag/retrieval/comercios_query.py`
    no requiera cambios.

Por qué queda fuera de esta versión:
    El buscador de Pluxee no expone una API pública y bloquea scraping
    automatizado. Sortear esa protección está fuera del alcance ético y
    técnico de este proyecto académico. Se documenta aquí como punto de
    extensión futuro (ej. vía acuerdo de datos con Pluxee o una API oficial),
    no como funcionalidad construida.
"""


def sincronizar_comercios_desde_pluxee() -> None:
    """Placeholder de la interfaz prevista. No implementado."""
    raise NotImplementedError(
        "Integración con el buscador real de Pluxee fuera de alcance — ver docstring del módulo."
    )
