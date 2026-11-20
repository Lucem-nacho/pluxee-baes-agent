"""Calcula las métricas de la evaluación a partir de la base de observabilidad
(solo ejecuciones con origen "evaluacion") y las guarda como JSON en
data/processed/eval/metricas.json.

Métricas: precisión de clasificación, de herramientas y de veredicto,
consistencia entre repeticiones, tasa de errores por tipo, p50/p95 de latencia
por etapa y por herramienta, y llamadas y tokens por consulta. Las definiciones
exactas están en `observabilidad/metricas.py`.

Todo se reporta también POR MODELO (el modelo registrado en cada ejecución):
si la cuota te obliga a usar más de uno, el bloque global los mezcla y el
script lo advierte; compara usando `por_modelo`. Las consultas de preparación
(`<caso>-prep`) y las ejecuciones sin caso no cuentan.

La precisión de veredicto sale de las etiquetas que llena el estudiante en
data/processed/eval/etiquetas.csv (columnas id_ejecucion, correcto = 1 | 0 |
parcial, comentario). Si el archivo no existe, se genera una plantilla con una
fila por ejecución y `correcto` vacío; llénala y vuelve a correr el script.

Uso:
    python scripts/calcular_metricas.py
    python scripts/calcular_metricas.py --incluir-ejemplos
    python scripts/calcular_metricas.py --etiquetas otro.csv --salida otro.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from observabilidad.casos import ErrorEsquema, cargar_casos
from observabilidad.metricas import (
    ErrorEtiquetas,
    calcular,
    cargar_ejecuciones,
    generar_plantilla_etiquetas,
    guardar_json,
    leer_etiquetas,
    resumen_consola,
)


def main() -> int:
    carpeta_eval = Path(settings.OBSERVABILIDAD_DB_PATH).parent / "eval"
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--casos-path", default=None, help="Ruta de casos.json (por defecto eval/casos.json).")
    parser.add_argument("--incluir-ejemplos", action="store_true", help="Incluye los casos marcados como ejemplo.")
    parser.add_argument("--etiquetas", default=str(carpeta_eval / "etiquetas.csv"), help="CSV de etiquetas del estudiante.")
    parser.add_argument("--salida", default=str(carpeta_eval / "metricas.json"), help="Ruta del JSON de salida.")
    args = parser.parse_args()

    avisos: list[str] = []
    try:
        casos = cargar_casos(args.casos_path, incluir_ejemplos=args.incluir_ejemplos, advertencias=avisos)
    except ErrorEsquema as error:
        print(error, file=sys.stderr)
        return 2
    for aviso in avisos:
        print(f"Aviso: {aviso}", file=sys.stderr)
    if not casos:
        print("Aviso: no hay casos activos (los marcados como ejemplo se ignoran; usa --incluir-ejemplos). Las métricas saldrán vacías.\n")

    ejecuciones = cargar_ejecuciones()

    etiquetas_path = Path(args.etiquetas)
    if etiquetas_path.exists():
        try:
            etiquetas = leer_etiquetas(etiquetas_path)
        except ErrorEtiquetas as error:
            print(error, file=sys.stderr)
            return 2
    else:
        escritas = generar_plantilla_etiquetas(etiquetas_path, casos, ejecuciones)
        etiquetas = {}
        print(f"No existía {etiquetas_path}: generé la plantilla con {escritas} fila(s). Llena la columna 'correcto' (1, 0 o parcial) y vuelve a correr.\n")

    metricas = calcular(casos, ejecuciones, etiquetas)
    salida = guardar_json(metricas, args.salida)
    print(resumen_consola(metricas))
    print(f"\nJSON: {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
