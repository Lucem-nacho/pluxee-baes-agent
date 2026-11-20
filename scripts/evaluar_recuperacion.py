"""Evalúa la recuperación del índice vectorial sin llamar a ningún LLM.

Para la pregunta y cada variante de cada caso con `fuente_esperada`, llama a
`recuperar()` y calcula hit@3, hit@5 y MRR (definiciones en
`observabilidad/eval_recuperacion.py`). Guarda un CSV en
data/processed/eval/ y muestra el resumen por consola.

Uso:
    python scripts/evaluar_recuperacion.py
    python scripts/evaluar_recuperacion.py --incluir-ejemplos
    python scripts/evaluar_recuperacion.py --k 10 --casos id1,id2

Requiere el índice construido (`python scripts/ingest_all.py`). No consume
cuota de Gemini.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from observabilidad.casos import ErrorEsquema, cargar_casos
from observabilidad.eval_recuperacion import K_POR_DEFECTO, KS, evaluar, guardar_csv, resumir


def _porcentaje(valor: float) -> str:
    return f"{valor * 100:5.1f}%"


def _linea(nombre: str, resumen: dict) -> str:
    hits = "  ".join(f"hit@{n} {_porcentaje(resumen[f'hit@{n}'])}" for n in KS)
    return f"  {nombre:<22} n={resumen['n']:<3} {hits}  MRR {resumen['mrr']:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--casos-path", default=None, help="Ruta de casos.json (por defecto eval/casos.json).")
    parser.add_argument("--incluir-ejemplos", action="store_true", help="Incluye los casos marcados como ejemplo.")
    parser.add_argument("--casos", default=None, help="Ids separados por coma para filtrar.")
    parser.add_argument("--k", type=int, default=K_POR_DEFECTO, help=f"Fragmentos a recuperar por consulta (por defecto {K_POR_DEFECTO}).")
    parser.add_argument("--salida", default=None, help="Ruta del CSV (por defecto data/processed/eval/recuperacion_<fecha>.csv).")
    args = parser.parse_args()

    try:
        casos = cargar_casos(args.casos_path, incluir_ejemplos=args.incluir_ejemplos)
    except ErrorEsquema as error:
        print(error, file=sys.stderr)
        return 2
    if args.casos:
        pedidos = {c.strip() for c in args.casos.split(",") if c.strip()}
        desconocidos = pedidos - {c["id"] for c in casos}
        if desconocidos:
            print(f"Ids que no existen (o son ejemplos sin --incluir-ejemplos): {sorted(desconocidos)}", file=sys.stderr)
            return 2
        casos = [c for c in casos if c["id"] in pedidos]

    evaluables = [c for c in casos if c["fuente_esperada"]]
    if not evaluables:
        print("No hay casos con 'fuente_esperada' para evaluar (¿solo hay ejemplos? usa --incluir-ejemplos).")
        return 0

    from rag.retrieval.vector_retriever import recuperar  # carga el modelo de embeddings: import tardío

    filas = evaluar(casos, recuperar, k=args.k)

    salida = Path(args.salida) if args.salida else (
        Path(settings.OBSERVABILIDAD_DB_PATH).parent / "eval" / f"recuperacion_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    guardar_csv(filas, salida)

    resumen = resumir(filas)
    print(f"Recuperación sobre {len(evaluables)} caso(s) con fuente esperada, {len(filas)} consulta(s), k={args.k}")
    print(_linea("GLOBAL", resumen["global"]))
    for categoria, r in resumen["por_categoria_caso"].items():
        print(_linea(categoria, r))
    sin_acierto = [f for f in filas if f["rango"] is None]
    if sin_acierto:
        print(f"\nSin acierto en el top-{args.k} ({len(sin_acierto)}):")
        for f in sin_acierto:
            print(f"  [{f['caso_id']}] {f['variante']}")
    print(f"\nCSV: {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
