"""Evalúa la recuperación del índice vectorial sin llamar a ningún LLM.

Para la pregunta y cada variante de cada caso con `fuente_esperada`, llama a
`recuperar()` y calcula hit@3, hit@5 y MRR (definiciones en
`observabilidad/eval_recuperacion.py`), siempre en dos versiones lado a lado:
la ESTRICTA (solo `fuente_esperada`) y la AMPLIADA (también cuenta cualquiera
de las `fuentes_alternativas` del caso). Guarda un CSV en data/processed/eval/
(con el top-3 de fragmentos de cada consulta, para revisarlo en Excel) y
muestra el resumen por consola.

Uso:
    python scripts/evaluar_recuperacion.py
    python scripts/evaluar_recuperacion.py --detalle
    python scripts/evaluar_recuperacion.py --incluir-ejemplos
    python scripts/evaluar_recuperacion.py --k 10 --casos id1,id2

`--detalle` imprime, para cada consulta sin acierto estricto en el top-k, los 5
primeros fragmentos recuperados: sirve para decidir qué frases agregar como
`fuentes_alternativas` en eval/casos.json (el script no las agrega solo).

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
from observabilidad.eval_recuperacion import (
    K_POR_DEFECTO,
    KS,
    TOP_DETALLE,
    evaluar,
    formatear_detalle,
    guardar_csv,
    resumir,
    separar_evaluables,
)


def _porcentaje(valor: float) -> str:
    return f"{valor * 100:5.1f}%"


def _metricas(bloque: dict) -> str:
    hits = "  ".join(f"hit@{n} {_porcentaje(bloque[f'hit@{n}'])}" for n in KS)
    return f"{hits}  MRR {bloque['mrr']:.3f}"


def _linea(nombre: str, resumen: dict) -> str:
    """Estricta y ampliada en la misma línea, con las mismas columnas."""
    return f"  {nombre:<22} n={resumen['n']:<3} | {_metricas(resumen['estricta'])} | {_metricas(resumen['ampliada'])}"


def _encabezado() -> str:
    ancho = len(_metricas({"hit@3": 0, "hit@5": 0, "mrr": 0}))
    return f"  {'':<22} {'':<5} | {'ESTRICTA (fuente_esperada)':<{ancho}} | AMPLIADA (+ fuentes_alternativas)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--casos-path", default=None, help="Ruta de casos.json (por defecto eval/casos.json).")
    parser.add_argument("--incluir-ejemplos", action="store_true", help="Incluye los casos marcados como ejemplo.")
    parser.add_argument("--casos", default=None, help="Ids separados por coma para filtrar.")
    parser.add_argument("--k", type=int, default=K_POR_DEFECTO, help=f"Fragmentos a recuperar por consulta (por defecto {K_POR_DEFECTO}).")
    parser.add_argument(
        "--detalle", action="store_true",
        help=f"Para cada consulta sin acierto estricto, imprime los {TOP_DETALLE} primeros fragmentos recuperados.",
    )
    parser.add_argument("--salida", default=None, help="Ruta del CSV (por defecto data/processed/eval/recuperacion_<fecha>.csv).")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k debe ser al menos 1")

    avisos: list[str] = []
    try:
        casos = cargar_casos(args.casos_path, incluir_ejemplos=args.incluir_ejemplos, advertencias=avisos)
    except ErrorEsquema as error:
        print(error, file=sys.stderr)
        return 2
    for aviso in avisos:
        print(f"Aviso: {aviso}", file=sys.stderr)
    if args.casos:
        pedidos = {c.strip() for c in args.casos.split(",") if c.strip()}
        desconocidos = pedidos - {c["id"] for c in casos}
        if desconocidos:
            print(f"Ids que no existen (o son ejemplos sin --incluir-ejemplos): {sorted(desconocidos)}", file=sys.stderr)
            return 2
        casos = [c for c in casos if c["id"] in pedidos]

    evaluables, omitidos = separar_evaluables(casos)
    if omitidos:
        print(f"Omití {len(omitidos)} caso(s) sin 'fuente_esperada' (no dependen del índice vectorial): {', '.join(c['id'] for c in omitidos)}")
    if not evaluables:
        print("No hay casos con 'fuente_esperada' para evaluar (¿solo hay ejemplos? usa --incluir-ejemplos).")
        return 0

    from rag.retrieval import vector_retriever  # carga el modelo de embeddings: import tardío

    filas = evaluar(casos, vector_retriever.recuperar, k=args.k)

    salida = Path(args.salida) if args.salida else (
        Path(settings.OBSERVABILIDAD_DB_PATH).parent / "eval" / f"recuperacion_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    guardar_csv(filas, salida)

    resumen = resumir(filas)
    print(f"Recuperación sobre {len(evaluables)} caso(s) con fuente esperada ({len(omitidos)} omitido(s)), {len(filas)} consulta(s), k={args.k}")
    print(_encabezado())
    print(_linea("GLOBAL", resumen["global"]))
    for categoria, r in resumen["por_categoria_caso"].items():
        print(_linea(categoria, r))
    con_alternativas = resumen["global"]["con_alternativas"]
    if con_alternativas:
        print(f"La ampliada usa las fuentes_alternativas de {con_alternativas} de las {len(filas)} consulta(s).")
    else:
        print("Ningún caso tiene 'fuentes_alternativas': la métrica ampliada es igual a la estricta.")

    sin_acierto = [f for f in filas if f["rango"] is None]
    if sin_acierto:
        print(f"\nSin acierto estricto en el top-{args.k} ({len(sin_acierto)}):")
        for f in sin_acierto:
            salvada = f" (la ampliada acierta en el rango {f['rango_ampliado']})" if f["rango_ampliado"] is not None else ""
            print(f"  [{f['caso_id']}] {f['variante']}{salvada}")
        if args.detalle:
            print(f"\nDetalle: los {TOP_DETALLE} primeros fragmentos de cada consulta sin acierto (menor distancia primero)")
            for f in sin_acierto:
                print()
                print("\n".join(formatear_detalle(f)))
        else:
            print("(usa --detalle para ver los fragmentos que sí se recuperaron)")
    print(f"\nCSV: {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
