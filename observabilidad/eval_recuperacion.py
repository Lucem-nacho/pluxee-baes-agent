"""Evaluación de la recuperación (sin LLM): ¿aparece la fuente esperada entre
los primeros fragmentos que devuelve `recuperar()`?

Para la pregunta original y cada variante de cada caso con `fuente_esperada`:

- `rango`: posición (1 = primero) del primer fragmento que contiene la fuente
  esperada, o vacío si no aparece entre los `k` recuperados.
- hit@3 / hit@5: 1 si `rango` <= 3 / <= 5, si no 0. El resumen es el promedio.
- MRR: promedio de 1/rango (0 cuando no hay acierto).

Un fragmento "contiene" la fuente esperada si el texto esperado aparece en el
texto del fragmento o en el nombre de su fuente, después de normalizar ambos:
se ignoran las mayúsculas y los espacios y saltos de línea repetidos (los PDFs
cortan las frases a mitad de renglón). Las tildes NO se ignoran; sí se
unifican las formas equivalentes de un mismo carácter (NFKC: una "é" escrita
como "e" + acento, o la ligadura "ﬁ" que deja la extracción de PDF).

Los casos con `fuente_esperada` vacía (comercios, derivación, memoria) se
omiten: `separar_evaluables` los devuelve aparte para poder informar cuántos fueron.

Mide la recuperación vectorial "cruda": no incluye el arbitraje por recencia
que aplica `buscar_normativa` después, ni lo que el agente reformula antes de
buscar. Los comercios no pasan por acá: viven en una tabla SQL, no en el
índice, por eso sus casos llevan `fuente_esperada` = null y se omiten.
"""

import csv
import unicodedata
from pathlib import Path
from typing import Callable

from observabilidad.casos import preguntas_del_caso

KS = (3, 5)
K_POR_DEFECTO = 10
COLUMNAS_CSV = ("caso_id", "categoria_caso", "variante", "fuente_esperada", "rango", "hit@3", "hit@5", "rr")


def normalizar(texto: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", texto).casefold().split())


def contiene_la_fuente(fragmento: dict, fuente_esperada: str) -> bool:
    esperado = normalizar(fuente_esperada)
    return esperado in normalizar(fragmento.get("texto", "")) or esperado in normalizar(
        fragmento.get("fuente", "")
    )


def rango_del_primer_acierto(fragmentos: list[dict], fuente_esperada: str) -> int | None:
    """Posición (desde 1) del primer fragmento que contiene la fuente esperada,
    tomando los fragmentos de menor a mayor distancia."""
    ordenados = sorted(fragmentos, key=lambda f: f.get("distancia", 0.0))
    for posicion, fragmento in enumerate(ordenados, start=1):
        if contiene_la_fuente(fragmento, fuente_esperada):
            return posicion
    return None


def hit_at_k(rangos: list[int | None], k: int) -> float:
    if not rangos:
        return 0.0
    return sum(1 for r in rangos if r is not None and r <= k) / len(rangos)


def mrr(rangos: list[int | None]) -> float:
    if not rangos:
        return 0.0
    return sum(1 / r for r in rangos if r is not None) / len(rangos)


def separar_evaluables(casos: list[dict]) -> tuple[list[dict], list[dict]]:
    """(evaluables, omitidos): los casos con `fuente_esperada` no vacía y los que
    no la tienen (vacía, null o ausente)."""
    evaluables = [c for c in casos if (c.get("fuente_esperada") or "").strip()]
    omitidos = [c for c in casos if not (c.get("fuente_esperada") or "").strip()]
    return evaluables, omitidos


def evaluar(
    casos: list[dict], recuperar_fn: Callable[[str, int], list[dict]], k: int = K_POR_DEFECTO
) -> list[dict]:
    """Una fila por (caso, pregunta o variante) de los casos con `fuente_esperada`."""
    filas = []
    for caso in separar_evaluables(casos)[0]:
        fuente = caso["fuente_esperada"]
        for texto in preguntas_del_caso(caso):
            rango = rango_del_primer_acierto(recuperar_fn(texto, k), fuente)
            filas.append(
                {
                    "caso_id": caso["id"],
                    "categoria_caso": caso["categoria_caso"],
                    "variante": texto,
                    "fuente_esperada": fuente,
                    "rango": rango,
                    **{f"hit@{n}": int(rango is not None and rango <= n) for n in KS},
                    "rr": 0.0 if rango is None else 1 / rango,
                }
            )
    return filas


def _resumen(filas: list[dict]) -> dict:
    rangos = [f["rango"] for f in filas]
    return {
        "n": len(filas),
        **{f"hit@{n}": hit_at_k(rangos, n) for n in KS},
        "mrr": mrr(rangos),
    }


def resumir(filas: list[dict]) -> dict:
    """Resumen global y por categoría de caso."""
    categorias = sorted({f["categoria_caso"] for f in filas})
    return {
        "global": _resumen(filas),
        "por_categoria_caso": {c: _resumen([f for f in filas if f["categoria_caso"] == c]) for c in categorias},
    }


def guardar_csv(filas: list[dict], ruta: str | Path) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_CSV)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({**fila, "rango": "" if fila["rango"] is None else fila["rango"]})
    return ruta
