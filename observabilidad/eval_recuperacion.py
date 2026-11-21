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
como "e" + acento, o dígitos de ancho completo).

Dos métricas lado a lado, siempre con las mismas columnas (hit@3, hit@5, MRR):

- ESTRICTA: solo cuenta `fuente_esperada`.
- AMPLIADA: cuenta `fuente_esperada` o cualquiera de `fuentes_alternativas`
  (frases literales que el estudiante decide agregar después de leer los
  fragmentos con `--detalle`; este módulo nunca las inventa). Si ningún caso
  tiene alternativas, las dos métricas coinciden.

Solo se consideran los `k` fragmentos más cercanos (menor distancia) de los que
devuelve `recuperar()`, que busca en dos colecciones y puede devolver hasta 2*k.

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
TOP_DETALLE = 5  # fragmentos que muestra --detalle por consulta sin acierto
TOP_CSV = 3  # fragmentos que se guardan en el CSV por consulta
CARACTERES_DETALLE = 200
CARACTERES_CSV = 150
METRICAS = ("estricta", "ampliada")
SEPARADOR_ALTERNATIVAS = " | "

COLUMNAS_CSV = (
    "caso_id", "categoria_caso", "variante", "fuente_esperada", "fuentes_alternativas",
    "rango", "hit@3", "hit@5", "rr",
    "rango_ampliado", "hit@3_ampliado", "hit@5_ampliado", "rr_ampliado",
    *(f"top{n}_{campo}" for n in range(1, TOP_CSV + 1) for campo in ("fuente", "texto", "distancia")),
)


def normalizar(texto: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", texto).casefold().split())


def _una_linea(texto: object, maximo: int) -> str:
    """Los primeros `maximo` caracteres del texto, sin saltos de línea."""
    return " ".join(str(texto).split())[:maximo]


def contiene_la_fuente(fragmento: dict, fuente_esperada: str) -> bool:
    esperado = normalizar(fuente_esperada)
    return esperado in normalizar(fragmento.get("texto", "")) or esperado in normalizar(
        fragmento.get("fuente", "")
    )


def ordenar_por_distancia(fragmentos: list[dict]) -> list[dict]:
    return sorted(fragmentos, key=lambda f: f.get("distancia", 0.0))


def _rango_con_alguna(ordenados: list[dict], frases: list[str]) -> int | None:
    for posicion, fragmento in enumerate(ordenados, start=1):
        if any(contiene_la_fuente(fragmento, frase) for frase in frases):
            return posicion
    return None


def rango_del_primer_acierto(fragmentos: list[dict], fuente_esperada: str) -> int | None:
    """Posición (desde 1) del primer fragmento que contiene la fuente esperada,
    tomando los fragmentos de menor a mayor distancia. Es el rango ESTRICTO."""
    return _rango_con_alguna(ordenar_por_distancia(fragmentos), [fuente_esperada])


def rango_ampliado(
    fragmentos: list[dict], fuente_esperada: str, fuentes_alternativas: list[str] | None = None
) -> int | None:
    """Como `rango_del_primer_acierto`, pero el fragmento también acierta si
    contiene cualquiera de las `fuentes_alternativas`."""
    return _rango_con_alguna(ordenar_por_distancia(fragmentos), [fuente_esperada, *(fuentes_alternativas or [])])


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


def _top(ordenados: list[dict], cantidad: int) -> list[dict]:
    return [
        {
            "posicion": posicion,
            "fuente": fragmento.get("fuente") or "(sin fuente)",
            "texto": fragmento.get("texto", ""),
            "distancia": fragmento.get("distancia"),  # None si el recuperador no la entrega
        }
        for posicion, fragmento in enumerate(ordenados[:cantidad], start=1)
    ]


def _campos_de_acierto(rango: int | None, sufijo: str = "") -> dict:
    return {
        f"rango{sufijo}": rango,
        **{f"hit@{n}{sufijo}": int(rango is not None and rango <= n) for n in KS},
        f"rr{sufijo}": 0.0 if rango is None else 1 / rango,
    }


def evaluar(
    casos: list[dict], recuperar_fn: Callable[[str, int], list[dict]], k: int = K_POR_DEFECTO
) -> list[dict]:
    """Una fila por (caso, pregunta o variante) de los casos con `fuente_esperada`.
    Cada fila trae el acierto estricto (`rango`, `hit@n`, `rr`), el ampliado (los
    mismos campos con sufijo `_ampliado`) y `top`: los mejores fragmentos."""
    filas = []
    for caso in separar_evaluables(casos)[0]:
        fuente = caso["fuente_esperada"]
        alternativas = list(caso.get("fuentes_alternativas") or [])
        for texto in preguntas_del_caso(caso):
            ordenados = ordenar_por_distancia(recuperar_fn(texto, k))[:k]  # el top-k de verdad
            estricto = _rango_con_alguna(ordenados, [fuente])
            ampliado = _rango_con_alguna(ordenados, [fuente, *alternativas])
            filas.append(
                {
                    "caso_id": caso["id"],
                    "categoria_caso": caso["categoria_caso"],
                    "variante": texto,
                    "fuente_esperada": fuente,
                    "fuentes_alternativas": alternativas,
                    **_campos_de_acierto(estricto),
                    **_campos_de_acierto(ampliado, "_ampliado"),
                    "top": _top(ordenados, max(TOP_DETALLE, TOP_CSV)),
                }
            )
    return filas


def _metricas(filas: list[dict], sufijo: str) -> dict:
    rangos = [f["rango" + sufijo] for f in filas]
    return {**{f"hit@{n}": hit_at_k(rangos, n) for n in KS}, "mrr": mrr(rangos)}


def _resumen(filas: list[dict]) -> dict:
    return {
        "n": len(filas),
        "con_alternativas": sum(1 for f in filas if f["fuentes_alternativas"]),  # consultas de casos con alternativas
        "estricta": _metricas(filas, ""),
        "ampliada": _metricas(filas, "_ampliado"),
    }


def resumir(filas: list[dict]) -> dict:
    """Resumen global y por categoría de caso; cada bloque trae las dos métricas
    (`estricta` y `ampliada`) con hit@3, hit@5 y mrr."""
    categorias = sorted({f["categoria_caso"] for f in filas})
    return {
        "global": _resumen(filas),
        "por_categoria_caso": {c: _resumen([f for f in filas if f["categoria_caso"] == c]) for c in categorias},
    }


def formatear_detalle(fila: dict, cantidad: int = TOP_DETALLE) -> list[str]:
    """Las líneas que `--detalle` imprime para una consulta: qué se esperaba y los
    primeros fragmentos recuperados (posición, fuente, 200 caracteres sin saltos
    de línea y distancia, si el recuperador la entregó)."""
    lineas = [f"[{fila['caso_id']}] {fila['variante']}", f"    esperada: {_una_linea(fila['fuente_esperada'], 300)}"]
    if fila["fuentes_alternativas"]:
        rango = fila["rango_ampliado"]
        lineas.append(
            "    alternativas: " + SEPARADOR_ALTERNATIVAS.join(_una_linea(a, 120) for a in fila["fuentes_alternativas"])
            + (f"  (la ampliada acierta en el rango {rango})" if rango is not None else "  (ninguna aparece tampoco)")
        )
    if not fila["top"]:
        lineas.append("    (el recuperador no devolvió fragmentos)")
    for fragmento in fila["top"][:cantidad]:
        distancia = fragmento["distancia"]
        puntaje = "" if distancia is None else f" (distancia {distancia:.4f})"
        lineas.append(f"    {fragmento['posicion']}. {fragmento['fuente']}{puntaje}")
        lineas.append(f"       {_una_linea(fragmento['texto'], CARACTERES_DETALLE)}")
    return lineas


def _fila_csv(fila: dict) -> dict:
    salida = {
        columna: ("" if fila.get(columna) is None else fila[columna])
        for columna in COLUMNAS_CSV
        if not columna.startswith("top") and columna != "fuentes_alternativas"
    }
    salida["fuentes_alternativas"] = SEPARADOR_ALTERNATIVAS.join(fila["fuentes_alternativas"])
    for n in range(1, TOP_CSV + 1):
        fragmento = fila["top"][n - 1] if len(fila["top"]) >= n else None
        salida[f"top{n}_fuente"] = "" if fragmento is None else fragmento["fuente"]
        salida[f"top{n}_texto"] = "" if fragmento is None else _una_linea(fragmento["texto"], CARACTERES_CSV)
        salida[f"top{n}_distancia"] = (
            "" if fragmento is None or fragmento["distancia"] is None else round(fragmento["distancia"], 4)
        )
    return salida


def guardar_csv(filas: list[dict], ruta: str | Path) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_CSV)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow(_fila_csv(fila))
    return ruta
