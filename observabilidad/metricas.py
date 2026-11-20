"""Métricas de la evaluación, calculadas a partir de la base de observabilidad.

Solo cuentan las ejecuciones con origen "evaluacion" que pertenecen al
conjunto de casos activo: se excluyen (y se informan) las que no tienen
`caso_id` (p. ej. las de `scripts/check_agent_executor.py`), las de casos o
variantes que ya no están en `casos.json`, y las de casos marcados "ejemplo"
cuando no se pidió incluirlos.

Qué ejecuciones entran en cada métrica:
- `tasa_errores`: todas las consideradas, ERROR incluidas (para eso existe).
- precisión, consistencia, latencias y consumo: solo las COMPLETAS (estado
  distinto de ERROR). Un 429 es del proveedor, no un acierto ni un fallo del
  agente. Para precisión y consistencia, además, si una misma (caso, variante,
  repetición) tiene varias ejecuciones se usa la última.

Definiciones:
- precisión de clasificación: `categoria` registrada == `categoria_esperada`.
- precisión de herramientas: el CONJUNTO de herramientas usadas == el conjunto
  esperado (el orden y las repeticiones no importan; usar una de más o una de
  menos cuenta como fallo).
- precisión de veredicto: viene de las etiquetas del estudiante (CSV con
  `id_ejecucion`, `correcto` = 1 | 0 | parcial, `comentario`). `precision_estricta`
  = correctas / etiquetadas; `precision_ponderada` cuenta `parcial` como 0,5.
- consistencia: para cada (caso, variante) con 2 o más repeticiones, qué
  fracción de las repeticiones coincide con la más frecuente, en veredicto, en
  secuencia de herramientas y en ambos a la vez (la métrica pedida).
  Las repeticiones se comparan solo entre la misma variante.
- veredicto de una ejecución (para la consistencia): HEURÍSTICO. ESCALAR ->
  derivar; ERROR -> error; sin_respuesta / limite_iteraciones -> no_confirmable;
  si no, se buscan marcas en el texto de la respuesta, con prioridad
  no_confirmable > prohibido > permitido, y si no hay ninguna, "indeterminado".
  Es frágil (p. ej. "sí puedes comprar X, pero no puedes comprar Y" sale
  prohibido): la columna OPCIONAL `veredicto_observado` del CSV de etiquetas
  (permitido | prohibido | no_confirmable | derivar) lo reemplaza.
- percentiles: interpolación lineal entre rangos vecinos (igual que numpy por
  defecto); p50 de 1..100 = 50,5.
"""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from observabilidad import store
from observabilidad.casos import VEREDICTOS, preguntas_del_caso
from observabilidad.eval_recuperacion import normalizar
from observabilidad.privacidad import enmascarar

COLUMNAS_ETIQUETAS = (
    "id_ejecucion",
    "correcto",
    "comentario",
    # A partir de acá son de contexto para quien etiqueta; no se leen.
    "caso_id",
    "variante",
    "repeticion",
    "veredicto_esperado",
    "respuesta",
)
VALORES_CORRECTO = ("1", "0", "parcial")
_PESO_CORRECTO = {"1": 1.0, "parcial": 0.5, "0": 0.0}

_MARCAS_NO_CONFIRMABLE = (
    "no mencionan", "no menciona", "no encontre", "no logre confirmar", "no cubre",
    "no hay informacion", "no tengo informacion", "no es posible confirmar",
    "confirmar con un ejecutivo", "consultar con un ejecutivo", "consultarlo con un ejecutivo",
)
_MARCAS_PROHIBIDO = (
    "no puedes", "no se puede", "esta prohibido", "prohibido", "no esta permitido",
    "no esta habilitado", "no es posible comprar", "no te permite",
)
_MARCAS_PERMITIDO = (
    "si puedes", "puedes comprar", "esta permitido", "se puede comprar",
    "es posible comprar", "si se puede",
)


class ErrorEtiquetas(ValueError):
    def __init__(self, errores: list[str], origen: str):
        self.errores = list(errores)
        detalle = "\n".join(f"  - {e}" for e in self.errores)
        super().__init__(f"{origen} tiene {len(self.errores)} problema(s):\n{detalle}")


# ------------------------------------------------------------------ utilidades


def percentil(valores: list[float], p: float) -> float | None:
    """Percentil `p` (0-100) con interpolación lineal; None si no hay valores."""
    if not valores:
        return None
    orden = sorted(valores)
    posicion = (len(orden) - 1) * p / 100
    bajo = math.floor(posicion)
    alto = min(bajo + 1, len(orden) - 1)
    return orden[bajo] + (orden[alto] - orden[bajo]) * (posicion - bajo)


def _resumen_latencia(valores: list[float]) -> dict:
    p50, p95 = percentil(valores, 50), percentil(valores, 95)
    return {
        "n": len(valores),
        "p50_ms": None if p50 is None else round(p50, 2),
        "p95_ms": None if p95 is None else round(p95, 2),
    }


def _precision(aciertos: list[bool]) -> dict:
    n = len(aciertos)
    return {"n": n, "correctas": sum(aciertos), "precision": (sum(aciertos) / n) if n else None}


def _media(valores: list[float]) -> float | None:
    return round(sum(valores) / len(valores), 2) if valores else None


# ------------------------------------------------------------- lectura de la base


def cargar_ejecuciones(db_path: str | Path | None = None) -> list[dict]:
    """Ejecuciones de origen "evaluacion", cada una con su lista de `pasos` y la
    tupla `herramientas` (en orden de uso)."""
    conn = store._conectar(db_path)
    try:
        ejecuciones = [dict(f) for f in conn.execute("SELECT * FROM ejecuciones WHERE origen = 'evaluacion' ORDER BY id")]
        por_id = {e["id"]: e for e in ejecuciones}
        for e in ejecuciones:
            e["pasos"] = []
        for paso in conn.execute("SELECT * FROM pasos ORDER BY id_ejecucion, orden"):
            if paso["id_ejecucion"] in por_id:
                por_id[paso["id_ejecucion"]]["pasos"].append(dict(paso))
        for e in ejecuciones:
            e["herramientas"] = tuple(p["herramienta"] for p in e["pasos"])
        return ejecuciones
    finally:
        conn.close()


def veredicto_observado(ejecucion: dict, override: str | None = None) -> str:
    if override:
        return override
    if ejecucion["estado"] == "ESCALAR":
        return "derivar"
    if ejecucion["estado"] == "ERROR":
        return "error"
    if ejecucion["tipo_error"] in ("sin_respuesta", "limite_iteraciones"):
        return "no_confirmable"
    texto = normalizar(ejecucion.get("respuesta") or "")
    for veredicto, marcas in (
        ("no_confirmable", _MARCAS_NO_CONFIRMABLE),
        ("prohibido", _MARCAS_PROHIBIDO),
        ("permitido", _MARCAS_PERMITIDO),
    ):
        if any(marca in texto for marca in marcas):
            return veredicto
    return "indeterminado"


# ------------------------------------------------------------------- selección


def seleccionar(casos: list[dict], ejecuciones: list[dict]) -> tuple[list[dict], int, int]:
    """(consideradas, sin_caso, fuera_del_conjunto) entre las de origen evaluacion."""
    por_id = {c["id"]: c for c in casos}
    textos = {c["id"]: {enmascarar(t) for t in preguntas_del_caso(c)} for c in casos}
    consideradas, sin_caso, fuera = [], 0, 0
    for e in ejecuciones:
        if e["origen"] != "evaluacion":
            continue
        if e["caso_id"] is None:
            sin_caso += 1
        elif e["caso_id"] not in por_id or e["variante"] not in textos[e["caso_id"]]:
            fuera += 1
        else:
            consideradas.append(e)
    return consideradas, sin_caso, fuera


def deduplicar(ejecuciones: list[dict]) -> list[dict]:
    """Una ejecución por (caso, variante, repetición): la de mayor id."""
    ultimas: dict[tuple, dict] = {}
    for e in ejecuciones:
        ultimas[(e["caso_id"], e["variante"], e["repeticion"])] = max(
            ultimas.get((e["caso_id"], e["variante"], e["repeticion"]), e), e, key=lambda x: x["id"]
        )
    return sorted(ultimas.values(), key=lambda x: x["id"])


# ---------------------------------------------------------------- etiquetas


def _delimitador(primera_linea: str) -> str:
    # Excel en español guarda los CSV con ';'.
    return ";" if primera_linea.count(";") > primera_linea.count(",") else ","


def leer_etiquetas(ruta: str | Path) -> dict[int, dict]:
    """{id_ejecucion: {"correcto", "comentario", "veredicto_observado"}} de las
    filas ya etiquetadas (`correcto` no vacío). Lanza `ErrorEtiquetas` con la
    línea de cada problema."""
    ruta = Path(ruta)
    origen = str(ruta)
    with open(ruta, newline="", encoding="utf-8-sig") as archivo:
        contenido = archivo.read()
    if not contenido.strip():
        return {}
    lector = csv.DictReader(contenido.splitlines(), delimiter=_delimitador(contenido.splitlines()[0]))
    faltantes = [c for c in ("id_ejecucion", "correcto") if c not in (lector.fieldnames or [])]
    if faltantes:
        raise ErrorEtiquetas([f"faltan las columnas {faltantes} en la fila de encabezado"], origen)

    etiquetas: dict[int, dict] = {}
    errores: list[str] = []
    for linea, fila in enumerate(lector, start=2):
        correcto = (fila.get("correcto") or "").strip().lower()
        if not correcto:
            continue  # sin etiquetar todavía
        try:
            id_ejecucion = int((fila.get("id_ejecucion") or "").strip())
        except ValueError:
            errores.append(f"línea {linea}: 'id_ejecucion' = {fila.get('id_ejecucion')!r} no es un número entero")
            continue
        if correcto not in VALORES_CORRECTO:
            errores.append(f"línea {linea}: 'correcto' = {fila.get('correcto')!r} no es válido; debe ser 1, 0 o parcial")
            continue
        observado = (fila.get("veredicto_observado") or "").strip().lower() or None
        if observado is not None and observado not in VEREDICTOS:
            errores.append(f"línea {linea}: 'veredicto_observado' = {observado!r} no es válido; debe ser uno de: {', '.join(VEREDICTOS)}")
            continue
        if id_ejecucion in etiquetas:
            errores.append(f"línea {linea}: la ejecución {id_ejecucion} está etiquetada dos veces")
            continue
        etiquetas[id_ejecucion] = {
            "correcto": correcto,
            "comentario": (fila.get("comentario") or "").strip(),
            "veredicto_observado": observado,
        }
    if errores:
        raise ErrorEtiquetas(errores, origen)
    return etiquetas


def generar_plantilla_etiquetas(ruta: str | Path, casos: list[dict], ejecuciones: list[dict]) -> int:
    """Crea el CSV de etiquetas con una fila por ejecución completa, con
    `correcto` y `comentario` vacíos y columnas de contexto (caso, pregunta,
    veredicto esperado, respuesta) para poder etiquetar sin abrir la base.
    Devuelve cuántas filas escribió."""
    consideradas, _, _ = seleccionar(casos, ejecuciones)
    unicas = deduplicar([e for e in consideradas if e["estado"] != "ERROR"])
    esperado = {c["id"]: c["veredicto_esperado"] for c in casos}

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(COLUMNAS_ETIQUETAS)
        for e in unicas:
            escritor.writerow(
                [e["id"], "", "", e["caso_id"], e["variante"], e["repeticion"], esperado[e["caso_id"]], e["respuesta"] or ""]
            )
    return len(unicas)


# ------------------------------------------------------------------- cálculo


def _consistencia(ejecuciones: list[dict], etiquetas: dict[int, dict]) -> dict:
    grupos: dict[tuple, list[dict]] = defaultdict(list)
    for e in ejecuciones:
        grupos[(e["caso_id"], e["variante"])].append(e)

    def sumar(conjunto: list[list[dict]]) -> dict:
        total = veredicto = secuencia = ambos = 0
        for grupo in conjunto:
            pares = [
                (veredicto_observado(e, (etiquetas.get(e["id"]) or {}).get("veredicto_observado")), e["herramientas"])
                for e in grupo
            ]
            total += len(pares)
            veredicto += Counter(p[0] for p in pares).most_common(1)[0][1]
            secuencia += Counter(p[1] for p in pares).most_common(1)[0][1]
            ambos += Counter(pares).most_common(1)[0][1]
        fraccion = lambda x: (x / total) if total else None
        return {
            "repeticiones": total,
            "mismo_veredicto": fraccion(veredicto),
            "misma_secuencia": fraccion(secuencia),
            "ambos": fraccion(ambos),
        }

    evaluables = [g for g in grupos.values() if len(g) >= 2]
    por_caso: dict[str, list[list[dict]]] = defaultdict(list)
    for (caso_id, _), grupo in grupos.items():
        if len(grupo) >= 2:
            por_caso[caso_id].append(grupo)
    return {
        "grupos_evaluables": len(evaluables),
        "grupos_sin_repeticion": len(grupos) - len(evaluables),
        **sumar(evaluables),
        "por_caso": {caso_id: sumar(gs) for caso_id, gs in sorted(por_caso.items())},
    }


def calcular(casos: list[dict], ejecuciones: list[dict], etiquetas: dict[int, dict] | None = None) -> dict:
    etiquetas = etiquetas or {}
    por_id = {c["id"]: c for c in casos}
    consideradas, sin_caso, fuera = seleccionar(casos, ejecuciones)
    completas = [e for e in consideradas if e["estado"] != "ERROR"]
    unicas = deduplicar(completas)

    clasificacion = [(e, e["categoria"] == por_id[e["caso_id"]]["categoria_esperada"]) for e in unicas]
    herramientas = [
        (e, set(e["herramientas"]) == set(por_id[e["caso_id"]]["herramientas_esperadas"])) for e in unicas
    ]

    # --- veredicto, a partir de las etiquetas del estudiante
    ids_completas = {e["id"] for e in completas}
    etiquetadas = [(e, etiquetas[e["id"]]["correcto"]) for e in completas if e["id"] in etiquetas]
    valores = [_PESO_CORRECTO[c] for _, c in etiquetadas]
    conteo = Counter(c for _, c in etiquetadas)
    veredicto = {
        "etiquetadas": len(etiquetadas),
        "correctas": conteo["1"],
        "parciales": conteo["parcial"],
        "incorrectas": conteo["0"],
        "sin_etiquetar": sum(1 for e in unicas if e["id"] not in etiquetas),
        "etiquetas_de_ejecuciones_ajenas": sum(1 for i in etiquetas if i not in ids_completas),
        "precision_estricta": (conteo["1"] / len(etiquetadas)) if etiquetadas else None,
        "precision_ponderada": (sum(valores) / len(valores)) if valores else None,
    }

    # --- errores
    total = len(consideradas)
    por_tipo = Counter(e["tipo_error"] for e in consideradas)
    errores = {
        "ejecuciones": total,
        "por_tipo": {
            tipo: {"n": por_tipo.get(tipo, 0), "tasa": (por_tipo.get(tipo, 0) / total) if total else None}
            for tipo in store.TIPOS_ERROR
        },
    }

    # --- latencias
    duraciones: dict[str, list[int]] = defaultdict(list)
    for e in completas:
        for paso in e["pasos"]:
            if paso["duracion_ms"] is not None:
                duraciones[paso["herramienta"]].append(paso["duracion_ms"])
    latencia = {
        "por_etapa": {
            etapa: _resumen_latencia([e[columna] for e in completas if e[columna] is not None])
            for etapa, columna in (
                ("total", "latencia_total_ms"),
                ("clasificador", "latencia_clasificador_ms"),
                ("agente", "latencia_agente_ms"),
            )
        },
        "por_herramienta": {h: _resumen_latencia(v) for h, v in sorted(duraciones.items())},
    }

    # --- tokens y llamadas por consulta
    def consumo(grupo: list[dict]) -> dict:
        t_in = [e["tokens_entrada"] for e in grupo if e["tokens_entrada"] is not None]
        t_out = [e["tokens_salida"] for e in grupo if e["tokens_salida"] is not None]
        return {
            "consultas": len(grupo),
            "llamadas_llm_promedio": _media([e["llamadas_llm"] for e in grupo]),
            "tokens_entrada_promedio": _media(t_in),
            "tokens_salida_promedio": _media(t_out),
            "consultas_con_tokens": len(t_in),
        }

    # --- por caso
    consistencia = _consistencia(unicas, etiquetas)
    por_caso = {}
    for caso_id in sorted({e["caso_id"] for e in unicas}):
        suyas = [e for e in unicas if e["caso_id"] == caso_id]
        por_caso[caso_id] = {
            "ejecuciones": len(suyas),
            "precision_clasificacion": _precision([ok for e, ok in clasificacion if e["caso_id"] == caso_id])["precision"],
            "precision_herramientas": _precision([ok for e, ok in herramientas if e["caso_id"] == caso_id])["precision"],
            "consistencia": (consistencia["por_caso"].get(caso_id) or {}).get("ambos"),
            "consumo": consumo(suyas),
        }

    return {
        "ejecuciones": {
            "evaluacion_total": sum(1 for e in ejecuciones if e["origen"] == "evaluacion"),
            "sin_caso_excluidas": sin_caso,
            "fuera_del_conjunto_excluidas": fuera,
            "consideradas": len(consideradas),
            "completas": len(completas),
            "con_error": len(consideradas) - len(completas),
            "unicas": len(unicas),
        },
        "precision_clasificacion": _precision([ok for _, ok in clasificacion]),
        "precision_herramientas": _precision([ok for _, ok in herramientas]),
        "precision_veredicto": veredicto,
        "consistencia": consistencia,
        "tasa_errores": errores,
        "latencia_ms": latencia,
        "consumo_por_consulta": consumo(completas),
        "por_caso": por_caso,
    }


def guardar_json(metricas: dict, ruta: str | Path) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


def _pct(valor: float | None) -> str:
    return "—" if valor is None else f"{valor * 100:.1f}%"


def resumen_consola(m: dict) -> str:
    e = m["ejecuciones"]
    lineas = [
        f"Ejecuciones de evaluación: {e['evaluacion_total']} en la base | consideradas {e['consideradas']} "
        f"(completas {e['completas']}, con error {e['con_error']}, únicas {e['unicas']})",
        f"  excluidas: {e['sin_caso_excluidas']} sin caso, {e['fuera_del_conjunto_excluidas']} fuera del conjunto activo",
        "",
    ]
    for nombre, clave in (("Precisión de clasificación", "precision_clasificacion"), ("Precisión de herramientas", "precision_herramientas")):
        r = m[clave]
        lineas.append(f"{nombre}: {_pct(r['precision'])} ({r['correctas']}/{r['n']})")
    v = m["precision_veredicto"]
    lineas.append(
        f"Precisión de veredicto: estricta {_pct(v['precision_estricta'])} | ponderada {_pct(v['precision_ponderada'])} "
        f"({v['etiquetadas']} etiquetadas, {v['sin_etiquetar']} sin etiquetar)"
    )
    c = m["consistencia"]
    lineas.append(
        f"Consistencia: veredicto {_pct(c['mismo_veredicto'])} | herramientas {_pct(c['misma_secuencia'])} | "
        f"ambos {_pct(c['ambos'])} ({c['grupos_evaluables']} grupos con repeticiones, {c['grupos_sin_repeticion']} sin)"
    )
    con_error = {t: x for t, x in m["tasa_errores"]["por_tipo"].items() if t != "ninguno" and x["n"]}
    lineas.append("Errores: " + (", ".join(f"{t} {x['n']} ({_pct(x['tasa'])})" for t, x in con_error.items()) or "ninguno"))
    lineas.append("")
    lineas.append("Latencia (p50 / p95, ms):")
    for etapa, r in m["latencia_ms"]["por_etapa"].items():
        lineas.append(f"  {etapa:<13} {r['p50_ms']} / {r['p95_ms']}  (n={r['n']})")
    for herramienta, r in m["latencia_ms"]["por_herramienta"].items():
        lineas.append(f"  {herramienta:<20} {r['p50_ms']} / {r['p95_ms']}  (n={r['n']})")
    k = m["consumo_por_consulta"]
    lineas.append("")
    lineas.append(
        f"Por consulta: {k['llamadas_llm_promedio']} llamadas al LLM, {k['tokens_entrada_promedio']} tokens de entrada, "
        f"{k['tokens_salida_promedio']} de salida (promedios; tokens en {k['consultas_con_tokens']}/{k['consultas']} consultas)"
    )
    return "\n".join(lineas)
