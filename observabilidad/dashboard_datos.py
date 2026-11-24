"""Lógica de datos del dashboard (`dashboard.py`), sin nada de Streamlit.

Todo lo que el dashboard calcula vive acá, en funciones que reciben y devuelven
DataFrames o diccionarios, para poder probarlas con bases temporales.

- LECTURA SOLO LECTURA: la base se abre con `mode=ro`. `store._conectar` crea el
  archivo y las tablas si no existen, y un dashboard que solo mira no debe
  fabricar una base vacía (ni dejar de avisar que no existe).
- Las definiciones son las de `observabilidad/metricas.py` (se reutilizan sus
  funciones): completas = estado distinto de ERROR; percentiles con
  interpolación lineal (`percentil`); precisión, consistencia y veredicto vía
  `calcular`; recuperación vía `resumir`.
- "Con error" en los indicadores = ejecuciones con `tipo_error` distinto de
  "ninguno" (incluye `limite_iteraciones` y `sin_respuesta`, que terminan con
  estado INFORMATIVA pero son fallas). Es la misma cuenta que el desglose por tipo.
- Los filtros de la barra lateral se aplican al DataFrame de ejecuciones; la
  calidad se calcula sobre lo que queda. La recuperación (un CSV aparte) no
  depende de esos filtros.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from config import settings
from observabilidad import store
from observabilidad.eval_recuperacion import KS, resumir
from observabilidad.metricas import ErrorEtiquetas, calcular, leer_etiquetas, percentil

SEPARADOR_ALTERNATIVAS = " | "
PATRON_CSV_RECUPERACION = "recuperacion_*.csv"
LIMITE_TRAZAS = 300

# Los seis tipos de error que muestra el dashboard (todos menos "ninguno").
ETIQUETAS_TIPO_ERROR = {
    "429": "429 (cuota)",
    "503": "503 (servicio no disponible)",
    "limite_iteraciones": "Límite de iteraciones",
    "sin_respuesta": "Sin respuesta",
    "clasificador_no_interpretable": "Clasificador no interpretable",
    "otro": "Otro",
}
ETAPAS = (
    ("total", "Total", "latencia_total_ms"),
    ("clasificador", "Clasificador", "latencia_clasificador_ms"),
    ("agente", "Agente", "latencia_agente_ms"),
)
COLUMNAS_EJECUCION = ("id", *store._COLUMNAS_EJECUCION)
COLUMNAS_PASO = ("id", "id_ejecucion", *store._COLUMNAS_PASO)


# ------------------------------------------------------------------ la base


@dataclass(frozen=True)
class EstadoBase:
    estado: str  # "ok" | "no_existe" | "sin_tablas" | "vacia" | "ilegible"
    total: int
    mensaje: str

    @property
    def utilizable(self) -> bool:
        return self.estado == "ok"


def ruta_base(db_path: str | Path | None = None) -> Path:
    return Path(db_path or settings.OBSERVABILIDAD_DB_PATH)


def carpeta_eval(db_path: str | Path | None = None) -> Path:
    """Donde los scripts dejan las etiquetas y los CSV de recuperación."""
    return ruta_base(db_path).parent / "eval"


def _conectar_lectura(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Conexión de SOLO LECTURA: falla si el archivo no existe, nunca lo crea."""
    ruta = ruta_base(db_path).resolve()
    return sqlite3.connect(f"{ruta.as_uri()}?mode=ro", uri=True)


def diagnosticar_base(db_path: str | Path | None = None) -> EstadoBase:
    ruta = ruta_base(db_path)
    if not ruta.exists():
        return EstadoBase(
            "no_existe", 0,
            f"Todavía no existe la base de observabilidad ({ruta}). Se crea sola cuando el agente "
            "responde una consulta (app.py) o al correr scripts/evaluar_agente.py.",
        )
    try:
        conn = _conectar_lectura(ruta)
        try:
            tablas = {f[0] for f in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            if "ejecuciones" not in tablas:
                return EstadoBase("sin_tablas", 0, f"El archivo {ruta} existe pero no tiene la tabla de ejecuciones.")
            total = conn.execute("SELECT COUNT(*) FROM ejecuciones").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error as error:
        return EstadoBase("ilegible", 0, f"No se pudo leer {ruta}: {error}")
    if total == 0:
        return EstadoBase(
            "vacia", 0,
            f"La base ({ruta}) existe pero todavía no tiene ejecuciones registradas. "
            "Usa el agente en app.py o corre scripts/evaluar_agente.py para generar datos.",
        )
    return EstadoBase("ok", total, f"{total} ejecución(es) en la base.")


def cargar_ejecuciones_df(db_path: str | Path | None = None) -> pd.DataFrame:
    """TODAS las ejecuciones (de la app y de la evaluación), ordenadas por id.
    Agrega `fecha` (datetime UTC) y `dia` (fecha UTC). Si la base viene de una
    versión anterior del esquema, las columnas que faltan quedan vacías."""
    conn = _conectar_lectura(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM ejecuciones ORDER BY id", conn)
    finally:
        conn.close()
    for columna in COLUMNAS_EJECUCION:
        if columna not in df.columns:
            df[columna] = None
    df["fecha"] = pd.to_datetime(df["fecha_utc"], utc=True, format="ISO8601", errors="coerce")
    df["dia"] = df["fecha"].dt.date
    return df.reset_index(drop=True)


def cargar_pasos_df(db_path: str | Path | None = None) -> pd.DataFrame:
    conn = _conectar_lectura(db_path)
    try:
        tablas = {f[0] for f in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "pasos" not in tablas:
            return pd.DataFrame(columns=list(COLUMNAS_PASO))
        return pd.read_sql_query("SELECT * FROM pasos ORDER BY id_ejecucion, orden", conn)
    finally:
        conn.close()


# ------------------------------------------------------------------ filtros


@dataclass
class Filtros:
    """Un valor vacío (None o tupla vacía) significa "sin filtro"."""

    desde: date | None = None  # inclusive, en UTC
    hasta: date | None = None  # inclusive, en UTC
    origenes: tuple[str, ...] = field(default_factory=tuple)
    modelos: tuple[str, ...] = field(default_factory=tuple)
    estados: tuple[str, ...] = field(default_factory=tuple)
    tipos_error: tuple[str, ...] = field(default_factory=tuple)


def valores_de(df: pd.DataFrame, columna: str) -> list[str]:
    if df.empty or columna not in df:
        return []
    return sorted(str(v) for v in df[columna].dropna().unique())


def rango_de_fechas(df: pd.DataFrame) -> tuple[date, date] | None:
    dias = df["dia"].dropna() if "dia" in df else []
    return (min(dias), max(dias)) if len(dias) else None


def aplicar_filtros(df: pd.DataFrame, filtros: Filtros) -> pd.DataFrame:
    mascara = pd.Series(True, index=df.index)
    if filtros.desde is not None:
        mascara &= df["dia"].apply(lambda d: d is not None and not pd.isna(d) and d >= filtros.desde)
    if filtros.hasta is not None:
        mascara &= df["dia"].apply(lambda d: d is not None and not pd.isna(d) and d <= filtros.hasta)
    for columna, valores in (
        ("origen", filtros.origenes),
        ("modelo", filtros.modelos),
        ("estado", filtros.estados),
        ("tipo_error", filtros.tipos_error),
    ):
        if valores:
            mascara &= df[columna].isin(valores)
    return df[mascara]


def _completas(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["estado"] != "ERROR"]


def _valores(serie: pd.Series) -> list[float]:
    return [float(v) for v in serie.dropna().tolist()]


def _media(serie: pd.Series) -> float | None:
    valores = _valores(serie)
    return sum(valores) / len(valores) if valores else None


# --------------------------------------------------------------- indicadores


def indicadores(df: pd.DataFrame) -> dict:
    """Los indicadores de arriba. Latencia, tokens y llamadas se calculan sobre
    las ejecuciones COMPLETAS (estado distinto de ERROR), como en metricas.py."""
    total = len(df)
    completas = _completas(df)
    con_error = int((df["tipo_error"] != "ninguno").sum()) if total else 0
    tokens = completas["tokens_entrada"] + completas["tokens_salida"]
    return {
        "ejecuciones": total,
        "completas": len(completas),
        "con_error": con_error,
        "con_estado_error": int((df["estado"] == "ERROR").sum()) if total else 0,
        "porcentaje_error": (con_error / total) if total else None,
        "latencia_total_p50_ms": percentil(_valores(completas["latencia_total_ms"]), 50),
        "latencia_total_p95_ms": percentil(_valores(completas["latencia_total_ms"]), 95),
        "tokens_promedio": _media(tokens),  # entrada + salida, solo donde hay ambos
        "tokens_entrada_promedio": _media(completas["tokens_entrada"]),
        "tokens_salida_promedio": _media(completas["tokens_salida"]),
        "consultas_con_tokens": int(tokens.notna().sum()) if len(completas) else 0,
        "llamadas_llm_promedio": _media(completas["llamadas_llm"]),
    }


# ------------------------------------------------------------------ latencia


def _fila_latencia(nombre: str, valores: list[float]) -> dict:
    return {"nombre": nombre, "n": len(valores), "p50_ms": percentil(valores, 50), "p95_ms": percentil(valores, 95)}


def latencia_por_etapa(df: pd.DataFrame) -> pd.DataFrame:
    completas = _completas(df)
    filas = [_fila_latencia(nombre, _valores(completas[columna])) for _, nombre, columna in ETAPAS]
    return pd.DataFrame(filas, columns=["nombre", "n", "p50_ms", "p95_ms"]).rename(columns={"nombre": "etapa"})


def latencia_por_herramienta(df: pd.DataFrame, pasos: pd.DataFrame) -> pd.DataFrame:
    ids = set(_completas(df)["id"])
    utiles = pasos[pasos["id_ejecucion"].isin(ids) & pasos["duracion_ms"].notna()]
    filas = [
        _fila_latencia(herramienta, _valores(grupo["duracion_ms"]))
        for herramienta, grupo in sorted(utiles.groupby("herramienta"), key=lambda x: x[0])
    ]
    return pd.DataFrame(filas, columns=["nombre", "n", "p50_ms", "p95_ms"]).rename(columns={"nombre": "herramienta"})


# ------------------------------------------------------------------- errores


def errores_por_tipo(df: pd.DataFrame) -> pd.DataFrame:
    """Los seis tipos de error (con ceros), su cantidad y su tasa sobre el total filtrado."""
    total = len(df)
    filas = []
    for tipo, etiqueta in ETIQUETAS_TIPO_ERROR.items():
        n = int((df["tipo_error"] == tipo).sum()) if total else 0
        filas.append({"tipo": tipo, "etiqueta": etiqueta, "n": n, "tasa": (n / total) if total else None})
    return pd.DataFrame(filas, columns=["tipo", "etiqueta", "n", "tasa"])


def errores_por_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """Errores por día (UTC) y tipo: una fila por cada día con ejecuciones (los
    días sin errores salen en cero) y una columna por tipo, más `ejecuciones`."""
    columnas = [*ETIQUETAS_TIPO_ERROR.values(), "ejecuciones"]
    con_fecha = df[df["dia"].notna()]
    if con_fecha.empty:
        return pd.DataFrame(columns=columnas, index=pd.Index([], name="dia"))
    dias = sorted(con_fecha["dia"].unique())
    tabla = pd.DataFrame(0, index=pd.Index(dias, name="dia"), columns=columnas)
    for dia, grupo in con_fecha.groupby("dia"):
        tabla.loc[dia, "ejecuciones"] = len(grupo)
        for tipo, etiqueta in ETIQUETAS_TIPO_ERROR.items():
            tabla.loc[dia, etiqueta] = int((grupo["tipo_error"] == tipo).sum())
    return tabla


# ------------------------------------------------------------------- calidad


def _valor_python(valor):
    if valor is None:
        return None
    try:
        if pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(valor, "item"):
        valor = valor.item()
    if isinstance(valor, float) and valor.is_integer():
        return int(valor)
    return valor


def ejecuciones_como_dicts(df: pd.DataFrame, pasos: pd.DataFrame) -> list[dict]:
    """Las ejecuciones en la forma que espera `metricas.calcular`: dicts con
    `pasos` y `herramientas` (en orden de uso) y None en lugar de NaN."""
    pasos_por_id: dict[int, list[dict]] = {}
    if not pasos.empty:
        for paso in pasos.sort_values(["id_ejecucion", "orden"]).to_dict("records"):
            pasos_por_id.setdefault(int(paso["id_ejecucion"]), []).append({k: _valor_python(v) for k, v in paso.items()})
    resultado = []
    for fila in df.to_dict("records"):
        e = {k: _valor_python(v) for k, v in fila.items() if k not in ("fecha", "dia")}
        e["pasos"] = pasos_por_id.get(e["id"], [])
        e["herramientas"] = tuple(p["herramienta"] for p in e["pasos"])
        resultado.append(e)
    return resultado


@dataclass(frozen=True)
class EstadoEtiquetas:
    etiquetas: dict
    estado: str  # "ok" | "faltan" | "vacio" | "invalido"
    mensaje: str


def cargar_etiquetas_estudiante(carpeta: str | Path) -> EstadoEtiquetas:
    ruta = Path(carpeta) / "etiquetas.csv"
    if not ruta.exists():
        return EstadoEtiquetas(
            {}, "faltan",
            f"Faltan las etiquetas del estudiante ({ruta}): sin ellas no hay precisión de veredicto. "
            "Corre `python scripts/calcular_metricas.py` para generar la plantilla y llena la columna 'correcto'.",
        )
    try:
        etiquetas = leer_etiquetas(ruta)
    except ErrorEtiquetas as error:
        return EstadoEtiquetas({}, "invalido", str(error))
    if not etiquetas:
        return EstadoEtiquetas(
            {}, "vacio",
            f"El archivo de etiquetas ({ruta}) no tiene ninguna fila etiquetada todavía: "
            "falta llenar la columna 'correcto' (1, 0 o parcial).",
        )
    return EstadoEtiquetas(etiquetas, "ok", f"{len(etiquetas)} ejecución(es) etiquetada(s).")


COLUMNAS_CALIDAD = (
    "categoria_caso", "ejecuciones", "precision_clasificacion", "precision_herramientas",
    "etiquetadas", "precision_veredicto", "consistencia", "grupos_con_repeticiones",
)


def _fila_calidad(nombre: str, m: dict) -> dict:
    return {
        "categoria_caso": nombre,
        "ejecuciones": m["ejecuciones"]["unicas"],
        "precision_clasificacion": m["precision_clasificacion"]["precision"],
        "precision_herramientas": m["precision_herramientas"]["precision"],
        "etiquetadas": m["precision_veredicto"]["etiquetadas"],
        "precision_veredicto": m["precision_veredicto"]["precision_estricta"],
        "consistencia": m["consistencia"]["ambos"],
        "grupos_con_repeticiones": m["consistencia"]["grupos_evaluables"],
    }


def calidad_por_categoria(casos: list[dict], ejecuciones: list[dict], etiquetas: dict[int, dict] | None = None) -> dict:
    """Precisión de clasificación, de herramientas y de veredicto, y consistencia,
    para todos los casos y por categoría de caso, con `metricas.calcular` sobre
    las ejecuciones ya filtradas. Devuelve `tabla` (una fila "Todas" y una por
    categoría, valores entre 0 y 1 o None), `aviso_modelos` y `ejecuciones`
    (el conteo de exclusiones de `calcular`)."""
    etiquetas = etiquetas or {}
    todas = calcular(casos, ejecuciones, etiquetas)
    filas = [_fila_calidad("Todas", todas)]
    for categoria in sorted({c["categoria_caso"] for c in casos}):
        propios = [c for c in casos if c["categoria_caso"] == categoria]
        filas.append(_fila_calidad(categoria, calcular(propios, ejecuciones, etiquetas)))
    return {
        "tabla": pd.DataFrame(filas, columns=list(COLUMNAS_CALIDAD)),
        "aviso_modelos": todas["aviso_modelos"],
        "modelos": todas["modelos"],
        "ejecuciones": todas["ejecuciones"],
    }


# ------------------------------------------------------------- recuperación


def ultimo_csv_recuperacion(carpeta: str | Path) -> Path | None:
    """El CSV de recuperación más reciente (los nombres llevan fecha y hora:
    `recuperacion_AAAAMMDD_HHMMSS.csv`), o None si no hay ninguno."""
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return None
    candidatos = sorted(carpeta.glob(PATRON_CSV_RECUPERACION), key=lambda p: p.name)
    return candidatos[-1] if candidatos else None


def _entero(texto: str) -> int | None:
    texto = (texto or "").strip()
    return int(float(texto)) if texto else None


def leer_recuperacion(ruta: str | Path) -> dict:
    """Lee un CSV de `scripts/evaluar_recuperacion.py` y arma la comparación
    estricta vs ampliada (hit@3, hit@5, MRR) global y por categoría de caso,
    con `eval_recuperacion.resumir`. Un CSV anterior a las alternativas no trae
    columnas ampliadas: en ese caso `tiene_ampliada` es False y la ampliada se
    iguala a la estricta. Lanza ValueError si el archivo no es un CSV de recuperación."""
    ruta = Path(ruta)
    crudo = pd.read_csv(ruta, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    faltantes = [c for c in ("caso_id", "categoria_caso", "rango") if c not in crudo.columns]
    if faltantes:
        raise ValueError(f"{ruta.name} no parece un CSV de recuperación: faltan las columnas {faltantes}")
    tiene_ampliada = "rango_ampliado" in crudo.columns

    filas = []
    for f in crudo.to_dict("records"):
        alternativas = [a for a in f.get("fuentes_alternativas", "").split(SEPARADOR_ALTERNATIVAS) if a.strip()]
        rango = _entero(f["rango"])
        filas.append(
            {
                "caso_id": f["caso_id"],
                "categoria_caso": f["categoria_caso"],
                "variante": f.get("variante", ""),
                "rango": rango,
                "rango_ampliado": _entero(f["rango_ampliado"]) if tiene_ampliada else rango,
                "fuentes_alternativas": alternativas,
            }
        )
    resumen = resumir(filas)

    def fila(nombre: str, bloque: dict) -> dict:
        salida = {"grupo": nombre, "consultas": bloque["n"]}
        for n in KS:
            salida[f"hit@{n} estricta"] = bloque["estricta"][f"hit@{n}"]
            salida[f"hit@{n} ampliada"] = bloque["ampliada"][f"hit@{n}"]
        salida["MRR estricta"] = bloque["estricta"]["mrr"]
        salida["MRR ampliada"] = bloque["ampliada"]["mrr"]
        return salida

    tabla = pd.DataFrame(
        [fila("Global", resumen["global"]), *(fila(c, b) for c, b in resumen["por_categoria_caso"].items())]
    )
    return {
        "archivo": ruta.name,
        "tabla": tabla,
        "consultas": len(filas),
        "con_alternativas": resumen["global"]["con_alternativas"],
        "tiene_ampliada": tiene_ampliada,
        "sin_acierto": pd.DataFrame(
            [
                {"caso_id": f["caso_id"], "variante": f["variante"], "rango_ampliado": f["rango_ampliado"]}
                for f in filas
                if f["rango"] is None
            ],
            columns=["caso_id", "variante", "rango_ampliado"],
        ),
    }


# -------------------------------------------------------------------- trazas


def opciones_de_traza(df: pd.DataFrame, limite: int = LIMITE_TRAZAS) -> list[tuple[int, str]]:
    """Las ejecuciones más recientes primero, como (id, texto para el selector)."""
    opciones = []
    for fila in df.sort_values("id", ascending=False).head(limite).to_dict("records"):
        consulta = " ".join(str(fila["consulta"]).split())
        consulta = consulta if len(consulta) <= 60 else consulta[:57] + "..."
        caso = f" · {fila['caso_id']}" if _valor_python(fila.get("caso_id")) else ""
        opciones.append((int(fila["id"]), f"#{int(fila['id'])} · {fila['fecha_utc'][:19]} · {fila['origen']}{caso} · {fila['estado']} · {consulta}"))
    return opciones


def traza_de(df: pd.DataFrame, pasos: pd.DataFrame, id_ejecucion: int) -> tuple[dict, pd.DataFrame]:
    """(la ejecución como dict, sus pasos ordenados). Lanza KeyError si no existe."""
    fila = df[df["id"] == id_ejecucion]
    if fila.empty:
        raise KeyError(id_ejecucion)
    ejecucion = {k: _valor_python(v) for k, v in fila.iloc[0].to_dict().items() if k not in ("fecha", "dia")}
    suyos = pasos[pasos["id_ejecucion"] == id_ejecucion].sort_values("orden")
    tabla = suyos[["orden", "herramienta", "argumentos", "duracion_ms", "exito", "tamano_salida"]].copy()
    tabla["exito"] = tabla["exito"].astype(bool)
    return ejecucion, tabla.reset_index(drop=True)
