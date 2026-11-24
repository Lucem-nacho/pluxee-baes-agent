"""Dashboard de observabilidad (EP3) del agente informativo BAES.

Solo interfaz: lee `data/processed/observabilidad.db` (en modo solo lectura: si no
existe, lo dice y NO la crea), las etiquetas y el CSV de recuperación de
`data/processed/eval/`, y muestra los indicadores, la latencia, los errores, la
calidad, la recuperación y un explorador de trazas. Todo el cálculo está en
`observabilidad/dashboard_datos.py`, que es lo que se prueba.

Abrir:  python -m streamlit run dashboard.py --server.fileWatcherType none

Los filtros de la barra lateral (fecha, origen, modelo, estado y tipo de error)
valen para todas las secciones menos la de recuperación, que sale de un CSV
aparte. Un filtro sin nada elegido significa "todos". Como la muestra es pequeña,
siempre se muestra cuántas ejecuciones quedan después de filtrar.
"""

import sqlite3
from datetime import date

import pandas as pd
import streamlit as st

from config import settings
from observabilidad import dashboard_datos as datos
from observabilidad.casos import ErrorEsquema, cargar_casos

MUESTRA_PEQUENA = 30  # por debajo de esto, los percentiles y porcentajes son poco estables

st.set_page_config(page_title="Observabilidad — Agente BAES", layout="wide")


# ---------------------------------------------------------------- formatos


def _ms(valor: float | None) -> str:
    return "—" if valor is None or pd.isna(valor) else f"{valor:,.0f} ms".replace(",", ".")


def _pct(valor: float | None) -> str:
    return "—" if valor is None or pd.isna(valor) else f"{valor * 100:.1f}%"


def _num(valor: float | None, decimales: int = 1) -> str:
    return "—" if valor is None or pd.isna(valor) else f"{valor:,.{decimales}f}".replace(",", ".")


def _en_porcentaje(tabla: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    salida = tabla.copy()
    for columna in columnas:
        salida[columna] = salida[columna].astype(float) * 100
    return salida


def _tabla_calidad(tabla: pd.DataFrame) -> pd.DataFrame:
    """La tabla de calidad lista para mostrar: porcentajes como texto, con «—» donde no hay dato."""
    salida = pd.DataFrame(
        {
            "Categoría de caso": tabla["categoria_caso"],
            "Ejecuciones": tabla["ejecuciones"],
            "Precisión de clasificación": tabla["precision_clasificacion"].map(_pct),
            "Precisión de herramientas": tabla["precision_herramientas"].map(_pct),
            "Etiquetadas": tabla["etiquetadas"],
            "Precisión de veredicto": tabla["precision_veredicto"].map(_pct),
            "Consistencia": tabla["consistencia"].map(_pct),
            "Grupos con repeticiones": tabla["grupos_con_repeticiones"],
        }
    )
    return salida


def _config_porcentaje(columnas: list[str]) -> dict:
    return {c: st.column_config.NumberColumn(c, format="%.1f%%") for c in columnas}


# ---------------------------------------------------------------- la página

st.title("Observabilidad del agente BAES")

base = datos.diagnosticar_base()
if not base.utilizable:
    st.warning(base.mensaje)
    st.stop()

try:
    todas = datos.cargar_ejecuciones_df()
    todos_los_pasos = datos.cargar_pasos_df()
except (sqlite3.Error, ValueError) as error:
    st.error(f"No se pudo leer la base de observabilidad: {error}")
    st.stop()

# ---- filtros (barra lateral)
st.sidebar.header("Filtros")
fechas = datos.rango_de_fechas(todas)
desde = hasta = None
if fechas is not None:
    elegido = st.sidebar.date_input("Fecha (UTC)", value=fechas, min_value=fechas[0], max_value=fechas[1])
    if isinstance(elegido, date):
        elegido = (elegido,)
    if len(elegido) == 1:
        desde = hasta = elegido[0]  # mientras se elige el segundo extremo
    elif len(elegido) == 2:
        desde, hasta = elegido
origenes = st.sidebar.multiselect("Origen", datos.valores_de(todas, "origen"), placeholder="Todos")
modelos = st.sidebar.multiselect("Modelo", datos.valores_de(todas, "modelo"), placeholder="Todos")
estados = st.sidebar.multiselect("Estado", datos.valores_de(todas, "estado"), placeholder="Todos")
tipos_error = st.sidebar.multiselect(
    "Tipo de error",
    datos.valores_de(todas, "tipo_error"),
    format_func=lambda t: datos.ETIQUETAS_TIPO_ERROR.get(t, "Sin error" if t == "ninguno" else t),
    placeholder="Todos",
)
st.sidebar.caption("Un filtro sin nada elegido equivale a «todos». La sección de recuperación no depende de los filtros.")

filtradas = datos.aplicar_filtros(
    todas,
    datos.Filtros(desde, hasta, tuple(origenes), tuple(modelos), tuple(estados), tuple(tipos_error)),
)
n = len(filtradas)
st.sidebar.metric("Ejecuciones tras filtrar", f"{n} de {len(todas)}")

if n == 0:
    st.warning(f"Ninguna de las {len(todas)} ejecuciones cumple los filtros elegidos. Amplía o quita algún filtro.")
    st.stop()
if n < MUESTRA_PEQUENA:
    st.info(
        f"Muestra pequeña: {n} ejecución(es) tras filtrar (de {len(todas)}). Los porcentajes y los "
        "percentiles son poco estables con tan pocos datos."
    )

# ---- 1. indicadores
ind = datos.indicadores(filtradas)
st.subheader("Indicadores")
columnas = st.columns(6)
columnas[0].metric("Ejecuciones", f"{ind['ejecuciones']}", help=f"{ind['completas']} completas (estado distinto de ERROR).")
columnas[1].metric(
    "Con error",
    _pct(ind["porcentaje_error"]),
    help=f"{ind['con_error']} ejecución(es) con un tipo de error distinto de «ninguno»; "
    f"{ind['con_estado_error']} terminaron en estado ERROR.",
)
columnas[2].metric("Latencia total p50", _ms(ind["latencia_total_p50_ms"]), help="Sobre las ejecuciones completas.")
columnas[3].metric("Latencia total p95", _ms(ind["latencia_total_p95_ms"]), help="Sobre las ejecuciones completas.")
columnas[4].metric(
    "Tokens por consulta",
    _num(ind["tokens_promedio"], 0),
    help=f"Entrada + salida, promedio de {ind['consultas_con_tokens']} consulta(s) con tokens registrados "
    f"(entrada {_num(ind['tokens_entrada_promedio'], 0)}, salida {_num(ind['tokens_salida_promedio'], 0)}).",
)
columnas[5].metric("Llamadas al LLM por consulta", _num(ind["llamadas_llm_promedio"]), help="Promedio sobre las ejecuciones completas.")

# ---- 2. latencia
st.subheader("Latencia")
st.caption(f"Ejecuciones completas ({ind['completas']} de {n}); p50 = mediana, p95 = percentil 95.")
izquierda, derecha = st.columns(2)
por_etapa = datos.latencia_por_etapa(filtradas)
por_herramienta = datos.latencia_por_herramienta(filtradas, todos_los_pasos)
with izquierda:
    st.markdown("**Por etapa**")
    st.bar_chart(por_etapa.set_index("etapa")[["p50_ms", "p95_ms"]], stack=False, sort=False)  # lado a lado: el p95 no se suma al p50
    st.dataframe(por_etapa, hide_index=True, width="stretch")
with derecha:
    st.markdown("**Por herramienta**")
    if por_herramienta.empty:
        st.info("Ninguna ejecución completa usó herramientas en este filtro.")
    else:
        st.bar_chart(por_herramienta.set_index("herramienta")[["p50_ms", "p95_ms"]], stack=False, sort=False)
        st.dataframe(por_herramienta, hide_index=True, width="stretch")

# ---- 3. errores
st.subheader("Errores")
por_tipo = datos.errores_por_tipo(filtradas)
izquierda, derecha = st.columns(2)
with izquierda:
    st.markdown("**Por tipo**")
    if ind["con_error"] == 0:
        st.success("No hay errores en las ejecuciones filtradas.")
    st.bar_chart(por_tipo.set_index("etiqueta")[["n"]], sort=False)
    st.dataframe(
        _en_porcentaje(por_tipo.drop(columns="tipo"), ["tasa"]),
        hide_index=True, width="stretch", column_config=_config_porcentaje(["tasa"]),
    )
with derecha:
    st.markdown("**Evolución por fecha (UTC)**")
    por_fecha = datos.errores_por_fecha(filtradas)
    grafico = por_fecha.drop(columns="ejecuciones")
    grafico.index = grafico.index.map(str)  # eje de categorías (un día por barra), no de tiempo continuo
    st.bar_chart(grafico, sort=False)
    st.dataframe(por_fecha, width="stretch")

# ---- 4. calidad
st.subheader("Calidad")
try:
    casos = cargar_casos()
except ErrorEsquema as error:
    casos = None
    st.error(f"No se pudo leer eval/casos.json, así que no hay calidad por categoría:\n\n{error}")

if casos is not None:
    etiquetas = datos.cargar_etiquetas_estudiante(datos.carpeta_eval())
    if etiquetas.estado == "invalido":
        st.error(f"El archivo de etiquetas tiene errores y no se usa:\n\n{etiquetas.mensaje}")
    elif etiquetas.estado != "ok":
        st.warning(etiquetas.mensaje)
    else:
        st.caption(f"Etiquetas del estudiante: {etiquetas.mensaje}")

    de_evaluacion = filtradas[filtradas["origen"] == "evaluacion"]
    if de_evaluacion.empty:
        st.info("No hay ejecuciones de evaluación en el filtro actual (la calidad solo se mide con ellas).")
    else:
        calidad = datos.calidad_por_categoria(
            casos, datos.ejecuciones_como_dicts(de_evaluacion, todos_los_pasos), etiquetas.etiquetas
        )
        if calidad["aviso_modelos"]:
            st.warning(f"{calidad['aviso_modelos']} Filtra por modelo para no mezclarlos.")
        e = calidad["ejecuciones"]
        if e["unicas"] == 0:
            st.info(
                f"Ninguna de las {e['evaluacion_total']} ejecuciones de evaluación del filtro cuenta para la calidad "
                f"({e['sin_caso_excluidas']} sin caso, {e['preparacion_excluidas']} de preparación, "
                f"{e['fuera_del_conjunto_excluidas']} de casos o preguntas que ya no están en eval/casos.json, "
                f"{e['con_error']} con error)."
            )
        else:
            st.caption(
                f"{e['unicas']} ejecución(es) únicas consideradas de {e['evaluacion_total']} de evaluación; "
                f"excluidas: {e['sin_caso_excluidas']} sin caso, {e['preparacion_excluidas']} de preparación, "
                f"{e['fuera_del_conjunto_excluidas']} fuera del conjunto activo; {e['con_error']} con error (no cuentan en la precisión)."
            )
            st.dataframe(_tabla_calidad(calidad["tabla"]), hide_index=True, width="stretch")
            st.caption(
                "Precisión de veredicto = respuestas etiquetadas como correctas / etiquetadas (solo con etiquetas). "
                "Consistencia = repeticiones iguales en veredicto y herramientas; necesita 2 o más repeticiones de la misma pregunta."
            )

# ---- 5. recuperación
st.subheader("Recuperación")
carpeta = datos.carpeta_eval()
ultimo = datos.ultimo_csv_recuperacion(carpeta)
if ultimo is None:
    st.info(f"No hay CSV de recuperación en {carpeta}. Genera uno con `python scripts/evaluar_recuperacion.py`.")
else:
    try:
        recuperacion = datos.leer_recuperacion(ultimo)
    except (ValueError, OSError, pd.errors.ParserError) as error:
        st.error(f"No se pudo leer {ultimo.name}: {error}")
    else:
        st.caption(
            f"Último CSV: {recuperacion['archivo']} · {recuperacion['consultas']} consulta(s). "
            "Estricta = solo `fuente_esperada`; ampliada = también las `fuentes_alternativas`."
        )
        if not recuperacion["tiene_ampliada"]:
            st.warning("Este CSV es anterior a las fuentes alternativas: no trae la métrica ampliada (se muestra igual a la estricta). Vuelve a correr `scripts/evaluar_recuperacion.py`.")
        elif recuperacion["con_alternativas"] == 0:
            st.info("Ningún caso tiene `fuentes_alternativas`: la métrica ampliada es igual a la estricta.")
        tabla = recuperacion["tabla"]
        columnas_pct = [c for c in tabla.columns if c.startswith("hit@")]
        st.dataframe(
            _en_porcentaje(tabla, columnas_pct),
            hide_index=True, width="stretch",
            column_config={**_config_porcentaje(columnas_pct), **{c: st.column_config.NumberColumn(c, format="%.3f") for c in tabla.columns if c.startswith("MRR")}},
        )
        if not recuperacion["sin_acierto"].empty:
            with st.expander(f"Consultas sin acierto estricto ({len(recuperacion['sin_acierto'])})"):
                st.dataframe(recuperacion["sin_acierto"], hide_index=True, width="stretch")

# ---- 6. explorador de trazas
st.subheader("Explorador de trazas")
opciones = datos.opciones_de_traza(filtradas)
etiquetas_de_opcion = dict(opciones)
elegida = st.selectbox(
    f"Ejecución (las {len(opciones)} más recientes del filtro)",
    [i for i, _ in opciones],
    format_func=lambda i: etiquetas_de_opcion[i],
)
ejecucion, pasos = datos.traza_de(filtradas, todos_los_pasos, elegida)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Estado", ejecucion["estado"])
c2.metric("Tipo de error", datos.ETIQUETAS_TIPO_ERROR.get(ejecucion["tipo_error"], "Sin error" if ejecucion["tipo_error"] == "ninguno" else ejecucion["tipo_error"]))
c3.metric("Latencia total", _ms(ejecucion["latencia_total_ms"]))
c4.metric("Llamadas al LLM", _num(ejecucion["llamadas_llm"], 0))
c5.metric("Modelo", ejecucion["modelo"])
st.markdown(f"**Consulta** ({ejecucion['origen']}" + (f", caso {ejecucion['caso_id']}, repetición {ejecucion['repeticion']}" if ejecucion.get("caso_id") else "") + ")")
st.text(ejecucion["consulta"])
st.markdown("**Respuesta**")
st.text(ejecucion["respuesta"] or "(sin respuesta registrada)")
st.markdown("**Pasos del agente**")
if pasos.empty:
    st.info("Esta ejecución no usó herramientas (por ejemplo, la derivó el clasificador de entrada o falló antes).")
else:
    st.dataframe(pasos, hide_index=True, width="stretch")

st.caption(f"Base: {settings.OBSERVABILIDAD_DB_PATH}")
