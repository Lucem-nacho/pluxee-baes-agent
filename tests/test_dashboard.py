"""Pruebas del dashboard (EP3, paso 3).

Con datos sintéticos y bases temporales: el `conftest` redirige la base de
observabilidad a una carpeta temporal en todas las pruebas, y estas pruebas ni
siquiera abren la memoria de largo plazo. Dos partes:

- las funciones de datos de `observabilidad/dashboard_datos.py` (valores
  calculados a mano) y el caso de base inexistente o vacía;
- la interfaz `dashboard.py`, ejecutada con `streamlit.testing` (sin navegador).
"""

import csv
import json
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from observabilidad import casos as modulo_casos
from observabilidad import dashboard_datos as datos
from observabilidad import store
from observabilidad.eval_recuperacion import evaluar, guardar_csv
from observabilidad.metricas import calcular, cargar_ejecuciones

RAIZ = Path(__file__).resolve().parents[1]

CASOS = [
    {"id": "A", "categoria_caso": "normativa", "pregunta": "pa", "variantes": [], "categoria_esperada": "INFORMATIVA",
     "herramientas_esperadas": ["buscar_normativa"], "veredicto_esperado": "prohibido", "fuente_esperada": "alcohol"},
    {"id": "B", "categoria_caso": "derivacion", "pregunta": "pb", "variantes": [], "categoria_esperada": "ESCALAR",
     "herramientas_esperadas": [], "veredicto_esperado": "derivar", "fuente_esperada": None},
    {"id": "C", "categoria_caso": "comercio", "pregunta": "pc", "variantes": [], "categoria_esperada": "INFORMATIVA",
     "herramientas_esperadas": ["buscar_comercio"], "veredicto_esperado": "permitido", "fuente_esperada": None},
]


def registrar(db, fecha, *, origen="app", modelo="m1", estado="INFORMATIVA", categoria="INFORMATIVA", tipo_error="ninguno",
              total=1000, clasificador=100, agente=900, llamadas=3, t_in=100, t_out=50, caso=None, variante=None,
              repeticion=None, respuesta="ok", consulta="una consulta", pasos=()):
    return store.registrar_ejecucion(
        {
            "fecha_utc": f"{fecha}T10:00:00.000+00:00", "origen": origen, "id_usuario_anonimo": "abc123abc123",
            "consulta": consulta, "categoria": categoria, "estado": estado, "latencia_total_ms": total,
            "latencia_clasificador_ms": clasificador, "latencia_agente_ms": agente, "llamadas_llm": llamadas,
            "tokens_entrada": t_in, "tokens_salida": t_out, "iteraciones_agente": None, "tipo_error": tipo_error,
            "modelo": modelo, "respuesta": respuesta, "caso_id": caso, "variante": variante, "repeticion": repeticion,
        },
        [
            {"orden": i, "herramienta": h, "argumentos": json.dumps({"q": "x"}), "duracion_ms": d, "exito": 1, "tamano_salida": 10}
            for i, (h, d) in enumerate(pasos, start=1)
        ],
        db_path=db,
    )


def sembrar(db):
    """Nueve ejecuciones en tres días y dos modelos; las métricas están calculadas
    a mano en cada prueba. Devuelve {nombre: id}."""
    ids = {}
    ids["e1"] = registrar(db, "2026-11-01", pasos=[("buscar_normativa", 100)])
    ids["e2"] = registrar(db, "2026-11-01", total=3000, clasificador=200, agente=2800, llamadas=4, t_in=200, t_out=100,
                          pasos=[("buscar_normativa", 300), ("buscar_comercio", 50)])
    ids["e3"] = registrar(db, "2026-11-02", estado="ERROR", categoria=None, tipo_error="429", total=500, agente=400,
                          llamadas=2, t_in=None, t_out=None, respuesta=None)
    ids["e4"] = registrar(db, "2026-11-02", origen="evaluacion", caso="A", variante="pa", repeticion=1, total=2000,
                          clasificador=200, agente=1800, t_in=300, t_out=100, respuesta="No puedes comprar alcohol.",
                          pasos=[("buscar_normativa", 200)])
    ids["e5"] = registrar(db, "2026-11-02", origen="evaluacion", caso="A", variante="pa", repeticion=2, total=4000,
                          clasificador=400, agente=3600, t_in=300, t_out=100, respuesta="Sí puedes comprar eso.",
                          pasos=[("buscar_normativa", 400)])
    ids["e6"] = registrar(db, "2026-11-03", origen="evaluacion", modelo="m2", estado="ESCALAR", categoria="ESCALAR",
                          caso="B", variante="pb", repeticion=1, total=600, clasificador=600, agente=None, llamadas=1,
                          t_in=None, t_out=None, respuesta=None)
    ids["e7"] = registrar(db, "2026-11-03", origen="evaluacion", modelo="m2", estado="ERROR", categoria=None,
                          tipo_error="503", caso="B", variante="pb", repeticion=2, total=300, clasificador=300, agente=None,
                          llamadas=1, t_in=None, t_out=None, respuesta=None)
    ids["e8"] = registrar(db, "2026-11-03", origen="evaluacion", modelo="m2", tipo_error="limite_iteraciones", caso="A",
                          variante="pa", repeticion=1, total=5000, clasificador=500, agente=4500, llamadas=5, t_in=400,
                          t_out=200, respuesta="texto", pasos=[("buscar_normativa", 900), ("buscar_comercio", 100)])
    ids["e9"] = registrar(db, "2026-11-03", origen="evaluacion", modelo="m2", caso="A-prep", variante="consulta previa",
                          repeticion=1, total=800, clasificador=100, agente=700, t_in=100, t_out=50)
    return ids


@pytest.fixture
def base(db_observabilidad):
    ids = sembrar(db_observabilidad)
    return db_observabilidad, ids


def cargar(db):
    return datos.cargar_ejecuciones_df(db), datos.cargar_pasos_df(db)


# ------------------------------------------------- la base: no existe / vacía / ok


def test_una_base_que_no_existe_se_informa_y_no_se_crea(tmp_path):
    ruta = tmp_path / "carpeta_nueva" / "obs.db"
    estado = datos.diagnosticar_base(ruta)
    assert (estado.estado, estado.total, estado.utilizable) == ("no_existe", 0, False)
    assert "Todavía no existe la base de observabilidad" in estado.mensaje and str(ruta) in estado.mensaje
    assert not ruta.exists() and not ruta.parent.exists()   # mirar no fabrica nada
    with pytest.raises(sqlite3.OperationalError):
        datos.cargar_ejecuciones_df(ruta)
    assert not ruta.exists()


def test_una_base_creada_pero_sin_ejecuciones_se_informa_como_vacia(tmp_path):
    ruta = tmp_path / "obs.db"
    store._conectar(ruta).close()   # tablas creadas, ninguna fila
    estado = datos.diagnosticar_base(ruta)
    assert (estado.estado, estado.total, estado.utilizable) == ("vacia", 0, False)
    assert "todavía no tiene ejecuciones" in estado.mensaje


def test_un_archivo_sin_tablas_y_uno_corrupto_se_diagnostican_sin_romper(tmp_path):
    sin_tablas = tmp_path / "sin_tablas.db"
    sqlite3.connect(sin_tablas).close()
    assert datos.diagnosticar_base(sin_tablas).estado == "sin_tablas"
    corrupto = tmp_path / "corrupto.db"
    corrupto.write_bytes(b"esto no es una base sqlite" * 50)
    estado = datos.diagnosticar_base(corrupto)
    assert estado.estado == "ilegible" and "No se pudo leer" in estado.mensaje


def test_una_base_con_datos_es_utilizable(base):
    db, _ = base
    estado = datos.diagnosticar_base(db)
    assert (estado.estado, estado.total, estado.utilizable) == ("ok", 9, True)


def test_la_conexion_del_dashboard_es_de_solo_lectura(base):
    db, _ = base
    conn = datos._conectar_lectura(db)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM ejecuciones")
    finally:
        conn.close()
    assert datos.diagnosticar_base(db).total == 9


def test_se_leen_todas_las_ejecuciones_de_cualquier_origen_con_su_fecha(base):
    df, pasos = cargar(base[0])
    assert len(df) == 9 and set(df["origen"]) == {"app", "evaluacion"}
    assert df["id"].is_monotonic_increasing
    assert list(df["dia"].unique()) == [date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3)]
    assert len(pasos) == 7   # e1 (1), e2 (2), e4 (1), e5 (1) y e8 (2); el resto no usó herramientas
    assert datos.rango_de_fechas(df) == (date(2026, 11, 1), date(2026, 11, 3))
    assert datos.valores_de(df, "modelo") == ["m1", "m2"] and datos.valores_de(df, "origen") == ["app", "evaluacion"]


def test_una_base_de_una_version_anterior_sin_columnas_de_caso_tambien_se_lee(tmp_path):
    ruta = tmp_path / "vieja.db"
    conn = sqlite3.connect(ruta)
    conn.executescript(
        """
        CREATE TABLE ejecuciones (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha_utc TEXT NOT NULL, origen TEXT NOT NULL,
            id_usuario_anonimo TEXT NOT NULL, consulta TEXT NOT NULL, categoria TEXT, estado TEXT NOT NULL,
            latencia_total_ms INTEGER NOT NULL, latencia_clasificador_ms INTEGER, latencia_agente_ms INTEGER,
            llamadas_llm INTEGER NOT NULL, tokens_entrada INTEGER, tokens_salida INTEGER, iteraciones_agente INTEGER,
            tipo_error TEXT NOT NULL, modelo TEXT NOT NULL, respuesta TEXT);
        INSERT INTO ejecuciones VALUES (1, '2026-11-01T10:00:00.000+00:00', 'app', 'abc', 'vieja', 'INFORMATIVA',
            'INFORMATIVA', 10, 5, 5, 2, NULL, NULL, 1, 'ninguno', 'm', 'r');
        """
    )
    conn.commit()
    conn.close()
    df = datos.cargar_ejecuciones_df(ruta)
    assert len(df) == 1 and df["caso_id"].isna().all() and df["repeticion"].isna().all()
    assert datos.cargar_pasos_df(ruta).empty    # ni siquiera existe la tabla de pasos
    assert datos.indicadores(df)["ejecuciones"] == 1


# ------------------------------------------------------------------- filtros


def test_cada_filtro_deja_solo_lo_que_corresponde_y_las_fechas_son_inclusivas(base):
    df, _ = cargar(base[0])
    assert len(datos.aplicar_filtros(df, datos.Filtros())) == 9   # sin filtros: todo
    assert len(datos.aplicar_filtros(df, datos.Filtros(origenes=("app",)))) == 3
    assert len(datos.aplicar_filtros(df, datos.Filtros(modelos=("m2",)))) == 4
    assert len(datos.aplicar_filtros(df, datos.Filtros(estados=("ERROR",)))) == 2
    assert len(datos.aplicar_filtros(df, datos.Filtros(estados=("ERROR", "ESCALAR")))) == 3
    assert len(datos.aplicar_filtros(df, datos.Filtros(tipos_error=("429", "503")))) == 2
    assert len(datos.aplicar_filtros(df, datos.Filtros(desde=date(2026, 11, 2)))) == 7
    assert len(datos.aplicar_filtros(df, datos.Filtros(hasta=date(2026, 11, 1)))) == 2
    assert len(datos.aplicar_filtros(df, datos.Filtros(desde=date(2026, 11, 2), hasta=date(2026, 11, 2)))) == 3   # un solo día
    assert len(datos.aplicar_filtros(df, datos.Filtros(desde=date(2026, 12, 1)))) == 0
    assert len(df) == 9   # el original no se toca


def test_los_filtros_se_combinan_con_y(base):
    df, _ = cargar(base[0])
    filtrado = datos.aplicar_filtros(df, datos.Filtros(origenes=("evaluacion",), modelos=("m2",), estados=("INFORMATIVA",)))
    assert list(filtrado["caso_id"]) == ["A", "A-prep"]
    vacio = datos.aplicar_filtros(df, datos.Filtros(origenes=("app",), modelos=("m2",)))
    assert vacio.empty
    assert datos.indicadores(vacio)["ejecuciones"] == 0 and datos.indicadores(vacio)["porcentaje_error"] is None


# --------------------------------------------------------------- indicadores


def test_indicadores_con_valores_calculados_a_mano(base):
    df, _ = cargar(base[0])
    i = datos.indicadores(df)
    assert (i["ejecuciones"], i["completas"]) == (9, 7)           # e3 y e7 terminaron en ERROR
    assert (i["con_error"], i["con_estado_error"]) == (3, 2)      # 429, 503 y límite de iteraciones
    assert i["porcentaje_error"] == pytest.approx(3 / 9)
    # latencia total de las completas: 600 800 1000 2000 3000 4000 5000
    assert i["latencia_total_p50_ms"] == 2000.0
    assert i["latencia_total_p95_ms"] == pytest.approx(4700.0)    # 4000 + 0,7 * 1000
    # tokens (entrada + salida) donde hay ambos: 150 300 400 400 600 150
    assert i["tokens_promedio"] == pytest.approx(2000 / 6) and i["consultas_con_tokens"] == 6
    assert i["tokens_entrada_promedio"] == pytest.approx((100 + 200 + 300 + 300 + 400 + 100) / 6)
    assert i["llamadas_llm_promedio"] == pytest.approx(22 / 7)    # 3 4 3 3 1 5 3


def test_los_indicadores_respetan_el_filtro(base):
    df, _ = cargar(base[0])
    solo_app = datos.indicadores(datos.aplicar_filtros(df, datos.Filtros(origenes=("app",))))
    assert (solo_app["ejecuciones"], solo_app["completas"], solo_app["con_error"]) == (3, 2, 1)
    assert solo_app["latencia_total_p50_ms"] == 2000.0            # (1000 + 3000) / 2
    assert solo_app["porcentaje_error"] == pytest.approx(1 / 3)


# ------------------------------------------------------------------- latencia


def test_latencia_por_etapa_y_por_herramienta_con_valores_a_mano(base):
    db, _ = base
    df, pasos = cargar(db)
    etapa = datos.latencia_por_etapa(df).set_index("etapa")
    assert etapa.loc["Total", ["n", "p50_ms", "p95_ms"]].tolist() == [7, 2000.0, pytest.approx(4700.0)]
    # clasificador: 100 100 200 200 400 500 600 -> p50 200, p95 500 + 0,7 * 100
    assert etapa.loc["Clasificador", ["n", "p50_ms", "p95_ms"]].tolist() == [7, 200.0, pytest.approx(570.0)]
    # agente (e6 no lo corrió): 700 900 1800 2800 3600 4500 -> p50 2300, p95 3600 + 0,75 * 900
    assert etapa.loc["Agente", ["n", "p50_ms", "p95_ms"]].tolist() == [6, 2300.0, pytest.approx(4275.0)]

    herramienta = datos.latencia_por_herramienta(df, pasos).set_index("herramienta")
    assert list(herramienta.index) == ["buscar_comercio", "buscar_normativa"]
    assert herramienta.loc["buscar_normativa", ["n", "p50_ms", "p95_ms"]].tolist() == [5, 300.0, pytest.approx(800.0)]   # 100 200 300 400 900
    assert herramienta.loc["buscar_comercio", ["n", "p50_ms", "p95_ms"]].tolist() == [2, 75.0, pytest.approx(97.5)]       # 50 100


def test_la_latencia_filtrada_no_cuenta_los_pasos_de_ejecuciones_fuera_del_filtro(base):
    db, _ = base
    df, pasos = cargar(db)
    solo_m1_app = datos.aplicar_filtros(df, datos.Filtros(origenes=("app",)))
    herramienta = datos.latencia_por_herramienta(solo_m1_app, pasos).set_index("herramienta")
    assert herramienta.loc["buscar_normativa", "n"] == 2 and herramienta.loc["buscar_comercio", "n"] == 1
    vacia = datos.latencia_por_herramienta(datos.aplicar_filtros(df, datos.Filtros(estados=("ESCALAR",))), pasos)
    assert vacia.empty and list(vacia.columns) == ["herramienta", "n", "p50_ms", "p95_ms"]


def test_la_latencia_coincide_con_la_de_metricas_py_para_las_mismas_ejecuciones(base):
    """El dashboard reutiliza las definiciones de metricas.py: sobre las ejecuciones
    de evaluación consideradas dan los mismos números que `calcular`."""
    db, _ = base
    df, pasos = cargar(db)
    esperado = calcular(CASOS, cargar_ejecuciones(db))["latencia_ms"]
    # las consideradas por calcular: evaluación, de los casos activos y sin la preparación
    consideradas = df[(df["origen"] == "evaluacion") & df["caso_id"].isin(["A", "B"])]
    etapas = datos.latencia_por_etapa(consideradas).set_index("etapa")
    for clave, nombre in (("total", "Total"), ("clasificador", "Clasificador"), ("agente", "Agente")):
        assert etapas.loc[nombre, "p50_ms"] == esperado["por_etapa"][clave]["p50_ms"]
        assert etapas.loc[nombre, "p95_ms"] == pytest.approx(esperado["por_etapa"][clave]["p95_ms"])
        assert etapas.loc[nombre, "n"] == esperado["por_etapa"][clave]["n"]
    herramientas = datos.latencia_por_herramienta(consideradas, pasos).set_index("herramienta")
    for nombre, r in esperado["por_herramienta"].items():
        assert herramientas.loc[nombre, ["n", "p50_ms"]].tolist() == [r["n"], r["p50_ms"]]


# -------------------------------------------------------------------- errores


def test_errores_por_tipo_trae_los_seis_tipos_con_ceros_y_su_tasa(base):
    df, _ = cargar(base[0])
    tabla = datos.errores_por_tipo(df).set_index("tipo")
    assert list(tabla.index) == ["429", "503", "limite_iteraciones", "sin_respuesta", "clasificador_no_interpretable", "otro"]
    assert tabla["n"].to_dict() == {"429": 1, "503": 1, "limite_iteraciones": 1, "sin_respuesta": 0, "clasificador_no_interpretable": 0, "otro": 0}
    assert tabla.loc["429", "tasa"] == pytest.approx(1 / 9) and tabla.loc["otro", "tasa"] == 0.0
    assert tabla.loc["limite_iteraciones", "etiqueta"] == "Límite de iteraciones"
    vacia = datos.errores_por_tipo(df.iloc[0:0])
    assert vacia["n"].tolist() == [0] * 6 and vacia["tasa"].isna().all()


def test_errores_por_fecha_incluye_los_dias_sin_errores_en_cero(base):
    df, _ = cargar(base[0])
    tabla = datos.errores_por_fecha(df)
    assert list(tabla.index) == [date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3)]
    assert tabla["ejecuciones"].tolist() == [2, 3, 4]
    assert tabla["429 (cuota)"].tolist() == [0, 1, 0]
    assert tabla["503 (servicio no disponible)"].tolist() == [0, 0, 1]
    assert tabla["Límite de iteraciones"].tolist() == [0, 0, 1]
    assert tabla.loc[date(2026, 11, 1)].drop("ejecuciones").sum() == 0   # el primer día no tuvo errores
    assert datos.errores_por_fecha(df.iloc[0:0]).empty


# -------------------------------------------------------------------- calidad


def test_ejecuciones_como_dicts_da_lo_mismo_que_metricas_cargar_ejecuciones(base):
    db, _ = base
    df, pasos = cargar(db)
    del_dashboard = datos.ejecuciones_como_dicts(df[df["origen"] == "evaluacion"], pasos)
    de_metricas = cargar_ejecuciones(db)
    assert [e["id"] for e in del_dashboard] == [e["id"] for e in de_metricas]
    for a, b in zip(del_dashboard, de_metricas):
        assert a == b   # incluidos None en vez de NaN, enteros y la tupla de herramientas
    assert calcular(CASOS, del_dashboard) == calcular(CASOS, de_metricas)


def test_calidad_por_categoria_con_valores_calculados_a_mano(base):
    db, ids = base
    df, pasos = cargar(db)
    etiquetas = {ids["e4"]: {"correcto": "1", "comentario": "", "veredicto_observado": None},
                 ids["e5"]: {"correcto": "0", "comentario": "", "veredicto_observado": None}}
    calidad = datos.calidad_por_categoria(CASOS, datos.ejecuciones_como_dicts(df[df["origen"] == "evaluacion"], pasos), etiquetas)
    tabla = calidad["tabla"].set_index("categoria_caso")

    assert list(tabla.index) == ["Todas", "comercio", "derivacion", "normativa"]
    # únicas (completas, sin la preparación): e4, e5, e6 y e8 -> A: 3, B: 1
    assert tabla["ejecuciones"].to_dict() == {"Todas": 4, "comercio": 0, "derivacion": 1, "normativa": 3}
    assert tabla.loc["Todas", "precision_clasificacion"] == 1.0
    assert tabla.loc["Todas", "precision_herramientas"] == 0.75          # e8 usó una herramienta de más
    assert tabla.loc["normativa", "precision_herramientas"] == pytest.approx(2 / 3)
    assert tabla.loc["derivacion", ["precision_clasificacion", "precision_herramientas"]].tolist() == [1.0, 1.0]
    assert pd.isna(tabla.loc["comercio", "precision_clasificacion"])     # sin ejecuciones: no hay precisión
    # veredicto: dos etiquetadas, una correcta
    assert tabla.loc["Todas", ["etiquetadas", "precision_veredicto"]].tolist() == [2, 0.5]
    assert tabla.loc["normativa", ["etiquetadas", "precision_veredicto"]].tolist() == [2, 0.5]
    assert tabla.loc["derivacion", "etiquetadas"] == 0 and pd.isna(tabla.loc["derivacion", "precision_veredicto"])
    # consistencia de (A, pa): tres repeticiones con veredictos y secuencias distintas -> 1/3
    assert tabla.loc["normativa", "consistencia"] == pytest.approx(1 / 3) and tabla.loc["normativa", "grupos_con_repeticiones"] == 1
    assert pd.isna(tabla.loc["derivacion", "consistencia"])
    # las ejecuciones de dos modelos se avisan
    assert calidad["modelos"] == ["m1", "m2"] and "2 modelos" in calidad["aviso_modelos"]
    assert calidad["ejecuciones"]["preparacion_excluidas"] == 1


def test_calidad_sin_etiquetas_deja_vacia_la_precision_de_veredicto_pero_calcula_el_resto(base):
    db, _ = base
    df, pasos = cargar(db)
    de_m1 = datos.aplicar_filtros(df, datos.Filtros(modelos=("m1",), origenes=("evaluacion",)))
    calidad = datos.calidad_por_categoria(CASOS, datos.ejecuciones_como_dicts(de_m1, pasos))
    tabla = calidad["tabla"].set_index("categoria_caso")
    assert calidad["aviso_modelos"] is None                              # un solo modelo: sin aviso
    assert tabla.loc["Todas", "etiquetadas"] == 0 and pd.isna(tabla.loc["Todas", "precision_veredicto"])
    assert tabla.loc["Todas", "precision_clasificacion"] == 1.0
    assert tabla.loc["normativa", "consistencia"] == 0.5                 # No puedes / Sí puedes: 1 de 2 coincide


def test_estado_de_las_etiquetas_faltan_vacio_invalido_y_ok(tmp_path):
    faltan = datos.cargar_etiquetas_estudiante(tmp_path)
    assert faltan.estado == "faltan" and "Faltan las etiquetas del estudiante" in faltan.mensaje and faltan.etiquetas == {}

    archivo = tmp_path / "etiquetas.csv"
    archivo.write_text("id_ejecucion,correcto,comentario\n1,,\n2,,\n", encoding="utf-8")
    vacio = datos.cargar_etiquetas_estudiante(tmp_path)
    assert vacio.estado == "vacio" and "ninguna fila etiquetada" in vacio.mensaje

    archivo.write_text("id_ejecucion,correcto,comentario\n1,quizas,\n", encoding="utf-8")
    invalido = datos.cargar_etiquetas_estudiante(tmp_path)
    assert invalido.estado == "invalido" and "línea 2" in invalido.mensaje

    archivo.write_text("id_ejecucion,correcto,comentario\n4,1,bien\n5,parcial,\n", encoding="utf-8")
    ok = datos.cargar_etiquetas_estudiante(tmp_path)
    assert ok.estado == "ok" and set(ok.etiquetas) == {4, 5} and "2 ejecución(es) etiquetada(s)" in ok.mensaje


# ----------------------------------------------------------------- recuperación


def frag(texto, distancia):
    return {"texto": texto, "distancia": distancia, "fuente": "Manual"}


CASOS_RECUPERACION = [
    {**CASOS[0], "id": "a", "pregunta": "qa", "variantes": ["qa2"], "fuente_esperada": "regla uno", "fuentes_alternativas": ["regla alternativa"]},
    {**CASOS[0], "id": "b", "pregunta": "qb", "variantes": [], "categoria_caso": "comercio", "fuente_esperada": "regla dos"},
]
RESULTADOS = {
    "qa": [frag("regla uno", 0.1)],                                                        # estricta 1, ampliada 1
    "qa2": [frag("x", 0.1), frag("y", 0.2), frag("la regla alternativa", 0.3)],           # estricta no, ampliada 3
    "qb": [frag("x", 0.1), frag("regla dos", 0.2)],                                        # estricta 2, ampliada 2
}


def csv_de_recuperacion(carpeta, nombre="recuperacion_20261120_100000.csv"):
    return guardar_csv(evaluar(CASOS_RECUPERACION, lambda consulta, k: RESULTADOS[consulta]), Path(carpeta) / nombre)


def test_el_ultimo_csv_de_recuperacion_es_el_de_nombre_con_fecha_mas_reciente(tmp_path):
    assert datos.ultimo_csv_recuperacion(tmp_path / "no_existe") is None
    assert datos.ultimo_csv_recuperacion(tmp_path) is None
    for nombre in ("recuperacion_20261119_235959.csv", "recuperacion_20261120_100000.csv", "recuperacion_20261118_080000.csv", "otro.csv", "etiquetas.csv"):
        (tmp_path / nombre).write_text("x", encoding="utf-8")
    assert datos.ultimo_csv_recuperacion(tmp_path).name == "recuperacion_20261120_100000.csv"


def test_la_recuperacion_muestra_estricta_y_ampliada_lado_a_lado(tmp_path):
    r = datos.leer_recuperacion(csv_de_recuperacion(tmp_path))
    assert (r["archivo"], r["consultas"], r["con_alternativas"], r["tiene_ampliada"]) == ("recuperacion_20261120_100000.csv", 3, 2, True)
    tabla = r["tabla"].set_index("grupo")
    assert list(tabla.index) == ["Global", "comercio", "normativa"]
    g = tabla.loc["Global"]
    assert g["consultas"] == 3
    assert g["hit@3 estricta"] == pytest.approx(2 / 3) and g["hit@3 ampliada"] == pytest.approx(1.0)   # la alternativa rescata qa2 en el rango 3
    assert g["hit@5 estricta"] == pytest.approx(2 / 3) and g["hit@5 ampliada"] == pytest.approx(1.0)
    assert g["MRR estricta"] == pytest.approx((1 + 0 + 1 / 2) / 3)
    assert g["MRR ampliada"] == pytest.approx((1 + 1 / 3 + 1 / 2) / 3)
    assert tabla.loc["comercio", "MRR estricta"] == tabla.loc["comercio", "MRR ampliada"] == 0.5
    assert r["sin_acierto"].to_dict("records") == [{"caso_id": "a", "variante": "qa2", "rango_ampliado": 3}]


def test_un_csv_de_recuperacion_anterior_a_las_alternativas_se_lee_y_avisa(tmp_path):
    viejo = tmp_path / "recuperacion_20261120_090000.csv"
    with open(viejo, "w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.writer(f)
        escritor.writerow(["caso_id", "categoria_caso", "variante", "fuente_esperada", "rango", "hit@3", "hit@5", "rr"])
        escritor.writerow(["a", "normativa", "qa", "regla", "1", "1", "1", "1.0"])
        escritor.writerow(["a", "normativa", "qa2", "regla", "", "0", "0", "0.0"])
    r = datos.leer_recuperacion(viejo)
    assert r["tiene_ampliada"] is False and r["con_alternativas"] == 0
    g = r["tabla"].set_index("grupo").loc["Global"]
    assert g["hit@3 estricta"] == g["hit@3 ampliada"] == 0.5 and g["MRR estricta"] == 0.5


def test_un_csv_que_no_es_de_recuperacion_se_rechaza_con_un_mensaje(tmp_path):
    ajeno = tmp_path / "recuperacion_20261120_090000.csv"
    ajeno.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no parece un CSV de recuperación"):
        datos.leer_recuperacion(ajeno)


# --------------------------------------------------------------------- trazas


def test_las_opciones_de_traza_van_de_la_mas_reciente_a_la_mas_antigua(base):
    df, _ = cargar(base[0])
    opciones = datos.opciones_de_traza(df)
    assert [i for i, _ in opciones] == list(range(9, 0, -1))
    assert opciones[0][1].startswith("#9 · 2026-11-03T10:00:00 · evaluacion · A-prep · INFORMATIVA")
    assert datos.opciones_de_traza(df, limite=2) == opciones[:2]
    larga = registrar(base[0], "2026-11-04", consulta="palabra " * 30)
    df, _ = cargar(base[0])
    assert datos.opciones_de_traza(df)[0][1].endswith("...") and len(datos.opciones_de_traza(df)[0][1]) < 140


def test_la_traza_trae_la_ejecucion_y_sus_pasos_con_herramienta_argumentos_duracion_y_exito(base):
    db, ids = base
    df, pasos = cargar(db)
    ejecucion, tabla = datos.traza_de(df, pasos, ids["e2"])
    assert (ejecucion["estado"], ejecucion["modelo"], ejecucion["llamadas_llm"], ejecucion["caso_id"]) == ("INFORMATIVA", "m1", 4, None)
    assert list(tabla.columns) == ["orden", "herramienta", "argumentos", "duracion_ms", "exito", "tamano_salida"]
    assert tabla[["orden", "herramienta", "duracion_ms"]].values.tolist() == [[1, "buscar_normativa", 300], [2, "buscar_comercio", 50]]
    assert tabla["exito"].tolist() == [True, True] and json.loads(tabla.loc[0, "argumentos"]) == {"q": "x"}

    _, sin_pasos = datos.traza_de(df, pasos, ids["e6"])   # derivada por el clasificador: sin herramientas
    assert sin_pasos.empty
    with pytest.raises(KeyError):
        datos.traza_de(df, pasos, 999)


# ======================================================================= interfaz


@pytest.fixture
def casos_de_prueba(tmp_path, monkeypatch):
    """eval/casos.json de prueba (no el real), para que la calidad salga de los datos sintéticos."""
    ruta = tmp_path / "casos_prueba.json"
    ruta.write_text(json.dumps({"casos": CASOS}), encoding="utf-8")
    monkeypatch.setattr(modulo_casos, "RUTA_CASOS", ruta)


def abrir():
    app = AppTest.from_file(str(RAIZ / "dashboard.py"), default_timeout=60)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def textos(elementos):
    return [e.value for e in elementos]


def metricas(app):
    return {m.label: m.value for m in app.metric}


def test_la_pagina_con_la_base_inexistente_muestra_un_mensaje_claro_y_no_crea_nada(db_observabilidad):
    assert not db_observabilidad.exists()
    app = abrir()
    assert any("Todavía no existe la base de observabilidad" in t for t in textos(app.warning))
    assert len(app.metric) == 0 and len(app.subheader) == 0     # no pinta secciones vacías
    assert not db_observabilidad.exists()                        # abrir el dashboard no crea la base


def test_la_pagina_con_la_base_vacia_muestra_un_mensaje_claro(db_observabilidad):
    store._conectar(db_observabilidad).close()
    app = abrir()
    assert any("todavía no tiene ejecuciones" in t for t in textos(app.warning))
    assert len(app.metric) == 0


def test_la_pagina_con_datos_muestra_las_seis_secciones_y_los_indicadores(base, casos_de_prueba):
    app = abrir()
    assert textos(app.subheader) == ["Indicadores", "Latencia", "Errores", "Calidad", "Recuperación", "Explorador de trazas"]
    m = metricas(app)
    assert m["Ejecuciones"] == "9" and m["Con error"] == "33.3%"
    assert m["Latencia total p50"] == "2.000 ms" and m["Latencia total p95"] == "4.700 ms"
    assert m["Tokens por consulta"] == "333" and m["Llamadas al LLM por consulta"] == "3.1"
    assert app.sidebar.metric[0].label == "Ejecuciones tras filtrar" and app.sidebar.metric[0].value == "9 de 9"


def test_la_pagina_avisa_lo_que_falta_en_calidad_y_en_recuperacion(base, casos_de_prueba):
    app = abrir()
    avisos = textos(app.warning)
    assert any("Faltan las etiquetas del estudiante" in t for t in avisos)
    assert any("2 modelos" in t for t in avisos)                 # la evaluación mezcla m1 y m2
    assert any("No hay CSV de recuperación" in t for t in textos(app.info))


def test_la_pagina_usa_las_etiquetas_y_el_csv_de_recuperacion_cuando_existen(base, casos_de_prueba, tmp_path):
    db, ids = base
    carpeta = tmp_path / "eval"
    carpeta.mkdir()
    (carpeta / "etiquetas.csv").write_text(f"id_ejecucion,correcto,comentario\n{ids['e4']},1,\n{ids['e5']},0,\n", encoding="utf-8")
    csv_de_recuperacion(carpeta)

    app = abrir()
    assert not any("Faltan las etiquetas" in t for t in textos(app.warning))
    assert any("2 ejecución(es) etiquetada(s)" in t for t in textos(app.caption))
    assert not any("No hay CSV de recuperación" in t for t in textos(app.info))
    assert any("recuperacion_20261120_100000.csv" in t for t in textos(app.caption))
    tablas = [d.value for d in app.dataframe]
    calidad = next(t for t in tablas if "Precisión de veredicto" in t.columns)
    todas = calidad[calidad["Categoría de caso"] == "Todas"].iloc[0]
    assert todas["Precisión de veredicto"] == "50.0%" and todas["Etiquetadas"] == 2
    assert calidad.loc[calidad["Categoría de caso"] == "comercio", "Precisión de clasificación"].iloc[0] == "—"   # sin ejecuciones
    recuperacion = next(t for t in tablas if "hit@3 ampliada" in t.columns)
    assert recuperacion.loc[0, "hit@3 estricta"] == pytest.approx(200 / 3) and recuperacion.loc[0, "hit@3 ampliada"] == 100.0


def test_un_archivo_de_etiquetas_con_errores_se_muestra_como_error_sin_romper_la_pagina(base, casos_de_prueba, tmp_path):
    carpeta = tmp_path / "eval"
    carpeta.mkdir()
    (carpeta / "etiquetas.csv").write_text("id_ejecucion,correcto,comentario\n1,quizas,\n", encoding="utf-8")
    app = abrir()
    assert any("tiene errores y no se usa" in t and "línea 2" in t for t in textos(app.error))


def test_los_filtros_cambian_el_conteo_y_los_indicadores(base, casos_de_prueba):
    app = abrir()
    modelos = next(w for w in app.sidebar.multiselect if w.label == "Modelo")
    modelos.select("m2").run()
    assert not app.exception
    assert app.sidebar.metric[0].value == "4 de 9"
    assert metricas(app)["Ejecuciones"] == "4"
    assert any("Muestra pequeña: 4 ejecución(es)" in t for t in textos(app.info))

    origen = next(w for w in app.sidebar.multiselect if w.label == "Origen")
    origen.select("app").run()                                        # m2 y app: no hay ninguna
    assert app.sidebar.metric[0].value == "0 de 9"
    assert any("Ninguna de las 9 ejecuciones cumple los filtros" in t for t in textos(app.warning))
    assert len(app.main.metric) == 0                                  # solo queda el conteo de la barra lateral


def test_el_filtro_de_tipo_de_error_usa_nombres_legibles(base, casos_de_prueba):
    app = abrir()
    tipo = next(w for w in app.sidebar.multiselect if w.label == "Tipo de error")
    assert tipo.options == ["429 (cuota)", "503 (servicio no disponible)", "Límite de iteraciones", "Sin error"]
    tipo.select("429 (cuota)").run()
    assert app.sidebar.metric[0].value == "1 de 9"
    assert metricas(app)["Con error"] == "100.0%"


def test_el_explorador_de_trazas_muestra_los_pasos_de_la_ejecucion_elegida(base, casos_de_prueba):
    db, ids = base
    app = abrir()
    traza = next(s for s in app.selectbox if s.label.startswith("Ejecución"))
    assert traza.value == 9                                           # la más reciente
    traza.select(ids["e2"]).run()
    assert not app.exception
    pasos = next(d.value for d in app.dataframe if "argumentos" in d.value.columns)
    assert pasos["herramienta"].tolist() == ["buscar_normativa", "buscar_comercio"]
    assert pasos["duracion_ms"].tolist() == [300, 50]

    traza.select(ids["e6"]).run()                                      # derivada de entrada: sin pasos
    assert any("no usó herramientas" in t for t in textos(app.info))


def test_si_ninguna_ejecucion_de_evaluacion_cuenta_para_la_calidad_se_explica_en_vez_de_mostrar_una_tabla_de_guiones(base):
    # sin `casos_de_prueba`: el eval/casos.json real no conoce los casos A, B ni C de los datos sintéticos
    app = abrir()
    assert any("Ninguna de las 6 ejecuciones de evaluación del filtro cuenta para la calidad" in t for t in textos(app.info))
    assert not any("Precisión de veredicto" in d.value.columns for d in app.dataframe)


def test_la_latencia_por_herramienta_ignora_los_pasos_de_las_ejecuciones_con_error(base):
    db, _ = base
    registrar(db, "2026-11-03", estado="ERROR", categoria=None, tipo_error="503", total=99999, respuesta=None,
              pasos=[("buscar_normativa", 99999)])   # falló después de usar una herramienta lenta
    df, pasos = cargar(db)
    herramienta = datos.latencia_por_herramienta(df, pasos).set_index("herramienta")
    assert herramienta.loc["buscar_normativa", "n"] == 5            # las mismas 5 duraciones de las completas
    assert herramienta.loc["buscar_normativa", "p95_ms"] == pytest.approx(800.0)


def test_los_numeros_enteros_de_la_ejecucion_no_se_muestran_como_decimales(base):
    db, ids = base
    df, pasos = cargar(db)
    # `repeticion` tiene vacíos en la base, así que pandas la lee como decimal: se devuelve entera
    ejecucion, _ = datos.traza_de(df, pasos, ids["e4"])
    assert ejecucion["repeticion"] == 1 and isinstance(ejecucion["repeticion"], int)
    assert isinstance(ejecucion["id"], int) and isinstance(ejecucion["latencia_total_ms"], int)
    assert isinstance(datos.ejecuciones_como_dicts(df, pasos)[3]["repeticion"], int)


def test_la_traza_de_una_ejecucion_de_evaluacion_indica_su_caso_y_repeticion(base, casos_de_prueba):
    db, ids = base
    app = abrir()
    next(s for s in app.selectbox if s.label.startswith("Ejecución")).select(ids["e4"]).run()
    assert any("caso A, repetición 1)" in m.value for m in app.markdown)


def test_si_el_filtro_deja_solo_ejecuciones_de_la_app_la_calidad_lo_explica(base, casos_de_prueba):
    app = abrir()
    next(w for w in app.sidebar.multiselect if w.label == "Origen").select("app").run()
    assert not app.exception
    assert app.sidebar.metric[0].value == "3 de 9"
    assert any("No hay ejecuciones de evaluación en el filtro actual" in t for t in textos(app.info))
    assert not any("Precisión de veredicto" in d.value.columns for d in app.dataframe)
