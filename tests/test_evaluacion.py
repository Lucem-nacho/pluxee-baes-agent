"""Pruebas del conjunto de evaluación y sus scripts (EP3, paso 2).

Todo con datos sintéticos y el modelo falso de `tests/modelo_falso.py`: sin
red ni Gemini. Cubren: columnas de caso y su migración, validación del
esquema, hit@k y MRR, reanudación, parada tras tres 429 y cada métrica de
`calcular_metricas` con valores calculados a mano.
"""

import csv
import itertools
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from modelo_falso import ESCALAR, error_429, error_503, llamada_tool, responder_con, respuesta_final

from config import settings
from memory.store import inicializar_db
from observabilidad import store
from observabilidad.casos import ErrorEsquema, cargar_casos, validar_casos
from observabilidad.eval_agente import (
    Combinacion,
    combinaciones,
    ejecutar,
    segundos_de_espera,
    separar_pendientes,
)
from observabilidad.eval_recuperacion import (
    evaluar,
    guardar_csv,
    hit_at_k,
    mrr,
    normalizar,
    rango_del_primer_acierto,
    resumir,
)
from observabilidad.metricas import (
    ErrorEtiquetas,
    calcular,
    cargar_ejecuciones,
    generar_plantilla_etiquetas,
    leer_etiquetas,
    percentil,
    veredicto_observado,
)

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def memoria_temporal(tmp_path, monkeypatch):
    # `responder` también escribe en la memoria de largo plazo (memoria.db).
    monkeypatch.setattr(settings, "MEMORIA_DB_PATH", tmp_path / "memoria_test.db")
    inicializar_db()


def filas(db_path, sql):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(f) for f in conn.execute(sql)]
    finally:
        conn.close()


# ============================================================ B. columnas de caso


ESQUEMA_V1 = """
CREATE TABLE ejecuciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT, fecha_utc TEXT NOT NULL, origen TEXT NOT NULL,
    id_usuario_anonimo TEXT NOT NULL, consulta TEXT NOT NULL, categoria TEXT, estado TEXT NOT NULL,
    latencia_total_ms INTEGER NOT NULL, latencia_clasificador_ms INTEGER, latencia_agente_ms INTEGER,
    llamadas_llm INTEGER NOT NULL, tokens_entrada INTEGER, tokens_salida INTEGER,
    iteraciones_agente INTEGER, tipo_error TEXT NOT NULL, modelo TEXT NOT NULL, respuesta TEXT
);
CREATE TABLE pasos (
    id INTEGER PRIMARY KEY AUTOINCREMENT, id_ejecucion INTEGER NOT NULL REFERENCES ejecuciones (id) ON DELETE CASCADE,
    orden INTEGER NOT NULL, herramienta TEXT NOT NULL, argumentos TEXT, duracion_ms INTEGER,
    exito INTEGER NOT NULL, tamano_salida INTEGER
);
INSERT INTO ejecuciones VALUES (1, '2026-11-01T10:00:00.000+00:00', 'app', 'abc123abc123', 'consulta vieja',
    'INFORMATIVA', 'INFORMATIVA', 10, 5, 5, 2, NULL, NULL, 1, 'ninguno', 'm', 'respuesta vieja');
INSERT INTO pasos VALUES (1, 1, 1, 'buscar_comercio', '{}', 3, 1, 9);
"""


def base_vieja(ruta):
    conn = sqlite3.connect(ruta)
    conn.executescript(ESQUEMA_V1)
    conn.commit()
    conn.close()


def columnas_de_ejecuciones(ruta):
    return [f["name"] for f in filas(ruta, "PRAGMA table_info(ejecuciones)")]


def test_la_migracion_agrega_las_columnas_a_una_base_existente_sin_perder_datos(tmp_path):
    ruta = tmp_path / "vieja.db"
    base_vieja(ruta)
    assert "caso_id" not in columnas_de_ejecuciones(ruta)

    store._conectar(ruta).close()

    assert {"caso_id", "variante", "repeticion"} <= set(columnas_de_ejecuciones(ruta))
    [vieja] = filas(ruta, "SELECT * FROM ejecuciones")
    assert vieja["consulta"] == "consulta vieja" and vieja["respuesta"] == "respuesta vieja"
    assert (vieja["caso_id"], vieja["variante"], vieja["repeticion"]) == (None, None, None)
    assert len(filas(ruta, "SELECT * FROM pasos")) == 1  # los pasos viejos siguen ahí


def test_la_migracion_es_idempotente_y_la_base_migrada_acepta_casos(tmp_path):
    ruta = tmp_path / "vieja.db"
    base_vieja(ruta)
    for _ in range(3):
        store._conectar(ruta).close()
    assert columnas_de_ejecuciones(ruta).count("caso_id") == 1

    nueva = {
        "origen": "evaluacion", "id_usuario_anonimo": "abc123abc123", "consulta": "q", "categoria": None,
        "estado": "INFORMATIVA", "latencia_total_ms": 1, "latencia_clasificador_ms": None,
        "latencia_agente_ms": None, "llamadas_llm": 1, "tokens_entrada": None, "tokens_salida": None,
        "iteraciones_agente": None, "tipo_error": "ninguno", "modelo": "m", "respuesta": None,
        "caso_id": "c1", "variante": "q", "repeticion": 2,
    }
    store.registrar_ejecucion(nueva, [], db_path=ruta)
    assert filas(ruta, "SELECT caso_id, repeticion FROM ejecuciones WHERE id = 2") == [{"caso_id": "c1", "repeticion": 2}]


def test_migrar_desde_varios_hilos_a_la_vez_no_falla(tmp_path):
    ruta = tmp_path / "vieja.db"
    base_vieja(ruta)
    errores = []

    def migrar():
        try:
            store._conectar(ruta).close()
        except Exception as error:  # pragma: no cover - lo que se quiere descartar
            errores.append(error)

    hilos = [threading.Thread(target=migrar) for _ in range(8)]
    [h.start() for h in hilos]
    [h.join() for h in hilos]
    assert errores == []
    assert columnas_de_ejecuciones(ruta).count("caso_id") == 1


def test_responder_guarda_caso_variante_y_repeticion(db_observabilidad):
    responder_con([respuesta_final("ok")], origen="evaluacion", caso_id="c1", variante="¿Mi RUT 12.345.678-9?", repeticion=2)
    responder_con([respuesta_final("ok")])  # las llamadas de siempre no cambian

    con_caso, sin_caso = filas(db_observabilidad, "SELECT caso_id, variante, repeticion, origen FROM ejecuciones ORDER BY id")
    assert con_caso == {"caso_id": "c1", "variante": "¿Mi RUT [RUT]?", "repeticion": 2, "origen": "evaluacion"}  # variante enmascarada
    assert (sin_caso["caso_id"], sin_caso["variante"], sin_caso["repeticion"], sin_caso["origen"]) == (None, None, None, "app")


def registrar(db, caso_id, variante, repeticion, *, origen="evaluacion", estado="INFORMATIVA", categoria="INFORMATIVA",
              tipo_error="ninguno", total=1000, clasificador=100, agente=900, llamadas=3, t_in=100, t_out=50,
              respuesta="No puedes comprar alcohol.", pasos=()):
    return store.registrar_ejecucion(
        {
            "origen": origen, "id_usuario_anonimo": "abc123abc123", "consulta": variante, "categoria": categoria,
            "estado": estado, "latencia_total_ms": total, "latencia_clasificador_ms": clasificador,
            "latencia_agente_ms": agente, "llamadas_llm": llamadas, "tokens_entrada": t_in, "tokens_salida": t_out,
            "iteraciones_agente": None, "tipo_error": tipo_error, "modelo": "m", "respuesta": respuesta,
            "caso_id": caso_id, "variante": variante, "repeticion": repeticion,
        },
        [
            {"orden": i, "herramienta": h, "argumentos": "{}", "duracion_ms": d, "exito": 1, "tamano_salida": 10}
            for i, (h, d) in enumerate(pasos, start=1)
        ],
        db_path=db,
    )


def test_combinaciones_evaluadas_no_cuenta_los_fallos_del_proveedor(db_observabilidad):
    registrar(db_observabilidad, "c", "hecha", 1)
    registrar(db_observabilidad, "c", "derivada", 1, estado="ESCALAR", categoria="ESCALAR")
    registrar(db_observabilidad, "c", "cuota", 1, estado="ERROR", tipo_error="429")
    registrar(db_observabilidad, "c", "caido", 1, estado="ERROR", tipo_error="503")
    registrar(db_observabilidad, "c", "bug", 1, estado="ERROR", tipo_error="otro")  # un error del código sí cuenta
    registrar(db_observabilidad, "c", "uso real", 1, origen="app")
    registrar(db_observabilidad, None, "sin caso", None)

    assert store.combinaciones_evaluadas() == {("c", "hecha", 1), ("c", "derivada", 1), ("c", "bug", 1)}


# ===================================================== C. validación del esquema


def caso(**cambios):
    base = {
        "id": "c1", "categoria_caso": "normativa", "pregunta": "¿Puedo comprar alcohol?", "variantes": ["¿Se puede comprar cerveza?"],
        "categoria_esperada": "INFORMATIVA", "herramientas_esperadas": ["buscar_normativa"],
        "veredicto_esperado": "prohibido", "fuente_esperada": "alcohol",
    }
    base.update(cambios)
    return {k: v for k, v in base.items() if v is not ...}


def errores_de(*casos):
    return validar_casos({"casos": list(casos)})


def test_un_caso_valido_no_tiene_errores():
    assert errores_de(caso()) == []
    assert errores_de(caso(fuente_esperada=None, variantes=[], ejemplo=True, notas="x")) == []


def test_el_casos_json_del_repositorio_es_valido_y_trae_solo_3_ejemplos():
    todos = cargar_casos(incluir_ejemplos=True)
    assert len(todos) == 3
    assert all(c["ejemplo"] is True for c in todos)
    assert {c["categoria_caso"] for c in todos} == {"normativa", "comercio", "derivar"}
    assert cargar_casos() == []  # por defecto los ejemplos se ignoran


@pytest.mark.parametrize(
    "cambios, esperado",
    [
        ({"id": ...}, "falta el campo 'id'"),
        ({"pregunta": ...}, "falta el campo 'pregunta'"),
        ({"id": "con espacio"}, "'id' debe ser texto sin espacios"),
        ({"categoria_caso": "normativo"}, "'categoria_caso' = 'normativo' no es válido; debe ser uno de: normativa, comercio, derivar, memoria"),
        ({"categoria_esperada": "informativa"}, "'categoria_esperada' = 'informativa' no es válido"),
        ({"veredicto_esperado": "prohibida"}, "'veredicto_esperado' = 'prohibida' no es válido; debe ser uno de: permitido, prohibido, no_confirmable, derivar"),
        ({"pregunta": "   "}, "'pregunta' debe ser un texto no vacío"),
        ({"variantes": "otra"}, "'variantes' debe ser una lista"),
        ({"variantes": [""]}, "la variante #1 debe ser un texto no vacío"),
        ({"variantes": ["¿PUEDO comprar alcohol?"]}, "la variante #1 repite la pregunta original"),
        ({"herramientas_esperadas": ["buscar_cosas"]}, "herramienta 'buscar_cosas' desconocida"),
        ({"herramientas_esperadas": ["buscar_normativa", "buscar_normativa"]}, "herramientas repetidas"),
        ({"herramientas_esperadas": "buscar_normativa"}, "'herramientas_esperadas' debe ser una lista"),
        ({"fuente_esperada": ""}, "'fuente_esperada' debe ser un texto no vacío o null"),
        ({"ejemplo": "si"}, "'ejemplo' debe ser true o false"),
        ({"veredicto_esperdo": "prohibido"}, "el campo 'veredicto_esperdo' no existe en el esquema"),
        # coherencia entre campos
        ({"categoria_esperada": "ESCALAR", "veredicto_esperado": "derivar"}, "el agente no llega a correr"),
        ({"categoria_esperada": "ESCALAR", "herramientas_esperadas": [], "veredicto_esperado": "prohibido"}, "debe ser 'derivar'"),
        ({"veredicto_esperado": "derivar"}, "solo ocurre si el agente llama a 'escalar_a_ejecutivo'"),
        ({"herramientas_esperadas": ["escalar_a_ejecutivo"]}, "si se espera 'escalar_a_ejecutivo'"),
    ],
)
def test_los_errores_del_esquema_dicen_que_caso_y_que_campo(cambios, esperado):
    mensajes = errores_de(caso(**cambios))
    assert any(esperado in m for m in mensajes), mensajes
    assert all(m.startswith("caso #1") for m in mensajes)  # siempre dice de qué caso habla


def test_un_id_repetido_se_senala_en_el_segundo_caso():
    mensajes = errores_de(caso(), caso(pregunta="otra pregunta"))
    assert any(m.startswith('caso #2 (id "c1")') and "repetido" in m for m in mensajes)


def test_se_informan_todos_los_problemas_y_no_solo_el_primero():
    assert len(errores_de(caso(categoria_caso="x", veredicto_esperado="y", pregunta=""))) == 3


@pytest.mark.parametrize(
    "datos, esperado",
    [
        ([], "objeto JSON con la clave 'casos'"),
        ({"casos": "nada"}, "'casos' debe ser una lista"),
        ({"casos": []}, "está vacío"),
        ({"casos": ["texto"]}, "caso #1: debe ser un objeto JSON"),
    ],
)
def test_forma_general_del_archivo(datos, esperado):
    assert any(esperado in m for m in validar_casos(datos))


def test_cargar_casos_con_archivo_roto_o_inexistente_lanza_un_error_claro(tmp_path):
    roto = tmp_path / "roto.json"
    roto.write_text('{"casos": [', encoding="utf-8")
    with pytest.raises(ErrorEsquema, match="no es JSON válido"):
        cargar_casos(roto)
    with pytest.raises(ErrorEsquema, match="no existe el archivo"):
        cargar_casos(tmp_path / "no_existe.json")

    invalido = tmp_path / "invalido.json"
    invalido.write_text(json.dumps({"casos": [caso(veredicto_esperado="x")]}), encoding="utf-8")
    with pytest.raises(ErrorEsquema) as capturada:
        cargar_casos(invalido)
    assert len(capturada.value.errores) == 1 and "veredicto_esperado" in str(capturada.value)


def test_los_ejemplos_se_filtran_salvo_que_se_pida():
    datos = {"casos": [caso(ejemplo=True), caso(id="c2", pregunta="otra")]}
    assert validar_casos(datos) == []


# ============================================== D1. hit@k, MRR y recuperación


def test_hit_at_k_y_mrr_con_valores_conocidos():
    rangos = [1, 4, None, 5, 2]
    assert hit_at_k(rangos, 3) == pytest.approx(2 / 5)      # rangos 1 y 2
    assert hit_at_k(rangos, 5) == pytest.approx(4 / 5)      # 1, 4, 5 y 2
    assert mrr(rangos) == pytest.approx((1 + 1 / 4 + 0 + 1 / 5 + 1 / 2) / 5)
    assert (hit_at_k([], 3), mrr([])) == (0.0, 0.0)
    assert (hit_at_k([None, None], 5), mrr([None, None])) == (0.0, 0.0)


def frag(texto, distancia, fuente="Manual BAES 2026"):
    return {"texto": texto, "distancia": distancia, "fuente": fuente}


def test_el_rango_usa_la_distancia_y_no_el_orden_de_llegada():
    fragmentos = [frag("sin relación", 0.9), frag("regla de ALCOHOL, tabaco", 0.4), frag("otro", 0.2)]
    assert rango_del_primer_acierto(fragmentos, "alcohol, tabaco") == 2


def test_la_fuente_esperada_se_busca_sin_mayusculas_tildes_ni_saltos_de_linea():
    assert normalizar("Sanción  DEFINITIVA\nde pérdida") == "sancion definitiva de perdida"
    assert rango_del_primer_acierto([frag("la\nsanción definitiva", 0.1)], "SANCIÓN definitiva") == 1
    # también cuenta si aparece en el nombre de la fuente
    assert rango_del_primer_acierto([frag("nada", 0.1, fuente="Resolución Exenta DN-00556/2026")], "dn-00556") == 1
    assert rango_del_primer_acierto([frag("nada", 0.1)], "alcohol") is None


def test_evaluar_calcula_una_fila_por_pregunta_y_omite_casos_sin_fuente():
    casos = [
        caso(id="a", variantes=["alcohol v1", "alcohol v2"]),
        caso(id="b", pregunta="comercio", variantes=[], fuente_esperada=None, categoria_caso="comercio"),
    ]
    resultados = {
        "¿Puedo comprar alcohol?": [frag("alcohol", 0.1)],                                       # rango 1
        "alcohol v1": [frag("x", 0.1), frag("x", 0.2), frag("x", 0.3), frag("alcohol", 0.4)],     # rango 4
        "alcohol v2": [frag("x", 0.1)],                                                         # sin acierto
    }
    pedidos = []

    def recuperar_falso(consulta, k):
        pedidos.append((consulta, k))
        return resultados[consulta]

    tabla = evaluar(casos, recuperar_falso, k=7)

    assert [(f["caso_id"], f["rango"], f["hit@3"], f["hit@5"]) for f in tabla] == [("a", 1, 1, 1), ("a", 4, 0, 1), ("a", None, 0, 0)]
    assert [round(f["rr"], 4) for f in tabla] == [1.0, 0.25, 0.0]
    assert [k for _, k in pedidos] == [7, 7, 7]          # el caso sin fuente no consulta nada
    resumen = resumir(tabla)
    assert resumen["global"]["n"] == 3
    assert resumen["global"]["hit@3"] == pytest.approx(1 / 3)
    assert resumen["global"]["hit@5"] == pytest.approx(2 / 3)
    assert resumen["global"]["mrr"] == pytest.approx((1 + 0.25 + 0) / 3)
    assert set(resumen["por_categoria_caso"]) == {"normativa"}


def test_el_csv_de_recuperacion_se_guarda_con_encabezado_y_rango_vacio_si_no_hay_acierto(tmp_path):
    tabla = evaluar([caso(variantes=[])], lambda c, k: [frag("nada", 0.1)])
    ruta = guardar_csv(tabla, tmp_path / "eval" / "r.csv")
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        [fila] = list(csv.DictReader(f))
    assert fila["rango"] == "" and fila["hit@3"] == "0" and fila["caso_id"] == "c1"


# ========================================================= D2. reanudación y 429


def responder_falso(consulta, historial, id_usuario, **kw):
    return responder_con([respuesta_final("ok")], consulta=consulta, id_usuario=id_usuario, **kw)


SIN_ESPERA = dict(pausa=0, dormir=lambda s: None, imprimir=lambda s: None)


def test_las_combinaciones_van_por_repeticion_y_cubren_todas_las_variantes():
    casos = [caso(id="a", variantes=["a2"]), caso(id="b", pregunta="pb", variantes=[])]
    todas = combinaciones(casos, repeticiones=2)
    assert [(c.caso_id, c.variante, c.repeticion) for c in todas] == [
        ("a", "¿Puedo comprar alcohol?", 1), ("a", "a2", 1), ("b", "pb", 1),
        ("a", "¿Puedo comprar alcohol?", 2), ("a", "a2", 2), ("b", "pb", 2),
    ]


def test_la_evaluacion_se_reanuda_saltando_lo_que_ya_esta_en_la_base(db_observabilidad):
    casos = [caso(id="a", variantes=["a2"]), caso(id="b", pregunta="pb", variantes=[])]
    todas = combinaciones(casos, repeticiones=2)  # 6 combinaciones

    pendientes, saltadas = separar_pendientes(todas, store.combinaciones_evaluadas())
    primera = ejecutar(pendientes, responder_falso, saltadas=saltadas, limite=4, **SIN_ESPERA)
    assert (primera.completadas, primera.llamadas, primera.detenida_por, primera.faltan) == (4, 4, "limite", 2)
    assert len(filas(db_observabilidad, "SELECT * FROM ejecuciones WHERE origen = 'evaluacion'")) == 4

    pendientes, saltadas = separar_pendientes(todas, store.combinaciones_evaluadas())
    assert (len(pendientes), saltadas) == (2, 4)
    segunda = ejecutar(pendientes, responder_falso, saltadas=saltadas, **SIN_ESPERA)
    assert (segunda.completadas, segunda.faltan, segunda.detenida_por, segunda.saltadas) == (2, 0, None, 4)

    pendientes, saltadas = separar_pendientes(todas, store.combinaciones_evaluadas())
    assert (pendientes, saltadas) == ([], 6)
    assert len(filas(db_observabilidad, "SELECT * FROM ejecuciones WHERE origen = 'evaluacion'")) == 6  # nada se repitió


def test_las_ejecuciones_quedan_marcadas_con_caso_variante_y_repeticion(db_observabilidad):
    pendientes = combinaciones([caso(id="a", variantes=["a2"])], repeticiones=1)
    ejecutar(pendientes, responder_falso, **SIN_ESPERA)
    assert filas(db_observabilidad, "SELECT origen, caso_id, variante, repeticion FROM ejecuciones ORDER BY id") == [
        {"origen": "evaluacion", "caso_id": "a", "variante": "¿Puedo comprar alcohol?", "repeticion": 1},
        {"origen": "evaluacion", "caso_id": "a", "variante": "a2", "repeticion": 1},
    ]


def combo(n=1):
    return [Combinacion("c", f"pregunta {i}", 1) for i in range(n)]


def responder_que_falla_con(errores):
    """Un responder que devuelve/lanza lo que diga la lista, en orden."""
    llamadas = []
    secuencia = iter(errores)

    def fn(consulta, historial, id_usuario, **kw):
        llamadas.append(consulta)
        accion = next(secuencia)
        if isinstance(accion, BaseException):
            raise accion
        return {"estado": "INFORMATIVA"}

    fn.llamadas = llamadas
    return fn


def corrida(pendientes, fn, **kw):
    esperas = []
    mensajes = []
    resultado = ejecutar(pendientes, fn, dormir=esperas.append, imprimir=mensajes.append, pausa=kw.pop("pausa", 15), **kw)
    return resultado, esperas, mensajes


def test_ante_un_429_espera_el_retry_delay_y_reintenta_una_vez():
    fn = responder_que_falla_con([error_429("7s"), "ok"])
    resultado, esperas, _ = corrida(combo(1), fn)
    assert fn.llamadas == ["pregunta 0", "pregunta 0"]  # la misma combinación, dos veces
    assert esperas == [8.0]                              # retryDelay (7s) + 1s de margen
    assert (resultado.completadas, resultado.fallidas, resultado.faltan, resultado.detenida_por) == (1, 0, 0, None)


def test_tras_tres_429_seguidos_se_detiene_limpio_e_informa_cuantas_faltan():
    fn = responder_que_falla_con([error_429("7s")] * 10)
    resultado, esperas, mensajes = corrida(combo(5), fn)
    # combinación 1: 429 -> espera -> reintento 429 (ya van 2); combinación 2: 429 (3) -> se detiene
    assert len(fn.llamadas) == 3
    assert esperas == [8.0, 15]                           # espera del reintento + pausa entre llamadas
    assert resultado.detenida_por == "cuota" and "3 respuestas 429 seguidas" in resultado.detalle
    assert (resultado.completadas, resultado.faltan) == (0, 5)   # nada se perdió: las 5 siguen pendientes


def test_un_exito_pone_en_cero_el_contador_de_429():
    # 429 + reintento exitoso, cinco veces seguidas: nunca hay 3 seguidos
    fn = responder_que_falla_con([error_429("1s"), "ok"] * 5)
    resultado, _, _ = corrida(combo(5), fn)
    assert (resultado.completadas, resultado.faltan, resultado.detenida_por) == (5, 0, None)
    assert len(fn.llamadas) == 10


def test_un_error_que_no_es_429_no_detiene_la_corrida_ni_cuenta_como_429():
    fn = responder_que_falla_con([error_429("1s"), RuntimeError("falló"), error_429("1s"), "ok", "ok"])
    resultado, _, mensajes = corrida(combo(3), fn, max_429_seguidos=2)
    # el RuntimeError corta la racha de 429; sin eso, el tercer 429 habría detenido todo
    assert resultado.detenida_por is None
    assert (resultado.completadas, resultado.fallidas) == (2, 1)
    assert any("RuntimeError" in m for m in mensajes)


def test_un_retry_delay_larguisimo_se_toma_como_cuota_diaria_y_no_se_duerme():
    fn = responder_que_falla_con([error_429("8336s")])
    resultado, esperas, _ = corrida(combo(3), fn)
    assert fn.llamadas == ["pregunta 0"] and esperas == []
    assert resultado.detenida_por == "cuota" and "cuota diaria" in resultado.detalle and "8336" in resultado.detalle
    assert resultado.faltan == 3


def test_el_limite_cuenta_tambien_los_reintentos():
    fn = responder_que_falla_con([error_429("1s"), "ok", "ok"])
    resultado, _, _ = corrida(combo(3), fn, limite=2)
    assert len(fn.llamadas) == 2 and resultado.detenida_por == "limite"
    assert (resultado.completadas, resultado.faltan) == (1, 2)


def test_la_pausa_se_hace_entre_llamadas_y_no_antes_de_la_primera():
    fn = responder_que_falla_con(["ok"] * 3)
    _, esperas, _ = corrida(combo(3), fn, pausa=15)
    assert esperas == [15, 15]


def test_un_429_real_queda_registrado_como_error_y_se_reintenta_en_la_proxima_corrida(db_observabilidad):
    def responder_sin_cuota(consulta, historial, id_usuario, **kw):
        return responder_con(error=error_429("1s"), consulta=consulta, id_usuario=id_usuario, **kw)

    pendientes = combinaciones([caso(id="a", variantes=["a2"]), caso(id="b", pregunta="pb", variantes=[])], 1)
    resultado, _, _ = corrida(pendientes, responder_sin_cuota)

    assert resultado.detenida_por == "cuota"
    errores = filas(db_observabilidad, "SELECT estado, tipo_error FROM ejecuciones")
    assert len(errores) == 3 and all(f == {"estado": "ERROR", "tipo_error": "429"} for f in errores)
    assert store.combinaciones_evaluadas() == set()      # ninguna cuenta como hecha

    pendientes, saltadas = separar_pendientes(pendientes, store.combinaciones_evaluadas())
    assert (len(pendientes), saltadas) == (3, 0)
    final = ejecutar(pendientes, responder_falso, **SIN_ESPERA)
    assert (final.completadas, final.faltan) == (3, 0)


def test_segundos_de_espera_lee_el_cuerpo_del_sdk_el_texto_y_la_cadena_de_causas():
    assert segundos_de_espera(error_429("34s")) == 34.0                      # RetryInfo, vía __cause__
    assert segundos_de_espera(RuntimeError("quota. Please retry in 59.68s.")) == pytest.approx(59.68)
    assert segundos_de_espera(RuntimeError("{'retryDelay': '12s'}")) == 12.0
    assert segundos_de_espera(RuntimeError("sin pista")) is None
    assert segundos_de_espera(error_503()) is None


# ====================================================== D3. métricas con valores conocidos

CASOS_METRICAS = [
    {"id": "A", "categoria_caso": "normativa", "pregunta": "pa", "variantes": ["pa2"], "categoria_esperada": "INFORMATIVA",
     "herramientas_esperadas": ["buscar_normativa"], "veredicto_esperado": "prohibido", "fuente_esperada": "alcohol"},
    {"id": "B", "categoria_caso": "derivar", "pregunta": "pb", "variantes": [], "categoria_esperada": "ESCALAR",
     "herramientas_esperadas": [], "veredicto_esperado": "derivar", "fuente_esperada": None},
]


def sembrar_metricas(db):
    """Datos sintéticos cuyas métricas están calculadas a mano en los tests."""
    assert validar_casos({"casos": CASOS_METRICAS}) == []
    ids = {}
    ids["A_pa_1"] = registrar(db, "A", "pa", 1, pasos=[("buscar_normativa", 100)])
    ids["A_pa_2"] = registrar(db, "A", "pa", 2, total=2000, clasificador=200, agente=1800, llamadas=4, t_in=200, t_out=80,
                              pasos=[("buscar_normativa", 200), ("buscar_comercio", 50)])
    ids["A_pa_3"] = registrar(db, "A", "pa", 3, estado="ERROR", tipo_error="429", total=500, agente=400, llamadas=2,
                              t_in=None, t_out=None, respuesta=None)
    ids["A_pa2_1"] = registrar(db, "A", "pa2", 1, total=3000, clasificador=300, agente=2700, t_in=300, t_out=90,
                               respuesta="Sí puedes comprar eso.", pasos=[("buscar_normativa", 300)])
    ids["B_1"] = registrar(db, "B", "pb", 1, estado="ESCALAR", categoria="ESCALAR", total=400, clasificador=400, agente=None,
                           llamadas=1, t_in=None, t_out=None, respuesta=None)
    ids["B_2"] = registrar(db, "B", "pb", 2, total=600, clasificador=100, agente=500, llamadas=2, t_in=40, t_out=20, respuesta="Todo bien")
    # ruido que NO debe contar:
    registrar(db, None, "humo", None)                       # sin caso (p. ej. check_agent_executor)
    registrar(db, "ZZ", "pz", 1)                            # caso que no está en casos.json
    registrar(db, "A", "texto viejo", 1)                    # variante que ya no existe
    registrar(db, "A", "pa", 1, origen="app")               # uso real
    return ids


def metricas(db, etiquetas=None, casos=CASOS_METRICAS):
    return calcular(casos, cargar_ejecuciones(db), etiquetas)


def test_conteo_de_ejecuciones_y_exclusiones(db_observabilidad):
    sembrar_metricas(db_observabilidad)
    e = metricas(db_observabilidad)["ejecuciones"]
    assert e == {"evaluacion_total": 9, "sin_caso_excluidas": 1, "fuera_del_conjunto_excluidas": 2,
                 "consideradas": 6, "completas": 5, "con_error": 1, "unicas": 5}


def test_precision_de_clasificacion_y_de_herramientas(db_observabilidad):
    sembrar_metricas(db_observabilidad)
    m = metricas(db_observabilidad)
    # clasificación: solo B_2 falla (esperaba ESCALAR) -> 4 de 5
    assert m["precision_clasificacion"] == {"n": 5, "correctas": 4, "precision": 0.8}
    # herramientas (conjunto): solo A_pa_2 falla (usó una de más) -> 4 de 5
    assert m["precision_herramientas"] == {"n": 5, "correctas": 4, "precision": 0.8}
    assert m["por_caso"]["A"]["precision_clasificacion"] == 1.0
    assert m["por_caso"]["B"]["precision_clasificacion"] == 0.5
    assert m["por_caso"]["A"]["precision_herramientas"] == pytest.approx(2 / 3)
    assert m["por_caso"]["B"]["precision_herramientas"] == 1.0


def test_las_herramientas_se_comparan_como_conjunto_sin_orden_ni_repeticiones(db_observabilidad):
    casos = [{**CASOS_METRICAS[0], "herramientas_esperadas": ["buscar_normativa", "buscar_comercio"]}]
    registrar(db_observabilidad, "A", "pa", 1, pasos=[("buscar_comercio", 5), ("buscar_normativa", 5), ("buscar_normativa", 5)])
    registrar(db_observabilidad, "A", "pa", 2, pasos=[("buscar_normativa", 5)])                        # falta una
    registrar(db_observabilidad, "A", "pa", 3, pasos=[("buscar_normativa", 5), ("buscar_comercio", 5), ("consultar_historial", 5)])  # sobra una
    m = calcular(casos, cargar_ejecuciones(db_observabilidad))
    assert m["precision_herramientas"] == {"n": 3, "correctas": 1, "precision": pytest.approx(1 / 3)}


def test_precision_de_veredicto_a_partir_de_las_etiquetas(db_observabilidad):
    ids = sembrar_metricas(db_observabilidad)
    etiquetas = {
        ids["A_pa_1"]: {"correcto": "1"}, ids["A_pa_2"]: {"correcto": "parcial"},
        ids["A_pa2_1"]: {"correcto": "0"}, ids["B_1"]: {"correcto": "1"},
        999: {"correcto": "1"},                                 # no es una ejecución de la evaluación
    }
    v = metricas(db_observabilidad, etiquetas)["precision_veredicto"]
    assert (v["etiquetadas"], v["correctas"], v["parciales"], v["incorrectas"]) == (4, 2, 1, 1)
    assert v["precision_estricta"] == 0.5
    assert v["precision_ponderada"] == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    assert v["sin_etiquetar"] == 1                              # B_2
    assert v["etiquetas_de_ejecuciones_ajenas"] == 1            # el 999
    assert metricas(db_observabilidad)["precision_veredicto"]["precision_estricta"] is None  # sin etiquetas


def test_consistencia_entre_repeticiones(db_observabilidad):
    sembrar_metricas(db_observabilidad)
    c = metricas(db_observabilidad)["consistencia"]
    # grupos con 2+ repeticiones: (A, pa) y (B, pb); (A, pa2) tiene una sola
    assert (c["grupos_evaluables"], c["grupos_sin_repeticion"], c["repeticiones"]) == (2, 1, 4)
    # (A,pa): mismo veredicto en 2/2 pero secuencias distintas; (B,pb): lo contrario
    assert c["mismo_veredicto"] == 0.75     # (2 + 1) / 4
    assert c["misma_secuencia"] == 0.75     # (1 + 2) / 4
    assert c["ambos"] == 0.5                # (1 + 1) / 4
    assert c["por_caso"]["A"] == {"repeticiones": 2, "mismo_veredicto": 1.0, "misma_secuencia": 0.5, "ambos": 0.5}
    assert c["por_caso"]["B"] == {"repeticiones": 2, "mismo_veredicto": 0.5, "misma_secuencia": 1.0, "ambos": 0.5}


def test_el_veredicto_observado_del_estudiante_reemplaza_a_la_heuristica(db_observabilidad):
    ids = sembrar_metricas(db_observabilidad)
    etiquetas = {ids["B_2"]: {"correcto": "0", "veredicto_observado": "derivar"}}
    c = metricas(db_observabilidad, etiquetas)["consistencia"]
    assert c["mismo_veredicto"] == 1.0      # (2 + 2) / 4: B_2 ahora cuenta como "derivar"
    assert c["ambos"] == 0.75               # (1 + 2) / 4


def test_tasa_de_errores_por_tipo(db_observabilidad):
    sembrar_metricas(db_observabilidad)
    t = metricas(db_observabilidad)["tasa_errores"]
    assert t["ejecuciones"] == 6
    assert t["por_tipo"]["429"] == {"n": 1, "tasa": pytest.approx(1 / 6)}
    assert t["por_tipo"]["ninguno"] == {"n": 5, "tasa": pytest.approx(5 / 6)}
    assert t["por_tipo"]["503"] == {"n": 0, "tasa": 0.0}
    assert set(t["por_tipo"]) == set(store.TIPOS_ERROR)


def test_percentiles_p50_y_p95_por_etapa_y_por_herramienta(db_observabilidad):
    sembrar_metricas(db_observabilidad)
    lat = metricas(db_observabilidad)["latencia_ms"]
    # total (completas): 400 600 1000 2000 3000 -> p50 = 1000; p95 = 2000 + 0,8 * 1000
    assert lat["por_etapa"]["total"] == {"n": 5, "p50_ms": 1000.0, "p95_ms": 2800.0}
    # clasificador: 100 100 200 300 400 -> p50 = 200; p95 = 300 + 0,8 * 100
    assert lat["por_etapa"]["clasificador"] == {"n": 5, "p50_ms": 200.0, "p95_ms": 380.0}
    # agente (B_1 no lo corrió): 500 900 1800 2700 -> p50 = 1350; p95 = 1800 + 0,85 * 900
    assert lat["por_etapa"]["agente"] == {"n": 4, "p50_ms": 1350.0, "p95_ms": 2565.0}
    assert lat["por_herramienta"]["buscar_normativa"] == {"n": 3, "p50_ms": 200.0, "p95_ms": 290.0}
    assert lat["por_herramienta"]["buscar_comercio"] == {"n": 1, "p50_ms": 50.0, "p95_ms": 50.0}


def test_percentil_con_interpolacion_lineal():
    assert percentil(list(range(1, 101)), 50) == pytest.approx(50.5)
    assert percentil(list(range(1, 101)), 95) == pytest.approx(95.05)
    assert percentil([7], 95) == 7 and percentil([], 50) is None
    assert percentil([3, 1, 2], 100) == 3 and percentil([3, 1, 2], 0) == 1


def test_llamadas_y_tokens_por_consulta(db_observabilidad):
    sembrar_metricas(db_observabilidad)
    k = metricas(db_observabilidad)["consumo_por_consulta"]
    assert k["consultas"] == 5
    assert k["llamadas_llm_promedio"] == 2.6            # (3 + 4 + 3 + 1 + 2) / 5
    assert k["tokens_entrada_promedio"] == 160.0        # (100 + 200 + 300 + 40) / 4: B_1 no tiene tokens
    assert k["tokens_salida_promedio"] == 60.0          # (50 + 80 + 90 + 20) / 4
    assert k["consultas_con_tokens"] == 4


def test_los_ejemplos_no_cuentan_cuando_el_caso_activo_no_los_incluye(db_observabilidad):
    sembrar_metricas(db_observabilidad)
    solo_b = [CASOS_METRICAS[1]]
    e = metricas(db_observabilidad, casos=solo_b)["ejecuciones"]
    assert e["consideradas"] == 2 and e["fuera_del_conjunto_excluidas"] == 6  # todo lo de A queda fuera


def test_si_una_combinacion_tiene_varias_ejecuciones_la_precision_usa_la_ultima(db_observabilidad):
    registrar(db_observabilidad, "A", "pa", 1, categoria="ESCALAR", estado="ESCALAR")   # primer intento: mal clasificada
    registrar(db_observabilidad, "A", "pa", 1, pasos=[("buscar_normativa", 10)])         # reintento: bien
    m = metricas(db_observabilidad)
    assert m["ejecuciones"]["completas"] == 2 and m["ejecuciones"]["unicas"] == 1
    assert m["precision_clasificacion"] == {"n": 1, "correctas": 1, "precision": 1.0}


@pytest.mark.parametrize(
    "estado, tipo_error, respuesta, esperado",
    [
        ("ESCALAR", "ninguno", None, "derivar"),
        ("ERROR", "429", None, "error"),
        ("INFORMATIVA", "limite_iteraciones", "texto cualquiera", "no_confirmable"),
        ("INFORMATIVA", "sin_respuesta", "", "no_confirmable"),
        ("INFORMATIVA", "ninguno", "No, no puedes comprar alcohol con la BAES.", "prohibido"),
        ("INFORMATIVA", "ninguno", "Sí, puedes comprar fármacos de venta libre.", "permitido"),
        ("INFORMATIVA", "ninguno", "Puedes comprar alimentos con hasta 2 sellos. Las fuentes no mencionan las bebidas energéticas.", "no_confirmable"),
        ("INFORMATIVA", "ninguno", "Hola, ¿en qué te ayudo?", "indeterminado"),
    ],
)
def test_veredicto_observado_heuristico(estado, tipo_error, respuesta, esperado):
    assert veredicto_observado({"estado": estado, "tipo_error": tipo_error, "respuesta": respuesta}) == esperado


# ----------------------------------------------------------- etiquetas (CSV)


def escribir_csv(ruta, texto, encoding="utf-8"):
    ruta.write_text(texto, encoding=encoding)
    return ruta


def test_leer_etiquetas_acepta_coma_punto_y_coma_bom_y_filas_sin_etiquetar(tmp_path):
    coma = escribir_csv(tmp_path / "a.csv", "id_ejecucion,correcto,comentario\n1,1,bien\n2,Parcial,\n3,,\n4, 0 ,mal\n", "utf-8-sig")
    assert leer_etiquetas(coma) == {
        1: {"correcto": "1", "comentario": "bien", "veredicto_observado": None},
        2: {"correcto": "parcial", "comentario": "", "veredicto_observado": None},
        4: {"correcto": "0", "comentario": "mal", "veredicto_observado": None},
    }
    excel_es = escribir_csv(tmp_path / "b.csv", "id_ejecucion;correcto;comentario;veredicto_observado\n7;1;ok;prohibido\n")
    assert leer_etiquetas(excel_es) == {7: {"correcto": "1", "comentario": "ok", "veredicto_observado": "prohibido"}}
    assert leer_etiquetas(escribir_csv(tmp_path / "vacio.csv", "")) == {}


@pytest.mark.parametrize(
    "contenido, esperado",
    [
        ("id_ejecucion,correcto,comentario\n1,si,\n", "línea 2: 'correcto' = 'si' no es válido; debe ser 1, 0 o parcial"),
        ("id_ejecucion,correcto,comentario\n1,1,\nabc,1,\n", "línea 3: 'id_ejecucion' = 'abc' no es un número entero"),
        ("id_ejecucion,correcto,comentario\n1,1,\n1,0,\n", "línea 3: la ejecución 1 está etiquetada dos veces"),
        ("id_ejecucion,correcto,comentario,veredicto_observado\n1,1,,quizas\n", "'veredicto_observado' = 'quizas' no es válido"),
        ("id,correcto\n1,1\n", "faltan las columnas ['id_ejecucion']"),
    ],
)
def test_las_etiquetas_invalidas_se_rechazan_con_la_linea_del_problema(tmp_path, contenido, esperado):
    with pytest.raises(ErrorEtiquetas, match="problema") as capturada:
        leer_etiquetas(escribir_csv(tmp_path / "e.csv", contenido))
    assert esperado in str(capturada.value)


def test_la_plantilla_de_etiquetas_trae_una_fila_por_ejecucion_completa_y_correcto_vacio(db_observabilidad, tmp_path):
    ids = sembrar_metricas(db_observabilidad)
    ruta = tmp_path / "eval" / "etiquetas.csv"

    escritas = generar_plantilla_etiquetas(ruta, CASOS_METRICAS, cargar_ejecuciones(db_observabilidad))

    assert escritas == 5   # sin la que falló con 429 ni el ruido
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        lector = csv.DictReader(f)
        assert lector.fieldnames[:3] == ["id_ejecucion", "correcto", "comentario"]
        tabla = list(lector)
    assert [int(f["id_ejecucion"]) for f in tabla] == [ids["A_pa_1"], ids["A_pa_2"], ids["A_pa2_1"], ids["B_1"], ids["B_2"]]
    assert all(f["correcto"] == "" and f["comentario"] == "" for f in tabla)
    assert tabla[0]["veredicto_esperado"] == "prohibido" and "No puedes" in tabla[0]["respuesta"]
    assert leer_etiquetas(ruta) == {}  # vacía: no hay nada etiquetado todavía


# --------------------------------------------------------------- scripts (CLI)


def correr(script, *args, db_path, **extra_env):
    entorno = {**os.environ, "OBSERVABILIDAD_DB_PATH": str(db_path), "PYTHONIOENCODING": "utf-8", **extra_env}
    return subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / script), *map(str, args)],
        cwd=RAIZ, env=entorno, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )


def test_calcular_metricas_genera_la_plantilla_calcula_y_escribe_el_json(db_observabilidad, tmp_path):
    ids = sembrar_metricas(db_observabilidad)
    casos = tmp_path / "casos.json"
    casos.write_text(json.dumps({"casos": CASOS_METRICAS}), encoding="utf-8")
    etiquetas, salida = tmp_path / "e" / "etiquetas.csv", tmp_path / "e" / "metricas.json"
    args = ("--casos-path", casos, "--etiquetas", etiquetas, "--salida", salida)

    primera = correr("calcular_metricas.py", *args, db_path=db_observabilidad)
    assert primera.returncode == 0, primera.stderr
    assert "generé la plantilla con 5 fila(s)" in primera.stdout
    resultado = json.loads(salida.read_text(encoding="utf-8"))
    assert resultado["precision_clasificacion"]["precision"] == 0.8
    assert resultado["precision_veredicto"]["etiquetadas"] == 0

    # la estudiante etiqueta una fila; la segunda corrida NO pisa el archivo y la lee
    texto = etiquetas.read_text(encoding="utf-8-sig").splitlines()
    texto[1] = texto[1].replace(f"{ids['A_pa_1']},,", f"{ids['A_pa_1']},1,", 1)
    etiquetas.write_text("\n".join(texto), encoding="utf-8-sig")
    segunda = correr("calcular_metricas.py", *args, db_path=db_observabilidad)
    assert segunda.returncode == 0, segunda.stderr and "generé la plantilla" not in segunda.stdout
    assert json.loads(salida.read_text(encoding="utf-8"))["precision_veredicto"]["etiquetadas"] == 1
    assert "Precisión de clasificación: 80.0% (4/5)" in segunda.stdout


def test_calcular_metricas_rechaza_etiquetas_invalidas_y_casos_invalidos(db_observabilidad, tmp_path):
    casos = tmp_path / "casos.json"
    casos.write_text(json.dumps({"casos": CASOS_METRICAS}), encoding="utf-8")
    malas = escribir_csv(tmp_path / "malas.csv", "id_ejecucion,correcto,comentario\n1,quizas,\n")

    r = correr("calcular_metricas.py", "--casos-path", casos, "--etiquetas", malas, "--salida", tmp_path / "m.json", db_path=db_observabilidad)
    assert r.returncode == 2 and "línea 2" in r.stderr and not (tmp_path / "m.json").exists()

    casos.write_text(json.dumps({"casos": [caso(veredicto_esperado="x")]}), encoding="utf-8")
    r = correr("calcular_metricas.py", "--casos-path", casos, "--etiquetas", malas, "--salida", tmp_path / "m.json", db_path=db_observabilidad)
    assert r.returncode == 2 and "veredicto_esperado" in r.stderr


def test_evaluar_agente_sin_casos_reales_o_sin_pendientes_no_llama_a_gemini(db_observabilidad, tmp_path):
    # solo hay ejemplos en eval/casos.json: se ignoran y el script termina sin importar el agente
    r = correr("evaluar_agente.py", db_path=db_observabilidad)
    assert r.returncode == 0 and "No hay casos para evaluar" in r.stdout
    assert not db_observabilidad.exists()

    # con todo ya hecho en la base tampoco hay nada que correr
    casos = tmp_path / "casos.json"
    casos.write_text(json.dumps({"casos": [caso(id="a", variantes=[])]}), encoding="utf-8")
    registrar(db_observabilidad, "a", "¿Puedo comprar alcohol?", 1)
    r = correr("evaluar_agente.py", "--casos-path", casos, "--repeticiones", "1", db_path=db_observabilidad)
    assert r.returncode == 0 and "1 ya estaban en la base, 0 pendientes" in r.stdout and "Nada que hacer" in r.stdout


@pytest.mark.parametrize("argumento", [("--repeticiones", "0"), ("--limite", "0"), ("--pausa", "-1")])
def test_evaluar_agente_rechaza_parametros_invalidos(db_observabilidad, argumento):
    r = correr("evaluar_agente.py", *argumento, db_path=db_observabilidad)
    assert r.returncode == 2 and "error:" in r.stderr


def test_evaluar_agente_rechaza_ids_inexistentes_y_casos_invalidos(db_observabilidad, tmp_path):
    r = correr("evaluar_agente.py", "--casos", "no_existe", db_path=db_observabilidad)
    assert r.returncode == 2 and "ejemplos sin --incluir-ejemplos" in r.stderr  # los ejemplos no cuentan sin esa opción

    casos = tmp_path / "casos.json"
    casos.write_text(json.dumps({"casos": [caso(id="a")]}), encoding="utf-8")
    r = correr("evaluar_agente.py", "--casos-path", casos, "--casos", "b", db_path=db_observabilidad)
    assert r.returncode == 2 and "no existen" in r.stderr

    casos.write_text(json.dumps({"casos": [caso(categoria_caso="x")]}), encoding="utf-8")
    r = correr("evaluar_agente.py", "--casos-path", casos, db_path=db_observabilidad)
    assert r.returncode == 2 and "categoria_caso" in r.stderr


def test_evaluar_recuperacion_sin_casos_con_fuente_termina_sin_cargar_el_indice(db_observabilidad):
    r = correr("evaluar_recuperacion.py", db_path=db_observabilidad)
    assert r.returncode == 0 and "No hay casos con 'fuente_esperada'" in r.stdout
