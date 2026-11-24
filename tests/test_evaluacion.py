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
from collections import Counter
from pathlib import Path

import pytest
from modelo_falso import ESCALAR, error_429, error_503, llamada_tool, responder_con, respuesta_final

from config import settings
from memory.store import consultar_historial_usuario, inicializar_db
from observabilidad import store
from observabilidad.casos import ErrorEsquema, advertencias_de_casos, cargar_casos, validar_casos
from observabilidad.eval_agente import (
    Combinacion,
    combinaciones,
    ejecutar,
    id_usuario_del_caso,
    planificar,
    preparacion_de,
    segundos_de_espera,
    separar_pendientes,
)
from observabilidad.eval_recuperacion import (
    COLUMNAS_CSV,
    evaluar,
    formatear_detalle,
    guardar_csv,
    hit_at_k,
    mrr,
    normalizar,
    rango_ampliado,
    rango_del_primer_acierto,
    resumir,
    separar_evaluables,
)
from observabilidad.metricas import (
    ErrorEtiquetas,
    calcular,
    cargar_ejecuciones,
    generar_plantilla_etiquetas,
    leer_etiquetas,
    percentil,
    resumen_consola,
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
              tipo_error="ninguno", modelo="m", total=1000, clasificador=100, agente=900, llamadas=3, t_in=100, t_out=50,
              respuesta="No puedes comprar alcohol.", pasos=()):
    return store.registrar_ejecucion(
        {
            "origen": origen, "id_usuario_anonimo": "abc123abc123", "consulta": variante, "categoria": categoria,
            "estado": estado, "latencia_total_ms": total, "latencia_clasificador_ms": clasificador,
            "latencia_agente_ms": agente, "llamadas_llm": llamadas, "tokens_entrada": t_in, "tokens_salida": t_out,
            "iteraciones_agente": None, "tipo_error": tipo_error, "modelo": modelo, "respuesta": respuesta,
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


def avisos_de(*casos):
    return advertencias_de_casos({"casos": list(casos)})


def test_un_caso_valido_no_tiene_errores():
    assert errores_de(caso()) == []
    assert errores_de(caso(fuente_esperada=None, variantes=[], ejemplo=True, notas="x")) == []


IDS_DE_LOS_14_CASOS = [
    "normativa-02", "normativa-03", "normativa-04a", "normativa-04b", "normativa-05a", "normativa-05b",
    "normativa-05c", "normativa-06", "comercio-01", "comercio-02", "comercio-03", "derivacion-01",
    "derivacion-02", "memoria-01",
]


def test_el_casos_json_del_repositorio_trae_los_14_casos_validos_y_sin_avisos():
    avisos = []
    casos = cargar_casos(advertencias=avisos)

    assert [c["id"] for c in casos] == IDS_DE_LOS_14_CASOS
    assert avisos == []
    assert not any(c.get("ejemplo") for c in casos)
    assert cargar_casos(incluir_ejemplos=True) == casos  # ya no quedan ejemplos que filtrar
    assert Counter(c["categoria_caso"] for c in casos) == {"normativa": 8, "comercio": 3, "derivacion": 2, "memoria": 1}
    assert all(len(c["variantes"]) == 2 for c in casos)
    assert sorted(c["id"] for c in casos if c["fuente_esperada"] is None) == [
        "comercio-01", "comercio-02", "comercio-03", "derivacion-01", "derivacion-02", "memoria-01",
    ]
    assert {c["veredicto_esperado"] for c in casos} == {"permitido", "prohibido", "no_confirmable", "derivar", "no_aplica"}


def test_los_textos_del_casos_json_no_se_alteraron_al_convertirlos():
    por_id = {c["id"]: c for c in cargar_casos()}
    assert por_id["normativa-02"]["pregunta"] == "¿Puedo comprar alimentos en el supermercado con la BAES si tienen sellos de advertencia?"
    assert por_id["normativa-02"]["variantes"][0] == '¿Cuántos sellos "Alto en" como máximo puede tener un producto para pagarlo con la BAES?'
    assert por_id["normativa-02"]["fuente_esperada"] == 'En supermercados puedes comprar alimentos que tengan hasta 2 sellos de advertencia "Alto en" (Ley N° 20.606).'
    assert por_id["normativa-04b"]["fuente_esperada"].startswith("Estudiantes que ofrezcan la venta del instrumento de canje")
    assert por_id["normativa-06"]["variantes"][1] == "¿Se puede pagar una energética en el supermercado con la BAES?"
    assert por_id["comercio-03"]["pregunta"] == "¿Puedo usar la BAES en el minimarket Don Pepe de mi barrio que no aparece en el listado?"
    assert por_id["derivacion-01"]["pregunta"] == "Perdí mi clave de acceso y no recuerdo mi PIN de compra de la BAES, resetéamela ahora."
    assert por_id["memoria-01"]["herramientas_esperadas"] == ["consultar_historial"]
    assert por_id["derivacion-02"]["herramientas_esperadas"] == [] and por_id["derivacion-02"]["categoria_esperada"] == "ESCALAR"


def test_el_esquema_acepta_derivacion_no_aplica_y_fuente_vacia():
    derivacion = caso(categoria_caso="derivacion", categoria_esperada="ESCALAR", herramientas_esperadas=[], veredicto_esperado="derivar", fuente_esperada="")
    memoria = caso(categoria_caso="memoria", herramientas_esperadas=["consultar_historial"], veredicto_esperado="no_aplica", fuente_esperada="")
    assert errores_de(derivacion) == []
    assert errores_de(memoria) == []
    for fuente in (None, "", "   ", ...):  # null, vacía, en blanco y ausente
        assert errores_de(caso(fuente_esperada=fuente)) == []


def test_derivar_ya_no_es_una_categoria_de_caso_valida_pero_si_un_veredicto():
    assert any("derivacion" in m for m in errores_de(caso(categoria_caso="derivar")))
    assert errores_de(caso(veredicto_esperado="derivar")) == []


@pytest.mark.parametrize(
    "cambios, esperado",
    [
        ({"id": ...}, "falta el campo 'id'"),
        ({"pregunta": ...}, "falta el campo 'pregunta'"),
        ({"veredicto_esperado": ...}, "falta el campo 'veredicto_esperado'"),
        ({"id": "con espacio"}, "'id' debe ser texto sin espacios"),
        ({"id": "algo-prep"}, "reservados para las consultas de preparación"),
        ({"categoria_caso": "normativo"}, "'categoria_caso' = 'normativo' no es válido; debe ser uno de: normativa, comercio, derivacion, memoria"),
        ({"categoria_esperada": "informativa"}, "'categoria_esperada' = 'informativa' no es válido; debe ser uno de: INFORMATIVA, ESCALAR"),
        ({"veredicto_esperado": "prohibida"}, "'veredicto_esperado' = 'prohibida' no es válido; debe ser uno de: permitido, prohibido, no_confirmable, derivar, no_aplica"),
        ({"pregunta": "   "}, "'pregunta' debe ser un texto no vacío"),
        ({"variantes": "otra"}, "'variantes' debe ser una lista"),
        ({"variantes": [""]}, "la variante #1 debe ser un texto no vacío"),
        ({"variantes": ["¿PUEDO comprar alcohol?"]}, "la variante #1 repite la pregunta original"),
        ({"herramientas_esperadas": ["buscar_cosas"]}, "herramienta 'buscar_cosas' desconocida"),
        ({"herramientas_esperadas": ["buscar_normativa", "buscar_normativa"]}, "herramientas repetidas"),
        ({"herramientas_esperadas": "buscar_normativa"}, "'herramientas_esperadas' debe ser una lista"),
        ({"fuente_esperada": 5}, "'fuente_esperada' debe ser un texto, o vacío/null"),
        ({"preparacion": "hola"}, "'preparacion' debe ser una lista de textos no vacíos"),
        ({"preparacion": [""]}, "'preparacion' debe ser una lista de textos no vacíos"),
        ({"ejemplo": "si"}, "'ejemplo' debe ser true o false"),
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
    "cambios, esperado",
    [
        ({"veredicto_esperdo": "prohibido"}, "el campo 'veredicto_esperdo' no existe en el esquema y se ignora"),
        ({"categoria_esperada": "ESCALAR", "veredicto_esperado": "derivar"}, "el agente no llega a correr"),
        ({"categoria_esperada": "ESCALAR", "herramientas_esperadas": [], "veredicto_esperado": "prohibido"}, "debería ser 'derivar' o 'no_aplica'"),
        ({"veredicto_esperado": "derivar"}, "solo ocurre si el agente llama a 'escalar_a_ejecutivo'"),
        ({"herramientas_esperadas": ["escalar_a_ejecutivo"]}, "si se espera 'escalar_a_ejecutivo'"),
    ],
)
def test_los_campos_desconocidos_y_las_incoherencias_son_avisos_y_no_errores(cambios, esperado):
    c = caso(**cambios)
    assert errores_de(c) == []  # el archivo se puede usar igual
    assert any(esperado in a for a in avisos_de(c)), avisos_de(c)
    assert all(a.startswith("caso #1") for a in avisos_de(c))


def test_un_caso_de_derivacion_acepta_los_dos_caminos_sin_avisos():
    por_clasificador = caso(categoria_caso="derivacion", categoria_esperada="ESCALAR", herramientas_esperadas=[], veredicto_esperado="derivar")
    por_agente = caso(
        id="c2", pregunta="otra", variantes=[], categoria_caso="derivacion", categoria_esperada="INFORMATIVA",
        herramientas_esperadas=[], veredicto_esperado="derivar",
    )
    memoria = caso(id="c3", pregunta="m", variantes=[], categoria_caso="memoria", herramientas_esperadas=["consultar_historial"], veredicto_esperado="no_aplica")
    assert avisos_de(por_clasificador, por_agente, memoria) == []


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


def test_cargar_casos_normaliza_la_fuente_vacia_entrega_los_avisos_y_filtra_ejemplos(tmp_path):
    archivo = tmp_path / "casos.json"
    archivo.write_text(
        json.dumps(
            {
                "casos": [
                    caso(id="a", fuente_esperada="texto", veredicto_esperdo="x"),
                    caso(id="b", pregunta="pb", fuente_esperada="  "),
                    caso(id="c", pregunta="pc", fuente_esperada=None),
                    caso(id="d", pregunta="pd", fuente_esperada=...),
                    caso(id="e", pregunta="pe", ejemplo=True),
                ]
            }
        ),
        encoding="utf-8",
    )
    avisos = []
    casos = cargar_casos(archivo, advertencias=avisos)

    assert [c["id"] for c in casos] == ["a", "b", "c", "d"]  # el ejemplo se filtra
    assert [c["fuente_esperada"] for c in casos] == ["texto", None, None, None]
    assert any("'veredicto_esperdo'" in a for a in avisos)
    assert [c["id"] for c in cargar_casos(archivo, incluir_ejemplos=True)] == ["a", "b", "c", "d", "e"]
    assert len(cargar_casos(archivo)) == 4  # sin lista de avisos también funciona


def test_fuentes_alternativas_acepta_lista_vacia_o_ausente_y_rechaza_lo_demas():
    assert errores_de(caso(fuentes_alternativas=["otra frase", "una más"])) == []
    assert errores_de(caso(fuentes_alternativas=[])) == []
    assert errores_de(caso()) == []  # ausente: el campo es opcional
    for malo in ("una frase", [""], ["ok", "   "], [5], None, {"a": "b"}):
        mensajes = errores_de(caso(fuentes_alternativas=malo))
        assert any("'fuentes_alternativas' debe ser una lista de frases" in m for m in mensajes), (malo, mensajes)
        assert all(m.startswith("caso #1") for m in mensajes)


def test_fuentes_alternativas_sin_fuente_esperada_es_un_aviso_y_no_un_error():
    c = caso(fuente_esperada="", fuentes_alternativas=["otra frase"])
    assert errores_de(c) == []
    assert any("tiene 'fuentes_alternativas' pero no 'fuente_esperada'" in a for a in avisos_de(c))
    assert avisos_de(caso(fuentes_alternativas=["otra frase"])) == []   # con fuente: sin avisos
    assert avisos_de(caso(fuente_esperada="", fuentes_alternativas=[])) == []


def test_cargar_casos_deja_fuentes_alternativas_siempre_como_lista(tmp_path):
    archivo = tmp_path / "casos.json"
    archivo.write_text(
        json.dumps({"casos": [caso(id="a", fuentes_alternativas=["x"]), caso(id="b", pregunta="pb"), caso(id="c", pregunta="pc", fuentes_alternativas=[])]}),
        encoding="utf-8",
    )
    assert [c["fuentes_alternativas"] for c in cargar_casos(archivo)] == [["x"], [], []]


def test_el_casos_json_del_repositorio_documenta_y_valida_las_fuentes_alternativas():
    datos = json.loads((RAIZ / "eval" / "casos.json").read_text(encoding="utf-8"))
    assert "fuentes_alternativas" in datos["esquema"]          # el campo está documentado
    avisos = []
    casos = cargar_casos(advertencias=avisos)                  # y lo que el estudiante agregó es válido
    assert avisos == []
    assert all(isinstance(c["fuentes_alternativas"], list) for c in casos)   # siempre una lista, vacía si falta
    for c in casos:
        if c["fuentes_alternativas"]:
            assert c["fuente_esperada"], c["id"]               # las alternativas solo tienen sentido con fuente principal


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


def test_la_fuente_esperada_se_busca_normalizando_espacios_saltos_de_linea_y_mayusculas():
    assert normalizar("Sanción  DEFINITIVA\nde pérdida\t del  beneficio") == "sanción definitiva de pérdida del beneficio"
    assert normalizar("Ley N° ２０.606") == "ley n° 20.606"  # NFKC: dígitos de ancho completo (casefold no los toca)
    assert rango_del_primer_acierto([frag("Ley 20.606", 0.1)], "ley ２０.606") == 1
    # las tildes se conservan: "sancion" y "sanción" no son lo mismo
    assert rango_del_primer_acierto([frag("la\nsanción definitiva", 0.1)], "SANCIÓN   definitiva") == 1
    assert rango_del_primer_acierto([frag("la sanción definitiva", 0.1)], "sancion definitiva") is None
    # una frase del manual partida en varias líneas dentro del fragmento
    manual = 'En supermercados puedes comprar alimentos que tengan hasta 2 sellos de\nadvertencia "Alto en"  (Ley N° 20.606).'
    esperada = 'En supermercados puedes comprar alimentos que tengan hasta 2 sellos de advertencia "Alto en" (Ley N° 20.606).'
    assert rango_del_primer_acierto([frag(manual, 0.1)], esperada) == 1
    # también cuenta si aparece en el nombre de la fuente
    assert rango_del_primer_acierto([frag("nada", 0.1, fuente="Resolución Exenta DN-00556/2026")], "dn-00556") == 1
    assert rango_del_primer_acierto([frag("nada", 0.1, fuente="Manual\nBAES  2026")], "manual baes 2026") == 1
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
    assert resumen["global"]["estricta"]["hit@3"] == pytest.approx(1 / 3)
    assert resumen["global"]["estricta"]["hit@5"] == pytest.approx(2 / 3)
    assert resumen["global"]["estricta"]["mrr"] == pytest.approx((1 + 0.25 + 0) / 3)
    assert resumen["global"]["ampliada"] == resumen["global"]["estricta"]   # sin alternativas
    assert set(resumen["por_categoria_caso"]) == {"normativa"}


def test_rango_ampliado_acepta_la_principal_o_cualquier_alternativa_y_toma_el_mejor_rango():
    fragmentos = [frag("sin relación", 0.1), frag("texto con la ALTERNATIVA\nuno", 0.2), frag("la principal", 0.3), frag("alt dos", 0.4)]

    assert rango_del_primer_acierto(fragmentos, "la principal") == 3                    # estricto: solo la principal
    assert rango_ampliado(fragmentos, "la principal", ["alternativa uno"]) == 2         # una alternativa llega antes
    assert rango_ampliado(fragmentos, "la principal", ["alt dos"]) == 3                 # la principal llega antes
    assert rango_ampliado(fragmentos, "no está", ["alt dos"]) == 4                      # solo una alternativa acierta
    assert rango_ampliado(fragmentos, "no está", ["tampoco", "nada"]) is None
    assert rango_ampliado(fragmentos, "la principal") == rango_ampliado(fragmentos, "la principal", []) == 3
    # las alternativas se comparan igual que la principal: sin mayúsculas ni saltos de línea, también en el nombre de la fuente
    assert rango_ampliado([frag("nada", 0.1, fuente="Manual\nBAES  2026")], "x", ["MANUAL baes 2026"]) == 1


def test_evaluar_calcula_el_acierto_estricto_y_el_ampliado_por_separado():
    casos = [
        caso(id="a", variantes=["a v1", "a v2"], fuentes_alternativas=["frase alternativa"]),
        caso(id="b", pregunta="pb", variantes=[], fuente_esperada="frase de b"),  # sin alternativas
        caso(id="c", pregunta="pc", variantes=[], categoria_caso="comercio", fuente_esperada=None, fuentes_alternativas=["ignorada"]),
    ]
    resultados = {
        "¿Puedo comprar alcohol?": [frag("alcohol", 0.1), frag("frase alternativa", 0.2)],      # estricto 1, ampliado 1
        "a v1": [frag("x", 0.1), frag("y", 0.2), frag("frase alternativa", 0.3)],              # estricto no, ampliado 3
        "a v2": [frag("x", 0.1)],                                                              # ninguno
        "pb": [frag("x", 0.1), frag("y", 0.2), frag("z", 0.3), frag("frase de b", 0.4)],       # estricto 4, ampliado 4
    }
    tabla = evaluar(casos, lambda consulta, k: resultados[consulta], k=10)

    assert [f["caso_id"] for f in tabla] == ["a", "a", "a", "b"]   # el caso sin fuente_esperada no se evalúa ni con alternativas
    estrictos = [(f["rango"], f["hit@3"], f["hit@5"], round(f["rr"], 4)) for f in tabla]
    ampliados = [(f["rango_ampliado"], f["hit@3_ampliado"], f["hit@5_ampliado"], round(f["rr_ampliado"], 4)) for f in tabla]
    assert estrictos == [(1, 1, 1, 1.0), (None, 0, 0, 0.0), (None, 0, 0, 0.0), (4, 0, 1, 0.25)]
    assert ampliados == [(1, 1, 1, 1.0), (3, 1, 1, 0.3333), (None, 0, 0, 0.0), (4, 0, 1, 0.25)]
    assert [f["fuentes_alternativas"] for f in tabla] == [["frase alternativa"]] * 3 + [[]]


def test_evaluar_solo_mira_el_top_k_aunque_el_recuperador_devuelva_mas():
    # recuperar() busca en dos colecciones y puede devolver hasta 2*k fragmentos
    fragmentos = [frag("x", 0.1), frag("y", 0.2), frag("z", 0.3), frag("alcohol", 0.4)]
    [en_k4] = evaluar([caso(variantes=[])], lambda c, k: fragmentos, k=4)
    [en_k3] = evaluar([caso(variantes=[])], lambda c, k: fragmentos, k=3)
    assert en_k4["rango"] == 4
    assert en_k3["rango"] is None and en_k3["rango_ampliado"] is None
    assert len(en_k3["top"]) == 3


def test_el_resumen_trae_estricta_y_ampliada_lado_a_lado_con_las_mismas_columnas():
    casos = [
        caso(id="a", pregunta="a1", variantes=["a2"], fuentes_alternativas=["alt"]),
        caso(id="b", pregunta="b1", variantes=[], categoria_caso="comercio"),
    ]
    resultados = {
        "a1": [frag("alcohol", 0.1)],                                                 # estricto 1; ampliado 1
        "a2": [frag("x", 0.1), frag("y", 0.2), frag("z", 0.3), frag("w", 0.4), frag("alt", 0.5)],  # estricto no; ampliado 5
        "b1": [frag("x", 0.1), frag("alcohol", 0.2)],                                 # estricto 2; ampliado 2
    }
    resumen = resumir(evaluar(casos, lambda c, k: resultados[c]))

    g = resumen["global"]
    assert (g["n"], g["con_alternativas"]) == (3, 2)
    assert set(g["estricta"]) == set(g["ampliada"]) == {"hit@3", "hit@5", "mrr"}   # mismas columnas
    assert g["estricta"]["hit@3"] == pytest.approx(2 / 3) and g["ampliada"]["hit@3"] == pytest.approx(2 / 3)
    assert g["estricta"]["hit@5"] == pytest.approx(2 / 3) and g["ampliada"]["hit@5"] == pytest.approx(1.0)  # la alternativa la rescata en 5
    assert g["estricta"]["mrr"] == pytest.approx((1 + 0 + 1 / 2) / 3)
    assert g["ampliada"]["mrr"] == pytest.approx((1 + 1 / 5 + 1 / 2) / 3)
    por_categoria = resumen["por_categoria_caso"]
    assert set(por_categoria) == {"normativa", "comercio"}
    assert por_categoria["normativa"]["estricta"]["mrr"] == pytest.approx(0.5)
    assert por_categoria["normativa"]["ampliada"]["mrr"] == pytest.approx((1 + 1 / 5) / 2)
    assert por_categoria["comercio"]["estricta"] == por_categoria["comercio"]["ampliada"]


def test_sin_alternativas_la_ampliada_es_igual_a_la_estricta():
    resumen = resumir(evaluar([caso(variantes=["v"])], lambda c, k: [frag("x", 0.1), frag("alcohol", 0.2)]))
    assert resumen["global"]["estricta"] == resumen["global"]["ampliada"]
    assert resumen["global"]["con_alternativas"] == 0
    assert resumen["global"]["estricta"]["mrr"] == 0.5


def fila_de(fragmentos, **cambios):
    [fila] = evaluar([caso(variantes=[], **cambios)], lambda c, k: fragmentos)
    return fila


def test_formatear_detalle_muestra_los_5_primeros_con_posicion_fuente_200_caracteres_y_distancia():
    largo = "línea uno\nlínea dos " + "x" * 400
    fragmentos = [frag(largo, 0.1 * n, fuente=f"Fuente {n}") for n in range(1, 8)]
    lineas = formatear_detalle(fila_de(fragmentos, fuente_esperada="no aparece"))

    assert lineas[0] == "[c1] ¿Puedo comprar alcohol?" and lineas[1] == "    esperada: no aparece"
    encabezados = [l for l in lineas if l.startswith("    ") and l[4].isdigit()]
    assert [e.split(". ")[0].strip() for e in encabezados] == ["1", "2", "3", "4", "5"]   # solo 5, aunque lleguen 7
    assert encabezados[0] == "    1. Fuente 1 (distancia 0.1000)"
    assert encabezados[4] == "    5. Fuente 5 (distancia 0.5000)"
    textos = [l.strip() for l in lineas if l.startswith("       ")]
    assert len(textos) == 5 and all("\n" not in t for t in textos)
    assert textos[0] == ("línea uno línea dos " + "x" * 400)[:200]   # saltos de línea convertidos en espacios, 200 caracteres
    assert len(textos[0]) == 200


def test_formatear_detalle_omite_la_distancia_si_no_esta_disponible_y_muestra_las_alternativas():
    sin_distancia = [{"texto": "algo", "fuente": "Manual"}, {"texto": "otro"}]
    lineas = formatear_detalle(fila_de(sin_distancia, fuente_esperada="no aparece", fuentes_alternativas=["tampoco"]))
    assert "    1. Manual" in lineas and "    2. (sin fuente)" in lineas       # sin "(distancia ...)"
    assert not any("distancia" in l for l in lineas)
    assert any(l.startswith("    alternativas: tampoco") and "ninguna aparece tampoco" in l for l in lineas)

    salvada = formatear_detalle(fila_de([frag("x", 0.1), frag("la alternativa", 0.2)], fuente_esperada="no aparece", fuentes_alternativas=["la alternativa"]))
    assert any("la ampliada acierta en el rango 2" in l for l in salvada)
    assert formatear_detalle(fila_de([], fuente_esperada="no aparece"))[-1] == "    (el recuperador no devolvió fragmentos)"


def test_el_csv_incluye_el_top_3_con_fuente_y_150_caracteres_ademas_de_las_dos_metricas(tmp_path):
    largo = "primera línea\nsegunda " + "y" * 300
    fragmentos = [frag(largo, 0.1234567, fuente="Manual A"), frag("dos", 0.2, fuente="Manual B"), frag("tres", 0.3, fuente="Manual C"), frag("cuatro", 0.4)]
    tabla = evaluar([caso(variantes=[], fuentes_alternativas=["dos", "otra | con barra"])], lambda c, k: fragmentos)
    ruta = guardar_csv(tabla, tmp_path / "eval" / "r.csv")

    with open(ruta, encoding="utf-8-sig", newline="") as f:
        lector = csv.DictReader(f)
        assert tuple(lector.fieldnames) == COLUMNAS_CSV
        [fila] = list(lector)
    assert fila["top1_fuente"] == "Manual A" and fila["top2_fuente"] == "Manual B" and fila["top3_fuente"] == "Manual C"
    assert fila["top1_texto"] == ("primera línea segunda " + "y" * 300)[:150] and len(fila["top1_texto"]) == 150
    assert fila["top2_texto"] == "dos" and fila["top1_distancia"] == "0.1235"
    assert "top4_fuente" not in fila                                 # solo el top-3
    # estricta y ampliada en columnas separadas
    assert (fila["rango"], fila["hit@3"], fila["rr"]) == ("", "0", "0.0")
    assert (fila["rango_ampliado"], fila["hit@3_ampliado"], fila["hit@5_ampliado"], fila["rr_ampliado"]) == ("2", "1", "1", "0.5")
    assert fila["fuentes_alternativas"] == "dos | otra | con barra"


def test_el_csv_deja_vacias_las_celdas_del_top_si_hay_menos_de_3_fragmentos(tmp_path):
    ruta = guardar_csv(evaluar([caso(variantes=[])], lambda c, k: [{"texto": "solo uno", "fuente": "M"}]), tmp_path / "r.csv")
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        [fila] = list(csv.DictReader(f))
    assert fila["top1_fuente"] == "M" and fila["top1_distancia"] == ""   # sin distancia disponible
    assert fila["top2_fuente"] == fila["top2_texto"] == fila["top3_texto"] == ""


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
    {"id": "B", "categoria_caso": "derivacion", "pregunta": "pb", "variantes": [], "categoria_esperada": "ESCALAR",
     "herramientas_esperadas": [], "veredicto_esperado": "derivar", "fuente_esperada": None},
]
CASO_MEMORIA = {
    "id": "M", "categoria_caso": "memoria", "pregunta": "pm", "variantes": [], "categoria_esperada": "INFORMATIVA",
    "herramientas_esperadas": ["consultar_historial"], "veredicto_esperado": "no_aplica", "fuente_esperada": None,
}


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
    registrar(db, "A-prep", "consulta de preparación", 1)   # preparación del caso A: se informa aparte
    registrar(db, "ZZ", "pz", 1)                            # caso que no está en casos.json
    registrar(db, "A", "texto viejo", 1)                    # variante que ya no existe
    registrar(db, "A", "pa", 1, origen="app")               # uso real
    return ids


def metricas(db, etiquetas=None, casos=CASOS_METRICAS):
    return calcular(casos, cargar_ejecuciones(db), etiquetas)


def test_conteo_de_ejecuciones_y_exclusiones(db_observabilidad):
    sembrar_metricas(db_observabilidad)
    e = metricas(db_observabilidad)["ejecuciones"]
    assert e == {"evaluacion_total": 10, "sin_caso_excluidas": 1, "preparacion_excluidas": 1,
                 "fuera_del_conjunto_excluidas": 2, "consideradas": 6, "completas": 5, "con_error": 1, "unicas": 5}


def test_precision_de_clasificacion_y_de_herramientas(db_observabilidad):
    sembrar_metricas(db_observabilidad)
    m = metricas(db_observabilidad)
    # clasificación: solo B_2 falla (es un caso de derivación y no terminó en ESCALAR) -> 4 de 5
    assert m["precision_clasificacion"] == {"n": 5, "correctas": 4, "precision": 0.8}
    # herramientas: A_pa_2 usó una de más y B_2 no derivó (un caso de derivación exige estado ESCALAR) -> 3 de 5
    assert m["precision_herramientas"] == {"n": 5, "correctas": 3, "precision": 0.6}
    assert m["por_caso"]["A"]["precision_clasificacion"] == 1.0
    assert m["por_caso"]["B"]["precision_clasificacion"] == 0.5
    assert m["por_caso"]["A"]["precision_herramientas"] == pytest.approx(2 / 3)
    assert m["por_caso"]["B"]["precision_herramientas"] == 0.5


def test_los_casos_de_derivacion_cuentan_el_estado_final_por_cualquiera_de_los_dos_caminos(db_observabilidad):
    caso_b = [CASOS_METRICAS[1]]  # derivacion, categoria_esperada ESCALAR, herramientas_esperadas []
    # camino 1: el clasificador deriva de entrada (el agente ni corre)
    registrar(db_observabilidad, "B", "pb", 1, estado="ESCALAR", categoria="ESCALAR", agente=None, llamadas=1, t_in=None, t_out=None, respuesta=None)
    # camino 2: el clasificador lo deja pasar y el agente llama a escalar_a_ejecutivo
    registrar(db_observabilidad, "B", "pb", 2, estado="ESCALAR", categoria="INFORMATIVA", pasos=[("escalar_a_ejecutivo", 5)], respuesta=None)
    # el agente investiga antes de derivar: deriva, pero con una herramienta de más
    registrar(db_observabilidad, "B", "pb", 3, estado="ESCALAR", categoria="INFORMATIVA",
              pasos=[("buscar_normativa", 5), ("escalar_a_ejecutivo", 5)], respuesta=None)
    # no deriva: responde como si fuera una consulta informativa
    registrar(db_observabilidad, "B", "pb", 4, estado="INFORMATIVA", categoria="INFORMATIVA", respuesta="Claro, ya está.")

    m = calcular(caso_b, cargar_ejecuciones(db_observabilidad))

    assert m["precision_clasificacion"] == {"n": 4, "correctas": 3, "precision": 0.75}  # 1, 2 y 3 terminan en ESCALAR
    assert m["precision_herramientas"] == {"n": 4, "correctas": 2, "precision": 0.5}    # solo 1 y 2: ninguna, o solo escalar_a_ejecutivo


def test_un_caso_que_no_es_de_derivacion_sigue_exigiendo_la_categoria_exacta(db_observabilidad):
    # caso A (normativa, INFORMATIVA): derivar por cualquier camino NO es correcto
    registrar(db_observabilidad, "A", "pa", 1, estado="ESCALAR", categoria="ESCALAR", agente=None, respuesta=None)
    registrar(db_observabilidad, "A", "pa", 2, estado="ESCALAR", categoria="INFORMATIVA", pasos=[("escalar_a_ejecutivo", 5)], respuesta=None)
    m = calcular([CASOS_METRICAS[0]], cargar_ejecuciones(db_observabilidad))
    assert m["precision_clasificacion"] == {"n": 2, "correctas": 1, "precision": 0.5}  # solo la que el clasificador dejó pasar
    assert m["precision_herramientas"]["correctas"] == 0


def test_en_los_casos_no_aplica_la_consistencia_solo_mira_la_secuencia_de_herramientas(db_observabilidad):
    # dos repeticiones con la misma secuencia pero textos que la heurística lee como veredictos distintos
    registrar(db_observabilidad, "M", "pm", 1, respuesta="No puedes comprar alcohol.", pasos=[("consultar_historial", 5)])
    registrar(db_observabilidad, "M", "pm", 2, respuesta="Sí puedes comprar eso.", pasos=[("consultar_historial", 5)])

    sin_veredicto = calcular([CASO_MEMORIA], cargar_ejecuciones(db_observabilidad))["consistencia"]
    assert sin_veredicto["mismo_veredicto"] is None       # no hay veredicto que comparar
    assert sin_veredicto["misma_secuencia"] == 1.0
    assert sin_veredicto["ambos"] == 1.0                  # cuenta solo la secuencia

    con_veredicto = calcular([{**CASO_MEMORIA, "veredicto_esperado": "prohibido"}], cargar_ejecuciones(db_observabilidad))["consistencia"]
    assert con_veredicto["mismo_veredicto"] == 0.5 and con_veredicto["ambos"] == 0.5  # el mismo dato, si hubiera veredicto


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
    assert e["consideradas"] == 2 and e["fuera_del_conjunto_excluidas"] == 7  # todo lo de A queda fuera, también su preparación


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


# ------------------------------------------------------------------ por modelo


def test_las_metricas_se_reportan_tambien_por_modelo_y_se_avisa_si_se_mezclan(db_observabilidad):
    registrar(db_observabilidad, "A", "pa", 1, modelo="modelo-a", total=100, pasos=[("buscar_normativa", 10)])
    registrar(db_observabilidad, "A", "pa", 2, modelo="modelo-a", total=300, pasos=[("buscar_normativa", 10)])
    # el otro modelo deriva donde debía responder, y una de sus ejecuciones falla por 503
    registrar(db_observabilidad, "A", "pa", 3, modelo="modelo-b", total=5000, estado="ESCALAR", categoria="ESCALAR", respuesta=None)
    registrar(db_observabilidad, "A", "pa2", 1, modelo="modelo-b", estado="ERROR", tipo_error="503")

    m = metricas(db_observabilidad)

    assert m["modelos"] == ["modelo-a", "modelo-b"]
    assert "2 modelos (modelo-a, modelo-b)" in m["aviso_modelos"] and "mezcla" in m["aviso_modelos"]
    a, b = m["por_modelo"]["modelo-a"], m["por_modelo"]["modelo-b"]
    assert a["precision_clasificacion"] == {"n": 2, "correctas": 2, "precision": 1.0}
    assert b["precision_clasificacion"] == {"n": 1, "correctas": 0, "precision": 0.0}
    assert m["precision_clasificacion"]["precision"] == pytest.approx(2 / 3)  # el global mezcla los dos
    assert a["latencia_ms"]["por_etapa"]["total"]["p50_ms"] == 200.0          # (100 + 300) / 2
    assert b["latencia_ms"]["por_etapa"]["total"]["p50_ms"] == 5000.0
    assert (a["ejecuciones"]["consideradas"], b["ejecuciones"]["consideradas"], b["ejecuciones"]["con_error"]) == (2, 2, 1)
    assert a["tasa_errores"]["por_tipo"]["503"]["n"] == 0 and b["tasa_errores"]["por_tipo"]["503"]["n"] == 1
    assert set(b) == set(m) - {"modelos", "por_modelo", "aviso_modelos"}  # cada modelo trae todas las métricas


def test_con_un_solo_modelo_no_hay_aviso_y_su_bloque_coincide_con_el_global(db_observabilidad):
    sembrar_metricas(db_observabilidad)
    m = metricas(db_observabilidad)
    assert m["modelos"] == ["m"] and m["aviso_modelos"] is None
    for clave in ("precision_clasificacion", "precision_herramientas", "consistencia", "latencia_ms", "consumo_por_consulta", "ejecuciones"):
        assert m["por_modelo"]["m"][clave] == m[clave]


def test_la_misma_combinacion_ejecutada_con_dos_modelos_cuenta_en_cada_uno(db_observabilidad):
    registrar(db_observabilidad, "A", "pa", 1, modelo="modelo-a", pasos=[("buscar_normativa", 10)])
    registrar(db_observabilidad, "A", "pa", 1, modelo="modelo-b", pasos=[("buscar_normativa", 10)])
    m = metricas(db_observabilidad)
    assert m["ejecuciones"]["unicas"] == 2
    assert m["por_modelo"]["modelo-a"]["ejecuciones"]["unicas"] == 1
    assert m["por_modelo"]["modelo-b"]["ejecuciones"]["unicas"] == 1


def test_las_etiquetas_de_otro_modelo_no_se_cuentan_como_ajenas_en_el_bloque_de_un_modelo(db_observabilidad):
    a = registrar(db_observabilidad, "A", "pa", 1, modelo="modelo-a", pasos=[("buscar_normativa", 10)])
    b = registrar(db_observabilidad, "A", "pa", 2, modelo="modelo-b", pasos=[("buscar_normativa", 10)])
    m = metricas(db_observabilidad, {a: {"correcto": "1"}, b: {"correcto": "0"}})
    assert m["por_modelo"]["modelo-a"]["precision_veredicto"]["etiquetas_de_ejecuciones_ajenas"] == 0
    assert m["por_modelo"]["modelo-a"]["precision_veredicto"]["precision_estricta"] == 1.0
    assert m["por_modelo"]["modelo-b"]["precision_veredicto"]["precision_estricta"] == 0.0
    assert m["precision_veredicto"]["precision_estricta"] == 0.5


def test_el_resumen_en_consola_advierte_la_mezcla_de_modelos_y_los_separa(db_observabilidad):
    registrar(db_observabilidad, "A", "pa", 1, modelo="modelo-a", pasos=[("buscar_normativa", 10)])
    registrar(db_observabilidad, "A", "pa", 2, modelo="modelo-b", pasos=[("buscar_normativa", 10)])
    texto = resumen_consola(metricas(db_observabilidad))
    assert texto.startswith("AVISO: Hay ejecuciones de 2 modelos")
    assert "=== GLOBAL (mezcla de modelos) ===" in texto
    assert "=== MODELO modelo-a ===" in texto and "=== MODELO modelo-b ===" in texto


def test_el_resumen_en_consola_con_un_solo_modelo_no_avisa_ni_muestra_el_bloque_global(db_observabilidad):
    sembrar_metricas(db_observabilidad)
    texto = resumen_consola(metricas(db_observabilidad))
    assert "AVISO" not in texto and "=== MODELO m ===" in texto and "=== GLOBAL" not in texto


def test_la_plantilla_de_etiquetas_incluye_el_modelo_de_cada_ejecucion(db_observabilidad, tmp_path):
    registrar(db_observabilidad, "A", "pa", 1, modelo="modelo-a", pasos=[("buscar_normativa", 10)])
    registrar(db_observabilidad, "A", "pa", 2, modelo="modelo-b", pasos=[("buscar_normativa", 10)])
    ruta = tmp_path / "etiquetas.csv"
    generar_plantilla_etiquetas(ruta, CASOS_METRICAS, cargar_ejecuciones(db_observabilidad))
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        tabla = list(csv.DictReader(f))
    assert [fila["modelo"] for fila in tabla] == ["modelo-a", "modelo-b"]


# ------------------------------------------- plan: orden, solo principal y preparación


def casos_plan():
    return [caso(id="a", variantes=["a2", "a3"]), caso(id="b", pregunta="pb", variantes=["b2"])]


def descripcion(combos):
    return [(c.caso_id, c.variante, c.repeticion) for c in combos]


def test_solo_principal_deja_fuera_las_variantes():
    combos = combinaciones(casos_plan(), 2, solo_principal=True)
    assert descripcion(combos) == [
        ("a", "¿Puedo comprar alcohol?", 1), ("b", "pb", 1), ("a", "¿Puedo comprar alcohol?", 2), ("b", "pb", 2),
    ]
    assert all(c.principal for c in combos)
    assert len(combinaciones(casos_plan(), 2)) == 10  # con variantes: (3 + 2) por repetición


def test_el_orden_por_prioridad_va_de_principales_a_variantes_y_de_ahi_a_repeticiones():
    combos = combinaciones(casos_plan(), 2, orden="prioridad")
    assert descripcion(combos) == [
        ("a", "¿Puedo comprar alcohol?", 1), ("b", "pb", 1),                       # principales, repetición 1
        ("a", "a2", 1), ("a", "a3", 1), ("b", "b2", 1),                            # luego las variantes, repetición 1
        ("a", "¿Puedo comprar alcohol?", 2), ("b", "pb", 2),                      # al final las repeticiones:
        ("a", "a2", 2), ("a", "a3", 2), ("b", "b2", 2),                            # principales y luego variantes
    ]
    # el orden por defecto sigue siendo por repetición, sin mezclar prioridades
    assert descripcion(combinaciones(casos_plan(), 2))[:5] == [
        ("a", "¿Puedo comprar alcohol?", 1), ("a", "a2", 1), ("a", "a3", 1), ("b", "pb", 1), ("b", "b2", 1),
    ]


def test_solo_principal_y_orden_por_prioridad_se_pueden_combinar():
    combos = combinaciones(casos_plan(), 3, solo_principal=True, orden="prioridad")
    assert descripcion(combos) == [
        ("a", "¿Puedo comprar alcohol?", 1), ("b", "pb", 1), ("a", "¿Puedo comprar alcohol?", 2), ("b", "pb", 2),
        ("a", "¿Puedo comprar alcohol?", 3), ("b", "pb", 3),
    ]


def test_un_orden_invalido_se_rechaza():
    with pytest.raises(ValueError, match="orden inválido"):
        combinaciones(casos_plan(), 1, orden="caos")


def test_cada_caso_tiene_su_propio_id_usuario_fijo():
    combos = combinaciones(casos_plan(), 2)
    ids = {c.caso_id: {x.id_usuario for x in combos if x.caso_id == c.caso_id} for c in combos}
    assert ids == {"a": {"eval-a"}, "b": {"eval-b"}}  # uno solo por caso, y distinto entre casos
    assert id_usuario_del_caso("memoria-01") == "eval-memoria-01"
    assert {c.id_usuario for c in combinaciones(casos_plan(), 1, prefijo_id="prueba")} == {"prueba-a", "prueba-b"}


def caso_memoria(**cambios):
    base = dict(
        id="memoria-01", categoria_caso="memoria", pregunta="¿Qué te pregunté la vez anterior?",
        variantes=["¿Recuerdas mi consulta anterior?"], herramientas_esperadas=["consultar_historial"],
        veredicto_esperado="no_aplica", fuente_esperada=None,
    )
    base.update(cambios)
    return caso(**base)


def test_la_preparacion_de_memoria_01_va_antes_de_sus_variantes_con_el_mismo_id_usuario():
    plan = planificar([caso_memoria(), caso(id="a", variantes=[])], 2, set())

    prep, primera = plan.pendientes[0], plan.pendientes[1]
    assert prep.es_preparacion and prep.caso_id == "memoria-01-prep" and prep.caso_base == "memoria-01"
    assert (prep.variante, prep.repeticion) == ("¿Puedo comprar alcohol con la BAES?", 1)
    assert prep.id_usuario == "eval-memoria-01" == primera.id_usuario  # el mismo id_usuario que el caso
    assert primera.caso_id == "memoria-01" and not primera.es_preparacion
    assert [c.caso_id for c in plan.pendientes].count("memoria-01-prep") == 1  # una sola, aunque haya 2 repeticiones
    assert (plan.preparaciones, plan.total, plan.saltadas) == (1, 6, 0)        # la preparación no cuenta como combinación del caso
    assert all(c.id_usuario == "eval-a" for c in plan.pendientes if c.caso_id == "a")


def test_la_preparacion_no_se_repite_si_ya_esta_hecha_ni_se_agrega_si_el_caso_no_tiene_pendientes():
    texto = "¿Puedo comprar alcohol con la BAES?"
    ya_hecha = {("memoria-01-prep", texto, 1)}
    plan = planificar([caso_memoria()], 1, ya_hecha)
    assert plan.preparaciones == 0 and all(not c.es_preparacion for c in plan.pendientes)

    # todo el caso ya hecho: no tiene sentido preparar nada
    hechas = {("memoria-01", "¿Qué te pregunté la vez anterior?", 1), ("memoria-01", "¿Recuerdas mi consulta anterior?", 1)}
    assert planificar([caso_memoria()], 1, hechas).pendientes == []


def test_el_campo_preparacion_reemplaza_al_valor_por_defecto_y_una_lista_vacia_lo_desactiva():
    assert preparacion_de(caso_memoria()) == ["¿Puedo comprar alcohol con la BAES?"]
    assert preparacion_de(caso_memoria(preparacion=["p1", "p2"])) == ["p1", "p2"]
    assert preparacion_de(caso_memoria(preparacion=[])) == []
    assert preparacion_de(caso(id="a")) == []  # solo memoria-01 trae una por defecto

    plan = planificar([caso(id="x", variantes=[], preparacion=["p1", "p2"])], 1, set())
    assert [(c.caso_id, c.variante) for c in plan.pendientes[:2]] == [("x-prep", "p1"), ("x-prep", "p2")]
    assert plan.preparaciones == 2


def test_la_preparacion_deja_historial_en_el_id_usuario_del_caso_y_las_metricas_la_excluyen(db_observabilidad):
    casos = [caso(id="normativa-x", variantes=[]), caso_memoria(variantes=[])]
    plan = planificar(casos, 1, set())

    def responder_por_caso(consulta, historial, id_usuario, **kw):
        guion = [llamada_tool("consultar_historial", {}), respuesta_final("ok")] if kw["caso_id"] == "memoria-01" else [respuesta_final("ok")]
        return responder_con(guion, consulta=consulta, id_usuario=id_usuario, **kw)

    resultado = ejecutar(plan.pendientes, responder_por_caso, **SIN_ESPERA)

    assert (resultado.completadas, resultado.faltan) == (3, 0)  # normativa-x, la preparación y memoria-01
    ejecuciones = filas(db_observabilidad, "SELECT caso_id, origen, variante FROM ejecuciones ORDER BY id")
    assert [e["caso_id"] for e in ejecuciones] == ["normativa-x", "memoria-01-prep", "memoria-01"]
    assert all(e["origen"] == "evaluacion" for e in ejecuciones)
    assert ejecuciones[1]["variante"] == "¿Puedo comprar alcohol con la BAES?"

    # la memoria de largo plazo: la preparación quedó en el historial del MISMO id_usuario que memoria-01...
    historial_memoria = {h["consulta_usuario"] for h in consultar_historial_usuario("eval-memoria-01")}
    assert "¿Puedo comprar alcohol con la BAES?" in historial_memoria
    # ...y no se filtró al de otro caso
    historial_otro = {h["consulta_usuario"] for h in consultar_historial_usuario("eval-normativa-x")}
    assert historial_otro == {"¿Puedo comprar alcohol?"}
    # memoria-01 encontró historial: lo que devolvió consultar_historial es más largo que el mensaje de "sin historial"
    [paso] = filas(db_observabilidad, "SELECT tamano_salida FROM pasos")
    assert paso["tamano_salida"] > len("Este estudiante no tiene interacciones registradas en sesiones anteriores.")

    m = calcular(casos, cargar_ejecuciones(db_observabilidad))
    assert m["ejecuciones"]["preparacion_excluidas"] == 1 and m["ejecuciones"]["consideradas"] == 2


def test_si_falla_la_preparacion_se_posterga_el_resto_del_caso():
    plan = planificar([caso_memoria(), caso(id="a", variantes=[])], 1, set())  # prep, pregunta, variante, a
    llamadas = []

    def fn(consulta, historial, id_usuario, **kw):
        llamadas.append(kw["caso_id"])
        if kw["caso_id"].endswith("-prep"):
            raise RuntimeError("falló")
        return {"estado": "INFORMATIVA"}

    mensajes = []
    resultado = ejecutar(plan.pendientes, fn, pausa=0, dormir=lambda s: None, imprimir=mensajes.append)

    assert llamadas == ["memoria-01-prep", "a"]  # memoria-01 no se ejecutó sin historial; los demás casos sí
    assert (resultado.completadas, resultado.fallidas, resultado.faltan) == (1, 1, 3)
    assert sum("pospuesta" in m and "memoria-01" in m for m in mensajes) == 2


# --------------------------------------------------------------- scripts (CLI)


def correr(script, *args, db_path, **extra_env):
    """Corre un script en un subproceso. Siempre con la API key VACÍA (un intento
    accidental de llamar a Gemini falla al instante) y la memoria aislada: estas
    pruebas no pueden gastar cuota ni escribir en data/processed/memoria.db."""
    entorno = {
        **os.environ,
        "OBSERVABILIDAD_DB_PATH": str(db_path),
        "MEMORIA_DB_PATH": str(Path(db_path).with_name("memoria_cli.db")),
        "GOOGLE_API_KEY": "",
        "PYTHONIOENCODING": "utf-8",
        **extra_env,
    }
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
    assert resultado["precision_herramientas"]["precision"] == 0.6
    assert resultado["precision_veredicto"]["etiquetadas"] == 0
    assert resultado["ejecuciones"]["preparacion_excluidas"] == 1

    # el estudiante etiqueta una fila; la segunda corrida NO pisa el archivo y la lee
    texto = etiquetas.read_text(encoding="utf-8-sig").splitlines()
    texto[1] = texto[1].replace(f"{ids['A_pa_1']},,", f"{ids['A_pa_1']},1,", 1)
    etiquetas.write_text("\n".join(texto), encoding="utf-8-sig")
    segunda = correr("calcular_metricas.py", *args, db_path=db_observabilidad)
    assert segunda.returncode == 0, segunda.stderr
    assert "generé la plantilla" not in segunda.stdout
    assert json.loads(salida.read_text(encoding="utf-8"))["precision_veredicto"]["etiquetadas"] == 1
    assert "Precisión de clasificación: 80.0% (4/5)" in segunda.stdout


def test_calcular_metricas_reporta_por_modelo_y_advierte_si_hay_mas_de_uno(db_observabilidad, tmp_path):
    registrar(db_observabilidad, "A", "pa", 1, modelo="modelo-a", pasos=[("buscar_normativa", 10)])
    registrar(db_observabilidad, "A", "pa", 2, modelo="modelo-b", pasos=[("buscar_normativa", 10)])
    casos = tmp_path / "casos.json"
    casos.write_text(json.dumps({"casos": CASOS_METRICAS}), encoding="utf-8")
    salida = tmp_path / "m.json"

    r = correr("calcular_metricas.py", "--casos-path", casos, "--etiquetas", tmp_path / "e.csv", "--salida", salida, db_path=db_observabilidad)

    assert r.returncode == 0, r.stderr
    assert "AVISO: Hay ejecuciones de 2 modelos (modelo-a, modelo-b)" in r.stdout
    assert "=== MODELO modelo-a ===" in r.stdout and "=== MODELO modelo-b ===" in r.stdout
    resultado = json.loads(salida.read_text(encoding="utf-8"))
    assert resultado["modelos"] == ["modelo-a", "modelo-b"] and set(resultado["por_modelo"]) == {"modelo-a", "modelo-b"}
    assert resultado["aviso_modelos"]


def test_calcular_metricas_rechaza_etiquetas_invalidas_y_casos_invalidos(db_observabilidad, tmp_path):
    casos = tmp_path / "casos.json"
    casos.write_text(json.dumps({"casos": CASOS_METRICAS}), encoding="utf-8")
    malas = escribir_csv(tmp_path / "malas.csv", "id_ejecucion,correcto,comentario\n1,quizas,\n")

    r = correr("calcular_metricas.py", "--casos-path", casos, "--etiquetas", malas, "--salida", tmp_path / "m.json", db_path=db_observabilidad)
    assert r.returncode == 2 and "línea 2" in r.stderr and not (tmp_path / "m.json").exists()

    casos.write_text(json.dumps({"casos": [caso(veredicto_esperado="x")]}), encoding="utf-8")
    r = correr("calcular_metricas.py", "--casos-path", casos, "--etiquetas", malas, "--salida", tmp_path / "m.json", db_path=db_observabilidad)
    assert r.returncode == 2 and "veredicto_esperado" in r.stderr


def test_evaluar_agente_imprime_el_modelo_activo_y_no_hace_nada_sin_casos_reales_o_sin_pendientes(db_observabilidad, tmp_path):
    solo_ejemplos = tmp_path / "ejemplos.json"
    solo_ejemplos.write_text(json.dumps({"casos": [caso(ejemplo=True)]}), encoding="utf-8")
    r = correr("evaluar_agente.py", "--casos-path", solo_ejemplos, db_path=db_observabilidad, GEMINI_MODEL="modelo-de-prueba")
    assert r.returncode == 0 and "No hay casos para evaluar" in r.stdout
    assert r.stdout.splitlines()[0] == "Modelo activo: modelo-de-prueba (proveedor: gemini)"  # lo primero que imprime
    assert not db_observabilidad.exists()

    # con todo ya hecho en la base tampoco hay nada que correr
    casos = tmp_path / "casos.json"
    casos.write_text(json.dumps({"casos": [caso(id="a", variantes=[])]}), encoding="utf-8")
    registrar(db_observabilidad, "a", "¿Puedo comprar alcohol?", 1)
    r = correr("evaluar_agente.py", "--casos-path", casos, "--repeticiones", "1", db_path=db_observabilidad)
    assert r.returncode == 0 and "1 ya estaban en la base, 0 pendientes" in r.stdout and "Nada que hacer" in r.stdout


def test_evaluar_agente_con_los_14_casos_reales_planifica_y_no_puede_llamar_a_gemini(db_observabilidad):
    r = correr(
        "evaluar_agente.py", "--solo-principal", "--orden", "--repeticiones", "1", "--pausa", "0", "--limite", "1",
        db_path=db_observabilidad, GEMINI_MODEL="modelo-de-prueba",
    )

    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[0] == "Modelo activo: modelo-de-prueba (proveedor: gemini)"
    assert (
        "14 caso(s), 1 repetición(es), solo la pregunta principal, orden prioridad: "
        "14 combinaciones, 0 ya estaban en la base, 14 pendientes (más 1 consulta(s) de preparación)."
    ) in r.stdout
    # sin API key el primer intento falla al instante: no hubo ninguna llamada real
    assert "[1/15] normativa-02 rep 1: ERROR RuntimeError" in r.stdout
    assert "Completadas: 0 | fallidas: 1" in r.stdout and "se alcanzó --limite 1" in r.stdout
    assert "Modelo: modelo-de-prueba" in r.stdout.split("--- Resumen ---")[1]
    assert filas(db_observabilidad, "SELECT caso_id, estado, tipo_error, modelo FROM ejecuciones") == [
        {"caso_id": "normativa-02", "estado": "ERROR", "tipo_error": "otro", "modelo": "modelo-de-prueba"}
    ]


@pytest.mark.parametrize("argumento", [("--repeticiones", "0"), ("--limite", "0"), ("--pausa", "-1"), ("--orden", "caos")])
def test_evaluar_agente_rechaza_parametros_invalidos(db_observabilidad, argumento):
    r = correr("evaluar_agente.py", *argumento, db_path=db_observabilidad)
    assert r.returncode == 2 and "error:" in r.stderr


def test_evaluar_agente_rechaza_ids_inexistentes_y_casos_invalidos(db_observabilidad, tmp_path):
    r = correr("evaluar_agente.py", "--casos", "no_existe", db_path=db_observabilidad)
    assert r.returncode == 2 and "Ids que no existen" in r.stderr and "no_existe" in r.stderr

    casos = tmp_path / "casos.json"
    casos.write_text(json.dumps({"casos": [caso(id="a")]}), encoding="utf-8")
    r = correr("evaluar_agente.py", "--casos-path", casos, "--casos", "b", db_path=db_observabilidad)
    assert r.returncode == 2 and "no existen" in r.stderr

    casos.write_text(json.dumps({"casos": [caso(categoria_caso="x")]}), encoding="utf-8")
    r = correr("evaluar_agente.py", "--casos-path", casos, db_path=db_observabilidad)
    assert r.returncode == 2 and "categoria_caso" in r.stderr


def test_evaluar_agente_muestra_los_avisos_del_esquema_pero_sigue(db_observabilidad, tmp_path):
    casos = tmp_path / "casos.json"
    casos.write_text(json.dumps({"casos": [caso(id="a", variantes=[], veredicto_esperdo="x")]}), encoding="utf-8")
    registrar(db_observabilidad, "a", "¿Puedo comprar alcohol?", 1)
    r = correr("evaluar_agente.py", "--casos-path", casos, "--repeticiones", "1", db_path=db_observabilidad)
    assert r.returncode == 0 and "Aviso: caso #1" in r.stderr and "'veredicto_esperdo'" in r.stderr


def test_evaluar_recuperacion_informa_cuantos_casos_omitio_y_no_carga_el_indice_si_no_hay_evaluables(db_observabilidad, tmp_path):
    casos = tmp_path / "casos.json"
    casos.write_text(
        json.dumps({"casos": [caso(id="a", fuente_esperada=""), caso(id="b", pregunta="pb", fuente_esperada=None)]}),
        encoding="utf-8",
    )
    r = correr("evaluar_recuperacion.py", "--casos-path", casos, db_path=db_observabilidad)
    assert r.returncode == 0, r.stderr
    assert "Omití 2 caso(s) sin 'fuente_esperada'" in r.stdout and "a, b" in r.stdout
    assert "No hay casos con 'fuente_esperada'" in r.stdout


def test_separar_evaluables_omite_los_casos_sin_fuente_y_evaluar_no_los_consulta():
    casos = [
        caso(id="a"),
        caso(id="b", pregunta="pb", fuente_esperada=""),
        caso(id="c", pregunta="pc", fuente_esperada=None),
        caso(id="d", pregunta="pd", fuente_esperada="   "),
        caso(id="e", pregunta="pe", fuente_esperada=...),  # sin el campo
    ]
    evaluables, omitidos = separar_evaluables(casos)
    assert [c["id"] for c in evaluables] == ["a"]
    assert [c["id"] for c in omitidos] == ["b", "c", "d", "e"]

    consultas = []
    evaluar(casos, lambda c, k: consultas.append(c) or [], k=3)
    assert consultas == ["¿Puedo comprar alcohol?", "¿Se puede comprar cerveza?"]  # solo las del caso con fuente
# ------------------------------- evaluar_recuperacion.py con un recuperador falso


def correr_recuperacion(monkeypatch, capsys, casos, resultados, *args):
    """Ejecuta `main()` del script en el mismo proceso, con un recuperador falso:
    sin índice, sin modelo de embeddings y sin red."""
    import importlib.util

    from rag.retrieval import vector_retriever

    consultas = []

    def recuperar_falso(consulta, k=4):
        consultas.append((consulta, k))
        return resultados[consulta]

    monkeypatch.setattr(vector_retriever, "recuperar", recuperar_falso)
    spec = importlib.util.spec_from_file_location("evaluar_recuperacion_script", RAIZ / "scripts" / "evaluar_recuperacion.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(sys, "argv", ["evaluar_recuperacion.py", "--casos-path", str(casos), *map(str, args)])

    codigo = script.main()
    salida = capsys.readouterr()
    return codigo, salida.out, salida.err, consultas


def casos_de_recuperacion(tmp_path, **alternativas):
    archivo = tmp_path / "casos.json"
    archivo.write_text(
        json.dumps(
            {
                "casos": [
                    caso(id="a", pregunta="pa", variantes=["pa2"], **alternativas),
                    caso(id="b", pregunta="pb", variantes=[], categoria_caso="comercio", fuente_esperada="frase de b"),
                    caso(id="c", pregunta="pc", variantes=[], categoria_caso="derivacion", fuente_esperada=""),
                ]
            }
        ),
        encoding="utf-8",
    )
    return archivo


RESULTADOS_CLI = {
    "pa": [frag("alcohol", 0.1, fuente="Manual A")],                                                     # acierta
    "pa2": [frag("otra cosa", 0.1, fuente="Manual B"), frag("regla\nalternativa", 0.2, fuente="Manual C")],  # solo la alternativa
    "pb": [frag("nada de nada", 0.1, fuente="Manual D")],                                                # no acierta
}


def test_evaluar_recuperacion_imprime_estricta_y_ampliada_lado_a_lado(monkeypatch, capsys, tmp_path):
    casos = casos_de_recuperacion(tmp_path, fuentes_alternativas=["regla alternativa"])
    salida_csv = tmp_path / "r.csv"

    codigo, out, err, consultas = correr_recuperacion(monkeypatch, capsys, casos, RESULTADOS_CLI, "--salida", salida_csv)

    assert codigo == 0 and err == ""
    assert [c for c, _ in consultas] == ["pa", "pa2", "pb"] and {k for _, k in consultas} == {10}   # el caso c no tiene fuente
    assert "Omití 1 caso(s) sin 'fuente_esperada' (no dependen del índice vectorial): c" in out
    assert "ESTRICTA (fuente_esperada)" in out and "AMPLIADA (+ fuentes_alternativas)" in out
    [global_] = [l for l in out.splitlines() if l.strip().startswith("GLOBAL")]
    estricta, ampliada = global_.split("|")[1:]
    assert "hit@3  33.3%" in estricta and "hit@5  33.3%" in estricta and "MRR 0.333" in estricta   # solo pa acierta
    assert "hit@3  66.7%" in ampliada and "hit@5  66.7%" in ampliada and "MRR 0.500" in ampliada   # pa y pa2 (rango 2)
    assert "La ampliada usa las fuentes_alternativas de 2 de las 3 consulta(s)." in out
    assert "[a] pa2 (la ampliada acierta en el rango 2)" in out and "[b] pb" in out
    assert salida_csv.exists()
    with open(salida_csv, encoding="utf-8-sig", newline="") as f:
        tabla = list(csv.DictReader(f))
    assert [(f["caso_id"], f["rango"], f["rango_ampliado"]) for f in tabla] == [("a", "1", "1"), ("a", "", "2"), ("b", "", "")]


def test_evaluar_recuperacion_sin_alternativas_avisa_que_las_dos_metricas_coinciden(monkeypatch, capsys, tmp_path):
    codigo, out, _, _ = correr_recuperacion(monkeypatch, capsys, casos_de_recuperacion(tmp_path), RESULTADOS_CLI, "--salida", tmp_path / "r.csv")
    assert codigo == 0 and "Ningún caso tiene 'fuentes_alternativas': la métrica ampliada es igual a la estricta." in out
    [global_] = [l for l in out.splitlines() if l.strip().startswith("GLOBAL")]
    estricta, ampliada = global_.split("|")[1:]
    assert estricta.strip() == ampliada.strip()


def test_detalle_imprime_los_fragmentos_solo_de_las_consultas_sin_acierto_estricto(monkeypatch, capsys, tmp_path):
    casos = casos_de_recuperacion(tmp_path, fuentes_alternativas=["regla alternativa"])
    codigo, out, _, _ = correr_recuperacion(monkeypatch, capsys, casos, RESULTADOS_CLI, "--salida", tmp_path / "r.csv", "--detalle")

    assert codigo == 0
    detalle = out.split("Detalle:")[1]
    assert "[a] pa2" in detalle and "[b] pb" in detalle
    assert "[a] pa\n" not in detalle                                   # pa acierta: no se detalla
    assert "    1. Manual B (distancia 0.1000)" in detalle and "    2. Manual C (distancia 0.2000)" in detalle
    assert "       regla alternativa" in detalle                      # el salto de línea del fragmento se vuelve espacio
    assert "(la ampliada acierta en el rango 2)" in detalle
    assert "usa --detalle" not in out


def test_sin_detalle_solo_se_sugiere_la_opcion(monkeypatch, capsys, tmp_path):
    codigo, out, _, _ = correr_recuperacion(monkeypatch, capsys, casos_de_recuperacion(tmp_path), RESULTADOS_CLI, "--salida", tmp_path / "r.csv")
    assert codigo == 0 and "Detalle:" not in out and "(usa --detalle para ver los fragmentos que sí se recuperaron)" in out
    assert "Manual D" not in out   # los fragmentos recuperados no se imprimen sin --detalle


def test_detalle_respeta_k_y_no_imprime_nada_si_todo_acierta(monkeypatch, capsys, tmp_path):
    todos = {"pa": [frag("alcohol", 0.1)], "pa2": [frag("alcohol", 0.1)], "pb": [frag("frase de b", 0.1)]}
    codigo, out, _, consultas = correr_recuperacion(
        monkeypatch, capsys, casos_de_recuperacion(tmp_path), todos, "--salida", tmp_path / "r.csv", "--detalle", "--k", "7"
    )
    assert codigo == 0 and {k for _, k in consultas} == {7}
    assert "Sin acierto" not in out and "Detalle:" not in out


def test_evaluar_recuperacion_rechaza_un_k_invalido(db_observabilidad):
    r = correr("evaluar_recuperacion.py", "--k", "0", db_path=db_observabilidad)
    assert r.returncode == 2 and "--k debe ser al menos 1" in r.stderr
