"""Pruebas de `observabilidad/`: el registro de cada ejecución de
`agent.orchestrator.responder`.

Usan un chat model FALSO que sigue un guion (como `tests/test_orchestrator.py`),
las herramientas reales y la base SQLite real en una carpeta temporal: sin red
ni cuota de Gemini. `tests/conftest.py` redirige la base de observabilidad a
esa carpeta en todas las pruebas.
"""

import hashlib
import itertools
import re
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from google.genai import errors as errores_sdk
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError

from agent import orchestrator
from config import settings
from memory.store import inicializar_db
from observabilidad import store
from observabilidad.medicion import tipo_de_error
from observabilidad.privacidad import _obtener_sal, anonimizar_id_usuario, enmascarar

RAIZ = Path(__file__).resolve().parents[1]
INFORMATIVA = {"categoria": "INFORMATIVA", "justificacion": "prueba"}


# ---------------------------------------------------------------- utilidades


def llamada_tool(nombre, args, id_="c1", uso=None):
    return AIMessage(
        content="",
        tool_calls=[{"name": nombre, "args": args, "id": id_, "type": "tool_call"}],
        usage_metadata=uso,
    )


def respuesta_final(texto, uso=None):
    return AIMessage(content=texto, usage_metadata=uso)


def uso(entrada, salida):
    return {"input_tokens": entrada, "output_tokens": salida, "total_tokens": entrada + salida}


class ModeloFalso(BaseChatModel):
    """Responde con los mensajes de `guion` en orden; si `error` está definido,
    lo lanza en vez de responder. `pausa` simula latencia del modelo."""

    guion: object = None
    error: object = None
    pausa: float = 0.0

    @property
    def _llm_type(self):
        return "falso"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(self.pausa)
        if self.error is not None:
            raise self.error
        return ChatResult(generations=[ChatGeneration(message=next(self.guion))])


@pytest.fixture(autouse=True)
def memoria_temporal(tmp_path, monkeypatch):
    # `responder` también escribe en la memoria de largo plazo (memoria.db).
    monkeypatch.setattr(settings, "MEMORIA_DB_PATH", tmp_path / "memoria_test.db")
    inicializar_db()


def responder_con(
    guion=None,
    *,
    error=None,
    pausa=0.0,
    consulta="¿Puedo comprar alcohol en el Jumbo?",
    id_usuario="u1",
    clasificacion=INFORMATIVA,
    **kwargs_responder,
):
    modelo = ModeloFalso(
        guion=iter(guion) if guion is not None and not hasattr(guion, "__next__") else guion,
        error=error,
        pausa=pausa,
    )
    with patch.object(orchestrator, "ChatGoogleGenerativeAI", return_value=modelo), patch.object(
        orchestrator, "clasificar", return_value=clasificacion
    ):
        return orchestrator.responder(consulta, "", id_usuario, **kwargs_responder)


def tabla(db_path, nombre):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(f) for f in conn.execute(f"SELECT * FROM {nombre} ORDER BY id")]
    finally:
        conn.close()


def error_429():
    sdk = errores_sdk.ClientError(
        429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "cuota"}}, None
    )
    envuelto = GoogleRateLimitError("Error calling model (RESOURCE_EXHAUSTED)")
    envuelto.__cause__ = sdk  # igual que `raise ... from e` en langchain-google-genai
    return envuelto


def error_503():
    return GoogleAPIError(code=503, response_json={"error": {"status": "UNAVAILABLE"}}, response=None)


# ----------------------------------------------------------- ejecución normal


def test_ejecucion_normal_se_registra_con_todas_las_metricas(db_observabilidad):
    def clasificar_lento(consulta, historial):
        time.sleep(0.05)
        return INFORMATIVA

    guion = [
        llamada_tool("buscar_comercio", {"nombre_o_categoria": "Jumbo"}, uso=uso(10, 5)),
        respuesta_final("Según el listado, en Jumbo no.", uso=uso(20, 8)),
    ]
    modelo = ModeloFalso(guion=iter(guion), pausa=0.05)
    with patch.object(orchestrator, "ChatGoogleGenerativeAI", return_value=modelo), patch.object(
        orchestrator, "clasificar", side_effect=clasificar_lento
    ):
        resultado = orchestrator.responder("¿Puedo comprar alcohol en el Jumbo?", "", "u1")

    assert resultado["estado"] == "INFORMATIVA"
    [fila] = tabla(db_observabilidad, "ejecuciones")
    assert fila["origen"] == "app"
    assert fila["categoria"] == "INFORMATIVA"
    assert fila["estado"] == "INFORMATIVA"
    assert fila["tipo_error"] == "ninguno"
    assert fila["modelo"] == settings.GEMINI_MODEL
    assert fila["respuesta"] == "Según el listado, en Jumbo no."
    assert fila["consulta"] == "¿Puedo comprar alcohol en el Jumbo?"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00", fila["fecha_utc"])
    # 1 llamada del clasificador + 2 del agente (una por herramienta, otra para la respuesta)
    assert fila["llamadas_llm"] == 3
    assert fila["iteraciones_agente"] == 2
    assert (fila["tokens_entrada"], fila["tokens_salida"]) == (30, 13)
    # las latencias se miden de verdad (las pausas del falso son de 50 ms)
    assert fila["latencia_clasificador_ms"] >= 50
    assert fila["latencia_agente_ms"] >= 100
    assert fila["latencia_total_ms"] >= fila["latencia_clasificador_ms"] + fila["latencia_agente_ms"]


def test_origen_por_defecto_es_app_y_se_puede_marcar_como_evaluacion(db_observabilidad):
    responder_con([respuesta_final("ok")])
    responder_con([respuesta_final("ok")], origen="evaluacion")
    assert [f["origen"] for f in tabla(db_observabilidad, "ejecuciones")] == ["app", "evaluacion"]


def test_origen_invalido_falla_antes_de_hacer_nada(db_observabilidad):
    with pytest.raises(ValueError, match="origen"):
        responder_con([respuesta_final("ok")], origen="produccion")
    assert not db_observabilidad.exists()


# --------------------------------------------------------------------- pasos


def test_los_pasos_quedan_asociados_a_su_ejecucion(db_observabilidad):
    def consulta_lenta(nombre):
        time.sleep(0.05)
        return []

    guion_1 = [
        llamada_tool("buscar_comercio", {"nombre_o_categoria": "Jumbo"}, id_="a"),
        llamada_tool("consultar_historial", {}, id_="b"),
        respuesta_final("listo"),
    ]
    guion_2 = [
        llamada_tool("buscar_comercio", {"nombre_o_categoria": "Subway"}, id_="c"),
        respuesta_final("listo"),
    ]
    responder_con(guion_1)
    with patch("agent.tools.consultar_comercio", side_effect=consulta_lenta):
        responder_con(guion_2)

    ejecuciones = tabla(db_observabilidad, "ejecuciones")
    pasos = tabla(db_observabilidad, "pasos")
    assert len(ejecuciones) == 2 and len(pasos) == 3

    de = lambda id_: [p for p in pasos if p["id_ejecucion"] == id_]
    primera, segunda = de(ejecuciones[0]["id"]), de(ejecuciones[1]["id"])
    assert [(p["orden"], p["herramienta"]) for p in primera] == [
        (1, "buscar_comercio"),
        (2, "consultar_historial"),
    ]
    assert [(p["orden"], p["herramienta"]) for p in segunda] == [(1, "buscar_comercio")]
    assert '"Jumbo"' in primera[0]["argumentos"]
    assert '"Subway"' in segunda[0]["argumentos"]
    assert all(p["exito"] == 1 and p["tamano_salida"] > 0 for p in pasos)
    assert segunda[0]["duracion_ms"] >= 50  # la herramienta se midió de verdad


def test_el_error_de_una_herramienta_queda_en_el_paso_y_se_relanza(db_observabilidad):
    with patch("agent.tools.recuperar", side_effect=RuntimeError("falló el índice")):
        with pytest.raises(RuntimeError, match="falló el índice"):
            responder_con([llamada_tool("buscar_normativa", {"consulta": "alcohol"})])

    [fila] = tabla(db_observabilidad, "ejecuciones")
    assert (fila["estado"], fila["tipo_error"], fila["respuesta"]) == ("ERROR", "otro", None)
    [paso] = tabla(db_observabilidad, "pasos")
    assert paso["id_ejecucion"] == fila["id"]
    assert paso["herramienta"] == "buscar_normativa"
    assert paso["exito"] == 0
    assert paso["duracion_ms"] is not None and paso["tamano_salida"] is None


# ----------------------------------------------------------------- ejecución fallida


@pytest.mark.parametrize(
    "error, tipo",
    [
        (error_429(), "429"),
        (error_503(), "503"),
        (RuntimeError("algo inesperado"), "otro"),
    ],
    ids=["429", "503", "otro"],
)
def test_ejecucion_que_falla_se_registra_y_se_relanza_la_misma_excepcion(
    db_observabilidad, error, tipo
):
    with pytest.raises(type(error)) as capturada:
        responder_con(error=error, pausa=0.02)
    assert capturada.value is error  # se relanza tal cual, sin envolver

    [fila] = tabla(db_observabilidad, "ejecuciones")
    assert fila["estado"] == "ERROR"
    assert fila["tipo_error"] == tipo
    assert fila["categoria"] == "INFORMATIVA"  # el clasificador sí llegó a decidir
    assert fila["respuesta"] is None
    assert fila["latencia_agente_ms"] >= 20  # también se mide lo que falla
    assert fila["llamadas_llm"] == 2  # clasificador + la llamada del agente que falló
    assert tabla(db_observabilidad, "pasos") == []


def test_si_el_clasificador_falla_se_registra_sin_categoria(db_observabilidad):
    with patch.object(orchestrator, "clasificar", side_effect=error_503()):
        with pytest.raises(GoogleAPIError):
            orchestrator.responder("hola", "", "u1")

    [fila] = tabla(db_observabilidad, "ejecuciones")
    assert (fila["estado"], fila["tipo_error"], fila["categoria"]) == ("ERROR", "503", None)
    assert fila["latencia_clasificador_ms"] is not None
    assert fila["latencia_agente_ms"] is None
    assert fila["llamadas_llm"] == 1


# ------------------------------------------------------ otros desenlaces


def test_derivacion_por_el_clasificador(db_observabilidad):
    resultado = responder_con(
        clasificacion={"categoria": "ESCALAR", "justificacion": "reclamo de cobro"},
        consulta="Me cobraron dos veces",
    )
    assert resultado["estado"] == "ESCALAR"

    [fila] = tabla(db_observabilidad, "ejecuciones")
    assert (fila["categoria"], fila["estado"], fila["tipo_error"]) == ("ESCALAR", "ESCALAR", "ninguno")
    assert fila["llamadas_llm"] == 1
    # el agente no corrió: sus métricas quedan vacías, no en cero
    assert fila["latencia_agente_ms"] is None
    assert fila["iteraciones_agente"] is None
    assert fila["tokens_entrada"] is None and fila["tokens_salida"] is None
    assert fila["respuesta"] is None
    assert tabla(db_observabilidad, "pasos") == []


def test_derivacion_a_mitad_del_agente(db_observabilidad):
    resultado = responder_con(
        [
            llamada_tool("escalar_a_ejecutivo", {"justificacion": "Pide modificar su saldo"}),
            respuesta_final("Tu consulta fue derivada."),
        ]
    )
    assert resultado["estado"] == "ESCALAR"

    [fila] = tabla(db_observabilidad, "ejecuciones")
    assert (fila["categoria"], fila["estado"], fila["tipo_error"]) == ("INFORMATIVA", "ESCALAR", "ninguno")
    [paso] = tabla(db_observabilidad, "pasos")
    assert paso["herramienta"] == "escalar_a_ejecutivo"


def test_clasificador_no_interpretable_tiene_su_propio_tipo_de_error(db_observabilidad):
    with patch("agent.classifier.generar_respuesta", return_value="esto no es JSON"):
        resultado = orchestrator.responder("¿qué onda?", "", "u1")

    assert resultado["estado"] == "ESCALAR"  # el comportamiento (derivar por seguridad) no cambia
    [fila] = tabla(db_observabilidad, "ejecuciones")
    assert (fila["categoria"], fila["estado"]) == ("ESCALAR", "ESCALAR")
    assert fila["tipo_error"] == "clasificador_no_interpretable"


def test_corte_por_limite_de_iteraciones_tiene_su_propio_tipo_de_error(db_observabilidad, monkeypatch):
    monkeypatch.setattr(orchestrator, "MAX_ITERACIONES_AGENTE", 2)
    bucle = (
        llamada_tool("buscar_comercio", {"nombre_o_categoria": f"x{n}"}, id_=f"c{n}")
        for n in itertools.count()
    )
    resultado = responder_con(bucle)

    assert resultado["respuesta"] == orchestrator.MENSAJE_SIN_RESPUESTA
    [fila] = tabla(db_observabilidad, "ejecuciones")
    assert (fila["estado"], fila["tipo_error"]) == ("INFORMATIVA", "limite_iteraciones")
    assert fila["iteraciones_agente"] == 2
    assert len(tabla(db_observabilidad, "pasos")) == 2
    assert "Agent stopped" not in fila["respuesta"]


def test_mensaje_sin_respuesta_por_salida_vacia_tiene_su_propio_tipo_de_error(db_observabilidad):
    resultado = responder_con([respuesta_final("")])

    assert resultado["respuesta"] == orchestrator.MENSAJE_SIN_RESPUESTA
    [fila] = tabla(db_observabilidad, "ejecuciones")
    assert (fila["estado"], fila["tipo_error"]) == ("INFORMATIVA", "sin_respuesta")
    assert fila["respuesta"] == orchestrator.MENSAJE_SIN_RESPUESTA


def test_la_respuesta_se_trunca_a_500_caracteres(db_observabilidad):
    responder_con([respuesta_final("a" * 900)])
    [fila] = tabla(db_observabilidad, "ejecuciones")
    assert fila["respuesta"] == "a" * 500


# ------------------------------------------------------------ privacidad

DATOS_PERSONALES = ["12.345.678-9", "juan.perez@correo.cl", "+56 9 1234 5678"]


def test_rut_correo_y_telefono_no_llegan_a_la_base(db_observabilidad):
    consulta = (
        "Mi RUT es 12.345.678-9, escríbeme a juan.perez@correo.cl "
        "o llámame al +56 9 1234 5678: ¿puedo comprar en el Jumbo?"
    )
    guion = [
        # el LLM puede repetir datos del estudiante en los argumentos de una tool
        llamada_tool("buscar_comercio", {"nombre_o_categoria": "Jumbo 12.345.678-9 juan.perez@correo.cl"}),
        respuesta_final("Hola, te contactaremos al +56 9 1234 5678 o a juan.perez@correo.cl."),
    ]
    responder_con(guion, consulta=consulta)

    contenido = db_observabilidad.read_bytes().decode("utf-8", errors="replace")
    for dato in DATOS_PERSONALES:
        assert dato not in contenido, f"{dato!r} llegó a la base"

    [fila] = tabla(db_observabilidad, "ejecuciones")
    assert "[RUT]" in fila["consulta"] and "[CORREO]" in fila["consulta"] and "[TELEFONO]" in fila["consulta"]
    assert "[CORREO]" in fila["respuesta"] and "[TELEFONO]" in fila["respuesta"]
    [paso] = tabla(db_observabilidad, "pasos")
    assert "[RUT]" in paso["argumentos"] and "[CORREO]" in paso["argumentos"]
    assert "Jumbo" in paso["argumentos"]  # lo que no es dato personal se conserva


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("mi rut es 12.345.678-9 gracias", "mi rut es [RUT] gracias"),
        ("rut 7.654.321-K ok", "rut [RUT] ok"),
        ("rut 12345678-9.", "rut [RUT]."),
        ("escríbeme a a.b+x@mail.duocuc.cl", "escríbeme a [CORREO]"),
        ("llámame al +56 9 1234 5678", "llámame al [TELEFONO]"),
        ("fono 912345678", "fono [TELEFONO]"),
        ("fijo 2 2123 4567", "fijo [TELEFONO]"),
        ("fijo (41) 234 5678", "fijo [TELEFONO]"),
    ],
)
def test_enmascarar_datos_personales(texto, esperado):
    assert enmascarar(texto) == esperado


@pytest.mark.parametrize(
    "texto",
    [
        "El monto es $60.000 al mes",
        "según la Ley N° 20.606 de etiquetado",
        "Resolución Exenta DN-00556/2026",
        "Actualizado el 2026-11-20 a las 14:30",
        "Tabla N° 16, página 33 del manual",
        "un saldo de $1.234.567",
        "hasta 2 sellos de advertencia",
    ],
)
def test_enmascarar_no_toca_texto_normativo_ni_montos_ni_fechas(texto):
    assert enmascarar(texto) == texto


def test_enmascarar_acepta_none():
    assert enmascarar(None) is None


def test_id_usuario_queda_anonimizado(db_observabilidad):
    responder_con([respuesta_final("ok")], id_usuario="alumno-secreto-123")
    responder_con([respuesta_final("ok")], id_usuario="alumno-secreto-123")
    responder_con([respuesta_final("ok")], id_usuario="otro-alumno")

    assert "alumno-secreto-123" not in db_observabilidad.read_bytes().decode("utf-8", errors="replace")
    a1, a2, b = (f["id_usuario_anonimo"] for f in tabla(db_observabilidad, "ejecuciones"))
    assert re.fullmatch(r"[0-9a-f]{12}", a1)
    assert a1 == a2  # el mismo usuario se puede agrupar
    assert a1 != b

    # es SHA-256 con sal: sal:id, truncado a 12 hex
    sal = _obtener_sal()
    assert a1 == hashlib.sha256(f"{sal}:alumno-secreto-123".encode()).hexdigest()[:12]
    assert "alumno-secreto-123" not in (db_observabilidad.with_name("observabilidad.sal")).read_text()


def test_la_sal_distingue_instalaciones(monkeypatch):
    sin_sal_a = anonimizar_id_usuario("alumno")
    monkeypatch.setattr(settings, "OBSERVABILIDAD_SAL", "otra-sal")
    assert anonimizar_id_usuario("alumno") != sin_sal_a


def test_la_sal_se_crea_una_sola_vez_aunque_haya_hilos_concurrentes(db_observabilidad):
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(lambda _: anonimizar_id_usuario("alumno"), range(24)))
    assert len(set(hashes)) == 1
    assert list(db_observabilidad.parent.glob("*.tmp")) == []  # sin temporales sobrantes


# ------------------------------------------------------------------- purga


def sembrar(db_path, edades_en_dias):
    ahora = datetime.now(timezone.utc)
    for dias in edades_en_dias:
        store.registrar_ejecucion(
            {
                "fecha_utc": (ahora - timedelta(days=dias)).isoformat(timespec="milliseconds"),
                "origen": "app",
                "id_usuario_anonimo": "abc123abc123",
                "consulta": f"consulta de hace {dias} días",
                "categoria": "INFORMATIVA",
                "estado": "INFORMATIVA",
                "latencia_total_ms": 10,
                "latencia_clasificador_ms": 5,
                "latencia_agente_ms": 5,
                "llamadas_llm": 2,
                "tokens_entrada": None,
                "tokens_salida": None,
                "iteraciones_agente": 1,
                "tipo_error": "ninguno",
                "modelo": "m",
                "respuesta": "r",
            },
            [
                {
                    "orden": 1,
                    "herramienta": "buscar_comercio",
                    "argumentos": "{}",
                    "duracion_ms": 1,
                    "exito": 1,
                    "tamano_salida": 3,
                }
            ],
            db_path=db_path,
        )


def test_la_purga_borra_lo_antiguo_y_sus_pasos_y_deja_lo_reciente(db_observabilidad):
    sembrar(db_observabilidad, [40, 31, 29, 1])

    borradas = store.purgar_anteriores_a(30)

    assert borradas == 2
    restantes = tabla(db_observabilidad, "ejecuciones")
    assert [f["consulta"] for f in restantes] == ["consulta de hace 29 días", "consulta de hace 1 días"]
    # los pasos de lo borrado se fueron con ello, sin huérfanos
    assert {p["id_ejecucion"] for p in tabla(db_observabilidad, "pasos")} == {f["id"] for f in restantes}


def _correr_purga(db_path, *args):
    entorno = {**__import__("os").environ, "OBSERVABILIDAD_DB_PATH": str(db_path), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / "purgar_observabilidad.py"), *args],
        cwd=RAIZ, env=entorno, capture_output=True, text=True, encoding="utf-8", timeout=120,
    )


def test_el_script_de_purga_usa_30_dias_por_defecto_y_acepta_otro_valor(db_observabilidad):
    sembrar(db_observabilidad, [40, 10, 1])

    por_defecto = _correr_purga(db_observabilidad)
    assert por_defecto.returncode == 0, por_defecto.stderr
    assert "1 ejecuciones con más de 30 días" in por_defecto.stdout
    assert len(tabla(db_observabilidad, "ejecuciones")) == 2

    siete = _correr_purga(db_observabilidad, "--dias", "7")
    assert siete.returncode == 0, siete.stderr
    assert "1 ejecuciones con más de 7 días" in siete.stdout
    assert [f["consulta"] for f in tabla(db_observabilidad, "ejecuciones")] == ["consulta de hace 1 días"]


def test_el_script_de_purga_rechaza_dias_negativos(db_observabilidad):
    sembrar(db_observabilidad, [40])
    resultado = _correr_purga(db_observabilidad, "--dias", "-1")
    assert resultado.returncode != 0
    assert len(tabla(db_observabilidad, "ejecuciones")) == 1


# ------------------------------------------------ observar no debe romper al agente


def test_un_fallo_al_registrar_no_cambia_lo_que_devuelve_responder(db_observabilidad):
    esperado = responder_con([respuesta_final("hola")])

    with patch.object(store, "registrar_ejecucion", side_effect=sqlite3.OperationalError("disco lleno")):
        obtenido = responder_con([respuesta_final("hola")])

    assert obtenido == esperado


def test_si_fallan_el_agente_y_el_registro_se_relanza_la_excepcion_del_agente(db_observabilidad):
    original = RuntimeError("falló el agente")
    with patch.object(store, "registrar_ejecucion", side_effect=sqlite3.OperationalError("disco lleno")):
        with pytest.raises(RuntimeError) as capturada:
            responder_con(error=original)
    assert capturada.value is original


# --------------------------------------------------------- clasificación de errores


def test_tipo_de_error_recorre_la_cadena_de_causas():
    assert tipo_de_error(error_429()) == "429"
    assert tipo_de_error(error_503()) == "503"
    assert tipo_de_error(errores_sdk.ServerError(503, {"error": {}}, None)) == "503"
    assert tipo_de_error(errores_sdk.ClientError(429, {"error": {}}, None)) == "429"
    assert tipo_de_error(errores_sdk.ClientError(400, {"error": {}}, None)) == "otro"
    assert tipo_de_error(ValueError("x")) == "otro"


# ------------------------------------------------------------------- app.py


def test_la_barra_lateral_avisa_que_las_consultas_se_registran():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(RAIZ / "app.py"))
    at.run()

    assert not at.exception
    textos = [c.value for c in at.sidebar.caption]
    assert any("se registran de forma anónima" in t for t in textos), textos
