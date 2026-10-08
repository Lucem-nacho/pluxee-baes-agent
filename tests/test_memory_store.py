"""Pruebas de memory/store.py — escalamientos persistentes y memoria de largo
plazo. Sin dependencias externas ni LLM: sqlite3 es stdlib, así que estas
pruebas corren siempre, incluso sin GOOGLE_API_KEY ni paquetes instalados."""

import tempfile
from pathlib import Path

import pytest

from memory.store import (
    consultar_historial_usuario,
    inicializar_db,
    registrar_escalamiento,
    registrar_interaccion,
)


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp) / "memoria_test.db"


def test_inicializar_db_crea_archivo_y_directorio(db_path):
    assert not db_path.exists()
    inicializar_db(db_path)
    assert db_path.exists()


def test_inicializar_db_es_idempotente(db_path):
    inicializar_db(db_path)
    inicializar_db(db_path)  # no debe fallar ni duplicar nada


def test_registrar_escalamiento_persiste_y_devuelve_paquete_compatible(db_path):
    resultado = registrar_escalamiento(
        consulta_usuario="Me cobraron dos veces en el mismo local",
        historial_contexto="Usuario: Me cobraron dos veces...",
        justificacion="Reclamo de cobro duplicado, requiere revisión de cuenta.",
        db_path=db_path,
    )

    # Misma forma que agent.escalation.derivar devolvía antes (compatibilidad con app.py)
    assert resultado["estado"] == "ESCALAR"
    assert resultado["consulta_usuario"] == "Me cobraron dos veces en el mismo local"
    assert "fecha_derivacion" in resultado
    assert resultado["id_escalamiento"] is not None


def test_dos_escalamientos_quedan_en_filas_distintas(db_path):
    r1 = registrar_escalamiento("consulta A", "hist A", "justif A", db_path=db_path)
    r2 = registrar_escalamiento("consulta B", "hist B", "justif B", db_path=db_path)
    assert r1["id_escalamiento"] != r2["id_escalamiento"]


def test_registrar_interaccion_rechaza_estado_invalido(db_path):
    with pytest.raises(ValueError):
        registrar_interaccion("user1", "consulta", "respuesta", estado="OTRO", db_path=db_path)


def test_consultar_historial_usuario_vacio_para_usuario_nuevo(db_path):
    assert consultar_historial_usuario("usuario_que_no_existe", db_path=db_path) == []


def test_consultar_historial_usuario_devuelve_mas_reciente_primero(db_path):
    registrar_interaccion("ignacio", "¿puedo comprar alcohol?", "No, está prohibido.", "INFORMATIVA", db_path=db_path)
    registrar_interaccion("ignacio", "¿y cigarros?", "Tampoco están permitidos.", "INFORMATIVA", db_path=db_path)

    historial = consultar_historial_usuario("ignacio", db_path=db_path)

    assert len(historial) == 2
    assert historial[0]["consulta_usuario"] == "¿y cigarros?"  # la más reciente, primero
    assert historial[1]["consulta_usuario"] == "¿puedo comprar alcohol?"


def test_consultar_historial_usuario_respeta_limite(db_path):
    for i in range(10):
        registrar_interaccion("ignacio", f"consulta {i}", f"respuesta {i}", "INFORMATIVA", db_path=db_path)

    assert len(consultar_historial_usuario("ignacio", limite=3, db_path=db_path)) == 3
    assert len(consultar_historial_usuario("ignacio", limite=100, db_path=db_path)) == 10


def test_historial_no_mezcla_usuarios_distintos(db_path):
    registrar_interaccion("ignacio", "consulta de ignacio", "respuesta", "INFORMATIVA", db_path=db_path)
    registrar_interaccion("otro_usuario", "consulta de otro", "respuesta", "INFORMATIVA", db_path=db_path)

    historial_ignacio = consultar_historial_usuario("ignacio", db_path=db_path)
    assert len(historial_ignacio) == 1
    assert historial_ignacio[0]["consulta_usuario"] == "consulta de ignacio"


def test_persiste_entre_conexiones_distintas(db_path):
    """La prueba central del módulo: simula lo que importa de verdad —
    que los datos sobrevivan a que el proceso se reinicie (o, en Streamlit,
    a que cada interacción corra en su propio hilo con su propia conexión)."""
    registrar_interaccion("ignacio", "consulta", "respuesta", "INFORMATIVA", db_path=db_path)

    # Nueva "sesión": ninguna conexión ni estado en memoria se reutiliza acá.
    historial = consultar_historial_usuario("ignacio", db_path=db_path)
    assert len(historial) == 1
