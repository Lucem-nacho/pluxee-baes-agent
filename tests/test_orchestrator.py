"""Pruebas de agent/orchestrator.py (EP2) con un chat model FALSO que sigue un
guion de tool calls — ejercitan el ciclo real de `AgentExecutor`, las tools
reales (`agent/tools.py`) y la memoria SQLite real, sin red ni cuota de
Gemini. NO validan la calidad de las respuestas del LLM (eso solo se ve con
`scripts/check_agent_executor.py`), sino el cableado y los casos borde que
aparecieron al correr el código con LangChain y Gemini de verdad:

- el contenido final puede venir como lista de bloques, no como `str`;
- el agente puede agotar `max_iterations` y la librería devuelve un texto
  interno en vez de lanzar una excepción.
"""

import itertools
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent import orchestrator
from config import settings
from memory.store import consultar_historial_usuario, inicializar_db, registrar_interaccion

INFORMATIVA = {"categoria": "INFORMATIVA", "justificacion": "prueba"}


def llamada_tool(nombre: str, args: dict, id_: str = "c1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": nombre, "args": args, "id": id_, "type": "tool_call"}]
    )


class ModeloFalso(BaseChatModel):
    """Responde con los mensajes de `guion` en orden. Si `guion` es un
    iterador infinito, simula un agente que nunca termina."""

    guion: object
    vistos: list = []

    @property
    def _llm_type(self):
        return "falso"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.vistos.append(messages)
        return ChatResult(generations=[ChatGeneration(message=next(self.guion))])


@pytest.fixture
def db_temporal(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MEMORIA_DB_PATH", tmp_path / "memoria_test.db")
    inicializar_db()


def responder_con(guion, consulta="¿Puedo comprar alcohol en el Jumbo?", id_usuario="u1", clasificacion=INFORMATIVA):
    modelo = ModeloFalso(guion=iter(guion) if not hasattr(guion, "__next__") else guion)
    with patch.object(orchestrator, "ChatGoogleGenerativeAI", return_value=modelo), patch.object(
        orchestrator, "clasificar", return_value=clasificacion
    ):
        return orchestrator.responder(consulta, "", id_usuario), modelo


def test_respuesta_final_como_lista_de_bloques_se_normaliza_a_texto(db_temporal):
    guion = [
        llamada_tool("buscar_comercio", {"nombre_o_categoria": "Jumbo"}),
        AIMessage(content=[{"type": "text", "text": "Según el listado, en Jumbo no."}]),
    ]
    resultado, _ = responder_con(guion)

    assert resultado["estado"] == "INFORMATIVA"
    assert resultado["respuesta"] == "Según el listado, en Jumbo no."
    assert any("buscar_comercio" in paso for paso in resultado["pasos"])
    # se pudo persistir en SQLite (una lista cruda lanzaba ProgrammingError)
    historial = consultar_historial_usuario("u1")
    assert historial[0]["respuesta"] == "Según el listado, en Jumbo no."


def test_el_agente_recibe_la_observacion_real_de_la_tool(db_temporal):
    guion = [
        llamada_tool("buscar_comercio", {"nombre_o_categoria": "Jumbo"}),
        AIMessage(content="ok"),
    ]
    _, modelo = responder_con(guion)

    observacion = [m for m in modelo.vistos[1] if m.type == "tool"][0].content
    assert "Jumbo Providencia" in observacion


def test_corte_por_limite_de_iteraciones_no_filtra_texto_interno(db_temporal, monkeypatch):
    monkeypatch.setattr(orchestrator, "MAX_ITERACIONES_AGENTE", 2)
    contador = itertools.count()
    bucle = (
        llamada_tool("buscar_normativa", {"consulta": f"bebidas {n}"}, id_=f"c{n}")
        for n in contador
    )
    resultado, _ = responder_con(bucle)

    assert resultado["estado"] == "INFORMATIVA"
    assert resultado["respuesta"] == orchestrator.MENSAJE_SIN_RESPUESTA
    assert "Agent stopped" not in resultado["respuesta"]
    assert consultar_historial_usuario("u1")[0]["respuesta"] == orchestrator.MENSAJE_SIN_RESPUESTA


def test_escalar_a_ejecutivo_a_mitad_de_razonamiento_devuelve_escalar(db_temporal):
    guion = [
        llamada_tool("escalar_a_ejecutivo", {"justificacion": "Pide modificar su saldo"}),
        AIMessage(content="Tu consulta fue derivada a un ejecutivo."),
    ]
    resultado, _ = responder_con(guion)

    assert resultado["estado"] == "ESCALAR"
    assert resultado["id_escalamiento"] >= 1
    assert resultado["justificacion_clasificador"] == "Pide modificar su saldo"
    assert consultar_historial_usuario("u1")[0]["estado"] == "ESCALAR"


def test_consultar_historial_solo_lee_el_usuario_de_la_sesion(db_temporal):
    registrar_interaccion("u1", "pregunta de u1", "respuesta a u1", "INFORMATIVA")
    registrar_interaccion("otro", "pregunta SECRETA de otro", "respuesta a otro", "INFORMATIVA")

    guion = [llamada_tool("consultar_historial", {}), AIMessage(content="ok")]
    _, modelo = responder_con(guion)

    observacion = [m for m in modelo.vistos[1] if m.type == "tool"][0].content
    assert "pregunta de u1" in observacion
    assert "SECRETA" not in observacion


def test_clasificador_escalar_no_invoca_al_agente(db_temporal):
    constructor_llm = MagicMock()
    clasificacion = {"categoria": "ESCALAR", "justificacion": "reclamo de cobro"}
    with patch.object(orchestrator, "ChatGoogleGenerativeAI", constructor_llm), patch.object(
        orchestrator, "clasificar", return_value=clasificacion
    ):
        resultado = orchestrator.responder("Me cobraron dos veces", "", "u1")

    assert resultado["estado"] == "ESCALAR"
    constructor_llm.assert_not_called()
    assert consultar_historial_usuario("u1")[0]["estado"] == "ESCALAR"
