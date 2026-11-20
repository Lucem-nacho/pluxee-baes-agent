"""Ayudas compartidas por las pruebas de evaluación: un chat model FALSO que
sigue un guion (sin red ni Gemini), errores de Gemini con la forma real, y
`responder_con`, que llama al `responder` de verdad con el modelo y el
clasificador reemplazados."""

import time
from unittest.mock import patch

from google.genai import errors as errores_sdk
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError

from agent import orchestrator

INFORMATIVA = {"categoria": "INFORMATIVA", "justificacion": "prueba"}
ESCALAR = {"categoria": "ESCALAR", "justificacion": "prueba"}


def llamada_tool(nombre, args, id_="c1", uso=None):
    return AIMessage(
        content="",
        tool_calls=[{"name": nombre, "args": args, "id": id_, "type": "tool_call"}],
        usage_metadata=uso,
    )


def respuesta_final(texto, uso=None):
    return AIMessage(content=texto, usage_metadata=uso)


class ModeloFalso(BaseChatModel):
    """Responde con los mensajes de `guion` en orden; si `error` está
    definido, lo lanza en vez de responder."""

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


def responder_con(guion=None, *, error=None, consulta="¿Puedo comprar alcohol?", id_usuario="u1",
                  clasificacion=INFORMATIVA, **kwargs_responder):
    """Llama a `orchestrator.responder` de verdad, con el modelo del agente y
    el clasificador reemplazados."""
    modelo = ModeloFalso(guion=iter(guion) if guion is not None else None, error=error)
    with patch.object(orchestrator, "ChatGoogleGenerativeAI", return_value=modelo), patch.object(
        orchestrator, "clasificar", return_value=clasificacion
    ):
        return orchestrator.responder(consulta, "", id_usuario, **kwargs_responder)


def error_429(retry_delay="34s"):
    """Un 429 como llega desde LangChain: GoogleRateLimitError sin código,
    encadenado al ClientError del SDK que trae el cuerpo con el RetryInfo."""
    cuerpo = {
        "error": {
            "status": "RESOURCE_EXHAUSTED",
            "message": "cuota",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}],
        }
    }
    envuelto = GoogleRateLimitError("Error calling model (RESOURCE_EXHAUSTED)")
    envuelto.__cause__ = errores_sdk.ClientError(429, cuerpo, None)
    return envuelto


def error_503():
    return GoogleAPIError(code=503, response_json={"error": {"status": "UNAVAILABLE"}}, response=None)
