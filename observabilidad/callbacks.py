"""Handler de LangChain que mide, durante una ejecución del `AgentExecutor`,
cada herramienta (duración, éxito, tamaño de la salida) y los LLM
(cantidad de llamadas y tokens).

Una instancia por ejecución. Debe pasarse en `executor.invoke(..., config=
{"callbacks": [handler]})` y NO en el constructor del `AgentExecutor`: los
callbacks del constructor no se heredan a las herramientas ni al modelo, así
que no verían ningún evento.

Los argumentos de cada herramienta se guardan ya enmascarados (los genera el
LLM y pueden repetir datos personales que el estudiante escribió).
"""

import json
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from observabilidad.privacidad import enmascarar


def milisegundos(desde: float) -> int:
    """Milisegundos transcurridos desde un `time.perf_counter()`."""
    return round((time.perf_counter() - desde) * 1000)


def _a_texto(salida: Any) -> str:
    # Según cómo se invoque la herramienta, la salida llega como str o como
    # ToolMessage (con el texto en `.content`).
    return str(getattr(salida, "content", salida))


class RegistroPasos(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.pasos: list[dict] = []
        self.llamadas_llm = 0
        self.tokens_entrada: int | None = None
        self.tokens_salida: int | None = None
        self._abiertos: dict[Any, tuple[dict, float]] = {}

    # --- herramientas ---

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        nombre = (serialized or {}).get("name") or kwargs.get("name") or "desconocida"
        argumentos = json.dumps(inputs, ensure_ascii=False, default=str) if inputs else input_str
        paso = {
            "orden": len(self.pasos) + 1,
            "herramienta": nombre,
            "argumentos": enmascarar(argumentos),
            "duracion_ms": None,
            "exito": 0,
            "tamano_salida": None,
        }
        self.pasos.append(paso)
        self._abiertos[run_id] = (paso, time.perf_counter())

    def on_tool_end(self, output, *, run_id, **kwargs):
        paso, inicio = self._abiertos.pop(run_id, (None, None))
        if paso is None:
            return
        paso["duracion_ms"] = milisegundos(inicio)
        paso["exito"] = 1
        paso["tamano_salida"] = len(_a_texto(output))

    def on_tool_error(self, error, *, run_id, **kwargs):
        paso, inicio = self._abiertos.pop(run_id, (None, None))
        if paso is None:
            return
        paso["duracion_ms"] = milisegundos(inicio)
        paso["exito"] = 0

    # --- modelo ---

    def on_llm_end(self, response, **kwargs):
        self.llamadas_llm += 1
        for generaciones in response.generations:
            for generacion in generaciones:
                uso = getattr(getattr(generacion, "message", None), "usage_metadata", None)
                if uso:
                    self.tokens_entrada = (self.tokens_entrada or 0) + (uso.get("input_tokens") or 0)
                    self.tokens_salida = (self.tokens_salida or 0) + (uso.get("output_tokens") or 0)

    def on_llm_error(self, error, **kwargs):
        # Una llamada que falló también consumió una petición.
        self.llamadas_llm += 1

    def pasos_finales(self) -> list[dict]:
        """Los pasos registrados. Una herramienta que empezó y nunca terminó
        (el proceso se cortó a mitad) queda como fallida, con duración vacía."""
        self._abiertos.clear()
        return self.pasos
