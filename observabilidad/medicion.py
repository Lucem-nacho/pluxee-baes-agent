"""Acumula los datos de UNA ejecución de `agent.orchestrator.responder` y los
escribe en la base al cerrar.

Definiciones de las métricas:

- `latencia_total_ms`: desde que `responder` arranca hasta antes de escribir
  el registro. Incluye todo (clasificador, agente, escrituras en la memoria).
- `latencia_clasificador_ms`: la llamada a `clasificar`, reintentos de
  `llm/client.py` incluidos (sus esperas cuentan como latencia).
- `latencia_agente_ms`: solo `executor.invoke`, sin armar el agente ni las
  escrituras posteriores. Vacía si el agente no llegó a correr.
- `llamadas_llm`: llamadas del clasificador (1 si corrió) + llamadas del
  modelo del agente. Los reintentos internos de `llm/client.py` y de
  `max_retries` de LangChain no se ven y no se cuentan.
- `iteraciones_agente`: pasadas del `AgentExecutor`, es decir, llamadas al
  modelo del agente (una por cada herramienta pedida, más la de la respuesta
  final). Es la cantidad que topea `MAX_ITERACIONES_AGENTE`.
- `tokens_entrada` / `tokens_salida`: SOLO los del modelo del agente (vienen
  de `usage_metadata` en el callback). Los del clasificador quedan fuera:
  `llm.client.generar_respuesta` devuelve solo el texto, y exponer el uso
  habría exigido cambiar su firma. Por eso son NULL cuando el clasificador
  deriva sin que el agente corra.
- `estado`: INFORMATIVA, ESCALAR o ERROR (la ejecución terminó en excepción).
- `categoria`: lo que decidió el clasificador; NULL si falló antes de decidir.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from config import settings
from observabilidad import store
from observabilidad.callbacks import RegistroPasos, milisegundos
from observabilidad.privacidad import anonimizar_id_usuario, enmascarar

logger = logging.getLogger(__name__)

LONGITUD_MAX_RESPUESTA = 500
_ETAPAS = ("clasificador", "agente")


def tipo_de_error(error: BaseException) -> str:
    """Clasifica una excepción como "429", "503" u "otro".

    Recorre la cadena `__cause__`: en la capa de LangChain, un 429 llega como
    `GoogleRateLimitError` sin código propio, encadenada al `ClientError` del
    SDK de Google que sí lo trae; un 503 llega como `GoogleAPIError` con `code`.
    """
    actual: BaseException | None = error
    for _ in range(8):  # tope por si hubiera un ciclo en la cadena
        if actual is None:
            break
        codigo = getattr(actual, "code", None)
        if codigo == 429 or any("RateLimit" in clase.__name__ for clase in type(actual).__mro__):
            return "429"
        if codigo == 503:
            return "503"
        actual = actual.__cause__
    return "otro"


@dataclass
class Medicion:
    origen: str
    id_usuario: str
    consulta: str
    inicio: float = field(default_factory=time.perf_counter)
    categoria: str | None = None
    estado: str | None = None
    tipo_error: str = "ninguno"
    respuesta: str | None = None
    latencia_clasificador_ms: int | None = None
    latencia_agente_ms: int | None = None
    llamadas_clasificador: int = 0
    registro_pasos: RegistroPasos | None = None

    @contextmanager
    def medir(self, etapa: str):
        """Mide una etapa con `time.perf_counter()`, también si lanza una
        excepción (la latencia de lo que falló también interesa)."""
        if etapa not in _ETAPAS:
            raise ValueError(f"etapa desconocida: {etapa!r}")
        if etapa == "clasificador":
            self.llamadas_clasificador = 1
        desde = time.perf_counter()
        try:
            yield
        finally:
            setattr(self, f"latencia_{etapa}_ms", milisegundos(desde))

    def registrar_resultado(self, resultado: dict) -> None:
        self.estado = resultado["estado"]
        if self.estado == "INFORMATIVA":
            self.respuesta = resultado.get("respuesta")

    def registrar_excepcion(self, error: BaseException) -> None:
        self.estado = "ERROR"
        self.tipo_error = tipo_de_error(error)

    def cerrar(self) -> None:
        """Escribe el registro. Nunca lanza: observar no debe romper al agente
        (si la base falla, se loguea y la ejecución sigue como si nada)."""
        total = milisegundos(self.inicio)
        try:
            rp = self.registro_pasos
            llamadas_agente = rp.llamadas_llm if rp else 0

            estado, tipo_error = self.estado, self.tipo_error
            if estado is None:
                # Cortada por algo que no es Exception (p. ej. Ctrl+C).
                estado = "ERROR"
                if tipo_error == "ninguno":
                    tipo_error = "otro"

            respuesta = enmascarar(self.respuesta)
            ejecucion = {
                "origen": self.origen,
                "id_usuario_anonimo": anonimizar_id_usuario(self.id_usuario),
                "consulta": enmascarar(self.consulta),
                "categoria": self.categoria,
                "estado": estado,
                "latencia_total_ms": total,
                "latencia_clasificador_ms": self.latencia_clasificador_ms,
                "latencia_agente_ms": self.latencia_agente_ms,
                "llamadas_llm": self.llamadas_clasificador + llamadas_agente,
                "tokens_entrada": rp.tokens_entrada if rp else None,
                "tokens_salida": rp.tokens_salida if rp else None,
                "iteraciones_agente": llamadas_agente if rp else None,
                "tipo_error": tipo_error,
                "modelo": settings.GEMINI_MODEL,
                "respuesta": respuesta[:LONGITUD_MAX_RESPUESTA] if respuesta is not None else None,
            }
            store.registrar_ejecucion(ejecucion, rp.pasos_finales() if rp else [])
        except Exception:
            logger.exception("observabilidad: no se pudo registrar la ejecución")
