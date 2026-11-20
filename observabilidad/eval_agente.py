"""Ejecución reanudable de la evaluación del agente contra Gemini real.

Recorre las combinaciones (caso, variante, repetición) y llama a `responder`
con `origen="evaluacion"`; cada ejecución queda en la base de observabilidad.

- Orden: por repetición y, dentro de cada una, todos los casos y variantes.
  Así una corrida cortada por la cuota deja cobertura de todos los casos en
  vez de pocos casos muy repetidos.
- Reanudable: salta lo que ya está en la base (`store.combinaciones_evaluadas`).
  Un intento que terminó en 429 o 503 no cuenta como hecho y se reintenta.
- 429: espera el `retryDelay` que informa Gemini y reintenta UNA vez esa
  combinación. Cada respuesta 429 suma a un contador que se pone en cero con
  cualquier ejecución que no sea 429; al llegar a 3 seguidas la corrida se
  detiene limpia. También se detiene si el proveedor pide esperar más de
  `espera_maxima` segundos (señal de cuota DIARIA agotada: quedarse dormido
  horas no tiene sentido).
- `limite` cuenta llamadas a `responder`, reintentos incluidos, porque cada
  una consume cuota.
"""

import re
import time
from dataclasses import dataclass
from typing import Callable

from observabilidad.casos import preguntas_del_caso
from observabilidad.medicion import tipo_de_error
from observabilidad.privacidad import enmascarar

MAX_429_SEGUIDOS = 3
ESPERA_MAXIMA_S = 300.0
ESPERA_POR_DEFECTO_S = 60.0

_PATRONES_ESPERA = (
    re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE),
    re.compile(r"retryDelay['\"]?\s*:\s*['\"]?([0-9]+(?:\.[0-9]+)?)s"),
)


@dataclass(frozen=True)
class Combinacion:
    caso_id: str
    variante: str  # texto exacto de la pregunta que se envía
    repeticion: int

    @property
    def clave(self) -> tuple[str, str, int]:
        # La base guarda la variante ya enmascarada: se compara en esos términos.
        return (self.caso_id, enmascarar(self.variante), self.repeticion)


@dataclass
class ResultadoCorrida:
    completadas: int = 0
    fallidas: int = 0
    saltadas: int = 0
    llamadas: int = 0
    faltan: int = 0
    detenida_por: str | None = None  # None | "limite" | "cuota" | "interrupcion"
    detalle: str = ""


def combinaciones(casos: list[dict], repeticiones: int) -> list[Combinacion]:
    return [
        Combinacion(caso["id"], texto, repeticion)
        for repeticion in range(1, repeticiones + 1)
        for caso in casos
        for texto in preguntas_del_caso(caso)
    ]


def separar_pendientes(
    todas: list[Combinacion], hechas: set[tuple[str, str, int]]
) -> tuple[list[Combinacion], int]:
    """(pendientes, cantidad de saltadas porque ya estaban en la base)."""
    pendientes = [c for c in todas if c.clave not in hechas]
    return pendientes, len(todas) - len(pendientes)


def segundos_de_espera(error: BaseException) -> float | None:
    """El `retryDelay` de un 429 de Gemini, o None si no se puede extraer.
    Busca en el cuerpo estructurado del error del SDK y, si no, en el texto,
    recorriendo la cadena de causas (LangChain envuelve el error del SDK)."""
    actual: BaseException | None = error
    for _ in range(8):
        if actual is None:
            break
        detalles = getattr(actual, "details", None)
        if isinstance(detalles, dict):
            try:
                for detalle in detalles["error"]["details"]:
                    if str(detalle.get("@type", "")).endswith("RetryInfo"):
                        return float(str(detalle["retryDelay"]).rstrip("s"))
            except (KeyError, TypeError, ValueError, AttributeError):
                pass
        texto = str(actual)
        for patron in _PATRONES_ESPERA:
            encontrado = patron.search(texto)
            if encontrado:
                return float(encontrado.group(1))
        actual = actual.__cause__
    return None


def ejecutar(
    pendientes: list[Combinacion],
    responder_fn: Callable[..., dict],
    *,
    saltadas: int = 0,
    limite: int | None = None,
    pausa: float = 15.0,
    id_usuario: str = "evaluacion",
    dormir: Callable[[float], None] = time.sleep,
    imprimir: Callable[[str], None] = print,
    max_429_seguidos: int = MAX_429_SEGUIDOS,
    espera_maxima: float = ESPERA_MAXIMA_S,
    espera_por_defecto: float = ESPERA_POR_DEFECTO_S,
) -> ResultadoCorrida:
    resultado = ResultadoCorrida(saltadas=saltadas)
    seguidos_429 = 0
    total = len(pendientes)

    def detener(motivo: str, detalle: str) -> None:
        resultado.detenida_por, resultado.detalle = motivo, detalle

    try:
        for numero, combo in enumerate(pendientes, start=1):
            etiqueta = f"[{numero}/{total}] {combo.caso_id} rep {combo.repeticion}"
            reintentada = False
            espera_ya_hecha = False
            while True:
                if limite is not None and resultado.llamadas >= limite:
                    detener("limite", f"se alcanzó --limite {limite}")
                    break
                if resultado.llamadas > 0 and not espera_ya_hecha:
                    dormir(pausa)
                espera_ya_hecha = False
                resultado.llamadas += 1
                try:
                    respuesta = responder_fn(
                        combo.variante,
                        "",
                        id_usuario,
                        origen="evaluacion",
                        caso_id=combo.caso_id,
                        variante=combo.variante,
                        repeticion=combo.repeticion,
                    )
                except Exception as error:
                    if tipo_de_error(error) != "429":
                        seguidos_429 = 0
                        resultado.fallidas += 1
                        imprimir(f"{etiqueta}: ERROR {type(error).__name__}; se sigue con la próxima")
                        break
                    seguidos_429 += 1
                    espera = segundos_de_espera(error) or espera_por_defecto
                    if seguidos_429 >= max_429_seguidos:
                        detener("cuota", f"{seguidos_429} respuestas 429 seguidas")
                        break
                    if espera > espera_maxima:
                        detener(
                            "cuota",
                            f"Gemini pide esperar {espera:.0f}s (más de {espera_maxima:.0f}s): "
                            "probablemente la cuota diaria está agotada",
                        )
                        break
                    if reintentada:
                        resultado.fallidas += 1
                        imprimir(f"{etiqueta}: 429 otra vez tras reintentar; se sigue con la próxima")
                        break
                    reintentada = True
                    imprimir(f"{etiqueta}: 429, espero {espera:.0f}s y reintento una vez")
                    dormir(espera + 1)
                    espera_ya_hecha = True
                    continue
                else:
                    seguidos_429 = 0
                    resultado.completadas += 1
                    imprimir(f"{etiqueta}: {respuesta.get('estado', '?')}")
                    break
            if resultado.detenida_por:
                break
    except KeyboardInterrupt:
        detener("interrupcion", "interrumpida con Ctrl+C")

    resultado.faltan = total - resultado.completadas
    return resultado
