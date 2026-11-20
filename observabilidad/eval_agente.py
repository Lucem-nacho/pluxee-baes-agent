"""Ejecución reanudable de la evaluación del agente contra Gemini real.

Recorre las combinaciones (caso, variante, repetición) y llama a `responder`
con `origen="evaluacion"`; cada ejecución queda en la base de observabilidad.

- Orden: por defecto ("repeticion"), por repetición y, dentro de cada una,
  todos los casos y variantes. Con orden "prioridad": primero las preguntas
  principales de todos los casos, luego las variantes y al final las
  repeticiones. En ambos, una corrida cortada por la cuota deja cobertura de
  todos los casos en vez de pocos casos muy repetidos.
- Cada caso se ejecuta con su propio id_usuario fijo (`eval-<caso_id>`), para
  que la memoria de largo plazo de un caso no se mezcle con la de otro.
- Preparación: algunos casos (memoria-01) necesitan historial previo del mismo
  id_usuario. Antes de la primera ejecución pendiente del caso se corre su
  consulta de preparación, registrada como caso `<id>-prep` (origen
  "evaluacion"); las métricas la excluyen. Si la preparación falla, el resto de
  ese caso se posterga para la próxima corrida: ejecutarlo sin historial daría
  un dato inválido.
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

from observabilidad.casos import SUFIJO_PREPARACION, preguntas_del_caso
from observabilidad.medicion import tipo_de_error
from observabilidad.privacidad import enmascarar

PREFIJO_ID_USUARIO = "eval"
ORDENES = ("repeticion", "prioridad")
# Consultas de preparación cuando el caso no define `preparacion` en casos.json.
PREPARACION_POR_DEFECTO = {"memoria-01": ("¿Puedo comprar alcohol con la BAES?",)}

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
    id_usuario: str = "evaluacion"
    principal: bool = True  # la pregunta principal del caso, no una variante
    es_preparacion: bool = False

    @property
    def caso_base(self) -> str:
        """El caso al que pertenece; para una preparación, el caso que prepara."""
        return self.caso_id.removesuffix(SUFIJO_PREPARACION) if self.es_preparacion else self.caso_id

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


def id_usuario_del_caso(caso_id: str, prefijo: str = PREFIJO_ID_USUARIO) -> str:
    return f"{prefijo}-{caso_id}"


def preparacion_de(caso: dict) -> list[str]:
    """Consultas de preparación del caso: las de su campo `preparacion` (una
    lista vacía las desactiva) o, si no lo define, las de PREPARACION_POR_DEFECTO."""
    propia = caso.get("preparacion")
    if propia is not None:
        return list(propia)
    return list(PREPARACION_POR_DEFECTO.get(caso["id"], ()))


def _prioridad(combo: Combinacion) -> tuple[int, int, int]:
    if combo.repeticion == 1:
        return (0 if combo.principal else 1, 0, 0)
    return (2, combo.repeticion, 0 if combo.principal else 1)


def combinaciones(
    casos: list[dict],
    repeticiones: int,
    *,
    solo_principal: bool = False,
    orden: str = "repeticion",
    prefijo_id: str = PREFIJO_ID_USUARIO,
) -> list[Combinacion]:
    """Las combinaciones (caso, pregunta, repetición) en el orden pedido.
    `solo_principal` deja fuera las variantes. No incluye las preparaciones."""
    if orden not in ORDENES:
        raise ValueError(f"orden inválido: {orden!r} (debe ser uno de {ORDENES})")
    todas = []
    for repeticion in range(1, repeticiones + 1):
        for caso in casos:
            preguntas = preguntas_del_caso(caso)
            for indice, texto in enumerate(preguntas[:1] if solo_principal else preguntas):
                todas.append(
                    Combinacion(
                        caso["id"],
                        texto,
                        repeticion,
                        id_usuario=id_usuario_del_caso(caso["id"], prefijo_id),
                        principal=(indice == 0),
                    )
                )
    return sorted(todas, key=_prioridad) if orden == "prioridad" else todas


def agregar_preparacion(
    pendientes: list[Combinacion], casos: list[dict], hechas: set[tuple[str, str, int]]
) -> list[Combinacion]:
    """Inserta la preparación de cada caso justo antes de su primera ejecución
    pendiente, salvo las que ya están hechas en la base."""
    por_id = {c["id"]: c for c in casos}
    resultado: list[Combinacion] = []
    vistos: set[str] = set()
    for combo in pendientes:
        if combo.caso_id not in vistos:
            vistos.add(combo.caso_id)
            for texto in preparacion_de(por_id[combo.caso_id]):
                prep = Combinacion(
                    f"{combo.caso_id}{SUFIJO_PREPARACION}",
                    texto,
                    1,
                    id_usuario=combo.id_usuario,
                    es_preparacion=True,
                )
                if prep.clave not in hechas:
                    resultado.append(prep)
        resultado.append(combo)
    return resultado


@dataclass
class Plan:
    pendientes: list[Combinacion]  # con las preparaciones incluidas
    total: int                     # combinaciones de los casos (sin preparaciones)
    saltadas: int                  # de ese total, las que ya estaban en la base
    preparaciones: int             # preparaciones pendientes incluidas en `pendientes`


def planificar(
    casos: list[dict],
    repeticiones: int,
    hechas: set[tuple[str, str, int]],
    *,
    solo_principal: bool = False,
    orden: str = "repeticion",
    prefijo_id: str = PREFIJO_ID_USUARIO,
) -> Plan:
    todas = combinaciones(casos, repeticiones, solo_principal=solo_principal, orden=orden, prefijo_id=prefijo_id)
    reales, saltadas = separar_pendientes(todas, hechas)
    con_prep = agregar_preparacion(reales, casos, hechas)
    return Plan(con_prep, len(todas), saltadas, len(con_prep) - len(reales))


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
    dormir: Callable[[float], None] = time.sleep,
    imprimir: Callable[[str], None] = print,
    max_429_seguidos: int = MAX_429_SEGUIDOS,
    espera_maxima: float = ESPERA_MAXIMA_S,
    espera_por_defecto: float = ESPERA_POR_DEFECTO_S,
) -> ResultadoCorrida:
    resultado = ResultadoCorrida(saltadas=saltadas)
    seguidos_429 = 0
    total = len(pendientes)
    bloqueados: set[str] = set()  # casos cuya preparación falló en esta corrida

    def detener(motivo: str, detalle: str) -> None:
        resultado.detenida_por, resultado.detalle = motivo, detalle

    try:
        for numero, combo in enumerate(pendientes, start=1):
            etiqueta = f"[{numero}/{total}] {combo.caso_id} rep {combo.repeticion}"
            if combo.caso_base in bloqueados and not combo.es_preparacion:
                imprimir(f"{etiqueta}: pospuesta, falló la preparación de {combo.caso_base}")
                continue
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
                        combo.id_usuario,
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
                        if combo.es_preparacion:
                            bloqueados.add(combo.caso_base)
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
                        if combo.es_preparacion:
                            bloqueados.add(combo.caso_base)
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
