"""Conjunto de evaluación (`eval/casos.json`): carga y validación del esquema.

Cada caso es una pregunta (con variantes de redacción) y lo que se espera que
haga el agente. Los scripts de evaluación leen los casos desde acá, así que un
casos.json mal escrito falla temprano y con un mensaje que dice QUÉ caso y QUÉ
campo corregir, en vez de producir métricas silenciosamente equivocadas.

Forma del archivo: un objeto con la clave "casos" (lista). Cualquier otra
clave de primer nivel (p. ej. "esquema", que documenta los campos) se ignora.

Dos niveles de problema:
- ERRORES (`validar_casos`, `ErrorEsquema`): el archivo no se puede usar
  (campo faltante, valor fuera de los permitidos, tipo equivocado, id repetido).
- AVISOS (`advertencias_de_casos`): el archivo se usa igual, pero algo merece
  una segunda mirada (un campo que no existe en el esquema, o una combinación
  de campos poco coherente). Las combinaciones válidas dependen de la categoría
  del caso (p. ej. un caso de derivación admite dos caminos), así que no se
  bloquea por ellas.
"""

import json
import re
import unicodedata
from pathlib import Path

RUTA_CASOS = Path(__file__).resolve().parents[1] / "eval" / "casos.json"

CATEGORIAS_CASO = ("normativa", "comercio", "derivacion", "memoria")
CATEGORIAS_ESPERADAS = ("INFORMATIVA", "ESCALAR")
VEREDICTOS = ("permitido", "prohibido", "no_confirmable", "derivar", "no_aplica")
# Las 4 herramientas de agent/tools.py.
HERRAMIENTAS = ("buscar_normativa", "buscar_comercio", "consultar_historial", "escalar_a_ejecutivo")

CAMPOS_OBLIGATORIOS = (
    "id",
    "categoria_caso",
    "pregunta",
    "variantes",
    "categoria_esperada",
    "herramientas_esperadas",
    "veredicto_esperado",
)
# `fuente_esperada` puede faltar, ser null o ser texto vacío: el caso simplemente
# no se evalúa en recuperación. `preparacion` son consultas que se ejecutan antes
# del caso con el mismo id_usuario (ver eval_agente.py).
CAMPOS_OPCIONALES = ("fuente_esperada", "fuentes_alternativas", "ejemplo", "notas", "preparacion")

# Los ids que terminan así quedan reservados para las ejecuciones de preparación.
SUFIJO_PREPARACION = "-prep"

_ID_VALIDO = re.compile(r"[A-Za-z0-9_-]+")


class ErrorEsquema(ValueError):
    """El archivo de casos no cumple el esquema. `errores` lista todos los
    problemas encontrados (no solo el primero)."""

    def __init__(self, errores: list[str], origen: str = "casos"):
        self.errores = list(errores)
        detalle = "\n".join(f"  - {e}" for e in self.errores)
        super().__init__(f"{origen} no cumple el esquema ({len(self.errores)} problema(s)):\n{detalle}")


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return " ".join(sin_tildes.casefold().split())


def _opciones(valores: tuple[str, ...]) -> str:
    return ", ".join(valores)


def _nombre(caso: dict, posicion: int) -> str:
    id_caso = caso.get("id")
    if isinstance(id_caso, str) and id_caso:
        return f'caso #{posicion} (id "{id_caso}")'
    return f"caso #{posicion} (sin id)"


def _validar_caso(caso: object, posicion: int, ids_vistos: set[str]) -> list[str]:
    if not isinstance(caso, dict):
        return [f"caso #{posicion}: debe ser un objeto JSON, no {type(caso).__name__}"]

    nombre = _nombre(caso, posicion)
    id_caso = caso.get("id")
    errores: list[str] = []

    for campo in CAMPOS_OBLIGATORIOS:
        if campo not in caso:
            errores.append(f"{nombre}: falta el campo '{campo}'")

    # --- id
    if "id" in caso:
        if not isinstance(id_caso, str) or not _ID_VALIDO.fullmatch(id_caso):
            errores.append(f"{nombre}: 'id' debe ser texto sin espacios, solo letras, números, '_' o '-' (llegó {id_caso!r})")
        elif id_caso.endswith(SUFIJO_PREPARACION):
            errores.append(f"{nombre}: los ids que terminan en '{SUFIJO_PREPARACION}' están reservados para las consultas de preparación")
        elif id_caso in ids_vistos:
            errores.append(f"{nombre}: 'id' repetido; cada caso necesita un id único")
        else:
            ids_vistos.add(id_caso)

    # --- campos de texto con valores permitidos
    for campo, validos in (
        ("categoria_caso", CATEGORIAS_CASO),
        ("categoria_esperada", CATEGORIAS_ESPERADAS),
        ("veredicto_esperado", VEREDICTOS),
    ):
        if campo in caso and caso[campo] not in validos:
            errores.append(f"{nombre}: '{campo}' = {caso[campo]!r} no es válido; debe ser uno de: {_opciones(validos)}")

    # --- pregunta y variantes
    pregunta = caso.get("pregunta")
    if "pregunta" in caso and (not isinstance(pregunta, str) or not pregunta.strip()):
        errores.append(f"{nombre}: 'pregunta' debe ser un texto no vacío")
    if "variantes" in caso:
        variantes = caso["variantes"]
        if not isinstance(variantes, list):
            errores.append(f"{nombre}: 'variantes' debe ser una lista de textos (puede ser vacía [])")
        else:
            vistas = {_normalizar(pregunta)} if isinstance(pregunta, str) and pregunta.strip() else set()
            for i, variante in enumerate(variantes, start=1):
                if not isinstance(variante, str) or not variante.strip():
                    errores.append(f"{nombre}: la variante #{i} debe ser un texto no vacío")
                elif _normalizar(variante) in vistas:
                    errores.append(f"{nombre}: la variante #{i} repite la pregunta original u otra variante: {variante!r}")
                else:
                    vistas.add(_normalizar(variante))

    # --- herramientas
    if "herramientas_esperadas" in caso:
        herramientas = caso["herramientas_esperadas"]
        if not isinstance(herramientas, list):
            errores.append(f"{nombre}: 'herramientas_esperadas' debe ser una lista (puede ser vacía [])")
        else:
            for h in herramientas:
                if h not in HERRAMIENTAS:
                    errores.append(f"{nombre}: herramienta {h!r} desconocida en 'herramientas_esperadas'; debe ser una de: {_opciones(HERRAMIENTAS)}")
            if len(set(map(str, herramientas))) != len(herramientas):
                errores.append(f"{nombre}: 'herramientas_esperadas' tiene herramientas repetidas; se compara como conjunto, lístalas una vez")

    # --- opcionales
    fuente = caso.get("fuente_esperada")
    if fuente is not None and not isinstance(fuente, str):
        errores.append(f"{nombre}: 'fuente_esperada' debe ser un texto, o vacío/null si el caso no se evalúa en recuperación")
    if "fuentes_alternativas" in caso:
        alternativas = caso["fuentes_alternativas"]
        if not isinstance(alternativas, list) or not all(isinstance(a, str) and a.strip() for a in alternativas):
            errores.append(f"{nombre}: 'fuentes_alternativas' debe ser una lista de frases (textos no vacíos); puede ser []")
    if "preparacion" in caso:
        preparacion = caso["preparacion"]
        if not isinstance(preparacion, list) or not all(isinstance(p, str) and p.strip() for p in preparacion):
            errores.append(f"{nombre}: 'preparacion' debe ser una lista de textos no vacíos")
    if "ejemplo" in caso and not isinstance(caso["ejemplo"], bool):
        errores.append(f"{nombre}: 'ejemplo' debe ser true o false")
    if "notas" in caso and not isinstance(caso["notas"], str):
        errores.append(f"{nombre}: 'notas' debe ser un texto")

    return errores


def _lista_de_casos(datos: object) -> list[dict]:
    if isinstance(datos, dict) and isinstance(datos.get("casos"), list):
        return [c for c in datos["casos"] if isinstance(c, dict)]
    return []


def validar_casos(datos: object) -> list[str]:
    """Devuelve la lista de ERRORES del contenido ya parseado (vacía si es válido)."""
    if not isinstance(datos, dict) or "casos" not in datos:
        return ["el archivo debe ser un objeto JSON con la clave 'casos' (una lista de casos)"]
    casos = datos["casos"]
    if not isinstance(casos, list):
        return ["'casos' debe ser una lista"]
    if not casos:
        return ["'casos' está vacío: agrega al menos un caso"]

    errores: list[str] = []
    ids_vistos: set[str] = set()
    for posicion, caso in enumerate(casos, start=1):
        errores.extend(_validar_caso(caso, posicion, ids_vistos))
    return errores


def advertencias_de_casos(datos: object) -> list[str]:
    """AVISOS sobre un contenido ya parseado: campos que no existen en el
    esquema y combinaciones de campos poco coherentes. No impiden usar el archivo."""
    avisos: list[str] = []
    conocidos = CAMPOS_OBLIGATORIOS + CAMPOS_OPCIONALES
    for posicion, caso in enumerate(_lista_de_casos(datos), start=1):
        nombre = _nombre(caso, posicion)
        for campo in caso:
            if campo not in conocidos:
                avisos.append(
                    f"{nombre}: el campo '{campo}' no existe en el esquema y se ignora "
                    f"(¿error de tipeo? campos válidos: {_opciones(conocidos)})"
                )

        fuente = caso.get("fuente_esperada")
        if caso.get("fuentes_alternativas") and not (isinstance(fuente, str) and fuente.strip()):
            avisos.append(
                f"{nombre}: tiene 'fuentes_alternativas' pero no 'fuente_esperada'; el caso se omite de la "
                "evaluación de recuperación y las alternativas no se usan"
            )

        categoria = caso.get("categoria_esperada")
        veredicto = caso.get("veredicto_esperado")
        herramientas = caso.get("herramientas_esperadas")
        if not isinstance(herramientas, list) or any(h not in HERRAMIENTAS for h in herramientas):
            continue  # ya es un error: no tiene sentido comparar
        es_derivacion = caso.get("categoria_caso") == "derivacion"

        if categoria == "ESCALAR" and herramientas:
            avisos.append(f"{nombre}: con 'categoria_esperada' = ESCALAR el agente no llega a correr, así que 'herramientas_esperadas' debería ser []")
        if categoria == "ESCALAR" and veredicto in VEREDICTOS and veredicto not in ("derivar", "no_aplica"):
            avisos.append(f"{nombre}: con 'categoria_esperada' = ESCALAR el 'veredicto_esperado' debería ser 'derivar' o 'no_aplica' (llegó {veredicto!r})")
        if (
            veredicto == "derivar"
            and categoria == "INFORMATIVA"
            and "escalar_a_ejecutivo" not in herramientas
            and not es_derivacion
        ):
            avisos.append(f"{nombre}: 'veredicto_esperado' = derivar con 'categoria_esperada' = INFORMATIVA solo ocurre si el agente llama a 'escalar_a_ejecutivo'; considera agregarla a 'herramientas_esperadas'")
        if "escalar_a_ejecutivo" in herramientas and veredicto in VEREDICTOS and veredicto not in ("derivar", "no_aplica"):
            avisos.append(f"{nombre}: si se espera 'escalar_a_ejecutivo', el 'veredicto_esperado' debería ser 'derivar' o 'no_aplica' (llegó {veredicto!r})")
    return avisos


def cargar_casos(
    ruta: str | Path | None = None,
    incluir_ejemplos: bool = False,
    advertencias: list[str] | None = None,
) -> list[dict]:
    """Lee y valida el archivo de casos. Por defecto se descartan los marcados
    con `"ejemplo": true` (los scripts de evaluación los ignoran salvo con
    `--incluir-ejemplos`). Lanza `ErrorEsquema` si el archivo es inválido. Si se
    pasa una lista en `advertencias`, se le agregan los avisos (ver arriba).

    `fuente_esperada` queda siempre presente: None cuando falta, es null o es
    texto en blanco. `fuentes_alternativas` queda siempre presente: lista
    (vacía cuando falta)."""
    ruta = Path(ruta or RUTA_CASOS)
    origen = str(ruta)
    try:
        texto = ruta.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise ErrorEsquema([f"no existe el archivo {ruta}"], origen) from None
    try:
        datos = json.loads(texto)
    except json.JSONDecodeError as error:
        raise ErrorEsquema(
            [f"no es JSON válido: {error.msg} (línea {error.lineno}, columna {error.colno})"], origen
        ) from None

    errores = validar_casos(datos)
    if errores:
        raise ErrorEsquema(errores, origen)
    if advertencias is not None:
        advertencias.extend(advertencias_de_casos(datos))

    casos = datos["casos"]
    for caso in casos:
        fuente = caso.get("fuente_esperada")
        caso["fuente_esperada"] = fuente if isinstance(fuente, str) and fuente.strip() else None
        caso["fuentes_alternativas"] = list(caso.get("fuentes_alternativas") or [])
    return casos if incluir_ejemplos else [c for c in casos if not c.get("ejemplo")]


def preguntas_del_caso(caso: dict) -> list[str]:
    """La pregunta original seguida de sus variantes: todos los textos que se
    le envían al agente (o al recuperador) para este caso."""
    return [caso["pregunta"], *caso["variantes"]]
