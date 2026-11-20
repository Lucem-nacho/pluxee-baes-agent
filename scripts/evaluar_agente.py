"""Corre los casos de eval/casos.json contra el agente real (Gemini) y deja
cada ejecución en la base de observabilidad, marcada con origen="evaluacion"
y con su caso, variante y repetición.

CUOTA: una ejecución informativa usa unas 3-4 solicitudes a Gemini y el nivel
gratuito permite alrededor de 20 por día, así que una evaluación completa
toma varios días. El script es reanudable: vuelve a correrlo y salta lo que
ya está en la base. Acota cada corrida con --limite, y gasta la cuota en orden
de prioridad con --solo-principal y --orden.

Uso:
    python scripts/evaluar_agente.py --solo-principal --orden --repeticiones 1
    python scripts/evaluar_agente.py --orden --repeticiones 3 --limite 5
    python scripts/evaluar_agente.py --casos normativa-02,memoria-01
    python scripts/evaluar_agente.py --incluir-ejemplos --repeticiones 1 --limite 1

Orden:
    (por defecto)   por repetición: todas las preguntas con repetición 1, luego 2...
    --orden         por prioridad: las preguntas principales de todos los casos,
                    luego las variantes, y al final las repeticiones.

Cada caso corre con su propio id_usuario fijo (eval-<caso_id>). Los casos que
necesitan historial (memoria-01) ejecutan antes una consulta de preparación,
registrada como "<caso>-prep" y excluida de las métricas.

Ante un 429 espera el retryDelay y reintenta una vez; con 3 seguidos se
detiene limpio e imprime cuántas combinaciones faltan. Los casos marcados
"ejemplo" se ignoran salvo con --incluir-ejemplos.

IMPORTANTE: la corrida usa el modelo de GEMINI_MODEL (se imprime al empezar) y
cada ejecución lo registra. Si cambias de modelo entre corridas, las métricas
se reportan también por modelo; la reanudación NO distingue modelos: una
combinación ya hecha con otro modelo se salta.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from observabilidad import store
from observabilidad.casos import ErrorEsquema, cargar_casos
from observabilidad.eval_agente import ORDENES, PREFIJO_ID_USUARIO, ejecutar, planificar


def _entero_positivo(texto: str) -> int:
    valor = int(texto)
    if valor < 1:
        raise argparse.ArgumentTypeError("debe ser 1 o mayor")
    return valor


def _no_negativo(texto: str) -> float:
    valor = float(texto)
    if valor < 0:
        raise argparse.ArgumentTypeError("no puede ser negativo")
    return valor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeticiones", type=_entero_positivo, default=3, help="Repeticiones por pregunta (por defecto 3).")
    parser.add_argument("--limite", type=_entero_positivo, default=None, help="Máximo de ejecuciones (llamadas a responder) en esta corrida.")
    parser.add_argument("--pausa", type=_no_negativo, default=15.0, help="Segundos entre llamadas (por defecto 15).")
    parser.add_argument("--casos", default=None, help="Ids separados por coma para filtrar.")
    parser.add_argument("--solo-principal", action="store_true", help="Corre solo la pregunta principal de cada caso, sin variantes.")
    parser.add_argument(
        "--orden", nargs="?", const="prioridad", default="repeticion", choices=ORDENES,
        help="'--orden' (o '--orden prioridad'): principales, luego variantes, luego repeticiones. Por defecto: por repetición.",
    )
    parser.add_argument("--incluir-ejemplos", action="store_true", help="Incluye los casos marcados como ejemplo.")
    parser.add_argument("--casos-path", default=None, help="Ruta de casos.json (por defecto eval/casos.json).")
    parser.add_argument("--prefijo-id", default=PREFIJO_ID_USUARIO, help=f"Prefijo del id_usuario de cada caso (por defecto '{PREFIJO_ID_USUARIO}', da eval-<caso_id>).")
    args = parser.parse_args()

    print(f"Modelo activo: {settings.GEMINI_MODEL} (proveedor: {settings.LLM_PROVIDER})")

    avisos: list[str] = []
    try:
        casos = cargar_casos(args.casos_path, incluir_ejemplos=args.incluir_ejemplos, advertencias=avisos)
    except ErrorEsquema as error:
        print(error, file=sys.stderr)
        return 2
    for aviso in avisos:
        print(f"Aviso: {aviso}", file=sys.stderr)
    if args.casos:
        pedidos = {c.strip() for c in args.casos.split(",") if c.strip()}
        desconocidos = pedidos - {c["id"] for c in casos}
        if desconocidos:
            print(f"Ids que no existen (o son ejemplos sin --incluir-ejemplos): {sorted(desconocidos)}", file=sys.stderr)
            return 2
        casos = [c for c in casos if c["id"] in pedidos]
    if not casos:
        print("No hay casos para evaluar. Los casos marcados como ejemplo se ignoran: agrega los tuyos o usa --incluir-ejemplos.")
        return 0

    plan = planificar(
        casos, args.repeticiones, store.combinaciones_evaluadas(),
        solo_principal=args.solo_principal, orden=args.orden, prefijo_id=args.prefijo_id,
    )
    reales = len(plan.pendientes) - plan.preparaciones
    print(
        f"{len(casos)} caso(s), {args.repeticiones} repetición(es)"
        f"{', solo la pregunta principal' if args.solo_principal else ''}, orden {args.orden}: "
        f"{plan.total} combinaciones, {plan.saltadas} ya estaban en la base, {reales} pendientes"
        + (f" (más {plan.preparaciones} consulta(s) de preparación)." if plan.preparaciones else ".")
    )
    if not plan.pendientes:
        print("Nada que hacer.")
        return 0
    if args.limite is None:
        print(
            f"Aviso: ~3-4 solicitudes a Gemini por ejecución y ~20 por día en el nivel gratuito: "
            f"{len(plan.pendientes)} ejecuciones son ~{len(plan.pendientes) * 3}-{len(plan.pendientes) * 4} solicitudes. Considera --limite."
        )

    from agent.orchestrator import responder  # import tardío: carga LangChain y el índice

    resultado = ejecutar(plan.pendientes, responder, saltadas=plan.saltadas, limite=args.limite, pausa=args.pausa)

    print("\n--- Resumen ---")
    print(f"Modelo: {settings.GEMINI_MODEL}")
    print(f"Completadas: {resultado.completadas} | fallidas: {resultado.fallidas} | saltadas (ya hechas): {resultado.saltadas} | llamadas a responder: {resultado.llamadas}")
    if resultado.detenida_por:
        print(f"Detenida: {resultado.detalle}.")
    print(f"Faltan {resultado.faltan} de {len(plan.pendientes)} ejecuciones pendientes. Vuelve a correr el script para continuar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
