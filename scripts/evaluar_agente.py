"""Corre los casos de eval/casos.json contra el agente real (Gemini) y deja
cada ejecución en la base de observabilidad, marcada con origen="evaluacion"
y con su caso, variante y repetición.

CUOTA: una ejecución informativa usa unas 3-4 solicitudes a Gemini y el nivel
gratuito permite alrededor de 20 por día, así que una evaluación completa
toma varios días. El script es reanudable: vuelve a correrlo y salta lo que
ya está en la base. Acota cada corrida con --limite.

Uso:
    python scripts/evaluar_agente.py --limite 5
    python scripts/evaluar_agente.py --repeticiones 3 --pausa 15
    python scripts/evaluar_agente.py --casos id1,id2
    python scripts/evaluar_agente.py --incluir-ejemplos --repeticiones 1 --limite 1

Ante un 429 espera el retryDelay y reintenta una vez; con 3 seguidos se
detiene limpio e imprime cuántas combinaciones faltan. Los casos marcados
"ejemplo" se ignoran salvo con --incluir-ejemplos.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from observabilidad import store
from observabilidad.casos import ErrorEsquema, cargar_casos
from observabilidad.eval_agente import combinaciones, ejecutar, separar_pendientes


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
    parser.add_argument("--incluir-ejemplos", action="store_true", help="Incluye los casos marcados como ejemplo.")
    parser.add_argument("--casos-path", default=None, help="Ruta de casos.json (por defecto eval/casos.json).")
    parser.add_argument("--id-usuario", default="evaluacion", help="id_usuario con el que se ejecuta (por defecto 'evaluacion').")
    args = parser.parse_args()

    try:
        casos = cargar_casos(args.casos_path, incluir_ejemplos=args.incluir_ejemplos)
    except ErrorEsquema as error:
        print(error, file=sys.stderr)
        return 2
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

    todas = combinaciones(casos, args.repeticiones)
    pendientes, saltadas = separar_pendientes(todas, store.combinaciones_evaluadas())
    print(f"{len(casos)} caso(s), {args.repeticiones} repetición(es): {len(todas)} combinaciones, {saltadas} ya estaban en la base, {len(pendientes)} pendientes.")
    if not pendientes:
        print("Nada que hacer.")
        return 0
    if args.limite is None:
        print(f"Aviso: ~3-4 solicitudes a Gemini por ejecución y ~20 por día en el nivel gratuito; considera --limite (hoy alcanza para unas {20 // 4}-{20 // 3} ejecuciones).")

    from agent.orchestrator import responder  # import tardío: carga LangChain y el índice

    resultado = ejecutar(
        pendientes, responder, saltadas=saltadas, limite=args.limite, pausa=args.pausa, id_usuario=args.id_usuario
    )

    print("\n--- Resumen ---")
    print(f"Completadas: {resultado.completadas} | fallidas: {resultado.fallidas} | saltadas (ya hechas): {resultado.saltadas} | llamadas a responder: {resultado.llamadas}")
    if resultado.detenida_por:
        print(f"Detenida: {resultado.detalle}.")
    print(f"Faltan {resultado.faltan} de {len(pendientes)} combinaciones pendientes. Vuelve a correr el script para continuar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
