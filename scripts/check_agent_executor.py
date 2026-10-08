"""Smoke test manual del AgentExecutor de EP2 — confirma que el agente
completo (clasificador -> tools -> respuesta) funciona end-to-end con
llamadas reales a Gemini.

IMPORTANTE, a diferencia de `scripts/check_llm_connection.py`: este script
NO se pudo ejecutar durante el desarrollo porque el entorno donde se
escribió el código (sandbox en la nube) no tenía acceso a internet para
instalar `langchain`/`langchain-google-genai`. Corre esto ANTES de confiar
en que `agent/orchestrator.py` y `agent/tools.py` compilan y funcionan de
verdad — si algo falla, el error va a decir exactamente qué línea revisar.

Tampoco es un test de pytest, por la misma razón que el script de EP1: hace
llamadas de red reales y consume cuota, así que no debe correr en una suite
automatizada por defecto.

Cada escenario puede gastar varias llamadas a Gemini (una por cada tool que
el agente decida llamar, más la respuesta final) — con el free tier a 5
req/min / 20 req/día, corre UN escenario a la vez (ver `--escenario`) en vez
de todos de una.

Uso:
    python scripts/check_agent_executor.py --escenario normativa
    python scripts/check_agent_executor.py --escenario comercio
    python scripts/check_agent_executor.py --escenario escalamiento
    python scripts/check_agent_executor.py --escenario memoria

Requiere GOOGLE_API_KEY en .env y el índice Chroma ya construido
(`python scripts/ingest_all.py`).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.orchestrator import responder

ESCENARIOS = {
    "normativa": {
        "descripcion": "Pregunta normativa general — debería llamar buscar_normativa y nada más.",
        "consulta": "¿Puedo comprar bebidas energéticas con mi BAES?",
    },
    "comercio": {
        "descripcion": "Pregunta sobre un comercio puntual — debería llamar buscar_comercio (y "
        "probablemente también buscar_normativa, para cruzar restricciones generales con las "
        "del comercio).",
        "consulta": "¿Puedo comprar alcohol en el Jumbo de Providencia con la BAES?",
    },
    "escalamiento": {
        "descripcion": "Reclamo de cobro — el clasificador debería derivar ANTES de llegar al "
        "agente (sin gastar llamadas de tool-calling).",
        "consulta": "Me cobraron dos veces en el mismo local, quiero que me devuelvan la plata.",
    },
    "memoria": {
        "descripcion": "Pregunta sobre el historial — debería llamar consultar_historial. Corre "
        "primero el escenario 'normativa' con el mismo --id-usuario para que haya algo que "
        "encontrar.",
        "consulta": "¿Qué te pregunté la última vez?",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--escenario", choices=sorted(ESCENARIOS), required=True)
    parser.add_argument("--id-usuario", default="smoke-test")
    parser.add_argument(
        "--historial",
        default="",
        help="Historial de conversación previo (string plano, opcional).",
    )
    args = parser.parse_args()

    escenario = ESCENARIOS[args.escenario]
    print(f"Escenario: {args.escenario} — {escenario['descripcion']}")
    print(f"Consulta: {escenario['consulta']!r}")
    print(f"id_usuario: {args.id_usuario!r}\n")

    resultado = responder(escenario["consulta"], args.historial, args.id_usuario)

    print(f"Estado: {resultado['estado']}")
    if resultado["estado"] == "INFORMATIVA":
        print(f"Pasos del agente: {resultado.get('pasos', [])}")
        print(f"Respuesta: {resultado['respuesta']!r}")
    else:
        print(f"Justificación: {resultado['justificacion_clasificador']!r}")
        print(f"Registro #{resultado.get('id_escalamiento')}")

    print("\nOK — revisa que 'Pasos del agente' tenga sentido para el escenario elegido.")


if __name__ == "__main__":
    main()
