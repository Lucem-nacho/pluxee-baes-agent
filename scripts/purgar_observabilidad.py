"""Borra de la base de observabilidad las ejecuciones con más de N días (y
sus pasos). Pensado para correrse a mano o desde un programador de tareas.

Uso:
    python scripts/purgar_observabilidad.py            # más de 30 días
    python scripts/purgar_observabilidad.py --dias 7

La base es la de `settings.OBSERVABILIDAD_DB_PATH`
(`data/processed/observabilidad.db` por defecto).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from observabilidad.store import purgar_anteriores_a

DIAS_POR_DEFECTO = 30


def _dias(valor: str) -> int:
    dias = int(valor)
    if dias < 0:
        raise argparse.ArgumentTypeError("--dias no puede ser negativo")
    return dias


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dias",
        type=_dias,
        default=DIAS_POR_DEFECTO,
        help=f"Se borran los registros con más de estos días (por defecto {DIAS_POR_DEFECTO}).",
    )
    args = parser.parse_args()

    borradas = purgar_anteriores_a(args.dias)
    print(
        f"Purga de {settings.OBSERVABILIDAD_DB_PATH}: {borradas} ejecuciones "
        f"con más de {args.dias} días borradas (junto con sus pasos)."
    )


if __name__ == "__main__":
    main()
