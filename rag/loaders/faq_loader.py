"""Carga de la FAQ (interna, semi-estática).

A diferencia del manual/resolución (`rag/loaders/pdf_loader.py`, donde
`fuente`/`fecha` son constantes fijas por documento), cada entrada de
data/faq/faq_baes.json trae su propia metadata `fuente`/`fecha`: los 8 pares
actuales citan el manual como origen (son paráfrasis suyas), pero una FAQ es
semi-estática por diseño y una entrada futura podría citar otra fuente o una
fecha de actualización distinta. El loader respeta lo que viene en el JSON
en vez de hardcodear un valor.

NO pasa por rag/indexing/chunking.py: cada par pregunta-respuesta ya es un
fragmento atómico, tal como especifica la arquitectura.
"""

import json
from pathlib import Path

FAQ_PATH = Path(__file__).resolve().parents[2] / "data" / "faq" / "faq_baes.json"

CAMPOS_REQUERIDOS = {"pregunta", "respuesta", "fuente", "fecha"}


def cargar_faq(path: str | Path = FAQ_PATH) -> list[dict]:
    """Devuelve un fragmento por cada par pregunta-respuesta, con
    texto = "Pregunta: ...\\nRespuesta: ..." (se embebe el par completo, no
    solo la pregunta, para que el contexto recuperado incluya la respuesta) y
    la metadata (`fuente`, `fecha`) tal como viene en el JSON."""
    with open(path, encoding="utf-8") as f:
        entradas = json.load(f)

    fragmentos = []
    for i, entrada in enumerate(entradas):
        faltantes = CAMPOS_REQUERIDOS - entrada.keys()
        if faltantes:
            raise ValueError(f"Entrada {i} de {path} sin campos requeridos: {faltantes}")
        fragmentos.append({
            "texto": f"Pregunta: {entrada['pregunta']}\nRespuesta: {entrada['respuesta']}",
            "fuente": entrada["fuente"],
            "fecha": entrada["fecha"],
            "faq_index": i,
        })
    return fragmentos
