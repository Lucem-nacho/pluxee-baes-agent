"""Utilidades de texto compartidas por el pipeline RAG."""

import re


def normalizar_para_embedding(texto: str) -> str:
    """Colapsa saltos de línea y espacios múltiples a uno solo.

    Se usa SOLO para calcular embeddings — nunca para lo que se cita o
    muestra al usuario final, que conserva el `texto` original tal cual sale
    del loader/chunker.

    Nota: se probó empíricamente como posible causa de la baja similitud
    entre una FAQ y su fila real en la tabla del manual (ver limitación de
    rag/arbitration.py en docs/arquitectura.md) y NO tuvo ningún efecto —
    el tokenizer de sentence-transformers/all-MiniLM-L6-v2 ya colapsa
    espacios/saltos de línea internamente antes de tokenizar. La causa real
    de ese caso fue otra (chunking de tablas, ver rag/indexing/chunking.py).
    Se deja esta función de todas formas como higiene general de texto antes
    de vectorizar, sin depender de que este modelo en particular ya lo haga.
    """
    return re.sub(r"\s+", " ", texto).strip()
