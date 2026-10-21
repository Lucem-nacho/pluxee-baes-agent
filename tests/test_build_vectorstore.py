"""Pruebas de `construir_indices` (rag/indexing/build_vectorstore.py): la
colección debe reflejar exactamente lo que hay HOY en las fuentes.

Reproduce un defecto real: con `upsert`, retirar un fragmento de una fuente
no lo quitaba del índice, y agregar uno al principio corría los ids
(`tipo-N`, N global) dejando huérfanos los viejos — se llegó a tener el mismo
tweet triplicado en el índice vivo. Usa el modelo de embeddings local real
(sin red de Gemini) sobre una carpeta Chroma temporal.
"""

import chromadb
import pytest

from config import settings
from rag.indexing import build_vectorstore


def _doc(tipo, texto):
    return {"texto": texto, "fuente": "prueba", "fecha": "2026-01", "tipo_fragmento": tipo}


def _docs(*textos):
    return [_doc("social", t) for t in textos]


def _contenido_del_indice(ruta):
    coleccion = chromadb.PersistentClient(path=str(ruta)).get_collection(
        settings.COLLECTION_MANUAL_FAQ_SOCIAL
    )
    return sorted(coleccion.get()["documents"])


@pytest.fixture
def indice_temporal(tmp_path, monkeypatch):
    ruta = tmp_path / "chroma"
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIR", str(ruta))
    return ruta


def _construir_con(monkeypatch, documentos):
    monkeypatch.setattr(
        build_vectorstore, "preparar_documentos_manual_faq_social", lambda: documentos
    )
    build_vectorstore.construir_indices()


def test_primera_corrida_crea_la_coleccion(indice_temporal, monkeypatch):
    _construir_con(monkeypatch, _docs("uno", "dos"))
    assert _contenido_del_indice(indice_temporal) == ["dos", "uno"]


def test_un_fragmento_retirado_de_las_fuentes_sale_del_indice(indice_temporal, monkeypatch):
    _construir_con(monkeypatch, _docs("uno", "dos", "tres"))
    _construir_con(monkeypatch, _docs("uno", "dos"))
    assert _contenido_del_indice(indice_temporal) == ["dos", "uno"]


def test_agregar_una_entrada_de_un_tipo_anterior_no_deja_duplicados_huerfanos(
    indice_temporal, monkeypatch
):
    # Caso real: agregar una entrada de FAQ corría los ids de los tweets
    # (social-1, social-2 -> social-2, social-3) y social-1 quedaba huérfano.
    _construir_con(monkeypatch, [_doc("faq", "f1"), _doc("social", "uno"), _doc("social", "dos")])
    _construir_con(
        monkeypatch,
        [_doc("faq", "f1"), _doc("faq", "f2"), _doc("social", "uno"), _doc("social", "dos")],
    )
    assert _contenido_del_indice(indice_temporal) == ["dos", "f1", "f2", "uno"]
