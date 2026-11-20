"""Aislamiento común de las pruebas."""

from pathlib import Path

import pytest

from config import settings


@pytest.fixture(autouse=True)
def db_observabilidad(tmp_path, monkeypatch) -> Path:
    """Redirige la base de observabilidad a una carpeta temporal en TODAS las
    pruebas. `agent.orchestrator.responder` registra cada ejecución, así que
    sin esto cualquier prueba que lo llame escribiría en
    `data/processed/observabilidad.db`. La sal del hash se guarda junto a la
    base, así que también queda en la carpeta temporal."""
    ruta = tmp_path / "observabilidad_test.db"
    monkeypatch.setattr(settings, "OBSERVABILIDAD_DB_PATH", str(ruta))
    monkeypatch.setattr(settings, "OBSERVABILIDAD_SAL", None)
    return ruta
