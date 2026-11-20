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


@pytest.fixture(autouse=True)
def sin_api_key_de_gemini(monkeypatch):
    """Ninguna prueba puede llamar a Gemini de verdad: sin clave, un intento
    accidental falla al instante en vez de gastar cuota. (Lección aprendida: una
    prueba que ejecutó un script con el casos.json real hizo ejecuciones
    reales.) `config.settings` copia la clave del .env al entorno al
    importarse, por eso se quita de los dos lados."""
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
