"""Observabilidad del agente (EP3): registro de cada ejecución de
`agent.orchestrator.responder` en una base SQLite propia
(`data/processed/observabilidad.db`), separada de `memoria.db`.

- `store.py`: esquema (tablas `ejecuciones` y `pasos`), escritura y purga.
- `privacidad.py`: enmascarado de datos personales y anonimización del id.
- `callbacks.py`: handler de LangChain que mide cada herramienta y los tokens.
- `medicion.py`: acumula los datos de una ejecución y los escribe al cerrar.

Registrar nunca debe romper ni alterar al agente: cualquier fallo al escribir
se loguea y se ignora (ver `Medicion.cerrar`).
"""
