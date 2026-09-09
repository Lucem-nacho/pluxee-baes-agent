# Agente informativo BAES — Pluxee Chile

Agente basado en LLM y RAG que responde consultas normativas sobre el
beneficio BAES, y deriva a un ejecutivo humano las consultas transaccionales
o complejas.

## Requisitos
- Python 3.10+
- Una API key gratuita de Google AI Studio: https://aistudio.google.com/apikey

## Instalación
\`\`\`bash
pip install -r requirements.txt
cp .env.example .env
# pega tu GOOGLE_API_KEY en .env
\`\`\`

## Construir el índice (una sola vez)
\`\`\`bash
python scripts/ingest_all.py
\`\`\`

## Ejecutar
\`\`\`bash
python -m streamlit run app.py --server.fileWatcherType none
\`\`\`
Se abre automáticamente en el navegador, en `localhost:8501`.

**Nota:** se usa `python -m streamlit` en vez de `streamlit` directo porque en
algunos entornos (especialmente Windows) el ejecutable no queda en el PATH
del sistema pese a estar instalado. El flag `--server.fileWatcherType none`
evita que Streamlit escanee innecesariamente los submódulos de la librería
`transformers`, lo que sin el flag puede generar cientos de advertencias
inofensivas en la terminal y una carga inicial más lenta (pantalla en negro
por varios segundos antes de que aparezca la interfaz — no es un error).

## Estructura
- `agent/` — clasificador, generador, escalamiento, orquestador
- `rag/` — carga de fuentes, chunking, embeddings, retrieval, arbitraje
- `llm/` — wrapper del proveedor LLM (Gemini, nivel gratuito)
- `data/` — fuentes (manual BAES real, FAQ, comercios simulados, social mock)
- `docs/arquitectura.md` — decisiones de diseño y limitaciones con evidencia

## Limitaciones conocidas
- Nivel gratuito de Gemini: 5 solicitudes/min, 20/día — evitar ráfagas de
  preguntas seguidas. Si aparece un error `429 RESOURCE_EXHAUSTED`, es la
  cuota diaria agotada, no una falla del sistema; probar con una API key
  propia resuelve esto de inmediato.
- El umbral de arbitraje por similitud (0.9) prioriza evitar falsos positivos
  sobre detectar todas las contradicciones reales entre fuentes.
- El listado de comercios es un dataset simulado (Pluxee no publica uno
  descargable).

## Uso de IA
Se usó Claude para apoyo en redacción de código, diagramas y documentación
técnica. Decisiones de diseño e interpretación de resultados son propias.