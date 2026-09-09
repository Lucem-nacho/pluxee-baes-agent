# Arquitectura — Agente Informativo BAES (Pluxee Chile)

## Organización y caso

Pluxee Chile administra el beneficio BAES (Junaeb) para estudiantes de
educación superior. El canal de atención se satura con consultas repetitivas
sobre restricciones normativas de compra. Objetivo: reducir en 40% el tiempo
de resolución de dudas normativas con respuestas automáticas, derivando a
ejecutivos humanos solo consultas complejas o transaccionales.

**Restricción clave:** el agente es estrictamente informativo. Nunca ejecuta
acciones sobre la cuenta del estudiante (recuperar clave, modificar saldo,
etc.), y debe mantener alta fidelidad respecto a la normativa de Junaeb.

## Flujo

1. Interfaz de chat (Streamlit, `app.py`) recibe la consulta.
2. `agent/classifier.py` clasifica en INFORMATIVA o ESCALAR, usando el
   historial de la conversación como contexto.
3. Si INFORMATIVA, pasa al pipeline RAG (`rag/`), que recupera de las
   fuentes vigentes en esta versión (ver "Fuentes descartadas/redundantes"
   más abajo para las dos que quedaron fuera):
   - Manual BAES (PDF) — interna, estática — chunking + vector store.
   - FAQ — interna, semi-estática — cada par pregunta-respuesta = 1 fragmento.
   - Listado de comercios asociados — interna, estructurada — consulta SQL
     sobre sqlite, **no se vectoriza**.
   - Twitter/X de Pluxee — externa, dinámica — ingesta simulada cada 24h,
     filtrando contenido normativo antes de vectorizar. Pendiente de
     implementar (`rag/loaders/social_loader.py`).
4. `rag/arbitration.py`: si dos fuentes se contradicen, prioriza la más
   reciente por metadata (`fuente`, `fecha`) — no por colección de origen —
   y conserva la anterior como contexto histórico.
5. LLM generador (`prompts/generador.py` + `llm/client.py`) redacta la
   respuesta final citando fuente y fecha.
6. Si ESCALAR, `agent/escalation.py` deriva a un ejecutivo humano con el
   contexto de la conversación adjunto.

## Stack técnico

| Componente | Elección |
|---|---|
| LLM | Google Gemini (`gemini-3.6-flash` explícito, free tier vía Google AI Studio, SDK `google-genai`) — proveedor abstraído en `llm/client.py` |
| Embeddings | `sentence-transformers` (local, gratis) |
| Vector store | ChromaDB, colección `manual_faq_social` (única en esta versión, ver limitaciones) |
| Tabla estructurada | sqlite3 (stdlib) sobre `data/comercios/comercios.csv` |
| PDF parsing | `pypdf` |
| Orquestación | Funciones Python planas (`agent/orchestrator.py`), sin framework de grafos — flujo lineal simple, fácil de depurar/explicar |
| Interfaz | Streamlit |

## Limitaciones conocidas y extensiones futuras (fuera de alcance de esta versión)

- **Disponibilidad y cuota del free tier de Gemini** (`llm/client.py`):
  probado con llamadas reales, el free tier devuelve `ServerError` 503
  ("high demand") de forma intermitente — en una corrida, 2 de 5 llamadas
  fallaron así; en otra corrida inmediatamente después, 5 de 5 funcionaron.
  `generar_respuesta` reintenta hasta 3 veces con backoff exponencial corto
  (1s, 2s) ante `ServerError`, pero NO ante otros `ClientError` (API key
  inválida, etc., que fallarían igual las 3 veces).
  - **Elección de modelo, con evidencia**: se probó la hipótesis de que el
    alias `gemini-flash-latest` resolvía a un modelo con cuota gratuita más
    restrictiva que un modelo Flash estable explícito. Resultado: no se
    confirmó del todo. `gemini-2.5-flash` (el modelo estable "obvio") ya no
    está disponible para keys nuevas (404 "no longer available to new
    users"; Google redirige a `gemini-3.6-flash`). Probando
    `gemini-3.6-flash` directamente, el free tier limita a **5
    requests/minuto** (`quotaId
    GenerateRequestsPerMinutePerProjectPerModel-FreeTier`) — se agota en
    segundos con llamadas seguidas, incluso más ajustado que lo que toleró
    `gemini-3.8-flash` (a donde resolvía el alias `-latest`) durante el
    resto de esta sesión. Se fijó `GEMINI_MODEL=gemini-3.6-flash` de todas
    formas por ser el reemplazo oficial que recomienda Google, no por tener
    mejor cuota.
  - **Mitigación implementada y verificada con un 429 real**: a diferencia
    de un `ClientError` genérico, un 429 (`RESOURCE_EXHAUSTED`) sí es
    recuperable — la propia API devuelve cuánto esperar (`retryDelay`).
    `generar_respuesta` reintenta UNA sola vez usando ese valor exacto
    (acotado a `MAX_ESPERA_CUOTA_SEGUNDOS=60`). Verificado agotando la cuota
    a propósito: absorbió una espera real de 44.8s y devolvió la respuesta
    sin propagar el error.
  - **Recomendación para la demo en vivo**: evitar ráfagas de preguntas
    seguidas — con 5 req/min de cuota, dos 429 consecutivos sin espacio
    entre medio propagan el segundo (el retry cubre uno solo por llamada).
    Espaciar las preguntas unos segundos evita depender del retry. Riesgo
    aceptado y documentado para que esto no se confunda con un bug del
    proyecto si ocurre en la presentación.

- **Colección "normativa_junaeb" (normativa Junaeb / Ley de Etiquetado
  externa)**: no se implementa en esta versión. Al revisar el manual real,
  su tabla de causales de pérdida (`Tabla N°15`, ver `docs/arquitectura.md`
  → "Fuentes reales usadas") ya cubre las restricciones de productos
  (alcohol, tabaco, fármacos) que era justo el tipo de contenido que esta
  fuente externa iba a aportar — se determinó redundante con el contenido
  ya indexado desde el manual, no un vacío real de cobertura. Mismo patrón
  que `rag/loaders/comercios_scraper.py` y `agent/validator.py`: documentada
  como decisión de alcance, no como funcionalidad pendiente por olvido.
- **Integración con el buscador real de comercios de Pluxee**
  (`rag/loaders/comercios_scraper.py`): el buscador web no expone API y
  bloquea scraping. Se usa un dataset simulado de 15 comercios en su lugar.
- **Validador de fidelidad post-generación** (`agent/validator.py`): no se
  implementa una verificación automática de que la respuesta del LLM no
  contenga afirmaciones no respaldadas por el contexto recuperado. Se mitiga
  parcialmente con la regla 1 del prompt del generador. Queda como mejora
  futura, no como funcionalidad construida.
- **Umbral de arbitraje por similitud semántica** (`rag/arbitration.py`):
  probado con datos reales, se encontró que dos causales de pérdida
  DISTINTAS que comparten la misma jerga legal de cierre ("es una causal de
  sanción definitiva...") pueden puntuar más similar entre sí (~0.715) que la
  FAQ y su propia fila real en la tabla del manual sobre el MISMO tema
  (~0.57). Se fijó `UMBRAL_MISMO_TEMA=0.9` a propósito para evitar ese falso
  positivo.
  - **Confirmado con una contradicción real deliberada** (no solo con datos
    que coinciden por casualidad): se agregó un tweet mock que actualiza a
    propósito la regla de fármacos ("ya no está prohibido comprar fármacos
    de venta libre", 2026-09-01) contradiciendo a la FAQ/manual ("es causal
    de sanción definitiva", 2026-01). `recuperar()` trae ambos fragmentos en
    el top-6 para la consulta "¿Puedo comprar fármacos con la BAES?", pero
    `resolver_contradicciones` los puso a los 6 en `vigente` — **no detectó
    la contradicción**. Medido directamente: similitud = **0.723**, prácticamente
    igual al falso positivo de arriba (0.715, gap de 0.008).
  - **Conclusión, no es un problema de calibración**: no existe un valor de
    `UMBRAL_MISMO_TEMA` que separe "mismo punto normativo, actualizado" de
    "puntos distintos, misma jerga legal" en este corpus con
    `all-MiniLM-L6-v2` — ambos casos caen en la misma banda de similitud
    (~0.71-0.72). Es una limitación estructural del método (similitud
    coseno de embeddings sobre texto administrativo en español con mucho
    boilerplate compartido), no algo resoluble subiendo o bajando el número.
  - **Por qué no es necesariamente fatal**: `resolver_contradicciones` no
    descarta nada — los 6 fragmentos (ambos lados de la contradicción, con
    sus fechas) igual llegan a `agent/generator.py` como `vigente`. La regla
    2 del prompt del generador (`prompts/generador.py`) le pide al LLM
    razonar sobre recencia directamente desde las fechas del contexto — una
    segunda capa de defensa con capacidad semántica real, distinta de la
    heurística de coseno.
  - **Prueba con Gemini real — resultado PARCIAL, no concluyente todavía**:
    se diseñó un control de sesgo de posición: correr `generar()` con los
    mismos 6 fragmentos en el orden original (tweet de la actualización
    primero, por el ordenamiento descendente interno de
    `resolver_contradicciones`) y de nuevo con el orden invertido (tweet al
    final). Si el generador acierta en ambos, es evidencia de razonamiento
    real sobre fechas; si solo acierta con el tweet primero, sería sesgo de
    posición — una limitación más seria que la del umbral de arbitraje.
    - **Corrida 1 (tweet primero) — ejecutada, resultado correcto**: "Te
      cuento que sí puedes comprar fármacos de venta libre en farmacias
      asociadas con tu BAES. La normativa fue actualizada recientemente,
      por lo que, según la actualización del 1 de septiembre de 2026 en el
      Twitter/X de Pluxee, esta compra ahora sí está permitida. Solo ten
      muy en cuenta que la prohibición de comprar alcohol y tabaco se
      mantiene y sigue siendo causal de pérdida definitiva del beneficio."
      Sin advertencias de formato (cita fuente y fecha explícitas, 3 frases).
    - **Corrida 2 (tweet al final) — NO se pudo ejecutar**: `ClientError`
      429 persistente (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`),
      incluso después del reintento automático de `llm/client.py` (que
      espera el `retryDelay` real, ~59s) — consistente con que es una cuota
      por DÍA, no por minuto, y no se repuso en el tiempo real transcurrido
      entre sesiones (el "día" de la cuenta de Google no equivale al
      "mañana" narrativo de esta conversación).
    - **Conclusión honesta**: el hallazgo NO está cerrado. Corrida 1 por sí
      sola es compatible tanto con "razonamiento real sobre fechas" como con
      "sesgo de posición" (repetir lo que aparece primero en el prompt) —
      no permite distinguir entre ambas hipótesis. La validación de que el
      diseño en dos capas (arbitraje heurístico + razonamiento del LLM)
      compensa la falla de la primera capa queda **pendiente de la Corrida
      2**, a ejecutar cuando la cuota diaria se reponga de verdad.
  - En la práctica, mientras no se resuelva esto, `rag/arbitration.py`
    funciona como un colapsador de casi-duplicados literales, no como un
    detector genuino de contradicciones entre fuentes con redacciones
    distintas.
- **Chunking de contenido tabular** (`rag/indexing/chunking.py`): la causa
  raíz del hallazgo anterior es que `CHUNK_SIZE=800` empaqueta varias filas
  de la tabla de causales del manual en un mismo chunk, mezclando 2-3
  causales distintas y diluyendo la señal semántica de cualquiera de ellas
  individualmente. Se probó (y se descartó) que fuera un problema de
  formato: normalizar espacios/saltos de línea antes de generar el embedding
  no cambió la similitud ni en la 16ª decimal (el tokenizer del modelo ya
  colapsa espacios internamente). Un chunking consciente de tablas (una fila
  = un chunk) resolvería esto, pero queda fuera de alcance por ahora.

## Fuentes reales usadas

- Manual de Orientaciones Técnicas BAES 2026 y Resolución Exenta DN-00556/2026:
  **ambos viven en el mismo archivo**,
  `data/raw/REX-DN-00556-2026_APRUEBA-MODIFICACION-MANUAL-BAES-2026.pdf`.
  Al inspeccionar el texto extraído se confirmó que el PDF de la resolución
  trae el manual anexado íntegro (páginas PDF 4-54, numeración interna 1-51),
  con el texto propio de la resolución en las páginas PDF 1-3 y 55.
  `rag/loaders/pdf_loader.py` separa ambas fuentes por rango de páginas,
  cada una con su propio campo `fuente` para las citas del generador. Ver
  `data/raw/README.md` para el detalle.
- Listado de comercios: dataset simulado (`data/comercios/comercios.csv`), no
  hay fuente real descargable disponible.

### Fuente descartada

- `data/raw/2026_MANUAL GUÍA USUARIO.pdf`: pese al nombre, es una guía de uso
  de la app Pluxee (activar tarjeta, pagar, cambiar PIN, agregar correo) —
  contenido procedimental sobre acciones de cuenta, no normativa de
  restricciones de compra, y adyacente a temas que deben ESCALAR. Se decidió
  (con el usuario) no indexarla en esta versión.
