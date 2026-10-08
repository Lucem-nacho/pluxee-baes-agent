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

*(Flujo de EP1. Ver "Flujo — EP2" más abajo para cómo cambian los pasos 3-6
al agregar framework de agentes, memoria y escritura.)*

## Flujo — EP2 (agente funcional, memoria y escritura)

EP2 pide extender el proyecto de EP1 con cuatro capacidades que el diseño
original no tenía: un framework de agentes real, memoria de largo plazo,
una herramienta de escritura, y planificación/decisiones adaptativas. Los
pasos 1-2 de EP1 (interfaz -> clasificador) no cambian; lo que cambia es
qué pasa después de clasificar como INFORMATIVA:

1. Interfaz de chat (`app.py`) recibe la consulta — ahora con un
   `id_usuario` editable en la barra lateral, que identifica la memoria de
   largo plazo del estudiante (sin ser un login real, ver limitaciones).
2. `agent/classifier.py` clasifica igual que en EP1, **sin cambios**: sigue
   siendo el filtro barato antes de invocar nada más costoso — evita gastar
   un ciclo completo de tool-calling (varias llamadas al LLM) en consultas
   claramente fuera de alcance.
3. Si ESCALAR, `memory.store.registrar_escalamiento` persiste la derivación
   en SQLite (tabla `escalamientos`) — la primera capacidad de ESCRITURA
   real del agente. En EP1, `agent/escalation.py` armaba el mismo paquete
   pero se perdía al cerrar la sesión.
4. Si INFORMATIVA, se arma un `AgentExecutor` de LangChain
   (`agent/orchestrator.py` + `agent/tools.py`) con 4 tools: `buscar_normativa`
   (envuelve `rag/retrieval/vector_retriever.py` + `rag/arbitration.py`,
   igual que el pipeline de EP1), `buscar_comercio` (envuelve
   `rag/retrieval/comercios_query.py`), `consultar_historial` (memoria de
   largo plazo, nueva) y `escalar_a_ejecutivo` (escritura, nueva — el
   agente puede decidir derivar A MITAD de su propio razonamiento, no solo
   el clasificador de entrada). El LLM decide él mismo qué tools llamar, en
   qué orden, y cuándo detenerse — a diferencia del camino fijo de EP1
   (recuperar -> comercios -> arbitrar -> generar, siempre en ese orden).
5. Toda interacción resuelta (por cualquiera de los dos caminos) se
   registra en `interacciones` (memoria de largo plazo) antes de devolver
   el resultado a `app.py`.

`agent/generator.py` y la llamada directa a `rag/arbitration.py` desde el
orquestador (ambos de EP1) ya no están en este camino — ver sus propios
docstrings, actualizados para documentar por qué se mantienen en el
repositorio sin tocar en vez de borrarse.

### Memoria: corto plazo vs. largo plazo

EP2 pide "mecanismos de memoria de corto y largo plazo" como dos cosas
distintas, así que se implementaron como dos mecanismos genuinamente
distintos, no una sola tabla con dos nombres:

| | Corto plazo | Largo plazo |
|---|---|---|
| Dónde vive | `st.session_state.mensajes` (navegador) | tabla `interacciones` en SQLite (disco) |
| Sobrevive a | Nada — se pierde al cerrar la pestaña | Cerrar la app, reiniciar el proceso |
| Quién la usa | El prompt del clasificador (vía `historial_contexto`) | La tool `consultar_historial`, que el LLM decide invocar o no |
| Identificada por | La sesión del navegador (implícito) | `id_usuario`, que el estudiante escribe a mano |

### Escritura: de EP1 a EP2

En EP1, `agent/escalation.py` devolvía un dict de derivación que `app.py`
mostraba y se perdía. EP2 pide una herramienta de ESCRITURA real dentro del
agente — no basta con que el agente "sepa" escalar, tiene que dejar un
registro persistente de que lo hizo. `memory/store.py` agrega la tabla
`escalamientos`, con el mismo patrón de conexión por llamada que
`rag/retrieval/comercios_query.py` ya había resuelto para el problema de
hilos de Streamlit (ver docstring de `memory/store.py`) — la misma lección
de EP1, aplicada de nuevo a una base en disco en vez de en memoria.

### Demostración de planificación y decisiones adaptativas (EP2, IE5/IE7)

La diferencia con EP1 no es solo técnica (framework vs. funciones planas):
es que el mismo agente puede terminar en caminos distintos según lo que
encuentra, sin que eso esté decidido de antemano:

- Una consulta normativa general solo debería disparar `buscar_normativa`.
- Una consulta sobre un comercio puntual debería disparar `buscar_comercio`
  y, razonablemente, también `buscar_normativa` (para cruzar la
  restricción particular del comercio con la regla general).
- Una consulta que el clasificador dejó pasar como INFORMATIVA pero que, al
  investigar, resulta ambigua o contradictoria, puede terminar en
  `escalar_a_ejecutivo` — una decisión que en EP1 solo podía tomar el
  clasificador, antes de ver ningún resultado de búsqueda.

`agent/orchestrator.py` expone estos pasos (`resultado["pasos"]`) y
`app.py` los muestra como nota interna junto a la respuesta, pensado
específicamente para la demo en vivo que pide la pauta de presentación
(IE7). Los 4 escenarios de `scripts/check_agent_executor.py` están
diseñados para mostrar esto con llamadas reales.

## Stack técnico

| Componente | Elección |
|---|---|
| LLM | Google Gemini (`gemini-3.6-flash` explícito, free tier vía Google AI Studio, SDK `google-genai`) — proveedor abstraído en `llm/client.py` |
| Embeddings | `sentence-transformers` (local, gratis) |
| Vector store | ChromaDB, colección `manual_faq_social` (única en esta versión, ver limitaciones) |
| Tabla estructurada | sqlite3 (stdlib) sobre `data/comercios/comercios.csv` |
| PDF parsing | `pypdf` |
| Orquestación (INFORMATIVA) | **EP2**: `AgentExecutor` de LangChain (`langchain-google-genai` + `langchain.agents`), tool-calling sobre `gemini-3.6-flash` — reemplaza el camino fijo de EP1 (funciones Python planas) |
| Orquestación (ESCALAR) | Sin cambios respecto a EP1: `agent/classifier.py` decide antes de llegar al agente |
| Memoria persistente | **EP2**: sqlite3 (stdlib) en disco, `memory/store.py` — tablas `escalamientos` (escritura) e `interacciones` (memoria de largo plazo) |
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

## Limitaciones conocidas y extensiones futuras — EP2

- **El código de EP2 no se pudo ejecutar durante el desarrollo**: a
  diferencia de EP1 (donde todo se probó con llamadas reales a Gemini
  durante el desarrollo), el entorno donde se escribieron
  `agent/tools.py`, `agent/orchestrator.py` y los cambios a `app.py` no
  tenía acceso a internet para instalar `langchain` / `langchain-google-genai`.
  Lo que SÍ se probó ahí mismo, con datos reales y sin depender de ningún
  paquete externo (sqlite3 es stdlib): `memory/store.py` completo — 10
  pruebas unitarias (`tests/test_memory_store.py`) más una prueba de
  concurrencia multi-hilo con 20 escrituras simultáneas, mismo patrón que
  ya se había verificado para `comercios_query.py` en EP1.
  - **Primera ejecución real (2026-10-08, LangChain 1.4.0 +
    langchain-google-genai 4.4.0 + Gemini)** — encontró 4 problemas que
    ninguna prueba sin red podía ver, todos corregidos:
    1. `from langchain.agents import AgentExecutor` no existe en LangChain
       1.x (se movió a `langchain-classic`) → import corregido y paquete
       agregado a `requirements.txt`.
    2. El contenido final del agente puede llegar como lista de bloques, no
       `str`, y SQLite lo rechazaba (`ProgrammingError`) →
       `_texto_de_salida` en `agent/orchestrator.py`.
    3. Ante una pregunta que el manual no cubre (bebidas energéticas), el
       agente reformulaba la búsqueda en bucle hasta agotar iteraciones, y
       la librería devolvía el texto interno `"Agent stopped due to max
       iterations."` (que se habría mostrado al estudiante y guardado en la
       memoria) → regla 5 en `prompts/agente.py` (máx. 2 búsquedas) y
       degradación a un mensaje honesto de "no encontré información" si
       igual se agota el límite.
    4. El clasificador (prompt de EP1) mandaba "¿Qué te pregunté la última
       vez?" a ESCALAR por "ambigua", así que la memoria de largo plazo era
       inalcanzable → ampliación mínima de `prompts/clasificador.py`.
    Además, el agente ignoraba `buscar_comercio` al nombrar un comercio
    (respondía solo con la norma general, lo que daría respuestas erróneas
    para comercios con restricción propia) → regla 6 en `prompts/agente.py`.
    Cubierto con 6 pruebas offline permanentes en `tests/test_orchestrator.py`
    (modelo falso, sin cuota).
  - **Verificación real de los 4 escenarios de `scripts/check_agent_executor.py`**:
    `normativa`, `comercio` y `escalamiento` OK con Gemini real.
    `memoria` quedó **parcial**: tras el arreglo del clasificador, la
    consulta pasó a la ruta del agente (la cuota se agotó justo ahí), pero
    NO está confirmado con Gemini real que el agente elija
    `consultar_historial`; la tool en sí sí está probada offline y solo lee
    el `id_usuario` de la sesión. Pendiente de re-correr con cuota.
  - Los escenarios se ejecutaron con `GEMINI_MODEL=gemini-flash-latest`
    como sobreescritura temporal por variable de entorno, porque la cuota
    diaria de `gemini-3.6-flash` (default del proyecto) se había agotado
    durante el diagnóstico. Valida el código, no necesariamente el
    comportamiento exacto de `gemini-3.6-flash`.
- **`id_usuario` no es autenticación real**: es un string que el estudiante
  escribe libremente en la barra lateral de `app.py`. Cualquiera que
  escriba el mismo `id_usuario` puede leer ese historial. Suficiente para
  demostrar memoria de largo plazo en un contexto académico, no para
  producción — ver docstring de `memory/store.py`.
- **Dos mecanismos de reintento distintos para el mismo LLM**: el
  clasificador (`llm/client.py`) tiene el manejo de 503/429 probado con
  llamadas reales en EP1 (espera el `retryDelay` exacto que devuelve la
  API). El `AgentExecutor` usa `ChatGoogleGenerativeAI` de
  `langchain-google-genai` directamente, con su propio `max_retries` — no
  pasa por `llm/client.py`, así que no hereda ese comportamiento
  específico. Unificar ambos caminos bajo un solo wrapper queda como
  mejora futura.
- **`MAX_ITERACIONES_AGENTE=6`** (`agent/orchestrator.py`): medido contra
  casos reales, SÍ se alcanzaba en una consulta fuera de cobertura (ver
  punto 3 arriba). Con la regla de máx. 2 búsquedas, los escenarios
  `normativa` y `comercio` terminaron en 2 llamadas a tools. Cada consulta
  informativa cuesta ~3-4 requests a Gemini (clasificador + 2-3 del
  agente), así que el free tier de 20/día alcanza para unas 5-6 consultas
  por modelo — recordarlo para la demo en vivo.
- **Errores de cuota/servidor en la ruta del agente no se manejan**: un 429
  o un 503 persistente en `ChatGoogleGenerativeAI` (observados ambos
  durante la verificación) propaga la excepción y `app.py` mostraría un
  traceback al estudiante. Lo mismo ocurre con el clasificador si agota sus
  reintentos. Aceptado por ahora; mejora futura: capturar en
  `agent/orchestrator.py` y responder con un mensaje de "servicio no
  disponible, intenta en unos minutos".

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
