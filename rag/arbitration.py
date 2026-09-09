"""Módulo de arbitraje entre fuentes que se contradicen.

Diseño confirmado: el arbitraje opera sobre la METADATA de cada fragmento
recuperado (campos `fuente` y `fecha`), NUNCA sobre la colección Chroma de
origen. Un fragmento de "normativa_junaeb" y uno de "manual_faq_social" se
comparan exactamente igual si responden la misma pregunta — la colección es
un detalle de indexación, no una señal de prioridad.

Cómo se detecta "el mismo punto normativo": los fragmentos recuperados para
una misma consulta ya vienen acotados (top-k por colección), así que agrupar
por similitud semántica entre ellos —reutilizando el mismo modelo de
embeddings que la recuperación— es suficiente para este alcance, sin
necesitar un clasificador de temas aparte. Dos fragmentos con similitud
coseno sobre UMBRAL_MISMO_TEMA se consideran el mismo punto.

LIMITACIÓN CONOCIDA (probada con datos reales, no hipotética): con este
corpus, dos causales de pérdida DISTINTAS que comparten la misma jerga legal
de cierre ("es una causal de sanción definitiva...") pueden puntuar más alto
entre sí (~0.71) que la FAQ y su propia fila real en la tabla del manual
sobre EL MISMO tema (~0.57) — la fila de tabla mezcla varias causales en un
mismo chunk (ver rag/indexing/chunking.py), lo que diluye su embedding.
No existe un umbral que separe ambos casos correctamente en este corpus.
UMBRAL_MISMO_TEMA se fija alto (0.9) a propósito: prioriza NUNCA agrupar por
error (que arriesgaría que el generador afirme que una regla "fue
actualizada" por otra sin relación real — un fallo de fidelidad activo) por
sobre detectar toda contradicción real. En la práctica, esto reduce el
agrupamiento a casi-duplicados literales (ej. chunks solapados por el
overlap del chunking); no detecta aún contradicciones genuinas entre fuentes
con redacciones distintas. Ver docs/arquitectura.md.

Regla de arbitraje dentro de cada grupo:
1. Se prioriza el fragmento con `fecha` más reciente (comparación lexicográfica
   de strings ISO "YYYY-MM" / "YYYY-MM-DD", válida porque están zero-padded).
2. El/los fragmento(s) más antiguos del grupo NO se descartan: pasan a
   "historico" para que el generador pueda mencionar que la regla fue
   actualizada (regla 2 de `prompts/generador.py`).
3. Un fragmento sin par semántico (no hay contradicción) va directo a
   "vigente" — no es que gane un desempate, es que no hay nada que arbitrar.
"""

from sentence_transformers import SentenceTransformer, util

from rag.text_utils import normalizar_para_embedding

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
UMBRAL_MISMO_TEMA = 0.9

_modelo = None


def _get_modelo() -> SentenceTransformer:
    global _modelo
    if _modelo is None:
        _modelo = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _modelo


def resolver_contradicciones(fragmentos: list[dict]) -> dict:
    """Recibe fragmentos de cualquier colección (mezclados) y devuelve
    {"vigente": [...], "historico": [...]}, agrupando por el punto normativo
    que abordan (similitud semántica) y priorizando por fecha dentro de
    cada grupo."""
    if not fragmentos:
        return {"vigente": [], "historico": []}

    modelo = _get_modelo()
    textos_norm = [normalizar_para_embedding(f["texto"]) for f in fragmentos]
    embeddings = modelo.encode(textos_norm, convert_to_tensor=True)
    similitudes = util.cos_sim(embeddings, embeddings)

    # Se procesa en orden de fecha descendente: así, cuando se arma un grupo,
    # el primer elemento visitado es siempre el más reciente del grupo.
    orden = sorted(range(len(fragmentos)), key=lambda i: fragmentos[i].get("fecha", ""), reverse=True)

    visitados = set()
    vigente, historico = [], []

    for i in orden:
        if i in visitados:
            continue
        visitados.add(i)
        grupo = [i]
        for j in orden:
            if j not in visitados and similitudes[i][j] >= UMBRAL_MISMO_TEMA:
                visitados.add(j)
                grupo.append(j)

        vigente.append(fragmentos[grupo[0]])
        historico.extend(fragmentos[idx] for idx in grupo[1:])

    return {"vigente": vigente, "historico": historico}
