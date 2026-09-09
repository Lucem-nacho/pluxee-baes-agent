"""Interfaz de chat (Streamlit) del agente informativo BAES.

Responsabilidad: recibir la consulta del estudiante, mantener el historial
de la conversación en `st.session_state`, invocar a
`agent.orchestrator.responder` y renderizar la respuesta (o el mensaje de
derivación a ejecutivo humano). No contiene lógica de clasificación,
retrieval ni generación — todo eso vive en `agent/orchestrator.py`.
"""

import streamlit as st

from agent.orchestrator import responder

ETIQUETAS_ROL = {"user": "Usuario", "assistant": "Asistente"}


def _formatear_historial(mensajes: list[dict]) -> str:
    """Convierte el historial de session_state al string plano que espera
    el prompt del clasificador ("Usuario: ...\\nAsistente: ...")."""
    return "\n".join(f"{ETIQUETAS_ROL[m['role']]}: {m['content']}" for m in mensajes)


def _renderizar_resultado(resultado: dict) -> str:
    """Traduce el dict de `agent.orchestrator.responder` a lo que se muestra
    al estudiante. Las advertencias de formato de `agent/generator.py` NO se
    muestran al estudiante — son una nota interna de depuración."""
    if resultado["estado"] == "ESCALAR":
        return (
            "Esta consulta requiere revisión de un ejecutivo humano — no puedo "
            "resolverla de forma automática (por ejemplo, involucra una acción "
            "sobre tu cuenta o un reclamo formal). Un ejecutivo revisará tu "
            "conversación completa a la brevedad."
        )
    return resultado["respuesta"]


st.set_page_config(page_title="Agente BAES · Pluxee", page_icon="🍽️")
st.title("Agente informativo BAES")
st.caption(
    "Resuelvo dudas normativas sobre tu Beca de Alimentación para la Educación "
    "Superior (BAES). No puedo recuperar claves, modificar tu saldo ni realizar "
    "ninguna otra acción sobre tu cuenta — esas consultas se derivan a un ejecutivo."
)

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []  # [{"role": "user"|"assistant", "content": str}]

for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

consulta = st.chat_input("Escribe tu consulta sobre la BAES...")

if consulta:
    historial_contexto = _formatear_historial(st.session_state.mensajes)

    st.session_state.mensajes.append({"role": "user", "content": consulta})
    with st.chat_message("user"):
        st.markdown(consulta)

    resultado = responder(consulta, historial_contexto)
    texto_respuesta = _renderizar_resultado(resultado)

    st.session_state.mensajes.append({"role": "assistant", "content": texto_respuesta})
    with st.chat_message("assistant"):
        st.markdown(texto_respuesta)
        if resultado["estado"] == "INFORMATIVA" and resultado.get("advertencias"):
            st.caption(
                "⚠️ Nota interna (no visible en producción): "
                + "; ".join(resultado["advertencias"])
            )
