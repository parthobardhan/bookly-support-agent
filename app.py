"""
Streamlit UI for Bookly support: text chat with the LangGraph agent.
"""

from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from agent import get_bookly_graph

load_dotenv()

REQUIRED_KEYS = (
    "OPENAI_API_KEY",
    "VOYAGE_API_KEY",
    "MONGODB_URI",
)


def _check_env() -> bool:
    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        st.error("Missing environment variables: " + ", ".join(missing))
        st.stop()
    return True


def _init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


def _render_transcript(messages) -> None:
    """Show human and assistant text (skip tool-only assistant turns for readability)."""
    for m in messages:
        if isinstance(m, HumanMessage):
            with st.chat_message("user"):
                st.markdown(m.content)
        elif isinstance(m, AIMessage) and (m.content or "").strip():
            with st.chat_message("assistant"):
                st.markdown(m.content)


def main() -> None:
    st.set_page_config(page_title="Bookly Support", page_icon=None)
    st.title("Bookly customer support")
    st.caption("Text chat with the Bookly agent (LangGraph + MongoDB memory).")

    _check_env()
    _init_session()

    graph = get_bookly_graph()
    cfg = {"configurable": {"thread_id": st.session_state.thread_id}}

    state = graph.get_state(cfg)
    msgs = state.values.get("messages", []) if state.values else []
    _render_transcript(msgs)

    prompt = st.chat_input("Message Bookly support")
    if prompt:
        graph.invoke({"messages": [HumanMessage(content=prompt)]}, config=cfg)
        st.rerun()


if __name__ == "__main__":
    main()
