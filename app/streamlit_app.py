"""LitraRag chat: ask questions about your documents.

Run with:   uv run streamlit run app/streamlit_app.py
Close this app before running `litrarag ingest` (embedded Qdrant allows one process).
"""

from collections.abc import Sequence

import streamlit as st

from litrarag.chat import Chat, clean_question, sources_for, start_chat
from litrarag.config import Settings
from litrarag.generator import GenerationError
from litrarag.logging_setup import setup_logging
from litrarag.schemas import SearchHit

st.set_page_config(page_title="LitraRag", page_icon="📚")


@st.cache_resource(show_spinner="Loading LitraRag…")
def get_chat() -> Chat:
    """Built ONCE, then reused on every rerun."""
    settings = Settings.from_env()
    setup_logging(settings.log_level, settings.log_dir)
    return start_chat(settings)


def show_sources(label: str, shown: Sequence[tuple[int, SearchHit]]) -> None:
    if not shown:
        return
    with st.expander(f"{label} ({len(shown)})"):
        for n, hit in shown:
            source = hit.chunk.metadata.get("source", "unknown")
            st.markdown(f"**[{n}] {source}, page {hit.chunk.page_number}**")
            preview = hit.chunk.text[:400]
            st.caption(preview + ("…" if len(hit.chunk.text) > 400 else ""))


st.title("📚 LitraRag")
st.caption("Answers from your own documents, running on your machine.")

try:
    chat = get_chat()
except Exception as err:  # e.g. Ollama not running, or storage locked by an ingest
    st.error(f"LitraRag could not start: {err}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far (the script reruns from the top on every message)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("label", ""), message.get("sources", []))

if raw_question := st.chat_input("Ask about your documents"):
    question = clean_question(raw_question)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents…"):
            hits = chat.retrieve(question)
        try:
            with st.spinner("Reading the passages…"):
                text = str(st.write_stream(chat.stream(question, hits)))
        except GenerationError as err:
            st.error(str(err))
            st.stop()
        label, shown = sources_for(text, hits)
        show_sources(label, shown)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "label": label, "sources": shown}
    )
