from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.build_index import build_index, PERSIST_DIR, KNOWLEDGE_BASE_DIR
from rag.rag_agent import generate_answer, stream_answer, retrieve_sources


load_dotenv()

st.set_page_config(page_title="RAG (Chroma) – Streamlit", page_icon="📄", layout="wide")


def ensure_index_exists() -> bool:
    return (PERSIST_DIR / "chroma.sqlite3").exists() or any(PERSIST_DIR.glob("**/*"))


def sidebar():
    with st.sidebar:
        st.header("Settings")
        st.caption("Provide your OpenAI key in .env or the environment.")

        openai_ok = bool(os.getenv("OPENAI_API_KEY"))
        st.write("OpenAI API Key:", "✅" if openai_ok else "❌ Missing")

        st.divider()
        st.subheader("Knowledge Base")
        st.write("Add PDFs to:")
        st.code(str(KNOWLEDGE_BASE_DIR))
        if st.button("Build / Update Index", type="primary"):
            with st.spinner("Building Chroma index..."):
                build_index()
            st.success("Index built.")

        st.divider()
        st.info(
            "This sample is inspired by the LangGraph-based template in agent-service-toolkit."
        )


def main():
    sidebar()

    st.title("📄 RAG over PDFs (Chroma)")
    st.caption("Drop PDFs in the knowledge base and chat with them.")

    if not ensure_index_exists():
        st.warning(
            "No Chroma index found. Add PDFs to the knowledge base and click 'Build / Update Index' in the sidebar."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            tokens = []
            for chunk in stream_answer(prompt):
                tokens.append(chunk)
                placeholder.markdown("".join(tokens))

            answer = "".join(tokens)
            sources = retrieve_sources(prompt)
            if sources:
                with st.expander("Sources"):
                    st.write("\n".join(f"- {s}" for s in sources))

        st.session_state.messages.append(("assistant", answer))


if __name__ == "__main__":
    main()

