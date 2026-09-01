"""
Streamlit chat UI for the RAG pipeline.

Run locally:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...      # or paste it into the sidebar at runtime
    streamlit run app.py

First run will download the 4-book corpus and build the FAISS index
(costs a small amount of OpenAI embedding usage); subsequent runs reload
the persisted index from vector_store/ instantly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from ingest import load_corpus, SOURCES
from vector_store import get_or_build_vector_store
from rag_chain import answer_question

st.set_page_config(page_title="RAG over Public Domain Novels", layout="wide")

st.title("📚 RAG Pipeline — Ask Questions About 4 Public-Domain Novels")
st.caption(
    "Corpus: " + ", ".join(k.replace("_", " ").title() for k in SOURCES)
    + ". Answers are grounded ONLY in retrieved excerpts, with citations."
)

# ---------------- API key handling ----------------
st.sidebar.header("Settings")
api_key = st.sidebar.text_input(
    "OpenAI API key",
    value=os.environ.get("OPENAI_API_KEY", ""),
    type="password",
    help="Not stored anywhere - used only for this session's requests.",
)
k = st.sidebar.slider("Chunks retrieved per question (k)", 1, 10, 4)

if not api_key:
    st.warning("Enter your OpenAI API key in the sidebar to build the index and ask questions.")
    st.stop()

# ---------------- Build / load index (cached per session) ----------------
if "vector_store" not in st.session_state:
    with st.spinner("Loading corpus and building/loading vector index (first run may take a minute)..."):
        chunks = load_corpus()
        st.session_state.vector_store = get_or_build_vector_store(chunks, api_key=api_key)
    st.success(f"Index ready — {len(chunks)} chunks across {len(SOURCES)} documents.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- Chat UI ----------------
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("Sources used"):
            for s in turn["sources"]:
                st.markdown(f"**{s['doc_id']}** (`{s['chunk_id']}`)")
                st.caption(s["excerpt"])

question = st.chat_input("Ask something about Pride and Prejudice, Sherlock Holmes, Dracula, or The Time Machine...")

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            try:
                result = answer_question(question, st.session_state.vector_store, api_key=api_key, k=k)
            except Exception as e:
                st.error(f"Error calling OpenAI: {e}")
                st.stop()
        st.write(result["answer"])
        with st.expander("Sources used"):
            for s in result["sources"]:
                st.markdown(f"**{s['doc_id']}** (`{s['chunk_id']}`)")
                st.caption(s["excerpt"])
    st.session_state.chat_history.append({
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
    })
