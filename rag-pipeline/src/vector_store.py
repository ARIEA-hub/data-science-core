"""
vector_store.py

Embeds document chunks with OpenAI embeddings and stores them in a local
FAISS index, persisted to disk so re-running the app doesn't re-embed
(and re-pay for) the whole corpus every time.

Requires an OpenAI API key with access to an embeddings model, set as the
OPENAI_API_KEY environment variable (or passed explicitly).

Note: langchain_community's FAISS wrapper is used here since it's still
the most widely documented path as of this writing; langchain-community
is in the process of being split into standalone integration packages
(see the `langchain-faiss` package) if you want to migrate later.
"""

import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from ingest import Chunk

VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vector_store")
EMBEDDING_MODEL = "text-embedding-3-small"  # cheap + good enough for a demo corpus


def chunks_to_documents(chunks: list[Chunk]) -> list[Document]:
    return [
        Document(page_content=c.text, metadata={"doc_id": c.doc_id, "chunk_id": c.chunk_id})
        for c in chunks
    ]


def build_vector_store(chunks: list[Chunk], api_key: str | None = None) -> FAISS:
    """Embeds all chunks and builds a fresh FAISS index. Expensive - call once, then persist."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Export it or pass api_key explicitly.")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    docs = chunks_to_documents(chunks)
    store = FAISS.from_documents(docs, embeddings)
    return store


def save_vector_store(store: FAISS, path: str = VECTOR_STORE_DIR) -> None:
    os.makedirs(path, exist_ok=True)
    store.save_local(path)


def load_vector_store(api_key: str | None = None, path: str = VECTOR_STORE_DIR) -> FAISS:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Export it or pass api_key explicitly.")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def get_or_build_vector_store(chunks: list[Chunk], api_key: str | None = None,
                               path: str = VECTOR_STORE_DIR) -> FAISS:
    """Loads a persisted index if present, otherwise builds + saves a new one."""
    if os.path.exists(path) and os.path.isdir(path):
        try:
            return load_vector_store(api_key, path)
        except Exception:
            pass  # fall through and rebuild if the persisted index is unreadable
    store = build_vector_store(chunks, api_key)
    save_vector_store(store, path)
    return store
