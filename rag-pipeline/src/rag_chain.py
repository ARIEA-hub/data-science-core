"""
rag_chain.py

Retrieval-augmented generation: retrieve top-k relevant chunks from the
FAISS store, then ask an OpenAI chat model to answer using ONLY those
chunks, with citations back to source document + chunk id. This keeps the
model grounded in the corpus instead of answering from parametric memory,
and makes hallucination detectable (citations point at retrievable text).
"""

import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

CHAT_MODEL = "gpt-4o-mini"  # cheap + fast, fine for a demo RAG pipeline

SYSTEM_PROMPT = """You are a question-answering assistant that must answer \
ONLY using the provided source excerpts. Each excerpt is labeled with its \
source document and chunk id.

Rules:
- If the excerpts don't contain enough information to answer, say so plainly \
  rather than guessing or using outside knowledge.
- Cite which document(s) you used, e.g. "(source: dracula)".
- Be concise. Do not repeat the excerpts verbatim at length; synthesize.
"""


def format_context(retrieved_docs) -> str:
    blocks = []
    for doc in retrieved_docs:
        doc_id = doc.metadata.get("doc_id", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        blocks.append(f"[source: {doc_id} | chunk: {chunk_id}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, store: FAISS, api_key: str | None = None,
                     k: int = 4, model: str = CHAT_MODEL) -> dict:
    """
    Returns {"answer": str, "sources": list[dict], "context": str}
    so the UI can show both the generated answer and what it was grounded in.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Export it or pass api_key explicitly.")

    retrieved = store.similarity_search(question, k=k)
    context = format_context(retrieved)

    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Source excerpts:\n\n{context}\n\nQuestion: {question}"},
    ]
    response = llm.invoke(messages)

    sources = [
        {"doc_id": d.metadata.get("doc_id"), "chunk_id": d.metadata.get("chunk_id"),
         "excerpt": d.page_content[:200] + ("..." if len(d.page_content) > 200 else "")}
        for d in retrieved
    ]

    return {"answer": response.content, "sources": sources, "context": context}
