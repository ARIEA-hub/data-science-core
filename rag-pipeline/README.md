# RAG Pipeline — Q&A over Public-Domain Novels

A small retrieval-augmented generation pipeline: ingest documents → chunk →
embed → store in FAISS → retrieve top-k on a question → ask an LLM to answer
using only what was retrieved, with citations.

## Stack

- **LangChain** for the embeddings/vector-store/chat-model glue
- **OpenAI API** — `text-embedding-3-small` for embeddings, `gpt-4o-mini` for generation
- **FAISS** (local, in-process) as the vector store
- **Streamlit** for the chat UI

## Corpus

Four public-domain novels, downloaded automatically from GitHub (GITenberg
mirrors of Project Gutenberg texts):

- *Pride and Prejudice* — Jane Austen
- *The Adventures of Sherlock Holmes* — Arthur Conan Doyle
- *Dracula* — Bram Stoker
- *The Time Machine* — H.G. Wells

Chosen deliberately for topical/tonal diversity so retrieval quality is
actually testable (a question about detectives should retrieve Sherlock
Holmes chunks, not Austen).

**To use your own documents:** edit `SOURCES` in `src/ingest.py`, or point
`load_corpus()` at a local folder of `.txt`/`.pdf` files instead. Nothing
downstream (chunking, embedding, retrieval, generation) needs to change.

## Run it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...          # or paste it into the app's sidebar
streamlit run app.py
```

First run downloads the corpus and builds the FAISS index (small embedding
API cost); the index is persisted to `vector_store/` so later runs load
instantly instead of re-embedding.

## Grounding & citations

The system prompt instructs the model to answer **only** from retrieved
excerpts and to say so if the excerpts don't cover the question, rather than
falling back on its own training knowledge. Every answer in the UI has an
expandable "Sources used" section showing exactly which document/chunk the
model was given — so a hallucinated claim is checkable against real text.

## Known limitation

This was built in a sandboxed dev environment with no network access to
`api.openai.com`, so the OpenAI-dependent parts (embeddings, generation)
could not be smoke-tested end-to-end here. The FAISS retrieval plumbing
*was* verified end-to-end using a deterministic mock embedding function
(see commit history). Test with a real key before relying on it.
