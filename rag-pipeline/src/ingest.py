"""
ingest.py

Downloads a small public-domain document corpus (4 full novels from
Project Gutenberg, mirrored on GitHub via the GITenberg project) and
splits each into overlapping chunks ready for embedding.

Corpus (all public domain, chosen for topical diversity so retrieval
quality is actually testable across documents):
  - Pride and Prejudice (Jane Austen)
  - The Adventures of Sherlock Holmes (Arthur Conan Doyle)
  - Dracula (Bram Stoker)
  - The Time Machine (H.G. Wells)

To use your own documents/PDFs instead: replace SOURCES below with your
own file paths, or point `load_corpus()` at a folder of .txt/.pdf files.
The rest of the pipeline (chunking, embedding, retrieval, generation) is
source-agnostic.
"""

import os
import re
import urllib.request
from dataclasses import dataclass

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SOURCES = {
    "pride_and_prejudice": "https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/master/1342.txt",
    "sherlock_holmes": "https://raw.githubusercontent.com/GITenberg/The-Adventures-of-Sherlock-Holmes_1661/master/1661.txt",
    "dracula": "https://raw.githubusercontent.com/GITenberg/Dracula_345/master/345.txt",
    "the_time_machine": "https://raw.githubusercontent.com/GITenberg/The-Time-Machine_35/master/35.txt",
}


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


def download_corpus(force: bool = False) -> dict:
    """Downloads each source to data/<doc_id>.txt if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = {}
    for doc_id, url in SOURCES.items():
        path = os.path.join(DATA_DIR, f"{doc_id}.txt")
        if force or not os.path.exists(path):
            print(f"Downloading {doc_id} ...")
            urllib.request.urlretrieve(url, path)
        paths[doc_id] = path
    return paths


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Removes the Project Gutenberg license header/footer, keeping the book body."""
    start_match = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.IGNORECASE)
    end_match = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.IGNORECASE)
    start = start_match.end() if start_match else 0
    end = end_match.start() if end_match else len(text)
    return text[start:end].strip()


def chunk_text(text: str, doc_id: str, chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """
    Simple sliding-window chunker over whitespace-normalized text.
    chunk_size / overlap are in characters (kept simple/dependency-free;
    swap for a token-based splitter if you move to a different model).
    """
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_str = text[start:end]
        chunks.append(Chunk(doc_id=doc_id, chunk_id=f"{doc_id}_{idx}", text=chunk_str))
        idx += 1
        start += chunk_size - overlap
    return chunks


def load_corpus(chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    paths = download_corpus()
    all_chunks = []
    for doc_id, path in paths.items():
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        clean = _strip_gutenberg_boilerplate(raw)
        all_chunks.extend(chunk_text(clean, doc_id, chunk_size, overlap))
    return all_chunks


if __name__ == "__main__":
    chunks = load_corpus()
    print(f"Loaded {len(chunks)} chunks across {len(SOURCES)} documents.")
    for doc_id in SOURCES:
        n = sum(1 for c in chunks if c.doc_id == doc_id)
        print(f"  {doc_id}: {n} chunks")
