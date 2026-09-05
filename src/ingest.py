"""
Ingestion pipeline: read raw knowledge-base docs, chunk them by section,
embed each chunk, and persist a searchable index to disk.

Run as: python src/ingest.py
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict
import numpy as np

from embeddings import get_embedder

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def chunk_markdown(text: str, source: str) -> List[Dict]:
    """Split a markdown doc into chunks along '## ' section headers.
    Each chunk keeps its section title and source filename as metadata —
    this is what lets the pipeline cite where an answer came from."""
    # Grab the top-level title (first '# ' line) for context
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else source

    # Split on '## ' section headers, keeping the header text
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# "):
            continue
        header_match = re.match(r"^##\s+(.+)$", section, re.MULTILINE)
        section_title = header_match.group(1).strip() if header_match else "Overview"
        chunks.append({
            "source": source,
            "doc_title": doc_title,
            "section_title": section_title,
            "text": section,
        })
    return chunks


def load_and_chunk_all(raw_dir: Path = RAW_DIR) -> List[Dict]:
    all_chunks = []
    for filepath in sorted(raw_dir.glob("*.md")):
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, source=filepath.name)
        all_chunks.extend(chunks)
    return all_chunks


def build_index(backend: str = "tfidf"):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_and_chunk_all()
    if not chunks:
        raise RuntimeError(f"No chunks found in {RAW_DIR} — add .md files first.")

    texts = [c["text"] for c in chunks]
    embedder = get_embedder(backend)
    vectors = embedder.fit_transform(texts)

    # Persist vectors, metadata, and the fitted embedder itself
    np.save(PROCESSED_DIR / "vectors.npy", vectors)
    with open(PROCESSED_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    import pickle
    with open(PROCESSED_DIR / "embedder.pkl", "wb") as f:
        pickle.dump(embedder, f)

    print(f"Ingested {len(chunks)} chunks from {len(list(RAW_DIR.glob('*.md')))} documents.")
    print(f"Vector shape: {vectors.shape}")
    print(f"Saved index to: {PROCESSED_DIR}")
    return chunks, vectors


if __name__ == "__main__":
    build_index(backend="tfidf")
