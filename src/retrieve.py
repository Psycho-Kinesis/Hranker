"""
Retrieval: given a query, embed it with the same fitted embedder used at
ingestion time, and return the top-k most similar chunks by cosine
similarity (vectors are pre-normalized, so this is just a dot product).
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


class Retriever:
    def __init__(self, processed_dir: Path = PROCESSED_DIR):
        self.vectors: np.ndarray = np.load(processed_dir / "vectors.npy")
        with open(processed_dir / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks: List[Dict] = json.load(f)
        with open(processed_dir / "embedder.pkl", "rb") as f:
            self.embedder = pickle.load(f)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        query_vec = self.embedder.transform([query])[0]  # shape (d,)
        scores = self.vectors @ query_vec  # cosine similarity, vectors are normalized
        top_k_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_idx:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results


if __name__ == "__main__":
    retriever = Retriever()
    test_query = "How do I calculate successive percentage change?"
    for r in retriever.search(test_query, k=3):
        print(f"[{r['score']:.3f}] {r['source']} — {r['section_title']}")
