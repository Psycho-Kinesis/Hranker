"""
Top-level RAG pipeline: wires retrieval + generation into a single
`ask()` call. This is the module both the CLI and the Streamlit app use.
"""

from __future__ import annotations
from typing import Dict
from retrieve import Retriever
from generate import generate_answer


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()

    def ask(self, query: str, k: int = 3) -> Dict:
        retrieved = self.retriever.search(query, k=k)
        matched = self._matched_terms(query)
        result = generate_answer(query, retrieved, matched_terms=matched)
        result["retrieved_chunks"] = [] if result.get("refused") else retrieved
        result["matched_terms"] = matched
        return result

    def _matched_terms(self, query: str):
        """Which query terms the retriever actually recognised, when the
        backend can say. None means the backend has no vocabulary concept."""
        getter = getattr(self.retriever.embedder, "matched_terms", None)
        return getter(query) if callable(getter) else None


if __name__ == "__main__":
    import sys

    pipeline = RAGPipeline()
    query = " ".join(sys.argv[1:]) or "What is the successive percentage change formula?"
    print(f"Q: {query}\n")

    result = pipeline.ask(query)
    print(f"A ({result['mode']} mode):\n{result['answer']}\n")
    terms = result.get("matched_terms")
    if terms is not None:
        print(f"Matched query terms: {', '.join(terms) if terms else '(none)'}")
    print("Sources:")
    for c in result["retrieved_chunks"]:
        print(f"  [{c['score']:.3f}] {c['source']} — {c['section_title']}")
