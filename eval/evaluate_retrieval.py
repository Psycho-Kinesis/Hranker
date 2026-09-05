"""
Evaluate retrieval quality: for each labeled question, check whether the
correct source document appears in the top-k retrieved chunks.

This reports real, measured numbers on the current index — nothing here is
a placeholder. Re-run after changing the embedding backend or chunking
strategy to see the actual effect.

Run as: python eval/evaluate_retrieval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from retrieve import Retriever  # noqa: E402

EVAL_FILE = Path(__file__).parent / "eval_questions.json"


def separation_margin(retriever, query: str, expected_source: str):
    """Score gap between the best chunk from the expected document and the
    best chunk from any other document.

    Hit-rate and top-1 accuracy are both saturated at 100% on this corpus, so
    neither can tell two retrievers apart. This can: it measures *how far*
    the right document beat the competition, not just whether it did. A
    positive margin means the correct document won; a shrinking margin is the
    early warning that retrieval is about to start failing, which is what a
    larger corpus with overlapping vocabulary would cause.
    """
    ranked = retriever.search(query, k=len(retriever.chunks))
    best_expected = next((r["score"] for r in ranked if r["source"] == expected_source), None)
    best_other = next((r["score"] for r in ranked if r["source"] != expected_source), None)
    if best_expected is None or best_other is None:
        return None
    return best_expected - best_other


def evaluate(k: int = 3):
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    retriever = Retriever()
    hits = 0
    top1_hits = 0
    results = []
    margins = []

    for item in questions:
        retrieved = retriever.search(item["query"], k=k)
        retrieved_sources = [r["source"] for r in retrieved]
        hit = item["expected_source"] in retrieved_sources
        top1_hit = retrieved_sources[0] == item["expected_source"] if retrieved_sources else False
        hits += hit
        top1_hits += top1_hit
        margin = separation_margin(retriever, item["query"], item["expected_source"])
        if margin is not None:
            margins.append(margin)
        results.append({
            "query": item["query"],
            "expected": item["expected_source"],
            "retrieved_top1": retrieved_sources[0] if retrieved_sources else None,
            "hit_at_k": hit,
            "margin": margin,
        })

    n = len(questions)
    print(f"Evaluated {n} questions, k={k}\n")
    for r in results:
        marker = "PASS" if r["hit_at_k"] else "FAIL"
        margin = f"{r['margin']:+.3f}" if r["margin"] is not None else "  n/a"
        print(f"[{marker}] margin {margin}  {r['query'][:52]:<52} -> {r['retrieved_top1']}")

    mean_margin = sum(margins) / len(margins) if margins else 0.0
    min_margin = min(margins) if margins else 0.0

    print(f"\nHit rate@{k}: {hits}/{n} = {hits/n:.1%}")
    print(f"Top-1 accuracy: {top1_hits}/{n} = {top1_hits/n:.1%}")
    print(f"Mean separation margin: {mean_margin:+.3f}   (higher is better)")
    print(f"Min separation margin:  {min_margin:+.3f}   (the closest call)")
    return {
        "hit_rate_at_k": hits / n,
        "top1_accuracy": top1_hits / n,
        "mean_separation_margin": mean_margin,
        "min_separation_margin": min_margin,
        "n": n,
        "k": k,
    }


if __name__ == "__main__":
    evaluate(k=3)
