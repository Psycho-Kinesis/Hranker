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


def evaluate(k: int = 3):
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    retriever = Retriever()
    hits = 0
    top1_hits = 0
    results = []

    for item in questions:
        retrieved = retriever.search(item["query"], k=k)
        retrieved_sources = [r["source"] for r in retrieved]
        hit = item["expected_source"] in retrieved_sources
        top1_hit = retrieved_sources[0] == item["expected_source"] if retrieved_sources else False
        hits += hit
        top1_hits += top1_hit
        results.append({
            "query": item["query"],
            "expected": item["expected_source"],
            "retrieved_top1": retrieved_sources[0] if retrieved_sources else None,
            "hit_at_k": hit,
        })

    n = len(questions)
    print(f"Evaluated {n} questions, k={k}\n")
    for r in results:
        marker = "PASS" if r["hit_at_k"] else "FAIL"
        print(f"[{marker}] {r['query'][:60]:<60} -> got: {r['retrieved_top1']}")

    print(f"\nHit rate@{k}: {hits}/{n} = {hits/n:.1%}")
    print(f"Top-1 accuracy: {top1_hits}/{n} = {top1_hits/n:.1%}")
    return {"hit_rate_at_k": hits / n, "top1_accuracy": top1_hits / n, "n": n, "k": k}


if __name__ == "__main__":
    evaluate(k=3)
