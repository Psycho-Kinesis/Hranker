"""
Generation: turn retrieved chunks + a student's question into a grounded
answer.

Two modes:
1. LLM mode (preferred) — calls the Claude API if ANTHROPIC_API_KEY is set
   in the environment. The prompt explicitly instructs the model to answer
   ONLY from the provided context and to say so if the context doesn't
   cover the question, which is what keeps a doubt-solving bot from
   hallucinating exam facts.
2. Offline fallback mode — if no API key is available (e.g. running in a
   sandbox with no LLM access), returns a clearly-labeled extractive
   summary of the top retrieved chunk instead of a generated answer. This
   keeps the pipeline runnable end-to-end for demos/testing without a key.
"""

from __future__ import annotations
import os
from typing import List, Dict

SYSTEM_PROMPT = """You are a doubt-solving assistant for students preparing for competitive \
exams (SSC, Banking, Railways, and similar). You will be given a student's question and a \
set of reference notes retrieved from the institute's knowledge base.

Rules:
1. Answer ONLY using the information in the provided reference notes. Do not use outside \
knowledge, even if you believe you know the answer.
2. If the notes do not contain enough information to answer the question, say so explicitly \
instead of guessing.
3. Keep the answer concise and exam-focused — a worked example or formula is more useful \
than a long essay.
4. At the end, cite which note(s) you used by their section title.
"""


def _format_context(chunks: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[Note {i}: {c['doc_title']} — {c['section_title']}]\n{c['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(query: str, chunks: List[Dict]) -> Dict:
    """Returns {'answer': str, 'mode': 'llm' | 'fallback', 'sources': [...]}"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    sources = [f"{c['doc_title']} — {c['section_title']}" for c in chunks]

    if api_key:
        try:
            return _generate_with_claude(query, chunks, api_key, sources)
        except Exception as e:
            # Fall through to offline mode rather than crashing the demo
            fallback = _generate_fallback(query, chunks, sources)
            fallback["answer"] = (
                f"[LLM call failed: {e}. Showing offline fallback instead.]\n\n"
                + fallback["answer"]
            )
            return fallback
    else:
        return _generate_fallback(query, chunks, sources)


def _generate_with_claude(query: str, chunks: List[Dict], api_key: str, sources: List[str]) -> Dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    context = _format_context(chunks)
    user_message = f"Reference notes:\n\n{context}\n\nStudent's question: {query}"

    response = client.messages.create(
        model="claude-sonnet-5",  # check docs.claude.com for the current recommended model string
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    answer_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return {"answer": answer_text, "mode": "llm", "sources": sources}


def _generate_fallback(query: str, chunks: List[Dict], sources: List[str]) -> Dict:
    """No-API-key mode: return the most relevant chunk verbatim with a
    clear label, so the pipeline is still demonstrable end-to-end."""
    if not chunks:
        return {
            "answer": "No relevant notes were found for this question.",
            "mode": "fallback",
            "sources": [],
        }
    top = chunks[0]
    answer = (
        f"(Offline fallback mode — no ANTHROPIC_API_KEY set, so this is the top-matching "
        f"note shown directly rather than an LLM-generated answer.)\n\n"
        f"From \"{top['doc_title']} — {top['section_title']}\":\n\n{top['text']}"
    )
    return {"answer": answer, "mode": "fallback", "sources": sources}
