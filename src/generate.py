"""
Generation: turn retrieved chunks + a student's question into a grounded
answer.

Two modes:
1. LLM mode (preferred) — calls Gemini if GEMINI_API_KEY is set, or Claude if
   ANTHROPIC_API_KEY is set. The same grounding prompt is used either way: it
   instructs the model to answer ONLY from the provided context and to say so
   if the context doesn't cover the question, which is what keeps a
   doubt-solving bot from hallucinating exam facts.
2. Offline fallback mode — if no API key is available (e.g. running in a
   sandbox with no LLM access), returns a clearly-labeled extractive
   summary of the top retrieved chunk instead of a generated answer. This
   keeps the pipeline runnable end-to-end for demos/testing without a key.

The provider is chosen by which key is present, so deploying with a different
one is a secrets change rather than a code change. Grounding, citation and
refusal behaviour live outside the provider call and are identical for both.
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


# Overridable, because free-tier model availability changes over time and a
# retired id fails as a 404 rather than degrading to something sensible.
#
# Not the newest model on purpose. Measured over 5 calls each: gemini-3.8-flash
# returned 503 UNAVAILABLE twice (3/5, avg 3.6s) while gemini-3.5-flash and
# gemini-3.6-flash were 5/5 (avg 5.6s and 5.9s). Two seconds of latency is a
# good trade for a demo that does not drop into fallback mode while someone is
# watching it.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"

# Gemini 3.x spends part of this budget reasoning before it writes anything, so
# a limit sized for the answer alone truncates it. Measured: 500 produced half
# a sentence. Claude's budget is separate because it is not drawn down the same
# way, and the grounding prompt keeps answers short on both.
GEMINI_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "2048"))


def _format_context(chunks: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[Note {i}: {c['doc_title']} — {c['section_title']}]\n{c['text']}"
        )
    return "\n\n".join(blocks)


def _build_user_message(query: str, chunks: List[Dict]) -> str:
    """The retrieved notes and the question, in one string. Shared by both
    providers so the grounding they receive is identical."""
    return f"Reference notes:\n\n{_format_context(chunks)}\n\nStudent's question: {query}"


# Below this cosine score nothing in the corpus is genuinely related. Measured,
# not guessed: the 12 labelled eval questions score 0.697 at worst, while
# clearly out-of-corpus questions ("explain photosynthesis", "who won the 2018
# World Cup") score 0.000, so anything in this gap is safe.
MIN_USEFUL_SCORE = 0.35

REFUSAL = (
    "I can't answer this from the current knowledge base — nothing in the "
    "notes is close enough to your question.\n\n"
    "The knowledge base covers percentages, time and work, blood relations, "
    "coding-decoding, fundamental rights, and major rivers of India."
)


def _no_topical_evidence(matched_terms) -> bool:
    """True when the retriever recognised nothing, or only bare numbers.

    The score threshold alone cannot catch "solve x + y = 10 and 2x = 8": the
    vectoriser drops the algebra and keeps "10", which matches "decreases the
    new price by 10%" and scores 0.989 — above every labelled eval question.

    A number is the giveaway. "10" or "1978" turns up incidentally in notes on
    any subject, so matching one says nothing about topic, whereas matching
    "percentage" does. IDF cannot express this: "10" scores 3.71 in this
    corpus, the highest of any term, while "percentage" scores 2.46 — so
    weighting by rarity would reject the good question and accept the bad one.

    Only fires when numbers are the *whole* of the evidence. "Article 32
    meaning" matches "article" as well as "32", so it is answered normally.
    """
    if matched_terms is None:
        return False  # backend has no vocabulary to miss (dense embeddings)
    return all(
        all(word.isnumeric() for word in term.split()) for term in matched_terms
    )


def generate_answer(query: str, chunks: List[Dict], matched_terms=None) -> Dict:
    """Returns {'answer': str, 'mode': 'llm' | 'fallback', 'sources': [...]}"""
    sources = [f"{c['doc_title']} — {c['section_title']}" for c in chunks]

    # Refuse before generating, in both modes. Sending notes the question has
    # no real overlap with invites the model to make a connection that isn't
    # there, and it spends an API call to do it.
    if not chunks or chunks[0].get("score", 0.0) < MIN_USEFUL_SCORE \
            or _no_topical_evidence(matched_terms):
        return {"answer": REFUSAL, "mode": "fallback", "refused": True, "sources": []}

    provider, api_key = _select_provider()
    if provider:
        try:
            return _PROVIDERS[provider](query, chunks, api_key, sources)
        except Exception as e:
            # Fall through to offline mode rather than crashing the demo
            fallback = _generate_fallback(query, chunks, sources)
            fallback["answer"] = (
                f"[{provider} call failed: {e}. Showing offline fallback instead.]\n\n"
                + fallback["answer"]
            )
            return fallback
    return _generate_fallback(query, chunks, sources)


def _select_provider():
    """Pick a provider from whichever key is set, so switching provider is a
    secrets change rather than a code change. GOOGLE_API_KEY is accepted too
    because the Google SDK and its docs use that name interchangeably."""
    for provider, variables in (
        ("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
        ("claude", ("ANTHROPIC_API_KEY",)),
    ):
        for variable in variables:
            key = os.environ.get(variable)
            if key:
                return provider, key
    return None, None


def _generate_with_gemini(query: str, chunks: List[Dict], api_key: str, sources: List[str]) -> Dict:
    """Google Gemini, via the `google-genai` SDK.

    The model is overridable because free-tier model availability changes and
    an unavailable id is a 404 rather than a fallback to something sensible.
    If this errors, set GEMINI_MODEL to a model your key can reach.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        contents=_build_user_message(query, chunks),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            # Gemini 3.x reasons before answering and those thinking tokens are
            # drawn from this same budget, so a limit sized for the visible
            # answer alone gets spent thinking and truncates mid-sentence. This
            # is deliberately generous; the prompt keeps answers short.
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        ),
    )

    # .text is None when the response carries no text part — a safety block or
    # a candidate that stopped before emitting one. Raising here routes into
    # the offline fallback rather than rendering an empty answer as if it were
    # a real one.
    text = response.text
    if not text:
        blocked = getattr(response.prompt_feedback, "block_reason", None)
        raise RuntimeError(f"no text returned (block_reason={blocked})")

    # A truncated answer still arrives as ordinary text, so say so rather than
    # presenting half a sentence as a complete one.
    if _hit_token_limit(response):
        text += "\n\n_(Answer truncated — it hit the output token limit.)_"
    return {"answer": text, "mode": "llm", "provider": "gemini",
            "refused": False, "sources": sources}


def _hit_token_limit(response) -> bool:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return False
    return str(getattr(candidates[0], "finish_reason", "")).endswith("MAX_TOKENS")


def _generate_with_claude(query: str, chunks: List[Dict], api_key: str, sources: List[str]) -> Dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL),
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(query, chunks)}],
    )
    answer_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    if not answer_text:
        raise RuntimeError(f"no text returned (stop_reason={response.stop_reason})")
    return {"answer": answer_text, "mode": "llm", "provider": "claude",
            "refused": False, "sources": sources}


_PROVIDERS = {"gemini": _generate_with_gemini, "claude": _generate_with_claude}


def _generate_fallback(query: str, chunks: List[Dict], sources: List[str]) -> Dict:
    """No-API-key mode: return the most relevant chunk verbatim with a
    clear label, so the pipeline is still demonstrable end-to-end.

    Callers reach this only after generate_answer has cleared the refusal
    checks, so the chunks here are known to be topically relevant.
    """
    if not chunks:
        return {"answer": REFUSAL, "mode": "fallback", "refused": True, "sources": []}
    top = chunks[0]
    answer = (
        f"(Offline fallback mode — no LLM available, so this is the top-matching "
        f"note shown directly rather than a generated answer. Set GEMINI_API_KEY "
        f"or ANTHROPIC_API_KEY to enable generation.)\n\n"
        f"From \"{top['doc_title']} — {top['section_title']}\":\n\n{top['text']}"
    )
    return {"answer": answer, "mode": "fallback", "refused": False, "sources": sources}
