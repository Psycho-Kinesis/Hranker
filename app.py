"""
Streamlit demo UI for the doubt-solving RAG assistant.

Run as: streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from ingest import PROCESSED_DIR, build_index
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Doubt Solving Assistant", page_icon="📚", layout="centered")

st.title("📚 Exam Prep Doubt-Solving Assistant")
st.caption(
    "A retrieval-augmented assistant that answers only from a curated knowledge base "
    "of exam-prep notes — built as a demo of the kind of doubt-resolution tool a "
    "B2B edtech platform could offer partner coaching institutes."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Your question is embedded and matched against a knowledge base of "
        "exam-prep notes (Quant, Reasoning, GS).\n"
        "2. The top-matching notes are retrieved.\n"
        "3. An LLM generates an answer **grounded only in those notes**, and "
        "cites the section it used.\n\n"
        "If no `ANTHROPIC_API_KEY` is set, the app falls back to showing the "
        "top-matching note directly instead of an LLM-generated answer."
    )
    st.divider()
    st.markdown("**Sample questions to try:**")
    st.markdown(
        "- What is the successive percentage change formula?\n"
        "- How do pipes and cisterns problems work?\n"
        "- Explain the only-son blood relation trick.\n"
        "- Is the right to property a fundamental right?\n"
        "- Which rivers flow into the Arabian Sea?"
    )


@st.cache_resource
def load_pipeline():
    """Load the pipeline, building the index first if it isn't there yet.

    data/processed/ is gitignored, so a fresh deploy (e.g. Streamlit
    Community Cloud) starts with no index at all. Building it on first run
    takes about a second for this corpus and keeps deployment a single step,
    rather than needing a shell command a hosted runtime doesn't offer.
    """
    if not (PROCESSED_DIR / "vectors.npy").exists():
        with st.spinner("Building the knowledge-base index (first run only)..."):
            build_index()
    return RAGPipeline()


try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error(
        "No index found, and it could not be built automatically. Check that "
        "`data/raw/` contains the knowledge-base markdown files, then run "
        "`python src/ingest.py`."
    )
    st.stop()

query = st.text_input("Ask a doubt:", placeholder="e.g. How do I solve successive percentage change questions?")
k = st.slider("Number of notes to retrieve", 1, 5, 3)

if query:
    with st.spinner("Searching notes and generating answer..."):
        result = pipeline.ask(query, k=k)

    mode_label = "🤖 LLM-generated (Claude)" if result["mode"] == "llm" else "📄 Offline fallback (no API key set)"
    st.markdown(f"**Mode:** {mode_label}")
    st.markdown("### Answer")
    st.write(result["answer"])

    with st.expander("📎 Retrieved source notes"):
        for chunk in result["retrieved_chunks"]:
            st.markdown(f"**{chunk['doc_title']} — {chunk['section_title']}**  \nscore: `{chunk['score']:.3f}`")
            st.text(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))
            st.divider()
