"""
Embedding backends for the RAG pipeline.

Two backends are provided:

1. TfidfEmbedder — pure scikit-learn TF-IDF + SVD, runs fully offline with no
   external model download. This is the default so the project runs anywhere,
   including sandboxed / no-internet environments.

2. SentenceTransformerEmbedder — wraps `sentence-transformers` for true dense
   semantic embeddings (recommended upgrade once you're running locally with
   full internet access, since it captures meaning rather than just word
   overlap). Swap it in by changing one line in ingest.py / rag_pipeline.py.

Both expose the same interface: fit(texts), transform(texts) -> np.ndarray,
so the rest of the pipeline never needs to know which one is active.
"""

from __future__ import annotations
import numpy as np
from typing import List


class TfidfEmbedder:
    """Offline-friendly embedder: TF-IDF vectors compressed with truncated SVD
    (a.k.a. Latent Semantic Analysis) so we still capture some topic-level
    similarity rather than pure keyword overlap, without needing to download
    any pretrained weights."""

    def __init__(self, n_components: int = 100, random_state: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
        )
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        # SVD components can't exceed min(n_samples, n_features) - 1
        n_comp = min(self.n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        n_comp = max(n_comp, 2)
        if n_comp != self.svd.n_components:
            from sklearn.decomposition import TruncatedSVD
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf_matrix)
        self._fitted = True

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        tfidf_matrix = self.vectorizer.transform(texts)
        vectors = self.svd.transform(tfidf_matrix)
        # L2-normalize so cosine similarity == dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return (vectors / norms).astype("float32")

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def matched_terms(self, text: str) -> List[str]:
        """The query terms that actually exist in the fitted vocabulary.

        Cosine similarity cannot express "I barely understood the question".
        L2 normalisation turns a query backed by one stray token and a query
        backed by ten strong ones into unit vectors alike, so a question that
        matched only an incidental number can score higher than a real one.
        Reporting which terms matched exposes that directly, without needing a
        confidence threshold that this representation cannot honestly provide.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before matched_terms().")
        analyzer = self.vectorizer.build_analyzer()
        vocabulary = self.vectorizer.vocabulary_
        seen, matched = set(), []
        for token in analyzer(text):
            if token in vocabulary and token not in seen:
                seen.add(token)
                matched.append(token)
        return matched


class SentenceTransformerEmbedder:
    """Dense semantic embeddings via sentence-transformers. Requires
    internet access to download the model weights on first use (e.g.
    'all-MiniLM-L6-v2') — use this when running locally/on Colab, not
    inside a network-restricted sandbox."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]) -> None:
        pass  # no fitting needed, model is pretrained

    def transform(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def matched_terms(self, text: str) -> List[str]:
        """Not applicable: a dense model has no vocabulary to miss, and
        embeds any text into a meaningful vector. Returning None (rather than
        an empty list) lets callers tell "no terms matched" apart from "this
        backend cannot answer the question"."""
        return None


def get_embedder(backend: str = "tfidf"):
    """Factory so the rest of the code can switch backends via a config
    string instead of importing classes directly."""
    if backend == "tfidf":
        return TfidfEmbedder()
    elif backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")
