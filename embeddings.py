"""
Local embedding backend used by the retrieval node.

Two backends are supported, selected via the ORBITDESK_MODEL_BACKEND
environment variable:

  * "local"  (default) -- loads a real Hugging Face model through
    sentence-transformers. This is the backend used for the actual
    assignment run and requires the model to have been downloaded once
    while network access was available.

  * "stub"   -- a deterministic, dependency-free bag-of-words cosine
    similarity model. It is used only by the automated routing tests
    (tests/test_graph_routing.py) so that graph *routing* can be verified
    quickly, in CI, and without depending on model wording or a model
    download. It is never used for the real demo runs or submitted
    sample outputs.

Both backends expose the same tiny interface: `.encode(texts) -> np.ndarray`
and `.model_name`, `.revision`, `.load_ms`, so the retrieval node does not
need to know which backend is active.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class StubEmbeddingBackend:
    """Deterministic bag-of-words backend used only for fast routing tests."""

    model_name = "stub-bow-cosine"
    revision = "n/a (test double, not used for real answers)"

    def __init__(self) -> None:
        t0 = time.time()
        self._vocab: dict[str, int] = {}
        self._frozen = False
        self.load_ms = (time.time() - t0) * 1000

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def encode(self, texts: list[str]) -> np.ndarray:
        tokenized = [self._tokenize(t) for t in texts]
        if not self._frozen:
            for tokens in tokenized:
                for tok in tokens:
                    self._vocab.setdefault(tok, len(self._vocab))
            # The first encode() call is always the corpus build in
            # RetrievalIndex.__init__; freeze the vocabulary afterwards so
            # later single-query encodes return vectors of the same fixed
            # dimension (out-of-vocabulary query words are simply ignored,
            # which is an acceptable trade-off for a test-only stub).
            self._frozen = True
        dim = max(len(self._vocab), 1)
        matrix = np.zeros((len(texts), dim), dtype="float32")
        for i, tokens in enumerate(tokenized):
            counts = Counter(tokens)
            for tok, count in counts.items():
                idx = self._vocab.get(tok)
                if idx is not None:
                    matrix[i, idx] = count
        return matrix


class LocalSentenceTransformerBackend:
    """Real local Hugging Face embedding model via sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        t0 = time.time()
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.load_ms = (time.time() - t0) * 1000
        self.revision = self._resolve_revision()

    def _resolve_revision(self) -> str:
        """Best-effort lookup of the exact snapshot commit hash that was
        loaded from the local Hugging Face cache, so the run can report an
        exact revision rather than a mutable branch name such as 'main'."""
        try:
            cache_root = Path.home() / ".cache" / "huggingface" / "hub"
            model_dir = "models--" + self.model_name.replace("/", "--")
            snapshots = cache_root / model_dir / "snapshots"
            if snapshots.exists():
                commits = [p.name for p in snapshots.iterdir() if p.is_dir()]
                if commits:
                    return commits[0]
        except Exception:
            pass
        return "main (exact snapshot hash unavailable in this environment)"

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)


def get_embedding_backend():
    backend = os.environ.get("ORBITDESK_MODEL_BACKEND", "local").lower()
    if backend == "stub":
        return StubEmbeddingBackend()
    return LocalSentenceTransformerBackend()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector `a` (1D) and matrix `b` (2D)."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return b_norm @ a_norm
