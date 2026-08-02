"""
Retrieval node: embeds the supplied knowledge-base / resolved-case chunks
once, then ranks them against the incoming question using cosine
similarity from a local Hugging Face embedding model.

Deterministic logic (not model reasoning) decides:
  * how many passages to keep (top-k, plus a minimum-score floor)
  * whether a top-scoring resolved case is superseded, in which case it is
    kept only as a flagged, non-authoritative reference (per README.md /
    KB-001 "Source Priority" and the assignment's data-precedence rule)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data_loader import Chunk, load_all_chunks
from .embeddings import cosine_similarity, get_embedding_backend
from .state import RetrievedPassage

TOP_K = 4
MIN_SCORE = 0.15  # below this, a passage is considered noise, not evidence


class RetrievalIndex:
    """Holds embedded chunks in memory. Built once per process."""

    def __init__(self, data_dir: Path, backend=None):
        self.backend = backend or get_embedding_backend()
        self.chunks: list[Chunk] = load_all_chunks(data_dir)
        texts = [f"{c.title}. {c.heading}. {c.text}" for c in self.chunks]
        self.embeddings: np.ndarray = self.backend.encode(texts)

    def search(self, query: str, top_k: int = TOP_K) -> list[RetrievedPassage]:
        query_vec = self.backend.encode([query])[0]
        scores = cosine_similarity(query_vec, self.embeddings)
        ranked_idx = np.argsort(-scores)[: top_k * 2]  # over-fetch, then filter

        results: list[RetrievedPassage] = []
        for idx in ranked_idx:
            score = float(scores[idx])
            if score < MIN_SCORE and len(results) >= 1:
                continue
            chunk = self.chunks[idx]
            results.append(
                RetrievedPassage(
                    source_id=chunk.source_id,
                    source_type=chunk.source_type,
                    title=chunk.title,
                    passage=chunk.text[:600],
                    score=round(score, 4),
                    superseded=(chunk.status == "superseded"),
                )
            )
            if len(results) >= top_k:
                break
        return results


def retrieval_node_factory(index: RetrievalIndex):
    """Returns a LangGraph node closure bound to a pre-built RetrievalIndex."""

    def retrieval_node(state: dict) -> dict:
        passages = index.search(state["question"])
        trace = state.get("node_trace", []) + ["retrieval"]
        return {"retrieved": passages, "node_trace": trace}

    return retrieval_node
