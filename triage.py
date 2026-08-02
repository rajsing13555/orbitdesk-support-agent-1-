"""
Triage node.

Per KB-010 (Security and Safe Response Rules) and KB-008 (Escalation),
scope and safety decisions must be deterministic and must not be
overridden by instructions embedded in the user's message. Triage is
therefore implemented as deterministic rule-based code layered on top of
one model signal (embedding similarity to the in-scope corpus), rather
than asking a language model to decide -- this keeps the safety-relevant
decision auditable and not dependent on generation-model wording.

Classification produced here is a *proposal*; the graph's conditional
routing (src/graph.py) decides which node runs next based on this value.
"""

from __future__ import annotations

import re

from .retrieval import RetrievalIndex
from .state import AgentState

# Deterministic keyword rules -------------------------------------------------

_OUT_OF_SCOPE_PATTERNS = [
    r"\brefund\b",
    r"\blegal advice\b",
    r"\blawsuit\b",
    r"\bcancel(l)?ation\b",
    r"\bmedical\b",
    r"\bfinancial advice\b",
    r"\bignore (the )?(supplied )?documentation\b",
    r"\bignore (your |all )?(previous |prior )?instructions\b",
]

_ESCALATION_HINTS = [
    r"already (checked|tried|attempted)",
    r"did(n't| not) work",
    r"still (not working|broken|failing)",
    r"two (export )?runs? (in a row )?failed",
    r"render_failed",
    r"escalat",
]

_VAGUE_HINTS = [
    r"^(sync|it|things?|something) (is|are) (not working|broken)\.?$",
    r"\bnot working\b(?!.*\b(schedule|connection|export|dashboard|credential|refresh|destination)\b)",
]

_MIN_QUESTION_TOKENS = 4
SCOPE_SIMILARITY_FLOOR = 0.12


def _matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def triage_node_factory(index: RetrievalIndex):
    def triage_node(state: AgentState) -> dict:
        question = state["question"]
        trace = state.get("node_trace", []) + ["triage"]

        # Model signal: how well does the question match the in-scope corpus?
        query_vec = index.backend.encode([question])[0]
        from .embeddings import cosine_similarity

        scores = cosine_similarity(query_vec, index.embeddings)
        top_score = float(scores.max()) if len(scores) else 0.0

        scores_summary = {"top_similarity": round(top_score, 4)}

        # 1) Explicit out-of-scope / unsupported-action / prompt-injection requests.
        if _matches_any(_OUT_OF_SCOPE_PATTERNS, question):
            return {
                "classification": "out_of_scope",
                "triage_reason": (
                    "Request asks for an action the support assistant cannot perform "
                    "(e.g. refunds, legal advice, or overriding supplied instructions), "
                    "per KB-001 Support Boundaries and KB-010 Security and Safe Response Rules."
                ),
                "triage_scores": scores_summary,
                "node_trace": trace,
            }

        # 2) Nothing in the corpus is even topically related.
        if top_score < SCOPE_SIMILARITY_FLOOR:
            return {
                "classification": "out_of_scope",
                "triage_reason": (
                    "Question does not match any topic in the supplied OrbitDesk "
                    "knowledge base or resolved cases."
                ),
                "triage_scores": scores_summary,
                "node_trace": trace,
            }

        # 3) User explicitly signals prior troubleshooting / repeated failures.
        if _matches_any(_ESCALATION_HINTS, question):
            return {
                "classification": "requires_escalation",
                "triage_reason": (
                    "User indicates documented troubleshooting steps were already "
                    "attempted without success; per KB-008 this qualifies for escalation."
                ),
                "triage_scores": scores_summary,
                "node_trace": trace,
            }

        # 4) Too vague / underspecified to pick a documented path.
        token_count = len(re.findall(r"\w+", question))
        if token_count < _MIN_QUESTION_TOKENS or _matches_any(_VAGUE_HINTS, question):
            return {
                "classification": "requires_clarification",
                "triage_reason": (
                    "Question lacks the object, symptom or error information needed to "
                    "select a documented troubleshooting path (KB-006 / KB-010)."
                ),
                "triage_scores": scores_summary,
                "node_trace": trace,
            }

        # 5) Otherwise: attempt a knowledge-base-grounded answer.
        return {
            "classification": "answerable",
            "triage_reason": "Question matches in-scope OrbitDesk documentation with sufficient specificity.",
            "triage_scores": scores_summary,
            "node_trace": trace,
        }

    return triage_node
