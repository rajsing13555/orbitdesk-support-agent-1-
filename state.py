"""
Shared typed state for the OrbitDesk Support Agent graph.

Every node reads from and writes to this single TypedDict. LangGraph merges
partial updates returned by each node into this shared state, so nodes never
pass ad-hoc arguments to one another -- this is what the assignment calls
"shared typed state".
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict


Classification = Literal[
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
    "safe_failure",
]


class RetrievedPassage(TypedDict):
    source_id: str          # e.g. "KB-004" or "CASE-1103"
    source_type: str        # "knowledge_base" | "resolved_case"
    title: str
    passage: str             # short excerpt used as evidence
    score: float              # cosine similarity, 0..1
    superseded: bool          # True for resolved cases marked "superseded"


class VerificationResult(TypedDict):
    passed: bool
    schema_valid: bool
    has_sources: bool
    evidence_supported: bool
    unsupported_action_detected: bool
    notes: list[str]


class AgentState(TypedDict, total=False):
    # ---- input ----
    question_id: str
    question: str

    # ---- triage ----
    classification: Classification
    triage_reason: str
    triage_scores: dict[str, float]

    # ---- retrieval ----
    retrieved: list[RetrievedPassage]

    # ---- generation ----
    answer: str
    confidence: float
    clarification_question: Optional[str]
    warnings: list[str]

    # ---- verification / control flow ----
    verification: VerificationResult
    retry_count: int
    max_retries: int
    final_response: dict[str, Any]

    # ---- observability ----
    node_trace: list[str]     # ordered list of node names that executed
    timings_ms: dict[str, float]
