"""
Verification node.

Checks, deterministically, whether the generated response:
  1. is supported by the retrieved evidence (lexical overlap heuristic --
     cheap, explainable, and does not require yet another model call),
  2. contains source references,
  3. matches the required output schema (jsonschema validation),
  4. avoids claiming an unsupported action was performed (KB-001 / KB-010).

If verification fails, the graph (src/graph.py) routes to a single
revision attempt; if that also fails, it routes to a safe-failure
response. A `retry_count` guard bounds this to prevent an infinite loop.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator

from .generation import _UNSUPPORTED_ACTION_VERBS
from .state import AgentState, VerificationResult

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "output_schema.json"
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = Draft202012Validator(_SCHEMA)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _evidence_overlap_ratio(answer: str, passages: list[dict]) -> float:
    if not passages:
        return 0.0
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    evidence_tokens: set[str] = set()
    for p in passages:
        evidence_tokens |= _tokenize(p["passage"]) | _tokenize(p["title"])
    if not evidence_tokens:
        return 0.0
    overlap = answer_tokens & evidence_tokens
    return len(overlap) / len(answer_tokens)


MIN_OVERLAP_FOR_ANSWERABLE = 0.18


def build_candidate_response(state: AgentState) -> dict:
    """Assemble the schema-shaped candidate response from current state."""
    passages = state.get("retrieved", [])
    classification = state["classification"]
    sources = [
        {"source_id": p["source_id"], "passage": p["passage"][:300]}
        for p in passages
        if not p["superseded"]
    ] if classification in ("answerable", "requires_escalation") else []

    requires_human = classification in ("requires_escalation", "requires_clarification")

    return {
        "classification": classification,
        "answer": state.get("answer", ""),
        "sources": sources,
        "confidence": state.get("confidence", 0.5),
        "requires_human": requires_human,
        "reason": state.get("triage_reason", ""),
        "clarification_question": state.get("clarification_question"),
        "warnings": state.get("warnings", []),
    }


def verification_node(state: AgentState) -> dict:
    trace = state.get("node_trace", []) + ["verification"]
    candidate = build_candidate_response(state)
    notes: list[str] = []

    # 1. Schema validity
    schema_errors = sorted(_VALIDATOR.iter_errors(candidate), key=lambda e: e.path)
    schema_valid = not schema_errors
    if not schema_valid:
        notes.append("Schema errors: " + "; ".join(e.message for e in schema_errors[:3]))

    # 2. Source references present (only required on evidence-based branches)
    needs_sources = candidate["classification"] in ("answerable", "requires_escalation")
    has_sources = (not needs_sources) or bool(candidate["sources"])
    if needs_sources and not has_sources:
        notes.append("No source references included for an evidence-based response.")

    # 3. Evidence support (only meaningful for the answerable branch)
    if candidate["classification"] == "answerable":
        overlap = _evidence_overlap_ratio(candidate["answer"], state.get("retrieved", []))
        evidence_supported = overlap >= MIN_OVERLAP_FOR_ANSWERABLE
        notes.append(f"Evidence token-overlap ratio: {overlap:.2f} (min {MIN_OVERLAP_FOR_ANSWERABLE}).")
    else:
        evidence_supported = True

    # 4. Unsupported-action guard
    lowered = candidate["answer"].lower()
    unsupported_action_detected = any(phrase in lowered for phrase in _UNSUPPORTED_ACTION_VERBS)
    if unsupported_action_detected:
        notes.append("Answer appears to claim an action the assistant cannot perform.")

    passed = (
        schema_valid
        and has_sources
        and evidence_supported
        and not unsupported_action_detected
        and bool(candidate["answer"].strip())
    )

    verification: VerificationResult = {
        "passed": passed,
        "schema_valid": schema_valid,
        "has_sources": has_sources,
        "evidence_supported": evidence_supported,
        "unsupported_action_detected": unsupported_action_detected,
        "notes": notes,
    }

    return {
        "verification": verification,
        "final_response": candidate,
        "node_trace": trace,
    }


def safe_failure_node(state: AgentState) -> dict:
    """Terminal fallback when a revised answer still fails verification."""
    trace = state.get("node_trace", []) + ["safe_failure"]
    response = {
        "classification": "safe_failure",
        "answer": (
            "I was not able to produce a response I could fully verify against the "
            "supplied OrbitDesk documentation for this question. Please rephrase with "
            "more specific details (affected object, error code, timestamps), or "
            "contact human support directly."
        ),
        "sources": [],
        "confidence": 0.2,
        "requires_human": True,
        "reason": "Verification failed after one revision attempt; returning a safe failure rather than an unverified answer.",
        "clarification_question": None,
        "warnings": state.get("verification", {}).get("notes", []),
    }
    return {"final_response": response, "node_trace": trace}
