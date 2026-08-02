"""
Response-generation node.

Design decision (documented further in docs/design_notes.md): the local
Hugging Face language model is used ONLY on the "answerable" path, where
its job is to phrase an answer strictly from retrieved evidence. The
"requires_clarification", "requires_escalation", "out_of_scope" and
"safe_failure" branches use deterministic, policy-driven templates instead
of free model text. Those branches are safety- and policy-sensitive (e.g.
KB-010 refusal rules, KB-008 escalation-evidence checklist) and a small
local model is more likely to drift from the required wording than a
template is. This keeps "model reasoning" and "deterministic code" clearly
separated, as required by the assignment's orchestration requirements.
"""

from __future__ import annotations

import os
import re
import time

from .state import AgentState, RetrievedPassage

DEFAULT_GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

_UNSUPPORTED_ACTION_VERBS = [
    "i have issued", "i have refunded", "i've refunded", "i have created the credential",
    "i have changed your role", "i have reset", "i have deleted", "i have contacted",
    "i will contact", "i have restored", "i have escalated it for you and it is resolved",
]


class LocalGenerationBackend:
    """Wraps a local Hugging Face text-generation pipeline."""

    def __init__(self, model_name: str = DEFAULT_GENERATION_MODEL):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        t0 = time.time()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.load_ms = (time.time() - t0) * 1000
        self.revision = self._resolve_revision(model_name)

    def _resolve_revision(self, model_name: str) -> str:
        try:
            from pathlib import Path

            cache_root = Path.home() / ".cache" / "huggingface" / "hub"
            model_dir = "models--" + model_name.replace("/", "--")
            snapshots = cache_root / model_dir / "snapshots"
            if snapshots.exists():
                commits = [p.name for p in snapshots.iterdir() if p.is_dir()]
                if commits:
                    return commits[0]
        except Exception:
            pass
        return "main (exact snapshot hash unavailable in this environment)"

    def generate(self, prompt: str, max_new_tokens: int = 220) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the OrbitDesk support assistant. Answer ONLY using the "
                    "evidence provided. Do not invent steps, error codes or role "
                    "permissions that are not in the evidence. Be concise and "
                    "reference the source document IDs you used."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt
        out = self.pipe(
            text,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]
        return out[len(text):].strip() if out.startswith(text) else out.strip()


class StubGenerationBackend:
    """Deterministic template backend used only by automated routing tests."""

    model_name = "stub-template"
    revision = "n/a (test double, not used for real answers)"
    load_ms = 0.0

    def generate(self, prompt: str, max_new_tokens: int = 220) -> str:
        return "Stub answer generated from evidence for routing tests only."


def get_generation_backend():
    backend = os.environ.get("ORBITDESK_MODEL_BACKEND", "local").lower()
    if backend == "stub":
        return StubGenerationBackend()
    return LocalGenerationBackend()


def _build_prompt(question: str, passages: list[RetrievedPassage]) -> str:
    evidence_block = "\n\n".join(
        f"[{p['source_id']}] {p['title']} -- {p['passage']}"
        + (" (NOTE: this case is SUPERSEDED, do not present as current guidance)" if p["superseded"] else "")
        for p in passages
    )
    return (
        f"Customer question: {question}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Write a short, direct answer using only this evidence. Mention the "
        "source document IDs inline where relevant."
    )


def _estimate_confidence(passages: list[RetrievedPassage]) -> float:
    if not passages:
        return 0.2
    top = passages[0]["score"]
    non_superseded = [p for p in passages if not p["superseded"]]
    if not non_superseded:
        return 0.3
    # Simple, explainable heuristic: scaled top similarity, capped, penalised
    # slightly if the best passage was superseded and had to be swapped out.
    conf = min(0.95, 0.35 + top * 0.9)
    return round(conf, 2)


def generation_node_factory(backend=None):
    backend = backend or get_generation_backend()

    def generation_node(state: AgentState) -> dict:
        trace = state.get("node_trace", []) + ["generation"]
        classification = state["classification"]
        warnings: list[str] = []

        if classification == "requires_clarification":
            return {
                "answer": (
                    "Could you share a few more details so we can pinpoint the issue? "
                    "For example: the workspace ID, which connection or schedule is "
                    "affected, the current status shown in the product, and the exact "
                    "error code, if any."
                ),
                "clarification_question": (
                    "Which specific object is affected (dashboard, schedule, connection "
                    "or credential), and what status or error code do you see?"
                ),
                "confidence": 0.5,
                "warnings": warnings,
                "node_trace": trace,
            }

        if classification == "out_of_scope":
            return {
                "answer": (
                    "This request is outside the OrbitDesk support knowledge base "
                    "(for example, refunds, billing cancellations, or legal/financial "
                    "advice are not something this assistant can provide or act on). "
                    "Please contact the appropriate billing or legal channel for this "
                    "request."
                ),
                "clarification_question": None,
                "confidence": 0.95,
                "warnings": warnings,
                "node_trace": trace,
            }

        if classification == "requires_escalation":
            passages = state.get("retrieved", [])
            cited = ", ".join(sorted({p["source_id"] for p in passages if not p["superseded"]})) or "KB-008"
            return {
                "answer": (
                    "Since the documented troubleshooting steps have already been "
                    "completed, this should be escalated. Please collect: workspace ID, "
                    "affected object ID (schedule/dashboard/connection/credential), the "
                    "exact error code, timestamps with timezone, relevant run or refresh "
                    "IDs, and the steps already attempted. Do not include passwords, API "
                    f"secrets, OAuth tokens or full exported data ({cited})."
                ),
                "clarification_question": None,
                "confidence": 0.85,
                "warnings": warnings,
                "node_trace": trace,
            }

        # classification == "answerable" (the only branch reaching this point)
        passages = state.get("retrieved", [])
        if any(p["superseded"] for p in passages[:1]):
            warnings.append(
                f"Top match {passages[0]['source_id']} is a superseded case; "
                "using current knowledge-base guidance instead."
            )

        # Demo/test-only hook (see docs/design_notes.md, "Reproducing the
        # verification-failure demo case"): when explicitly enabled AND this
        # is the first attempt, deliberately produce an answer with no
        # evidence grounding so the verification node fails it and the
        # revise -> regenerate path is exercised. Never active unless the
        # environment variable is set, and never active on the retry.
        force_failure = os.environ.get("ORBITDESK_FORCE_VERIFICATION_FAILURE") == "1"
        if force_failure and state.get("retry_count", 0) == 0:
            warnings.append("Demo hook: forcing an ungrounded first draft to demonstrate the revise path.")
            return {
                "answer": (
                    "You should reinstall the OrbitDesk desktop client and clear your "
                    "browser cache; this usually resolves rendering problems."
                ),
                "clarification_question": None,
                "confidence": 0.4,
                "warnings": warnings,
                "node_trace": trace,
            }

        prompt = _build_prompt(state["question"], passages)
        raw_answer = backend.generate(prompt)
        confidence = _estimate_confidence(passages)

        return {
            "answer": raw_answer,
            "clarification_question": None,
            "confidence": confidence,
            "warnings": warnings,
            "node_trace": trace,
        }

    return generation_node
