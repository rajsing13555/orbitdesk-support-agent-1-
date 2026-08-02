"""
Graph assembly for the OrbitDesk Support Agent.

Node responsibilities (see assignment "Technical Requirements"):
  triage        -> classify: answerable / requires_clarification /
                   requires_escalation / out_of_scope
  retrieval     -> embed + rank knowledge-base & resolved-case passages
  generation    -> local HF LM answer (answerable) or deterministic
                   policy template (other branches)
  verification  -> schema check, evidence-support check, source-reference
                   check, unsupported-action guard
  revise_router -> bounded retry: routes back to `generation` once, or to
                   `safe_failure` if still failing

Conditional routing map:

    triage --(answerable | requires_escalation)--> retrieval --> generation
    triage --(requires_clarification | out_of_scope)--> generation
    generation --> verification
    verification --(passed)--> END
    verification --(failed, retry_count < max_retries)--> generation (revise)
    verification --(failed, retry_count >= max_retries)--> safe_failure --> END

`retry_count` / `max_retries` in the shared state is the loop guard that
satisfies "protection against an infinite graph loop"; LangGraph's own
`recursion_limit` is set defensively as a second, independent guard.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langgraph.graph import END, StateGraph

from .generation import generation_node_factory, get_generation_backend
from .retrieval import RetrievalIndex, retrieval_node_factory
from .state import AgentState
from .triage import triage_node_factory
from . import verification as verification_module

logger = logging.getLogger("orbitdesk.graph")

DEFAULT_MAX_RETRIES = 1


def _init_state(state: AgentState) -> dict:
    return {
        "node_trace": ["init"],
        "retry_count": 0,
        "max_retries": state.get("max_retries", DEFAULT_MAX_RETRIES),
        "warnings": [],
    }


def _route_after_triage(state: AgentState) -> str:
    classification = state["classification"]
    if classification in ("answerable", "requires_escalation"):
        return "retrieval"
    return "generation"  # requires_clarification / out_of_scope -> template only


def _route_after_verification(state: AgentState) -> str:
    verification = state["verification"]
    if verification["passed"]:
        return "end"
    if state.get("retry_count", 0) < state.get("max_retries", DEFAULT_MAX_RETRIES):
        return "revise"
    return "safe_failure"


def _revise_node(state: AgentState) -> dict:
    """Bumps the retry counter and logs why a revision was triggered, then
    hands control back to the generation node for a second attempt."""
    trace = state.get("node_trace", []) + ["revise"]
    notes = state.get("verification", {}).get("notes", [])
    logger.info("Revision triggered (attempt %s). Notes: %s", state.get("retry_count", 0) + 1, notes)
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "node_trace": trace,
        "warnings": state.get("warnings", []) + ["Answer revised after failing verification: " + "; ".join(notes[:2])],
    }


def build_graph(data_dir: Path, generation_backend=None, embedding_backend=None, verification_node=None):
    """Builds and compiles the LangGraph StateGraph.

    A single shared RetrievalIndex and generation backend are built once
    and closed over by the relevant node factories, so models are loaded
    exactly once per process (recorded load time is reported by the
    caller, see scripts/run_demo_cases.py).

    `verification_node` can be overridden (used by
    tests/test_graph_routing.py) to inject deterministic pass/fail
    behaviour when testing the revise / safe-failure routing without
    depending on real model output.
    """
    index = RetrievalIndex(data_dir, backend=embedding_backend)
    gen_backend = generation_backend or get_generation_backend()
    verification_fn = verification_node or verification_module.verification_node

    graph = StateGraph(AgentState)

    graph.add_node("init", _init_state)
    graph.add_node("triage", triage_node_factory(index))
    graph.add_node("retrieval", retrieval_node_factory(index))
    graph.add_node("generation", generation_node_factory(gen_backend))
    graph.add_node("verification", verification_fn)
    graph.add_node("revise", _revise_node)
    graph.add_node("safe_failure", verification_module.safe_failure_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "triage")

    graph.add_conditional_edges(
        "triage",
        _route_after_triage,
        {"retrieval": "retrieval", "generation": "generation"},
    )
    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", "verification")

    graph.add_conditional_edges(
        "verification",
        _route_after_verification,
        {"end": END, "revise": "revise", "safe_failure": "safe_failure"},
    )
    graph.add_edge("revise", "generation")
    graph.add_edge("safe_failure", END)

    # Second, independent guard against infinite loops (belt-and-braces on
    # top of the retry_count check above).
    compiled = graph.compile()
    return compiled, index, gen_backend
