# OrbitDesk Support Agent -- Local-First Support Agent Network

A graph-orchestrated support agent for the fictional **OrbitDesk** product,
built for the AI Engineer Internship assignment. The full workflow --
triage, retrieval, generation and verification -- runs on locally loaded
Hugging Face models via **LangGraph**. No remote language-model API is
called at any point.

## 1. Architecture

```
init -> triage -> retrieval -> generation -> verification -> END (verified)
          |                                        |
          | (clarification / out-of-scope)         | (failed, retries left)
          v                                        v
       generation (template)                     revise -> generation (retry)
          |                                        |
          v                                        | (failed, no retries left)
     verification -> END                            v
                                               safe_failure -> END
```

See `diagrams/graph_diagram.png` for the rendered diagram (regenerate any
time with `python diagrams/generate_diagram.py`).

| Node | Responsibility | Model / code |
|---|---|---|
| `triage` | Classify the request: `answerable`, `requires_clarification`, `requires_escalation`, `out_of_scope` | Deterministic keyword rules **+** one local embedding-similarity signal (in-scope check). Kept rule-based because safety/scope decisions must be auditable (KB-010). |
| `retrieval` | Rank knowledge-base sections and resolved cases against the question | Local Hugging Face **embedding** model (`sentence-transformers/all-MiniLM-L6-v2`) + cosine similarity (deterministic code) |
| `generation` | Produce the answer | Local Hugging Face **causal LM** (`Qwen/Qwen2.5-0.5B-Instruct`) for the `answerable` branch only; deterministic policy templates for clarification / escalation / out-of-scope (see `docs/design_notes.md` for why) |
| `verification` | Schema check, source-reference check, evidence-overlap check, unsupported-action guard | Deterministic code (`jsonschema` + lexical overlap heuristic) |
| `revise` / `safe_failure` | Bounded retry (max 1) then safe terminal fallback | Deterministic code; `retry_count` in shared state + LangGraph `recursion_limit=25` are two independent loop guards |

Shared typed state is defined in `src/state.py` (`AgentState`, a
`TypedDict`) and flows through every node — this is what the assignment
calls "shared typed state." `node_trace` on that state is appended to by
every node and is the execution log referenced in the video-recording
requirement.

## 2. Repository layout

```
data/                     Supplied knowledge base, resolved cases, schema (copied, unmodified)
src/
  state.py                Shared TypedDict state
  data_loader.py           Parses KB markdown + resolved cases into retrievable chunks
  embeddings.py            Local embedding backend (+ stub backend for tests)
  retrieval.py             Embedding-based retrieval node
  triage.py                 Rule + embedding-based triage node
  generation.py             Local LM generation node (+ stub backend for tests)
  verification.py           Schema / evidence / safety verification node
  graph.py                   LangGraph StateGraph assembly and routing
scripts/
  run_cli.py                Interactive CLI for new natural-language questions
  run_demo_cases.py          Runs the 5 (+1 bonus) required test cases, writes sample_outputs/
tests/
  test_graph_routing.py      Automated routing tests (stub backend, wording-independent)
diagrams/
  generate_diagram.py         Regenerates graph_diagram.png
  graph_diagram.png
sample_outputs/
  demo_results.json           Output of the last `run_demo_cases.py` run
docs/
  design_notes.md              Trade-offs, limitations, "what I'd improve with more time"
```

## 3. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run of `run_cli.py` or `run_demo_cases.py` downloads the two
models below from the Hugging Face Hub (requires internet **once**).
Every run after that is fully offline (see Section 6).

## 4. Models used

| Role | Model | Notes |
|---|---|---|
| Embedding / retrieval | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, CPU-friendly, well-established for semantic search |
| Response generation | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B params, instruction-tuned, runs comfortably on CPU; quantized/larger models can be swapped in via `DEFAULT_GENERATION_MODEL` in `src/generation.py` |

**Exact revisions:** rather than hardcoding a Hugging Face branch name
such as `"main"` (which is mutable and therefore not a precise revision),
both backends resolve and log the **exact local snapshot commit hash**
from the Hugging Face cache directory the first time they load a model
(`_resolve_revision()` in `src/embeddings.py` and `src/generation.py`).
This is printed at the start of every `run_cli.py` / `run_demo_cases.py`
run and is included in `sample_outputs/demo_results.json` under
`embedding_model.revision` / `generation_model.revision`. Record the
values printed on your machine in your submission notes.

**Load time and latency:** both scripts print model load time
(`load_ms`) and per-question end-to-end latency (`latency_ms`); these are
also saved in `sample_outputs/demo_results.json`.

## 5. Running it

```bash
# Interactive CLI
python scripts/run_cli.py
python scripts/run_cli.py --question "Can a Viewer create API credentials?"

# Required demonstration cases (writes sample_outputs/demo_results.json)
python scripts/run_demo_cases.py

# Automated routing tests (fast, stub backend, no model download)
ORBITDESK_MODEL_BACKEND=stub pytest tests/ -v
```

## 6. Running fully offline

After the first successful run (which downloads the two models above),
disable networking and re-run `scripts/run_cli.py` or
`scripts/run_demo_cases.py` -- both will load the models from the local
Hugging Face cache (`~/.cache/huggingface`) with no network calls.

## 7. Required test cases

`scripts/run_demo_cases.py` runs six cases (the five required by the
assignment, plus one bonus escalation case):

1. **Directly answerable** -- Viewer / API credential question (single document, `KB-005` + `KB-002`).
2. **Requires two documents** -- timezone-change export question (`KB-003` + `KB-004`).
3. **Ambiguous, needs clarification** -- "sync is not working" (routes to `requires_clarification`, skips retrieval).
4. **Out of scope** -- refund / legal-advice / prompt-injection attempt (routes to `out_of_scope`, skips retrieval and the language model entirely).
5. **Initial answer fails verification** -- run with `ORBITDESK_FORCE_VERIFICATION_FAILURE=1` set for that one case only, so the first draft is deliberately ungrounded, `verification` fails it, `revise` runs once, and the second draft passes. See `docs/design_notes.md` for why this demo hook exists and how it's gated.
6. **Bonus: escalation** -- two consecutive `render_failed` events (`KB-004` + `KB-008`).

`tests/test_graph_routing.py` additionally asserts, without depending on
any generated wording, that: the clarification and out-of-scope branches
never call `retrieval`; the escalation branch does call it; a forced
verification failure produces exactly one `revise` + a second
`generation` call; and a *persistent* verification failure still
terminates (at `safe_failure`, with `retry_count` capped at
`max_retries`) instead of looping forever.

## 8. Output schema

`data/output_schema.json` (supplied) is used as-is for validation in
`src/verification.py`. Two small additive fields were kept from the
schema as delivered: `clarification_question` and `warnings`, both
optional, to carry the clarification prompt and any verification/routing
notes without breaking the required fields.

## 9. Hardware used for this run

*(fill in for your machine before submitting)*

- CPU:
- RAM:
- GPU / accelerator (if any):
- OS:
- Approximate embedding-model load time:
- Approximate generation-model load time:
- Approximate per-question latency:

## 10. AI assistant disclosure

This codebase was developed with substantial assistance from an AI coding
assistant (Anthropic Claude), which generated the initial project
structure, LangGraph wiring, retrieval/verification logic, tests and this
README from the supplied assignment material. All design decisions
(model choice, node boundaries, retry/verification policy) were reviewed
before submission. This disclosure follows the assignment's requirement
that AI-assistant use be stated in the README.

## 11. Known limitations / what I'd improve with more time

See `docs/design_notes.md` for the full list, including: the lexical
overlap heuristic used for the evidence-support check (an embedding-based
entailment/similarity check would be more robust than token overlap); the
rule-based triage step (a small local classifier head could replace the
keyword list); and no cross-encoder re-ranking pass before generation.
