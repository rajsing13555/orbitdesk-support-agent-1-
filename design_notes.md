# Design Notes

## Key trade-offs

**1. The local LM is only used on the `answerable` path.**
Clarification, escalation and out-of-scope responses are deterministic
templates, not model output. These branches are the ones governed by
explicit safety/policy rules in `KB-008` and `KB-010` (what to collect,
what never to ask for, refusing out-of-scope requests without answering
from general knowledge). A 0.5B local model is far more likely to drift
from that required wording under paraphrasing than a template is, and the
assignment's own verification requirement ("avoids inventing unsupported
instructions") is much easier to guarantee for a fixed template than for
freshly generated text. The trade-off is a less "conversational" feel on
those branches, in exchange for policy-exact, auditable output.

**2. Triage is rule-based plus one embedding signal, not model-classified.**
Scope/safety classification is the highest-stakes decision in the graph.
Keeping it in deterministic code (with the retrieved-corpus similarity
score as the one model-derived input) keeps it inspectable and testable
without depending on a language model's judgment call about whether a
request is "safe." The trade-off is that the keyword list is necessarily
incomplete -- a paraphrased refund request the list doesn't recognise
would fall through to the embedding-similarity / specificity checks
instead, which are weaker signals for *safety* than for *topic*.

**3. Evidence support is checked with lexical token overlap, not a second
embedding/entailment model.**
This keeps the verification node fast, deterministic, and free of a third
model dependency. It is a blunt instrument: a well-grounded answer that
happens to paraphrase heavily could score a lower overlap than a
plagiarized-looking one. With more time this would be replaced with a
sentence-level embedding similarity between the answer and the cited
passages (reusing the same embedding backend already loaded for
retrieval, so no extra model would be needed).

**4. Chunking is section-level (markdown `##` headings), not fixed-size.**
This matches the knowledge base's own structure (each `##` section is
already a coherent troubleshooting step or rule) and produces more
citable, single-topic passages than a fixed token-window chunker would,
at the cost of some chunks being longer or shorter than ideal for the
embedding model's context.

## Reproducing the verification-failure demo case

Required test case 5 ("a case where the initial generated answer fails
verification") needs a *reliable*, repeatable way to trigger a failure --
waiting for a 0.5B local model to happen to hallucinate on a specific
question would make the demo flaky. `generation.py` therefore has a
narrow, explicitly-gated hook:

```python
force_failure = os.environ.get("ORBITDESK_FORCE_VERIFICATION_FAILURE") == "1"
if force_failure and state.get("retry_count", 0) == 0:
    ...
```

It is inert unless the environment variable is set, and even then only
fires on the *first* attempt for that single invocation -- the retry
always regenerates normally. `scripts/run_demo_cases.py` sets this
variable only around the one case designed to exercise the path, and
`tests/test_graph_routing.py` exercises the same routing more strictly by
directly injecting a verification function that fails once (and,
separately, one that always fails, to prove the retry ceiling actually
stops the loop). This keeps the demonstration deterministic without
weakening the real verification logic used on every other question.

## Known limitations

- The stub embedding/generation backends (`ORBITDESK_MODEL_BACKEND=stub`)
  exist solely so `tests/test_graph_routing.py` can run instantly and
  offline in CI; they are never used for the demo cases or a real answer.
- The lexical-overlap evidence check (see trade-off 3) is a heuristic
  proxy for "is this answer grounded," not a semantic entailment check.
- Triage's keyword list is illustrative, not exhaustive; it is layered
  with the embedding scope check specifically so a fully novel phrasing
  still degrades gracefully to `requires_clarification` or an
  out-of-scope embedding-similarity floor rather than being answered from
  the model's general knowledge.
- Retrieval is single-stage cosine similarity; there is no re-ranking
  pass. For a knowledge base this small (10 documents), a re-ranker's
  benefit is marginal, but it would matter at larger scale.
- Confidence is a simple function of top retrieval similarity, not a
  calibrated probability.

## What I would improve with more time

1. Replace the lexical-overlap evidence check with an embedding-similarity
   / lightweight entailment check between the answer and its cited
   passages.
2. Add a small local cross-encoder re-ranking pass after the initial
   cosine-similarity retrieval, particularly useful once the knowledge
   base grows beyond a handful of documents.
3. Expand the triage rule set based on a larger, labelled set of example
   questions, and consider a small fine-tuned/zero-shot classification
   head as a second opinion alongside the current rules.
4. Track resolved-case "supersedes" relationships explicitly (e.g.
   `CASE-0914` -> superseded by `KB-005`'s "Legacy Personal Tokens"
   section) so verification can cite *why* a case was excluded, not just
   that it was.
