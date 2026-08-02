"""
Minimal interactive CLI for the OrbitDesk Support Agent.

Usage:
    python scripts/run_cli.py
    python scripts/run_cli.py --question "Can a Viewer create API credentials?"

The graph and local models are loaded once at startup; each question you
type is then run through the full triage -> retrieval -> generation ->
verification graph, and both a readable answer and the structured JSON
response are printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph import build_graph  # noqa: E402

DATA_DIR = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="OrbitDesk Support Agent CLI")
    parser.add_argument("--question", "-q", type=str, default=None, help="Ask a single question and exit.")
    args = parser.parse_args()

    print("Loading local models and building the graph...")
    compiled_graph, index, gen_backend = build_graph(DATA_DIR)
    print(f"  Embedding model : {index.backend.model_name} (revision {index.backend.revision})")
    print(f"  Generation model: {gen_backend.model_name} (revision {gen_backend.revision})")
    print("Ready. Type a question, or 'quit' to exit.\n")

    def ask(question: str) -> None:
        result = compiled_graph.invoke(
            {"question_id": "cli", "question": question},
            config={"recursion_limit": 25},
        )
        print(f"\nNode trace: {' -> '.join(result.get('node_trace', []))}")
        print("\nAnswer:\n" + result["final_response"]["answer"])
        print("\nStructured response:")
        print(json.dumps(result["final_response"], indent=2))
        print()

    if args.question:
        ask(args.question)
        return

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break
        ask(question)


if __name__ == "__main__":
    main()
