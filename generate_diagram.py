"""
Generates diagrams/graph_diagram.png -- a static picture of the compiled
LangGraph, for the assignment's "Graph diagram uploaded as a PNG or JPG"
submission requirement. Run once: `python diagrams/generate_diagram.py`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parent / "graph_diagram.png"

NODE_STYLE = dict(boxstyle="round,pad=0.35,rounding_size=0.08", linewidth=1.6)
COLORS = {
    "control": "#e8eefc",
    "model": "#fdece0",
    "decision": "#eafbea",
    "terminal": "#f6e8ee",
}

nodes = {
    "init":          {"pos": (0.5, 9.4),  "label": "init\n(load state,\nretry_count=0)", "kind": "control"},
    "triage":        {"pos": (0.5, 8.0),  "label": "triage\n(rules + embedding\nscope check)", "kind": "decision"},
    "retrieval":     {"pos": (-1.6, 6.4), "label": "retrieval\n(local HF embedding\nmodel, cosine top-k)", "kind": "model"},
    "generation":    {"pos": (0.5, 4.8),  "label": "generation\n(local HF LM for\nanswerable; templates\nfor other branches)", "kind": "model"},
    "verification":  {"pos": (0.5, 3.2),  "label": "verification\n(schema, sources,\nevidence overlap,\nunsupported-action guard)", "kind": "decision"},
    "revise":        {"pos": (2.9, 4.0),  "label": "revise\n(retry_count += 1,\nmax 1 retry)", "kind": "control"},
    "safe_failure":  {"pos": (2.9, 1.8),  "label": "safe_failure\n(terminal fallback)", "kind": "terminal"},
    "end_clarify":   {"pos": (3.0, 8.0),  "label": "END\n(requires_clarification /\nout_of_scope)", "kind": "terminal"},
    "end_pass":      {"pos": (-2.0, 1.8), "label": "END\n(verified answer)", "kind": "terminal"},
}

edges = [
    ("init", "triage", None),
    ("triage", "retrieval", "answerable /\nrequires_escalation"),
    ("triage", "end_clarify", "requires_clarification /\nout_of_scope\n(template, no LM call)"),
    ("retrieval", "generation", None),
    ("end_clarify", "generation", None),  # visually route templates through generation too
    ("generation", "verification", None),
    ("verification", "end_pass", "passed"),
    ("verification", "revise", "failed &\nretry_count < max_retries"),
    ("verification", "safe_failure", "failed &\nretry_count >= max_retries"),
    ("revise", "generation", "loop back\n(bounded)"),
]

fig, ax = plt.subplots(figsize=(11, 10))
ax.set_xlim(-4, 5)
ax.set_ylim(0.5, 10.2)
ax.axis("off")
ax.set_title(
    "OrbitDesk Support Agent -- LangGraph Workflow",
    fontsize=15, fontweight="bold", pad=14,
)

box_wh = {}
for name, spec in nodes.items():
    x, y = spec["pos"]
    color = COLORS[spec["kind"]]
    text = spec["label"]
    width = 0.09 * max(len(line) for line in text.split("\n")) + 0.5
    height = 0.34 * (text.count("\n") + 1) + 0.35
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        linewidth=1.6, edgecolor="#333333", facecolor=color, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=9.3, zorder=4)
    box_wh[name] = (width, height)


def edge_anchor(name, other_pos):
    x, y = nodes[name]["pos"]
    w, h = box_wh[name]
    ox, oy = other_pos
    dx, dy = ox - x, oy - y
    if abs(dx) / (w / 2 + 1e-6) > abs(dy) / (h / 2 + 1e-6):
        return (x + (w / 2 if dx > 0 else -w / 2), y)
    return (x, y + (h / 2 if dy > 0 else -h / 2))


for src, dst, label in edges:
    p0 = edge_anchor(src, nodes[dst]["pos"])
    p1 = edge_anchor(dst, nodes[src]["pos"])
    arrow = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.3, color="#555555", zorder=2,
        connectionstyle="arc3,rad=0.08",
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my, label, fontsize=7.6, ha="center", va="center",
                color="#444444", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

legend_items = [
    ("Deterministic / control-flow code", COLORS["control"]),
    ("Local Hugging Face model call", COLORS["model"]),
    ("Decision / check", COLORS["decision"]),
    ("Terminal state", COLORS["terminal"]),
]
for i, (label, color) in enumerate(legend_items):
    y = 10.05 - i * 0.32
    ax.add_patch(FancyBboxPatch((-3.95, y - 0.1), 0.22, 0.2, facecolor=color, edgecolor="#333333", linewidth=1))
    ax.text(-3.6, y, label, fontsize=8.3, va="center")

fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"Wrote {OUT}")
