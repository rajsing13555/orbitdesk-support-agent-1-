"""
Loads and chunks the supplied knowledge base and resolved cases.

Deterministic, model-free code: parses YAML front-matter, splits each
markdown document into section-level chunks (so retrieval can point at a
specific passage rather than a whole document), and normalises resolved
cases into the same chunk shape so both sources can be embedded and ranked
together.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    source_id: str        # KB-004 / CASE-1103
    source_type: str      # "knowledge_base" | "resolved_case"
    title: str
    heading: str           # section heading this chunk came from
    text: str
    status: str = "current"   # "current" | "resolved" | "escalated" | "superseded"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passage_id(self) -> str:
        safe_heading = re.sub(r"\s+", "_", self.heading.strip().lower())[:40]
        return f"{self.source_id}#{safe_heading}" if safe_heading else self.source_id


_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    fm_block, body = match.groups()
    meta: dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip("[]")
    return meta, body


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split a markdown body into (heading, text) chunks on '##' headings.

    The '#' title line is treated as an implicit first section named
    "Overview" so short intro paragraphs are still retrievable.
    """
    sections: list[tuple[str, str]] = []
    current_heading = "Overview"
    current_lines: list[str] = []

    for line in body.splitlines():
        h2 = re.match(r"^##\s+(.*)", line)
        h1 = re.match(r"^#\s+(.*)", line)
        if h2:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = h2.group(1).strip()
            current_lines = []
        elif h1:
            # top-level title -- keep as part of Overview, not a new section
            continue
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return [(h, t) for h, t in sections if t.strip()]


def load_knowledge_base(kb_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        source_id = meta.get("document_id", path.stem)
        title = meta.get("title", path.stem)
        status = meta.get("status", "current")
        for heading, text in _split_sections(body):
            chunks.append(
                Chunk(
                    source_id=source_id,
                    source_type="knowledge_base",
                    title=title,
                    heading=heading,
                    text=text,
                    status=status,
                    metadata={"file": path.name, "tags": meta.get("tags", "")},
                )
            )
    return chunks


def load_resolved_cases(cases_path: Path) -> list[Chunk]:
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for case in data.get("cases", []):
        text_parts = [f"Title: {case['title']}"]
        if case.get("symptoms"):
            text_parts.append("Symptoms: " + "; ".join(case["symptoms"]))
        if case.get("resolution"):
            text_parts.append("Resolution steps: " + "; ".join(case["resolution"]))
        if case.get("important_limit"):
            text_parts.append("Important limit: " + case["important_limit"])
        if case.get("superseded_reason"):
            text_parts.append("Superseded reason: " + case["superseded_reason"])
        text = "\n".join(text_parts)
        chunks.append(
            Chunk(
                source_id=case["case_id"],
                source_type="resolved_case",
                title=case["title"],
                heading="Resolved case",
                text=text,
                status=case.get("status", "resolved"),
                metadata={
                    "source_documents": case.get("source_documents", []),
                    "product_version": case.get("product_version"),
                },
            )
        )
    return chunks


def load_all_chunks(data_dir: Path) -> list[Chunk]:
    kb_chunks = load_knowledge_base(data_dir / "knowledge_base")
    case_chunks = load_resolved_cases(data_dir / "resolved_cases.json")
    return kb_chunks + case_chunks
