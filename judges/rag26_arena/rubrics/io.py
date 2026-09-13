"""Load/write organizer-compatible rubric JSONL (plan Section 28: one deterministic ordering
policy, kept stable, because it improves prompt-cache reuse downstream)."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import RubricTopic


def load_jsonl(path: Path) -> list[RubricTopic]:
    """Raise on a structurally invalid row (missing field, wrong type) -- this is meant for our
    own generated output and well-formed organizer releases. For the raw dev file's own quirks
    (unknown axis strings, a zero weight), `RubricCriterion`/`RubricTopic` parse them as-is; use
    `scripts/analyze_researchrubrics.py` (which reads the raw JSON directly) for pure statistics
    where a preventable parse error would be unwelcome."""
    topics: list[RubricTopic] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                topics.append(RubricTopic.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{path}:{lineno}: invalid rubric row: {exc}") from exc
    return topics


def dump_jsonl(topics: list[RubricTopic], path: Path) -> None:
    """Deterministic ordering: topics by `qid`, criteria in each topic kept in their given order
    (selection-stage ordering is meaningful; do not re-sort criteria here)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(topics, key=lambda t: t.qid)
    with open(path, "w", encoding="utf-8") as f:
        for topic in ordered:
            f.write(topic.model_dump_json() + "\n")


def check_unique_qids(topics: list[RubricTopic]) -> list[str]:
    """Return duplicate qids, if any (plan Section 42 unit test: 'duplicate qids')."""
    seen: set[str] = set()
    dupes: list[str] = []
    for t in topics:
        if t.qid in seen:
            dupes.append(t.qid)
        seen.add(t.qid)
    return dupes
