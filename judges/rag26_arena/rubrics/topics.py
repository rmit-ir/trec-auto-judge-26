"""Stage 1: load and normalize TREC RAG topics for rubric generation (docs/trec-rag-2026-
researchrubrics-generation-plan.md Section 13). No candidate response data enters here -- only
the public topics file.

Named `Topic` (not `RubricTopic`) to avoid colliding with `schema.RubricTopic`, which is the
*output* of generation (a topic plus its rubric criteria); this module handles the *input*.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Topic:
    qid: str
    narrative: str

    @property
    def content_hash(self) -> str:
        """Deterministic hash for provenance/cache keys (Section 13: 'deterministic content
        hash for provenance')."""
        return hashlib.sha256(f"{self.qid}\x00{self.narrative}".encode()).hexdigest()


# TREC RAG topic files vary in which narrative fields they carry: 2025-era files split
# title/background/problem_statement; the 2026 file (local-data/rag26/topics/
# trec_rag_2026_queries.jsonl) carries only `title`. Join whichever of these are present, in
# this fixed order, so the narrative is deterministic regardless of which fields a release ships.
_NARRATIVE_FIELDS = ("title", "background", "problem_statement")


def _narrative_from_row(row: dict) -> str:
    parts = [str(row[f]).strip() for f in _NARRATIVE_FIELDS if row.get(f)]
    return "\n\n".join(parts)


def load_topics(path: Path, *, qid_field: str = "request_id") -> list[Topic]:
    """Preserve file order and the organizer's exact qid (plan: 'preserve the organizer topic ID
    exactly'). Raises on a missing qid or an empty narrative rather than silently skipping."""
    topics: list[Topic] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if qid_field not in row:
                raise ValueError(f"{path}:{lineno}: missing '{qid_field}'")
            narrative = _narrative_from_row(row)
            if not narrative:
                raise ValueError(f"{path}:{lineno}: empty narrative for qid={row[qid_field]!r}")
            topics.append(Topic(qid=str(row[qid_field]), narrative=narrative))
    return topics


def duplicate_qids(topics: list[Topic]) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []
    for t in topics:
        if t.qid in seen:
            dupes.append(t.qid)
        seen.add(t.qid)
    return dupes
