"""Stage 3: evidence-grounded topic research pack (docs/trec-rag-2026-researchrubrics-generation-
plan.md Section 15). Response-independent by construction: the only inputs are a `Topic` (from
`topics.py`) and a search client (`search_index.SearchClient`) -- never candidate answers,
citations, or retrieval traces.

Query decomposition itself (Section 15.1: "ask a planner model for ~5-10 search intents") needs
an LLM call and is deliberately NOT implemented here -- this module takes an already-produced
query list via an injected `planner` callable, so it can be exercised end-to-end with a
deterministic dummy planner in tests today, and wired to a real LLM-backed planner later without
changing this pipeline. That is the "research-pack pipeline scaffold with dummy/contract tests"
requested before any expensive LLM sweep.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Protocol

from .. import search_index
from .topics import Topic

RESEARCH_PACK_BUILDER_VERSION = "v1"


class QueryPlanner(Protocol):
    def __call__(self, topic: Topic) -> list[str]: ...


@dataclass(frozen=True)
class ResearchPackConfig:
    """Backend defaults per plan Section 15.2 -- tuning knobs for the *retrieval*, not rubric
    hyperparameters. Keep `k_per_query`/`doc_limit` here, not scattered as magic numbers."""

    k_per_query: int = 20
    doc_limit: int = 40  # plan recommends retaining 30-50
    with_text: bool = True


@dataclass(frozen=True)
class ResearchPack:
    topic_qid: str
    topic_content_hash: str
    queries: tuple[str, ...]
    hits: tuple[search_index.SearchHit, ...]
    config: ResearchPackConfig

    def cache_key(self, *, backend_identity: str | None) -> str:
        """Section 33: topic hash + exact planned query list + backend identity/version (when
        exposed by /health) + k + backend parameters + builder version."""
        payload = {
            "builder_version": RESEARCH_PACK_BUILDER_VERSION,
            "topic_content_hash": self.topic_content_hash,
            "queries": list(self.queries),
            "k_per_query": self.config.k_per_query,
            "doc_limit": self.config.doc_limit,
            "with_text": self.config.with_text,
            "backend_identity": backend_identity,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def build_research_pack(topic: Topic, search_client: search_index.SearchClient,
                         planner: QueryPlanner,
                         config: ResearchPackConfig = ResearchPackConfig()) -> ResearchPack:
    """Deterministic given a deterministic `planner` and search client: same topic + same planner
    + same config + same backend state -> same pack. Never issues search queries derived from
    candidate answers/citations (plan Section 15.2) -- `planner` only ever sees `topic`."""
    queries = _dedupe_preserve_order(planner(topic))

    results = search_client.search_batch(queries, k=config.k_per_query, with_text=config.with_text)

    by_docid: dict[str, search_index.SearchHit] = {}
    for hits in results:
        for hit in hits:
            if hit.docid not in by_docid:  # first query to surface a docid wins (deterministic
                by_docid[hit.docid] = hit  # given a deterministic query order)

    retained = list(by_docid.values())[: config.doc_limit]
    return ResearchPack(
        topic_qid=topic.qid,
        topic_content_hash=topic.content_hash,
        queries=tuple(queries),
        hits=tuple(retained),
        config=config,
    )
