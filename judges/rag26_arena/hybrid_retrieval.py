"""Application-layer hybrid retriever: routes research intents to dense and/or sparse backends,
normalizes both wire formats into one shape, deduplicates by chunk and parent document, and
fuses with deterministic reciprocal-rank fusion (RRF) -- per the live-endpoint retrieval
architecture the user specified on 2026-09-12 (dense :8088 / sparse :8085, see
judges/rag26_arena/README.md).

Query/intent decomposition itself (deciding *what* to search for and whether each intent is
semantic/lexical/hybrid) needs an LLM call and is NOT implemented here -- callers inject an
`IntentPlanner`, mirroring `rubrics/research.py`'s `QueryPlanner` pattern. This module is the
deterministic retrieval/fusion layer underneath that planner; it never sees candidate responses.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol

from .rubrics.topics import Topic
from .search_index import (
    SearchClient,
    SearchHit,
    SearchUnavailableError,
    dense_client_from_env,
    sparse_client_from_env,
)

BackendName = Literal["dense", "sparse"]
IntentKind = Literal["semantic", "lexical", "hybrid"]

DEFAULT_RRF_K = 60

# A chunk id looks like `shard_00924_8806_p7`; the parent document is everything before the
# final `_p<digits>` passage/chunk suffix. A whole-document id with no such suffix (e.g. an
# unchunked corpus's `shard_00459_61697`) is already its own parent.
_CHUNK_SUFFIX_RE = re.compile(r"^(?P<parent>.+)_p\d+$")


def parent_docid(chunk_id: str) -> str:
    match = _CHUNK_SUFFIX_RE.match(chunk_id)
    return match.group("parent") if match else chunk_id


@dataclass(frozen=True)
class ResearchIntent:
    """One planned search intent for a topic (Section 'Per-topic retrieval pipeline' steps 1-3).
    `kind` decides routing: semantic -> dense only, lexical -> sparse only, hybrid -> both."""

    text: str
    kind: IntentKind = "hybrid"


class IntentPlanner(Protocol):
    def __call__(self, topic: Topic) -> list[ResearchIntent]: ...


@dataclass(frozen=True)
class BackendOccurrence:
    """Provenance for one (backend, query) hit on a chunk -- everything needed to explain a
    fused result later without re-running retrieval."""

    backend: BackendName
    intent: str
    query: str
    rank: int
    score: float


@dataclass(frozen=True)
class ResearchHit:
    chunk_id: str
    parent_docid: str
    text: str | None
    occurrences: tuple[BackendOccurrence, ...]
    fused_score: float
    fused_rank: int


def reciprocal_rank_fusion(occurrences_by_chunk: dict[str, list[BackendOccurrence]], *,
                            rrf_k: int = DEFAULT_RRF_K) -> dict[str, float]:
    """Standard RRF: score(d) = sum over every (backend, query) occurrence of 1/(rrf_k + rank).
    Deterministic given deterministic occurrences; never averages raw dense/BM25 scores, which
    are not on comparable scales (plan requirement)."""
    return {
        chunk_id: sum(1.0 / (rrf_k + occ.rank) for occ in occs)
        for chunk_id, occs in occurrences_by_chunk.items()
    }


@dataclass(frozen=True)
class HybridConfig:
    k_per_query: int = 20
    doc_limit: int = 40  # plan recommends retaining ~30-50
    rrf_k: int = DEFAULT_RRF_K
    with_text: bool = True
    # If a chunk's text wasn't returned by whichever backend first surfaced it (e.g. a
    # sparse-only occurrence retrieved with with_text=False), fetch it from the dense backend's
    # /doc/{docid} as a fallback -- sparse has no per-document-fetch endpoint at all.
    fill_missing_text_from_dense: bool = True


class HybridResearchRetriever:
    """Wraps a dense and a sparse `SearchClient`. Never issues a query derived from a candidate
    answer/citation -- `retrieve()` only ever sees planner-produced `ResearchIntent`s."""

    def __init__(self, dense: SearchClient, sparse: SearchClient, config: HybridConfig = HybridConfig()):
        self.dense = dense
        self.sparse = sparse
        self.config = config

    def retrieve(self, intents: list[ResearchIntent]) -> list[ResearchHit]:
        dense_intents = [i for i in intents if i.kind in ("semantic", "hybrid")]
        sparse_intents = [i for i in intents if i.kind in ("lexical", "hybrid")]

        # One batch call per backend regardless of how many intents route to it (the dense
        # service caps in-flight requests per worker at 1 -- batching, not concurrent fan-out).
        dense_results = (
            self.dense.search_batch([i.text for i in dense_intents], k=self.config.k_per_query,
                                     with_text=self.config.with_text)
            if dense_intents else []
        )
        sparse_results = (
            self.sparse.search_batch([i.text for i in sparse_intents], k=self.config.k_per_query,
                                      with_text=self.config.with_text)
            if sparse_intents else []
        )

        occurrences_by_chunk: dict[str, list[BackendOccurrence]] = {}
        text_by_chunk: dict[str, str | None] = {}

        def absorb(backend: BackendName, backend_intents: list[ResearchIntent],
                   results: list[list[SearchHit]]) -> None:
            for intent, hits in zip(backend_intents, results):
                for hit in hits:
                    occurrences_by_chunk.setdefault(hit.docid, []).append(BackendOccurrence(
                        backend=backend, intent=intent.text, query=intent.text,
                        rank=hit.rank, score=hit.score,
                    ))
                    if hit.text is not None and text_by_chunk.get(hit.docid) is None:
                        text_by_chunk[hit.docid] = hit.text

        absorb("dense", dense_intents, dense_results)
        absorb("sparse", sparse_intents, sparse_results)

        fused_scores = reciprocal_rank_fusion(occurrences_by_chunk, rrf_k=self.config.rrf_k)
        # Deterministic tie-break: chunk_id ascending when fused scores are equal.
        ordered = sorted(occurrences_by_chunk, key=lambda c: (-fused_scores[c], c))
        retained = ordered[: self.config.doc_limit]

        if self.config.fill_missing_text_from_dense:
            for chunk_id in retained:
                if text_by_chunk.get(chunk_id) is None:
                    try:
                        text_by_chunk[chunk_id] = self.dense.get_document(chunk_id)
                    except SearchUnavailableError:
                        pass  # leave as None; an outage here shouldn't fail the whole pack

        return [
            ResearchHit(
                chunk_id=chunk_id, parent_docid=parent_docid(chunk_id),
                text=text_by_chunk.get(chunk_id),
                occurrences=tuple(occurrences_by_chunk[chunk_id]),
                fused_score=fused_scores[chunk_id], fused_rank=rank,
            )
            for rank, chunk_id in enumerate(retained, start=1)
        ]

    def health(self) -> dict:
        return {"dense": self._safe_health(self.dense), "sparse": self._safe_health(self.sparse)}

    @staticmethod
    def _safe_health(client: SearchClient) -> dict:
        try:
            return client.health()
        except SearchUnavailableError as exc:
            return {"status": "unavailable", "detail": str(exc)}


HYBRID_PACK_BUILDER_VERSION = "v1"


@dataclass(frozen=True)
class HybridResearchPack:
    """The Stage-3 output: a topic's fused, deduplicated, provenance-carrying research pack
    (docs/trec-rag-2026-researchrubrics-generation-plan.md Section 15). Every hit's
    `occurrences` already IS the provenance record (backend, query/intent, rank, raw score);
    nothing here needs a separate log to be reproducible."""

    topic_qid: str
    topic_content_hash: str
    intents: tuple[ResearchIntent, ...]
    hits: tuple[ResearchHit, ...]
    config: HybridConfig

    def cache_key(self, *, dense_identity: str | None, sparse_identity: str | None) -> str:
        """Retrieval-cache key (plan 'Provenance and caching'): topic hash + exact intents +
        both backend identities/versions (from /health, when exposed) + config + builder
        version. Changing rubric-generation prompts must not invalidate this; changing retrieval
        settings or backend identity must."""
        payload = {
            "builder_version": HYBRID_PACK_BUILDER_VERSION,
            "topic_content_hash": self.topic_content_hash,
            "intents": [{"text": i.text, "kind": i.kind} for i in self.intents],
            "config": {
                "k_per_query": self.config.k_per_query,
                "doc_limit": self.config.doc_limit,
                "rrf_k": self.config.rrf_k,
                "with_text": self.config.with_text,
            },
            "dense_identity": dense_identity,
            "sparse_identity": sparse_identity,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def build_hybrid_research_pack(topic: Topic, retriever: HybridResearchRetriever,
                                planner: IntentPlanner,
                                config: HybridConfig | None = None) -> HybridResearchPack:
    """Deterministic given a deterministic `planner` and backends: same topic + same planner +
    same config + same backend state -> same pack. `planner` only ever sees `topic` -- never a
    candidate response, citation, or retrieval trace (plan 'Forbidden' list)."""
    if config is not None and config != retriever.config:
        retriever = HybridResearchRetriever(retriever.dense, retriever.sparse, config)
    intents = planner(topic)
    hits = retriever.retrieve(intents)
    return HybridResearchPack(
        topic_qid=topic.qid,
        topic_content_hash=topic.content_hash,
        intents=tuple(intents),
        hits=tuple(hits),
        config=retriever.config,
    )


def hybrid_retriever_from_env(env: Mapping[str, str] | None = None,
                               config: HybridConfig = HybridConfig()) -> HybridResearchRetriever:
    """Reads RAG26_DENSE_SEARCH_URL and RAG26_SPARSE_SEARCH_URL (never hardcoded); raises
    SearchNotConfiguredError naming whichever is missing."""
    return HybridResearchRetriever(dense_client_from_env(env), sparse_client_from_env(env), config)
