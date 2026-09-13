"""Tests for the hybrid dense+sparse retriever (docs/trec-rag-2026-researchrubrics-generation-
plan.md "Retrieval architecture for rubric generation" / "Fusion"). Dummy clients throughout --
no live services, no LLM calls, deterministic RRF.
"""

import pytest

from judges.rag26_arena.hybrid_retrieval import (
    BackendOccurrence,
    HybridConfig,
    HybridResearchRetriever,
    ResearchIntent,
    build_hybrid_research_pack,
    parent_docid,
    reciprocal_rank_fusion,
)
from judges.rag26_arena.rubrics.topics import Topic
from judges.rag26_arena.search_index import DummySearchClient, SearchHit, SearchUnavailableError


def _hit(docid, rank, score, text="t"):
    return SearchHit(docid=docid, rank=rank, score=score, text=text)


# --- parent_docid ---------------------------------------------------------------------------

def test_parent_docid_strips_passage_suffix():
    assert parent_docid("shard_00924_8806_p7") == "shard_00924_8806"


def test_parent_docid_leaves_whole_doc_id_unchanged():
    assert parent_docid("shard_00459_61697") == "shard_00459_61697"


def test_parent_docid_only_strips_trailing_pN_pattern():
    assert parent_docid("shard_p_weird_p3") == "shard_p_weird"
    assert parent_docid("no_suffix_here") == "no_suffix_here"


# --- reciprocal_rank_fusion -------------------------------------------------------------------

def test_rrf_sums_across_multiple_occurrences():
    occs = {
        "d1": [BackendOccurrence(backend="dense", intent="i", query="i", rank=1, score=0.9),
               BackendOccurrence(backend="sparse", intent="i", query="i", rank=3, score=8.0)],
        "d2": [BackendOccurrence(backend="dense", intent="i", query="i", rank=2, score=0.8)],
    }
    scores = reciprocal_rank_fusion(occs, rrf_k=60)
    assert scores["d1"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["d2"] == pytest.approx(1 / 62)
    assert scores["d1"] > scores["d2"]  # two backends agreeing outranks one


# --- HybridResearchRetriever ------------------------------------------------------------------

def test_routes_semantic_to_dense_only():
    dense = DummySearchClient(hits_by_query={"q": [_hit("d1", 1, 0.9)]})
    sparse = DummySearchClient(hits_by_query={"q": [_hit("s1", 1, 5.0)]})
    retriever = HybridResearchRetriever(dense, sparse)
    hits = retriever.retrieve([ResearchIntent(text="q", kind="semantic")])
    assert [h.chunk_id for h in hits] == ["d1"]
    assert hits[0].occurrences[0].backend == "dense"


def test_routes_lexical_to_sparse_only():
    dense = DummySearchClient(hits_by_query={"q": [_hit("d1", 1, 0.9)]})
    sparse = DummySearchClient(hits_by_query={"q": [_hit("s1", 1, 5.0)]})
    retriever = HybridResearchRetriever(dense, sparse)
    hits = retriever.retrieve([ResearchIntent(text="q", kind="lexical")])
    assert [h.chunk_id for h in hits] == ["s1"]
    assert hits[0].occurrences[0].backend == "sparse"


def test_routes_hybrid_to_both_and_fuses():
    dense = DummySearchClient(hits_by_query={"q": [_hit("shared", 1, 0.9), _hit("dense_only", 2, 0.8)]})
    sparse = DummySearchClient(hits_by_query={"q": [_hit("shared", 1, 9.0), _hit("sparse_only", 2, 8.0)]})
    retriever = HybridResearchRetriever(dense, sparse)
    hits = retriever.retrieve([ResearchIntent(text="q", kind="hybrid")])
    by_id = {h.chunk_id: h for h in hits}
    assert set(by_id) == {"shared", "dense_only", "sparse_only"}
    assert len(by_id["shared"].occurrences) == 2  # both backends surfaced it
    assert by_id["shared"].fused_rank == 1  # ranked first: two agreeing occurrences


def test_never_averages_raw_scores_only_fuses_by_rank():
    # dense score 0.9 (cosine-ish) vs sparse score 9000.0 (Lucene) must not be blended directly.
    dense = DummySearchClient(hits_by_query={"q": [_hit("d1", 1, 0.9)]})
    sparse = DummySearchClient(hits_by_query={"q": [_hit("s1", 1, 9000.0)]})
    retriever = HybridResearchRetriever(dense, sparse)
    hits = retriever.retrieve([ResearchIntent(text="q", kind="hybrid")])
    # both rank 1 in their own backend -> equal RRF score, not dominated by sparse's huge raw score
    assert hits[0].fused_score == hits[1].fused_score


def test_parent_docid_derivation_on_chunked_ids():
    dense = DummySearchClient(hits_by_query={"q": [_hit("shard_00001_1_p3", 1, 0.9)]})
    sparse = DummySearchClient()
    retriever = HybridResearchRetriever(dense, sparse)
    hits = retriever.retrieve([ResearchIntent(text="q", kind="semantic")])
    assert hits[0].parent_docid == "shard_00001_1"


def test_doc_limit_truncates_deterministically():
    hits_by_query = {"q": [_hit(f"d{i}", i + 1, 1.0 - i * 0.01) for i in range(10)]}
    dense = DummySearchClient(hits_by_query=hits_by_query)
    sparse = DummySearchClient()
    retriever = HybridResearchRetriever(dense, sparse, HybridConfig(doc_limit=3))
    hits = retriever.retrieve([ResearchIntent(text="q", kind="semantic")])
    assert len(hits) == 3
    assert [h.chunk_id for h in hits] == ["d0", "d1", "d2"]


def test_provenance_records_backend_query_rank_score():
    dense = DummySearchClient(hits_by_query={"my query": [_hit("d1", 1, 0.42)]})
    sparse = DummySearchClient()
    retriever = HybridResearchRetriever(dense, sparse)
    hits = retriever.retrieve([ResearchIntent(text="my query", kind="semantic")])
    occ = hits[0].occurrences[0]
    assert occ.backend == "dense"
    assert occ.query == "my query"
    assert occ.intent == "my query"
    assert occ.rank == 1
    assert occ.score == 0.42


def test_fills_missing_text_from_dense_when_sparse_omitted_it():
    dense = DummySearchClient(docs={"s1": "resolved via dense doc-fetch"})
    sparse = DummySearchClient(hits_by_query={"q": [_hit("s1", 1, 5.0, text=None)]})
    retriever = HybridResearchRetriever(dense, sparse)
    hits = retriever.retrieve([ResearchIntent(text="q", kind="lexical")])
    assert hits[0].text == "resolved via dense doc-fetch"


def test_missing_text_fallback_failure_does_not_crash_pipeline():
    class BrokenDense(DummySearchClient):
        def get_document(self, docid):
            raise SearchUnavailableError("down")

    dense = BrokenDense()
    sparse = DummySearchClient(hits_by_query={"q": [_hit("s1", 1, 5.0, text=None)]})
    retriever = HybridResearchRetriever(dense, sparse)
    hits = retriever.retrieve([ResearchIntent(text="q", kind="lexical")])
    assert hits[0].text is None  # left unresolved, no exception propagated


def test_health_reports_both_backends_and_never_raises():
    class BrokenDense(DummySearchClient):
        def health(self):
            raise SearchUnavailableError("dense down")

    retriever = HybridResearchRetriever(BrokenDense(), DummySearchClient())
    health = retriever.health()
    assert health["dense"]["status"] == "unavailable"
    assert health["sparse"]["status"] == "ok"


def test_build_hybrid_research_pack_is_deterministic_and_planner_sees_only_topic():
    seen = []

    def planner(topic):
        seen.append(topic)
        return [ResearchIntent(text="q", kind="hybrid")]

    dense = DummySearchClient(hits_by_query={"q": [_hit("d1", 1, 0.9)]})
    sparse = DummySearchClient(hits_by_query={"q": [_hit("d1", 1, 5.0)]})
    retriever = HybridResearchRetriever(dense, sparse)
    topic = Topic(qid="t1", narrative="n")

    pack = build_hybrid_research_pack(topic, retriever, planner)
    assert seen == [topic]
    assert pack.topic_qid == "t1"
    assert pack.topic_content_hash == topic.content_hash
    assert [h.chunk_id for h in pack.hits] == ["d1"]

    pack2 = build_hybrid_research_pack(topic, retriever, planner)
    assert pack.cache_key(dense_identity="v1", sparse_identity="v1") == \
        pack2.cache_key(dense_identity="v1", sparse_identity="v1")


def test_hybrid_pack_cache_key_changes_with_backend_identity():
    dense = DummySearchClient(hits_by_query={"q": [_hit("d1", 1, 0.9)]})
    sparse = DummySearchClient()
    retriever = HybridResearchRetriever(dense, sparse)
    topic = Topic(qid="t1", narrative="n")
    pack = build_hybrid_research_pack(topic, retriever, lambda t: [ResearchIntent(text="q", kind="semantic")])
    assert pack.cache_key(dense_identity="v1", sparse_identity=None) != \
        pack.cache_key(dense_identity="v2", sparse_identity=None)


def test_retrieve_never_passes_anything_but_intent_text_to_backends():
    seen = []

    class RecordingDense(DummySearchClient):
        def search_batch(self, queries, *, k=20, with_text=True, **kw):
            seen.extend(queries)
            return super().search_batch(queries, k=k, with_text=with_text, **kw)

    retriever = HybridResearchRetriever(RecordingDense(), DummySearchClient())
    retriever.retrieve([ResearchIntent(text="only this text", kind="semantic")])
    assert seen == ["only this text"]
