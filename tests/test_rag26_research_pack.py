"""Tests for Stage 1 topic loading and the Stage 3 research-pack pipeline scaffold
(docs/trec-rag-2026-researchrubrics-generation-plan.md Sections 13, 15). Dummy planner + dummy
search client throughout -- no LLM calls, response-independent by construction.
"""

import json

import pytest

from judges.rag26_arena.rubrics.research import ResearchPackConfig, build_research_pack
from judges.rag26_arena.rubrics.topics import Topic, duplicate_qids, load_topics
from judges.rag26_arena.search_index import DummySearchClient, SearchHit


# --- topics.py ----------------------------------------------------------------------------------

def test_load_topics_title_only_schema(tmp_path):
    path = tmp_path / "topics.jsonl"
    path.write_text(json.dumps({"request_id": "rag2026-0", "title": "Explain X."}) + "\n")
    topics = load_topics(path)
    assert topics == [Topic(qid="rag2026-0", narrative="Explain X.")]


def test_load_topics_joins_title_background_problem_statement(tmp_path):
    path = tmp_path / "topics.jsonl"
    row = {"request_id": "14", "title": "T", "background": "B", "problem_statement": "P"}
    path.write_text(json.dumps(row) + "\n")
    topics = load_topics(path)
    assert topics[0].narrative == "T\n\nB\n\nP"


def test_load_topics_preserves_file_order_and_exact_qid(tmp_path):
    path = tmp_path / "topics.jsonl"
    rows = [{"request_id": "b", "title": "second"}, {"request_id": "a", "title": "first"}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    topics = load_topics(path)
    assert [t.qid for t in topics] == ["b", "a"]


def test_load_topics_missing_qid_field_raises(tmp_path):
    path = tmp_path / "topics.jsonl"
    path.write_text(json.dumps({"title": "no id"}) + "\n")
    with pytest.raises(ValueError, match="missing 'request_id'"):
        load_topics(path)


def test_load_topics_empty_narrative_raises(tmp_path):
    path = tmp_path / "topics.jsonl"
    path.write_text(json.dumps({"request_id": "x", "title": ""}) + "\n")
    with pytest.raises(ValueError, match="empty narrative"):
        load_topics(path)


def test_content_hash_is_deterministic_and_qid_sensitive():
    t1 = Topic(qid="a", narrative="same text")
    t2 = Topic(qid="a", narrative="same text")
    t3 = Topic(qid="b", narrative="same text")
    assert t1.content_hash == t2.content_hash
    assert t1.content_hash != t3.content_hash


def test_duplicate_qids_detected():
    topics = [Topic(qid="a", narrative="x"), Topic(qid="a", narrative="y"), Topic(qid="b", narrative="z")]
    assert duplicate_qids(topics) == ["a"]


# --- research.py ---------------------------------------------------------------------------

def _dummy_hit(docid, text="t"):
    return SearchHit(docid=docid, rank=1, score=1.0, text=text)


def test_build_research_pack_dedupes_queries_and_docids_deterministically():
    topic = Topic(qid="q1", narrative="What is X and how does it compare to Y?")

    def planner(t):
        # deliberately includes a duplicate query, and two queries that surface the same docid
        return ["explicit asks", "explicit asks", "counterarguments"]

    client = DummySearchClient(hits_by_query={
        "explicit asks": [_dummy_hit("d1"), _dummy_hit("d2")],
        "counterarguments": [_dummy_hit("d2"), _dummy_hit("d3")],  # d2 already seen
    })

    pack = build_research_pack(topic, client, planner)
    assert pack.queries == ("explicit asks", "counterarguments")  # dedup preserves first occurrence
    assert [h.docid for h in pack.hits] == ["d1", "d2", "d3"]  # union, first-seen order
    assert pack.topic_qid == "q1"
    assert pack.topic_content_hash == topic.content_hash


def test_build_research_pack_respects_doc_limit():
    topic = Topic(qid="q1", narrative="n")
    client = DummySearchClient(hits_by_query={
        "q": [_dummy_hit(f"d{i}") for i in range(10)],
    })
    pack = build_research_pack(topic, client, lambda t: ["q"], ResearchPackConfig(doc_limit=3))
    assert len(pack.hits) == 3


def test_build_research_pack_never_calls_planner_with_anything_but_topic():
    seen_args = []

    def planner(t):
        seen_args.append(t)
        return ["q"]

    topic = Topic(qid="q1", narrative="n")
    client = DummySearchClient(hits_by_query={"q": []})
    build_research_pack(topic, client, planner)
    assert seen_args == [topic]


def test_cache_key_changes_when_queries_or_backend_identity_change():
    topic = Topic(qid="q1", narrative="n")
    client = DummySearchClient(hits_by_query={"q": [_dummy_hit("d1")], "q2": [_dummy_hit("d1")]})
    pack_a = build_research_pack(topic, client, lambda t: ["q"])
    pack_b = build_research_pack(topic, client, lambda t: ["q2"])
    assert pack_a.cache_key(backend_identity="v1") != pack_b.cache_key(backend_identity="v1")
    assert pack_a.cache_key(backend_identity="v1") != pack_a.cache_key(backend_identity="v2")
    # same inputs -> same key (deterministic, replay-safe)
    pack_a2 = build_research_pack(topic, client, lambda t: ["q"])
    assert pack_a.cache_key(backend_identity="v1") == pack_a2.cache_key(backend_identity="v1")
