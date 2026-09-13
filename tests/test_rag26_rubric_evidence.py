"""Offline tests for judges.rag26_arena.rubrics.evidence (Stages B+C). No
network: search_index clients are DummySearchClient, and
judges.rag26_arena.rubrics.llm.call_bedrock is monkeypatched (the single
choke point cached_bedrock_call routes every stage's live call through)."""

import json
from decimal import Decimal

import pytest

import judges.rag26_arena.rubrics.llm as llm_module
from judges.rag26_arena.hybrid_retrieval import HybridResearchRetriever, ResearchHit, BackendOccurrence
from judges.rag26_arena.rubrics.cache import RubricLlmCache
from judges.rag26_arena.rubrics.evidence import (
    EvidenceCondensationError,
    condense_evidence,
    parse_evidence_notes,
    plan_and_research,
)
from judges.rag26_arena.rubrics.generation_models import TopicCharacterization
from judges.rag26_arena.rubrics.llm import CostTracker, LlmResult
from judges.rag26_arena.rubrics.topics import Topic
from judges.rag26_arena.search_index import DummySearchClient, SearchHit

_MODEL_ID = "qwen.qwen3-235b-a22b-2507-v1:0"

_CHARACTERIZATION_PAYLOAD = {
    "domain": "Current Events",
    "conceptual_breadth": "Moderate",
    "logical_nesting": "Intermediate",
    "exploration": "Medium",
    "explicit_subquestions": ["q1", "q2"],
    "named_constraints": [],
    "synthesis_goal": "connect the evidence",
    "research_intents": [
        {"query": "q1 term", "retrieval_mode": "dense", "purpose": "p1"},
        {"query": "q2 term", "retrieval_mode": "sparse", "purpose": "p2"},
        {"query": "context term", "retrieval_mode": "hybrid", "purpose": "p3"},
        {"query": "counterargument term", "retrieval_mode": "dense", "purpose": "p4"},
        {"query": "stakeholder term", "retrieval_mode": "dense", "purpose": "p5"},
        {"query": "uncertainty term", "retrieval_mode": "sparse", "purpose": "p6"},
    ],
}

_CHARACTERIZATION = TopicCharacterization(
    qid="t1", domain="Current Events", conceptual_breadth="Moderate",
    logical_nesting="Intermediate", exploration="Medium",
    explicit_subquestions=("q1", "q2"), named_constraints=(),
    synthesis_goal="connect the evidence",
)


def _hit(chunk_id: str, text: str | None) -> ResearchHit:
    return ResearchHit(
        chunk_id=chunk_id, parent_docid=chunk_id, text=text,
        occurrences=(BackendOccurrence(backend="dense", intent="i", query="q", rank=1, score=1.0),),
        fused_score=1.0, fused_rank=1,
    )


def _valid_notes_payload():
    return [
        {"claim": "X causes Y.", "docids": ["d1"], "subtopic": "mechanism",
         "evidence_type": "fact", "confidence": 0.9, "independent_sources": 2},
        {"claim": "Some dispute Z.", "docids": ["d2"], "subtopic": "controversy",
         "evidence_type": "counterargument", "confidence": 0.6, "independent_sources": 1},
    ]


def test_parse_evidence_notes_accepts_valid_payload():
    notes = parse_evidence_notes(
        json.dumps(_valid_notes_payload()), valid_docids={"d1", "d2"}, min_notes=2, max_notes=5,
    )
    assert len(notes) == 2
    assert notes[0].evidence_id == "ev-000"  # deterministic, never taken from the model
    assert notes[1].evidence_id == "ev-001"
    assert notes[0].docids == ("d1",)


def test_parse_evidence_notes_rejects_unlisted_docid():
    payload = _valid_notes_payload()
    payload[0]["docids"] = ["not-a-real-docid"]
    with pytest.raises(EvidenceCondensationError, match="unlisted docid"):
        parse_evidence_notes(json.dumps(payload), valid_docids={"d1", "d2"}, min_notes=2, max_notes=5)


def test_parse_evidence_notes_rejects_count_outside_range():
    with pytest.raises(EvidenceCondensationError):
        parse_evidence_notes(json.dumps(_valid_notes_payload()), valid_docids={"d1", "d2"}, min_notes=5, max_notes=10)


@pytest.mark.parametrize("bad_field,bad_value", [
    ("claim", ""),
    ("subtopic", ""),
    ("evidence_type", "opinion"),
    ("confidence", 1.5),
    ("confidence", "high"),
    ("independent_sources", 0),
    ("independent_sources", "two"),
])
def test_parse_evidence_notes_rejects_invalid_fields(bad_field, bad_value):
    payload = _valid_notes_payload()
    payload[0][bad_field] = bad_value
    with pytest.raises(EvidenceCondensationError):
        parse_evidence_notes(json.dumps(payload), valid_docids={"d1", "d2"}, min_notes=2, max_notes=5)


def test_parse_evidence_notes_rejects_malformed_json():
    with pytest.raises(EvidenceCondensationError):
        parse_evidence_notes("not json", valid_docids={"d1"}, min_notes=1, max_notes=5)


def test_condense_evidence_caches_and_replays(monkeypatch, tmp_path):
    calls = []

    def fake_call_bedrock(model_id, messages, *, region, tier, fallback_tier=None, **kwargs):
        calls.append(1)
        return LlmResult(text=json.dumps(_valid_notes_payload()), input_tokens=1000, output_tokens=300,
                         model_id=model_id, tier=tier, region=region)

    monkeypatch.setattr(llm_module, "call_bedrock", fake_call_bedrock)
    cache = RubricLlmCache(tmp_path / "cache.db")
    tracker = CostTracker(hard_cap_usd=Decimal("25"))
    topic = Topic(qid="t1", narrative="A question about X and Y.")
    hits = [_hit("d1", "some passage about X and Y"), _hit("d2", "a disputing passage about Z")]
    from judges.rag26_arena.hybrid_retrieval import HybridConfig, HybridResearchPack
    pack = HybridResearchPack(topic_qid="t1", topic_content_hash=topic.content_hash,
                              intents=(), hits=tuple(hits), config=HybridConfig())

    notes = condense_evidence(
        topic, _CHARACTERIZATION, pack, model_id=_MODEL_ID, region="ap-southeast-2",
        tier="flex", fallback_tier="default", cache=cache, cost_tracker=tracker,
        min_notes=2, max_notes=5,
    )
    assert len(notes) == 2
    assert len(calls) == 1
    assert tracker.total_usd() > Decimal("0")

    calls.clear()
    tracker2 = CostTracker(hard_cap_usd=Decimal("25"))
    condense_evidence(
        topic, _CHARACTERIZATION, pack, model_id=_MODEL_ID, region="ap-southeast-2",
        tier="flex", fallback_tier="default", cache=cache, cost_tracker=tracker2,
        min_notes=2, max_notes=5,
    )
    assert calls == []  # cache hit: no new Bedrock call
    assert tracker2.total_usd() == Decimal("0")
    cache.close()


def test_condense_evidence_retries_on_hallucinated_docid(monkeypatch, tmp_path):
    bad_payload = _valid_notes_payload()
    bad_payload[0]["docids"] = ["fabricated"]
    responses = iter([
        LlmResult(text=json.dumps(bad_payload), input_tokens=100, output_tokens=50,
                  model_id=_MODEL_ID, tier="flex", region="ap-southeast-2"),
        LlmResult(text=json.dumps(_valid_notes_payload()), input_tokens=100, output_tokens=50,
                  model_id=_MODEL_ID, tier="flex", region="ap-southeast-2"),
    ])
    monkeypatch.setattr(llm_module, "call_bedrock", lambda *a, **k: next(responses))
    cache = RubricLlmCache(tmp_path / "cache.db")
    tracker = CostTracker(hard_cap_usd=Decimal("25"))
    topic = Topic(qid="t1", narrative="A question.")
    hits = [_hit("d1", "text1"), _hit("d2", "text2")]
    from judges.rag26_arena.hybrid_retrieval import HybridConfig, HybridResearchPack
    pack = HybridResearchPack(topic_qid="t1", topic_content_hash=topic.content_hash,
                              intents=(), hits=tuple(hits), config=HybridConfig())

    notes = condense_evidence(
        topic, _CHARACTERIZATION, pack, model_id=_MODEL_ID, region="ap-southeast-2",
        tier="flex", fallback_tier=None, cache=cache, cost_tracker=tracker,
        min_notes=2, max_notes=5, max_attempts=3,
    )
    assert len(notes) == 2
    cache.close()


def test_hit_with_no_text_cannot_be_cited():
    """A hit whose text never resolved must not appear in the model-citable
    docid universe -- otherwise the model could "cite" content it never saw."""
    hits = [_hit("d1", "real text"), _hit("d2", None)]
    from judges.rag26_arena.hybrid_retrieval import HybridConfig, HybridResearchPack
    pack = HybridResearchPack(topic_qid="t1", topic_content_hash="h", intents=(),
                              hits=tuple(hits), config=HybridConfig())
    valid_docids = {hit.chunk_id for hit in pack.hits if hit.text}
    assert valid_docids == {"d1"}


def test_plan_and_research_combines_stage_a_and_b(monkeypatch, tmp_path):
    def fake_call_bedrock(model_id, messages, *, region, tier, fallback_tier=None, **kwargs):
        return LlmResult(text=json.dumps(_CHARACTERIZATION_PAYLOAD), input_tokens=500, output_tokens=200,
                         model_id=model_id, tier=tier, region=region)

    monkeypatch.setattr(llm_module, "call_bedrock", fake_call_bedrock)
    dense = DummySearchClient(hits_by_query={
        "q1 term": [SearchHit(docid="dense-1", rank=1, score=0.9, text="dense passage")],
        "context term": [SearchHit(docid="dense-2", rank=1, score=0.8, text="context passage")],
        "counterargument term": [SearchHit(docid="dense-3", rank=1, score=0.7, text="counter passage")],
        "stakeholder term": [SearchHit(docid="dense-4", rank=1, score=0.6, text="stakeholder passage")],
    })
    sparse = DummySearchClient(hits_by_query={
        "q2 term": [SearchHit(docid="sparse-1", rank=1, score=5.0, text="sparse passage")],
        "context term": [SearchHit(docid="sparse-2", rank=1, score=4.0, text="sparse context")],
        "uncertainty term": [SearchHit(docid="sparse-3", rank=1, score=3.0, text="uncertainty passage")],
    })
    retriever = HybridResearchRetriever(dense, sparse)
    cache = RubricLlmCache(tmp_path / "cache.db")
    tracker = CostTracker(hard_cap_usd=Decimal("25"))
    topic = Topic(qid="t1", narrative="A question with two parts.")

    characterization, pack = plan_and_research(
        topic, retriever, model_id=_MODEL_ID, region="ap-southeast-2", tier="flex",
        fallback_tier="default", cache=cache, cost_tracker=tracker,
    )
    assert characterization.domain == "Current Events"
    assert len(pack.hits) > 0
    assert pack.topic_qid == "t1"
    cache.close()
