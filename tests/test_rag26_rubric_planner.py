"""Offline tests for judges.rag26_arena.rubrics.planner (Stage A). No
network: judges.rag26_arena.rubrics.planner.call_bedrock is monkeypatched."""

import json
from decimal import Decimal

import pytest

import judges.rag26_arena.rubrics.planner as planner_module
from judges.rag26_arena.rubrics.cache import RubricLlmCache
from judges.rag26_arena.rubrics.llm import BedrockUnavailableError, CostTracker, LlmResult
from judges.rag26_arena.rubrics.planner import CharacterizationError, characterize_topic, parse_characterization
from judges.rag26_arena.rubrics.topics import Topic

_VALID_PAYLOAD = {
    "domain": "Current Events",
    "conceptual_breadth": "Moderate",
    "logical_nesting": "Intermediate",
    "exploration": "Medium",
    "explicit_subquestions": ["why do leaves change color", "will the tree be okay"],
    "named_constraints": ["explain for a 5-year-old"],
    "synthesis_goal": "connect pigment chemistry to what a child observes outdoors",
    "research_intents": [
        {"query": "chlorophyll breakdown autumn", "retrieval_mode": "dense", "purpose": "core mechanism"},
        {"query": "anthocyanin leaf pigment", "retrieval_mode": "dense", "purpose": "red/purple color"},
        {"query": "deciduous vs evergreen leaf retention", "retrieval_mode": "hybrid", "purpose": "why some stay green"},
        {"query": "carotenoid pigment leaf", "retrieval_mode": "dense", "purpose": "yellow/orange color"},
        {"query": "abscission layer formation", "retrieval_mode": "sparse", "purpose": "leaf drop mechanism"},
        {"query": "temperature photoperiod leaf senescence", "retrieval_mode": "hybrid", "purpose": "seasonal trigger"},
    ],
}


def test_parse_characterization_accepts_valid_payload():
    characterization, intents = parse_characterization("leaf", json.dumps(_VALID_PAYLOAD))
    assert characterization.qid == "leaf"
    assert characterization.conceptual_breadth == "Moderate"
    assert len(intents) == 6
    assert intents[0].kind == "semantic"  # dense -> semantic
    assert any(i.kind == "lexical" for i in intents)  # sparse -> lexical
    assert any(i.kind == "hybrid" for i in intents)


@pytest.mark.parametrize("bad_field,bad_value", [
    ("conceptual_breadth", "Extreme"),
    ("logical_nesting", "Bottomless"),
    ("exploration", "Infinite"),
    ("domain", ""),
    ("synthesis_goal", ""),
    ("explicit_subquestions", "not a list"),
    ("named_constraints", [1, 2]),
])
def test_parse_characterization_rejects_invalid_fields(bad_field, bad_value):
    payload = {**_VALID_PAYLOAD, bad_field: bad_value}
    with pytest.raises(CharacterizationError):
        parse_characterization("t1", json.dumps(payload))


def test_parse_characterization_rejects_malformed_json():
    with pytest.raises(CharacterizationError):
        parse_characterization("t1", "not json at all")


def test_parse_characterization_rejects_invalid_retrieval_mode():
    payload = {**_VALID_PAYLOAD, "research_intents": [
        {"query": "x", "retrieval_mode": "boolean", "purpose": "y"},
    ] * 6}
    with pytest.raises(CharacterizationError):
        parse_characterization("t1", json.dumps(payload))


def test_parse_characterization_rejects_empty_query():
    payload = {**_VALID_PAYLOAD, "research_intents": [
        {"query": "  ", "retrieval_mode": "dense", "purpose": "y"},
    ] * 6}
    with pytest.raises(CharacterizationError):
        parse_characterization("t1", json.dumps(payload))


def test_parse_characterization_rejects_too_few_intents_for_subquestions():
    payload = {
        **_VALID_PAYLOAD,
        "explicit_subquestions": ["q1", "q2", "q3", "q4", "q5", "q6", "q7"],
        "research_intents": _VALID_PAYLOAD["research_intents"][:6],
    }
    with pytest.raises(CharacterizationError):
        parse_characterization("t1", json.dumps(payload))


def test_characterize_topic_calls_bedrock_on_cache_miss_and_caches_result(monkeypatch, tmp_path):
    calls = []

    def fake_call_bedrock(model_id, messages, *, region, tier, fallback_tier=None, **kwargs):
        calls.append((model_id, tier))
        return LlmResult(text=json.dumps(_VALID_PAYLOAD), input_tokens=500, output_tokens=200,
                         model_id=model_id, tier=tier, region=region)

    monkeypatch.setattr(planner_module, "call_bedrock", fake_call_bedrock)
    cache = RubricLlmCache(tmp_path / "cache.db")
    tracker = CostTracker(hard_cap_usd=Decimal("25"))
    topic = Topic(qid="leaf", narrative="Why do leaves change color in fall?")

    characterization, intents = characterize_topic(
        topic, model_id="qwen.qwen3-235b-a22b-2507-v1:0", region="ap-southeast-2",
        tier="flex", fallback_tier="default", cache=cache, cost_tracker=tracker,
    )
    assert characterization.qid == "leaf"
    assert len(intents) == 6
    assert len(calls) == 1
    assert tracker.total_usd() > Decimal("0")
    assert tracker.records[0].cache_hit is False

    # Second call for the same topic must hit the cache: no new Bedrock call, zero added cost.
    calls.clear()
    tracker2 = CostTracker(hard_cap_usd=Decimal("25"))
    characterize_topic(
        topic, model_id="qwen.qwen3-235b-a22b-2507-v1:0", region="ap-southeast-2",
        tier="flex", fallback_tier="default", cache=cache, cost_tracker=tracker2,
    )
    assert calls == []
    assert tracker2.records[0].cache_hit is True
    assert tracker2.total_usd() == Decimal("0")
    cache.close()


def test_characterize_topic_retries_on_malformed_output_then_succeeds(monkeypatch, tmp_path):
    model_id = "qwen.qwen3-235b-a22b-2507-v1:0"
    responses = iter([
        LlmResult(text="not json", input_tokens=100, output_tokens=10,
                  model_id=model_id, tier="flex", region="ap-southeast-2"),
        LlmResult(text=json.dumps(_VALID_PAYLOAD), input_tokens=500, output_tokens=200,
                  model_id=model_id, tier="flex", region="ap-southeast-2"),
    ])
    monkeypatch.setattr(planner_module, "call_bedrock", lambda *a, **k: next(responses))
    cache = RubricLlmCache(tmp_path / "cache.db")
    tracker = CostTracker(hard_cap_usd=Decimal("25"))
    topic = Topic(qid="t1", narrative="A question.")

    characterization, intents = characterize_topic(
        topic, model_id=model_id, region="ap-southeast-2", tier="flex", fallback_tier=None,
        cache=cache, cost_tracker=tracker, max_attempts=3,
    )
    assert characterization.qid == "t1"
    cache.close()


def test_characterize_topic_exhausted_retries_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(planner_module, "call_bedrock", lambda *a, **k: LlmResult(
        text="not json", input_tokens=10, output_tokens=5, model_id="m", tier="flex", region="r",
    ))
    cache = RubricLlmCache(tmp_path / "cache.db")
    tracker = CostTracker(hard_cap_usd=Decimal("25"))
    topic = Topic(qid="t1", narrative="A question.")
    with pytest.raises(CharacterizationError):
        characterize_topic(
            topic, model_id="m", region="r", tier="flex", fallback_tier=None,
            cache=cache, cost_tracker=tracker, max_attempts=2,
        )
    cache.close()


def test_characterize_topic_propagates_bedrock_unavailable_after_retries(monkeypatch, tmp_path):
    def always_unavailable(*a, **k):
        raise BedrockUnavailableError("down")

    monkeypatch.setattr(planner_module, "call_bedrock", always_unavailable)
    cache = RubricLlmCache(tmp_path / "cache.db")
    tracker = CostTracker(hard_cap_usd=Decimal("25"))
    topic = Topic(qid="t1", narrative="A question.")
    with pytest.raises(CharacterizationError):
        characterize_topic(
            topic, model_id="m", region="r", tier="flex", fallback_tier=None,
            cache=cache, cost_tracker=tracker, max_attempts=2,
        )
    cache.close()
