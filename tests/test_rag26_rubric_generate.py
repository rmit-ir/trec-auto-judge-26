"""Offline tests for judges.rag26_arena.rubrics.generate (D1-D5 axis
candidate generators). No network: judges.rag26_arena.rubrics.llm.call_bedrock
is monkeypatched."""

import json
from decimal import Decimal

import pytest

import judges.rag26_arena.rubrics.llm as llm_module
from judges.rag26_arena.rubrics.cache import RubricLlmCache
from judges.rag26_arena.rubrics.generate import (
    CandidateGenerationError,
    generate_axis_candidates,
    parse_explicit_candidates,
    parse_implicit_candidates,
    parse_low_volume_candidates,
    parse_negative_candidates,
    parse_synthesis_candidates,
)
from judges.rag26_arena.rubrics.generation_models import EvidenceNote, TopicCharacterization
from judges.rag26_arena.rubrics.llm import CostTracker, LlmResult
from judges.rag26_arena.rubrics.topics import Topic

_MODEL_ID = "qwen.qwen3-235b-a22b-2507-v1:0"
_NARRATIVE = "Why do leaves change color in fall, and will the tree be okay?"
_HAYSTACKS = [_NARRATIVE, "Why do leaves change color", "Will the tree be okay"]

_CHARACTERIZATION = TopicCharacterization(
    qid="leaf", domain="plant biology", conceptual_breadth="Moderate",
    logical_nesting="Intermediate", exploration="Medium",
    explicit_subquestions=("Why do leaves change color", "Will the tree be okay"),
    named_constraints=(), synthesis_goal="connect pigment chemistry to tree health",
)

_EVIDENCE = (
    EvidenceNote(evidence_id="ev-000", claim="Chlorophyll breaks down in autumn.",
                docids=("d1",), subtopic="mechanism", evidence_type="fact", confidence=0.9),
    EvidenceNote(evidence_id="ev-001", claim="Trees reabsorb nutrients before leaf drop.",
                docids=("d2",), subtopic="tree health", evidence_type="fact", confidence=0.8),
)


def _valid_item(**overrides):
    item = {
        "criterion": "The response explains why leaves change color.",
        "polarity": "positive",
        "rationale": "Directly asked for.",
        "narrative_spans": ["Why do leaves change color"],
        "evidence_ids": [],
        "confidence": 0.9,
    }
    item.update(overrides)
    return item


# --- D1 explicit -----------------------------------------------------------

def test_parse_explicit_accepts_valid_payload():
    payload = [_valid_item() for _ in range(6)]
    candidates = parse_explicit_candidates("leaf", json.dumps(payload), narrative_haystacks=_HAYSTACKS,
                                           min_count=5, max_count=7)
    assert len(candidates) == 6
    assert all(c.axis == "Explicit Criteria" for c in candidates)
    assert all(c.polarity == "positive" for c in candidates)
    assert candidates[0].candidate_id == "leaf-explicit-000"
    assert candidates[1].candidate_id == "leaf-explicit-001"


def test_parse_explicit_rejects_count_outside_range():
    payload = [_valid_item() for _ in range(2)]
    with pytest.raises(CandidateGenerationError):
        parse_explicit_candidates("leaf", json.dumps(payload), narrative_haystacks=_HAYSTACKS, min_count=5, max_count=7)


def test_parse_explicit_rejects_malformed_json():
    with pytest.raises(CandidateGenerationError):
        parse_explicit_candidates("leaf", "not json", narrative_haystacks=_HAYSTACKS, min_count=1, max_count=5)


def test_parse_explicit_rejects_negative_polarity():
    payload = [_valid_item(polarity="negative") for _ in range(5)]
    with pytest.raises(CandidateGenerationError, match="polarity"):
        parse_explicit_candidates("leaf", json.dumps(payload), narrative_haystacks=_HAYSTACKS, min_count=5, max_count=7)


def test_parse_explicit_rejects_invented_narrative_span():
    payload = [_valid_item(narrative_spans=["this text does not appear anywhere"]) for _ in range(5)]
    with pytest.raises(CandidateGenerationError, match="not found verbatim"):
        parse_explicit_candidates("leaf", json.dumps(payload), narrative_haystacks=_HAYSTACKS, min_count=5, max_count=7)


def test_parse_explicit_accepts_a_named_constraint_as_a_narrative_span():
    """Regression test: named_constraints are Stage A's own paraphrase of the
    narrative (not a raw substring), but they are still legitimate citable
    context -- the haystack generate_axis_candidates builds must include
    them, not just the raw narrative and explicit_subquestions."""
    haystacks = [_NARRATIVE, "Why do leaves change color", "5-year-old audience"]
    payload = [_valid_item(narrative_spans=["5-year-old audience"])] + [_valid_item() for _ in range(4)]
    candidates = parse_explicit_candidates("leaf", json.dumps(payload), narrative_haystacks=haystacks,
                                           min_count=5, max_count=7)
    assert len(candidates) == 5


def test_parse_explicit_rejects_empty_list():
    with pytest.raises(CandidateGenerationError):
        parse_explicit_candidates("leaf", "[]", narrative_haystacks=_HAYSTACKS, min_count=5, max_count=7)


@pytest.mark.parametrize("bad_field,bad_value", [
    ("criterion", ""), ("confidence", 1.5), ("confidence", "high"), ("rationale", ""),
])
def test_parse_explicit_rejects_invalid_common_fields(bad_field, bad_value):
    payload = [_valid_item(**{bad_field: bad_value})] + [_valid_item() for _ in range(4)]
    with pytest.raises(CandidateGenerationError):
        parse_explicit_candidates("leaf", json.dumps(payload), narrative_haystacks=_HAYSTACKS, min_count=5, max_count=7)


# --- D2 implicit -------------------------------------------------------------

def test_parse_implicit_accepts_valid_payload_with_evidence():
    payload = [_valid_item(evidence_ids=["ev-000"]) for _ in range(7)]
    candidates = parse_implicit_candidates("leaf", json.dumps(payload), valid_evidence_ids={"ev-000", "ev-001"},
                                           min_count=7, max_count=10)
    assert len(candidates) == 7
    assert all(c.axis == "Implicit Criteria" for c in candidates)
    assert candidates[0].evidence_ids == ("ev-000",)


def test_parse_implicit_allows_no_evidence_for_reasoning_candidates():
    payload = [_valid_item(evidence_ids=[]) for _ in range(7)]
    candidates = parse_implicit_candidates("leaf", json.dumps(payload), valid_evidence_ids={"ev-000"},
                                           min_count=7, max_count=10)
    assert len(candidates) == 7


def test_parse_implicit_rejects_unknown_evidence_id():
    payload = [_valid_item(evidence_ids=["ev-999"])] + [_valid_item() for _ in range(6)]
    with pytest.raises(CandidateGenerationError, match="unknown evidence_id"):
        parse_implicit_candidates("leaf", json.dumps(payload), valid_evidence_ids={"ev-000"}, min_count=7, max_count=10)


def test_parse_implicit_rejects_negative_polarity():
    payload = [_valid_item(polarity="negative") for _ in range(7)]
    with pytest.raises(CandidateGenerationError):
        parse_implicit_candidates("leaf", json.dumps(payload), valid_evidence_ids=set(), min_count=7, max_count=10)


# --- D3 synthesis ------------------------------------------------------------

def test_parse_synthesis_accepts_valid_payload():
    payload = [_valid_item() for _ in range(3)]
    candidates = parse_synthesis_candidates("leaf", json.dumps(payload), valid_evidence_ids=set(),
                                            min_count=3, max_count=5)
    assert all(c.axis == "Synthesis of Information" for c in candidates)


def test_parse_synthesis_rejects_count_outside_range():
    payload = [_valid_item() for _ in range(1)]
    with pytest.raises(CandidateGenerationError):
        parse_synthesis_candidates("leaf", json.dumps(payload), valid_evidence_ids=set(), min_count=3, max_count=5)


# --- D4 negative -------------------------------------------------------------

def _valid_negative_item(**overrides):
    item = {
        "criterion": "The response never presents a disputed conclusion as settled fact.",
        "axis": "Implicit Criteria", "polarity": "negative", "rationale": "Materially damaging if violated.",
        "narrative_spans": [], "evidence_ids": [], "confidence": 0.8,
    }
    item.update(overrides)
    return item


def test_parse_negative_accepts_valid_payload_with_model_supplied_axis():
    payload = [_valid_negative_item() for _ in range(3)]
    candidates = parse_negative_candidates("leaf", json.dumps(payload), valid_evidence_ids=set(),
                                           min_count=2, max_count=4)
    assert all(c.axis == "Implicit Criteria" for c in candidates)
    assert all(c.polarity == "negative" for c in candidates)


def test_parse_negative_rejects_unknown_axis():
    payload = [_valid_negative_item(axis="Not A Real Axis")] + [_valid_negative_item() for _ in range(2)]
    with pytest.raises(CandidateGenerationError, match="invalid axis"):
        parse_negative_candidates("leaf", json.dumps(payload), valid_evidence_ids=set(), min_count=2, max_count=4)


def test_parse_negative_rejects_positive_polarity():
    payload = [_valid_negative_item(polarity="positive")] + [_valid_negative_item() for _ in range(2)]
    with pytest.raises(CandidateGenerationError):
        parse_negative_candidates("leaf", json.dumps(payload), valid_evidence_ids=set(), min_count=2, max_count=4)


# --- D5 low-volume ------------------------------------------------------------

def _valid_low_volume_payload():
    return {
        "references": [_valid_item()],
        "communication": [_valid_item(), _valid_item()],
        "instruction_following": [_valid_item()],
    }


def test_parse_low_volume_accepts_valid_payload():
    candidates = parse_low_volume_candidates("leaf", json.dumps(_valid_low_volume_payload()))
    axes = {c.axis for c in candidates}
    assert axes == {"References & Citation Quality", "Communication Quality", "Instruction Following"}
    assert len(candidates) == 4


def test_parse_low_volume_rejects_missing_key():
    payload = _valid_low_volume_payload()
    del payload["communication"]
    with pytest.raises(CandidateGenerationError, match="communication"):
        parse_low_volume_candidates("leaf", json.dumps(payload))


def test_parse_low_volume_rejects_count_outside_per_axis_range():
    payload = _valid_low_volume_payload()
    payload["references"] = [_valid_item(), _valid_item(), _valid_item()]  # max is 2
    with pytest.raises(CandidateGenerationError, match="references"):
        parse_low_volume_candidates("leaf", json.dumps(payload))


def test_parse_low_volume_rejects_malformed_json():
    with pytest.raises(CandidateGenerationError):
        parse_low_volume_candidates("leaf", "not json")


# --- orchestration ------------------------------------------------------------

def test_generate_axis_candidates_runs_all_five_stages_in_order(monkeypatch, tmp_path):
    calls = []

    def fake_call_bedrock(model_id, messages, *, region, tier, fallback_tier=None, **kwargs):
        prompt = messages[-1]["content"]
        calls.append(prompt[:40])
        if "EXPLICIT" in prompt:
            payload = [_valid_item() for _ in range(6)]
        elif "IMPLICIT" in prompt:
            payload = [_valid_item(evidence_ids=["ev-000"]) for _ in range(10)]
        elif "SYNTHESIS" in prompt:
            payload = [_valid_item() for _ in range(3)]
        elif "NEGATIVE" in prompt:
            payload = [_valid_negative_item() for _ in range(3)]
        elif "three axes" in prompt:
            return LlmResult(text=json.dumps(_valid_low_volume_payload()), input_tokens=100, output_tokens=50,
                             model_id=model_id, tier=tier, region=region)
        else:
            raise AssertionError(f"unrecognized stage prompt: {prompt!r}")
        return LlmResult(text=json.dumps(payload), input_tokens=100, output_tokens=50,
                         model_id=model_id, tier=tier, region=region)

    monkeypatch.setattr(llm_module, "call_bedrock", fake_call_bedrock)
    cache = RubricLlmCache(tmp_path / "cache.db")
    tracker = CostTracker(hard_cap_usd=Decimal("25"))
    topic = Topic(qid="leaf", narrative=_NARRATIVE)

    candidates = generate_axis_candidates(
        topic, _CHARACTERIZATION, _EVIDENCE, model_id=_MODEL_ID, region="ap-southeast-2",
        tier="flex", fallback_tier="default", cache=cache, cost_tracker=tracker,
    )

    assert len(calls) == 5  # exactly one call per D1-D5 stage
    stages = [c.generator_stage for c in candidates]
    # Deterministic stage order: explicit, implicit, synthesis, negative, then low-volume axes.
    assert stages[:6] == ["explicit"] * 6
    assert stages[6:16] == ["implicit"] * 10
    assert stages[16:19] == ["synthesis"] * 3
    assert stages[19:22] == ["negative"] * 3
    assert set(stages[22:]) == {"low_volume_references", "low_volume_communication", "low_volume_instruction_following"}

    # Deterministic, code-assigned candidate IDs -- never taken from the model.
    ids = [c.candidate_id for c in candidates]
    assert ids == sorted(ids, key=ids.index)  # stable order
    assert len(ids) == len(set(ids))  # all unique
    assert tracker.total_usd() > Decimal("0")
    cache.close()


def test_generate_axis_candidates_caches_every_stage_independently(monkeypatch, tmp_path):
    call_count = {"n": 0}

    def fake_call_bedrock(model_id, messages, *, region, tier, fallback_tier=None, **kwargs):
        call_count["n"] += 1
        prompt = messages[-1]["content"]
        if "EXPLICIT" in prompt:
            payload = [_valid_item() for _ in range(6)]
        elif "IMPLICIT" in prompt:
            payload = [_valid_item() for _ in range(10)]
        elif "SYNTHESIS" in prompt:
            payload = [_valid_item() for _ in range(3)]
        elif "NEGATIVE" in prompt:
            payload = [_valid_negative_item() for _ in range(3)]
        else:
            return LlmResult(text=json.dumps(_valid_low_volume_payload()), input_tokens=1, output_tokens=1,
                             model_id=model_id, tier=tier, region=region)
        return LlmResult(text=json.dumps(payload), input_tokens=1, output_tokens=1,
                         model_id=model_id, tier=tier, region=region)

    monkeypatch.setattr(llm_module, "call_bedrock", fake_call_bedrock)
    cache = RubricLlmCache(tmp_path / "cache.db")
    topic = Topic(qid="leaf", narrative=_NARRATIVE)

    generate_axis_candidates(topic, _CHARACTERIZATION, _EVIDENCE, model_id=_MODEL_ID, region="ap-southeast-2",
                             tier="flex", fallback_tier="default", cache=cache, cost_tracker=CostTracker(Decimal("25")))
    assert call_count["n"] == 5

    call_count["n"] = 0
    tracker2 = CostTracker(hard_cap_usd=Decimal("25"))
    generate_axis_candidates(topic, _CHARACTERIZATION, _EVIDENCE, model_id=_MODEL_ID, region="ap-southeast-2",
                             tier="flex", fallback_tier="default", cache=cache, cost_tracker=tracker2)
    assert call_count["n"] == 0  # every stage replayed from cache
    assert tracker2.total_usd() == Decimal("0")
    cache.close()


def test_generate_axis_candidates_retries_a_single_stage_after_invalid_response(monkeypatch, tmp_path):
    responses = iter([
        LlmResult(text="not json", input_tokens=1, output_tokens=1, model_id=_MODEL_ID, tier="flex", region="r"),
        LlmResult(text=json.dumps([_valid_item() for _ in range(6)]), input_tokens=1, output_tokens=1,
                  model_id=_MODEL_ID, tier="flex", region="r"),
    ])

    def fake_call_bedrock(model_id, messages, *, region, tier, fallback_tier=None, **kwargs):
        prompt = messages[-1]["content"]
        if "EXPLICIT" in prompt:
            return next(responses)
        if "IMPLICIT" in prompt:
            return LlmResult(text=json.dumps([_valid_item() for _ in range(10)]), input_tokens=1, output_tokens=1,
                             model_id=model_id, tier=tier, region=region)
        if "SYNTHESIS" in prompt:
            return LlmResult(text=json.dumps([_valid_item() for _ in range(3)]), input_tokens=1, output_tokens=1,
                             model_id=model_id, tier=tier, region=region)
        if "NEGATIVE" in prompt:
            return LlmResult(text=json.dumps([_valid_negative_item() for _ in range(3)]), input_tokens=1, output_tokens=1,
                             model_id=model_id, tier=tier, region=region)
        return LlmResult(text=json.dumps(_valid_low_volume_payload()), input_tokens=1, output_tokens=1,
                         model_id=model_id, tier=tier, region=region)

    monkeypatch.setattr(llm_module, "call_bedrock", fake_call_bedrock)
    cache = RubricLlmCache(tmp_path / "cache.db")
    tracker = CostTracker(hard_cap_usd=Decimal("25"))
    topic = Topic(qid="leaf", narrative=_NARRATIVE)

    candidates = generate_axis_candidates(
        topic, _CHARACTERIZATION, _EVIDENCE, model_id=_MODEL_ID, region="ap-southeast-2",
        tier="flex", fallback_tier="default", cache=cache, cost_tracker=tracker, max_attempts=3,
    )
    assert any(c.generator_stage == "explicit" for c in candidates)
    cache.close()


def test_generate_axis_candidates_never_sees_candidate_response_data():
    """Structural guard: the orchestration signature accepts only a Topic,
    a TopicCharacterization, and EvidenceNotes -- there is no parameter
    through which a candidate RAG26 answer/citation could enter."""
    import inspect
    from judges.rag26_arena.rubrics.generate import generate_axis_candidates as fn
    params = set(inspect.signature(fn).parameters)
    assert params == {
        "topic", "characterization", "evidence_notes", "model_id", "region", "tier",
        "fallback_tier", "cache", "cost_tracker", "max_attempts",
    }
