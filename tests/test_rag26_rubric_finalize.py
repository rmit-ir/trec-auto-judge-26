"""Stages E/F/H/I with synthetic candidates (no network, no LLM)."""
import json

import pytest

from judges.rag26_arena.rubrics.finalize import (
    FinalizeError, apply_critic, parse_critic, parse_importance, prenormalize, select_final, to_rubric_topic,
)
from judges.rag26_arena.rubrics.generation_models import RubricCandidate, TopicCharacterization
from judges.rag26_arena.rubrics.validate import validate_topic


def cand(i, axis="Implicit Criteria", polarity="positive", text=None, conf=0.5):
    return RubricCandidate(
        candidate_id=f"q-{i:03d}", criterion=text or f"The response covers synthetic point number {i} in detail",
        axis=axis, polarity=polarity, rationale="r", narrative_spans=(), evidence_ids=(), confidence=conf,
        generator_stage="t",
    )


def test_prenormalize_dedupes_and_bounds():
    cs = [cand(1, text="The response  covers X thoroughly"), cand(2, text="the response covers x thoroughly"),
          cand(3, text="short"), cand(4, axis="Bogus")]
    assert [c.candidate_id for c in prenormalize(cs)] == ["q-001"]


def test_critic_parse_and_apply():
    cs = [cand(i) for i in range(5)]
    reply = json.dumps([
        {"id": "q-000", "verdict": "ATOMIC"},
        {"id": "q-001", "verdict": "SPLIT_REQUIRED", "split": ["The response A one two three", "The response B one two three"]},
        {"id": "q-002", "verdict": "REDUNDANT"}, {"id": "q-003", "verdict": "TOO_VAGUE"}, {"id": "q-004", "verdict": "ATOMIC"},
    ])
    out = apply_critic(cs, parse_critic(reply, cs))
    assert [c.candidate_id for c in out] == ["q-000", "q-001-s0", "q-001-s1", "q-004"]


def test_critic_never_drops_protected_candidates():
    for c in (cand(0, axis="Explicit Criteria"), cand(1, polarity="negative"), cand(2, axis="References & Citation Quality")):
        assert len(apply_critic([c], {c.candidate_id: ("REDUNDANT", ())})) == 1
    assert apply_critic([cand(3)], {"q-003": ("REDUNDANT", ())}) == []


@pytest.mark.parametrize("reply", [
    "nope", json.dumps([{"id": "zzz", "verdict": "ATOMIC"}]),
    json.dumps([{"id": "q-000", "verdict": "MAYBE"}]),
])
def test_critic_rejects_malformed(reply):
    with pytest.raises(FinalizeError):
        parse_critic(reply, [cand(0)])


def test_critic_unusable_split_falls_back_to_atomic():
    cs = [cand(0)]
    d = parse_critic(json.dumps([{"id": "q-000", "verdict": "SPLIT_REQUIRED", "split": ["only one"]}]), cs)
    assert d["q-000"][0] == "ATOMIC"


def test_importance_polarity_consistency():
    cs = [cand(0), cand(1, axis="Communication Quality", polarity="negative")]
    ok = json.dumps([{"id": "q-000", "importance_class": "CRITICAL_POSITIVE", "strong": True},
                     {"id": "q-001", "importance_class": "DETRIMENTAL", "strong": False}])
    assert parse_importance(ok, cs)["q-001"] == ("DETRIMENTAL", False)
    bad = json.dumps([{"id": "q-000", "importance_class": "DETRIMENTAL", "strong": True},
                      {"id": "q-001", "importance_class": "DETRIMENTAL", "strong": False}])
    with pytest.raises(FinalizeError):
        parse_importance(bad, cs)
    with pytest.raises(FinalizeError):  # missing id
        parse_importance(json.dumps([{"id": "q-000", "importance_class": "MINOR_POSITIVE", "strong": False}]), cs)


def test_select_final_end_to_end_valid():
    axes = ["Explicit Criteria"] * 6 + ["Implicit Criteria"] * 14 + ["Synthesis of Information"] * 4 + [
        "Communication Quality", "Instruction Following", "References & Citation Quality"]
    cs = [cand(i, axis=a, conf=(i % 10) / 10) for i, a in enumerate(axes)]
    cs.append(cand(99, axis="Implicit Criteria", polarity="negative"))
    imp = {c.candidate_id: ("IMPORTANT_POSITIVE", False) if c.polarity == "positive" else ("CRITICAL_NEGATIVE", True)
           for c in cs}
    imp["q-000"] = ("CRITICAL_POSITIVE", True)
    sel = select_final(cs, imp, target=22)
    topic = to_rubric_topic(TopicCharacterization("q", "d", "Simple", "Shallow", "Low", (), (), ""), sel)
    assert validate_topic(topic) == []
    assert 22 <= len(sel) <= 40
    assert {"Explicit Criteria", "References & Citation Quality"} <= topic.axes_present
    assert any(c.candidate_id == "q-099" for c, _ in sel)  # critical negative kept
    assert select_final(cs, imp, target=22) == sel  # deterministic


def test_arena_prompt_rubric_conditioning_and_fallback(tmp_path):
    from autojudge_base import Request
    from judges.rag26_arena.judge import RAG26ArenaJudge
    from judges.rag26_arena.prompts import arena_messages
    from judges.rag26_arena.rubrics.io import dump_jsonl
    from judges.rag26_arena.rubrics.schema import RubricCriterion, RubricTopic

    rt = RubricTopic(qid="t1", domain="d", conceptual_breadth="Simple", logical_nesting="Shallow", exploration="Low",
                     rubrics=(RubricCriterion(criterion="The response is minor", weight=1.0, axis="Communication Quality"),
                              RubricCriterion(criterion="The response hallucinates", weight=-4.0, axis="Implicit Criteria"),
                              RubricCriterion(criterion="The response answers the ask", weight=5.0, axis="Explicit Criteria")))
    path = tmp_path / "r.jsonl"
    dump_jsonl([rt], path)
    texts = RAG26ArenaJudge._load_rubric_text(True, str(path))
    assert texts["t1"].splitlines() == ["+ The response answers the ask", "- The response hallucinates", "+ The response is minor"]
    assert RAG26ArenaJudge._load_rubric_text(False, str(path)) == {}
    with pytest.raises(Exception):
        RAG26ArenaJudge._load_rubric_text(True, str(tmp_path / "missing.jsonl"))

    req = Request(request_id="t1", title="Q?")
    plain = arena_messages(req, "a", "b", max_answer_chars=100)[1]["content"]
    cond = arena_messages(req, "a", "b", max_answer_chars=100, rubric_text=texts["t1"])[1]["content"]
    assert "Criteria written" not in plain and "- The response hallucinates" in cond
    assert arena_messages(req, "a", "b", max_answer_chars=100, rubric_text=None)[1]["content"] == plain
