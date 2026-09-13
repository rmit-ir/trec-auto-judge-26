"""Tests for the ResearchRubrics-compatible schema, IO, weight mapping, and validation gates
(docs/trec-rag-2026-researchrubrics-generation-plan.md Section 42). Synthetic topics only, plus
one check against the real organizer dev file when it's present locally (not committed; see
local-data/researchrubrics-dev/README.md) -- skipped cleanly otherwise.
"""

from pathlib import Path

import pytest

from judges.rag26_arena.rubrics import io as rubric_io
from judges.rag26_arena.rubrics.schema import CANONICAL_AXES, RubricCriterion, RubricTopic
from judges.rag26_arena.rubrics.stats import compute_stats
from judges.rag26_arena.rubrics.validate import validate_topic
from judges.rag26_arena.rubrics.weights import importance_to_weight, is_mandatory

REPO = Path(__file__).parent.parent
DEV_FILE = REPO / "local-data" / "researchrubrics-dev" / "research-rubrics-dev-rubrics.jsonl"


def _criterion(text="The response does X.", weight=4.0, axis="Explicit Criteria"):
    return RubricCriterion(criterion=text, weight=weight, axis=axis)


def _topic(qid="q1", rubrics=None, domain="STEM", cb="Moderate", ln="Intermediate", ex="Medium"):
    return RubricTopic(
        qid=qid, domain=domain, conceptual_breadth=cb, logical_nesting=ln, exploration=ex,
        rubrics=rubrics if rubrics is not None else [_criterion()],
    )


# --- schema -----------------------------------------------------------------------------------

def test_polarity_and_mandatory():
    assert _criterion(weight=5.0).polarity == "positive"
    assert _criterion(weight=-2.0).polarity == "negative"
    assert _criterion(weight=0.0).polarity == "neutral"
    assert _criterion(weight=4.0).mandatory is True
    assert _criterion(weight=3.0).mandatory is False
    assert _criterion(weight=-4.0).mandatory is True


def test_criterion_requires_non_empty_text():
    with pytest.raises(Exception):
        RubricCriterion(criterion="", weight=1.0, axis="Explicit Criteria")


def test_topic_accepts_non_canonical_axis_without_raising():
    # The organizer's own dev file has a "Miscellaneous" axis row; schema must not reject it.
    topic = _topic(rubrics=[_criterion(axis="Miscellaneous")])
    assert topic.axes_present == {"Miscellaneous"}
    assert "Miscellaneous" not in CANONICAL_AXES


# --- weights ------------------------------------------------------------------------------------

def test_importance_to_weight_mapping():
    assert importance_to_weight("CRITICAL_POSITIVE", strong=True) == 5.0
    assert importance_to_weight("CRITICAL_POSITIVE", strong=False) == 4.0
    assert importance_to_weight("MINOR_POSITIVE", strong=True) == 1.0
    assert importance_to_weight("MINOR_NEGATIVE", strong=False) == -1.0
    assert importance_to_weight("DETRIMENTAL", strong=True) == -3.0
    assert importance_to_weight("CRITICAL_NEGATIVE", strong=False) == -4.0


def test_is_mandatory_threshold():
    assert is_mandatory(4.0) and is_mandatory(-4.0) and is_mandatory(5.0)
    assert not is_mandatory(3.0) and not is_mandatory(-3.9)


# --- validate -----------------------------------------------------------------------------------

def test_validate_topic_passes_on_healthy_topic():
    rubrics = [_criterion(f"The response does {i}.", weight=w, axis=a) for i, (w, a) in enumerate([
        (5.0, "Explicit Criteria"), (4.0, "Explicit Criteria"), (3.0, "Explicit Criteria"),
        (2.0, "Implicit Criteria"), (2.0, "Implicit Criteria"), (1.0, "Implicit Criteria"),
        (3.0, "Synthesis of Information"), (2.0, "Synthesis of Information"),
        (1.0, "Communication Quality"), (1.0, "Instruction Following"),
        (-1.0, "Implicit Criteria"), (-2.0, "Explicit Criteria"), (-3.0, "Synthesis of Information"),
        (1.0, "References & Citation Quality"), (2.0, "Explicit Criteria"),
    ])]
    assert validate_topic(_topic(rubrics=rubrics)) == []


def test_validate_topic_flags_too_few_criteria():
    assert "too_few_criteria(1<15)" in validate_topic(_topic())


def test_validate_topic_flags_zero_weight():
    rubrics = [_criterion(f"c{i}", weight=1.0) for i in range(14)] + [_criterion("bad", weight=0.0)]
    issues = validate_topic(_topic(rubrics=rubrics))
    assert any("zero_weight" in i for i in issues)


def test_validate_topic_flags_unknown_axis():
    rubrics = [_criterion(f"c{i}", weight=1.0) for i in range(14)] + [_criterion("bad", axis="Vibes")]
    issues = validate_topic(_topic(rubrics=rubrics))
    assert any("unknown_axis" in i and "Vibes" in i for i in issues)


def test_validate_topic_flags_no_mandatory_positive():
    rubrics = [_criterion(f"c{i}", weight=2.0) for i in range(15)]
    assert "no_mandatory_positive_criterion" in validate_topic(_topic(rubrics=rubrics))


def test_validate_topic_flags_duplicate_criterion():
    rubrics = [_criterion(f"c{i}", weight=1.0) for i in range(13)]
    rubrics.append(_criterion("The response does X.", weight=4.0))  # dup of _criterion() default
    rubrics.append(_criterion("The response does X.", weight=1.0))
    issues = validate_topic(_topic(rubrics=rubrics))
    assert any("duplicate_criterion" in i for i in issues)


# --- io -----------------------------------------------------------------------------------------

def test_dump_jsonl_is_deterministic_and_sorted_by_qid(tmp_path):
    topics = [_topic(qid="b"), _topic(qid="a")]
    out = tmp_path / "out.jsonl"
    rubric_io.dump_jsonl(topics, out)
    lines1 = out.read_text().splitlines()
    rubric_io.dump_jsonl(topics, out)
    lines2 = out.read_text().splitlines()
    assert lines1 == lines2
    assert lines1[0].startswith('{"qid":"a"') or '"qid":"a"' in lines1[0]
    qids_in_order = [line.split('"qid":"')[1].split('"')[0] for line in lines1]
    assert qids_in_order == ["a", "b"]


def test_load_jsonl_round_trips(tmp_path):
    topics = [_topic(qid="x", rubrics=[_criterion("The response covers X.", weight=3.0)])]
    out = tmp_path / "out.jsonl"
    rubric_io.dump_jsonl(topics, out)
    loaded = rubric_io.load_jsonl(out)
    assert len(loaded) == 1
    assert loaded[0].qid == "x"
    assert loaded[0].rubrics[0].weight == 3.0


def test_load_jsonl_rejects_malformed_row(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"qid": "x", "domain": "STEM"}\n')  # missing required fields
    with pytest.raises(ValueError, match="invalid rubric row"):
        rubric_io.load_jsonl(bad)


def test_check_unique_qids_detects_duplicates():
    topics = [_topic(qid="a"), _topic(qid="a"), _topic(qid="b")]
    assert rubric_io.check_unique_qids(topics) == ["a"]


# --- stats ----------------------------------------------------------------------------------

def test_compute_stats_on_synthetic_topics():
    topics = [_topic(qid="a"), _topic(qid="b", rubrics=[_criterion(axis="Implicit Criteria")])]
    s = compute_stats(topics).to_dict()
    assert s["n_topics"] == 2
    assert s["n_criteria_total"] == 2
    assert s["axis_counts"]["Explicit Criteria"] == 1
    assert s["axis_counts"]["Implicit Criteria"] == 1


@pytest.mark.skipif(not DEV_FILE.is_file(),
                     reason="organizer ResearchRubrics dev file not present locally "
                            "(see local-data/researchrubrics-dev/README.md)")
def test_organizer_dev_file_loads_and_matches_known_shape():
    topics = rubric_io.load_jsonl(DEV_FILE)
    assert len(topics) == 30
    assert rubric_io.check_unique_qids(topics) == []
    s = compute_stats(topics).to_dict()
    # Measured 2026-09-12 on this dev subset; document the reason before changing this test if
    # the shipped dev file is ever updated.
    assert 20 <= s["criteria_per_topic"]["min"] <= 25
    assert s["zero_weight_count"] >= 1  # organizer anomaly noted in dev-rubric-stats.md
    assert "Miscellaneous" in s["non_canonical_axes"]
