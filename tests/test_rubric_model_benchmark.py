from decimal import Decimal
from pathlib import Path

import pytest

from scripts import benchmark_rubric_models as b


def _criterion(text: str) -> dict[str, str]:
    return {
        "criterion": text,
        "axis": "Explicit Criteria",
        "polarity": "positive",
    }


def test_load_topics_keeps_first_headerless_row(tmp_path: Path) -> None:
    path = tmp_path / "topics.tsv"
    path.write_text("q1\tFirst prompt\nq2\tSecond prompt\n", encoding="utf-8")
    assert b.load_topics(path) == {"q1": "First prompt", "q2": "Second prompt"}


def test_blind_mapping_is_deterministic_permutation() -> None:
    keys = [x.key for x in b.CANDIDATES]
    first = b.blind_mapping("qid-1", keys)
    second = b.blind_mapping("qid-1", keys)
    assert first == second
    assert set(first) == {"A", "B", "C", "D"}
    assert set(first.values()) == set(keys)


def test_blind_judge_prompt_contains_no_model_identity() -> None:
    candidates = {
        spec.key: b.CandidateRubric(
            [_criterion(f"The response covers requirement {i} for {spec.key}.")]
        )
        for i, spec in enumerate(b.CANDIDATES)
    }
    # Replace the visible criterion text so the fake fixture itself does not
    # leak model keys. Real parsed candidate schema has no model metadata.
    for i, rubric in enumerate(candidates.values()):
        rubric.rubrics[0]["criterion"] = f"The response covers requirement {i}."
    mapping = b.blind_mapping("qid-2", list(candidates))
    reference = {
        "rubrics": [
            {
                "criterion": "The response covers the core requirement.",
                "axis": "Explicit Criteria",
                "weight": 5,
            }
        ]
    }
    prompt = b.build_blind_judge_user(
        "Do the task.", reference, candidates, mapping
    )
    for spec in b.CANDIDATES:
        assert spec.key not in prompt
        assert spec.display_name not in prompt
        assert spec.model_id not in prompt


def test_validate_candidate_rejects_invalid_axis() -> None:
    items = [_criterion(f"The response covers requirement {i}.") for i in range(12)]
    items[3]["axis"] = "Made Up Axis"
    with pytest.raises(ValueError, match="invalid axis"):
        b.validate_candidate({"rubrics": items})


def test_budget_blocks_call_that_can_cross_cap() -> None:
    budget = b.Budget(Decimal("0.000001"))
    with pytest.raises(b.BudgetExceeded):
        budget.reserve_check(
            b.JUDGE,
            estimated_input_tokens=1000,
            max_output_tokens=1000,
        )
