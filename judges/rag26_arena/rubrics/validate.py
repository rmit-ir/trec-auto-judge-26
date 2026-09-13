"""Quality gates a *generated* topic rubric must pass before acceptance (plan Section 46).
Not applied to the organizer's own dev file -- that file is reference data, not something we
validate; see `scripts/analyze_researchrubrics.py` for reference-data statistics instead.

Returns a list of gate names violated (empty = passes); callers decide whether that means
"reject the whole topic" or "regenerate the failed stage" (Section 46's bounded-regeneration
policy is a generation-pipeline concern, not implemented here since generation itself is a later
milestone).
"""

from __future__ import annotations

from .schema import CANONICAL_AXES, RubricTopic

MIN_CRITERIA = 15
MAX_CRITERIA = 40


def validate_topic(topic: RubricTopic) -> list[str]:
    issues: list[str] = []

    n = len(topic.rubrics)
    if n < MIN_CRITERIA:
        issues.append(f"too_few_criteria({n}<{MIN_CRITERIA})")
    if n > MAX_CRITERIA:
        issues.append(f"too_many_criteria({n}>{MAX_CRITERIA})")

    if not any(r.polarity == "positive" and r.mandatory for r in topic.rubrics):
        issues.append("no_mandatory_positive_criterion")

    axes = topic.axes_present
    if axes and axes <= {next(iter(axes))} and n > 1:
        issues.append("single_axis_only")

    unknown_axes = sorted(axes - set(CANONICAL_AXES))
    if unknown_axes:
        issues.append(f"unknown_axis({','.join(unknown_axes)})")

    for r in topic.rubrics:
        if not (-5 <= r.weight <= 5):
            issues.append(f"weight_out_of_range({r.criterion!r}:{r.weight})")
        if r.weight == 0:
            issues.append(f"zero_weight({r.criterion!r})")
        if not r.criterion.strip():
            issues.append("empty_criterion")

    seen: set[str] = set()
    for r in topic.rubrics:
        key = r.criterion.strip().lower()
        if key in seen:
            issues.append(f"duplicate_criterion({r.criterion!r})")
        seen.add(key)

    return issues
