"""Structural statistics over a set of rubric topics (plan Section 12, Stage 0). Used both by
`scripts/analyze_researchrubrics.py` against the organizer dev file and, later, to compare a
generator's own output against that reference -- so this must not assume organizer-clean input
(unknown axis strings, a zero weight): it counts and reports such rows rather than rejecting them.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .schema import CANONICAL_AXES, PUBLISHED_AXIS_PRIORS, RubricTopic

_EXAMPLE_MARKER = re.compile(r"\(e\.g\.,")


@dataclass
class RubricStats:
    n_topics: int
    n_criteria_total: int
    criteria_per_topic: dict[str, float]  # mean/median/min/max
    axis_counts: dict[str, int]
    axis_proportions: dict[str, float]
    axis_prior_deviation: dict[str, float]  # observed - published, canonical axes only
    axes_per_topic_mean: float
    non_canonical_axes: dict[str, int]
    positive_negative_counts: dict[str, int]
    weight_histogram: dict[str, int]  # str(weight) -> count, for stable JSON key order
    zero_weight_count: int
    mandatory_optional_counts: dict[str, int]
    mandatory_rate_by_axis: dict[str, float]
    negative_rate_by_complexity: dict[str, float]  # keyed by "conceptual_breadth:value" etc.
    criterion_length_chars: dict[str, float]  # mean/median/min/max
    example_use_rate: float
    domain_counts: dict[str, int]
    axis_by_domain: dict[str, dict[str, int]]
    complexity_counts: dict[str, dict[str, int]]  # field -> value -> count
    exact_duplicate_criterion_rate: float
    common_openings: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_topics": self.n_topics,
            "n_criteria_total": self.n_criteria_total,
            "criteria_per_topic": self.criteria_per_topic,
            "axis_counts": self.axis_counts,
            "axis_proportions": self.axis_proportions,
            "axis_prior_deviation": self.axis_prior_deviation,
            "axes_per_topic_mean": self.axes_per_topic_mean,
            "non_canonical_axes": self.non_canonical_axes,
            "positive_negative_counts": self.positive_negative_counts,
            "weight_histogram": self.weight_histogram,
            "zero_weight_count": self.zero_weight_count,
            "mandatory_optional_counts": self.mandatory_optional_counts,
            "mandatory_rate_by_axis": self.mandatory_rate_by_axis,
            "negative_rate_by_complexity": self.negative_rate_by_complexity,
            "criterion_length_chars": self.criterion_length_chars,
            "example_use_rate": self.example_use_rate,
            "domain_counts": self.domain_counts,
            "axis_by_domain": self.axis_by_domain,
            "complexity_counts": self.complexity_counts,
            "exact_duplicate_criterion_rate": self.exact_duplicate_criterion_rate,
            "common_openings": self.common_openings,
        }


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _opening(criterion: str) -> str:
    words = criterion.strip().split()
    return " ".join(words[:2]) if words else ""


def compute_stats(topics: list[RubricTopic]) -> RubricStats:
    all_criteria = [(t, r) for t in topics for r in t.rubrics]
    n_criteria_total = len(all_criteria)

    per_topic_counts = [len(t.rubrics) for t in topics]
    axis_counts: Counter[str] = Counter(r.axis for _, r in all_criteria)
    axes_per_topic = [len(t.axes_present) for t in topics]

    pos_neg: Counter[str] = Counter(r.polarity for _, r in all_criteria)
    weight_hist: Counter[str] = Counter(str(r.weight) for _, r in all_criteria)
    zero_weight_count = sum(1 for _, r in all_criteria if r.weight == 0)

    mand_opt: Counter[str] = Counter("mandatory" if r.mandatory else "optional" for _, r in all_criteria)

    mandatory_by_axis: dict[str, float] = {}
    for axis in axis_counts:
        rows = [r for _, r in all_criteria if r.axis == axis]
        mandatory_by_axis[axis] = sum(r.mandatory for r in rows) / len(rows) if rows else 0.0

    negative_rate_by_complexity: dict[str, float] = {}
    for field_name in ("conceptual_breadth", "logical_nesting", "exploration", "domain"):
        by_value: dict[str, list[RubricTopic]] = {}
        for t in topics:
            by_value.setdefault(getattr(t, field_name), []).append(t)
        for value, ts in by_value.items():
            rows = [r for t in ts for r in t.rubrics]
            rate = sum(1 for r in rows if r.polarity == "negative") / len(rows) if rows else 0.0
            negative_rate_by_complexity[f"{field_name}:{value}"] = rate

    lengths = [float(len(r.criterion)) for _, r in all_criteria]
    example_rate = (sum(1 for _, r in all_criteria if _EXAMPLE_MARKER.search(r.criterion))
                     / n_criteria_total if n_criteria_total else 0.0)

    domain_counts: Counter[str] = Counter(t.domain for t in topics)
    axis_by_domain: dict[str, dict[str, int]] = {}
    for t in topics:
        bucket = axis_by_domain.setdefault(t.domain, {})
        for r in t.rubrics:
            bucket[r.axis] = bucket.get(r.axis, 0) + 1

    complexity_counts: dict[str, dict[str, int]] = {}
    for field_name in ("conceptual_breadth", "logical_nesting", "exploration"):
        complexity_counts[field_name] = dict(Counter(getattr(t, field_name) for t in topics))

    dup_count = 0
    for t in topics:
        seen: set[str] = set()
        for r in t.rubrics:
            key = r.criterion.strip().lower()
            if key in seen:
                dup_count += 1
            seen.add(key)
    dup_rate = dup_count / n_criteria_total if n_criteria_total else 0.0

    openings: Counter[str] = Counter(_opening(r.criterion) for _, r in all_criteria)

    axis_proportions = {a: c / n_criteria_total for a, c in axis_counts.items()} if n_criteria_total else {}
    axis_prior_deviation = {
        a: axis_proportions.get(a, 0.0) - prior for a, prior in PUBLISHED_AXIS_PRIORS.items()
    }
    non_canonical = {a: c for a, c in axis_counts.items() if a not in CANONICAL_AXES}

    return RubricStats(
        n_topics=len(topics),
        n_criteria_total=n_criteria_total,
        criteria_per_topic=_summary([float(c) for c in per_topic_counts]),
        axis_counts=dict(axis_counts),
        axis_proportions=axis_proportions,
        axis_prior_deviation=axis_prior_deviation,
        axes_per_topic_mean=statistics.fmean(axes_per_topic) if axes_per_topic else 0.0,
        non_canonical_axes=non_canonical,
        positive_negative_counts=dict(pos_neg),
        weight_histogram=dict(sorted(weight_hist.items(), key=lambda kv: float(kv[0]))),
        zero_weight_count=zero_weight_count,
        mandatory_optional_counts=dict(mand_opt),
        mandatory_rate_by_axis=mandatory_by_axis,
        negative_rate_by_complexity=negative_rate_by_complexity,
        criterion_length_chars=_summary(lengths),
        example_use_rate=example_rate,
        domain_counts=dict(domain_counts),
        axis_by_domain=axis_by_domain,
        complexity_counts=complexity_counts,
        exact_duplicate_criterion_rate=dup_rate,
        common_openings=dict(openings.most_common(10)),
    )
