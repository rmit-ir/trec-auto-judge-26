"""Deterministic importance-class -> numeric weight mapping (plan Section 8). The generation
model must never choose an integer -5..5 directly -- it classifies ordinal importance, and this
module maps that class onto the ResearchRubrics scale, so weight assignment can't drift from
model to model or call to call.
"""

from __future__ import annotations

from typing import Literal

ImportanceClass = Literal[
    "CRITICAL_POSITIVE", "IMPORTANT_POSITIVE", "MINOR_POSITIVE",
    "MINOR_NEGATIVE", "DETRIMENTAL", "CRITICAL_NEGATIVE",
]

# class -> (weak, strong) weight pair. MINOR_* classes have only one value (weak == strong);
# `strong` picks between the pair for the other four classes.
_CLASS_WEIGHTS: dict[ImportanceClass, tuple[float, float]] = {
    "CRITICAL_POSITIVE": (4.0, 5.0),
    "IMPORTANT_POSITIVE": (2.0, 3.0),
    "MINOR_POSITIVE": (1.0, 1.0),
    "MINOR_NEGATIVE": (-1.0, -1.0),
    "DETRIMENTAL": (-2.0, -3.0),
    "CRITICAL_NEGATIVE": (-4.0, -5.0),
}


def importance_to_weight(importance: ImportanceClass, *, strong: bool) -> float:
    """`strong` is the deterministic secondary rule from Section 8: true when the criterion is
    required for basic correctness, false when it is merely strongly desirable/undesirable."""
    weak, strong_value = _CLASS_WEIGHTS[importance]
    return strong_value if strong else weak


def is_mandatory(weight: float) -> bool:
    """|weight| >= 4 -> mandatory/critical (Section 8 operational rule)."""
    return abs(weight) >= 4
