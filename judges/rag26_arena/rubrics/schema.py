"""Organizer-compatible ResearchRubrics schema (docs/trec-rag-2026-researchrubrics-generation-plan.md
Section 3-4). `RubricCriterion`/`RubricTopic` are deliberately permissive at parse time -- they
must load the real organizer `research-rubrics-dev-rubrics.jsonl` as-is, including rows outside
the documented six-axis taxonomy or with unusual weights (the dev file has both; see
`local-data/researchrubrics-dev/README.md` and `scripts/analyze_researchrubrics.py`). Strict
acceptance rules for OUR generated output live in `validate.py`, not here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# The six canonical ResearchRubrics axes (plan Section 4). An axis outside this set is not a
# parse error (the organizer dev file itself contains one, "Miscellaneous") -- code that cares
# should check membership explicitly via `is_canonical_axis`.
CANONICAL_AXES: tuple[str, ...] = (
    "Explicit Criteria",
    "Implicit Criteria",
    "Synthesis of Information",
    "References & Citation Quality",
    "Communication Quality",
    "Instruction Following",
)

# Published ResearchRubrics-wide axis proportions (plan Section 5) -- soft priors for comparison,
# never fixed quotas.
PUBLISHED_AXIS_PRIORS: dict[str, float] = {
    "Implicit Criteria": 0.394,
    "Explicit Criteria": 0.277,
    "Synthesis of Information": 0.158,
    "Communication Quality": 0.078,
    "Instruction Following": 0.058,
    "References & Citation Quality": 0.035,
}

CONCEPTUAL_BREADTH_VALUES: tuple[str, ...] = ("Simple", "Moderate", "High")
LOGICAL_NESTING_VALUES: tuple[str, ...] = ("Shallow", "Intermediate", "Deep")
EXPLORATION_VALUES: tuple[str, ...] = ("Low", "Medium", "High")


def is_canonical_axis(axis: str) -> bool:
    return axis in CANONICAL_AXES


class RubricCriterion(BaseModel):
    """One rubric row, matching the organizer JSONL's `rubrics[]` entries exactly (`criterion`,
    `weight`, `axis` -- no other fields belong here; provenance/rationale live in a separate
    sidecar per the plan, not yet implemented)."""

    model_config = {"frozen": True}

    criterion: str = Field(..., min_length=1)
    weight: float
    axis: str = Field(..., min_length=1)

    @property
    def polarity(self) -> str:
        """'positive' / 'negative' / 'neutral'. A weight of exactly 0 is not expected from our
        own generator (validate.py rejects it) but does occur at least once in the organizer dev
        file, so this must not raise."""
        if self.weight > 0:
            return "positive"
        if self.weight < 0:
            return "negative"
        return "neutral"

    @property
    def mandatory(self) -> bool:
        """Operational rule (plan Section 8): |weight| >= 4 -> mandatory/critical."""
        return abs(self.weight) >= 4


class RubricTopic(BaseModel):
    """One organizer-compatible JSONL row: a topic's full rubric plus its three complexity
    labels (plan Section 3, Section 7)."""

    model_config = {"frozen": True}

    qid: str = Field(..., min_length=1)
    domain: str = Field(..., min_length=1)
    conceptual_breadth: str
    logical_nesting: str
    exploration: str
    rubrics: tuple[RubricCriterion, ...]

    @property
    def axes_present(self) -> set[str]:
        return {r.axis for r in self.rubrics}
