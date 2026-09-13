"""Internal typed generation/provenance objects (execution-plan Section 7).

None of these leak into the organizer-compatible final JSONL -- that remains
exactly `schema.RubricTopic`/`RubricCriterion` (qid, domain,
conceptual_breadth, logical_nesting, exploration, rubrics[].{criterion,
weight, axis}). These are the pipeline's internal working representation:
provenance, rationale, and confidence live here so they can be inspected in
the provenance sidecar without ever reaching the compatibility artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Optional

EvidenceType = Literal["fact", "number", "argument", "counterargument", "context"]
Polarity = Literal["positive", "negative"]
ReviewVerdict = Literal["KEEP", "REVISE", "DROP", "SPLIT", "MERGE"]

# Same six values as rubrics.weights.ImportanceClass; duplicated here as plain
# str rather than imported so this module has no dependency on weights.py's
# Literal type identity -- callers that need the strict type still import it
# from weights.py for `importance_to_weight()`.
ImportanceClass = Literal[
    "CRITICAL_POSITIVE", "IMPORTANT_POSITIVE", "MINOR_POSITIVE",
    "MINOR_NEGATIVE", "DETRIMENTAL", "CRITICAL_NEGATIVE",
]


@dataclass(frozen=True)
class TopicCharacterization:
    """Stage A output (execution-plan Section 9): task domain/complexity plus
    the explicit decomposition later generators condition on."""

    qid: str
    domain: str
    conceptual_breadth: Literal["Simple", "Moderate", "High"]
    logical_nesting: Literal["Shallow", "Intermediate", "Deep"]
    exploration: Literal["Low", "Medium", "High"]
    explicit_subquestions: tuple[str, ...]
    named_constraints: tuple[str, ...]
    synthesis_goal: str


@dataclass(frozen=True)
class EvidenceNote:
    """Stage C output (execution-plan Section 11): a compact, attributable
    claim extracted from the fused ClimbMix research pack -- never a raw
    passage. Rubric generation reads these, not the raw retrieved text."""

    evidence_id: str
    claim: str
    docids: tuple[str, ...]
    subtopic: str
    evidence_type: EvidenceType
    confidence: float
    independent_sources: int = 1


@dataclass(frozen=True)
class RubricCandidate:
    """One criterion candidate before atomicity/dedup/weighting/selection
    (execution-plan Section 7/12). `importance_class`/`strong_importance` are
    filled in by Stage H (weights.py's `importance_to_weight()` maps them to
    the final numeric weight) -- never chosen by the same call that wrote the
    criterion text."""

    candidate_id: str
    criterion: str
    axis: str
    polarity: Polarity
    rationale: str
    narrative_spans: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    confidence: float
    generator_stage: str
    importance_class: Optional[ImportanceClass] = None
    strong_importance: Optional[bool] = None
    example: Optional[str] = None


@dataclass(frozen=True)
class ReviewDecision:
    """Stage F (atomicity/dedupe critic) and Stage J (Luna final audit) both
    emit these against candidate/criterion IDs; deterministic code applies
    the decision, never the model itself."""

    candidate_id: str
    verdict: ReviewVerdict
    reason_code: str
    replacement_text: Optional[str] = None
    replacement_axis: Optional[str] = None


@dataclass(frozen=True)
class UsageRecord:
    """One priced LLM call, for the required per-stage token/cost accounting
    (execution-plan Section 4/23). Never carries prompt or response text --
    only what cost accounting and the safe progress log need."""

    stage: str
    model_id: str
    tier: str
    region: str
    input_tokens: int
    output_tokens: int
    estimated_usd: Decimal
    cache_hit: bool
    qid: Optional[str] = None
