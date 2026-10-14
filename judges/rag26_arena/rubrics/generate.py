"""D1-D5: axis-specific rubric candidate generation (execution-plan Section
12 / docs/rag26-rubric-next-stage-handoff.md). Response-independent: inputs
are only the request narrative, its Stage A characterization, and Stage C
evidence notes -- never a candidate answer.

Deliberately NOT done here (next milestone): atomicity/dedupe critic,
examples, weight assignment, final selection, Luna audit.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

from .cache import RubricLlmCache, cache_key
from .generation_models import EvidenceNote, RubricCandidate, TopicCharacterization
from .llm import CostTracker, cached_bedrock_call
from .prompts import (
    EXPLICIT_PROMPT_VERSION,
    IMPLICIT_PROMPT_VERSION,
    LOW_VOLUME_PROMPT_VERSION,
    NEGATIVE_PROMPT_VERSION,
    SYNTHESIS_PROMPT_VERSION,
    explicit_candidates_messages,
    implicit_candidates_messages,
    low_volume_candidates_messages,
    negative_candidates_messages,
    synthesis_candidates_messages,
)
from .schema import CANONICAL_AXES
from .topics import Topic

EXPLICIT_AXIS = "Explicit Criteria"
IMPLICIT_AXIS = "Implicit Criteria"
SYNTHESIS_AXIS = "Synthesis of Information"

_LOW_VOLUME_AXIS_BY_KEY = {
    "references": "References & Citation Quality",
    "communication": "Communication Quality",
    "instruction_following": "Instruction Following",
}
_LOW_VOLUME_RANGE_BY_KEY = {
    "references": (1, 2),
    "communication": (1, 3),
    "instruction_following": (1, 2),
}

# Pre-normalization candidate-count targets by conceptual_breadth (handoff
# tables for D1-D4). Broader than the eventual 24-28 final target -- the
# next milestone (atomicity/dedupe/selection) removes redundancy.
_COUNT_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "explicit": {"Simple": (5, 7), "Moderate": (6, 9), "High": (8, 11)},
    "implicit": {"Simple": (7, 10), "Moderate": (10, 14), "High": (12, 17)},
    "synthesis": {"Simple": (2, 3), "Moderate": (3, 5), "High": (4, 7)},
    "negative": {"Simple": (2, 4), "Moderate": (3, 6), "High": (4, 7)},
}


def _count_range(stage: str, breadth: str) -> tuple[int, int]:
    return _COUNT_RANGES[stage][breadth]


class CandidateGenerationError(ValueError):
    """The model's structured output failed validation. Treated exactly like
    a call failure (bounded retry), never a partial accept."""


def _extract_json_array(text: str) -> Optional[list]:
    decoder = json.JSONDecoder()
    index = text.find("[")
    while index >= 0:
        try:
            value, _end = decoder.raw_decode(text, index)
            if isinstance(value, list):
                return value
        except json.JSONDecodeError:
            pass
        index = text.find("[", index + 1)
    return None


def _extract_json_object(text: str) -> Optional[dict]:
    decoder = json.JSONDecoder()
    index = text.find("{")
    while index >= 0:
        try:
            value, _end = decoder.raw_decode(text, index)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        index = text.find("{", index + 1)
    return None


def _validate_candidate_item(
    item: object,
    *,
    valid_evidence_ids: set[str],
    forced_polarity: Optional[str] = None,
    allowed_axes: Optional[Sequence[str]] = None,
    narrative_haystacks: Optional[Sequence[str]] = None,
) -> dict:
    """Common field validation shared by every D1-D5 item shape. Raises
    CandidateGenerationError on any structural problem. Axis is validated
    here only when `allowed_axes` is given (D4's model-supplied axis, or a
    caller passing a single-axis set for symmetry) -- D1/D2/D3/D5 assign
    axis deterministically in code and never ask the model for it, so an
    "unknown axis" failure mode structurally cannot occur there."""
    if not isinstance(item, dict):
        raise CandidateGenerationError(f"candidate is not an object: {item!r}")

    criterion = item.get("criterion")
    polarity = item.get("polarity")
    rationale = item.get("rationale")
    narrative_spans = item.get("narrative_spans", [])
    evidence_ids = item.get("evidence_ids", [])
    confidence = item.get("confidence")

    if not isinstance(criterion, str) or not criterion.strip():
        raise CandidateGenerationError("empty/missing criterion")
    if polarity not in ("positive", "negative"):
        raise CandidateGenerationError(f"invalid polarity: {polarity!r}")
    if forced_polarity is not None and polarity != forced_polarity:
        raise CandidateGenerationError(f"expected polarity {forced_polarity!r}, got {polarity!r}")
    if not isinstance(rationale, str) or not rationale.strip():
        raise CandidateGenerationError("empty/missing rationale")
    if not isinstance(narrative_spans, list) or not all(isinstance(s, str) for s in narrative_spans):
        raise CandidateGenerationError("narrative_spans must be a list of strings")
    if not isinstance(evidence_ids, list) or not all(isinstance(s, str) for s in evidence_ids):
        raise CandidateGenerationError("evidence_ids must be a list of strings")
    unknown_evidence = [e for e in evidence_ids if e not in valid_evidence_ids]
    if unknown_evidence:
        raise CandidateGenerationError(f"cites unknown evidence_id(s): {unknown_evidence}")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        raise CandidateGenerationError("confidence must be a number in [0, 1]")
    if narrative_haystacks is not None:
        # narrative_spans are optional provenance metadata, not rubric
        # semantics. Strong models occasionally paraphrase a requested span
        # despite being told to copy it verbatim; keep only spans that are
        # actually grounded rather than rejecting an otherwise valid
        # criterion and retrying the identical prompt.
        narrative_spans = [
            s for s in narrative_spans
            if any(s in haystack for haystack in narrative_haystacks)
        ]

    axis: Optional[str] = None
    if allowed_axes is not None:
        axis = item.get("axis")
        if axis not in allowed_axes:
            raise CandidateGenerationError(f"invalid axis: {axis!r}")

    result = {
        "criterion": criterion.strip(), "polarity": polarity, "rationale": rationale.strip(),
        "narrative_spans": tuple(s.strip() for s in narrative_spans),
        "evidence_ids": tuple(evidence_ids), "confidence": float(confidence),
    }
    if axis is not None:
        result["axis"] = axis
    return result


def _build_candidates(items: Sequence[dict], *, axis: Optional[str], generator_stage: str, qid: str) -> list[RubricCandidate]:
    return [
        RubricCandidate(
            candidate_id=f"{qid}-{generator_stage}-{index:03d}",
            criterion=item["criterion"], axis=item.get("axis", axis), polarity=item["polarity"],
            rationale=item["rationale"], narrative_spans=item["narrative_spans"],
            evidence_ids=item["evidence_ids"], confidence=item["confidence"],
            generator_stage=generator_stage,
        )
        for index, item in enumerate(items)
    ]


def parse_explicit_candidates(
    qid: str, text: str, *, narrative_haystacks: Sequence[str], min_count: int, max_count: int,
) -> list[RubricCandidate]:
    parsed = _extract_json_array(text)
    if parsed is None:
        raise CandidateGenerationError("no JSON array found in response")
    if not (min_count <= len(parsed) <= max_count):
        raise CandidateGenerationError(f"{len(parsed)} explicit candidates outside [{min_count}, {max_count}]")
    items = [
        _validate_candidate_item(item, valid_evidence_ids=set(), forced_polarity="positive",
                                 narrative_haystacks=narrative_haystacks)
        for item in parsed
    ]
    return _build_candidates(items, axis=EXPLICIT_AXIS, generator_stage="explicit", qid=qid)


def parse_implicit_candidates(
    qid: str, text: str, *, valid_evidence_ids: set[str], min_count: int, max_count: int,
) -> list[RubricCandidate]:
    parsed = _extract_json_array(text)
    if parsed is None:
        raise CandidateGenerationError("no JSON array found in response")
    if not (min_count <= len(parsed) <= max_count):
        raise CandidateGenerationError(f"{len(parsed)} implicit candidates outside [{min_count}, {max_count}]")
    items = [
        _validate_candidate_item(item, valid_evidence_ids=valid_evidence_ids, forced_polarity="positive")
        for item in parsed
    ]
    return _build_candidates(items, axis=IMPLICIT_AXIS, generator_stage="implicit", qid=qid)


def parse_synthesis_candidates(
    qid: str, text: str, *, valid_evidence_ids: set[str], min_count: int, max_count: int,
) -> list[RubricCandidate]:
    parsed = _extract_json_array(text)
    if parsed is None:
        raise CandidateGenerationError("no JSON array found in response")
    if not (min_count <= len(parsed) <= max_count):
        raise CandidateGenerationError(f"{len(parsed)} synthesis candidates outside [{min_count}, {max_count}]")
    items = [
        _validate_candidate_item(item, valid_evidence_ids=valid_evidence_ids, forced_polarity="positive")
        for item in parsed
    ]
    return _build_candidates(items, axis=SYNTHESIS_AXIS, generator_stage="synthesis", qid=qid)


def parse_negative_candidates(
    qid: str, text: str, *, valid_evidence_ids: set[str], min_count: int, max_count: int,
) -> list[RubricCandidate]:
    """Negative criteria can belong to any of the six canonical axes, so axis
    is model-supplied here and validated -- the one D-stage where "unknown
    axis" is a real failure mode."""
    parsed = _extract_json_array(text)
    if parsed is None:
        raise CandidateGenerationError("no JSON array found in response")
    if not (min_count <= len(parsed) <= max_count):
        raise CandidateGenerationError(f"{len(parsed)} negative candidates outside [{min_count}, {max_count}]")
    items = [
        _validate_candidate_item(item, valid_evidence_ids=valid_evidence_ids, forced_polarity="negative",
                                 allowed_axes=CANONICAL_AXES)
        for item in parsed
    ]
    return _build_candidates(items, axis=None, generator_stage="negative", qid=qid)


def parse_low_volume_candidates(qid: str, text: str) -> list[RubricCandidate]:
    """No evidence vocabulary is ever shown to this stage's prompt (References/
    Communication/Instruction Following criteria are meta-requirements, not
    factual claims) -- so evidence_ids must always be empty here, never a
    model-invented id validated against an unseen set."""
    parsed = _extract_json_object(text)
    if parsed is None:
        raise CandidateGenerationError("no JSON object found in response")

    candidates: list[RubricCandidate] = []
    for key, axis in _LOW_VOLUME_AXIS_BY_KEY.items():
        items_raw = parsed.get(key)
        if not isinstance(items_raw, list):
            raise CandidateGenerationError(f"missing/invalid {key!r} list")
        min_count, max_count = _LOW_VOLUME_RANGE_BY_KEY[key]
        if not (min_count <= len(items_raw) <= max_count):
            raise CandidateGenerationError(f"{key}: {len(items_raw)} candidates outside [{min_count}, {max_count}]")
        items = [_validate_candidate_item(item, valid_evidence_ids=set()) for item in items_raw]
        candidates.extend(_build_candidates(items, axis=axis, generator_stage=f"low_volume_{key}", qid=qid))
    return candidates


def generate_axis_candidates(
    topic: Topic,
    characterization: TopicCharacterization,
    evidence_notes: Sequence[EvidenceNote],
    *,
    model_id: str,
    region: str,
    tier: str,
    fallback_tier: Optional[str],
    cache: RubricLlmCache,
    cost_tracker: CostTracker,
    max_attempts: int = 3,
) -> list[RubricCandidate]:
    """Runs D1-D5 in deterministic order, feeding each stage's output forward
    as read-only context for the next (never mutated afterward). Does not
    deduplicate, weight, or select -- that is the next milestone."""
    valid_evidence_ids = {note.evidence_id for note in evidence_notes}
    # Stage A's explicit_subquestions/named_constraints are shown to D1 as
    # legitimate citable context ("the supplied structured characterization"),
    # even though they're Stage A's own paraphrase rather than a raw
    # substring of the original narrative -- so both belong in the haystack.
    narrative_haystacks = [
        topic.narrative, *characterization.explicit_subquestions, *characterization.named_constraints,
    ]
    breadth = characterization.conceptual_breadth

    def call(stage: str, prompt_version: str, messages: list[dict[str, str]], parse) -> list[RubricCandidate]:
        key = cache_key(stage=stage, prompt_version=prompt_version, model_id=model_id,
                        tier=tier, region=region, messages=messages)
        return cached_bedrock_call(
            key, messages, model_id=model_id, region=region, tier=tier, fallback_tier=fallback_tier,
            cache=cache, cost_tracker=cost_tracker, stage=stage, qid=topic.qid, parse=parse,
            error_type=CandidateGenerationError, max_attempts=max_attempts,
        )

    min_c, max_c = _count_range("explicit", breadth)
    explicit = call(
        "explicit", EXPLICIT_PROMPT_VERSION,
        explicit_candidates_messages(topic.narrative, characterization, min_count=min_c, max_count=max_c),
        # Prompt keeps the breadth target; the parse floor is lower because a
        # topic with few literal asks legitimately yields fewer explicit
        # criteria, and retrying the identical prompt cannot change that.
        lambda text: parse_explicit_candidates(topic.qid, text, narrative_haystacks=narrative_haystacks,
                                               min_count=min(min_c, 3), max_count=max_c),
    )

    min_c, max_c = _count_range("implicit", breadth)
    implicit = call(
        "implicit", IMPLICIT_PROMPT_VERSION,
        implicit_candidates_messages(topic.narrative, characterization, evidence_notes, explicit,
                                     min_count=min_c, max_count=max_c),
        lambda text: parse_implicit_candidates(topic.qid, text, valid_evidence_ids=valid_evidence_ids,
                                               min_count=min_c, max_count=max_c),
    )

    min_c, max_c = _count_range("synthesis", breadth)
    synthesis = call(
        "synthesis", SYNTHESIS_PROMPT_VERSION,
        synthesis_candidates_messages(topic.narrative, characterization, evidence_notes, explicit, implicit,
                                      min_count=min_c, max_count=max_c),
        lambda text: parse_synthesis_candidates(topic.qid, text, valid_evidence_ids=valid_evidence_ids,
                                                min_count=min_c, max_count=max_c),
    )

    min_c, max_c = _count_range("negative", breadth)
    positives = explicit + implicit + synthesis
    negative = call(
        "negative", NEGATIVE_PROMPT_VERSION,
        negative_candidates_messages(topic.narrative, characterization, evidence_notes, positives,
                                     min_count=min_c, max_count=max_c, allowed_axes=CANONICAL_AXES),
        lambda text: parse_negative_candidates(topic.qid, text, valid_evidence_ids=valid_evidence_ids,
                                               min_count=min_c, max_count=max_c),
    )

    low_volume = call(
        "low_volume", LOW_VOLUME_PROMPT_VERSION,
        low_volume_candidates_messages(topic.narrative, characterization),
        lambda text: parse_low_volume_candidates(topic.qid, text),
    )

    return explicit + implicit + synthesis + negative + low_volume


def print_candidate_diagnostics(qid: str, candidates: Sequence[RubricCandidate], cost_tracker: CostTracker) -> None:
    """Safe summary-only diagnostics (execution-plan requirement): counts and
    aggregates, never full criterion/rationale/evidence bodies in bulk."""
    by_axis: dict[str, int] = {}
    positives = negatives = grounded = 0
    for candidate in candidates:
        by_axis[candidate.axis] = by_axis.get(candidate.axis, 0) + 1
        if candidate.polarity == "positive":
            positives += 1
        else:
            negatives += 1
        if candidate.evidence_ids:
            grounded += 1

    print(f"[rag26-rubrics] qid={qid}: {len(candidates)} candidates ({positives} positive, {negatives} negative, {grounded} evidence-grounded)")
    for axis, count in sorted(by_axis.items()):
        print(f"  {axis}: {count}")
    cost_tracker.print_summary()
