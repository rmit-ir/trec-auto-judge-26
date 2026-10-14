"""Stages E, F, H, I: deterministic pre-normalization, atomicity/dedupe critic,
independent importance classification, deterministic final selection
(execution-plan Sections 13, 14, 16, 17). Response-independent: candidates
only, never a candidate answer.

V1 simplifications versus the full plan (see docs/rag26-tira-submission-plan.md
time budget): no targeted-examples pass (Stage G), no independent Luna audit
(Stage J), one critic pass with no repair loop.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Callable, Optional, Sequence

from .cache import RubricLlmCache, cache_key
from .generation_models import RubricCandidate, TopicCharacterization
from .llm import CostTracker, cached_bedrock_call
from .schema import CANONICAL_AXES, RubricCriterion, RubricTopic
from .weights import ImportanceClass, importance_to_weight

CRITIC_PROMPT_VERSION = "v4"
IMPORTANCE_PROMPT_VERSION = "v3"

MIN_CRITERION_CHARS = 15
MAX_CRITERION_CHARS = 400
FINAL_COUNT_BY_BREADTH = {"Simple": 22, "Moderate": 26, "High": 31}
HARD_MAX = 40

_DROPPABLE_AXES = ("Implicit Criteria", "Synthesis of Information")
CRITIC_VERDICTS = ("ATOMIC", "SPLIT_REQUIRED", "REDUNDANT", "TOO_VAGUE", "UNVERIFIABLE", "OUT_OF_SCOPE")
_POSITIVE_CLASSES = ("CRITICAL_POSITIVE", "IMPORTANT_POSITIVE", "MINOR_POSITIVE")
_NEGATIVE_CLASSES = ("MINOR_NEGATIVE", "DETRIMENTAL", "CRITICAL_NEGATIVE")


class FinalizeError(ValueError):
    """Structured critic/classifier output failed validation (treated as a
    call failure -> bounded retry)."""


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


# ---- Stage E ---------------------------------------------------------------

def prenormalize(candidates: Sequence[RubricCandidate]) -> list[RubricCandidate]:
    """Whitespace-normalize, drop empty/out-of-bounds/unknown-axis, exact
    casefold dedupe (first wins), stable sort by axis then id."""
    seen: set[str] = set()
    kept: list[RubricCandidate] = []
    axis_order = {axis: i for i, axis in enumerate(CANONICAL_AXES)}
    for cand in sorted(candidates, key=lambda c: (axis_order.get(c.axis, 99), c.candidate_id)):
        text = re.sub(r"\s+", " ", cand.criterion).strip()
        if cand.axis not in axis_order or not (MIN_CRITERION_CHARS <= len(text) <= MAX_CRITERION_CHARS):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        kept.append(dataclasses.replace(cand, criterion=text))
    return kept


# ---- Stage F ---------------------------------------------------------------

def critic_messages(narrative: str, candidates: Sequence[RubricCandidate]) -> list[dict[str, str]]:
    listing = "\n".join(f"{c.candidate_id} [{c.axis}; {c.polarity}] {c.criterion}" for c in candidates)
    prompt = (
        "You audit rubric criteria that will be used to grade answers to the request below. "
        "For EVERY criterion id, give exactly one verdict:\n"
        "ATOMIC (one checkable requirement, keep), SPLIT_REQUIRED (bundles 2-3 independently "
        "checkable requirements; give them in \"split\"), REDUNDANT (substantially duplicates "
        "another listed criterion; keep the better one), TOO_VAGUE, UNVERIFIABLE (cannot be "
        "judged from the answer text alone), OUT_OF_SCOPE (not required by the request).\n"
        "For positive criteria, treat retrieved-evidence specificity as a scope problem: if a criterion "
        "requires a named study, technique, organization, law, case, event, statistic, example, or "
        "source-specific detail that the request does not name and does not genuinely need, mark it "
        "OUT_OF_SCOPE. Evidence may support a broader requirement, but it must not become a hidden "
        "mandatory answer outline. A source-specific detail is acceptable only when the request itself "
        "requires it or the detail is indispensable to satisfying the stated information need. "
        "Do NOT mark a criterion OUT_OF_SCOPE merely because it is implicit rather than named verbatim: "
        "keep broad decision-relevant concepts, standard distinctions, mechanisms, and trade-offs that "
        "are reasonably needed to explain, compare, evaluate, or otherwise satisfy the user's request. "
        "OUT_OF_SCOPE is for arbitrary topic expansion or unnecessary specific examples, not for useful "
        "general dimensions of the requested analysis.\n"
        "Negative criteria are deliberate failure-mode checks: never mark one REDUNDANT merely because "
        "it is the inverse of a positive criterion. "
        "Prefer ATOMIC; only flag clear problems. Each split item must start with \"The response\" "
        "(or \"The response does not\" for negative criteria).\n"
        "Reply with ONLY a JSON array: [{\"id\": \"...\", \"verdict\": \"...\", \"split\": [\"...\"]}] "
        "(\"split\" only for SPLIT_REQUIRED).\n\n"
        f"Request:\n{narrative}\n\nCriteria:\n{listing}"
    )
    return [
        {"role": "system", "content": "You are a careful rubric auditor. Return only the requested JSON."},
        {"role": "user", "content": prompt},
    ]


def parse_critic(text: str, candidates: Sequence[RubricCandidate]) -> dict[str, tuple[str, tuple[str, ...]]]:
    parsed = _extract_json_array(text)
    if parsed is None:
        raise FinalizeError("no JSON array in critic reply")
    known = {c.candidate_id for c in candidates}
    out: dict[str, tuple[str, tuple[str, ...]]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            raise FinalizeError("critic item is not an object")
        cid, verdict = item.get("id"), item.get("verdict")
        if cid not in known:
            raise FinalizeError(f"critic cites unknown id {cid!r}")
        if verdict not in CRITIC_VERDICTS:
            raise FinalizeError(f"invalid verdict {verdict!r}")
        split: tuple[str, ...] = ()
        if verdict == "SPLIT_REQUIRED":
            raw = item.get("split")
            if (isinstance(raw, list) and 2 <= len(raw) <= 3
                    and all(isinstance(s, str) and s.strip() for s in raw)):
                split = tuple(s.strip() for s in raw)
            else:  # unusable split instruction -> keep the criterion as written
                verdict = "ATOMIC"
        out[cid] = (verdict, split)
    if len(out) < len(known) * 0.8:
        raise FinalizeError(f"critic covered only {len(out)}/{len(known)} criteria")
    return out


def apply_critic(
    candidates: Sequence[RubricCandidate], decisions: dict[str, tuple[str, tuple[str, ...]]],
) -> list[RubricCandidate]:
    """Deterministic application. Unlisted ids are ATOMIC. Explicit-axis
    criteria, negative (failure-mode) criteria and the low-volume axes are
    never dropped by the critic (selection trims later): a critic that sees
    only the positive mirror of a negative would otherwise delete whole axes."""
    result: list[RubricCandidate] = []
    for cand in candidates:
        verdict, split = decisions.get(cand.candidate_id, ("ATOMIC", ()))
        if verdict == "ATOMIC":
            result.append(cand)
        elif verdict == "SPLIT_REQUIRED":
            # Keep deliberately low-volume axes and negative failure-mode checks
            # compact. Splitting them adds little grading value and can create
            # surface-negation polarity ambiguity (for example, a desirable
            # reference-quality clause phrased as "does not misrepresent").
            if cand.polarity == "negative" or (
                cand.axis not in _DROPPABLE_AXES and cand.axis != "Explicit Criteria"
            ):
                result.append(cand)
            else:
                for i, text in enumerate(split):
                    result.append(dataclasses.replace(
                        cand, candidate_id=f"{cand.candidate_id}-s{i}", criterion=text,
                    ))
        elif cand.polarity == "negative" or cand.axis not in _DROPPABLE_AXES:
            result.append(cand)
        # else: dropped (REDUNDANT/TOO_VAGUE/UNVERIFIABLE/OUT_OF_SCOPE)
    return prenormalize(result)


# ---- Stage H ---------------------------------------------------------------

def importance_messages(narrative: str, candidates: Sequence[RubricCandidate]) -> list[dict[str, str]]:
    listing = "\n".join(f"{c.candidate_id} [{c.axis}; {c.polarity}] {c.criterion}" for c in candidates)
    prompt = (
        "Classify how important each rubric criterion is for judging answers to the request below.\n"
        "Positive criteria (a good answer satisfies them): CRITICAL_POSITIVE (omitting it makes the "
        "answer fail the user's core need), IMPORTANT_POSITIVE, MINOR_POSITIVE (polish).\n"
        "Negative criteria (a bad answer exhibits them): CRITICAL_NEGATIVE (occurrence invalidates "
        "the answer), DETRIMENTAL, MINOR_NEGATIVE.\n"
        "Set \"strong\": true if the criterion is required for basic correctness, false if merely "
        "strongly desirable/undesirable. Positive criteria must get a *_POSITIVE class and negative "
        "criteria a *_NEGATIVE/DETRIMENTAL class. Be discriminating: at most about a fifth of all "
        "criteria should be CRITICAL. Follow each criterion's declared polarity label, not surface "
        "negation in its wording: a [positive] criterion phrased as 'avoids', 'does not overstate', or "
        "similar still describes desirable good-answer behavior and must receive a *_POSITIVE class. "
        "For positive criteria, do not reward retrieved-example "
        "specificity merely because it is factual: a named study, technique, organization, case, event, "
        "statistic, or source-specific detail absent from the request should be MINOR_POSITIVE with "
        "strong=false unless that exact detail is genuinely necessary to satisfy the user's stated need. "
        "Prefer importance for the underlying concept or decision requirement over a particular example.\n"
        "Reply with ONLY a JSON array covering every id: "
        "[{\"id\": \"...\", \"importance_class\": \"...\", \"strong\": true}].\n\n"
        f"Request:\n{narrative}\n\nCriteria:\n{listing}"
    )
    return [
        {"role": "system", "content": "You are a careful rubric weighting analyst. Return only the requested JSON."},
        {"role": "user", "content": prompt},
    ]


def parse_importance(text: str, candidates: Sequence[RubricCandidate]) -> dict[str, tuple[str, bool]]:
    parsed = _extract_json_array(text)
    if parsed is None:
        raise FinalizeError("no JSON array in importance reply")
    polarity = {c.candidate_id: c.polarity for c in candidates}
    out: dict[str, tuple[str, bool]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            raise FinalizeError("importance item is not an object")
        cid, klass, strong = item.get("id"), item.get("importance_class"), item.get("strong")
        if cid not in polarity:
            raise FinalizeError(f"importance cites unknown id {cid!r}")
        allowed = _POSITIVE_CLASSES if polarity[cid] == "positive" else _NEGATIVE_CLASSES
        if klass not in allowed:
            raise FinalizeError(f"class {klass!r} inconsistent with {polarity[cid]} polarity for {cid!r}")
        if not isinstance(strong, bool):
            raise FinalizeError("strong must be a boolean")
        out[cid] = (klass, strong)
    missing = set(polarity) - set(out)
    if missing:
        raise FinalizeError(f"importance missing {len(missing)} id(s)")
    return out


# ---- Stage I ---------------------------------------------------------------

def select_final(
    candidates: Sequence[RubricCandidate], importance: dict[str, tuple[str, bool]], *, target: int,
) -> list[tuple[RubricCandidate, float]]:
    """Deterministic selection. Priority: every explicit criterion, one best
    per remaining axis, critical negatives, then by |weight| desc, confidence
    desc, id. Result is returned in canonical axis order."""
    weighted = [
        (c, importance_to_weight(importance[c.candidate_id][0], strong=importance[c.candidate_id][1]))
        for c in candidates
    ]
    target = min(target, HARD_MAX)
    rank = lambda cw: (-abs(cw[1]), -cw[0].confidence, cw[0].candidate_id)
    chosen: dict[str, tuple[RubricCandidate, float]] = {}

    for cw in sorted((cw for cw in weighted if cw[0].axis == "Explicit Criteria"), key=rank):
        chosen[cw[0].candidate_id] = cw
    for axis in CANONICAL_AXES:
        best = sorted((cw for cw in weighted if cw[0].axis == axis), key=rank)[:1]
        for cw in best:
            chosen.setdefault(cw[0].candidate_id, cw)
    for cw in sorted((cw for cw in weighted if cw[1] <= -4), key=rank):
        chosen.setdefault(cw[0].candidate_id, cw)
    for cw in sorted(weighted, key=rank):
        if len(chosen) >= target:
            break
        chosen.setdefault(cw[0].candidate_id, cw)

    axis_order = {axis: i for i, axis in enumerate(CANONICAL_AXES)}
    return sorted(chosen.values(), key=lambda cw: (axis_order[cw[0].axis], cw[0].candidate_id))[:HARD_MAX]


def to_rubric_topic(
    characterization: TopicCharacterization, selected: Sequence[tuple[RubricCandidate, float]],
) -> RubricTopic:
    return RubricTopic(
        qid=characterization.qid, domain=characterization.domain,
        conceptual_breadth=characterization.conceptual_breadth,
        logical_nesting=characterization.logical_nesting, exploration=characterization.exploration,
        rubrics=tuple(RubricCriterion(criterion=c.criterion, weight=w, axis=c.axis) for c, w in selected),
    )


# ---- Orchestration of F + H + I for one topic ------------------------------

def finalize_topic(
    narrative: str,
    characterization: TopicCharacterization,
    candidates: Sequence[RubricCandidate],
    *,
    model_id: str, region: str, tier: str, fallback_tier: Optional[str],
    cache: RubricLlmCache, cost_tracker: CostTracker, max_attempts: int = 3,
) -> RubricTopic:
    qid = characterization.qid
    normalized = prenormalize(candidates)

    def call(stage: str, version: str, messages: list[dict[str, str]], parse: Callable):
        key = cache_key(stage=stage, prompt_version=version, model_id=model_id, tier=tier, region=region,
                        messages=messages)
        return cached_bedrock_call(
            key, messages, model_id=model_id, region=region, tier=tier, fallback_tier=fallback_tier,
            cache=cache, cost_tracker=cost_tracker, stage=stage, qid=qid, parse=parse,
            error_type=FinalizeError, max_attempts=max_attempts,
        )

    decisions = call("critic", CRITIC_PROMPT_VERSION, critic_messages(narrative, normalized),
                     lambda text: parse_critic(text, normalized))
    atomic = apply_critic(normalized, decisions)
    importance = call("importance", IMPORTANCE_PROMPT_VERSION, importance_messages(narrative, atomic),
                      lambda text: parse_importance(text, atomic))
    target = FINAL_COUNT_BY_BREADTH[characterization.conceptual_breadth]
    return to_rubric_topic(characterization, select_final(atomic, importance, target=target))
