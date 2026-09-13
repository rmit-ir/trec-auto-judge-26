"""Stage B (ClimbMix retrieval) glue + Stage C (evidence condensation)
(execution-plan Sections 10-11). Response-independent: retrieval intents come
only from Stage A's planner output, never from a candidate answer/citation.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

from ..hybrid_retrieval import (
    HybridConfig,
    HybridResearchPack,
    HybridResearchRetriever,
    ResearchHit,
    ResearchIntent,
    build_hybrid_research_pack,
)
from .cache import RubricLlmCache, cache_key
from .generation_models import EvidenceNote, TopicCharacterization
from .llm import CostTracker, LlmResult, call_bedrock
from .planner import characterize_topic
from .topics import Topic

EVIDENCE_PROMPT_VERSION = "v1"

_EVIDENCE_SYSTEM = "You are a careful evidence analyst. Return only the requested JSON."
_VALID_EVIDENCE_TYPES = {"fact", "number", "argument", "counterargument", "context"}


def plan_and_research(
    topic: Topic,
    retriever: HybridResearchRetriever,
    *,
    model_id: str,
    region: str,
    tier: str,
    fallback_tier: Optional[str],
    cache: RubricLlmCache,
    cost_tracker: CostTracker,
    hybrid_config: Optional[HybridConfig] = None,
    max_attempts: int = 3,
) -> tuple[TopicCharacterization, HybridResearchPack]:
    """Stage A + Stage B combined: characterize/plan intents (one cached
    Bedrock call), then run those intents through the deterministic hybrid
    retriever (no LLM call -- see hybrid_retrieval.py). The already-produced
    intents are wrapped as a trivial IntentPlanner so build_hybrid_research_pack's
    existing provenance/cache-key machinery is reused rather than duplicated."""
    characterization, intents = characterize_topic(
        topic, model_id=model_id, region=region, tier=tier, fallback_tier=fallback_tier,
        cache=cache, cost_tracker=cost_tracker, max_attempts=max_attempts,
    )
    pack = build_hybrid_research_pack(
        topic, retriever, planner=lambda _topic: intents, config=hybrid_config,
    )
    return characterization, pack


def evidence_condensation_messages(
    narrative: str,
    characterization: TopicCharacterization,
    hits: Sequence[ResearchHit],
    *,
    max_chars_per_doc: int = 1200,
) -> list[dict[str, str]]:
    """Stage C prompt (execution-plan Section 11). Only hits with resolved
    text can be cited -- a hit with no text contributes nothing rather than
    being silently dropped from the docid universe the model may cite."""
    passages = "\n\n".join(
        f"[{hit.chunk_id}] {hit.text[:max_chars_per_doc]}"
        for hit in hits if hit.text
    )
    prompt = (
        "Condense the passages below into compact, attributable evidence notes "
        "for building a topic rubric later. Ground every note ONLY in the "
        "bracketed docids provided -- never cite a docid that is not listed, "
        "and never invent a fact absent from the passages. Preserve numbers, "
        "dates, and named entities exactly as given. If evidence conflicts "
        "across passages, write a disagreement note rather than asserting one "
        "side as settled. Distinguish core decision-relevant evidence from "
        "background -- do not turn every passage into a note.\n\n"
        f"Request:\n{narrative}\n\n"
        f"Domain: {characterization.domain}\n"
        f"Explicit subquestions: {list(characterization.explicit_subquestions)}\n"
        f"Synthesis goal: {characterization.synthesis_goal}\n\n"
        f"Passages (cite only by the bracketed id):\n{passages}\n\n"
        "Reply with ONLY a JSON array of evidence notes, each object exactly "
        'shaped: {"claim": "...", "docids": ["..."], "subtopic": "...", '
        '"evidence_type": "fact|number|argument|counterargument|context", '
        '"confidence": 0.0-1.0, "independent_sources": integer>=1}'
    )
    return [
        {"role": "system", "content": _EVIDENCE_SYSTEM},
        {"role": "user", "content": prompt},
    ]


class EvidenceCondensationError(ValueError):
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


def parse_evidence_notes(
    text: str,
    *,
    valid_docids: set[str],
    min_notes: int,
    max_notes: int,
) -> list[EvidenceNote]:
    """Evidence IDs are never taken from the model -- deterministic code
    assigns them (`ev-000`, ...) so ID uniqueness/format is never a model
    failure mode (same "don't let the model choose the final value"
    principle as weights.py's importance_to_weight())."""
    parsed = _extract_json_array(text)
    if parsed is None:
        raise EvidenceCondensationError("no JSON array found in response")
    if not (min_notes <= len(parsed) <= max_notes):
        raise EvidenceCondensationError(f"{len(parsed)} evidence notes outside [{min_notes}, {max_notes}]")

    notes: list[EvidenceNote] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise EvidenceCondensationError(f"evidence note is not an object: {item!r}")
        claim = item.get("claim")
        docids = item.get("docids")
        subtopic = item.get("subtopic")
        evidence_type = item.get("evidence_type")
        confidence = item.get("confidence")
        independent_sources = item.get("independent_sources", 1)

        if not isinstance(claim, str) or not claim.strip():
            raise EvidenceCondensationError(f"note {index}: empty/missing claim")
        if not isinstance(docids, list) or not docids or not all(isinstance(d, str) for d in docids):
            raise EvidenceCondensationError(f"note {index}: docids must be a non-empty list of strings")
        unknown = [d for d in docids if d not in valid_docids]
        if unknown:
            raise EvidenceCondensationError(f"note {index}: cites unlisted docid(s) {unknown}")
        if not isinstance(subtopic, str) or not subtopic.strip():
            raise EvidenceCondensationError(f"note {index}: empty/missing subtopic")
        if evidence_type not in _VALID_EVIDENCE_TYPES:
            raise EvidenceCondensationError(f"note {index}: invalid evidence_type {evidence_type!r}")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise EvidenceCondensationError(f"note {index}: confidence must be a number in [0, 1]")
        if isinstance(independent_sources, bool) or not isinstance(independent_sources, int) or independent_sources < 1:
            raise EvidenceCondensationError(f"note {index}: independent_sources must be a positive integer")

        notes.append(EvidenceNote(
            evidence_id=f"ev-{index:03d}",
            claim=claim.strip(),
            docids=tuple(docids),
            subtopic=subtopic.strip(),
            evidence_type=evidence_type,
            confidence=float(confidence),
            independent_sources=independent_sources,
        ))
    return notes


def condense_evidence(
    topic: Topic,
    characterization: TopicCharacterization,
    pack: HybridResearchPack,
    *,
    model_id: str,
    region: str,
    tier: str,
    fallback_tier: Optional[str],
    cache: RubricLlmCache,
    cost_tracker: CostTracker,
    min_notes: int = 25,
    max_notes: int = 45,
    max_chars_per_doc: int = 1200,
    max_attempts: int = 3,
) -> list[EvidenceNote]:
    """Stage C end-to-end for one topic (execution-plan Section 11). Never
    calls candidate-response data -- `pack` was built purely from Stage A/B."""
    messages = evidence_condensation_messages(
        topic.narrative, characterization, pack.hits, max_chars_per_doc=max_chars_per_doc,
    )
    valid_docids = {hit.chunk_id for hit in pack.hits if hit.text}
    key = cache_key(
        stage="evidence", prompt_version=EVIDENCE_PROMPT_VERSION,
        model_id=model_id, tier=tier, region=region, messages=messages,
    )
    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        cached = cache.get(key) if attempt == 0 else None
        if cached is not None:
            try:
                notes = parse_evidence_notes(cached.text, valid_docids=valid_docids, min_notes=min_notes, max_notes=max_notes)
            except EvidenceCondensationError as error:
                last_error = error
            else:
                cost_tracker.record_bedrock(
                    LlmResult(text=cached.text, input_tokens=cached.input_tokens,
                              output_tokens=cached.output_tokens, model_id=model_id, tier=tier, region=region),
                    stage="evidence", cache_hit=True, qid=topic.qid,
                )
                return notes

        try:
            call = call_bedrock(model_id, messages, region=region, tier=tier, fallback_tier=fallback_tier)
        except Exception as error:
            last_error = error
            continue

        try:
            notes = parse_evidence_notes(call.text, valid_docids=valid_docids, min_notes=min_notes, max_notes=max_notes)
        except EvidenceCondensationError as error:
            last_error = error
            continue

        cache.put(key, text=call.text, input_tokens=call.input_tokens, output_tokens=call.output_tokens)
        cost_tracker.record_bedrock(call, stage="evidence", cache_hit=False, qid=topic.qid)
        return notes

    raise EvidenceCondensationError(
        f"condense_evidence failed after {max_attempts} attempt(s) for qid={topic.qid}: {last_error}"
    )
