"""Stage A: task characterization + retrieval-intent planning, combined into
one Qwen3 235B call per topic (execution-plan Section 9). Response-
independent: only ``Topic.narrative`` (the public request) ever enters the
prompt.
"""

from __future__ import annotations

import json
from typing import Optional

from ..hybrid_retrieval import ResearchIntent
from .cache import RubricLlmCache, cache_key
from .generation_models import TopicCharacterization
from .llm import CostTracker, LlmResult, call_bedrock
from .prompts import CHARACTERIZE_PROMPT_VERSION, characterize_and_plan_messages
from .topics import Topic

_VALID_BREADTH = {"Simple", "Moderate", "High"}
_VALID_NESTING = {"Shallow", "Intermediate", "Deep"}
_VALID_EXPLORATION = {"Low", "Medium", "High"}
_VALID_RETRIEVAL_MODE = {"dense", "sparse", "hybrid"}
_MIN_INTENTS = 6
_MAX_INTENTS = 12

# hybrid_retrieval.ResearchIntent's `kind` routes dense-only/sparse-only/both;
# the plan's public retrieval_mode vocabulary maps onto it directly.
_KIND_BY_RETRIEVAL_MODE = {"dense": "semantic", "sparse": "lexical", "hybrid": "hybrid"}


class CharacterizationError(ValueError):
    """The model's structured output failed validation. Callers treat this
    exactly like a call failure (bounded retry, never a partial accept)."""


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


def parse_characterization(qid: str, text: str) -> tuple[TopicCharacterization, list[ResearchIntent]]:
    parsed = _extract_json_object(text)
    if parsed is None:
        raise CharacterizationError("no JSON object found in response")

    breadth = parsed.get("conceptual_breadth")
    nesting = parsed.get("logical_nesting")
    exploration = parsed.get("exploration")
    domain = parsed.get("domain")
    synthesis_goal = parsed.get("synthesis_goal")
    subquestions = parsed.get("explicit_subquestions")
    constraints = parsed.get("named_constraints")
    intents_raw = parsed.get("research_intents")

    if breadth not in _VALID_BREADTH:
        raise CharacterizationError(f"invalid conceptual_breadth: {breadth!r}")
    if nesting not in _VALID_NESTING:
        raise CharacterizationError(f"invalid logical_nesting: {nesting!r}")
    if exploration not in _VALID_EXPLORATION:
        raise CharacterizationError(f"invalid exploration: {exploration!r}")
    if not isinstance(domain, str) or not domain.strip():
        raise CharacterizationError("missing/empty domain")
    if not isinstance(synthesis_goal, str) or not synthesis_goal.strip():
        raise CharacterizationError("missing/empty synthesis_goal")
    if not isinstance(subquestions, list) or not all(isinstance(s, str) and s.strip() for s in subquestions):
        raise CharacterizationError("invalid explicit_subquestions")
    if not isinstance(constraints, list) or not all(isinstance(s, str) for s in constraints):
        raise CharacterizationError("invalid named_constraints")
    if not isinstance(intents_raw, list) or not intents_raw:
        raise CharacterizationError("missing/empty research_intents")

    intents: list[ResearchIntent] = []
    for item in intents_raw:
        if not isinstance(item, dict):
            raise CharacterizationError(f"research_intent is not an object: {item!r}")
        query = item.get("query")
        mode = item.get("retrieval_mode")
        purpose = item.get("purpose", "")
        if not isinstance(query, str) or not query.strip():
            raise CharacterizationError("research_intent has an empty query")
        if mode not in _VALID_RETRIEVAL_MODE:
            raise CharacterizationError(f"invalid retrieval_mode: {mode!r}")
        if not isinstance(purpose, str):
            raise CharacterizationError("research_intent purpose must be a string")
        intents.append(ResearchIntent(text=query.strip(), kind=_KIND_BY_RETRIEVAL_MODE[mode], purpose=purpose.strip()))

    # Structural completeness proxy only (execution-plan Section 9's "at least
    # one intent per explicit subquestion"): exact semantic correspondence
    # needs a model judgment, not deterministic code. This just rejects a
    # degenerate response that lists many subquestions but far fewer intents.
    if len(intents) < max(1, len(subquestions)):
        raise CharacterizationError(
            f"only {len(intents)} research_intents for {len(subquestions)} explicit_subquestions"
        )
    if not (_MIN_INTENTS <= len(intents) <= _MAX_INTENTS):
        raise CharacterizationError(f"research_intents count {len(intents)} outside the planned {_MIN_INTENTS}-10 range")

    characterization = TopicCharacterization(
        qid=qid, domain=domain.strip(),
        conceptual_breadth=breadth, logical_nesting=nesting, exploration=exploration,
        explicit_subquestions=tuple(s.strip() for s in subquestions),
        named_constraints=tuple(s.strip() for s in constraints),
        synthesis_goal=synthesis_goal.strip(),
    )
    return characterization, intents


def characterize_topic(
    topic: Topic,
    *,
    model_id: str,
    region: str,
    tier: str,
    fallback_tier: Optional[str],
    cache: RubricLlmCache,
    cost_tracker: CostTracker,
    max_attempts: int = 3,
) -> tuple[TopicCharacterization, list[ResearchIntent]]:
    """Stage A end-to-end for one topic: cache lookup, Bedrock call on miss,
    parse+validate, cost tracking. A schema-invalid reply is treated as a
    call failure and retried (bounded), never partially accepted."""
    messages = characterize_and_plan_messages(topic.narrative)
    key = cache_key(
        stage="characterize", prompt_version=CHARACTERIZE_PROMPT_VERSION,
        model_id=model_id, tier=tier, region=region, messages=messages,
    )
    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        cached = cache.get(key) if attempt == 0 else None
        if cached is not None:
            try:
                result = parse_characterization(topic.qid, cached.text)
            except CharacterizationError as error:
                last_error = error  # stale/corrupt cache entry: fall through to a live call
            else:
                cost_tracker.record_bedrock(
                    LlmResult(
                        text=cached.text, input_tokens=cached.input_tokens,
                        output_tokens=cached.output_tokens, model_id=model_id, tier=tier, region=region,
                    ),
                    stage="characterize", cache_hit=True, qid=topic.qid,
                )
                return result

        try:
            call = call_bedrock(model_id, messages, region=region, tier=tier, fallback_tier=fallback_tier)
        except Exception as error:  # BedrockUnavailableError or an unexpected transport error
            last_error = error
            continue

        try:
            result = parse_characterization(topic.qid, call.text)
        except CharacterizationError as error:
            last_error = error
            continue

        cache.put(key, text=call.text, input_tokens=call.input_tokens, output_tokens=call.output_tokens)
        cost_tracker.record_bedrock(call, stage="characterize", cache_hit=False, qid=topic.qid)
        return result

    raise CharacterizationError(
        f"characterize_topic failed after {max_attempts} attempt(s) for qid={topic.qid}: {last_error}"
    )
