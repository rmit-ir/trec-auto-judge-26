"""Offline LLM adapters for rubric construction (execution-plan Section 6).

Two providers, deliberately kept separate from the runtime judge:

- Bedrock (Qwen3 235B/32B): no existing minima-llm backend for Bedrock, so
  this module owns a small boto3 Converse wrapper with bounded retry and an
  explicit (never silent) Flex -> Standard tier fallback.
- Luna (GPT-5.6, independent reviewer): routed through the SAME injected
  OPENAI_BASE_URL/OPENAI_MODEL/OPENAI_API_KEY endpoint and minima-llm client
  the runtime judge uses (already configured to Luna). minima-llm already
  retries and caches (via CACHE_DIR) internally, so this is a thin wrapper,
  not a second Azure adapter.

Never imported by ``RAG26ArenaJudge`` at runtime (execution-plan Section 6:
"offline construction and runtime judging remain separate concerns").
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from minima_llm import MinimaLlmConfig, MinimaLlmRequest, MinimaLlmResponse, OpenAIMinimaLlm

from .cache import RubricLlmCache
from .generation_models import UsageRecord

_T = TypeVar("_T")

_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from bedrock_cost import estimate_cost as bedrock_estimate_cost  # noqa: E402


@dataclass(frozen=True)
class LlmResult:
    text: str
    input_tokens: int
    output_tokens: int
    model_id: str
    tier: str
    region: str


class BedrockUnavailableError(RuntimeError):
    """Every attempt, including any tier fallback, failed."""


def _to_converse_messages(messages: Sequence[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    """Split OpenAI-style [{"role","content"}] into Converse's separate
    `system` list and `messages` list (Converse has no "system" role)."""
    system_blocks = [{"text": m["content"]} for m in messages if m["role"] == "system"]
    turns = [
        {"role": m["role"], "content": [{"text": m["content"]}]}
        for m in messages if m["role"] != "system"
    ]
    return system_blocks, turns


def call_bedrock(
    model_id: str,
    messages: Sequence[dict[str, str]],
    *,
    region: str,
    tier: str = "flex",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    max_attempts: int = 3,
    retry_backoff_seconds: float = 1.0,
    fallback_tier: Optional[str] = None,
) -> LlmResult:
    """One logical Bedrock Converse call with bounded retry per tier, and an
    explicit tier fallback (e.g. Flex -> Standard, execution-plan Section 5's
    fallback order) -- never a silent fallback to a different model."""
    system_blocks, turns = _to_converse_messages(messages)
    tiers_to_try = [tier] + ([fallback_tier] if fallback_tier and fallback_tier != tier else [])
    last_error: Optional[BaseException] = None

    for tier_index, attempt_tier in enumerate(tiers_to_try):
        for attempt in range(max_attempts):
            if attempt:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
            try:
                client = boto3.client("bedrock-runtime", region_name=region)
                request: dict[str, Any] = {
                    "modelId": model_id,
                    "messages": turns,
                    "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
                }
                if system_blocks:
                    request["system"] = system_blocks
                if attempt_tier != "default":
                    request["serviceTier"] = {"type": attempt_tier}
                response = client.converse(**request)
            except (BotoCoreError, ClientError) as error:
                last_error = error
                continue
            content = response.get("output", {}).get("message", {}).get("content", [])
            text = "".join(block.get("text", "") for block in content)
            if not text:
                last_error = RuntimeError("Bedrock returned no content blocks")
                continue
            usage = response.get("usage", {})
            return LlmResult(
                text=text,
                input_tokens=usage.get("inputTokens", 0),
                output_tokens=usage.get("outputTokens", 0),
                model_id=model_id, tier=attempt_tier, region=region,
            )
        if tier_index + 1 < len(tiers_to_try):
            print(
                f"[rag26-rubrics] Bedrock {model_id} tier={attempt_tier!r} exhausted "
                f"{max_attempts} attempt(s) ({type(last_error).__name__ if last_error else 'no content'}); "
                f"falling back to tier={tiers_to_try[tier_index + 1]!r}"
            )

    raise BedrockUnavailableError(
        f"Bedrock {model_id} unavailable in {region} after trying tier(s) {tiers_to_try}: {last_error}"
    )


def call_luna(messages: Sequence[dict[str, str]], *, request_id: str = "rubric-luna-0") -> LlmResult:
    """One Luna review call through the injected OpenAI-compatible endpoint.
    minima-llm's OpenAIMinimaLlm already retries and caches internally
    (MAX_ATTEMPTS / CACHE_DIR env vars), so this issues exactly one logical
    request and raises on failure rather than adding a second retry layer."""
    config = MinimaLlmConfig.from_env()
    backend = OpenAIMinimaLlm(config)
    request = MinimaLlmRequest(request_id=request_id, messages=list(messages))
    [result] = asyncio.run(backend.run_batched([request]))
    if not isinstance(result, MinimaLlmResponse) or not result.text:
        raise RuntimeError(f"Luna call {request_id!r} failed: {result!r}")
    return LlmResult(
        text=result.text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        model_id=config.model, tier="default", region="",
    )


def cached_bedrock_call(
    key: str,
    messages: Sequence[dict[str, str]],
    *,
    model_id: str,
    region: str,
    tier: str,
    fallback_tier: Optional[str],
    cache: RubricLlmCache,
    cost_tracker: CostTracker,
    stage: str,
    qid: Optional[str],
    parse: Callable[[str], _T],
    error_type: type[Exception] = RuntimeError,
    max_attempts: int = 3,
) -> _T:
    """Shared cache-lookup / Bedrock-call / parse-with-retry loop used by every
    generation stage (planner.py's characterize_topic, evidence.py's
    condense_evidence, generate.py's axis generators). ``parse`` raises on any
    structural problem with the reply text; that is treated exactly like a
    call failure -- bounded retry, never a partial accept, and a stale/corrupt
    cache entry (should not happen since only validated text is ever cached,
    but costs nothing to handle) falls through to a live call rather than
    raising immediately."""
    last_error: Optional[BaseException] = None
    for attempt in range(max_attempts):
        cached = cache.get(key) if attempt == 0 else None
        if cached is not None:
            try:
                result = parse(cached.text)
            except Exception as error:
                last_error = error
            else:
                cost_tracker.record_bedrock(
                    LlmResult(text=cached.text, input_tokens=cached.input_tokens,
                              output_tokens=cached.output_tokens, model_id=model_id, tier=tier, region=region),
                    stage=stage, cache_hit=True, qid=qid,
                )
                return result

        try:
            call = call_bedrock(model_id, messages, region=region, tier=tier, fallback_tier=fallback_tier)
        except Exception as error:
            last_error = error
            continue

        # A real, billed call succeeded at the transport level -- record its cost
        # now, before parsing. AWS billed for these tokens whether or not the
        # reply passes our own structural validation next.
        cost_tracker.record_bedrock(call, stage=stage, cache_hit=False, qid=qid)

        try:
            result = parse(call.text)
        except Exception as error:
            last_error = error
            continue

        cache.put(key, text=call.text, input_tokens=call.input_tokens, output_tokens=call.output_tokens)
        return result

    raise error_type(f"{stage} failed after {max_attempts} attempt(s) for qid={qid}: {last_error}")


class BudgetExceededError(RuntimeError):
    """Raised by CostTracker.check_budget() when accumulated spend exceeds
    the configured hard cap (execution-plan Section 4's $25 workstream cap).
    Local/in-process only -- for cross-process enforcement while a batch
    subprocess runs, use scripts/cost_guard.py against the same cache dir."""


@dataclass
class CostTracker:
    """Accumulates UsageRecords and reports the required per-stage/pre-batch
    budget information (execution-plan Section 4/23). Never stores prompt or
    response text -- only the numeric fields UsageRecord already carries."""

    hard_cap_usd: Decimal
    records: list[UsageRecord] = field(default_factory=list)

    def record_bedrock(self, result: LlmResult, *, stage: str, cache_hit: bool, qid: Optional[str] = None) -> UsageRecord:
        cost = Decimal("0") if cache_hit else bedrock_estimate_cost(
            result.model_id, result.input_tokens, result.output_tokens, tier=result.tier,
        )
        record = UsageRecord(
            stage=stage, model_id=result.model_id, tier=result.tier, region=result.region,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            estimated_usd=cost, cache_hit=cache_hit, qid=qid,
        )
        self.records.append(record)
        return record

    def total_usd(self) -> Decimal:
        return sum((r.estimated_usd for r in self.records), Decimal("0"))

    def by_stage(self) -> dict[str, Decimal]:
        totals: dict[str, Decimal] = {}
        for record in self.records:
            totals[record.stage] = totals.get(record.stage, Decimal("0")) + record.estimated_usd
        return totals

    def check_budget(self) -> None:
        total = self.total_usd()
        if total > self.hard_cap_usd:
            raise BudgetExceededError(
                f"rubric-construction spend ${total} exceeds hard cap ${self.hard_cap_usd}"
            )

    def print_summary(self) -> None:
        print(f"[rag26-rubrics] spend so far: ${self.total_usd():.4f} of ${self.hard_cap_usd} cap")
        for stage, cost in sorted(self.by_stage().items()):
            calls = sum(1 for r in self.records if r.stage == stage)
            hits = sum(1 for r in self.records if r.stage == stage and r.cache_hit)
            print(f"  {stage}: ${cost:.4f} ({calls} call(s), {hits} cache hit(s))")


def print_budget_banner(
    *, model_id: str, tier: str, region: str, topic_count: int, stage: str,
    ceiling_input_tokens: int, ceiling_output_tokens: int, ceiling_usd: Decimal,
    cache_status: str,
) -> None:
    """Required before each paid batch (execution-plan Section 4)."""
    print(
        "[rag26-rubrics] about to spend:\n"
        f"  model={model_id} tier={tier} region={region}\n"
        f"  stage={stage} topics={topic_count}\n"
        f"  ceiling: {ceiling_input_tokens:,} input + {ceiling_output_tokens:,} output tokens, ${ceiling_usd}\n"
        f"  cache: {cache_status}"
    )
