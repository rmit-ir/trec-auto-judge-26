"""Response-independent disk cache for offline rubric-construction LLM calls
(execution-plan Section 6 / 33).

Reuses ``minima_llm.PromptCache`` (SQLite, multi-process safe, already a
project dependency) rather than building a new cache -- it already does
exactly what's needed: a deterministic key -> (text, raw_json) store. This
module only adds the deterministic key derivation and a typed
hit/miss-with-usage result, so cost can be tracked on cache hits too (a hit
still has known token counts, just no new spend).

Cache keys must never depend on candidate-response content -- callers pass
only topic/prompt/model/config material into ``cache_key()``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from minima_llm import PromptCache


def cache_key(
    *,
    stage: str,
    prompt_version: str,
    model_id: str,
    tier: str,
    region: str,
    messages: Sequence[dict],
) -> str:
    """Deterministic key: stable JSON serialization, no timestamps/UUIDs/
    unordered dict rendering. ``messages`` must already be response-
    independent -- this function does not itself enforce that."""
    payload = {
        "stage": stage,
        "prompt_version": prompt_version,
        "model_id": model_id,
        "tier": tier,
        "region": region,
        "messages": list(messages),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedCall:
    text: str
    input_tokens: int
    output_tokens: int
    cache_hit: bool


class RubricLlmCache:
    """Thin wrapper binding ``PromptCache`` to this module's key scheme and
    usage bookkeeping. Never stores prompt/response text alongside anything
    that would make it look like judge output -- this cache is for offline
    construction only and is never read by ``RAG26ArenaJudge`` at runtime."""

    def __init__(self, db_path: Path | str):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = PromptCache(str(db_path))

    def get(self, key: str) -> Optional[CachedCall]:
        hit = self._cache.get(key)
        if hit is None:
            return None
        text, raw = hit
        usage = (raw or {}).get("usage", {}) or {}
        return CachedCall(
            text=text,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            cache_hit=True,
        )

    def put(self, key: str, *, text: str, input_tokens: int, output_tokens: int, extra: Optional[dict[str, Any]] = None) -> None:
        raw = {"usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}, **(extra or {})}
        self._cache.put(key, text, raw)

    def close(self) -> None:
        self._cache.close()
