#!/usr/bin/env python3
"""Blind, budget-capped bake-off for ResearchRubrics constructor models.

Candidates see only public development topic prompts. GPT-6.1 Sol sees the
prompt, the organizer development rubric as a semantic reference, and
anonymized candidate rubrics labelled A-D. It never receives model identities.

Example:
    set -a; source .env; set +a
    python scripts/benchmark_rubric_models.py --out-dir evaluation/rubric-model-bakeoff
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

import boto3
from aws_bedrock_token_generator import provide_token
from botocore.exceptions import BotoCoreError, ClientError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS = ROOT / "local-data/researchrubrics-dev/research-rubrics-topics-dev.tsv"
DEFAULT_REFERENCES = ROOT / "local-data/researchrubrics-dev/research-rubrics-dev-rubrics.jsonl"

DEFAULT_QIDS = (
    "683a58c9a7e7fe4e7695846f",
    "683a58c9a7e7fe4e7695848b",
    "684397d188c1deceb49af31d",
    "6847465956a0f6376a605387",
    "6847465956a0f6376a6053c9",
    "6847465956a0f6376a60542d",
)

AXES = (
    "Explicit Criteria",
    "Implicit Criteria",
    "Synthesis of Information",
    "References & Citation Quality",
    "Communication Quality",
    "Instruction Following",
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    provider: str
    model_id: str
    region: str
    tier: str
    input_per_million: Decimal
    output_per_million: Decimal
    normalized_input_per_million: Decimal
    normalized_output_per_million: Decimal
    max_output_tokens: int = 7000
    reasoning_effort: str | None = None


# USD / 1M tokens. Checked 2026-10-14 against AWS public pricing.
# Quality calls use Standard for predictable latency. normalized_* stores the
# production deployment price: Qwen/GLM Sydney Flex and Gemma US-East Flex.
# Luna has no Flex tier; Mantle US-East is used for the live call and the
# normalized price records the cheaper Global CRIS production alternative.
CANDIDATES: tuple[ModelSpec, ...] = (
    ModelSpec(
        "luna", "GPT-6 Luna", "mantle_bearer",
        "openai.gpt-6-luna", "us-east-1", "default",
        Decimal("0.11"), Decimal("0.55"),
        Decimal("0.10"), Decimal("0.50"),
        reasoning_effort="high",
    ),
    ModelSpec(
        "qwen235", "Qwen3 235B A22B 2507", "bedrock_converse",
        "qwen.qwen3-235b-a22b-2507-v1:0", "ap-southeast-2", "default",
        Decimal("0.2266"), Decimal("0.9064"),
        Decimal("0.1133"), Decimal("0.4532"),
    ),
    ModelSpec(
        "gemma4", "Gemma 4 26B-A4B", "mantle_bearer",
        "google.gemma-4-26b-a4b", "us-east-1", "default",
        Decimal("0.13"), Decimal("0.40"),
        Decimal("0.065"), Decimal("0.20"),
        reasoning_effort="high",
    ),
    ModelSpec(
        "glm47flash", "GLM-4.7 Flash", "bedrock_converse",
        "zai.glm-4.7-flash", "ap-southeast-2", "default",
        Decimal("0.0721"), Decimal("0.412"),
        Decimal("0.03605"), Decimal("0.206"),
    ),
)

JUDGE = ModelSpec(
    "judge", "GPT-6.1 Sol", "mantle_bearer",
    "openai.gpt-6.1-sol", "us-east-1", "default",
    Decimal("2.20"), Decimal("11.00"),
    Decimal("2.00"), Decimal("10.00"),
    max_output_tokens=8000,
    reasoning_effort="high",
)

GENERATION_SYSTEM = (
    "You are constructing a ResearchRubrics evaluation rubric. Work only from "
    "the user's request; do not assume candidate answers exist. Return only JSON."
)

AXIS_GUIDE = """Allowed axes:
- Explicit Criteria: requirements the user directly asked for.
- Implicit Criteria: expert expectations, prerequisites, caveats, confounders,
  uncertainty, decision-relevant context, or missing considerations reasonably
  necessary even though not literally requested.
- Synthesis of Information: relationships requiring comparison, reconciliation,
  weighing, connection, distinction, explanation, or contextualization.
- References & Citation Quality: topic-specific evidentiary support, attribution,
  source quality, or handling of conflicting evidence.
- Communication Quality: topic-specific clarity/usability, not generic style.
- Instruction Following: actual scope, audience, format, or decision constraints,
  excluding factual content requirements."""

GENERATION_PROMPT = """Construct a high-quality rubric for grading responses to
this request.

{axis_guide}

Aim for roughly 20-30 atomic criteria; the organizer development set averages
about 25, but do not invent filler to reach a quota. Cover every distinct
explicit ask, important implicit expectations, genuine synthesis requirements,
and a small number of topic-specific reference/communication/instruction
criteria. Include negative criteria only for materially harmful failures
(factual reversal, causal overclaim, fabricated certainty, materially
misleading omission, source misattribution, scope violation, etc.); do not
create negatives that merely mirror positive criteria.

Each criterion must be observable from the eventual answer, specific enough
for a grader to apply, and mostly atomic. Avoid duplicates and arbitrary
requirements not justified by the request. Preserve named entities, dates,
comparisons, stakeholders, and requested decisions.

Return ONLY this JSON object:
{{
  "rubrics": [
    {{
      "criterion": "The response ...",
      "axis": "<one allowed axis>",
      "polarity": "positive|negative"
    }}
  ]
}}

Request:
{prompt}
"""

JUDGE_SYSTEM = (
    "You are an expert rubric-quality evaluator. Candidate rubrics are blindly "
    "labelled A-D. Never infer or speculate about their source model. Evaluate "
    "only their content and return only JSON."
)

JUDGE_PROMPT = """Judge four candidate ResearchRubrics for the request below.
The organizer's human development rubric is a semantic reference. Do NOT reward
surface-wording overlap; reward semantic coverage and rubric quality. Reference
weights indicate relative importance/polarity, but candidates need not reproduce
the numeric weights.

Score each candidate independently:
- reference_coverage (0-40): important human-reference obligations, weighted
  toward high-|weight| criteria.
- precision (0-20): avoids arbitrary requirements, factual overreach, and
  unjustified scope expansion.
- atomicity_judgeability (0-15): specific, observable, mostly atomic,
  non-duplicative, practical criteria.
- implicit_synthesis (0-15): strong implicit expectations and genuine synthesis.
- axis_failure_modes (0-10): sensible axes and useful non-mirror failure modes.

The total must equal the five dimensions (0-100). Rank A-D best to worst. Ties
are allowed only when genuinely indistinguishable. Do not use length alone as
a quality signal.

Request:
{prompt}

HUMAN REFERENCE RUBRIC:
{reference}

BLIND CANDIDATE RUBRICS:
{candidates}

Return ONLY:
{{
  "scores": {{
    "A": {{
      "reference_coverage": 0,
      "precision": 0,
      "atomicity_judgeability": 0,
      "implicit_synthesis": 0,
      "axis_failure_modes": 0,
      "total": 0,
      "notes": "brief evidence-based assessment"
    }},
    "B": {{...}},
    "C": {{...}},
    "D": {{...}}
  }},
  "ranking": ["A", "B", "C", "D"],
  "winner": "A",
  "confidence": 0.0,
  "decision_notes": "brief explanation"
}}
"""


@dataclass
class Usage:
    qid: str
    phase: str
    model_key: str
    model_id: str
    provider: str
    tier: str
    input_tokens: int
    output_tokens: int
    estimated_usd: Decimal
    normalized_usd: Decimal
    attempt: int


@dataclass
class CandidateRubric:
    rubrics: list[dict[str, str]]


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(
        self, hard_cap: Decimal, prior_spend: Decimal = Decimal("0")
    ):
        self.hard_cap = hard_cap
        self.prior_spend = prior_spend
        self.records: list[Usage] = []

    @property
    def tracked_spend(self) -> Decimal:
        return sum((x.estimated_usd for x in self.records), Decimal("0"))

    @property
    def spent(self) -> Decimal:
        return self.prior_spend + self.tracked_spend

    def reserve_check(
        self,
        spec: ModelSpec,
        estimated_input_tokens: int,
        max_output_tokens: int,
    ) -> None:
        worst = token_cost(
            spec, estimated_input_tokens, max_output_tokens, normalized=False
        )
        if self.spent + worst > self.hard_cap:
            raise BudgetExceeded(
                f"next-call reserve ${worst:.4f} would exceed "
                f"${self.hard_cap} cap (spent=${self.spent:.4f})"
            )

    def record(self, usage: Usage) -> None:
        if self.spent + usage.estimated_usd > self.hard_cap:
            raise BudgetExceeded(
                f"recorded spend would exceed ${self.hard_cap}: "
                f"${self.spent + usage.estimated_usd:.4f}"
            )
        self.records.append(usage)


def token_cost(
    spec: ModelSpec, input_tokens: int, output_tokens: int, *, normalized: bool
) -> Decimal:
    inp = (
        spec.normalized_input_per_million if normalized
        else spec.input_per_million
    )
    out = (
        spec.normalized_output_per_million if normalized
        else spec.output_per_million
    )
    return (
        Decimal(input_tokens) * inp + Decimal(output_tokens) * out
    ) / Decimal(1_000_000)


def rough_tokens(text: str) -> int:
    return max(1, (len(text) + 2) // 3)


def load_topics(path: Path) -> dict[str, str]:
    """Load the organizer's headerless qid<TAB>prompt development TSV."""
    topics: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = csv.reader(f, delimiter="\t")
        for lineno, row in enumerate(rows, 1):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(
                    f"{path}:{lineno}: expected 2 tab-separated fields, got {len(row)}"
                )
            qid, prompt = row
            if qid in topics:
                raise ValueError(f"{path}:{lineno}: duplicate qid {qid}")
            topics[qid] = prompt
    return topics


def load_references(path: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                refs[row["qid"]] = row
    return refs


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj


def validate_candidate(obj: dict[str, Any]) -> CandidateRubric:
    items = obj.get("rubrics")
    if not isinstance(items, list):
        raise ValueError("missing rubrics array")
    if not 12 <= len(items) <= 36:
        raise ValueError(f"unexpected criterion count {len(items)}")
    clean: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"rubrics[{i}] not object")
        criterion = str(item.get("criterion", "")).strip()
        axis = str(item.get("axis", "")).strip()
        polarity = str(item.get("polarity", "")).strip().lower()
        if len(criterion) < 12:
            raise ValueError(f"rubrics[{i}] criterion too short")
        if axis not in AXES:
            raise ValueError(f"rubrics[{i}] invalid axis {axis!r}")
        if polarity not in {"positive", "negative"}:
            raise ValueError(f"rubrics[{i}] invalid polarity {polarity!r}")
        norm = re.sub(r"\s+", " ", criterion.casefold())
        if norm in seen:
            raise ValueError(f"duplicate criterion at index {i}")
        seen.add(norm)
        clean.append(
            {"criterion": criterion, "axis": axis, "polarity": polarity}
        )
    return CandidateRubric(clean)


def validate_judgment(
    obj: dict[str, Any], labels: Sequence[str]
) -> dict[str, Any]:
    scores = obj.get("scores")
    ranking = obj.get("ranking")
    winner = obj.get("winner")
    if not isinstance(scores, dict) or set(scores) != set(labels):
        raise ValueError("judge scores must contain exactly all blind labels")
    if (
        not isinstance(ranking, list)
        or len(ranking) != len(labels)
        or set(ranking) != set(labels)
    ):
        raise ValueError("judge ranking must be a permutation of blind labels")
    if winner not in labels or ranking[0] != winner:
        raise ValueError("judge winner must equal ranking[0]")
    dims = (
        ("reference_coverage", 40),
        ("precision", 20),
        ("atomicity_judgeability", 15),
        ("implicit_synthesis", 15),
        ("axis_failure_modes", 10),
    )
    for label in labels:
        row = scores[label]
        total = 0.0
        for name, ceiling in dims:
            value = float(row[name])
            if value < 0 or value > ceiling:
                raise ValueError(f"{label}.{name} outside 0-{ceiling}")
            total += value
        if abs(float(row["total"]) - total) > 0.51:
            raise ValueError(f"{label}.total does not equal dimensions")
    return obj


def call_bedrock_converse(
    spec: ModelSpec, system: str, user: str
) -> tuple[str, int, int]:
    client = boto3.client("bedrock-runtime", region_name=spec.region)
    request: dict[str, Any] = {
        "modelId": spec.model_id,
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {
            "maxTokens": spec.max_output_tokens,
            "temperature": 0.0,
        },
    }
    if spec.tier != "default":
        request["serviceTier"] = {"type": spec.tier}
    if spec.reasoning_effort:
        request["additionalModelRequestFields"] = {
            "reasoning_effort": spec.reasoning_effort
        }
    response = client.converse(**request)
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    text = "".join(x.get("text", "") for x in blocks)
    usage = response.get("usage", {})
    if not text:
        raise RuntimeError(f"{spec.key} returned no text")
    return (
        text,
        int(usage.get("inputTokens", 0)),
        int(usage.get("outputTokens", 0)),
    )


def call_mantle_bearer(
    spec: ModelSpec, system: str, user: str
) -> tuple[str, int, int]:
    """Call Mantle Chat Completions with a short-lived regional bearer token."""
    url = (
        f"https://bedrock-mantle.{spec.region}.api.aws"
        "/openai/v1/chat/completions"
    )
    payload: dict[str, Any] = {
        "model": spec.model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": spec.max_output_tokens,
    }
    if spec.tier != "default":
        payload["service_tier"] = spec.tier
    if spec.reasoning_effort:
        payload["reasoning_effort"] = spec.reasoning_effort

    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    token = provide_token(region=spec.region)
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(
            f"Mantle HTTP {exc.code} for {spec.key}: {detail}"
        ) from exc

    try:
        text = data["choices"][0]["message"]["content"]
        usage = data["usage"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"unexpected Mantle response shape for {spec.key}"
        ) from exc
    if not text:
        raise RuntimeError(f"{spec.key} returned no text")
    return (
        str(text),
        int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
    )


def invoke(
    spec: ModelSpec, system: str, user: str, *, judge: bool = False
) -> tuple[str, int, int]:
    del judge
    if spec.provider == "bedrock_converse":
        return call_bedrock_converse(spec, system, user)
    if spec.provider == "mantle_bearer":
        return call_mantle_bearer(spec, system, user)
    raise ValueError(f"unknown provider {spec.provider}")


def generate_one(
    qid: str, prompt: str, spec: ModelSpec, budget: Budget
) -> CandidateRubric:
    user = GENERATION_PROMPT.format(axis_guide=AXIS_GUIDE, prompt=prompt)
    last_error: Exception | None = None
    for attempt in (1, 2):
        budget.reserve_check(
            spec,
            rough_tokens(GENERATION_SYSTEM + user),
            spec.max_output_tokens,
        )
        raw, input_tokens, output_tokens = invoke(
            spec, GENERATION_SYSTEM, user
        )
        budget.record(
            Usage(
                qid=qid,
                phase="generate",
                model_key=spec.key,
                model_id=spec.model_id,
                provider=spec.provider,
                tier=spec.tier,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_usd=token_cost(
                    spec, input_tokens, output_tokens, normalized=False
                ),
                normalized_usd=token_cost(
                    spec, input_tokens, output_tokens, normalized=True
                ),
                attempt=attempt,
            )
        )
        try:
            return validate_candidate(parse_json_object(raw))
        except Exception as exc:
            last_error = exc
            if attempt == 1:
                user += (
                    "\n\nYour previous reply did not satisfy the JSON schema "
                    f"({type(exc).__name__}: {exc}). Return a corrected complete "
                    "JSON object only."
                )
    raise RuntimeError(
        f"{spec.key} failed to produce valid rubric for {qid}: {last_error}"
    )


def human_reference_text(row: dict[str, Any]) -> str:
    lines = []
    for item in row["rubrics"]:
        weight = item.get("weight", 0)
        polarity = (
            "positive" if weight > 0
            else "negative" if weight < 0
            else "neutral"
        )
        lines.append(
            f"- [{item.get('axis', '')}] [weight={weight}; {polarity}] "
            f"{item.get('criterion', '')}"
        )
    return "\n".join(lines)


def candidate_text(rubric: CandidateRubric) -> str:
    return "\n".join(
        f"- [{x['axis']}] [{x['polarity']}] {x['criterion']}"
        for x in rubric.rubrics
    )


def blind_mapping(
    qid: str, model_keys: Sequence[str]
) -> dict[str, str]:
    labels = [chr(ord("A") + i) for i in range(len(model_keys))]
    digest = hashlib.sha256(
        f"rubric-bakeoff-v1:{qid}".encode()
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    shuffled = list(model_keys)
    rng.shuffle(shuffled)
    return dict(zip(labels, shuffled))


def build_blind_judge_user(
    prompt: str,
    reference: dict[str, Any],
    candidates: dict[str, CandidateRubric],
    mapping: dict[str, str],
) -> str:
    labels = list(mapping)
    blinded = "\n\n".join(
        f"=== {label} ===\n{candidate_text(candidates[mapping[label]])}"
        for label in labels
    )
    return JUDGE_PROMPT.format(
        prompt=prompt,
        reference=human_reference_text(reference),
        candidates=blinded,
    )


def judge_one(
    qid: str,
    prompt: str,
    reference: dict[str, Any],
    candidates: dict[str, CandidateRubric],
    mapping: dict[str, str],
    budget: Budget,
) -> dict[str, Any]:
    labels = list(mapping)
    user = build_blind_judge_user(prompt, reference, candidates, mapping)
    last_error: Exception | None = None
    for attempt in (1, 2):
        budget.reserve_check(
            JUDGE,
            rough_tokens(JUDGE_SYSTEM + user),
            JUDGE.max_output_tokens,
        )
        raw, input_tokens, output_tokens = invoke(
            JUDGE, JUDGE_SYSTEM, user, judge=True
        )
        budget.record(
            Usage(
                qid=qid,
                phase="judge",
                model_key=JUDGE.key,
                model_id=JUDGE.model_id,
                provider=JUDGE.provider,
                tier=JUDGE.tier,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_usd=token_cost(
                    JUDGE, input_tokens, output_tokens, normalized=False
                ),
                normalized_usd=token_cost(
                    JUDGE, input_tokens, output_tokens, normalized=True
                ),
                attempt=attempt,
            )
        )
        try:
            return validate_judgment(
                parse_json_object(raw), labels
            )
        except Exception as exc:
            last_error = exc
            if attempt == 1:
                user += (
                    "\n\nYour previous reply failed JSON validation "
                    f"({type(exc).__name__}: {exc}). Return one corrected "
                    "complete JSON object only."
                )
    raise RuntimeError(f"judge failed for {qid}: {last_error}")


def usage_to_json(x: Usage) -> dict[str, Any]:
    row = asdict(x)
    row["estimated_usd"] = str(x.estimated_usd)
    row["normalized_usd"] = str(x.normalized_usd)
    return row


def usage_from_json(row: dict[str, Any]) -> Usage:
    data = dict(row)
    data["estimated_usd"] = Decimal(str(data["estimated_usd"]))
    data["normalized_usd"] = Decimal(str(data["normalized_usd"]))
    return Usage(**data)


def conservative_preflight(
    qids: Sequence[str], topics: dict[str, str]
) -> Decimal:
    total = Decimal("0")
    for qid in qids:
        gen_user = GENERATION_PROMPT.format(
            axis_guide=AXIS_GUIDE, prompt=topics[qid]
        )
        gen_input = rough_tokens(GENERATION_SYSTEM + gen_user)
        # Reserve two full attempts for every candidate.
        for spec in CANDIDATES:
            total += token_cost(
                spec, gen_input * 2, spec.max_output_tokens * 2,
                normalized=False,
            )
        # Very conservative judge allowance: two 45k-input/8k-output calls.
        total += token_cost(
            JUDGE, 90_000, JUDGE.max_output_tokens * 2,
            normalized=False,
        )
    return total


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def aggregate(
    qids: Sequence[str],
    mappings: dict[str, dict[str, str]],
    judgments: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, CandidateRubric]],
    budget: Budget,
) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for spec in CANDIDATES:
        totals, ranks, counts = [], [], []
        wins = 0
        for qid in qids:
            inverse = {
                model: label for label, model in mappings[qid].items()
            }
            label = inverse[spec.key]
            judgment = judgments[qid]
            totals.append(float(judgment["scores"][label]["total"]))
            ranks.append(judgment["ranking"].index(label) + 1)
            wins += int(judgment["winner"] == label)
            counts.append(len(candidates[qid][spec.key].rubrics))
        records = [
            u for u in budget.records
            if u.phase == "generate" and u.model_key == spec.key
        ]
        actual = sum(
            (u.estimated_usd for u in records), Decimal("0")
        )
        normalized = sum(
            (u.normalized_usd for u in records), Decimal("0")
        )
        by_model[spec.key] = {
            "display_name": spec.display_name,
            "mean_judge_score": round(statistics.mean(totals), 3),
            "median_judge_score": round(statistics.median(totals), 3),
            "mean_rank": round(statistics.mean(ranks), 3),
            "wins": wins,
            "topics": len(totals),
            "mean_criteria": round(statistics.mean(counts), 2),
            "generation_actual_usd": str(actual),
            "generation_normalized_usd": str(normalized),
        }

    quality_order = sorted(
        by_model,
        key=lambda k: (
            -by_model[k]["mean_judge_score"],
            by_model[k]["mean_rank"],
            Decimal(by_model[k]["generation_normalized_usd"]),
        ),
    )
    top_score = by_model[quality_order[0]]["mean_judge_score"]
    competitive = [
        k for k in quality_order
        if by_model[k]["mean_judge_score"] >= top_score - 3.0
    ]
    value_winner = min(
        competitive,
        key=lambda k: Decimal(
            by_model[k]["generation_normalized_usd"]
        ),
    )
    judge_cost = sum(
        (
            u.estimated_usd for u in budget.records
            if u.phase == "judge"
        ),
        Decimal("0"),
    )
    return {
        "quality_order": quality_order,
        "blind_quality_winner": quality_order[0],
        "competitive_within_3_points": competitive,
        "cost_effective_winner": value_winner,
        "models": by_model,
        "judge_actual_usd": str(judge_cost),
        "total_actual_usd": str(budget.spent),
        "hard_cap_usd": str(budget.hard_cap),
        "selection_rule": (
            "Rank by blind GPT-6.1 Sol mean quality; models within 3 "
            "points of the top are treated as competitive, then choose "
            "the lowest normalized production generation cost."
        ),
    }


def run(args: argparse.Namespace) -> int:
    topics = load_topics(args.topics)
    refs = load_references(args.references)
    qids = args.qid or list(DEFAULT_QIDS[: args.limit])
    missing = [
        q for q in qids if q not in topics or q not in refs
    ]
    if missing:
        raise ValueError(f"missing development qids: {missing}")

    ceiling = conservative_preflight(qids, topics)
    print(
        f"[bakeoff] topics={len(qids)} candidates={len(CANDIDATES)}"
    )
    print(
        f"[bakeoff] hard cap=${args.budget_usd}; "
        f"two-attempt ceiling=${ceiling:.4f}"
    )
    if ceiling + args.prior_spend_usd > args.budget_usd:
        raise BudgetExceeded(
            f"preflight ${ceiling:.4f} + prior-spend reserve "
            f"${args.prior_spend_usd:.4f} exceeds "
            f"${args.budget_usd} cap; reduce --limit"
        )
    for qid in qids:
        meta = refs[qid]
        print(
            f"  {qid} | {meta.get('domain')} | "
            f"{meta.get('conceptual_breadth')}/"
            f"{meta.get('logical_nesting')}/"
            f"{meta.get('exploration')}"
        )
    if args.dry_run:
        return 0

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    usage_path = out_dir / "usage.json"
    budget = Budget(args.budget_usd, args.prior_spend_usd)
    if usage_path.exists():
        existing_usage = json.loads(usage_path.read_text(encoding="utf-8"))
        budget.records = [usage_from_json(row) for row in existing_usage]
        print(
            f"[bakeoff] resume: {len(budget.records)} usage record(s), "
            f"tracked=${budget.tracked_spend:.4f}, "
            f"prior reserve=${budget.prior_spend:.4f}"
        )
    all_candidates: dict[
        str, dict[str, CandidateRubric]
    ] = {}
    mappings: dict[str, dict[str, str]] = {}
    judgments: dict[str, dict[str, Any]] = {}

    for qid in qids:
        print(f"[bakeoff] generate qid={qid}")
        per_model: dict[str, CandidateRubric] = {}
        for spec in CANDIDATES:
            candidate_path = (
                out_dir / "candidates" / qid / f"{spec.key}.json"
            )
            if candidate_path.exists():
                rubric = validate_candidate(
                    json.loads(candidate_path.read_text(encoding="utf-8"))
                )
                per_model[spec.key] = rubric
                print(
                    f"  {spec.key}: resume criteria={len(rubric.rubrics)} "
                    f"spent=${budget.spent:.4f}"
                )
                continue

            started = time.time()
            rubric = generate_one(
                qid, topics[qid], spec, budget
            )
            per_model[spec.key] = rubric
            save_json(candidate_path, asdict(rubric))
            save_json(
                usage_path,
                [usage_to_json(x) for x in budget.records],
            )
            print(
                f"  {spec.key}: criteria={len(rubric.rubrics)} "
                f"spent=${budget.spent:.4f} "
                f"elapsed={time.time() - started:.1f}s"
            )
        all_candidates[qid] = per_model

        mapping = blind_mapping(
            qid, [x.key for x in CANDIDATES]
        )
        mappings[qid] = mapping
        save_json(
            out_dir / "blind-cases" / f"{qid}.json",
            {
                "qid": qid,
                "prompt": topics[qid],
                "reference": refs[qid],
                "candidates": {
                    label: asdict(per_model[model_key])
                    for label, model_key in mapping.items()
                },
            },
        )

        judgment_path = out_dir / "judgments" / f"{qid}.json"
        if judgment_path.exists():
            judgment = validate_judgment(
                json.loads(judgment_path.read_text(encoding="utf-8")),
                list(mapping),
            )
            print(
                f"[bakeoff] judge qid={qid}: resume "
                f"winner={judgment['winner']}"
            )
        else:
            print(
                f"[bakeoff] judge qid={qid} with GPT-6.1 Sol high"
            )
            judgment = judge_one(
                qid, topics[qid], refs[qid],
                per_model, mapping, budget,
            )
            save_json(
                usage_path,
                [usage_to_json(x) for x in budget.records],
            )
            save_json(judgment_path, judgment)
            print(
                f"  winner={judgment['winner']} "
                f"confidence={judgment.get('confidence')} "
                f"spent=${budget.spent:.4f}"
            )
        judgments[qid] = judgment

    save_json(
        out_dir / "hidden-model-mapping.json", mappings
    )
    save_json(
        out_dir / "candidate-models.json",
        {
            x.key: {
                "display_name": x.display_name,
                "model_id": x.model_id,
                "provider": x.provider,
                "region": x.region,
                "tier": x.tier,
                "input_per_million": str(
                    x.input_per_million
                ),
                "output_per_million": str(
                    x.output_per_million
                ),
                "normalized_input_per_million": str(
                    x.normalized_input_per_million
                ),
                "normalized_output_per_million": str(
                    x.normalized_output_per_million
                ),
            }
            for x in CANDIDATES
        },
    )
    summary = aggregate(
        qids, mappings, judgments,
        all_candidates, budget,
    )
    save_json(out_dir / "summary.json", summary)

    print("[bakeoff] final quality order:")
    for idx, key in enumerate(
        summary["quality_order"], 1
    ):
        row = summary["models"][key]
        print(
            f"  {idx}. {row['display_name']}: "
            f"score={row['mean_judge_score']:.3f}, "
            f"mean-rank={row['mean_rank']:.3f}, "
            f"wins={row['wins']}, "
            f"normalized-generation="
            f"${Decimal(row['generation_normalized_usd']):.4f}"
        )
    quality = summary["models"][
        summary["blind_quality_winner"]
    ]["display_name"]
    value = summary["models"][
        summary["cost_effective_winner"]
    ]["display_name"]
    print(f"[bakeoff] blind quality winner={quality}")
    print(f"[bakeoff] cost-effective winner={value}")
    print(
        f"[bakeoff] total actual spend="
        f"${budget.spent:.4f} / ${budget.hard_cap}"
    )
    print(f"[bakeoff] results={out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topics", type=Path, default=DEFAULT_TOPICS
    )
    parser.add_argument(
        "--references", type=Path, default=DEFAULT_REFERENCES
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "evaluation/rubric-model-bakeoff",
    )
    parser.add_argument(
        "--budget-usd", type=Decimal,
        default=Decimal("20.00"),
    )
    parser.add_argument(
        "--prior-spend-usd", type=Decimal,
        default=Decimal("0"),
        help="reserve for untracked calls from an interrupted run",
    )
    parser.add_argument(
        "--limit", type=int, default=6,
        choices=range(1, len(DEFAULT_QIDS) + 1),
    )
    parser.add_argument(
        "--qid", action="append", default=[],
        help="explicit public dev qid; repeatable",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (
        BudgetExceeded, BotoCoreError, ClientError,
        RuntimeError, ValueError,
    ) as exc:
        print(
            f"[bakeoff] ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
