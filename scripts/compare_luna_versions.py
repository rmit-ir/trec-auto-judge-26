#!/usr/bin/env python3
"""Blind GPT-5.6 Luna vs GPT-6 Luna rubric-construction comparison.

Uses only public ResearchRubrics development topics/reference rubrics.
GPT-5.6 Luna is called through the configured Azure OpenAI-compatible endpoint.
GPT-6 Luna candidate rubrics are reused from the completed model bake-off.
GPT-6.1 Sol (high) judges A/B candidates without model identities.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import urllib.error
import urllib.request
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from scripts import benchmark_rubric_models as b


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/luna56-vs-luna6"
SOURCE = ROOT / "evaluation/rubric-model-bakeoff"
QIDS = b.DEFAULT_QIDS[:3]

LUNA56_INPUT = Decimal("0.20")
LUNA56_OUTPUT = Decimal("1.20")
LUNA6_INPUT = Decimal("0.10")
LUNA6_OUTPUT = Decimal("0.50")


def cost(inp: int, out: int, pin: Decimal, pout: Decimal) -> Decimal:
    return (Decimal(inp) * pin + Decimal(out) * pout) / Decimal(1_000_000)


def azure_luna56(system: str, user: str) -> tuple[str, int, int]:
    base = os.environ["OPENAI_BASE_URL"].rstrip("/")
    key = os.environ["OPENAI_API_KEY"]
    payload = {
        "model": "gpt-5.6-luna",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": 7000,
        "reasoning_effort": "high",
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Azure HTTP {exc.code}: {detail}") from exc
    text = data["choices"][0]["message"]["content"]
    usage = data["usage"]
    return str(text), int(usage["prompt_tokens"]), int(usage["completion_tokens"])


def mapping(qid: str) -> dict[str, str]:
    models = ["luna56", "luna6"]
    seed = int.from_bytes(hashlib.sha256(("luna-head2head:" + qid).encode()).digest()[:8], "big")
    random.Random(seed).shuffle(models)
    return {"A": models[0], "B": models[1]}


def judge_prompt(
    prompt: str,
    reference: dict[str, Any],
    candidates: dict[str, b.CandidateRubric],
    blind: dict[str, str],
) -> str:
    blocks = "\n\n".join(
        f"=== {label} ===\n{b.candidate_text(candidates[key])}"
        for label, key in blind.items()
    )
    return f"""Compare two candidate ResearchRubrics for the request below.

The organizer's human development rubric is a semantic reference. Candidate
rubrics are blindly labelled A and B. Do not infer or speculate about their
source. Judge only the rubric content; do not reward wording overlap or length.

Score each candidate:
- reference_coverage (0-40): coverage of important human-reference obligations,
  weighted toward high-|weight| criteria.
- precision (0-20): avoids arbitrary requirements, factual overreach, and
  unjustified scope expansion.
- atomicity_judgeability (0-15): observable, specific, mostly atomic,
  non-duplicative criteria.
- implicit_synthesis (0-15): captures implicit expectations and genuine
  synthesis requirements.
- axis_failure_modes (0-10): sensible axes and useful, non-mirror failure modes.

Total must equal the five dimensions. Choose a winner; tie only if genuinely
indistinguishable.

REQUEST:
{prompt}

HUMAN REFERENCE RUBRIC:
{b.human_reference_text(reference)}

BLIND CANDIDATES:
{blocks}

Return ONLY JSON:
{{
  "scores": {{
    "A": {{
      "reference_coverage": 0,
      "precision": 0,
      "atomicity_judgeability": 0,
      "implicit_synthesis": 0,
      "axis_failure_modes": 0,
      "total": 0,
      "notes": "brief assessment"
    }},
    "B": {{
      "reference_coverage": 0,
      "precision": 0,
      "atomicity_judgeability": 0,
      "implicit_synthesis": 0,
      "axis_failure_modes": 0,
      "total": 0,
      "notes": "brief assessment"
    }}
  }},
  "ranking": ["A", "B"],
  "winner": "A",
  "confidence": 0.0,
  "decision_notes": "brief explanation"
}}"""


def validate_pair(obj: dict[str, Any]) -> dict[str, Any]:
    return b.validate_judgment(obj, ["A", "B"])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidates").mkdir(exist_ok=True)
    (OUT / "judgments").mkdir(exist_ok=True)

    topics = b.load_topics(b.DEFAULT_TOPICS)
    refs = b.load_references(b.DEFAULT_REFERENCES)
    old_usage_path = SOURCE / "usage.json"
    old_usage = json.loads(old_usage_path.read_text(encoding="utf-8"))
    luna6_usage = {
        row["qid"]: row
        for row in old_usage
        if row["phase"] == "generate" and row["model_key"] == "luna"
        and row["attempt"] == 1
    }

    rows: list[dict[str, Any]] = []
    total56 = Decimal("0")
    total6 = Decimal("0")
    judge_cost = Decimal("0")

    for qid in QIDS:
        prompt = topics[qid]
        gen_user = b.GENERATION_PROMPT.format(axis_guide=b.AXIS_GUIDE, prompt=prompt)

        c56_path = OUT / "candidates" / f"{qid}.luna56.json"
        if c56_path.exists():
            c56_obj = json.loads(c56_path.read_text(encoding="utf-8"))
            c56 = b.validate_candidate(c56_obj["rubric"])
            in56 = int(c56_obj["usage"]["input_tokens"])
            out56 = int(c56_obj["usage"]["output_tokens"])
        else:
            raw, in56, out56 = azure_luna56(b.GENERATION_SYSTEM, gen_user)
            c56 = b.validate_candidate(b.parse_json_object(raw))
            c56_obj = {
                "rubric": asdict(c56),
                "usage": {"input_tokens": in56, "output_tokens": out56},
            }
            b.save_json(c56_path, c56_obj)

        c6_path = SOURCE / "candidates" / qid / "luna.json"
        c6 = b.validate_candidate(json.loads(c6_path.read_text(encoding="utf-8")))
        u6 = luna6_usage[qid]
        in6, out6 = int(u6["input_tokens"]), int(u6["output_tokens"])

        price56 = cost(in56, out56, LUNA56_INPUT, LUNA56_OUTPUT)
        price6 = cost(in6, out6, LUNA6_INPUT, LUNA6_OUTPUT)
        total56 += price56
        total6 += price6

        candidates = {"luna56": c56, "luna6": c6}
        blind = mapping(qid)
        jpath = OUT / "judgments" / f"{qid}.json"
        if jpath.exists():
            judgment = validate_pair(json.loads(jpath.read_text(encoding="utf-8")))
            ju = json.loads((OUT / "judgments" / f"{qid}.usage.json").read_text(encoding="utf-8"))
            jin, jout = int(ju["input_tokens"]), int(ju["output_tokens"])
        else:
            user = judge_prompt(prompt, refs[qid], candidates, blind)
            rawj, jin, jout = b.invoke(b.JUDGE, b.JUDGE_SYSTEM, user, judge=True)
            judgment = validate_pair(b.parse_json_object(rawj))
            b.save_json(jpath, judgment)
            b.save_json(
                OUT / "judgments" / f"{qid}.usage.json",
                {"input_tokens": jin, "output_tokens": jout},
            )
        judge_cost += b.token_cost(b.JUDGE, jin, jout, normalized=False)

        inverse = {model: label for label, model in blind.items()}
        l56 = inverse["luna56"]
        l6 = inverse["luna6"]
        rows.append({
            "qid": qid,
            "blind_mapping": blind,
            "luna56_score": float(judgment["scores"][l56]["total"]),
            "luna6_score": float(judgment["scores"][l6]["total"]),
            "winner": blind[judgment["winner"]],
            "confidence": judgment.get("confidence"),
            "luna56_input_tokens": in56,
            "luna56_output_tokens": out56,
            "luna6_input_tokens": in6,
            "luna6_output_tokens": out6,
            "luna56_cost_usd": str(price56),
            "luna6_cost_usd": str(price6),
        })

    avg56 = sum(r["luna56_score"] for r in rows) / len(rows)
    avg6 = sum(r["luna6_score"] for r in rows) / len(rows)
    wins56 = sum(r["winner"] == "luna56" for r in rows)
    wins6 = sum(r["winner"] == "luna6" for r in rows)
    summary = {
        "topics": len(rows),
        "mean_score_luna56": round(avg56, 3),
        "mean_score_luna6": round(avg6, 3),
        "wins_luna56": wins56,
        "wins_luna6": wins6,
        "generation_cost_luna56_usd": str(total56),
        "generation_cost_luna6_usd": str(total6),
        "judge_cost_usd": str(judge_cost),
        "rows": rows,
    }
    b.save_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
