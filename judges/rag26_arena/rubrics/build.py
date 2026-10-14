"""Offline rubric-construction driver: Stages A-D (existing) + E/F/H/I
(finalize.py) for a list of topics, resumable per topic.

    python -m judges.rag26_arena.rubrics.build \\
        --topics local-data/rag26/topics/trec_rag_2026_queries.jsonl \\
        --out judges/rag26_arena/rubrics/frozen/rag26-rubrics.jsonl \\
        --work-dir rubric-work --limit 3

Response-independent: reads only the public topics file and the ClimbMix
services. Never touches candidate responses. Each finished topic is written to
`<work-dir>/<qid>.json` immediately, so an interrupted run resumes without
re-paying; `--out` is (re)assembled from those files in topics-file order.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from decimal import Decimal
from pathlib import Path

from ..hybrid_retrieval import HybridConfig, HybridResearchRetriever
from ..search_index import dense_client_from_env, sparse_client_from_env
from .cache import RubricLlmCache
from .evidence import condense_evidence, plan_and_research
from .finalize import finalize_topic
from .generate import generate_axis_candidates
from .io import dump_jsonl
from .llm import CostTracker
from .schema import RubricTopic
from .topics import load_topics
from .validate import validate_topic

MODEL_ID = "openai.gpt-6-luna"
REGION = "us-east-1"


def build_topic(topic, retriever, *, cache, tracker, model_id, region, tier, fallback_tier) -> RubricTopic:
    common = dict(model_id=model_id, region=region, tier=tier, fallback_tier=fallback_tier,
                  cache=cache, cost_tracker=tracker)
    characterization, pack = plan_and_research(topic, retriever, **common)
    notes = condense_evidence(topic, characterization, pack, min_notes=12, **common)
    candidates = generate_axis_candidates(topic, characterization, notes, **common)
    return finalize_topic(topic.narrative, characterization, candidates, **common)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("rubric-work"))
    parser.add_argument("--limit", type=int, default=None, help="first N topics only (rehearsal)")
    parser.add_argument("--budget-usd", type=Decimal, default=Decimal("25"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--tier", default="default")
    parser.add_argument("--fallback-tier", default=None)
    args = parser.parse_args(argv)

    topics = load_topics(args.topics)[: args.limit]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cache = RubricLlmCache(args.work_dir / "rubric_llm_cache.db")
    tracker = CostTracker(hard_cap_usd=args.budget_usd)
    retriever = HybridResearchRetriever(
        dense_client_from_env(), sparse_client_from_env(), HybridConfig(),
    )

    failed: list[str] = []
    for topic in topics:
        path = args.work_dir / f"{topic.qid}.json"
        if path.exists():
            continue
        try:
            rubric = build_topic(topic, retriever, cache=cache, tracker=tracker, model_id=args.model_id,
                                 region=args.region, tier=args.tier, fallback_tier=args.fallback_tier)
            issues = validate_topic(rubric)
            if issues:
                raise ValueError(f"validation: {issues[:3]}")
        except Exception as error:  # one bad topic must not sink the batch
            failed.append(topic.qid)
            print(f"[rubric-build] qid={topic.qid} FAILED: {type(error).__name__}: {str(error)[:200]}", file=sys.stderr)
            continue
        path.write_text(json.dumps(rubric.model_dump(), ensure_ascii=False, sort_keys=True), encoding="utf-8")
        print(f"[rubric-build] qid={topic.qid} ok criteria={len(rubric.rubrics)} spend=${tracker.total_usd():.3f}")
        tracker.check_budget()

    done = [RubricTopic(**json.loads(p.read_text(encoding="utf-8")))
            for t in topics if (p := args.work_dir / f"{t.qid}.json").exists()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(done, args.out)
    print(f"[rubric-build] wrote {len(done)}/{len(topics)} topics to {args.out}; failed={failed}")
    tracker.print_summary()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
