#!/usr/bin/env python3
"""Orchestrate one 2025 pilot evaluation experiment (docs/evaluation-2025-plan.md Phase P).

Reuses run_all_datasets.py's subprocess plumbing (`auto-judge run`, `auto-judge-evaluate
meta-evaluate`) instead of reimplementing it. This script only adds: split/fold -> --topic
selection from evaluation/splits/<dataset>.yml, a never-collide experiment directory, and a
manifest.yml recording the git commit + exact config for every run.

    uv run python scripts/run_eval_2025.py --judge judges/naive/workflow.yml \\
        --dataset rag25-generation --split fold-0 --dry-run

Drop --dry-run to actually invoke `auto-judge run` (an LLM-calling judge will spend against
whatever OPENAI_* endpoint is configured -- check that budget first, see scripts/cost_guard.py).
Add --meta-evaluate to also correlate against the dataset's truth leaderboard; this is skipped
cleanly (not an error) for any 2025 dataset until the official truth bundle is fetched, see
evaluation/README.md.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from run_all_datasets import load_datasets, run_workflow  # noqa: E402  (reuse, don't reimplement)

DATASETS_2025 = REPO / "evaluation" / "datasets-2025.yml"
SPLITS_DIR = REPO / "evaluation" / "splits"
RESULTS_DIR = REPO / "evaluation" / "results"


def split_topics(dataset_name: str, split: str) -> list[str] | None:
    """--split -> topic-id list via evaluation/splits/<dataset>.yml. 'all' -> None (no filter,
    i.e. every topic in the dataset's topics file, not just the assessed subset)."""
    if split == "all":
        return None
    path = SPLITS_DIR / f"{dataset_name}.yml"
    if not path.is_file():
        raise SystemExit(f"no split file for {dataset_name}: {path} "
                          f"(run: uv run python scripts/prepare_eval_2025.py --make-splits)")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if split == "assessed":
        return doc["assessed_topics"]
    folds = (doc.get("topic_folds") or {}).get("folds", {})
    if split not in folds:
        known = ["assessed", *sorted(folds)]
        raise SystemExit(f"{dataset_name}: unknown --split '{split}' (known: {known})")
    return folds[split]


def split_run_holdout(dataset_name: str, which: str) -> list[str] | None:
    """--runs known|held-out -> run-ID list via the split file's `run_holdout` (docs/
    evaluation-2025-plan.md Phase D4). 'all' -> None (no filter)."""
    if which == "all":
        return None
    path = SPLITS_DIR / f"{dataset_name}.yml"
    if not path.is_file():
        raise SystemExit(f"no split file for {dataset_name}: {path} "
                          f"(run: uv run python scripts/prepare_eval_2025.py --make-splits)")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    key = {"known": "known", "held-out": "held_out"}[which]
    return doc["run_holdout"][key]


def unique_dir(base: Path) -> Path:
    """base, or base-1/base-2/... if it already exists -- an experiment directory is never
    reused, so a later run can't silently clobber an earlier one's results."""
    if not base.exists():
        return base
    n = 1
    while (candidate := base.parent / f"{base.name}-{n}").exists():
        n += 1
    return candidate


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.strip()


def write_manifest(out_dir: Path, **fields: Any) -> None:
    (out_dir / "manifest.yml").write_text(yaml.safe_dump(fields, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", required=True, help="path to a judge's workflow.yml")
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME",
                         help="dataset name from evaluation/datasets-2025.yml (repeatable)")
    parser.add_argument("--split", default="all", help="fold-N | assessed | all (default: all)")
    parser.add_argument("--runs", default="all", choices=["all", "known", "held-out"],
                         help="anonymous run-ID holdout quadrant (docs/evaluation-2025-plan.md "
                              "Phase D4); 'known'/'held-out' need evaluation/splits/<dataset>.yml")
    parser.add_argument("--variant", default=None, help="workflow variant")
    parser.add_argument("--meta-evaluate", action="store_true",
                         help="also run auto-judge-evaluate meta-evaluate (skipped cleanly if truth is missing)")
    parser.add_argument("--dry-run", action="store_true",
                         help="print the exact commands; write nothing, run nothing")
    args = parser.parse_args()

    workflow = Path(args.judge)
    if not workflow.is_file():
        print(f"Error: judge workflow not found: {workflow}", file=sys.stderr)
        return 1

    all_datasets = {d.name: d for d in load_datasets(DATASETS_2025)}
    unknown = [n for n in args.dataset if n not in all_datasets]
    if unknown:
        print(f"Error: unknown dataset(s): {unknown} (known: {sorted(all_datasets)})", file=sys.stderr)
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in args.dataset:
        dataset = all_datasets[name]
        try:
            topics = split_topics(name, args.split)
            runs = split_run_holdout(name, args.runs)
        except SystemExit as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        extra_args = [a for t in (topics or []) for a in ("--topic", str(t))]
        extra_args += [a for r in (runs or []) for a in ("--run", str(r))]

        stamp = f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
        leaf = f"{name}-{args.split}" + (f"-runs_{args.runs}" if args.runs != "all" else "")
        out_dir = unique_dir(RESULTS_DIR / f"{stamp}-{workflow.parent.name}-{leaf}")
        correlations_path = out_dir / "correlations.jsonl" if args.meta_evaluate else None

        preview = (f"auto-judge run --workflow {workflow} --rag-responses {dataset.responses} "
                   f"--rag-topics {dataset.topics} --out-dir {out_dir}")
        if args.variant:
            preview += f" --variant {args.variant}"
        preview += "".join(f" --topic {t}" for t in (topics or []))
        preview += "".join(f" --run {r}" for r in (runs or []))
        print(f"\n=== {name} (split={args.split}, runs={args.runs}, judge={workflow.parent.name}) ===")
        print(f"  {preview}")
        if args.meta_evaluate:
            if dataset.truth:
                print(f"  # then: auto-judge-evaluate meta-evaluate --truth-leaderboard "
                      f"{dataset.truth} --output {correlations_path} ... {out_dir}/*.eval.txt")
            else:
                print("  # (skip meta-evaluation: no 'truth' set for this dataset)")

        if args.dry_run:
            continue

        out_dir.mkdir(parents=True)
        started = datetime.datetime.now(datetime.timezone.utc)
        common_fields = dict(
            experiment_id=out_dir.name, commit_sha=git_sha(), judge=str(workflow), dataset=name,
            split=args.split, runs=args.runs, variant=args.variant, topics=topics,
            run_ids=runs, meta_evaluate=args.meta_evaluate,
        )
        write_manifest(out_dir, **common_fields, started_at=started.isoformat())
        # topics_filter/runs_filter below are used by run_workflow only for directory-naming and
        # its own native "prio1"/"assessed" shortcuts (which add --run/--topic from
        # dataset.prio1_runs/assessed_topics themselves); prefixed so they never collide with
        # those two magic strings -- our own --topic/--run selection is already in extra_args.
        success = run_workflow(workflow, dataset, out_dir, f"split-{args.runs}", f"split-{args.split}",
                                extra_args, variant=args.variant, meta_evaluate=args.meta_evaluate,
                                meta_evaluate_output=correlations_path)
        finished = datetime.datetime.now(datetime.timezone.utc)
        write_manifest(
            out_dir, **common_fields, started_at=started.isoformat(), finished_at=finished.isoformat(),
            wall_time_seconds=(finished - started).total_seconds(), success=success,
        )
        if not success:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
