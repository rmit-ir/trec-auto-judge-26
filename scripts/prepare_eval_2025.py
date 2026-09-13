#!/usr/bin/env python3
"""Validate and report on the 2025 pilot evaluation datasets (docs/evaluation-2025-plan.md
Phase C). No LLM calls. Never opens response/report content -- only directory listings
(run_id filenames), topic files (fully readable per the data policy), and truth leaderboards
(readable at any topic).

    uv run python scripts/prepare_eval_2025.py                # validate + print dataset card
    uv run python scripts/prepare_eval_2025.py --dataset NAME  # just one dataset
    uv run python scripts/prepare_eval_2025.py --make-splits   # (re)write evaluation/splits/*.yml

Exit code: 1 on a structural problem (bad manifest, missing responses/topics dir, truth that
fails to parse, duplicate truth cells). Missing truth is reported but is not a structural
problem -- it is the expected, documented blocker until the official bundle is published.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from run_all_datasets import _resolve_from_release  # noqa: E402  (reuse, don't reimplement)

MANIFEST = REPO / "evaluation" / "datasets-2025.yml"
SPLITS_DIR = REPO / "evaluation" / "splits"
SPLIT_SEED = 20250912  # fixed; see evaluation/README.md "Splits"


def load_manifest(path: Path = MANIFEST) -> list[dict[str, Any]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cfg.get("datasets", [])


def topic_ids_of(topics_path: Path) -> list[str]:
    """request_id of every topic, in file order (topics are unrestricted at any index)."""
    ids = []
    with open(topics_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)["request_id"]))
    return ids


def run_ids_of(responses_dir: Path) -> list[str]:
    """run_id of every run = filename, sorted (run_id is readable at any topic)."""
    return sorted(p.name for p in responses_dir.iterdir() if not p.name.startswith("."))


class Problem(Exception):
    """A structural problem (non-zero exit), as opposed to the expected missing-truth blocker."""


def check_dataset(entry: dict[str, Any]) -> dict[str, Any]:
    name = entry["name"]
    resolved = _resolve_from_release(entry["from_release"], name)
    if resolved is None:
        raise Problem(f"{name}: track '{entry['from_release']['track']}' not fetched -- "
                       f"run ./fetch_datasets.py --pilot")
    responses, topics, prio1_runs, assessed_topics, _released_truth = resolved
    responses_dir, topics_path = Path(responses), Path(topics)

    if not responses_dir.is_dir():
        raise Problem(f"{name}: responses dir missing: {responses_dir}")
    if not topics_path.is_file():
        raise Problem(f"{name}: topics file missing: {topics_path}")

    runs = run_ids_of(responses_dir)
    topics_all = topic_ids_of(topics_path)

    missing_prio1 = sorted(set(prio1_runs) - set(runs))
    missing_assessed = sorted(set(assessed_topics) - set(topics_all))

    truth_path = REPO / entry["truth"]
    truth_status = "missing"
    truth_measures: set[str] = set()
    if truth_path.is_file():
        truth_status = "present"
        truth_measures = check_truth(name, truth_path, entry["truth_format"], runs, topics_all)

    return {
        "name": name,
        "runs": len(runs),
        "topics": len(topics_all),
        "assessed": len(assessed_topics),
        "missing_prio1_runs": missing_prio1,
        "missing_assessed_topics": missing_assessed,
        "truth_status": truth_status,
        "truth_path": str(truth_path),
        "truth_measures": sorted(truth_measures),
        "role": entry["role"],
        "prio1_runs": prio1_runs,
        "assessed_topics": assessed_topics,
        "all_runs": runs,
        "all_topics": topics_all,
    }


def check_truth(name: str, truth_path: Path, truth_format: str, runs: list[str],
                 topics_all: list[str]) -> set[str]:
    """Parse truth via autojudge_evaluate (no parallel parser) and check overlap + duplicates."""
    from autojudge_evaluate.eval_results import load as load_eval_result

    try:
        result = load_eval_result(
            truth_path, truth_format, has_header=False, drop_aggregates=False,
            recompute_aggregates=False, verify=True, on_missing="warn",
        )
    except Exception as exc:  # noqa: BLE001 -- surface as a Problem, not a traceback
        raise Problem(f"{name}: truth at {truth_path} failed to parse as {truth_format}: {exc}") from exc

    seen: set[tuple[str, str, str]] = set()
    for e in result.entries:
        cell = (e.run_id, e.topic_id, e.measure)
        if cell in seen:
            raise Problem(f"{name}: duplicate truth cell (run={e.run_id}, topic={e.topic_id}, "
                           f"measure={e.measure}) in {truth_path}")
        seen.add(cell)

    unknown_runs = sorted(result.run_ids - set(runs))
    unknown_topics = sorted(result.topic_ids - set(topics_all))
    if unknown_runs:
        print(f"  warn: {name}: truth has {len(unknown_runs)} run_id(s) absent from responses: "
              f"{unknown_runs[:5]}{'...' if len(unknown_runs) > 5 else ''}", file=sys.stderr)
    if unknown_topics:
        print(f"  warn: {name}: truth has {len(unknown_topics)} topic_id(s) absent from topics file: "
              f"{unknown_topics[:5]}{'...' if len(unknown_topics) > 5 else ''}", file=sys.stderr)
    return result.measures


def print_card(reports: list[dict[str, Any]]) -> None:
    header = f"{'Dataset':<20} {'Runs':>5} {'Topics':>7} {'Assessed':>9} {'Truth':>9}  Measures"
    print(header)
    for r in reports:
        measures = ", ".join(r["truth_measures"][:5]) + ("..." if len(r["truth_measures"]) > 5 else "")
        print(f"{r['name']:<20} {r['runs']:>5} {r['topics']:>7} {r['assessed']:>9} "
              f"{r['truth_status']:>9}  {measures}")


def make_splits(reports: list[dict[str, Any]]) -> None:
    """Deterministic splits (docs/evaluation-2025-plan.md Phase D). Topic folds only for the
    development dataset (rag25-generation); every dataset gets a run-ID holdout so the four
    known/held-out x known/held-out generalization quadrants can be formed against it."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for r in reports:
        rng = random.Random(SPLIT_SEED)
        run_holdout = sorted(r["all_runs"])
        rng.shuffle(run_holdout)
        cut = max(1, round(len(run_holdout) * 0.8))
        known_runs, held_out_runs = sorted(run_holdout[:cut]), sorted(run_holdout[cut:])

        assessed = sorted(r["assessed_topics"], key=str)
        doc: dict[str, Any] = {
            "dataset": r["name"],
            "role": r["role"],
            "seed": SPLIT_SEED,
            "algorithm": "random.Random(seed).shuffle(sorted(ids)) then slice",
            "run_holdout": {"known": known_runs, "held_out": held_out_runs, "ratio": "80/20"},
            "assessed_topics": assessed,
        }

        if r["role"] == "development":
            rng2 = random.Random(SPLIT_SEED)
            shuffled = list(assessed)
            rng2.shuffle(shuffled)
            k = 4
            folds = [sorted(shuffled[i::k]) for i in range(k)]
            doc["topic_folds"] = {"k": k, "folds": {f"fold-{i}": f for i, f in enumerate(folds)}}

        out = SPLITS_DIR / f"{r['name']}.yml"
        out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[], help="restrict to dataset(s) by name")
    parser.add_argument("--make-splits", action="store_true", help="(re)write evaluation/splits/*.yml")
    args = parser.parse_args()

    entries = load_manifest()
    if args.dataset:
        known = {e["name"] for e in entries}
        unknown = set(args.dataset) - known
        if unknown:
            print(f"Unknown dataset(s): {sorted(unknown)} (known: {sorted(known)})", file=sys.stderr)
            return 1
        entries = [e for e in entries if e["name"] in args.dataset]

    reports = []
    blocked = []
    for entry in entries:
        try:
            r = check_dataset(entry)
        except Problem as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if r["missing_prio1_runs"]:
            print(f"note: {r['name']}: {len(r['missing_prio1_runs'])} prio1_runs listed in "
                  f"datasets.yml have no run file: {r['missing_prio1_runs']} "
                  f"(released archive dropped/renamed them after the manifest was generated)",
                  file=sys.stderr)
        if r["missing_assessed_topics"]:
            print(f"note: {r['name']}: {len(r['missing_assessed_topics'])} assessed_topics not "
                  f"found in the topics file: {r['missing_assessed_topics']}", file=sys.stderr)
        if r["truth_status"] == "missing":
            blocked.append(r["name"])
        reports.append(r)

    print_card(reports)

    if args.make_splits:
        make_splits(reports)

    if blocked:
        print(f"\nBLOCKED (truth missing, see evaluation/README.md): {', '.join(blocked)}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
