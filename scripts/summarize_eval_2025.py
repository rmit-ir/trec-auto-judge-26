#!/usr/bin/env python3
"""Summarize 2025 pilot evaluation experiments (docs/evaluation-2025-plan.md Phase Q).

Reads every evaluation/results/*/manifest.yml (written by scripts/run_eval_2025.py) and prints
a console table, plus optional JSONL/CSV.

    uv run python scripts/summarize_eval_2025.py
    uv run python scripts/summarize_eval_2025.py --jsonl out.jsonl --csv out.csv

Correlation/cost columns (truth_measure, eval_measure, correlation, ci_low, ci_high,
pairwise_accuracy, llm_calls, tokens, estimated_cost) are populated from a sibling
`correlations.jsonl` next to manifest.yml when one exists; none of the current experiments have
one yet because `auto-judge-evaluate meta-evaluate` prints its table to stdout rather than a
file, and no 2025 dataset has truth to correlate against regardless (see evaluation/README.md).
Wiring `--output correlations.jsonl` into run_eval_2025.py's meta-evaluate step is the natural
next step once a truth bundle exists -- until then these columns are blank, not fabricated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "evaluation" / "results"

FIELDS = [
    "experiment_id", "commit_sha", "dataset", "split", "judge", "variant",
    "truth_measure", "eval_measure", "correlation", "estimate", "ci_low", "ci_high",
    "pairwise_accuracy", "llm_calls", "tokens", "estimated_cost", "wall_time_seconds", "success",
]


def load_experiments(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    rows = []
    for manifest_path in sorted(results_dir.glob("*/manifest.yml")):
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        row = {field: manifest.get(field) for field in FIELDS}
        row["judge"] = Path(manifest.get("judge", "")).parent.name or manifest.get("judge")

        correlations_path = manifest_path.parent / "correlations.jsonl"
        if correlations_path.is_file():
            with open(correlations_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = dict(row)
                        entry.update(json.loads(line))
                        rows.append(entry)
            continue
        rows.append(row)
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    cols = ["experiment_id", "dataset", "judge", "split", "correlation", "wall_time_seconds", "success"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) if rows else len(c) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "") or "").ljust(widths[c]) for c in cols))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--jsonl", type=Path, help="write full rows as JSON Lines")
    parser.add_argument("--csv", type=Path, help="write full rows as CSV")
    args = parser.parse_args()

    rows = load_experiments(args.results_dir)
    if not rows:
        print(f"No experiments found under {args.results_dir}", file=sys.stderr)
        return 0

    print_table(rows)

    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({k: r.get(k) for k in FIELDS} for r in rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
