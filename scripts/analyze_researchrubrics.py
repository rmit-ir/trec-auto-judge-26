#!/usr/bin/env python3
"""Analyze the organizer ResearchRubrics development file (docs/trec-rag-2026-researchrubrics-
generation-plan.md Stage 0 / Section 12). No LLM calls; pure statistics over public organizer
development data (local-data/researchrubrics-dev/, not restricted 2026 response content).

    uv run python scripts/analyze_researchrubrics.py

Writes evaluation/researchrubrics/dev-rubric-stats.json (machine-readable) and
dev-rubric-stats.md (human-readable summary). These are the reference numbers the rubric
generator's criterion-count policy and axis mix must be calibrated against (plan Section 6) --
not the published ResearchRubrics-wide proportions blindly, and not RAG25 report-level rankings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from judges.rag26_arena.rubrics.io import check_unique_qids, load_jsonl  # noqa: E402
from judges.rag26_arena.rubrics.schema import PUBLISHED_AXIS_PRIORS  # noqa: E402
from judges.rag26_arena.rubrics.stats import compute_stats  # noqa: E402

DEFAULT_INPUT = REPO / "local-data" / "researchrubrics-dev" / "research-rubrics-dev-rubrics.jsonl"
DEFAULT_OUT_DIR = REPO / "evaluation" / "researchrubrics"


def render_markdown(stats_dict: dict, dupes: list[str]) -> str:
    cpt = stats_dict["criteria_per_topic"]
    lines = [
        "# ResearchRubrics organizer dev-file statistics",
        "",
        f"- topics: {stats_dict['n_topics']}",
        f"- criteria total: {stats_dict['n_criteria_total']}",
        f"- criteria/topic: mean={cpt['mean']:.2f} median={cpt['median']:.1f} "
        f"min={cpt['min']:.0f} max={cpt['max']:.0f}",
        f"- distinct axes/topic (mean): {stats_dict['axes_per_topic_mean']:.2f} "
        "(published ResearchRubrics-wide: ~4.74)",
        f"- duplicate qids: {dupes or 'none'}",
        f"- zero-weight criteria: {stats_dict['zero_weight_count']} "
        "(organizer data anomaly -- our own generator must not produce these, see validate.py)",
        f"- non-canonical axis rows: {stats_dict['non_canonical_axes'] or 'none'}",
        f"- exact-duplicate-criterion rate: {stats_dict['exact_duplicate_criterion_rate']:.3f}",
        f"- example-use rate ('(e.g.,'): {stats_dict['example_use_rate']:.3f}",
        "",
        "## Axis proportions: observed vs. published ResearchRubrics-wide prior",
        "",
        "| Axis | Observed | Published prior | Deviation |",
        "|---|---:|---:|---:|",
    ]
    for axis, prior in PUBLISHED_AXIS_PRIORS.items():
        observed = stats_dict["axis_proportions"].get(axis, 0.0)
        lines.append(f"| {axis} | {observed:.3f} | {prior:.3f} | {observed - prior:+.3f} |")
    for axis, count in stats_dict["non_canonical_axes"].items():
        lines.append(f"| {axis} (non-canonical) | {count / stats_dict['n_criteria_total']:.3f} | - | - |")

    lines += [
        "",
        "## Domain counts",
        "",
        "| Domain | Topics |",
        "|---|---:|",
    ]
    for domain, count in sorted(stats_dict["domain_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {domain} | {count} |")

    lines += [
        "",
        "## Complexity label counts",
        "",
    ]
    for field_name, counts in stats_dict["complexity_counts"].items():
        lines.append(f"- `{field_name}`: " + ", ".join(f"{v}={c}" for v, c in counts.items()))

    lines += [
        "",
        "## Mandatory rate by axis",
        "",
        "| Axis | Mandatory rate |",
        "|---|---:|",
    ]
    for axis, rate in sorted(stats_dict["mandatory_rate_by_axis"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {axis} | {rate:.3f} |")

    lines += [
        "",
        "## Common criterion openings (top 10)",
        "",
        "| Opening | Count |",
        "|---|---:|",
    ]
    for opening, count in stats_dict["common_openings"].items():
        lines.append(f"| {opening!r} | {count} |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Error: {args.input} not found. See local-data/researchrubrics-dev/README.md "
              "for how to obtain it.", file=sys.stderr)
        return 1

    topics = load_jsonl(args.input)
    dupes = check_unique_qids(topics)
    stats_dict = compute_stats(topics).to_dict()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "dev-rubric-stats.json").write_text(
        json.dumps(stats_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "dev-rubric-stats.md").write_text(
        render_markdown(stats_dict, dupes), encoding="utf-8")

    print(f"{stats_dict['n_topics']} topics, {stats_dict['n_criteria_total']} criteria "
          f"-> {args.out_dir}/dev-rubric-stats.{{json,md}}")
    if dupes:
        print(f"WARNING: duplicate qids: {dupes}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
