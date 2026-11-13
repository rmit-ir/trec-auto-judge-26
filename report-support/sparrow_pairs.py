"""Which pairs does SPARROW judge, and are they the same across methods?

  seed EVAL.txt        Seed graph per (variant, topic), from run_id/topic_id identifiers only
                       (first two columns of a scored .eval.txt). No response content, no votes.
  overlap LOG LOG ...  Jaccard overlap of pair sets between runs, from `pair_log: true` files
                       (<filebase>.pairs.jsonl). Adaptive/reversal rows derive from votes, so
                       those logs are restricted: the data owner runs this, agents do not.
"""
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from judges.ragree_choice.sparrow import sparse_seed_graph  # noqa: E402

WF = yaml.safe_load((ROOT / "judges/rag26_arena/workflow.yml").read_text())
SPARROW_VARIANTS = ["arena-support", "rubric-support-arena", "rubric-v2-support-arena"]


def seed(eval_path):
    cands = defaultdict(set)
    for line in Path(eval_path).read_text().splitlines():
        run, topic = line.split()[:2]
        if topic != "all":
            cands[topic].add(run)
    graphs = {}
    for v in SPARROW_VARIANTS:
        s = {**WF["judge_settings"], **WF["variants"][v].get("judge_settings", {})}
        graphs[v] = {t: set(sparse_seed_graph(sorted(c), s["initial_graph_degree"], s["full_round_robin_max_candidates"]))
                     for t, c in cands.items()}
    n = {len(c) for c in cands.values()}
    sizes = {len(g) for g in graphs[SPARROW_VARIANTS[0]].values()}
    first = next(iter(graphs[SPARROW_VARIANTS[0]].values()))
    out = {
        "topics": len(cands), "candidates_per_topic": sorted(n),
        "possible_pairs_per_topic": [k * (k - 1) // 2 for k in n],
        "seed_edges_per_topic": sorted(sizes),
        "seed_identical_across_variants": all(graphs[v] == graphs[SPARROW_VARIANTS[0]] for v in SPARROW_VARIANTS),
        "seed_identical_across_topics": all(g == first for g in graphs[SPARROW_VARIANTS[0]].values()),
        "adaptive_budget_per_topic": WF["judge_settings"]["adaptive_comparisons_per_topic"],
    }
    k = next(iter(n)); rest = k * (k - 1) // 2 - next(iter(sizes)); b = out["adaptive_budget_per_topic"]
    out["chance_expected_adaptive_overlap_pairs"] = b * b / rest  # two independent uniform draws of b from the non-seed pairs
    return out


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    sets = defaultdict(set)
    for r in rows:
        sets[r["phase"]].add((r["topic"], *sorted((r["a"], r["b"]))))
    return sets


def overlap(paths):
    logs = {p: load(p) for p in paths}
    out = {}
    for phase in ("seed", "adaptive", "reversal"):
        for p, q in itertools.combinations(paths, 2):
            a, b = logs[p][phase], logs[q][phase]
            out[f"{phase}: {Path(p).name} vs {Path(q).name}"] = {
                "jaccard": len(a & b) / len(a | b) if a | b else None, "shared": len(a & b), "sizes": [len(a), len(b)]}
    return out


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    print(json.dumps({"seed": seed, "overlap": overlap}[cmd](*args) if cmd == "seed" else overlap(args), indent=2))
