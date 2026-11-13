"""Aggregate only runner .eval.txt scores; never reads answers or prompt caches.

Run with the implementation checkout as the sole argument (SciPy required).
Prints no candidate or topic identifiers; does not change judge configuration.
"""
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from statistics import mean

from scipy.stats import kendalltau, spearmanr


def load(path, measures):
    rows = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        assert len(fields) == 4, "Unexpected scored-output schema"
        run, topic, measure, value = fields
        key = (run, topic, measure)
        assert key not in rows, "Duplicate score"
        rows[key] = float(value)
        assert math.isfinite(rows[key]), "Nonfinite score"
    runs = sorted({r for r, _, _ in rows})
    topics = sorted({t for _, t, _ in rows} - {"all"})
    assert len(runs) == 83 and len(topics) == 119
    metric_names = {m for _, _, m in rows}
    assert len(metric_names) == measures
    assert len(rows) == 83 * 120 * measures
    assert all((r, t, m) in rows for r in runs for t in topics + ["all"]
               for m in metric_names), "Incomplete score grid"
    return rows, runs, topics


def analyze(root):
    base = Path("output/rag26-generation")
    paths = {"Plain": base / "rag26_arena/arena-support-all-all/rag26-arena.eval.txt",
             "V1": base / "rag26_arena/rubric-support-arena-all-all/rag26-arena.eval.txt",
             "V2": base / "rag26_arena/rubric-v2-support-arena-all-all/rag26-arena.eval.txt",
             "RAGree": base / "ragree/default-all-all/ragree.eval.txt"}
    data = {name: load(root / path, 7 if name == "RAGree" else 16)
            for name, path in paths.items()}
    runs, topics = data["Plain"][1:]
    assert all((r, t) == (runs, topics) for _, r, t in data.values())

    def values(name, topic, measure):
        return [data[name][0][run, topic, measure] for run in runs]

    def topic_value(name, topic, measure):
        v = values(name, topic, measure)
        assert len(set(v)) == 1, "Expected repeated topic-level diagnostic"
        return v[0]

    result = {"source_sha256": {name: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                 for name, p in paths.items()},
              "topics": len(topics), "candidates_per_topic": len(runs),
              "diagnostics": {}, "cross_run_agreement": {}}
    for name in ("Plain", "V1", "V2"):
        checked = [topic_value(name, t, "POSITION_REVERSAL_CHECKED") for t in topics]
        disagreements = [n * topic_value(name, t, "POSITION_REVERSAL_DISAGREEMENT_RATE")
                         for t, n in zip(topics, checked)]
        assert all(n in (0, 1, 2) for n in checked)
        assert all(abs(d - round(d)) < 1e-8 for d in disagreements)
        stable = [topic_value(name, t, "RANKING_STABLE") for t in topics]
        assert set(stable) <= {0, 1}
        result["diagnostics"][name] = {
            "successful_reversals": int(sum(checked)),
            "reversal_disagreements": int(round(sum(disagreements))),
            "pooled_reversal_disagreement_rate": sum(disagreements) / sum(checked),
            "mean_topic_both_bad_rate": mean(topic_value(name, t, "BOTH_BAD_RATE") for t in topics),
            "mean_topic_cycle_rate": mean(topic_value(name, t, "CYCLE_RATE") for t in topics),
            "kemeny_winner_agreement_topics": int(sum(stable)),
            "mean_ballot_coverage": mean(mean(values(name, t, "BALLOT_COVERAGE")) for t in topics),
            "observed_condorcet_winners": sum(sum(values(name, t, "CONDORCET_WINNER")) for t in topics),
            "topics_with_top_score_ties": sum(values(name, t, "RAG26_ARENA_SCORE").count(
                max(values(name, t, "RAG26_ARENA_SCORE"))) > 1 for t in topics)}
    for a, b in combinations(("Plain", "V1", "V2"), 2):
        tau, rho, agree = [], [], []
        for t in topics:
            x, y = (values(name, t, "RAG26_ARENA_SCORE") for name in (a, b))
            tau.append(float(kendalltau(x, y, variant="b").statistic))
            rho.append(float(spearmanr(x, y).statistic))
            agree.append({i for i, v in enumerate(x) if v == max(x)} ==
                         {i for i, v in enumerate(y) if v == max(y)})
        assert all(math.isfinite(v) for v in tau + rho)
        result["cross_run_agreement"][a + " vs " + b] = {
            "mean_topic_kendall_tau_b": mean(tau), "mean_topic_spearman_rho": mean(rho),
            "same_top_score_set_topics": sum(agree)}
    correlations = {}
    for name, metric in (("Plain", "RAG26_CIT_PREC_FIRST"),
                         ("Plain", "RAG26_CIT_REC_FIRST"),
                         ("RAGree", "ESTIMATED_COVERAGE"),
                         ("RAGree", "CONTEXT_FAITHFULNESS")):
        coefficients, sample_sizes = [], []
        for t in topics:
            x, y = values("Plain", t, "RAG26_ARENA_SCORE"), values(name, t, metric)
            mask = (values("Plain", t, "RAG26_SUPPORT_EVALUATED") if name == "Plain"
                    else [1] * len(runs))
            pairs = [(a, b) for a, b, available in zip(x, y, mask) if available == 1]
            if len(pairs) >= 2 and len({a for a, _ in pairs}) > 1 and len({b for _, b in pairs}) > 1:
                coefficients.append(float(spearmanr(*zip(*pairs)).statistic))
                sample_sizes.append(len(pairs))
        correlations[name + ":" + metric] = {"mean_topic_spearman_rho": mean(coefficients),
            "topics": len(coefficients), "candidate_count_range": [min(sample_sizes), max(sample_sizes)]}
    result["plain_preference_correlations"] = correlations
    return result


if __name__ == "__main__":
    assert math.isclose(kendalltau([1, 2, 3], [3, 2, 1]).statistic, -1)
    assert math.isclose(spearmanr([1, 1, 3], [2, 2, 6]).statistic, 1)
    assert math.isclose(kendalltau([1, 1, 3], [1, 2, 3]).statistic, 2 / math.sqrt(6))
    assert len(sys.argv) == 2, "Pass implementation checkout path"
    print(json.dumps(analyze(Path(sys.argv[1])), indent=2, allow_nan=False))
