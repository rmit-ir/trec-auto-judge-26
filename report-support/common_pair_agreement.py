"""Agreement restricted to pairs the SPARROW/Arena methods compared directly ("common pairs").

Common pairs = seed-graph pairs (topic, run_a, run_b): identical for all three Arena variants
(see docs/sparrow-pair-selection-analysis.md). Inputs: scored per-topic .eval.txt rows only,
copies under report-support/data/eval/. Reads no response content and no votes.

Outputs (report-support/data/): common_pairs.tsv, common_pair_stats.json,
common_pair_tau_b.csv (pooled), common_pair_tau_b_full.csv (same metrics, all pairs, pooled),
common_pair_agreement.png.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from judges.ragree_choice.sparrow import sparse_seed_graph  # noqa: E402

EV, OUT = HERE / "data" / "eval", HERE / "data"
FILES = {"arena": "arena-support", "rubric1": "rubric-support-arena", "rubric2": "rubric-v2-support-arena",
         "ragree": "ragree-default"}
# Same 12 columns as docs/judge-agreement-analysis.md; lower-is-better negated.
COLS = [("arena", "RAG26_ARENA_SCORE", 1), ("rubric1", "RAG26_ARENA_SCORE", 1), ("rubric2", "RAG26_ARENA_SCORE", 1),
        ("arena", "RAG26_CIT_PREC_ALL", 1), ("arena", "RAG26_CIT_REC_ALL", 1)] + \
       [("ragree", m, -1 if m == "CRITICAL_ERROR_RATE" else 1) for m in
        ["CITATION_COVERAGE", "CITATION_PRECISION", "CLAIM_F1", "CONTEXT_FAITHFULNESS", "CRITICAL_ERROR_RATE",
         "ESTIMATED_COVERAGE", "EVIDENCE_COVERAGE_RATE"]]
NAMES = [f"{j}: {m}" + (" (inv.)" if s < 0 else "") for j, m, s in COLS]


def load(name):
    d = pd.read_csv(EV / f"{FILES[name]}.eval.txt", sep="\t", names=["run", "topic", "measure", "value"])
    return d[d.topic != "all"]


data = {k: load(k) for k in FILES}
runs = sorted(data["arena"].run.unique())
topics = sorted(data["arena"].topic.unique())
for d in data.values():
    assert sorted(d.run.unique()) == runs and sorted(d.topic.unique()) == topics

# 1. common pairs (ids only)
ri = {r: i for i, r in enumerate(runs)}
edges = sparse_seed_graph(runs, 4, 5)  # arena defaults, shared by all variants; same for every topic
pairs = pd.DataFrame([(t, a, b) for t in topics for a, b in edges], columns=["topic", "run_a", "run_b"])
pairs.to_csv(OUT / "common_pairs.tsv", sep="\t", index=False)
ia = np.array([ri[a] for a, _ in edges]); ib = np.array([ri[b] for _, b in edges])

# 2. score tensor [topic, run, col]
T = np.full((len(topics), len(runs), len(COLS)), np.nan)
for c, (j, m, s) in enumerate(COLS):
    p = data[j][data[j].measure == m].pivot(index="topic", columns="run", values="value")
    T[:, :, c] = s * p.loc[topics, runs].to_numpy()
assert np.isfinite(T).all()
S = np.sign(T[:, ia, :] - T[:, ib, :])  # [topic, pair, col]
allia, allib = np.triu_indices(len(runs), 1)
SF = np.sign(T[:, allia, :] - T[:, allib, :])


def tau_b_matrix(S):
    n_t, n_p, n_c = S.shape
    X = S.reshape(-1, n_c)
    num = X.T @ X
    nz = (X != 0).astype(float)
    den = np.sqrt(np.outer(nz.sum(0), nz.sum(0)))
    return num / den


def tau_b_topic_mean(S):
    out = np.zeros((S.shape[2],) * 2)
    for i in range(S.shape[2]):
        for j in range(S.shape[2]):
            x, y = S[:, :, i], S[:, :, j]
            den = np.sqrt((x != 0).sum(1) * (y != 0).sum(1))
            out[i, j] = np.nanmean(np.where(den > 0, (x * y).sum(1) / np.where(den > 0, den, 1), np.nan))
    return out


tau_c, tau_f = tau_b_matrix(S), tau_b_matrix(SF)
tau_topic = tau_b_topic_mean(S)
pd.DataFrame(tau_c, NAMES, NAMES).to_csv(OUT / "common_pair_tau_b.csv")
pd.DataFrame(tau_f, NAMES, NAMES).to_csv(OUT / "common_pair_tau_b_full.csv")
pd.DataFrame(tau_topic, NAMES, NAMES).to_csv(OUT / "common_pair_tau_b_topic_mean.csv")

# 3. descriptive statistics
deg = np.bincount(np.concatenate([ia, ib]), minlength=len(runs))
iu = np.triu_indices(len(COLS), 1)
arena_idx = [0, 1, 2]
all_arena_same = ((S[:, :, 0] == S[:, :, 1]) & (S[:, :, 1] == S[:, :, 2])).mean()
nonzero_arena = (S[:, :, arena_idx] != 0).all(2)
dir_arena = ((S[:, :, 0] == S[:, :, 1]) & (S[:, :, 1] == S[:, :, 2]))[nonzero_arena].mean()
ragree_cov = COLS.index(("ragree", "ESTIMATED_COVERAGE", 1))
claim = COLS.index(("ragree", "CLAIM_F1", 1))
stats = {
    "topics": len(topics), "runs": len(runs), "pairs_per_topic": len(edges), "common_pairs_total": len(pairs),
    "possible_pairs_per_topic": len(allia), "fraction_of_all_pairs": len(edges) / len(allia),
    "runs_degree": {"min": int(deg.min()), "median": float(np.median(deg)), "max": int(deg.max()),
                    "histogram": {int(k): int(v) for k, v in zip(*np.unique(deg, return_counts=True))}},
    "tie_rate_common": dict(zip(NAMES, (S == 0).mean((0, 1)).round(4).tolist())),
    "tie_rate_all_pairs": dict(zip(NAMES, (SF == 0).mean((0, 1)).round(4).tolist())),
    "arena_variants_same_direction_all_pairs": float(all_arena_same),
    "arena_variants_same_direction_given_no_ties": float(dir_arena),
    "pooled_tau_b_common_vs_full": {
        "mean_abs_diff_offdiag": float(np.abs(tau_c - tau_f)[iu].mean()),
        "max_abs_diff_offdiag": float(np.abs(tau_c - tau_f)[iu].max()),
    },
}
(OUT / "common_pair_stats.json").write_text(json.dumps(stats, indent=1))

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, M, title in zip(axes, [tau_c, tau_c - tau_f], ["Common (seed) pairs: pooled tau_b", "Common minus all pairs"]):
    lim = 1 if M is tau_c else max(0.05, np.abs(M).max())
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(NAMES)), NAMES, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(NAMES)), NAMES, fontsize=7); ax.set_title(title, fontsize=10)
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, shrink=0.7)
fig.tight_layout(); fig.savefig(OUT / "common_pair_agreement.png", dpi=150)
print(json.dumps(stats, indent=1))
