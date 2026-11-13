#!/usr/bin/env python3
"""Pairwise rank agreement heatmap (tie-aware Kendall tau_b and tauAP_b) between judges/metrics.

Ports tau_b and tauAP_b from Urbano & Marrero (ICTIR 2017), R package `ircor`
(https://github.com/julian-urbano/ircor). Both are symmetric 'agreement' variants.
Reads only aggregate (topic == 'all') rows of *.eval.txt.

Usage: agreement_heatmap.py OUT_PREFIX name=path [name=path ...]
  e.g. names: arena, rubric1, rubric2, ragree  (arena-style files have RAG26_ARENA_SCORE)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

LOWER_IS_BETTER = {"CRITICAL_ERROR_RATE", "BOTH_BAD_RATE"}


def tau_b(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx = np.sign(x[:, None] - x[None, :])
    sy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(len(x), 1)
    sx, sy = sx[iu], sy[iu]
    return (sx * sy).sum() / np.sqrt((sx != 0).sum() * (sy != 0).sum())


def _tauap_b_ties(x, y):
    # x, y already negated (ascending order = best first), as in ircor
    rx = rankdata(x)                       # average ranks
    ry = rankdata(y, method="ordinal")     # ties.method = "first"
    p = rankdata(y, method="min")
    not_top = np.where(p != 1)[0]
    c_all = 0.0
    for i in not_top:
        above = np.where(p < p[i])[0]
        c = (np.sign(rx[i] - rx[above]) == np.sign(ry[i] - ry[above])).sum()
        c_all += c / (p[i] - 1)
    return 2 / len(not_top) * c_all - 1


def tauap_b(x, y):
    x, y = -np.asarray(x, float), -np.asarray(y, float)   # decreasing = TRUE
    return (_tauap_b_ties(x, y) + _tauap_b_ties(y, x)) / 2


def selfcheck():  # values from the ircor README
    x = np.round(np.array([.06, .2, .27, .37, .57, .63, .66, .9, .91, .94]) * 5) / 5
    y = np.round(np.array([.37, .06, .2, .27, .57, .66, .63, .91, .9, .94]) * 5) / 5
    assert abs(tau_b(x, y) - 0.75) < 1e-6
    assert abs(tauap_b(x, y) - 0.6269841) < 1e-6


def corr_matrix(df, fn):
    cols = df.columns
    m = pd.DataFrame(np.nan, index=cols, columns=cols)
    for a in cols:
        for b in cols:
            ok = df[[a, b]].dropna()
            m.loc[a, b] = fn(ok[a].values, ok[b].values)
    return m


def main():
    selfcheck()
    prefix, runs = sys.argv[1], {}
    for a in sys.argv[2:]:
        n, p = a.split("=", 1)
        d = pd.read_csv(p, sep="\t", names=["run", "topic", "measure", "value"])
        runs[n] = d[d.topic == "all"].pivot(index="run", columns="measure", values="value")

    cols = {}
    for n, d in runs.items():
        if "RAG26_ARENA_SCORE" in d:
            cols[f"{n}: ARENA_SCORE"] = d["RAG26_ARENA_SCORE"]
    first_arena = next(n for n, d in runs.items() if "RAG26_ARENA_SCORE" in d)
    for m in ["RAG26_CIT_PREC_ALL", "RAG26_CIT_REC_ALL"]:
        cols[f"{first_arena}: {m.removeprefix('RAG26_')}"] = runs[first_arena][m]
    for n, d in runs.items():
        if "RAG26_ARENA_SCORE" not in d:
            for m in d.columns:
                s = -d[m] if m in LOWER_IS_BETTER else d[m]
                cols[f"{n}: {m}" + (" (inv.)" if m in LOWER_IS_BETTER else "")] = s
    df = pd.DataFrame(cols)

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.2), constrained_layout=True)
    for ax, (title, fn) in zip(axes, [("Kendall τ$_b$ (tie-aware)", tau_b),
                                      ("τ$_{AP,b}$ (tie-aware, top-weighted)", tauap_b)]):
        m = corr_matrix(df, fn)
        m.to_csv(f"{prefix}_{'tau_b' if fn is tau_b else 'tauap_b'}.csv")
        im = ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(m)), m.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(m)), m.index, fontsize=8)
        for i in range(len(m)):
            for j in range(len(m)):
                v = m.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 else "black")
        ax.set_title(title, fontsize=12)
        ax.set_xticks(np.arange(-.5, len(m)), minor=True)
        ax.set_yticks(np.arange(-.5, len(m)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=axes, shrink=0.7, label="rank correlation over 83 systems")
    fig.suptitle("Agreement between judges and metrics (system rankings, aggregate over topics)", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(f"{prefix}.{ext}", dpi=200)
    print("wrote", prefix + ".png/.pdf and CSVs")


if __name__ == "__main__":
    main()
