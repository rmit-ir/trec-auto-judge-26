#!/usr/bin/env python3
"""Per-topic Krippendorff's alpha (ordinal) across judges/metrics, 'annotators' = metric columns,
'units' = systems. Each annotator's scores are converted to within-topic ranks (ties share the
average rank) so measures on different scales share one ordinal domain. Higher = better for all
(lower-is-better measures are negated). Reads *.eval.txt per-topic rows (scored outputs only).

Usage: topic_alpha.py OUT_PREFIX name=path [name=path ...]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

LOWER_IS_BETTER = {"CRITICAL_ERROR_RATE", "BOTH_BAD_RATE"}


def alpha_ordinal(R):
    """R: units x raters array of ordinal values (NaN = missing)."""
    cats = np.unique(R[~np.isnan(R)])
    idx = np.searchsorted(cats, R)
    ok = ~np.isnan(R)
    H = np.zeros((R.shape[0], len(cats)))
    for j in range(R.shape[1]):
        m = ok[:, j]
        np.add.at(H, (np.where(m)[0], idx[m, j]), 1)
    m_u = H.sum(1)
    keep = m_u >= 2
    H, m_u = H[keep], m_u[keep]
    o = (H.T / (m_u - 1)) @ H - np.diag((H / (m_u - 1)[:, None]).sum(0))
    n_c = o.sum(1)
    n = n_c.sum()
    cum = np.cumsum(n_c)
    # delta^2(c,k) = (sum_{g=c..k} n_g - (n_c + n_k)/2)^2
    lo, hi = np.minimum.outer(np.arange(len(cats)), np.arange(len(cats))), \
             np.maximum.outer(np.arange(len(cats)), np.arange(len(cats)))
    d2 = (cum[hi] - cum[lo] + n_c[lo] - (n_c[lo] + n_c[hi]) / 2) ** 2
    Do = (o * d2).sum() / n
    De = (np.outer(n_c, n_c) * d2).sum() / (n * (n - 1))
    return 1 - Do / De


def selfcheck():
    rng = np.random.default_rng(0)
    x = rng.permutation(60).astype(float)
    assert abs(alpha_ordinal(np.c_[x, x, x]) - 1) < 1e-9
    r = np.c_[rng.permutation(300), rng.permutation(300), rng.permutation(300)].astype(float)
    assert abs(alpha_ordinal(r)) < 0.1
    assert alpha_ordinal(np.c_[x, 59 - x]) < -0.9      # reversed raters


def load(args):
    runs = {}
    for a in args:
        n, p = a.split("=", 1)
        d = pd.read_csv(p, sep="\t", names=["run", "topic", "measure", "value"])
        runs[n] = d[d.topic != "all"].pivot_table(index=["topic", "run"], columns="measure", values="value")
    cols = {}
    for n, d in runs.items():
        if "RAG26_ARENA_SCORE" in d:
            cols[f"{n}: ARENA_SCORE"] = d["RAG26_ARENA_SCORE"]
    first = next(n for n, d in runs.items() if "RAG26_ARENA_SCORE" in d)
    for m in ["RAG26_CIT_PREC_ALL", "RAG26_CIT_REC_ALL"]:
        cols[f"{first}: {m.removeprefix('RAG26_')}"] = runs[first][m]
    for n, d in runs.items():
        if "RAG26_ARENA_SCORE" not in d:
            for m in d.columns:
                cols[f"{n}: {m}"] = -d[m] if m in LOWER_IS_BETTER else d[m]
    return pd.DataFrame(cols)


def main():
    selfcheck()
    prefix, df = sys.argv[1], load(sys.argv[2:])
    sets = {
        "arena judges (3)": [c for c in df if "ARENA_SCORE" in c],
        "all measures (12)": list(df),
    }
    res = {}
    for name, cols in sets.items():
        res[name] = {}
        for t, g in df[cols].groupby(level="topic"):
            R = np.apply_along_axis(lambda v: rankdata(v, nan_policy="omit"), 0, g.values)
            res[name][t] = alpha_ordinal(R)
    out = pd.DataFrame(res)
    out.index.name = "topic"
    out.to_csv(f"{prefix}.csv")
    print(out.describe().round(3).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1, 2.2]})
    ax[0].boxplot([out[c].dropna() for c in out], tick_labels=list(out), showmeans=True)
    for i, c in enumerate(out, 1):
        ax[0].scatter(np.random.default_rng(1).normal(i, .05, len(out)), out[c], s=8, alpha=.4)
    ax[0].set_ylabel("Krippendorff α (ordinal)")
    ax[0].set_title("Distribution over 119 topics")
    srt = out.sort_values(out.columns[-1])
    for c in out:
        ax[1].plot(range(len(srt)), srt[c].values, marker=".", label=c)
    ax[1].set_xlabel("topics, sorted by α of all measures")
    ax[1].set_title("Per-topic agreement")
    ax[1].legend()
    for a in ax:
        a.axhline(0.667, ls=":", c="gray"); a.axhline(0, c="k", lw=.5); a.grid(alpha=.3)
    fig.savefig(f"{prefix}.png", dpi=200); fig.savefig(f"{prefix}.pdf")


if __name__ == "__main__":
    main()
