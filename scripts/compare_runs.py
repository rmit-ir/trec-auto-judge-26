#!/usr/bin/env python3
"""Compare AutoJudge submissions from their aggregate (topic == 'all') rows in *.eval.txt.

Usage: compare_runs.py name=path [name=path ...]
Format per line: run_id <tab> topic <tab> measure <tab> value. Reads scored outputs only.
"""
import sys
import pandas as pd
from scipy.stats import kendalltau, spearmanr

runs = {}
for a in sys.argv[1:]:
    name, path = a.split("=", 1)
    df = pd.read_csv(path, sep="\t", names=["run", "topic", "measure", "value"])
    runs[name] = df[df.topic == "all"].pivot(index="run", columns="measure", values="value")

arena = [n for n, d in runs.items() if "RAG26_ARENA_SCORE" in d]
print("systems per run:", {n: len(d) for n, d in runs.items()})

print("\n== mean of arena diagnostics (over systems) ==")
cols = ["BALLOT_COVERAGE", "BOTH_BAD_RATE", "CYCLE_RATE", "POSITION_REVERSAL_DISAGREEMENT_RATE",
        "RANKING_STABLE", "RAG26_ARENA_SCORE"]
print(pd.DataFrame({n: runs[n][cols].mean() for n in arena}).round(4).to_string())

def corr_table(series):
    names = list(series)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x = pd.concat([series[a], series[b]], axis=1, join="inner").dropna()
            rows.append((a, b, len(x), spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0],
                         kendalltau(x.iloc[:, 0], x.iloc[:, 1])[0]))
    return pd.DataFrame(rows, columns=["a", "b", "n", "spearman", "kendall"]).round(3).to_string(index=False)

print("\n== rank correlation: RAG26_ARENA_SCORE across arena runs ==")
print(corr_table({n: runs[n]["RAG26_ARENA_SCORE"] for n in arena}))

if "ragree" in runs:
    print("\n== rank correlation: ragree measures vs each arena RAG26_ARENA_SCORE ==")
    for m in runs["ragree"].columns:
        for n in arena:
            x = pd.concat([runs["ragree"][m], runs[n]["RAG26_ARENA_SCORE"]], axis=1, join="inner").dropna()
            print(f"{m:26s} vs {n:14s} spearman={spearmanr(x.iloc[:,0], x.iloc[:,1])[0]:.3f} n={len(x)}")
