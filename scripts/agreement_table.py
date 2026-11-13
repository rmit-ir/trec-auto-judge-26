#!/usr/bin/env python3
"""Merge agreement_heatmap.py / topic_alpha.py CSVs into one long TSV: analysis, a, b, value."""
import sys
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "docs/figures"
parts = []
for name, f in [("kendall_tau_b", "judge_agreement_tau_b.csv"), ("tau_ap_b", "judge_agreement_tauap_b.csv")]:
    m = pd.read_csv(f"{d}/{f}", index_col=0)
    s = m.stack().rename("value").reset_index()
    s.columns = ["a", "b", "value"]
    parts.append(s.assign(analysis=name))
t = pd.read_csv(f"{d}/topic_alpha.csv", index_col=0).stack().rename("value").reset_index()
t.columns = ["a", "b", "value"]  # a = topic, b = annotator set
parts.append(t.assign(analysis="krippendorff_alpha_ordinal_per_topic"))
out = pd.concat(parts)[["analysis", "a", "b", "value"]]
out.to_csv(f"{d}/agreement_values.tsv", sep="\t", index=False, float_format="%.4f")
print(len(out), "rows")
