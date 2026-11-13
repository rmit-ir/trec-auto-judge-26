# Evidence gaps before final submission

- Recover exact executed source snapshots: all four saved configurations
  record dirty working trees. The relevant runtime source is committed and unchanged
  between those base commits and the synchronized HEAD, but no saved uncommitted
  patch was found. Preserve exact uploaded outputs, cache provenance,
  and model/deployment versions before making reproducibility claims.
- Complete billing reconciliation remains open, but cost evidence is substantially
  recovered: the RAGree run has documented call/token totals and a $54.31 estimate;
  Arena runs are documented at about $90 including V2. Itemized bake-off usage
  ledgers exist. Construction cache copies overlap and do not capture every billed
  attempt; no total invoice or cold-cache comparison was recovered.
  The rough estimate for retained successful construction calls is $5–6 under
  the documented production-model rate assumption; this is not a complete bill
  and is explicitly qualified in the manuscript. See `documentation-review.md` and
  `evidence/cost-ledger-review.json`.
- No new judge run, model call, truth download, or tuning was performed for this report.

## Resolved submission evidence

The author confirmed two code submissions, RAGree and Choice. The supplied TIRA
upload screenshot confirms the four data submissions, all exclusively on RAG26;
see `evidence/tira-rag26-run-uploads.png`. Historical handoff notes describing
planned Arena code uploads are not the final submission record.

## Post-conference work (outside this submission)

The author confirmed that official results will be released only after the TREC
conference, while this report is submitted beforehand. Official correlations,
rankings, and human-agreement analysis are therefore future work, not missing
evidence required for this pre-conference report.
