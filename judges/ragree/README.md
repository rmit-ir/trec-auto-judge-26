# RAGree Evidence Judge

`RAGreeEvidenceJudge` is a first-cut, LLM-based implementation of the **Evidence
Judge** lane from the [RAGree design proposal](../../docs/ragree-judge-design.md).
It is deliberately scoped down from that document — read the doc's
"Implementation status in this repo" section for the full list of what's built
vs. not. In short: this checks claim grounding, citation quality, and
LLM-estimated answer coverage. It does **not** implement the Preference Judge,
the Trajectory Judge, retrieval precision/recall, or any of the training
pipeline in the design doc.

## What it measures

For each response, sentences are treated as "claims" (see the design doc for
why full atomic sub-sentence decomposition wasn't used). Each sentence with a
citation that resolves to a document in `Report.documents` gets one LLM call
judging `SUPPORTED` / `PARTIALLY_SUPPORTED` / `CONTRADICTED` /
`NOT_ENOUGH_EVIDENCE`, plus a per-citation support verdict. Coverage against
query-only LLM-generated nugget questions (`RAGreeNuggetCreator`) gives an
*estimate* of recall, not a measurement against gold nuggets.

- `CONTEXT_FAITHFULNESS` — grounding in cited context
- `ESTIMATED_COVERAGE` — LLM-judged nugget coverage (not gold recall)
- `CLAIM_F1` — harmonic mean of the two above
- `CITATION_PRECISION` — per-citation-link support rate
- `CITATION_COVERAGE` — fraction of substantive sentences carrying a citation
- `CRITICAL_ERROR_RATE` — fraction of claims verified CONTRADICTED
- `EVIDENCE_COVERAGE_RATE` — fraction of citations that were actually
  checkable (transparency measure: low values mean the other measures above
  rest on little evidence, not that the run is unfaithful — see the measure's
  full description in `RAGREE_SPEC`)

A run whose submissions don't include `Report.documents` (inline
cited/retrieved document text) will have `EVIDENCE_COVERAGE_RATE` near 0 and
`CONTEXT_FAITHFULNESS`/`CITATION_PRECISION` near 0 as a result — this is a
data-availability limit of this judge, not a claim that such runs are
unfaithful.

## Run locally

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=...
export CACHE_DIR=./cache-ragree

auto-judge run \
    --workflow judges/ragree/workflow.yml \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output-ragree/
```

```bash
auto-judge-evaluate meta-evaluate \
    --truth-leaderboard data/kiddie/eval/kiddie_fake.eval.ir_measures.txt \
    --truth-format ir_measures --truth-header \
    --eval-format ir_measures --on-missing default \
    output-ragree/ragree.eval.txt
```

The kiddie dataset is synthetic, so these numbers are not meaningful, but they
do exercise every code path (kiddie's runs happen to carry `Report.documents`,
so `EVIDENCE_COVERAGE_RATE` is 1.0 there) — a pipeline smoke test, not a
quality signal (same caveat as the other example judges).
