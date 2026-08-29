# RAGree Choice / SPARROW

`RAGreeChoiceJudge` is the preferential-voting complement to the existing
RAGree Evidence Judge. It ranks the candidate responses (one run per system)
for each topic using pointwise evidence briefs followed by sparse, hard
pairwise ballots:

1. a successfully evaluated evidence eligibility gate vetoes responses with a
   critical error or inadequate factual/citation support; unavailable LLM calls
   are retried and then marked unavailable rather than judged as zero;
2. a connected sparse seed graph collects pairwise `A` / `B` / `TIE` /
   `ABSTAIN` ballots;
3. a weighted Davidson--Bradley--Terry fit estimates online utilities;
4. a max-heap schedules additional uncertain top-`k` boundary pairs;
5. Condorcet, cycle, and Kemeny--Young diagnostics audit the result.

The Kemeny audit is exact only through `exact_kemeny_limit` (default 8). Larger
elections use a BT-seeded adjacent-swap local-search heuristic and are labelled
as such in code; they are a cheap diagnostic, not an exact independent solver.

The complete rationale and configuration are in
[the RAGree Choice design](../../docs/ragree-choice.md).

## Important scope

The AutoJudge workflow receives one injected OpenAI-compatible endpoint through
`llm_config`; it therefore produces one ballot source per workflow run. It
never hard-codes a Bedrock model, Region, or credential. The reusable
[`sparrow.py`](sparrow.py) core accepts each ballot's `judge_id` and calibrated
`weight`, so a separate Bedrock orchestrator can combine multiple independent
model families after calibration. Do not pretend repeated calls to one endpoint
are independent family votes.

The evidence brief is a prompted, retrieved-context check. It is not a
human-calibrated factuality certificate and does not yet implement gold nugget
recall, retrieval precision/recall, trajectory scoring, conformal calibration,
or a persistent multi-model jury.

## Run locally

```bash
set -a; source .env; set +a
auto-judge run \
  --workflow judges/ragree_choice/workflow.yml \
  --rag-responses data/kiddie/runs/repgen/ \
  --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
  --out-dir ./output-ragree-choice/
```

Use `--limit-topics 1 --limit-runs 2` while iterating. The synthetic kiddie
dataset is a pipeline test, not a quality benchmark.

## Measures

- `CHOICE_SCORE` — candidate-pool-relative Davidson--BT win score (0-100).
- `EVIDENCE_EVALUATED` — whether a valid evidence brief was obtained after retry.
- `EVIDENCE_ELIGIBLE` — evidence-veto gate result, only meaningful when evaluated.
- `POINTWISE_EVIDENCE_SCORE` — scoped pointwise evidence estimate.
- `PAIRWISE_WIN_RATE` — observed hard-ballot performance.
- `CONDORCET_WINNER`, `RANKING_STABLE`, and `CYCLE_RATE` — voting audits.
- `BALLOT_COVERAGE` — sparse graph coverage, not an error metric.
- `NO_ACCEPTABLE_ANSWER` — every candidate was evaluated and failed the evidence
  gate; temporary endpoint failures cannot trigger it.

## Failure policy

Malformed, wrong-typed, and failed LLM replies are retried up to
`llm_max_attempts` with exponential backoff. A candidate without a valid brief
gets `EVIDENCE_EVALUATED = 0` and is excluded from the election; it is neither
an evidence failure nor a safety pass. A correctly parsed `critical_error:
true` vetoes the candidate. An explicit pairwise `ABSTAIN` is a recorded ballot,
while a failed or malformed pairwise reply is retried and never consumes an
adaptive comparison edge.
