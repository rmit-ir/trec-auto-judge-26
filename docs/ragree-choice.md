# RAGree Choice, powered by SPARROW

RAGree Choice is a second RAGree judge design for settings where each topic has
multiple candidate system responses and the decision is preferential: select a
winner, top-`k`, or acceptable answer rather than independently grade every
response on one scale.

Its rule is simple: **every judge gets a vote; evidence gets a veto.**

## What this repository implements

[`judges/ragree_choice/`](../judges/ragree_choice/) implements a runnable
AutoJudge prototype with these components:

| SPARROW component | Implementation |
| --- | --- |
| Pointwise-first brief | One structured, evidence-grounded LLM brief per candidate response |
| Evidence House | Strictly parsed gate on critical error, factual-support estimate, and citation-validity estimate |
| Sparse acquisition | Full round robin for up to five candidates; deterministic near-regular graph above that |
| Hard ballot | `A`, `B`, `TIE`, or `ABSTAIN` JSON ballot |
| Operational aggregation | Weighted Davidson--Bradley--Terry fit with Davidson ties |
| Adaptive scheduling | Max-heap of unseen pairs, prioritising top-`k` boundary uncertainty |
| Selective position reversal | Bounded A/B-swap recheck of top-`k`-boundary and tied ballots; disagreement collapses to `ABSTAIN`, not `TIE` |
| Voting audit | Condorcet winner, directed three-cycle rate, and Kemeny--Young ranking |

The configurable defaults are in
[`workflow.yml`](../judges/ragree_choice/workflow.yml). The reusable
provider-independent ranking routines are in
[`sparrow.py`](../judges/ragree_choice/sparrow.py).

## What it deliberately does not claim to implement

An AutoJudge workflow receives one injected LLM endpoint. It cannot honestly
infer a genuinely independent multi-model jury from repeated calls to that one
endpoint. The local prototype therefore uses one ballot source, with an
explicit `judge_weight`; the `Ballot` data model retains `judge_id` and weight
fields so an external Bedrock orchestration service can supply independently
calibrated ballots from several families.

The following remain future, calibration-dependent work:

- human-calibrated judge reliability, error-correlation, and bias coefficients;
- actual multiple-Bedrock-model voter assignment and cost-aware routing;
- claim-level gold precision/recall, authoritative-corpus verification,
  retrieval metrics, and trajectory checks;
- reasoning arbiter escalation, conformal abstention, and human-review queue
  (selective position reversal is implemented — see below — but only that
  one Level-3 escalation step);
- stable scores against a frozen reference-anchor pool.

`CHOICE_SCORE` is therefore a pool-relative score. It should not be compared
across changing candidate pools without the planned frozen anchors.

## Run a local smoke test

Use the injected endpoint from the local, gitignored `.env` file:

```bash
set -a; source .env; set +a
auto-judge run \
  --workflow judges/ragree_choice/workflow.yml \
  --rag-responses data/kiddie/runs/repgen/ \
  --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
  --limit-topics 1 --limit-runs 2 \
  --out-dir ./output-ragree-choice/
```

`kiddie` is synthetic and unrestricted. Its output verifies the pipeline, not
the human alignment or ranking quality of the method.

## Interpreting the decision

An ineligible answer is assigned zero preference wins and cannot defeat an
eligible answer. If all candidates fail the gate, the judge marks every row for
the topic with `NO_ACCEPTABLE_ANSWER = 1`; it does not declare the least bad
response a winner.

That verdict has a strict availability condition: every candidate must first
have a schema-valid evidence brief after bounded retry, and every such brief
must fail eligibility. A timeout, malformed response, or wrong-typed field is
`EVIDENCE_EVALUATED = 0`, not a zero factual score and not an inferred absence
of a critical error. Those candidates are excluded from the election pending a
later run. Likewise, an explicit `ABSTAIN` is a completed pairwise ballot, but
a failed/malformed pairwise call is retried and remains available to SPARROW's
adaptive scheduler.

For graphs above the full-round-robin limit, an `initial_graph_degree` of 0 or
1 is treated as the minimum connected degree of 2. The deterministic cycle is
always created before any additional sparse edges, so no candidate is silently
left out of the election.

The Kemeny result is an audit, not the live aggregation rule. If it selects a
different winner from the Davidson--BT ranking, `RANKING_STABLE = 0`; a future
orchestrator should collect a targeted independent ballot or escalate to review.
For more than `exact_kemeny_limit` candidates (8 by default), this repository
uses a BT-seeded adjacent-swap local-search Kemeny heuristic rather than an
exact solver, so treat that large-election audit as a diagnostic only.

## Selective position reversal

`position_reversal_budget` (default 2 per topic) bounds how many already-cast
ballots are re-checked with the answer order swapped: directional ballots
whose pair straddles the top-`k` utility boundary first, then `TIE` ballots.
This is deliberately not every pair — reversing everything catches position
bias but does nothing for verbosity bias, at double the cost, which is why the
design only asks for it on the comparisons that can actually change the
decision that matters.

Reconciliation never double-counts the pair as two independent ballots. If the
reversed vote, mapped back to the original A/B orientation, agrees with the
first vote, the original ballot stands unchanged. A genuine disagreement
becomes `ABSTAIN`, not `TIE`: a swapped answer changing the winner is evidence
the decision was position-sensitive, not evidence the two answers are equally
good, and treating it as a tie would inject a false equality signal into the
Davidson--BT fit. A failed or unparseable reversal call after retries leaves
the original ballot exactly as it was and is not counted as checked —
`POSITION_REVERSAL_CHECKED = 0` disambiguates "not checked" from "checked, no
disagreement" for `POSITION_REVERSAL_DISAGREEMENT_RATE`.
