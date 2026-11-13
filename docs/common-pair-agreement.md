# Agreement on common pairs

"Common pairs" are the pairs the Arena/SPARROW methods compared directly in every variant, and which
every other method (e.g. `ragree-default`) can also be compared on. Inputs are scored per-topic
`*.eval.txt` rows only (copies in `report-support/data/eval/`); no response content or votes.

Script: `report-support/common_pair_agreement.py`. Outputs in `report-support/data/`:
`common_pairs.tsv` (the pair file), `common_pair_stats.json`, `common_pair_tau_b*.csv`,
`common_pair_agreement.png`.

## What counts as a common pair

- **Seed pairs**: deterministic, identical for `arena-support`, `rubric-support-arena`,
  `rubric-v2-support-arena` (see `sparrow-pair-selection-analysis.md`).
- **Adaptive/reversal pairs are excluded**: they depend on votes, are not recorded in scored outputs,
  and are restricted data. They may differ per variant.
- `ragree-default` is pointwise (it judges each response alone), so it never judged pairs; it is
  compared on the same pairs through its score order. `ragree_choice` (also SPARROW, same seed graph
  settings) was not among the uploaded runs.
- A pair's preference under a metric = sign of the score difference for that topic (ties allowed);
  `CRITICAL_ERROR_RATE` is negated. Arena scores come from a Davidson-BT fit on all of a topic's
  ballots, so they reflect, but are not equal to, the direct vote on a pair.

## Descriptive statistics

| | |
|---|---|
| Topics x runs | 119 x 83 |
| Common pairs per topic | 166 of 3403 possible (4.9%) |
| Common pairs total | 19,754 (`common_pairs.tsv`) |
| Pair set across topics | same 166 run pairs in every topic |
| Comparisons per run (degree) | 4 for 81 runs; 3 for one run; 5 for one run |
| Arena variants prefer the same side (all 3) | 72.7% of pairs (73.6% excluding ties) |

Tie rates (share of pairs with equal scores), common pairs vs all pairs:

| Metric | Common | All |
|---|---|---|
| Arena score (arena / rubric1 / rubric2) | 0.33% / 0.63% / 0.62% | 0.02% / 0.03% / 0.03% |
| Arena citation precision / recall (all) | 0.68% / 0.66% | 0.45% / 0.43% |
| ragree CITATION_PRECISION | 1.0% | 0.8% |
| ragree CLAIM_F1 | 3.0% | 3.0% |
| ragree CONTEXT_FAITHFULNESS | 0.9% | 0.8% |
| ragree ESTIMATED_COVERAGE | 31% | 32% |
| ragree CITATION_COVERAGE | 69% | 70% |
| ragree CRITICAL_ERROR_RATE | 91% | 91% |
| ragree EVIDENCE_COVERAGE_RATE | 98% | 98% |

The last three ragree measures barely discriminate between runs, so their agreement values are
based on few non-tied pairs. Arena scores tie 10-20x more often on common pairs than on all
pairs: directly compared pairs more often end in a tie/abstain, so the fit gives them equal scores.

## Agreement (pooled tie-aware Kendall tau_b over pair directions)

Figure: `report-support/data/common_pair_agreement.png` (left: common pairs; right: common minus all
pairs). Full matrices: `common_pair_tau_b.csv`, `common_pair_tau_b_full.csv` (all pairs),
`common_pair_tau_b_topic_mean.csv` (mean over topics).

| Pair of metrics | Common pairs | All pairs |
|---|---|---|
| arena vs rubric1 arena score | 0.64 | 0.56 |
| arena vs rubric2 | 0.64 | 0.57 |
| rubric1 vs rubric2 | 0.65 | 0.59 |
| arena score vs ragree ESTIMATED_COVERAGE | 0.52 | 0.45 |
| arena score vs ragree CLAIM_F1 | 0.29 | 0.24 |
| arena score vs ragree CITATION_PRECISION | 0.03 | 0.02 |
| arena score vs ragree CONTEXT_FAITHFULNESS | -0.04 | -0.04 |
| arena CIT_PREC_ALL vs ragree CONTEXT_FAITHFULNESS | 0.78 | 0.77 |
| arena CIT_PREC_ALL vs ragree CITATION_PRECISION | 0.72 | 0.72 |

## Findings

- The pattern is the same as in `judge-agreement-analysis.md`: two families. Preference (arena score)
  correlates with ragree's coverage estimate (0.52) and, weaker, CLAIM_F1 (0.29); citation support
  (arena citation lane and ragree faithfulness/precision) forms a separate group (0.7-0.8); the two
  families are uncorrelated (about 0 or slightly negative).
- Restricting to directly compared pairs raises preference-related agreement by about 0.05-0.08
  (largest differences between any two metrics: 0.078; mean absolute change 0.015). Support-related
  agreements are unchanged. A plausible reading is that pairs the arena compared directly are the
  ones whose order the arena fit is most anchored on, but this subset is a fixed, near-regular
  graph, not a random sample, so treat the gain as descriptive.
- The three arena variants agree on direction for only about 73% of common pairs (tau_b about 0.64
  per pair, versus 0.86-0.95 between system-level rankings). Pair-level, per-topic ordering is much
  noisier than aggregate ranking; the rubric changes many individual pair preferences while barely
  changing the overall system ranking.
- Not measured: agreement of the actual direct votes on the seed pairs (restricted) and anything
  involving adaptive pairs. See the `pair_log` procedure in `sparrow-pair-selection-analysis.md`.
