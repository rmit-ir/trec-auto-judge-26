# SPARROW pair selection across methods

Question: did the SPARROW-based methods (`arena-support`, `rubric-support-arena`,
`rubric-v2-support-arena`; all `RAG26ArenaJudge`) always judge the same pairs, or different ones?

Script: `report-support/sparrow_pairs.py`. Inputs for `seed` are only run_id/topic_id columns of a
scored `.eval.txt` (83 runs x 119 topics); no response content or votes.

## Result: seed pairs, deterministic and identical

`sparse_seed_graph` (`judges/ragree_choice/sparrow.py`) has no randomness: it sorts candidate ids,
builds a cycle, then tops up by greedy min-degree. With 83 candidates and `initial_graph_degree: 4`
every topic gets 166 of 3403 possible pairs (4.9%). All three variants share these settings
(rubrics and citation support do not touch pair selection), and every topic has the same 83
candidates, so:

- seed pairs are identical across the three variants (verified by the script);
- the seed graph is the same set of run_id pairs on every topic (verified).

```bash
.venv/bin/python report-support/sparrow_pairs.py seed path/to/any-arena.eval.txt
```

## Not random, but not guaranteed identical: adaptive and reversal pairs

After the seed ballots, 4 adaptive pairs per topic come from `select_adaptive_pairs`: a
deterministic max-heap over unseen pairs, ranked by top-1 boundary bonus plus
`1 / (1 + |utility gap|)`, with utilities fitted to that variant's seed votes. Position-reversal
re-checks (2 per topic) are chosen from cast ballots the same way. No random sampling anywhere,
but because votes differ by variant (the rubric changes prompts), adaptive and reversal pairs may
differ between variants. If two variants fitted the same utilities, they would pick the same pairs.
For scale: two independent uniform draws of 4 from the 3237 non-seed pairs overlap by 0.005 pairs
per topic in expectation, so any overlap well above that reflects shared structure, not chance.

The saved `.eval.txt` outputs record no pair identities, and adaptive/reversal pair sets are derived
from votes, which the data policy treats as restricted. They were therefore not measured here.

## Measuring it (data owner)

Arena judge accepts `pair_log: true` (judge setting), writing `<filebase>.pairs.jsonl` with rows
`{phase: seed|adaptive|reversal, topic, a, b}` (ids only, no votes). Re-run the variants with it
(cached prompts make this cheap), then:

```bash
.venv/bin/python report-support/sparrow_pairs.py overlap arena.pairs.jsonl rubric1.pairs.jsonl rubric2.pairs.jsonl
```

This prints per-phase Jaccard and shared counts. Treat the logs as restricted: do not commit them,
and do not paste their contents into agent sessions.
