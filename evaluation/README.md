# 2025 RAGree evaluation framework

Local benchmark for RAGree using the unrestricted 2025 pilot datasets (`rag25`, `ragtime25`,
`dragun25`) with human ground truth, per `docs/evaluation-2025-plan.md`. Kept separate from the
2026 AutoJudge datasets, which stay blind evaluation data — see the [data-handling
policy digest](../CLAUDE.md#restricted-evaluation-data--the-short-version) and
`docs/evaluation-2025-plan.md` §23 (2026 boundary).

## Status (2026-09-12)

**Blocked on truth data.** `uv run python scripts/prepare_eval_2025.py` (see below) reports the
current state directly from the filesystem:

```text
Dataset               Runs  Topics  Assessed     Truth
rag25-generation        51     105        22   missing
rag25-auggen            25     105        22   missing
ragtime25-repgen        59     122        16   missing
dragun25-repgen         28      30        30   missing
```

### The blocker, precisely

`./fetch_datasets.py --pilot` downloads `anonymized-runs-{rag25,ragtime25,dragun25}-v2.1.tar.gz`
from the release server. Each extracts `runs/`, `topics/`, `metadata/`, and the track's own
`datasets.yml` — **no `eval/` directory and no `truth:` key in that `datasets.yml`**. Contrast
with the 2026 test tracks (`rag26`, `ragtime26`), whose archives *do* ship an `eval/` (with
explicit placeholder values, since assessment hasn't happened for those yet — see
`local-data/rag26/README.md`). The organizer's own `stats.txt` files committed alongside these
pilot releases (`local-data/*/stats.txt`) report real `eval/` statistics (measures, assessed-topic
counts, qrels label distributions) generated from a path outside this checkout
(`/home/dietz/trec-auto-judge/...`) — the truth exists on the organizer's side, it just was not
included in the pilot `anonymized-runs-*` tarballs fetched here, and no separate truth/eval
tarball is referenced anywhere in `datasets.yml` or `fetch_datasets.py` for the pilot category.

**Action needed from a human:** ask the TREC AutoJudge organizers (contact in
`local-data/rag26/README.md`: Laura Dietz, `dietz@cs.unh.edu`) for the official 2025 pilot
truth/assessment bundle (or its download coordinates, if it is meant to be served the same way
via `TREC_AUTOJUDGE_USER`/`PASSWORD`), and drop it at:

```text
local-data/rag25/eval/rag.generation.official.eval.jsonl
local-data/rag25/eval/rag.auggen.official.eval.jsonl
local-data/ragtime25/eval/ragtime.repgen.official.eval.jsonl
local-data/dragun25/eval/dragun.repgen.official.eval.jsonl
```

(paths already wired into `evaluation/datasets-2025.yml`'s `truth:` field). Once a file lands
there, re-run `uv run python scripts/prepare_eval_2025.py` — it re-derives truth status from the
filesystem on every run, so no other change is needed for it to pick the bundle up. Do not
substitute a hand-built or reverse-engineered truth file — see `docs/evaluation-2025-plan.md`
§4 (Phase A3).

### Resolved discrepancy: RAGTIME25 `repgen-prio1`

`local-data/ragtime25/datasets.yml` lists 13 `prio1_runs` for `repgen`; the organizer's
`stats.txt` reports only 11 in `repgen-prio1`. `prepare_eval_2025.py` checks the listed run IDs
against the actual run files present and finds `lichen` and `radish` (2 of the 13) have no run
file in the fetched `runs/repgen/` (59 files) — 13 − 2 = 11, matching `stats.txt` exactly. Most
likely those two runs were withdrawn or renamed between when the priority list and the final
anonymized archive were generated. This check runs generically for every dataset, not just
RAGTIME (see `missing_prio1_runs` in `prepare_eval_2025.check_dataset`).

## What this directory is

| File | Purpose |
|---|---|
| `datasets-2025.yml` | Manifest: which 2025 datasets, their role, expected truth path/format. Does **not** retype `responses`/`topics`/`prio1_runs`/`assessed_topics` — those resolve from each fetched track's own `local-data/<track>/datasets.yml` via `from_release`, the same mechanism `run_all_datasets.py` already uses. |
| `splits/*.yml` | Deterministic splits (seed `20250912`, `random.Random(seed).shuffle(sorted(ids))` then slice — see file header). Regenerate with `scripts/prepare_eval_2025.py --make-splits`. |
| `results/` | One directory per experiment, gitignored (matches the repo's blanket `results/` ignore; the checked-in `.gitignore` here just keeps the empty directory present). |

Response/topic data itself is never copied here — always referenced under `local-data/`.

## Roles and splits (`docs/evaluation-2025-plan.md` §7)

- **`rag25-generation` — development.** 4 deterministic topic folds over the 22 assessed
  topics, for tuning/ablations/prompt changes. Every dataset also gets an 80/20 anonymous
  run-ID holdout (`run_holdout.known`/`held_out` in its split file) for the four
  known/held-out × known/held-out generalization quadrants, selectable via `run_eval_2025.py
  --runs known|held-out`.
- **`rag25-auggen` — validation.** Distribution-shift check after a candidate is selected on
  `rag25-generation`; not merged into the tuning pool.
- **`ragtime25-repgen`, `dragun25-repgen` — frozen cross-track test.** Run once a candidate is
  frozen; do not tune against them.

## Commands

```bash
# Validate datasets + print the dataset card (no LLM calls, safe to run any time)
uv run python scripts/prepare_eval_2025.py

# (Re)generate evaluation/splits/*.yml deterministically
uv run python scripts/prepare_eval_2025.py --make-splits

# Print the exact commands for one experiment without running anything
uv run python scripts/run_eval_2025.py --judge judges/naive/workflow.yml \
    --dataset rag25-generation --split fold-0 --dry-run

# Actually run it (naive has uses_llm: false — the only judge safe to run before an
# explicit go-ahead on model spend; see "Deferred" below for LLM-judge baselines)
uv run python scripts/run_eval_2025.py --judge judges/naive/workflow.yml \
    --dataset rag25-generation --split fold-0

# Summarize experiments under evaluation/results/
uv run python scripts/summarize_eval_2025.py
```

`run_eval_2025.py` reuses `run_all_datasets.py`'s subprocess plumbing (`auto-judge run`,
`auto-judge-evaluate meta-evaluate`) rather than reimplementing it; it adds split→`--topic`
resolution, `--runs known|held-out` (the anonymous run-ID holdout, §7 D4) →`--run` resolution, a
never-collide experiment directory (`evaluation/results/<timestamp>-<judge>-<dataset>-<split>
[-runs_<runs>][-N]/`), a `manifest.yml` per experiment recording the git commit SHA and exact
config, and (added an optional `output`/`meta_evaluate_output` param to `run_all_datasets.py`'s
`run_meta_evaluate`/`run_workflow`, default `None` so every other caller is unaffected) a
`correlations.jsonl` next to `manifest.yml` when `--meta-evaluate` is passed. `--meta-evaluate`
is skipped cleanly (not an error) for every 2025 dataset today, since none has truth yet.

## Verified so far

Ran end-to-end with `judges/naive` (no LLM calls, `uses_llm: false`) against `rag25-generation`
fold-0 and `dragun25-repgen` (`--split assessed --runs known --meta-evaluate`): produced
`tmp-naive.eval.txt`/`.eval.measures.yml`/`.config.yml`, a `manifest.yml` with commit SHA and
timings, and confirmed `--meta-evaluate` fails cleanly (a clear message, exit 0, no crash) when
truth is absent — before spending on an LLM judge.

## Deferred (not built this pass)

Per `docs/evaluation-2025-plan.md` §26 ("before launching any expensive LLM experiments"),
scope for this pass stopped at validation, splits, tests, and a dry-run-capable harness. Still
open, in roughly the order the plan recommends next:

- **Cheap LLM baselines** (`judges/testing_simple_leaderboard`, `judges/ragree`,
  `judges/ragree_choice`) — needs an explicit go-ahead on model spend (all three have
  `uses_llm: true`); use `scripts/cost_guard.py` (already in the repo) to cap it.
- **Correlation computation itself** — `--meta-evaluate` now writes `correlations.jsonl`
  machine-readably (via `auto-judge-evaluate meta-evaluate --output`), but it's an empty/failed
  file until a dataset has truth: Kendall/Spearman/tauAP-b/Kendall@k (§9) only exist once the
  bundle lands.
- **Pairwise topic-level evaluation, bootstrap CIs, SPARROW degree/adaptive-budget/reversal
  sweep, baselines/ablations grid, cost/token accounting wired into experiment manifests** —
  §§8–17 of the plan. None of this can produce a real number without truth; `scripts/
  azure_cost.py`/`bedrock_cost.py`/`cost_guard.py` already exist in the repo for the cost side
  and should be reused, not reimplemented, when this is picked up.

**Recommended next step:** get the truth bundle (see "The blocker" above), then run
`scripts/prepare_eval_2025.py` again — a `truth_status: present` there is the signal to start on
the deferred list, beginning with the cheap LLM baselines under a `cost_guard.py` budget cap.
