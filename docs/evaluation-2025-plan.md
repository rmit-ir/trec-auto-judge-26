# RAGree 2025 Evaluation Plan

## Purpose

Build a reproducible local evaluation harness for RAGree using the unrestricted 2025 pilot datasets with human ground-truth labels. Use the 2025 data for **component validation, transfer diagnostics, robustness studies, and tuning only where the 2025 and target evaluation semantics actually match**. Keep the 2026 AutoJudge datasets as blind evaluation data and do not use their restricted response content or truth for tuning.

Important correction discovered after this plan was first drafted: **TREC RAG 2025 and TREC RAG 2026 do not use the same report-quality evaluation.** The official 2025 RAG overview describes response evaluation based on generated/post-edited nuggets, `vital`/`okay` importance, strict vital-nugget recall, sub-narrative coverage, and sentence-level citation support. The public 2026 guidelines instead add system-vs-system battles and individualized nugget/rubric scoring in addition to weighted citation precision/recall. Therefore 2025 report-level ranking truth must not be treated as a direct optimization target for the 2026 Arena/ResearchRubrics-style judge.

This plan is intended to be executed by Claude Code (Sonnet) in this repository.

---

## 0. Mandatory constraints before doing any work

1. Read `AGENTS.md` and `CLAUDE.md` in full.
2. Read `.claude/skills/autojudge-develop/SKILL.md` in full.
3. Read the canonical TREC AutoJudge data-handling policy before touching evaluation data.
4. Treat these datasets as unrestricted development data, per the repository policy:
   - `rag25`
   - `ragtime25`
   - `dragun25`
   - `kiddie`
5. Do not inspect, summarize, fingerprint, compare, or tune on restricted 2026 response content beyond what the data policy explicitly permits.
6. Do not attempt to map anonymized animal-style run IDs to teams, systems, institutions, or model families.
7. Use `uv` for the Python environment and dependency execution. The project environment should remain under the existing development setup; do not switch the project to ad-hoc `pip` workflows.
8. Do not hardcode API credentials, endpoints, or model IDs. Continue using the injected `llm_config` contract.

---

# 1. Current known data state

The repository currently contains locally fetched 2025 pilot inputs under:

- `local-data/rag25/`
- `local-data/ragtime25/`
- `local-data/dragun25/`

The local manifests and statistics confirm the following datasets:

## RAG 2025 generation

- responses: `local-data/rag25/runs/generation`
- topics: `local-data/rag25/topics/trec_rag_2025_requests.jsonl`
- approximately 51 response runs
- 105 topics in the run collection
- assessed-topic subset listed in `local-data/rag25/datasets.yml`
- expected official truth file:
  `rag.generation.official.eval.jsonl`

## RAG 2025 augmented generation

- responses: `local-data/rag25/runs/auggen`
- topics: `local-data/rag25/topics/trec_rag_2025_requests.jsonl`
- approximately 25 response runs
- expected official truth file:
  `rag.auggen.official.eval.jsonl`

## RAGTIME 2025 response generation

- responses: `local-data/ragtime25/runs/repgen`
- topics: `local-data/ragtime25/topics/ragtime25_main_all.jsonl`
- approximately 59 response runs
- expected official truth file:
  `ragtime.repgen.official.eval.jsonl`

## DRAGUN 2025 response generation

- responses: `local-data/dragun25/runs/repgen`
- topics: `local-data/dragun25/topics/trec-2025-dragun-requests-generated.jsonl`
- approximately 28 response runs
- expected official truth file:
  `dragun.repgen.official.eval.jsonl`

The checked-in dataset README files and AutoJudge tooling identify these official truth filenames. However, at the time this plan was written, those official truth files were not present in the fetched `local-data/*/eval/` locations.

Important: do not reconstruct anonymized run identities from public TREC data. Prefer the organizer-provided anonymized truth bundle that matches the local anonymous run IDs.

---

# 2. Goal

Create a local benchmark that answers four different questions separately:

1. **2025 transfer ranking:** Does a judge correlate with the 2025 human/official measures it is actually intended to approximate (nugget/sub-narrative coverage and support)? Treat this as a transfer diagnostic, not a proxy for the 2026 Arena objective.
2. **Pairwise algorithm robustness:** Given a fixed set of ballots or a generic pairwise judge, how stable and efficient are SPARROW/Bradley-Terry ranking mechanics? Do not label 2025 score-induced pairs as "2026 human preference".
3. **Evidence quality:** Does the pointwise evidence layer track human/reference measures such as strict vital nugget recall, sub-narrative coverage, citation support, and related 2025 assessment signals?
4. **Efficiency:** How much ranking quality is retained by sparse SPARROW acquisition compared with exhaustive pairwise comparison, and at what cost?

Do not collapse these into a single score, and do not use 2025 ranking correlation to select 2026-specific rubric/Arena semantics unless an experiment explicitly targets a metric shared across years.

---

# 3. Desired repository structure

Create the following, unless an equivalent structure already exists and is clearly preferable:

```text
evaluation/
├── README.md
├── datasets-2025.yml
├── splits/
│   ├── rag25-generation.yml
│   ├── rag25-auggen.yml
│   ├── ragtime25-repgen.yml
│   └── dragun25-repgen.yml
└── results/
    └── .gitignore

scripts/
├── prepare_eval_2025.py
├── run_eval_2025.py
└── summarize_eval_2025.py
```

If the existing project structure suggests a better naming convention, preserve the intent but keep the responsibilities separated.

Do not copy large response datasets into `evaluation/`. Reference files under `local-data/`.

Do not commit human-truth files if the repository policy or release terms imply they should remain local.

---

# 4. Phase A — inspect and recover the 2025 truth bundle

## A1. Inspect local manifests and existing files

Check:

```text
local-data/rag25/datasets.yml
local-data/rag25/stats.txt
local-data/ragtime25/datasets.yml
local-data/ragtime25/stats.txt
local-data/dragun25/datasets.yml
local-data/dragun25/stats.txt
```

Also inspect the relevant `data/*/README.md` files.

Do not infer truth paths from memory if the manifests provide them.

## A2. Check whether official truth exists elsewhere locally

Search filenames only for:

```text
rag.generation.official.eval.jsonl
rag.auggen.official.eval.jsonl
ragtime.repgen.official.eval.jsonl
dragun.repgen.official.eval.jsonl
```

If found, verify that they correspond to the anonymized 2025 runs before wiring them in.

## A3. If truth is missing

Do not fabricate it and do not reverse-map public run identities.

Instead:

1. inspect the existing fetch/release mechanism (`fetch_datasets.py`, release metadata, documentation);
2. determine whether the pilot release has a truth/download option that was not fetched;
3. document the exact missing artifact and expected destination;
4. if retrieval requires credentials or a user-only action, stop that subtask and report the concrete command/action the user must perform;
5. continue implementing the harness so it fails cleanly with a useful message until truth is present.

Expected destinations should ideally be:

```text
local-data/rag25/eval/rag.generation.official.eval.jsonl
local-data/rag25/eval/rag.auggen.official.eval.jsonl
local-data/ragtime25/eval/ragtime.repgen.official.eval.jsonl
local-data/dragun25/eval/dragun.repgen.official.eval.jsonl
```

Only use different paths if the official release layout dictates them.

---

# 5. Phase B — add a canonical 2025 evaluation manifest

Create `evaluation/datasets-2025.yml`.

Each dataset entry should include at minimum:

```yaml
name:
responses:
topics:
truth:
truth_format: jsonl
assessed_topics:
priority_runs:
role:
```

Recommended roles:

```text
rag25-generation   -> development
rag25-auggen       -> validation
ragtime25-repgen   -> frozen-cross-track-test
dragun25-repgen    -> frozen-cross-track-test
```

Also include enough metadata for reproducibility:

```yaml
track:
year: 2025
task:
```

Do not manually retype long assessed-topic lists if the source manifest can be loaded programmatically. Prefer a small manifest that references the authoritative release manifest where practical.

---

# 6. Phase C — build `prepare_eval_2025.py`

This script should perform validation and dataset preparation only. It must not call an LLM.

## Required checks

For each dataset:

1. responses directory exists;
2. topics file exists;
3. truth file exists;
4. truth parses in the declared format;
5. response run IDs overlap truth run IDs;
6. assessed topic IDs overlap topic IDs;
7. truth contains no duplicate `(run, topic, measure)` cells unless the format explicitly permits them;
8. truth aggregate rows are valid or can be checked with existing AutoJudge tooling;
9. no output artifact accidentally copies response text;
10. run/topic counts are reported.

Use `auto-judge-evaluate` wherever it already provides the required parsing/verification rather than implementing a parallel truth parser unnecessarily.

The script should print a compact dataset card such as:

```text
Dataset                Runs  Topics  Assessed  Truth measures
rag25-generation         51     105        22  ...
rag25-auggen              25     105        22  ...
ragtime25-repgen          59     122        16  ...
dragun25-repgen           28      30        30  ...
```

The exact counts should come from the files, not hardcoded values.

## Important discrepancy to investigate

The existing RAGTIME metadata appeared to list more priority run IDs in `datasets.yml` than the `stats.txt` summary reported for `repgen-prio1`. Resolve this deterministically and document the reason before using priority-run subsets in experiments.

---

# 7. Phase D — deterministic splits

Do not use one ad-hoc random train/test split.

## D1. Topic folds for RAG25 Generation

Use the assessed-topic subset only.

Create approximately 4 deterministic folds by topic. Keep all runs for a topic in the same fold.

Use a stable seed and deterministic sort order. Record the seed and algorithm in the split YAML.

Purpose:

- inner development
- threshold tuning
- prompt changes
- algorithm settings
- ablations

## D2. RAG25 AugGen as validation environment

Do not merge AugGen into the same tuning pool by default.

Use it as a distribution-shift validation set after candidate methods have been selected on RAG25 Generation.

## D3. Frozen cross-track test

Treat:

- RAGTIME25 RepGen
- DRAGUN25 RepGen

as frozen cross-track test datasets for local evaluation.

Once the first baseline results are produced, do not tune hyperparameters specifically to improve these tracks unless the experimental phase is explicitly reset and the distinction is documented.

## D4. Anonymous run holdout

Because run identities are anonymized and must not be reverse-mapped, do not create model-family splits.

Instead create a deterministic anonymous run-ID holdout, e.g. 80/20, using only anonymous IDs.

Use this to create four generalization quadrants:

```text
known topics     x known runs
held-out topics  x known runs
known topics     x held-out runs
held-out topics  x held-out runs
```

The last quadrant is the strongest local generalization test.

Do not claim that anonymous run holdout is equivalent to a generator-family holdout.

---

# 8. Phase E — define evaluation targets before running experiments

Do not select whichever truth measure gives the best result after the fact.

Create an explicit mapping configuration in the evaluation manifest or a separate evaluation config.

## RAGree Choice outputs

Current relevant outputs include:

```text
CHOICE_SCORE
EVIDENCE_ELIGIBLE
EVIDENCE_EVALUATED
POINTWISE_EVIDENCE_SCORE
PAIRWISE_WIN_RATE
CONDORCET_WINNER
RANKING_STABLE
CYCLE_RATE
BALLOT_COVERAGE
NO_ACCEPTABLE_ANSWER
POSITION_REVERSAL_CHECKED
POSITION_REVERSAL_DISAGREEMENT_RATE
```

## Classify them

### Primary ranking outputs

- `CHOICE_SCORE`
- `PAIRWISE_WIN_RATE`

### Evidence outputs

- `POINTWISE_EVIDENCE_SCORE`
- `EVIDENCE_ELIGIBLE`

### Robustness/diagnostic outputs

- `CONDORCET_WINNER`
- `RANKING_STABLE`
- `CYCLE_RATE`
- `POSITION_REVERSAL_CHECKED`
- `POSITION_REVERSAL_DISAGREEMENT_RATE`
- `BALLOT_COVERAGE`
- `NO_ACCEPTABLE_ANSWER`

Do not optimize directly against diagnostic measures as if they were human quality labels.

---

# 9. Phase F — system-level meta-evaluation

Use `auto-judge-evaluate meta-evaluate` as the authoritative implementation where possible.

For each judge output measure against each pre-registered human truth measure, compute:

Primary:

- Kendall tau-b or the AutoJudge equivalent `kendall`

Secondary:

- Spearman rho
- `tauap_b`
- Kendall@5 where applicable
- Kendall@15 where applicable
- Pearson as a score-magnitude diagnostic, not the headline metric

Store machine-readable results as JSONL.

Recommended directory pattern:

```text
evaluation/results/<timestamp-or-run-id>/
  manifest.yml
  per-dataset/
  correlations.jsonl
  summary.json
```

Do not rely only on terminal output.

---

# 10. Phase G — topic-level pairwise evaluation

Implement pairwise evaluation because SPARROW is fundamentally a preference-ranking method.

For each assessed topic and pair of systems `(A, B)`:

1. derive the human ordering from the selected official truth measure;
2. derive the RAGree ordering from `CHOICE_SCORE` or observed pairwise result;
3. compare the two.

Predefine tie handling before comparing variants.

Possible rule:

- exact tie when the human metric is exactly equal;
- optional tolerance only if justified by the metric scale and declared in config.

Report:

- directional pairwise accuracy;
- balanced accuracy;
- tie precision;
- tie recall;
- tie F1;
- confusion matrix;
- pairwise coverage;
- 95% confidence intervals via topic-level bootstrap.

Do not bootstrap individual pairs as though they were independent; resample topics.

---

# 11. Phase H — evidence-layer evaluation

Evaluate the pointwise evidence lane independently of the final preference ranking.

For RAGTIME especially, inspect available official truth measures and pre-register sensible mappings, including measures such as those already referenced by AutoJudge evaluation tooling:

```text
nugget_coverage
correct_nuggets
f1
supporting_citations
citation_support
relevant_citations
sentence_support
correctly_cited_sentences
```

Potential comparisons:

```text
POINTWISE_EVIDENCE_SCORE <-> nugget_coverage
POINTWISE_EVIDENCE_SCORE <-> citation_support
POINTWISE_EVIDENCE_SCORE <-> sentence_support
POINTWISE_EVIDENCE_SCORE <-> correct_nuggets
```

Use:

- Kendall/Spearman for ordinal or continuous labels;
- macro-F1 for legitimate categorical labels;
- AUROC/AUPRC only when a real binary target exists;
- calibration metrics only for outputs intended to be probabilistic.

Do not invent binary labels merely to report AUROC.

---

# 12. Phase I — SPARROW algorithm evaluation

The purpose is to separate LLM-judge quality from sparse-comparison/ranking quality.

## Reference condition

Use the existing `exhaustive-small` variant, or an equivalent setting, to obtain full round-robin comparison when computationally practical:

```yaml
full_round_robin_max_candidates: 999
adaptive_comparisons_per_topic: 0
```

Reuse the same cached LLM ballots whenever technically possible so that sparse-vs-exhaustive comparisons are not confounded by model sampling.

## Sparse configurations

At minimum evaluate a small grid such as:

```text
initial_graph_degree:             2, 3, 4, 6
adaptive_comparisons_per_topic:  0, 2, 4, 8
position_reversal_budget:        0, 1, 2, 4
```

Do not run the full Cartesian product blindly if cost is excessive. Use a staged design:

1. degree sweep;
2. adaptive-budget sweep around the best Pareto region;
3. reversal-budget sweep around the resulting candidate.

## Report

For sparse vs exhaustive:

- winner agreement;
- ranking Kendall tau;
- top-3 set overlap where applicable;
- human-ground-truth correlation;
- pairwise calls/topic;
- fraction of all possible pairs evaluated;
- reversal calls/topic;
- wall-clock time;
- estimated token use;
- estimated monetary cost.

The key result should be a Pareto curve of quality versus cost/comparisons.

---

# 13. Phase J — baselines and ablations

Run all feasible baselines under identical dataset/split conditions.

Minimum baselines:

1. `judges/naive`
2. `judges/testing_simple_leaderboard`
3. `judges/ragree`
4. `judges/ragree_choice`
5. `ragree_choice` exhaustive condition

Add ablations for RAGree Choice where implementation makes this clean:

6. pointwise evidence only
7. pairwise preference without evidence gate
8. evidence gate + pairwise preference without adaptive sparse acquisition
9. sparse SPARROW without position reversal
10. full SPARROW with position reversal

If an ablation would require risky invasive code changes, first add a workflow setting/variant rather than forking large code paths.

Keep all tunable settings in workflow YAML, not hardcoded in Python.

---

# 14. Phase K — hyperparameter discipline

Separate settings into two groups.

## Quality/calibration parameters

Examples:

```text
factual_support_threshold
citation_validity_threshold
```

Tune on RAG25 Generation **only when the parameter controls a component whose target semantics match 2025 truth**, especially citation support, nugget/coverage behavior, generic failure handling, or calibration against those measures.

Do **not** tune the following primarily against 2025 report-level rankings:

```text
ResearchRubrics axis composition
ResearchRubrics criterion wording/weighting
rubric-conditioned Arena prompt semantics
2026 system-battle tie policy
2026 combined Arena/rubric weighting
```

Those components must instead be developed from the public 2026 specification, organizer-provided ResearchRubrics development artifacts, the ResearchRubrics benchmark itself, and other directly relevant preference data where available. RAG25 may be used afterward as an out-of-objective transfer/sanity check.

## Compute/algorithm parameters

Examples:

```text
initial_graph_degree
adaptive_comparisons_per_topic
position_reversal_budget
```

Choose from a Pareto frontier balancing:

```text
human agreement higher
cost lower
latency lower
```

Do not pick a substantially more expensive configuration for a negligible, statistically uncertain correlation increase.

Record every selected value and the data split used to select it.

---

# 15. Phase L — statistical uncertainty

All headline scores must include uncertainty estimates.

Use topic-level bootstrap, recommended 5,000 resamples unless runtime is excessive.

For a metric `M`, report:

```text
M = estimate
95% CI [lower, upper]
```

For comparing variants A and B, bootstrap the difference:

```text
Delta M = M(A) - M(B)
95% CI [lower, upper]
```

Do not infer significance from point estimates alone.

Store bootstrap seed and resampling settings in the result manifest.

---

# 16. Phase M — robustness evaluation

Add explicit robustness experiments.

## Position bias

Compare:

- reversal disabled;
- configured reversal budget;
- higher reversal budget where affordable.

Report final-ranking changes as well as `POSITION_REVERSAL_DISAGREEMENT_RATE`.

## Candidate-order invariance

Verify that deterministic reordering of candidate input does not change final ranking except for genuine model nondeterminism.

The implementation already sorts response comparisons by run ID where required; test this explicitly.

## Prompt robustness

For prompts implementing **2025-shared semantics** (citation support, nugget assignment, evidence checks), use 2025 development folds to test a small set of semantically equivalent prompts.

For **2026-specific Arena/rubric prompts**, do not select wording by maximizing RAG25 report-level correlation. Prefer public RAGDoll/organizer prompt semantics and ResearchRubrics-based validation. RAG25 can be reported as a transfer diagnostic only.

Do not tune prompt wording against frozen cross-track test data.

## Failure handling

Exercise:

- malformed LLM JSON;
- explicit `ABSTAIN`;
- timeout/provider failure;
- one unavailable evidence brief;
- empty report.

Where possible, make these unit/integration tests rather than expensive LLM experiments.

---

# 17. Phase N — cost and resource accounting

The repository already includes Azure/Bedrock cost tools. Extend the evaluation harness to record, where available:

```text
LLM request count
pointwise evidence call count
pairwise ballot count
position-reversal call count
input tokens
output tokens
cache hits
cache misses
elapsed wall time
estimated cost
```

Do not log prompt or response content from restricted 2026 datasets.

For 2025 unrestricted data, content logging is technically allowed by the data policy, but evaluation infrastructure should still default to content-free structured logs for reproducibility and safety.

Suggested aggregate table:

```text
method | dataset | Kendall | pairwise accuracy | calls/topic | tokens/topic | cost/topic | wall time
```

---

# 18. Phase O — primary endpoint and reporting discipline

Before comparing experimental results, define the endpoint **per component**, because there is no single 2025 measure that reproduces the 2026 evaluation.

Recommended 2025 primary endpoints:

- for citation-support components: correlation/agreement with the official 2025 support measures;
- for nugget/coverage components: correlation with strict vital-nugget recall / sub-narrative coverage or the designated official 2025 response measure;
- for SPARROW mechanics: agreement with an exhaustive ranking produced from the **same ballot source**, plus call/cost reduction;
- for generic report scores: report system-level Kendall as a transfer diagnostic, not as the selection objective for the 2026 Arena/rubric judge.

There should be **no claim** that maximizing `CHOICE_SCORE` against 2025 truth optimizes 2026 system-battle accuracy.

Do not choose the human truth measure after seeing which one gives the largest correlation.

Secondary endpoints:

- Spearman rho
- tauAP-b
- top-k Kendall
- pairwise agreement
- evidence correlations
- citation metrics
- nugget metrics
- position robustness
- efficiency/cost

The final report must show per-track results, not only a pooled number.

---

# 19. Phase P — implementation of `run_eval_2025.py`

The runner should orchestrate experiments, not duplicate AutoJudge internals.

Prefer subprocess calls to the existing commands:

```text
auto-judge run
auto-judge-evaluate meta-evaluate
```

Use existing `run_all_datasets.py` functionality where it cleanly fits. Refactor reusable pieces rather than maintaining two divergent dataset runners if practical.

Required runner behavior:

1. choose dataset(s);
2. choose judge workflow;
3. choose variant;
4. choose split/fold;
5. create a unique output directory;
6. snapshot config and Git commit SHA;
7. run AutoJudge;
8. run meta-evaluation;
9. collect runtime/cost metadata;
10. write one machine-readable experiment manifest;
11. never overwrite a previous result silently.

Useful CLI shape:

```bash
uv run python scripts/run_eval_2025.py \
  --judge judges/ragree_choice/workflow.yml \
  --dataset rag25-generation \
  --split fold-0 \
  --variant default
```

Add a `--dry-run` mode that prints exact commands without executing LLM calls.

---

# 20. Phase Q — implementation of `summarize_eval_2025.py`

This script should read experiment result artifacts and produce:

1. a concise console table;
2. JSON/JSONL summaries;
3. CSV or TSV suitable for plotting;
4. optional Markdown summary.

Do not make plotting a prerequisite for the first implementation.

Minimum fields:

```text
experiment_id
commit_sha
dataset
split
judge
variant
model/endpoint identifier if safely available
truth_measure
eval_measure
correlation
estimate
ci_low
ci_high
pairwise_accuracy
llm_calls
tokens
estimated_cost
wall_time
```

---

# 21. Tests

Add tests before relying on the harness for expensive experiments.

Minimum test coverage:

## Dataset manifest

- paths resolve correctly;
- missing truth produces a clear error;
- assessed-topic filtering is deterministic;
- split generation is deterministic.

## Truth handling

- duplicate truth cells detected;
- missing runs/topics reported;
- legitimate aggregate rows handled correctly.

## Pairwise metrics

- A>B, B>A, tie cases;
- tie tolerance behavior;
- missing values;
- topic-level bootstrap reproducibility.

## Result handling

- previous experiment directories are not overwritten;
- experiment manifest contains commit SHA and config;
- JSONL output round-trips.

## Existing project tests

Run the full suite with the project's declared extras via `uv`.

The previous healthy baseline was:

```text
57 passed, 2 xfailed
```

Do not assume that count must remain identical if legitimate tests are added; require that all non-xfail tests pass.

---

# 22. First experiment sequence

Once truth is available and validation passes, execute experiments in this order.

## Experiment 0 — harness smoke test

Use `kiddie` or one tiny 2025 subset to verify the full pipeline without spending significant model budget.

## Experiment 1 — cheap baselines

Run:

- naive;
- simple leaderboard relevance judge where an LLM endpoint is available.

Confirm meta-evaluation outputs and result summaries.

## Experiment 2 — RAGree evidence judge

Run original `ragree` on RAG25 Generation development folds.

Measure evidence-target correlations.

## Experiment 3 — RAGree Choice transfer diagnostic

Run default `ragree_choice` on RAG25 Generation development folds.

Measure:

- system-level correlation with 2025 truth as a **transfer diagnostic**;
- score-induced pairwise ordering agreement, clearly labelled as ordering agreement rather than human preference accuracy;
- evidence/support correlations;
- cost.

Do not select 2026 Arena/rubric prompt semantics because this experiment scores well or poorly: the 2025 response-evaluation objective is different.

## Experiment 4 — exhaustive reference

Run full round-robin on a manageable subset, using cached ballots where possible.

Compare sparse vs exhaustive.

## Experiment 5 — sparse-efficiency sweep

Evaluate graph degree and adaptive-comparison budgets.

Select a Pareto region, not simply the maximum point estimate.

## Experiment 6 — position reversal

Evaluate reversal budget around the selected sparse configuration.

## Experiment 7 — 2025 validation / transfer check

Freeze **2025-shared components** and run on RAG25 AugGen.

For 2026-specific Arena/rubric components, treat this run as a transfer check only and do not retune those components to improve AugGen correlation.

## Experiment 8 — frozen cross-track test

Run the frozen candidate on:

- RAGTIME25 RepGen;
- DRAGUN25 RepGen.

Produce the final local 2025 evaluation report.

---

# 23. 2026 boundary

The phrase "2025 method is frozen" applies only to components legitimately developed against 2025 labels. The full 2026 specialist judge should instead be frozen after combining:

- 2025-validated shared components (especially citation support / coverage mechanics);
- ResearchRubrics structural and functional validation;
- public RAGDoll/2026 Arena semantics;
- search-index-grounded rubric generation developed without candidate responses.

Then:

- 2026 data may be used according to its policy for execution correctness and official evaluation;
- do not inspect restricted response content;
- do not use restricted 2026 topic-level outputs to tune prompts, thresholds, variants, or ranking algorithms;
- do not choose a new method because it looked better on restricted test data;
- treat official 2026 evaluation as the blind external result.

---

# 24. Deliverables

The implementation is complete when all of the following exist and work:

1. `evaluation/README.md`
2. `evaluation/datasets-2025.yml`
3. deterministic split files under `evaluation/splits/`
4. `scripts/prepare_eval_2025.py`
5. `scripts/run_eval_2025.py`
6. `scripts/summarize_eval_2025.py`
7. tests for manifests, splits, metrics, bootstrap, and result handling
8. documented instructions for obtaining the missing official 2025 truth bundle if it is not already available
9. at least one successful smoke evaluation end-to-end
10. baseline 2025 results if truth and an LLM endpoint are available
11. machine-readable experiment manifests
12. cost/resource metadata in the experiment outputs
13. all existing and new tests passing

---

# 25. Acceptance criteria

Do not consider the task complete merely because scripts exist.

The evaluation framework must satisfy all of these:

- reproducible from a clean checkout plus local 2025 release data;
- uses `uv`;
- does not duplicate or fork AutoJudge functionality unnecessarily;
- never requires deanonymizing runs;
- keeps 2026 restricted data out of tuning;
- fails clearly when truth is missing;
- deterministic splits;
- no silent overwrite of experiments;
- every experiment records Git SHA and exact judge configuration;
- component-specific primary metrics are pre-registered;
- 2025 metrics are never presented as equivalent to the 2026 pairwise/Rubric evaluation when they are not;
- confidence intervals are produced for headline metrics;
- sparse SPARROW can be compared fairly with exhaustive ranking;
- quality/cost tradeoff can be quantified;
- baseline and ablation comparisons are automated;
- test suite remains green.

---

# 26. How Claude Code should proceed

Work incrementally and keep commits logically separated.

Recommended order:

1. inspect current local 2025 data and truth availability;
2. report findings before large changes;
3. create manifest + README;
4. implement validation/preparation script;
5. implement deterministic splits;
6. add tests;
7. implement runner dry-run mode;
8. implement real runner;
9. implement summary/bootstrap metrics;
10. run smoke tests;
11. if truth is available, run cheap baseline evaluation;
12. report the next experimental step and estimated model-call scope before launching expensive RAGree sweeps;
13. explicitly label each proposed sweep as either `2025-shared-component tuning` or `2026-specific transfer diagnostic` before spending model budget.

Do not launch a large paid LLM sweep without first verifying the dry-run command set, cache configuration, expected number of calls, and cost guard.

When blocked by missing truth or credentials, continue all work that does not require them and leave an exact actionable blocker instead of stopping the whole implementation.
