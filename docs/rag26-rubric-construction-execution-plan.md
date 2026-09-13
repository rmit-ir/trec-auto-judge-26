# RAG26 Rubric Construction — Execution Plan

**Prepared:** 2026-09-13  
**Purpose:** build and freeze the response-independent ResearchRubrics-style rubric artifact for all 119 TREC RAG 2026 topics, ready for `RAG26ArenaJudge` integration and TIRA packaging.

This document is the operational plan for rubric construction. It narrows and updates the broader design in:

- `docs/trec-rag-2026-researchrubrics-generation-plan.md`
- `docs/rag26-tira-submission-plan.md`

It does **not** replace their methodological/data-handling rules. Where they conflict on implementation order, this execution plan takes precedence for the rubric-construction workstream.

---

# 1. Current verified state

Latest relevant commits:

```text
8cfee44 Rubric construction: Stage B/C -- ClimbMix retrieval + evidence condensation
8001ece Rubric construction: typed models, response-independent cache, Bedrock/Luna adapters, Stage A planner
356c5ac Rubric-construction preflight: stale README, Bedrock Flex pricing, targeted smoke tool
1b5fb0a Add RAGDoll-compatible citation-support lane to RAG26ArenaJudge
15cbe1e Add RAG26ArenaJudge: arena-only pairwise preference judge (SPARROW-ranked)
```

Current full test status:

```text
236 passed, 2 xfailed
```

Rubric-construction status:

```text
Stage A: topic characterization + research-intent planning       DONE / live-proven
Stage B: dense+sparse ClimbMix retrieval + RRF                  DONE / live-proven
Stage C: evidence condensation                                  DONE / live-proven
Next:    axis-specific criterion generation                     IN PROGRESS
```

Current cumulative paid rubric-construction spend is approximately `$0.0024`, confirming that model quality and pipeline reliability matter much more than shaving tiny fractions of a cent from individual construction stages.

Current RAG26 runtime judge already has:

- `judges/rag26_arena/judge.py`
- `judges/rag26_arena/prompts.py`
- `judges/rag26_arena/models.py`
- `judges/rag26_arena/support.py`
- `judges/rag26_arena/workflow.yml`
- Arena/SPARROW ranking
- deterministic candidate ordering
- retry/ABSTAIN semantics
- position-reversal diagnostics
- independent citation-support lane
- `arena-only` and `arena-support` variants
- deterministic cache replay verified on kiddie

The TIRA Docker dry-run is delegated to the Docker-capable-machine handoff and is not a blocker for continuing response-independent rubric construction on this machine.

The current `judges/rag26_arena/README.md` is stale and still describes the package as foundation-only. Update it before or alongside rubric implementation so coding agents do not follow obsolete status information.

---

# 2. Non-negotiable data boundary

Rubric construction must remain **100% response-independent**.

Allowed inputs:

```text
public RAG26 Request/topic narrative
public TREC RAG 2026 instructions
ClimbMix corpus/search results
ResearchRubrics public development topics/rubrics
synthetic fixtures
unrestricted 2025 pilot material where semantics genuinely transfer
```

Forbidden inputs:

```text
candidate RAG26 answer text
candidate citations
candidate response-derived claims
candidate retrieval traces
candidate prompt caches
candidate scores/leaderboard behavior used to choose rubric content
```

The rubric generator must be frozen before production generation of the 119 RAG26 topic rubrics. No manual per-topic editing based on candidate system outputs.

Read `AGENTS.md` / `CLAUDE.md` before touching evaluation data.

---

# 3. Model decision

## 3.1 Primary model: Qwen3 235B A22B 2507 on Amazon Bedrock Flex

Use this as the **default semantic model for rubric construction**:

```text
provider: Amazon Bedrock
model: Qwen3 235B A22B 2507
Bedrock Runtime model ID: qwen.qwen3-235b-a22b-2507-v1:0
region: ap-southeast-2 (Sydney)
service tier: flex
```

Why this model:

- reasoning support;
- structured-output support;
- very large context window, sufficient for a full topic research pack plus rubric candidates;
- materially stronger capacity than the 32B model while remaining extremely cheap;
- available in Sydney;
- Flex is explicitly intended for non-latency-sensitive evaluation/content-processing workloads;
- current Sydney Flex pricing snapshot (2026-09-13):

```text
input:  $0.1133 / 1M tokens
output: $0.4532 / 1M tokens
```

Standard fallback for the same model:

```text
input:  $0.2266 / 1M tokens
output: $0.9064 / 1M tokens
```

Use Standard automatically only when Flex is unavailable/throttled beyond the bounded retry policy. Do not silently switch model family.

The AWS model card currently marks the model Active and supports Bedrock Runtime / Converse and structured outputs. Re-check availability immediately before the production run.

## 3.2 V1 simplification: keep Qwen3 235B for construction + internal critique

Do **not** introduce Qwen3 32B into the first production pipeline yet.

The live Stage A/C probes show Qwen3 235B Flex is so inexpensive that the savings from switching mechanical stages to 32B are negligible relative to the engineering and behavioural complexity of adding another model deployment.

For V1, use the same Qwen3 235B Flex model for:

- topic characterization;
- research-intent planning;
- evidence condensation;
- explicit/implicit/synthesis/negative generation;
- atomicity/overlap critique;
- ordinal importance classification;
- bounded Qwen repair after the independent audit.

Benefits:

- one Bedrock deployment/model contract to debug;
- one response-independent cache namespace/model behaviour;
- stronger reasoning on every semantic stage;
- easier reproducibility;
- essentially no meaningful budget penalty at current Flex pricing.

After B0/B1/B2 development evaluation is complete, a **cost-ablation only** may test Qwen3 32B on atomicity/weighting. Adopt it only if semantic metrics and failure rates are statistically indistinguishable from 235B. Do not make that ablation part of the critical path.

## 3.3 Independent final reviewer: GPT-5.6 Luna

Use one independent cross-family review pass per accepted topic rubric.

Preferred endpoint: existing Azure OpenAI-compatible deployment if available.

Current Global Standard pricing snapshot:

```text
input:  $0.20 / 1M
output: $1.20 / 1M
```

Bedrock Global inference is an acceptable fallback at the same public base pricing if Azure is unavailable.

Why use Luna at all:

- independent model family reduces correlated self-review errors;
- final-review prompts are small relative to evidence-generation prompts;
- total cost is negligible;
- ResearchRubrics explicitly warns that naive LLM-generated/rephrased criteria can drift; an independent critic is worth the small cost.

Do not use Luna as the main generator unless Qwen3 235B fails the development gate.

## 3.4 Escalation model

Do **not** put GPT-5.6 Terra/Sol in the normal pipeline.

Only allow a selective Terra escalation when:

- Qwen3 235B and Luna disagree on a mandatory criterion;
- a topic fails final validation twice;
- a reviewer marks unresolved factual/semantic contradiction.

Require explicit cost approval before enabling this escalation globally.

---

# 4. Cost envelope

The main cost risk is workflow complexity, not model tokens.

Use this conservative per-topic planning envelope:

```text
Qwen3 235B Flex:
  ~154k input tokens/topic
  ~23.5k output tokens/topic
  # includes generation plus atomicity/importance stages that were previously
  # budgeted to a smaller helper model

GPT-5.6 Luna:
  ~20k input tokens/topic
  ~3k output tokens/topic
```

At current pricing, for 119 production topics this is approximately:

```text
Qwen3 235B Flex: ~$3.34
Luna audit:      ~$0.90
--------------------------------
base estimate:   ~$4.24
```

The observed live spend so far is even lower than this conservative envelope.

Allow 50% overhead for retries/repair:

```text
expected production rubric-generation budget: <$7
soft cap:  $10
hard cap:  $20
```

The 30-topic ResearchRubrics development calibration should add only a few dollars.

Set a total rubric-workstream hard cap of **$25** before any optional Terra escalation.

Before each paid batch, print:

```text
model
service tier
number of topics
planned stage(s)
estimated input/output token ceiling
estimated USD ceiling
cache status
```

No large paid call sequence without `scripts/cost_guard.py` or equivalent explicit budget enforcement.

---

# 5. First implementation task — model/config preflight

Before building the generator:

1. refresh the local Bedrock pricing snapshot;
2. add Qwen3 235B A22B 2507 Flex to the machine-readable pricing table; keep Qwen3 32B as an optional later ablation, not a V1 dependency;
3. update `scripts/bedrock_cost.py` tests as needed;
4. extend/create a targeted model smoke command that can test one selected model instead of spending on the entire historic model list;
5. make one minimal Qwen3 235B Flex structured-output request;
6. verify:
   - credentials;
   - Sydney access;
   - Flex accepted;
   - model ID correct;
   - JSON/structured output works;
   - input/output token usage returned;
   - cost calculator agrees with current price table.

Do not begin a 30-topic or 119-topic run until this passes.

Fallback order:

```text
Qwen3 235B Flex Sydney
-> Qwen3 235B Standard Sydney
-> stop and report
```

Do not silently fall back to an unrelated model.

---

# 6. Package layout to implement

Reuse existing:

```text
judges/rag26_arena/rubrics/
    schema.py
    weights.py
    validate.py
    io.py
    stats.py
    topics.py
    research.py
```

Add focused modules rather than one monolith:

```text
judges/rag26_arena/rubrics/
    generation_models.py   # internal typed generation/provenance models
    llm.py                 # Bedrock/Azure offline generation adapters + usage records
    prompts.py             # versioned prompt builders
    planner.py             # task characterization + ResearchIntent generation
    evidence.py            # research-pack condensation/evidence notes
    generate.py            # axis generators + orchestration
    normalize.py           # atomicity/dedupe/contradiction normalization
    review.py              # weighting + final independent review/repair
    cache.py               # response-independent disk cache
    evaluate.py            # dev-set structural/semantic fidelity
```

CLI/scripts:

```text
scripts/generate_rag26_rubrics.py
scripts/evaluate_rag26_rubric_generator.py
scripts/smoke_rag26_rubric_model.py
```

Do not make the runtime `RAG26ArenaJudge` import boto3/Azure-specific generator code during normal TIRA execution. Offline construction and runtime judging remain separate concerns.

---

# 7. Internal typed objects

Add typed internal objects that never leak into the organizer-compatible final JSONL.

## Topic characterization

```python
TopicCharacterization:
    qid
    domain
    conceptual_breadth
    logical_nesting
    exploration
    explicit_subquestions
    named_constraints
    synthesis_goal
```

## Research intent

Reuse/extend existing `ResearchIntent`:

```python
ResearchIntent:
    query
    retrieval_mode: dense | sparse | hybrid
    purpose
```

## Evidence note

```python
EvidenceNote:
    evidence_id
    claim
    docids
    subtopic
    evidence_type: fact | number | argument | counterargument | context
    confidence
    independent_sources
```

## Rubric candidate

```python
RubricCandidate:
    candidate_id
    criterion
    axis
    polarity
    importance_class | None
    strong_importance | None
    narrative_support
    evidence_claim_ids
    rationale
    example | None
    confidence
    generator_stage
```

## Review decision

```python
ReviewDecision:
    candidate_id
    verdict: KEEP | REVISE | DROP | SPLIT | MERGE
    reason_code
    replacement_text | None
    replacement_axis | None
```

The final compatibility artifact remains exactly:

```text
qid
domain
conceptual_breadth
logical_nesting
exploration
rubrics[].criterion
rubrics[].weight
rubrics[].axis
```

---

# 8. Prompt versioning

Every prompt builder must expose a stable version constant, e.g.:

```text
CHARACTERIZE_PROMPT_VERSION = "v1"
EVIDENCE_PROMPT_VERSION = "v1"
EXPLICIT_PROMPT_VERSION = "v1"
IMPLICIT_PROMPT_VERSION = "v1"
SYNTHESIS_PROMPT_VERSION = "v1"
NEGATIVE_PROMPT_VERSION = "v1"
NORMALIZE_PROMPT_VERSION = "v1"
WEIGHT_PROMPT_VERSION = "v1"
FINAL_REVIEW_PROMPT_VERSION = "v1"
```

Prompt hashes + versions go into provenance/manifest.

No timestamps, random IDs, unordered sets, or nondeterministic serialization in prompts/cache keys.

Temperature should be `0` (or the closest deterministic setting supported by the model) for all stages unless a specific development experiment demonstrates a benefit from diversity.

If diversity is required, generate deterministic multiple candidates via separately named prompt variants rather than uncontrolled sampling.

---

# 9. Stage A — characterization + research planning

Combine task characterization and retrieval-intent planning into **one Qwen3 235B call per topic** to reduce calls and keep the decomposition coherent.

Input:

- topic narrative;
- ResearchRubrics definitions for domain/complexity;
- retrieval-mode definitions.

Output:

```json
{
  "domain": "...",
  "conceptual_breadth": "Simple|Moderate|High",
  "logical_nesting": "Shallow|Intermediate|Deep",
  "exploration": "Low|Medium|High",
  "explicit_subquestions": ["..."],
  "named_constraints": ["..."],
  "synthesis_goal": "...",
  "research_intents": [
    {"query": "...", "retrieval_mode": "hybrid", "purpose": "..."}
  ]
}
```

Target 6–10 intents/topic.

Require intent coverage for:

- each explicit ask;
- names/entities/rare terminology;
- quantitative/date evidence;
- context;
- counterarguments;
- stakeholder effects;
- uncertainty/limitations;
- causal confounds;
- evidence capable of falsifying simplistic conclusions.

Validation:

- no empty query;
- at least one intent per explicit subquestion;
- lexical facts routed sparse/hybrid;
- conceptual questions routed dense/hybrid;
- no candidate response information.

---

# 10. Stage B — ClimbMix retrieval

Use the existing live backends:

```text
dense:  http://127.0.0.1:8088
sparse: http://127.0.0.1:8085
```

Do not change their wire contracts.

For each intent:

```text
k = 20 per selected backend
```

Then:

1. normalize;
2. dedupe exact chunk IDs;
3. derive/dedupe parent doc IDs;
4. deterministic RRF;
5. retain a broad top pool (~60–100 chunks/topic) before evidence condensation.

Do not average raw dense and BM25 scores.

Record provenance:

```text
qid
intent id
query
backend
rank
chunk id
parent docid
raw score
RRF rank/score
```

Search outage != zero hits. Fail/retry the topic stage on backend unavailability.

---

# 11. Stage C — evidence condensation

Use one Qwen3 235B call/topic where context permits.

Input:

- narrative;
- characterization;
- top fused passages, deterministically truncated;
- provenance IDs.

Target input ceiling:

```text
<= 80k tokens/topic
```

If the research pack exceeds that ceiling, deterministic chunking into 2 passes is allowed, followed by one merge pass.

Output 25–45 compact evidence notes.

Rules:

- preserve numerical values/dates/entities accurately;
- factual notes require either one strong source or >=2 independent sources where feasible;
- contested evidence becomes a disagreement note, not a false consensus;
- avoid turning every retrieved fact into a rubric requirement;
- distinguish core decision-relevant evidence from background.

Cache this stage separately because it is expensive to reconstruct and independent of later rubric prompt experiments.

---

# 12. Stage D — axis-specific candidate generation

Use Qwen3 235B Flex for semantic-heavy generation.

Do **not** one-shot the entire rubric.

Run separate deterministic calls:

## D1. Explicit criteria

Input primarily narrative + explicit decomposition.

Target approximately:

```text
6–10 candidates
```

Every explicit ask must map to >=1 candidate.

## D2. Implicit criteria

Input narrative + characterization + evidence notes + explicit candidates.

Target approximately:

```text
10–16 candidates
```

This is the highest-value generation call.

Require one-line provenance justification for each candidate explaining why it is a reasonable expert expectation rather than arbitrary scope expansion.

## D3. Synthesis criteria

Input narrative + evidence notes + current candidates.

Target:

```text
3–6 candidates
```

A synthesis criterion must require relations such as compare/reconcile/weigh/connect/explain rather than list isolated facts.

## D4. Negative failure modes

Input narrative + evidence notes + current positive criteria.

Target:

```text
3–8 candidates
```

Prioritise materially damaging failures:

- factual reversal;
- causal overclaim;
- wrong number/denominator;
- fabricated certainty;
- misleading omission;
- one-sided treatment when balance is required;
- stale/out-of-scope claims;
- source misattribution.

Avoid exact duplicate negations of positive criteria unless the commission error is materially worse than omission.

## D5. Low-volume axes

One structured call can generate sections for:

```text
References & Citation Quality
Communication Quality
Instruction Following
```

Target only a small number where applicable.

Citation criteria should not duplicate the dedicated RAGDoll citation-support lane.

---

# 13. Stage E — deterministic pre-normalization

Before another LLM call:

1. normalize whitespace;
2. stable-sort by axis + candidate ID;
3. reject empty criteria;
4. exact-casefold dedupe;
5. reject unknown axes;
6. check criterion length bounds;
7. flag likely multi-requirement criteria using deterministic punctuation/conjunction heuristics;
8. flag obvious positive/negative mirror duplicates.

Do not delete semantically similar candidates automatically beyond exact duplicates.

---

# 14. Stage F — atomicity/deduplication critic

Use Qwen3 235B Flex directly for V1. Do not add a smaller-model escalation ladder until the development benchmark proves it is safe and worthwhile.

For each candidate classify:

```text
ATOMIC
SPLIT_REQUIRED
REDUNDANT
TOO_VAGUE
UNVERIFIABLE
OUT_OF_SCOPE
```

Provide all candidates in the same topic so the critic can identify overlap.

Output explicit candidate IDs and merge/split targets.

Deterministic code applies the decisions.

Bound repair to one critic revision pass, then escalate/fail rather than looping indefinitely.

---

# 15. Stage G — targeted examples

Examples are optional and added only after criterion semantics are stable.

The dev file uses `(e.g., ...)` in ~47% of criteria, but this is a **style prior, not a quota**.

Use Qwen3 235B only for criteria flagged as materially clearer with examples.

Rules:

- 1–4 short examples maximum;
- examples must derive from the narrative/evidence pack;
- examples illustrate, not exhaust, acceptable content;
- examples cannot introduce a new requirement;
- no verbose rephrasing of the criterion.

A semantic-preservation check must reject an example insertion if it changes the criterion's scope.

---

# 16. Stage H — independent importance/weight classification

Do not let the criterion-writing call choose its own final numeric weight.

Use Qwen3 235B Flex to classify every surviving candidate into:

```text
CRITICAL_POSITIVE
IMPORTANT_POSITIVE
MINOR_POSITIVE
MINOR_NEGATIVE
DETRIMENTAL
CRITICAL_NEGATIVE
```

plus:

```text
strong: true|false
```

Then use existing deterministic `importance_to_weight()`.

The prompt should answer:

1. Would omission/violation make the response fail the user's core need?
2. Does it affect factual validity/core decision, or only polish?
3. For negative criteria, is occurrence invalidating or merely degrading?

Use the existing mandatory rule:

```text
abs(weight) >= 4
```

---

# 17. Stage I — deterministic final selection

The generator will usually have more candidates than the final target.

Select deterministically using this priority:

1. retain every validated mandatory explicit criterion;
2. retain critical negative criteria;
3. retain highest-confidence/highest-evidence implicit criteria;
4. retain meaningful synthesis criteria;
5. retain applicable instruction/communication/reference criteria;
6. remove redundant lower-value candidates;
7. use ResearchRubrics axis proportions only as soft priors;
8. fit the complexity-based criterion count range.

Initial final count policy:

```text
Simple/low:       20–23
Moderate/medium:  24–28
High/deep:        29–34
hard max:         40
```

Organizer dev reference:

```text
30 topics
755 criteria
mean 25.17/topic
median 25
min 20
max 32
```

Observed dev axis proportions:

```text
Implicit      41.7%
Explicit      27.9%
Synthesis     14.3%
Communication  7.8%
Instruction    5.7%
References      2.4%
```

Use these for diagnostics, not fixed quotas.

---

# 18. Stage J — independent Luna final audit

Run exactly one GPT-5.6 Luna review call on every candidate final rubric.

Luna sees:

- public narrative;
- characterization;
- compact evidence notes;
- final selected rubric with weights/axes.

It does **not** see candidate responses.

Require structured output:

```text
PASS
or
REPAIR_REQUIRED
```

with reason codes only:

```text
MISSING_EXPLICIT
MISSING_IMPORTANT_IMPLICIT
BAD_SYNTHESIS
OUT_OF_SCOPE
UNVERIFIABLE
REDUNDANT
BAD_AXIS
BAD_WEIGHT_CLASS
CONTRADICTION
GAMABLE
UNSUPPORTED_FACTUAL_CRITERION
```

If PASS: accept.

If REPAIR_REQUIRED:

1. Qwen3 235B performs a targeted repair using only flagged IDs/reasons;
2. deterministic validation reruns;
3. Luna rechecks once.

Maximum:

```text
2 Luna reviews/topic
1 Qwen repair/topic
```

If still failing, mark topic `NEEDS_ESCALATION` and stop that topic. Do not loop.

---

# 19. Stage K — generated-topic validation

Extend existing `validate_topic()` and pipeline validation to enforce:

```text
15 <= criteria <= 40
no zero weights
all weights in [-5, 5]
>=1 mandatory positive
known axes only
no exact duplicates
all explicit asks covered
not single-axis
negative polarity consistent
no mandatory unverifiable candidate
no unsupported factual examples
no mandatory contradictions
evidence-dependent candidates have provenance
```

Add topic-level generation status:

```text
PASS
REPAIRED
FAILED
```

Production artifact must contain only PASS/REPAIRED topics.

---

# 20. Development calibration on ResearchRubrics dev

Do **not** run all 119 production topics first.

Use the 30 public organizer dev prompts/rubrics.

## Baselines

Generate three versions:

```text
B0: one-shot topic-only Qwen3 235B
B1: staged topic-only Qwen3 235B
B2: staged + ClimbMix + review pipeline (production candidate)
```

Use identical final schema/validation.

## Structural metrics

Compare against organizer dev:

- criteria count mean/median/range;
- axis proportions;
- active axes/topic;
- mandatory rate by axis;
- positive/negative weight distribution;
- example-use rate;
- exact duplicate rate;
- validation failure rate.

## Semantic fidelity

Use **Luna as the independent matcher**, not the generating model.

For each dev topic give Luna:

- gold criterion IDs/text/axis/mandatory status;
- generated criterion IDs/text/axis/mandatory status.

Ask for one-to-one semantic matches with labels:

```text
EQUIVALENT
GENERATED_COVERS_GOLD
PARTIAL
NO_MATCH
```

Derive:

```text
gold criterion recall
gold mandatory recall
generated precision
semantic F1
axis agreement on matched criteria
mandatory-status agreement
```

Enforce deterministic one-to-one matching in code after the model proposes candidates.

## Selection gate

B2 must:

- pass schema/validation for 100% of dev topics after bounded repair;
- not reduce mandatory recall relative to B1;
- improve either semantic F1 or mandatory recall over B0 materially;
- avoid major axis-distribution collapse;
- keep mean criterion count in the observed ResearchRubrics range;
- have no unexplained large cost/latency regression.

If B2 fails this gate, do not produce RAG26 rubrics. Diagnose the failing stage on the public dev set first.

---

# 21. Small production rehearsal

After dev calibration, freeze **all code/prompts/models/settings** except bug fixes.

Run exactly 3 public RAG26 Requests selected deterministically by topic-file order:

```text
first topic
middle topic
last topic
```

Because Requests are unrestricted this is safe.

Inspect only the generated response-independent rubrics/research provenance, never candidate reports.

Verify:

- retrieval health;
- sensible evidence packs;
- generation log/provenance completeness;
- cost estimate vs actual;
- no malformed outputs;
- deterministic rerun/cache hits;
- criterion counts/axes reasonable.

Do not tune based on candidate system outputs.

If the 3-topic rehearsal passes, proceed to all 119.

---

# 22. Freeze protocol

Before the 119-topic run record:

```text
generator git SHA
Qwen3 235B model ID + tier + region
Luna deployment/model ID
all prompt versions
retrieval URLs/backend identities
RRF parameters
criteria-count policy
cache schema version
validator version
```

Generate a manifest hash from this configuration.

No prompt/model/selection changes during the production run.

Only execution bug fixes are allowed; if a bug fix changes semantics, increment generator version and restart all 119 topics under the new frozen version.

---

# 23. Production run for 119 RAG26 topics

Command target:

```bash
uv run python scripts/generate_rag26_rubrics.py \
  --topics <rag26-topics.jsonl> \
  --dense-url http://127.0.0.1:8088 \
  --sparse-url http://127.0.0.1:8085 \
  --model qwen.qwen3-235b-a22b-2507-v1:0 \
  --region ap-southeast-2 \
  --service-tier flex \
  --review-model <azure-luna-deployment> \
  --cache-dir <response-independent-cache> \
  --out-dir local-data/generated/rag26-rubrics-v1 \
  --budget-usd 20
```

Exact CLI names may differ after implementation, but these capabilities are required.

Safe progress log:

```text
qid index/119
stage
candidate count
retained count
active axes
LLM call count
cache hit/miss
tokens in/out
estimated/actual cost
retry count
elapsed
status
```

Do not print giant evidence text/prompts into logs.

Allow bounded topic-level resume from cache/checkpoints.

---

# 24. Production outputs

Development/local full provenance:

```text
local-data/generated/rag26-rubrics-v1/
    rubrics.jsonl
    provenance.jsonl
    stats.json
    manifest.json
    generation-summary.jsonl
```

Then copy only TIRA/runtime-required response-independent artifacts into:

```text
artifacts/rag26/
    rubrics.jsonl
    manifest.json
    stats.json
```

Do not bundle raw research-pack caches or large passage text into the TIRA image unless the runtime judge actually requires them. The rubric-conditioned Arena should need only the compact rubric artifact.

Final artifact gate:

```text
119/119 qids
no duplicate qids
all schemas valid
all topic validators PASS/REPAIRED
no candidate-response content
stable deterministic serialization
manifest hashes match
```

---

# 25. Runtime integration after artifact freeze

Only after the 119-topic rubric file is frozen:

1. add frozen-rubric loader to `RAG26ArenaJudge`;
2. add `rubric-arena` variant;
3. then add `rubric-support-arena` variant;
4. keep citation support independent;
5. do not feed pointwise scores/ranking metadata into Arena prompts;
6. Arena sees only:

```text
Request
compact frozen rubric
Answer A
Answer B
```

If a qid has no frozen rubric (e.g. kiddie), fall back deterministically to `arena-only`.

Pointwise rubric scoring can be added later and must not block the rubric-conditioned Arena submission path.

---

# 26. Tests required before production generation

Add unit tests for:

```text
structured response parsing
prompt determinism
cache-key determinism
characterization validation
research-intent validation
evidence note validation
candidate IDs/order
axis generation
negative polarity
exact dedupe
atomicity critic decisions
split/merge application
example semantic-preservation guard
importance->weight mapping integration
Luna review repair loop
bounded retry
budget stop
resume/checkpoint
stable JSONL serialization
manifest hashing
119-topic coverage verifier
```

Integration tests:

```text
synthetic topic end-to-end with dummy models/search
1 public ResearchRubrics dev topic with real cache replay
small ClimbMix live smoke (manual/marked live)
3-topic production rehearsal
```

Normal pytest must not require network or spend money.

---

# 27. Immediate execution order

Implement in this order:

```text
1. update stale rag26_arena README/status
2. refresh Bedrock pricing + add Qwen3 235B Flex config; leave 32B as an optional later cost ablation
3. targeted Qwen3 235B Flex smoke test
4. typed generation/provenance models
5. response-independent LLM cache + usage/cost accounting
6. prompt builders/versioning
7. characterization + intent planner
8. ClimbMix research/evidence condensation
9. explicit generator
10. implicit generator
11. synthesis generator
12. negative generator
13. low-volume axes generator
14. deterministic pre-normalization
15. Qwen235 atomicity/dedupe critic
16. targeted examples
17. independent importance/weight classifier
18. deterministic final selection
19. Luna final reviewer + bounded repair
20. extend validators
21. generation CLI
22. ResearchRubrics dev evaluator
23. run B0/B1/B2 on 30 dev topics
24. select/fix production recipe using dev results only
25. freeze generator configuration
26. 3-topic RAG26 rehearsal
27. run 119-topic production generation
28. verify/freeze artifacts
29. integrate `rubric-arena`
30. integrate `rubric-support-arena`
```

Do not skip from step 3 directly to a 119-topic generation run.

---

# 28. Definition of done

Rubric construction is complete when:

- generator code/tests are committed;
- ResearchRubrics dev calibration shows the staged/evidence-grounded pipeline is preferable to the one-shot baseline;
- model/prompt/retrieval configuration is frozen;
- all 119 RAG26 topic rubrics generated automatically;
- every rubric passes validation after bounded repair;
- compatibility artifact matches organizer schema;
- response-independent provenance/manifest exists;
- production cost is within the hard cap;
- deterministic rerun uses cache successfully;
- `artifacts/rag26/rubrics.jsonl` is ready for Docker packaging;
- no candidate response influenced rubric construction.

The primary model choice for this workstream is **Qwen3 235B A22B 2507 on Bedrock Flex**. Do not reopen broad model selection unless the public ResearchRubrics development gate demonstrates a concrete quality problem.
