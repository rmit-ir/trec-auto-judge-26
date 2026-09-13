# RAGree RAG26 — TIRA Submission Plan

**Prepared:** 2026-09-13  
**Deadline:** 2026-09-30  
**Primary dataset:** `rag26-generation` / TIRA `rag26-20260827_1-test`

## Goal

Prioritise a valid, reproducible TREC RAG 2026 AutoJudge submission before expanding to RAGTIME.

The key principle is:

> Build the thinnest valid RAG26 TIRA submission first. Then add rubric and citation quality layers without breaking the submission path.

A complete campaign submission requires both:

1. a **data submission** — local leaderboard output uploaded to TIRA; and
2. a **code submission** — reproducible judge code submitted through `tira-cli code-submission`.

The RAG26 test dataset contains 83 runs and 119 topics. All runs must be judged.

---

# 1. Mandatory data-handling boundary

Before any work touching evaluation data, read in full:

- `AGENTS.md`
- `CLAUDE.md`
- `.claude/skills/autojudge-develop/SKILL.md`
- `.claude/skills/autojudge-submit/SKILL.md` if present

The AutoJudge data policy remains binding throughout implementation:

- the judge may process all RAG26 reports;
- developers/coding agents may inspect report content only for the organizer-defined permitted first 10 topics;
- full-set scored outputs may be checked for execution correctness/completeness, not used for tuning;
- restricted prompt caches, traces, rationales, or report-derived artifacts must not be inspected;
- no deanonymisation of run IDs;
- all prompt/model/threshold/variant decisions must be based on unrestricted data, public development material, synthetic fixtures, or the permitted window.

Do not use the local RAG26 placeholder/random truth to select models, prompts, or variants.

---

# 2. Current known repository state

Last verified state before this plan:

```text
branch: main
full tests: 146 passed, 2 xfailed
```

The current RAG26 foundation under `judges/rag26_arena/` already includes:

- ResearchRubrics-compatible schemas;
- deterministic rubric weight mapping;
- rubric validation and deterministic I/O;
- ResearchRubrics development statistics/analyzer;
- topic loading/normalisation;
- dense ClimbMix HTTP client;
- sparse BM25 HTTP client;
- hybrid dense+sparse retrieval;
- deterministic RRF;
- chunk -> parent-document normalisation;
- response-independent research-pack structures/provenance;
- live search smoke tooling;
- unit/contract tests.

Do not rewrite working retrieval infrastructure unless a concrete defect is found.

Still missing before RAG26 is submission-ready:

- `RAG26ArenaJudge`;
- `workflow.yml`;
- Arena prompt/parser;
- SPARROW integration in the new judge;
- citation-support lane;
- LLM research-intent planner;
- persistent topic research cache;
- rubric generation/review;
- frozen topic-artifact generation;
- pointwise rubric scorer;
- rubric-conditioned Arena;
- TIRA-safe artifact packaging;
- end-to-end kiddie + TIRA dry-run.

---

# 3. Live ClimbMix search infrastructure

These services are available for **offline topic research and artifact generation only**.

## Dense semantic search

```text
http://127.0.0.1:8088
```

Backed by:

```text
/scratch/fast/kun/projects/trec-rag-26/data/built-indexes/climbmix-chunked-ft
```

Verified properties:

```text
n_docs: 921,892,634
vector dim: 256
metric: MIPS
kind: DiskANN
model: RMIT-ADMS/jina-v5-nano-trecrag26-agent-256d
```

Primary API:

```text
GET  /health
POST /search
POST /search/batch
GET  /doc/{docid}
POST /doc/batch
```

Use for conceptual/paraphrase/synthesis-oriented retrieval.

## Sparse BM25 search

```text
http://127.0.0.1:8085
```

Backed by:

```text
/scratch/fast/kun/projects/trec-rag-26/data/built-indexes/climbmix-bm25-chunked
```

Configuration:

```text
BM25 k1 = 0.9
BM25 b  = 0.4
```

API:

```text
POST /api/search
```

Request:

```json
{"query": "distinctive lexical terms", "hits": 20}
```

Treat this service as bag-of-words / OR-style search. Do not assume full Lucene Boolean semantics.

Use for names, dates, legislation, rare terms, technical terminology, identifiers, and exact lexical anchors.

## Runtime rule

The submitted TIRA judge must **not** depend on either local service.

Instead:

```text
public Request
   -> dense/sparse ClimbMix research
   -> frozen response-independent topic artifacts
   -> bundle artifacts into Docker image
   -> TIRA runtime reads frozen artifacts + candidate Report
```

Do not bundle the indexes themselves.

---

# 4. Phase 0 — inspect, clean, checkpoint

Before new implementation:

1. inspect `git status` and `git diff`;
2. inspect all untracked RAG26 work;
3. run relevant RAG26 tests;
4. run full pytest;
5. remove/ignore accidental `__pycache__`, caches, output directories, or build files;
6. create a logical checkpoint commit for the existing working foundation.

Exit gate:

```text
existing RAG26 foundation understood
relevant tests green
full tests green
current implementation checkpointed
```

---

# 5. Phase 1 — Arena-only runnable RAG26 judge

This is the highest-priority implementation task.

Create at minimum:

```text
judges/rag26_arena/
    judge.py
    prompts.py
    models.py
    workflow.yml
```

Reuse mature election/ranking infrastructure from:

```text
judges/ragree_choice/
```

Reuse where possible:

- deterministic response ordering;
- sparse connected comparison graph;
- Davidson/Bradley-Terry fitting;
- adaptive comparisons;
- position-reversal checks;
- cycle/Condorcet diagnostics;
- retry/failure handling.

Do not carry the old evidence gate into the RAG26 default.

## Arena semantics

Use public RAGDoll/TREC RAG 2026-style preference semantics:

```text
Which response would the user rather receive?
```

Evaluate:

- intent fulfilment;
- correctness/plausibility;
- completeness;
- specificity;
- synthesis;
- calibration;
- directness/usefulness;
- explanation quality;
- organisation/noise control.

Do not reward verbosity by itself.

Strict verdicts:

```text
A
B
TIE
BOTH_BAD
ABSTAIN
```

Malformed/unavailable LLM call:

```text
retry -> ABSTAIN
```

Never silently convert model failure into a loss or low-quality score.

## Initial SPARROW settings

```yaml
full_round_robin_max_candidates: 5
initial_graph_degree: 4
adaptive_comparisons_per_topic: 4
position_reversal_budget: 2
exact_kemeny_limit: 8
```

For 83 systems, degree 4 requires approximately:

```text
83 * 4 / 2 ~= 166 initial comparisons/topic
```

rather than 3,403 exhaustive comparisons/topic.

Always sort responses by `run_id` before constructing pairs.

Prompt construction must be byte-deterministic for cache replay.

## Minimum output measures

Expose at least:

```text
RAG26_ARENA_SCORE
PAIRWISE_WIN_RATE
CONDORCET_WINNER
RANKING_STABLE
CYCLE_RATE
BALLOT_COVERAGE
POSITION_REVERSAL_DISAGREEMENT_RATE
```

## Tests

Add tests for:

- A/B mapping;
- TIE;
- BOTH_BAD;
- ABSTAIN;
- malformed JSON;
- timeout/provider error;
- deterministic response order;
- position reversal;
- graph connectivity;
- one candidate;
- empty responses;
- small and large candidate pools;
- stable workflow import/config.

Exit gate:

```text
arena-only judge imports
workflow parses
kiddie can run end-to-end
```

---

# 6. Phase 2 — prove TIRA vertical slice immediately

Before rubric generation, prove the submission path.

Run:

```bash
auto-judge run \
  --workflow judges/rag26_arena/workflow.yml \
  --variant arena-only \
  --rag-responses data/kiddie/runs/repgen/ \
  --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
  --out-dir output-kiddie-rag26
```

Use `$CACHE_DIR`.

Confirm deterministic cache replay with the endpoint disabled.

The judge must support unknown/non-RAG26 qids gracefully so kiddie works without RAG26 artifacts.

Then make Docker/Podman work on the submission host and run a TIRA dry-run:

```bash
tira-cli code-submission \
  --dry-run \
  --path . \
  --cache-behaviour deterministic \
  --mount-cache '$CACHE_DIR=EMPTY_DIR' \
  --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
  --task trec-auto-judge \
  --dataset kiddie-20260605-training \
  --command 'auto-judge run --workflow /auto-judge/judges/rag26_arena/workflow.yml --variant arena-only --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```

Exit gate:

```text
arena-only kiddie green
cache-only replay green
container build green
TIRA dry-run green
```

**Status: PASSED (2026-09-13, SHA `b27b3ac`, Docker Desktop 4.88.1/engine 29.7.2).**
Both `arena-only` and `arena-support` dry runs built the image, produced valid kiddie
output, and reproduced byte-identical results from `$CACHE_DIR` with the endpoint
disabled — no code changes required. See
`docs/rag26-tira-dryrun-handoff.md` for the full transcript notes. One unrelated
blocker: `tira-cli verify-installation --task trec-auto-judge --team <TEAM>` 500s
server-side regardless of team-name spelling tried; login and Docker checks pass
standalone. Not a gate blocker (the dry-run itself needs no team scoping) but
worth raising with TIRA support before the real (non-dry-run) submission, since
that path likely does need it.

Do not make rubric generation the critical path before this gate is passed.

---

# 7. Phase 3 — citation-support lane

Implement the public RAGDoll-compatible citation-support semantics.

Soft support mapping:

```text
NONE    = 0.0
PARTIAL = 0.5
FULL    = 1.0
```

Hard support:

```text
FULL = 1
else = 0
```

Expose:

```text
RAG26_CIT_PREC_FIRST
RAG26_CIT_REC_FIRST
RAG26_CIT_PREC_ALL
RAG26_CIT_REC_ALL
RAG26_CIT_HARD_PREC
RAG26_CIT_HARD_REC
RAG26_SUPPORT_EVALUATED
```

Use documents/references present in the runtime `Report`.

Do not call ClimbMix during TIRA report scoring.

Batch several support decisions per call where context allows.

Avoid one LLM call per citation.

Exit gate:

- synthetic fixtures reproduce intended weighted/hard metrics;
- malformed/unavailable support judgment is tracked as unavailable, not low quality;
- no report/citation text leaks into normal scored outputs.

---

# 8. Phase 4 — LLM research-intent planner

Implement an injected/pluggable `IntentPlanner` for the existing hybrid retrieval pipeline.

For each public RAG26 Request generate approximately 5–10 typed intents:

```python
ResearchIntent(
    query: str,
    retrieval_mode: Literal["dense", "sparse", "hybrid"],
    purpose: str,
)
```

Cover:

- explicit subquestions;
- named entities;
- exact terminology;
- dates/numbers;
- quantitative evidence;
- relevant context;
- counterarguments;
- uncertainty;
- stakeholder impacts;
- causal caveats;
- evidence that could falsify simplistic conclusions.

Routing guidance:

```text
name/date/number/rare-term queries -> sparse
conceptual/paraphrase queries      -> dense
important/ambiguous facets         -> hybrid
```

Keep deterministic RRF.

Never average BM25 raw scores and dense raw scores.

---

# 9. Phase 5 — research-pack filtering and caching

RRF is retrieval, not final evidence selection.

Add a lightweight relevance/evidence-quality stage after fusion.

Build compact evidence notes such as:

```json
{
  "claim": "...",
  "docids": ["..."],
  "subtopic": "...",
  "evidence_type": "fact|number|argument|counterargument|context",
  "confidence": 0.9
}
```

Avoid giant raw-document prompts.

Add persistent topic-research caching.

Cache key should include:

```text
topic/request hash
research intents
retrieval configuration
index identities
builder/prompt version
```

Never include candidate responses.

---

# 10. Phase 6 — ResearchRubrics-style generation

Generate rubric axes separately rather than one-shot generation.

Required axes:

```text
Explicit Criteria
Implicit Criteria
Synthesis of Information
References & Citation Quality
Communication Quality
Instruction Following
```

Also generate high-value negative/failure criteria.

## Explicit criteria

Cover every explicit request requirement.

## Implicit criteria

Use Request + response-independent evidence pack to identify:

- important context;
- caveats;
- meaningful omissions;
- competing explanations;
- stakeholder implications;
- uncertainty;
- relevant temporal background;
- counterarguments.

## Synthesis criteria

Require actual relations:

```text
compare
reconcile
connect
distinguish
weigh
explain
assess
```

Reject pseudo-synthesis that only asks for lists of facts.

## Negative criteria

Prioritise:

- factual reversal;
- causal overclaim;
- fabricated certainty;
- misleading omission;
- denominator error;
- stale/out-of-scope fact;
- one-sided treatment;
- source misattribution;
- irrelevant tangent.

---

# 11. Phase 7 — rubric quality control

Use a constrained review pipeline:

```text
axis generators
  -> atomicity/verifiability critic
  -> deduplication
  -> ordinal importance classifier
  -> final coverage/quality reviewer
```

Do not initially implement an elaborate three-reviewer cascade unless the public development benchmark shows it is worthwhile.

LLM importance output should be ordinal, not arbitrary integer weights.

Use existing deterministic mapping into the `-5..+5` ResearchRubrics scale.

Use the organizer ResearchRubrics development set as the primary benchmark.

Measure:

- criteria/topic;
- axis distribution;
- semantic expert-criterion recall;
- mandatory recall;
- negative-criterion recall;
- polarity accuracy;
- axis correctness;
- duplicate rate;
- scoring agreement where possible.

Do not select RAG26 rubric design based on RAG25 system-ranking correlation; RAG25 report evaluation semantics differ materially from 2026.

---

# 12. Phase 8 — freeze all 119 topic artifacts

Once the generator is frozen, generate all public RAG26 Requests exactly once.

Create:

```text
artifacts/rag26/
    rubrics.jsonl
    manifest.json
    artifact-stats.json
```

All artifacts must be response-independent.

`manifest.json` should record at least:

```text
schema version
generator git SHA
prompt versions
model IDs by role
dense index identity
sparse index identity
retrieval configuration
RRF settings
input topic-file hash
artifact hashes
creation timestamp
```

No candidate response content.

No response-derived prompt cache.

No manual per-topic editing based on system outputs.

Artifact gate:

```text
119/119 qids present
no duplicate qids
schema valid
weights valid
axes valid
rubric arrays non-empty
```

Update Docker packaging so frozen artifacts are available inside TIRA.

Do not bundle ClimbMix indexes.

---

# 13. Phase 9 — pointwise rubric scorer

At runtime load frozen rubric by qid.

Batch criteria rather than one call per criterion.

Target approximately 8–15 criteria/call depending on report length/context.

Verdicts:

```text
SATISFIED
PARTIAL
NOT_SATISFIED
```

Use terse structured responses; do not emit verbose rationales in normal outputs.

Expose:

```text
RAG26_RUBRIC_SCORE
RAG26_RUBRIC_MANDATORY_RECALL
RAG26_RUBRIC_OPTIONAL_SCORE
RAG26_RUBRIC_NEGATIVE_PENALTY
RAG26_RUBRIC_EVALUATED
```

Unavailable judgment != failed criterion.

---

# 14. Phase 10 — rubric-conditioned Arena

Only after Arena and rubric scoring work independently.

Define variants:

```text
arena-only
rubric-arena
rubric-support-arena
```

Pairwise Arena may receive:

```text
Request
compact frozen rubric
Answer A
Answer B
```

It must not receive:

```text
system names
run metadata
pointwise rubric scores
citation scores
current ranking
gate results
```

Keep Arena/SPARROW as the ranking backbone.

Keep citation/rubric metrics independently reported.

Likely primary production variant:

```text
rubric-support-arena
```

but do not choose based on full hidden RAG26 outputs.

---

# 15. Model strategy

Separate offline artifact generation from runtime judging.

## Offline topic/rubric generation

Start with a strong but inexpensive model, preferably available through Bedrock:

```text
Qwen3 235B A22B
```

Use a stronger fallback such as DeepSeek V3.x if needed for difficult synthesis/generation.

Use GPT-5.6 Terra only as a selective reviewer/adjudicator.

Do not use GPT-5.6 Sol for bulk generation.

## Local full-data runtime

Primary candidate:

```text
Qwen3 32B
```

Secondary comparison:

```text
GPT-5.6 Luna
```

Choose using unrestricted/public development signals, schema reliability, reversal stability, citation-support quality, and cost — never full hidden RAG26 evaluation outputs.

## Submitted code

TIRA runtime must remain model-agnostic and consume only injected environment configuration:

```text
OPENAI_BASE_URL
OPENAI_API_KEY
OPENAI_MODEL
CACHE_DIR
```

Do not hardcode Azure/Bedrock model IDs into production runtime logic.

---

# 16. Cost controls

Before any expensive run use existing:

```text
scripts/bedrock_cost.py
scripts/azure_cost.py
scripts/cost_guard.py
```

Report:

```text
expected call count
max input/output tokens
model price assumptions
worst-case estimated cost
cache state
budget cap
```

Do not run a large paid sweep without explicit approval.

## Pairwise cost

For 83 systems:

```text
exhaustive: 3,403 battles/topic
119 topics: >404k pairwise battles
```

Degree-4 sparse graph:

```text
~166 initial battles/topic
~19,754 initial battles total
```

plus adaptive/reversal/retry calls.

SPARROW is therefore required for practical cost.

## Pointwise rubric cost

If each response uses 2–3 rubric batches:

```text
83 * 119 * ~2.5 ~= 24,700 calls
```

This means the primary submission may reasonably use:

```text
rubric-conditioned Arena + citation support
```

while keeping full pointwise rubric scoring as an optional diagnostic variant if it materially increases cost.

## Prompt minimisation

- batch judgments;
- strip irrelevant metadata;
- avoid sending retrieval provenance unless needed;
- use compact rubric text;
- deterministically truncate evidence;
- return terse JSON rather than prose rationales.

---

# 17. Prompt-cache requirements

Use `$CACHE_DIR` and disk-based caching.

Prompts must be byte-deterministic:

- stable response order;
- stable pair order;
- stable criterion order;
- stable citation order where semantics allow;
- stable JSON serialisation;
- no timestamps/UUIDs/temp paths.

Cache key should include model/request parameters but not endpoint URL.

The full RAG26 prompt cache contains restricted report-derived content and must not be inspected by coding agents/developers.

For primary TIRA code submission use:

```text
--cache-behaviour deterministic
--mount-cache '$CACHE_DIR=EMPTY_DIR'
```

unless a deliberately warm-cache submission is separately required.

---

# 18. Pre-submission hardening

Test at minimum:

- malformed LLM output;
- endpoint timeout;
- provider/rate-limit error;
- partial cache;
- endpoint unavailable;
- empty report;
- missing citation;
- malformed citation index;
- duplicate citation;
- one candidate;
- small candidate pool;
- large candidate pool;
- deterministic candidate order;
- unknown qid fallback;
- missing frozen rubric;
- corrupted artifact;
- cache replay;
- output schema;
- no report text in scored outputs/logs.

Critical invariant:

```text
unavailable judgment != negative judgment
```

---

# 19. Full local RAG26 data run

Before starting, freeze and record:

```text
git SHA
workflow variant
artifact hash
model
prompt versions
SPARROW settings
cost estimate/budget
```

Then run all 83 runs × 119 topics.

The judge may process all reports. Do not inspect restricted report text, LLM traffic, or prompt-cache entries.

Only inspect safe structural/scored outputs for:

```text
completion
all expected runs/topics represented
valid output schema
missing outputs
NaN/invalid numbers
runtime failures
```

Do not use local placeholder/random truth correlation as a quality signal.

Upload the data submission to:

```text
rag26-20260827_1-test
```

---

# 20. Final code submission gate

Before real TIRA code submission require:

```text
git status --porcelain -> empty
full pytest green
Docker/Podman preflight green
kiddie end-to-end green
cache-only replay deterministic
frozen artifacts bundled
no local search dependency
no credentials in repo/image
README updated
starter example judges removed if required
```

Run TIRA dry-run first.

Then submit the frozen RAG26 workflow/variant, scoped to RAG26 if desired/allowed.

---

# 21. Required implementation order

Do not change this order without a concrete reason:

```text
1. inspect/checkpoint current work
2. Arena-only runnable judge
3. kiddie end-to-end
4. deterministic cache replay
5. TIRA dry-run
6. citation support
7. offline intent planner
8. research-pack filtering/cache
9. rubric generator/review
10. freeze 119 rubrics
11. pointwise rubric scorer
12. rubric-conditioned Arena
13. cost comparison/model selection
14. freeze production variant
15. full RAG26 data submission
16. final TIRA code submission
```

This ensures there is a valid submission path early rather than making the entire research programme a prerequisite for submission.

---

# 22. Definition of success

Primary target submission:

```text
RAGree-RAG26
variant: rubric-support-arena
```

with:

- Arena/SPARROW ranking aligned to public TREC RAG 2026 preference evaluation;
- frozen response-independent ResearchRubrics-style topic artifacts;
- RAGDoll-compatible citation-support metrics;
- deterministic caching/replay;
- model-agnostic TIRA runtime;
- no dependency on local ClimbMix services at TIRA runtime;
- matching local data submission and code submission.

The highest-priority near-term milestone is not the final rubric generator. It is:

> `arena-only -> kiddie -> deterministic cache replay -> TIRA dry-run`.
