# Sonnet Handoff — Prioritise RAG26 TIRA Submission

We are now prioritising **TREC RAG 2026 (`rag26-generation`) above RAGTIME**.

The canonical execution plan is:

```text
docs/rag26-tira-submission-plan.md
```

Read that file in full before implementing anything.

Also read in full:

- `AGENTS.md`
- `CLAUDE.md`
- `.claude/skills/autojudge-develop/SKILL.md`
- `.claude/skills/autojudge-submit/SKILL.md` if present
- `docs/trec-rag-2026-judge-implementation-plan.md`
- `docs/trec-rag-2026-researchrubrics-generation-plan.md`
- `judges/rag26_arena/README.md`

The AutoJudge data-handling policy is mandatory. Do not inspect restricted RAG26 `Report` content outside the organizer-defined permitted first-10-topic window. Full-dataset judge runs are allowed, but full-set outputs may only be used for execution/completeness checks, not tuning.

## Current known state

Last verified full suite:

```text
146 passed, 2 xfailed
```

Current RAG26 foundation already includes:

- ResearchRubrics-compatible schemas;
- deterministic rubric weight mapping;
- rubric validation and I/O;
- ResearchRubrics dev-set analyzer/statistics;
- topic normalization;
- dense ClimbMix client;
- sparse BM25 client;
- hybrid retrieval;
- deterministic RRF;
- chunk -> parent-document normalization;
- response-independent research-pack structures/provenance;
- live-search smoke tooling;
- unit/contract tests.

Do not rewrite working retrieval infrastructure unless you find a concrete defect.

## Live ClimbMix services

Dense:

```text
http://127.0.0.1:8088
```

Backed by:

```text
/scratch/fast/kun/projects/trec-rag-26/data/built-indexes/climbmix-chunked-ft
```

Key properties:

```text
~921,892,634 chunks
256-dimensional DiskANN
metric=MIPS
model=RMIT-ADMS/jina-v5-nano-trecrag26-agent-256d
```

Sparse:

```text
http://127.0.0.1:8085
```

Backed by:

```text
/scratch/fast/kun/projects/trec-rag-26/data/built-indexes/climbmix-bm25-chunked
BM25 k1=0.9
BM25 b=0.4
```

Sparse request:

```json
{"query": "distinctive lexical terms", "hits": 20}
```

Treat sparse as bag-of-words / OR-style search. Do not assume full Lucene Boolean semantics.

These services are **offline artifact-generation dependencies only**. The submitted TIRA judge must not require either service at runtime.

## Primary objective

Build the shortest reliable path to a valid **RAG26 data + code submission to TIRA**.

Do not spend another long pass only extending design documents.

Do not make rubric generation the critical path before a runnable judge exists.

## Required implementation order

### 1. Inspect and checkpoint current work

First inspect:

```text
git status
git diff
untracked RAG26 files
current tests
```

Review existing work rather than replacing it.

Run relevant RAG26 tests and full pytest.

Clean accidental `__pycache__`, cache, output, or build artifacts.

Create a logical checkpoint commit once the current foundation is understood and green.

### 2. Build Arena-only RAG26 judge

Create the first runnable specialist judge under:

```text
judges/rag26_arena/
    judge.py
    prompts.py
    models.py
    workflow.yml
```

Reuse mature SPARROW/election machinery from:

```text
judges/ragree_choice/
```

Reuse where possible:

- deterministic response ordering;
- sparse connected graph;
- Davidson/Bradley-Terry fit;
- adaptive comparisons;
- position reversal;
- cycle/Condorcet diagnostics;
- retry/failure handling.

Do **not** carry over the old evidence gate into the default RAG26 Arena.

Use public RAGDoll/TREC RAG 2026 preference semantics:

```text
Which answer would the user rather receive?
```

Judge:

- intent fulfilment;
- correctness/plausibility;
- completeness;
- specificity;
- synthesis;
- calibration;
- explanation/usefulness;
- organisation/noise control.

Do not blindly reward verbosity.

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

Never convert provider failure into a loss or low score.

Use initial settings:

```yaml
full_round_robin_max_candidates: 5
initial_graph_degree: 4
adaptive_comparisons_per_topic: 4
position_reversal_budget: 2
exact_kemeny_limit: 8
```

Always sort responses by `run_id` before pairing.

Prompt construction must be byte-deterministic for cache replay.

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

Add comprehensive unit tests for parsing, ties, both-bad, unavailable calls, deterministic ordering, reversal, graph connectivity, one-candidate and empty-response cases.

### 3. Prove the TIRA vertical slice immediately

Before rubric generation:

1. run `arena-only` on `kiddie`;
2. use `$CACHE_DIR`;
3. confirm deterministic cache-only replay with endpoint disabled;
4. inspect/fix Docker or Podman setup;
5. run `tira-cli code-submission --dry-run` on kiddie.

Unknown/non-RAG26 qids must have a safe fallback so kiddie works without frozen RAG26 artifacts.

Do not proceed deeply into rubric generation until this works.

### 4. Add citation-support lane

Implement RAGDoll-compatible support semantics:

```text
NONE    -> 0.0
PARTIAL -> 0.5
FULL    -> 1.0
```

Hard support:

```text
FULL -> 1
else -> 0
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

Use citation/reference documents already available through the runtime `Report`.

Do not call ClimbMix during TIRA report scoring.

Batch support judgments; avoid one call per citation.

### 5. Implement offline LLM intent planning

Add an injected/pluggable `IntentPlanner` for the existing hybrid retrieval pipeline.

Generate approximately 5–10 intents per public Request:

```python
ResearchIntent(
    query=...,
    retrieval_mode="dense" | "sparse" | "hybrid",
    purpose=...,
)
```

Cover explicit subquestions, names, dates/numbers, quantitative evidence, relevant context, counterarguments, uncertainty, stakeholder effects, causal caveats, and evidence that could falsify simplistic conclusions.

Routing:

```text
name/date/number/rare-term -> sparse
conceptual/paraphrase      -> dense
important/ambiguous        -> hybrid
```

Keep deterministic RRF. Never average raw BM25 and dense scores.

### 6. Add research-pack filtering and persistent caching

RRF is retrieval, not final evidence selection.

Add lightweight relevance/evidence-quality filtering after fusion.

Use compact evidence notes rather than huge raw-document prompts.

Add persistent response-independent cache keyed by topic/intents/retrieval/index/prompt versions.

Never include candidate responses in artifact-generation cache keys.

### 7. Implement ResearchRubrics generation

Generate axes separately:

```text
Explicit Criteria
Implicit Criteria
Synthesis of Information
References & Citation Quality
Communication Quality
Instruction Following
negative failure criteria
```

Pipeline:

```text
axis generators
 -> atomicity/verifiability critic
 -> dedupe
 -> ordinal importance classifier
 -> final coverage/quality reviewer
```

Do not one-shot all criteria.

Do not let the model invent arbitrary integer weights; use the existing deterministic ordinal-class -> `-5..+5` mapping.

Use the organizer ResearchRubrics development set as the primary benchmark.

Evaluate structural/semantic fidelity including mandatory recall, axis distribution, semantic criterion recall, negative recall, polarity, duplication and scoring agreement where possible.

Do **not** optimise RAG26 rubric generation against RAG25 report-level leaderboard correlation.

### 8. Freeze all 119 RAG26 artifacts

Once the generator is frozen, generate all public Requests exactly once.

Create:

```text
artifacts/rag26/
    rubrics.jsonl
    manifest.json
    artifact-stats.json
```

Artifacts must be fully response-independent.

Manifest should record:

```text
schema version
generator git SHA
prompt versions
models by role
dense/sparse index identities
retrieval/RRF settings
input topic-file hash
artifact hashes
```

No candidate response material.

No manual per-topic editing based on participant outputs.

Verify 119/119 qids.

Bundle artifacts into Docker. Do not bundle ClimbMix indexes.

### 9. Add pointwise rubric scorer

Load frozen rubric by qid.

Batch roughly 8–15 criteria per LLM call, depending on context.

Verdicts:

```text
SATISFIED
PARTIAL
NOT_SATISFIED
```

Expose:

```text
RAG26_RUBRIC_SCORE
RAG26_RUBRIC_MANDATORY_RECALL
RAG26_RUBRIC_OPTIONAL_SCORE
RAG26_RUBRIC_NEGATIVE_PENALTY
RAG26_RUBRIC_EVALUATED
```

Unavailable judgment != failed criterion.

### 10. Add rubric-conditioned Arena

Define variants:

```text
arena-only
rubric-arena
rubric-support-arena
```

Pairwise Arena may see:

```text
Request
compact frozen rubric
Answer A
Answer B
```

It must not see:

```text
system names
run metadata
pointwise rubric scores
citation scores
current ranking
gate output
```

Arena/SPARROW remains the ranking backbone.

Citation and rubric metrics remain independently reported.

### 11. Model/cost strategy

Separate offline generation from runtime scoring.

Offline generation starting candidate:

```text
Qwen3 235B A22B
```

Use DeepSeek V3.x or similar if needed for hard synthesis/review.

Use GPT-5.6 Terra selectively, not for bulk work.

Do not use Sol for bulk generation.

Primary local full-data runtime candidate:

```text
Qwen3 32B
```

Secondary comparison:

```text
GPT-5.6 Luna
```

TIRA code must remain model-agnostic and use only injected:

```text
OPENAI_BASE_URL
OPENAI_API_KEY
OPENAI_MODEL
CACHE_DIR
```

Before any large paid run use:

```text
scripts/bedrock_cost.py
scripts/azure_cost.py
scripts/cost_guard.py
```

Report call count, max token assumptions, model price, cache state and worst-case cost before execution.

Do not launch an expensive sweep without explicit approval.

### 12. Cost-aware production variant

Estimate the full call budget before enabling every lane.

83 systems imply:

```text
3,403 exhaustive battles/topic
>404k exhaustive battles over 119 topics
```

Degree-4 SPARROW is roughly:

```text
166 initial battles/topic
~19,754 initial battles total
```

plus adaptive/reversal/retry calls.

If pointwise rubric scoring adds ~2–3 calls per response, it may add roughly another ~25k calls.

Therefore consider keeping pointwise rubric scoring as an optional diagnostics variant if it materially increases cost.

Likely primary variant:

```text
rubric-support-arena
```

but do not select based on full hidden RAG26 results.

### 13. Hardening

Test:

- malformed LLM output;
- timeout/rate limit;
- endpoint unavailable;
- partial cache;
- empty report;
- missing/bad/duplicate citations;
- one candidate;
- small/large pools;
- unknown qid fallback;
- missing/corrupt artifact;
- deterministic ordering;
- cache replay;
- output schema;
- no report text in scored outputs/logs.

Critical invariant:

```text
unavailable judgment != negative judgment
```

### 14. Full local RAG26 run

Before full execution freeze and record:

```text
git SHA
workflow variant
artifact hash
model
prompt versions
SPARROW settings
cost estimate/budget
```

Then run all 83 systems × 119 topics.

The judge may process all reports, but do not inspect restricted report content, prompt/completion traffic or response-derived cache entries.

For the full set, inspect only safe output for execution correctness/completeness.

Do not use local placeholder/random truth as a quality signal.

Upload data submission to:

```text
rag26-20260827_1-test
```

### 15. Final TIRA code submission

Before real submission require:

```text
git clean
full pytest green
container preflight green
kiddie end-to-end green
deterministic cache replay green
frozen artifacts bundled
no local search runtime dependency
no credentials in repo/image
README updated
starter examples removed if required
```

Run TIRA dry-run first, then submit the frozen RAG26 workflow/variant.

## Working style

Implement incrementally.

After each meaningful phase report:

- files changed;
- tests run/results;
- what now works end-to-end;
- cost/call-count impact;
- blockers;
- exact next recommended step.

Do not keep expanding design docs instead of implementing.

The immediate milestone is:

> `arena-only -> kiddie -> deterministic cache replay -> TIRA dry-run`.
