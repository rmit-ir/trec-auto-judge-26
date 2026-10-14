# RAG26 Arena / ResearchRubrics

Status as of 2026-10-13: **runnable judge; rubric pipeline built, frozen artifact pending**.
`RAG26ArenaJudge` (Arena/SPARROW pairwise preference + independent citation-support lane)
is implemented, tested, and TIRA-submission-shaped. Rubric generation (this file's other
half) is under active construction per `docs/rag26-rubric-construction-execution-plan.md`
-- see "Deliberately not built yet" below for exactly where that stands.

Canonical design docs:

- `docs/trec-rag-2026-judge-implementation-plan.md` -- judge architecture
- `docs/trec-rag-2026-researchrubrics-generation-plan.md` -- rubric-generation methodology
- `docs/rag26-tira-submission-plan.md` -- submission-path priorities/phases
- `docs/rag26-rubric-construction-execution-plan.md` -- **operational** rubric-construction
  plan; takes precedence over the two generation docs above on implementation order

## The runtime judge

- `judge.py` -- `RAG26ArenaJudge`. Pairwise Arena preference judging (`A`/`B`/`TIE`/
  `BOTH_BAD`/`ABSTAIN`) via SPARROW (reused unmodified from `judges/ragree_choice/sparrow.py`:
  sparse seed graph, adaptive acquisition, Davidson-Bradley-Terry fit, Condorcet/Kemeny/cycle
  audits, position reversal). No evidence gate -- every response in a topic is
  comparison-eligible; preference, rubric, and citation-support signals stay separate
  measures rather than collapsing into one gated score.
- `models.py` -- vote (`RAW_VOTES`, `BALLOT_VOTE`) and citation-support (`SUPPORT_LEVELS`)
  vocabularies.
- `prompts.py` -- the Arena pairwise prompt and the batched citation-support prompt.
- `support.py` -- RAGDoll-compatible citation-support metrics (first/all weighted + hard
  precision/recall), pure data/metrics, no LLM/IO, so it's unit-testable without a backend.
  Runs entirely off document text already resolved onto the `Report` -- never calls the
  offline ClimbMix index at scoring time, so it works unmodified inside a TIRA submission.
- `workflow.yml` -- variants `arena-only` (default), `arena-support` (turns on the
  independent citation-support lane), `exhaustive-small`, `no-reversal`.

Verified: kiddie end-to-end on both `arena-only` and `arena-support`, with deterministic
cache-only replay confirmed by re-running against a broken/unreachable endpoint from the
same warm `$CACHE_DIR` and diffing byte-identical output. The Docker/Podman TIRA dry-run
itself was delegated to a machine with a container runtime
(`docs/rag26-tira-dryrun-handoff.md`) since the original development host has neither and
no root/sudo to install one; it is not a blocker for the response-independent rubric-
construction work below, which needs no container runtime at all.

## Rubric construction (in progress)

Building the frozen, response-independent ResearchRubrics-style rubric artifact for all 119
RAG26 topics. The production constructor is **GPT-6 Luna with high reasoning** through
Bedrock Mantle (`us-east-1`), selected by the blind model bake-off in
`evaluation/rubric-model-bakeoff/`: GPT-6.1 Sol (high) judged anonymized outputs from Luna,
Qwen3 235B, Gemma 4 26B-A4B, and GLM-4.7 Flash, with Luna winning all tested topics.
The older execution plan still documents the pre-bake-off Qwen design and should be read as
historical rationale where its model choice conflicts with this README/current code.

### What's here already (pre-dates this workstream, reused as-is)

- `rubrics/schema.py` -- organizer-compatible `RubricCriterion`/`RubricTopic` (the exact
  `research-rubrics-dev-rubrics.jsonl` shape: `qid`, `domain`, `conceptual_breadth`,
  `logical_nesting`, `exploration`, `rubrics: [{criterion, weight, axis}]`). Deliberately
  permissive at parse time -- see the module docstring for why (the real organizer dev file has
  a non-canonical axis value and a zero weight; `validate.py` is the strict side, not this).
- `rubrics/weights.py` -- deterministic importance-class -> weight mapping: a model classifies
  ordinal importance, this maps it onto the -5..5 scale, so weight assignment never drifts by
  model or call.
- `rubrics/validate.py` -- quality gates a *generated* topic must pass. Not applied to the
  organizer dev file, which is reference data, not our output.
- `rubrics/io.py` -- load/write organizer-compatible JSONL, deterministic ordering.
- `rubrics/stats.py` + `scripts/analyze_researchrubrics.py` -- structural statistics over the
  dev file; see `evaluation/researchrubrics/dev-rubric-stats.md` for the actual numbers
  measured on it.
- `rubrics/topics.py` -- load/normalize the public TREC RAG topic file into `Topic(qid,
  narrative)`. No candidate response data enters here.
- `rubrics/research.py` -- single-backend research-pack pipeline (kept for the simple/dummy
  case): dedupes an already-produced query list against one `SearchClient`, unions hits by
  `docid`, produces a deterministic, cache-keyed `ResearchPack`.
- `search_index.py` -- the per-backend HTTP adapters (`SearchServeClient` dense,
  `SparseBM25Client` sparse, `DummySearchClient` for tests/ablation). See "Search-index
  clients" below.
- `hybrid_retrieval.py` -- the dense+sparse hybrid retriever: routes research intents to one
  or both backends, normalizes both wire formats, derives parent-document ids, fuses with
  deterministic RRF, and builds a `HybridResearchPack` with full per-hit provenance.
- `scripts/live_search_smoke.py` -- manual live smoke check (not part of the offline suite).

## Search-index clients

Two concrete HTTP adapters, matching the two real backends verified live on 2026-09-12:

- `SearchServeClient` -- the FastAPI `trec-rag-26/tasks/search_serve` dense/ANN contract
  (`POST /search`, `POST /search/batch`, `GET /doc/{docid}`, `POST /doc/batch`, `GET /health`).
- `SparseBM25Client` -- `POST /api/search` only, an Elasticsearch-shaped response
  (`hits.hits[]._id/_score/_source.contents`, no `rank` field -- derived from array position). No
  batch endpoint, no per-document-fetch endpoint on this backend at all; `health()` uses
  `/actuator/health` (Spring Boot). A non-200 raises `SearchUnavailableError` rather than
  looking like a valid zero-hit search (regression-tested).
- `DummySearchClient` -- in-memory, for tests and the prompt-only ablation/fallback.

Nothing is hardcoded: `dense_client_from_env()` reads `RAG26_DENSE_SEARCH_URL` (+ optional
`RAG26_DENSE_SEARCH_API_KEY`), `sparse_client_from_env()` reads `RAG26_SPARSE_SEARCH_URL` (+
optional `RAG26_SPARSE_SEARCH_API_KEY`); both share `RAG26_SEARCH_TIMEOUT_SECONDS`. Both are
needed simultaneously (the hybrid retriever, not a backend choice) --
`hybrid_retriever_from_env()` in `hybrid_retrieval.py` builds both at once.

Live endpoints (offline artifact-generation dependencies only -- the runtime `RAG26ArenaJudge`
must never call either at TIRA scoring time):

- Dense, `http://127.0.0.1:8088` -- DiskANN over a 256-d `jina-v5-nano-trecrag26-agent-256d`
  encoder, `n_docs: 921,892,634`.
- Sparse, `http://127.0.0.1:8085` -- Lucene BM25 (`k1=0.9, b=0.4`) over the same corpus,
  chunked identically (`shard_XXXXX_YYYYY_pN`).

## Rubric lane: V1 pipeline and runtime integration (2026-10-13)

Offline construction now runs end to end for one topic and in batch:

- `rubrics/finalize.py` -- Stage E (deterministic pre-normalization), F (atomicity/dedupe
  critic), H (independent importance classifier -> `importance_to_weight()`), I (deterministic
  final selection by complexity: Simple 22 / Moderate 26 / High 31 target, hard max 40).
  The critic can never drop explicit, negative (failure-mode) or low-volume-axis criteria --
  selection trims them instead; without this the critic deleted every negative as a "mirror".
  An unusable `SPLIT_REQUIRED` instruction falls back to keeping the criterion.
- `rubrics/build.py` -- resumable driver (`python -m judges.rag26_arena.rubrics.build ...`):
  Stage A-C (existing) + D1-D5 + finalize, one `<work-dir>/<qid>.json` per finished topic,
  assembles the organizer-compatible JSONL in topics-file order. Defaults to Bedrock **Standard**
  tier: Flex timed out (316 s) on a trivial call during the submission window. Needs the
  ClimbMix services (`RAG26_DENSE_SEARCH_URL`, `RAG26_SPARSE_SEARCH_URL`) and AWS credentials.
  Model/region/tier are CLI flags; the model choice is being benchmarked separately.
- Rehearsal (Qwen3 235B Standard, ap-southeast-2, topics 0-2): 3/3 valid, 26-31 criteria,
  all six axes present, 6-7 negatives per topic, about $0.007/topic. Known weak spot: the
  importance classifier is generous (9-14 mandatory criteria of 26-31).
- Deliberately skipped for the time budget: targeted-examples pass (Stage G), independent
  Luna audit (Stage J), ResearchRubrics dev calibration (plan Section 20).

Runtime (no ClimbMix/Bedrock dependency):

- `workflow.yml` variants `rubric-arena` and `rubric-support-arena` set
  `rubric_arena_enabled: true`; `rubric_path` defaults to
  `rubrics/frozen/rag26-rubrics.jsonl` (relative to `judges/rag26_arena/`, so it ships in the
  Docker image via `ADD judges`). The Arena prompt then also shows a compact rubric block
  (`+` desirable / `-` undesirable, strongest weights first, no weights shown). Still no
  answers, scores or support labels in the prompt.
- A qid with no frozen rubric gets the exact plain `arena-only` prompt (byte-identical, so
  existing prompt caches stay valid). An enabled variant whose artifact file is missing raises.
- The frozen artifact is not committed yet; until it is, the two rubric variants cannot run.

## Deliberately not built yet

- Pointwise rubric scorer (plan Phase 9). Not needed for `rubric-arena`.
- Stage G examples, Stage J Luna audit, dev-set B0/B1/B2 fidelity comparison.
- The frozen 119-topic artifact `judges/rag26_arena/rubrics/frozen/rag26-rubrics.jsonl`.

## Submission status (2026-10-13)

- `arena-support` TIRA dry-run passed on a Docker host (`docs/rag26-tira-dryrun-handoff.md`).
- Model: `gpt-5.6-luna` is the primary and only model for the submission. `gpt-6-luna`
  (ASE endpoint, vars `OPEAI_ASE_URL` / `AZURE_ASE_KEY`, deployment `gpt-6-luna`) is a backup
  only; the cache key includes the model name, so the two are never mixed in one run.
- First full run lost the entire citation-support lane: minima-llm `BatchConfig.max_failures`
  (default 25, env `BATCH_MAX_FAILURES`) aborts a whole batch and discards its results, and
  ~24 content-filter rejections tripped it. `judge.py` now builds its backend with
  `batch.max_failures=None`, so a rejected request is left unjudged ("unavailable", not
  negative) instead of sinking the batch (regression test in
  `tests/test_rag26_rubric_finalize.py`). Cache keys and prompts are unchanged.
- Rerun of `arena-support` on `rag26-generation` (83 runs x 119 topics, 16 measures, no
  NaN) completed; support lane populated. Cost of the rerun about $23 (first run about $30).
- Remaining: `tira-cli upload` of the data submission, and the real `tira-cli code-submission`
  (no `--dataset`) from a Docker host.
