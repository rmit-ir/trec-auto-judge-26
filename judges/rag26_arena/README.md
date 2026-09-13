# RAG26 Arena / ResearchRubrics

Status as of 2026-09-13: **runnable judge + rubric-construction workstream in progress**.
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
RAG26 topics, per `docs/rag26-rubric-construction-execution-plan.md`. Primary model: Qwen3
235B A22B 2507 on Bedrock Flex (`ap-southeast-2`); Qwen3 32B Flex for low-risk
atomicity/dedup/importance-classification stages only; GPT-5.6 Luna as the independent final
reviewer. See that plan for the full model/cost rationale -- it is not reopened here.

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

## Deliberately not built yet

- Rubric generation itself (typed generation/provenance models, the LLM-calling axis
  generators, atomicity/dedupe critic, weight classifier, Luna final review, deterministic
  selection) -- `docs/rag26-rubric-construction-execution-plan.md` Sections 6-19. This is the
  current workstream; see that plan's Section 27 for the exact build order and this file's
  git history / commit messages for how far it has actually progressed.
- Research-intent decomposition -- deciding what ~6-10 intents to search for per topic, and
  whether each is semantic/lexical/hybrid, needs an LLM call (plan Section 9's Stage A).
  `hybrid_retrieval.py` takes an already-produced `list[ResearchIntent]` via an injected
  `IntentPlanner`; tests use a deterministic dummy one.
- Persistent retrieval/research-pack and rubric-generation disk cache (plan's `cache.py`).
  `HybridResearchPack.cache_key()` gives the deterministic key a cache would use, but nothing
  disk/DB-backed exists yet.
- The frozen 119-topic rubric artifact (`artifacts/rag26/rubrics.jsonl`) and its
  `RAG26ArenaJudge` integration (`rubric-arena`, `rubric-support-arena` variants) --
  execution-plan Sections 21-25. Not started until the ResearchRubrics dev-set B0/B1/B2
  comparison (Section 20) passes its gate.
- ResearchRubrics fidelity evaluation (structural + Luna-matched semantic comparison against
  the organizer dev set) -- execution-plan Section 20.

See `docs/rag26-rubric-construction-execution-plan.md` Section 27 ("Immediate execution
order") for the full remaining sequence.
