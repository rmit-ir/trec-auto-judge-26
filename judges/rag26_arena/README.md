# RAG26 Arena / ResearchRubrics -- foundation

Status as of 2026-09-12: **foundation only**. No `workflow.yml`, no judge class, no LLM calls
yet. This directory currently holds the lowest-risk pieces from the two design docs:

- `docs/trec-rag-2026-judge-implementation-plan.md`
- `docs/trec-rag-2026-researchrubrics-generation-plan.md`

## What's here

- `rubrics/schema.py` -- organizer-compatible `RubricCriterion`/`RubricTopic` (the exact
  `research-rubrics-dev-rubrics.jsonl` shape: `qid`, `domain`, `conceptual_breadth`,
  `logical_nesting`, `exploration`, `rubrics: [{criterion, weight, axis}]`). Deliberately
  permissive at parse time -- see the module docstring for why (the real organizer dev file has
  a non-canonical axis value and a zero weight; `validate.py` is the strict side, not this).
- `rubrics/weights.py` -- deterministic importance-class -> weight mapping (plan Section 8): a
  model classifies ordinal importance, this maps it onto the -5..5 scale, so weight assignment
  never drifts by model or call.
- `rubrics/validate.py` -- quality gates a *generated* topic must pass (plan Section 46). Not
  applied to the organizer dev file, which is reference data, not our output.
- `rubrics/io.py` -- load/write organizer-compatible JSONL, deterministic ordering.
- `rubrics/stats.py` + `scripts/analyze_researchrubrics.py` -- Stage 0 structural statistics
  over the dev file; see `evaluation/researchrubrics/dev-rubric-stats.md` for the actual
  numbers measured on it.
- `rubrics/topics.py` -- Stage 1: load/normalize the public TREC RAG topic file into `Topic(qid,
  narrative)`. No candidate response data enters here.
- `rubrics/research.py` -- single-backend research-pack pipeline (kept for the simple/dummy
  case): dedupes an already-produced query list against one `SearchClient`, unions hits by
  `docid`, produces a deterministic, cache-keyed `ResearchPack`.
- `search_index.py` -- the per-backend HTTP adapters. See below.
- `hybrid_retrieval.py` -- the dense+sparse hybrid retriever: routes research intents to one or
  both backends, normalizes both wire formats, derives parent-document ids, fuses with
  deterministic RRF, and builds a `HybridResearchPack` with full per-hit provenance. This is the
  pipeline the live endpoints below actually exercise.
- `scripts/live_search_smoke.py` -- manual live smoke check (not part of the offline suite).

## Search-index clients

Two concrete HTTP adapters, matching the two real backends **verified live** on 2026-09-12
(corrected from an earlier, wrong assumption -- see "What changed" below):

- `SearchServeClient` -- the FastAPI `trec-rag-26/tasks/search_serve` dense/ANN contract
  (`POST /search`, `POST /search/batch`, `GET /doc/{docid}`, `POST /doc/batch`, `GET /health`).
  Field-for-field verified against the live service.
- `SparseBM25Client` -- `POST /api/search` only, an Elasticsearch-shaped response
  (`hits.hits[]._id/_score/_source.contents`, no `rank` field -- derived from array position). No
  batch endpoint, no per-document-fetch endpoint on this backend at all; `health()` uses
  `/actuator/health` (Spring Boot). Field-for-field verified against the live service.
- `DummySearchClient` -- in-memory, for tests and the prompt-only ablation/fallback.

Nothing is hardcoded: `dense_client_from_env()` reads `RAG26_DENSE_SEARCH_URL` (+ optional
`RAG26_DENSE_SEARCH_API_KEY`), `sparse_client_from_env()` reads `RAG26_SPARSE_SEARCH_URL` (+
optional `RAG26_SPARSE_SEARCH_API_KEY`); both share `RAG26_SEARCH_TIMEOUT_SECONDS`. Both are
needed simultaneously (the hybrid retriever, not a backend choice) --
`hybrid_retriever_from_env()` in `hybrid_retrieval.py` builds both at once.

### What changed since the first draft of this README

The first version of this module assumed the sparse backend would be the public castorini
`pyserini-rest-api` (`GET /v1/{index}/search`, `candidates[]`) based on the sibling `trec-rag-26`
checkout's skill docs, and used a single `RAG26_SEARCH_BASE_URL`/`RAG26_SEARCH_BACKEND` pair to
select one active backend. The user then supplied the actual live contract: the real sparse
service is a different, Elasticsearch-shaped `/api/search` (see `search_index.py`'s
`SparseBM25Client` docstring), and both dense and sparse run simultaneously for hybrid retrieval,
not as alternatives. `PyseriniRestClient`, `SearchConfig`/`load_search_config`/
`search_client_from_env` were removed rather than kept alongside as dead code, and replaced with
`SparseBM25Client` and the dual-URL factories above.

### Live endpoint status (2026-09-12, corrected)

Both backends verified live and working end-to-end (`scripts/live_search_smoke.py`):

- **Dense, `http://127.0.0.1:8088`** (the user's earlier "8888" was a typo) -- `/health` reports
  `n_docs: 921,892,634`, DiskANN index over a 256-d `jina-v5-nano-trecrag26-agent-256d` encoder.
  `/search`, `/search/batch`, `/doc/{docid}` all verified against real queries.
- **Sparse, `http://127.0.0.1:8085`** -- Lucene BM25 (`k1=0.9, b=0.4`) over the same corpus,
  chunked identically (`shard_XXXXX_YYYYY_pN`). `/api/search` and `/actuator/health` verified
  against real queries.

Building `SparseBM25Client` against the live service (before the corrected contract was known)
surfaced a real bug, fixed and regression-tested
(`test_sparse_route_not_mounted_becomes_unavailable_not_empty` in
`tests/test_rag26_search_index.py`): a 404 from an unmounted route must not look like a valid
zero-hit search. The fix: any non-200 on `/api/search` raises `SearchUnavailableError`; a
genuine empty result is a 200 with `hits.hits: []` (also verified live).

See `scripts/live_search_smoke.py`'s output in the session report for example normalized hits
from both backends and the fused hybrid result.

## Deliberately not built yet

- **Research-intent decomposition itself** -- deciding what ~5-10 intents to search for, and
  whether each is semantic/lexical/hybrid, needs an LLM call. `hybrid_retrieval.py` takes an
  already-produced `list[ResearchIntent]` via an injected `IntentPlanner`; tests use a
  deterministic dummy one. This is the next real piece of "wire the research pack into
  ResearchRubrics generation".
- **Persistent retrieval/research-pack cache store.** `HybridResearchPack.cache_key()` gives the
  deterministic key (topic hash + intents + backend identities + config + builder version) that
  a cache would use, but no disk/DB-backed cache was built -- retrieval itself is fast and free
  (no LLM calls yet), so there's nothing expensive to cache yet. Wire this to `CACHE_DIR` /
  Minima's cache behavior once rubric-generation prompts (which *are* expensive) are added.
- `judges/rag26_arena/judge.py`, `workflow.yml`, `RAG26ArenaJudge` (judge-plan Phase 1).
- Rubric *generation* (the LLM-calling axis generators, reviewers, weight-assignment calls --
  judge-plan Phase 4; researchrubrics-plan Stages 2, 4-14) and reranking/evidence-quality
  filtering of the fused research pack (currently: RRF rank + `doc_limit` truncation only, no
  relevance reranking on top).
- Citation-support lane, Arena pairwise judge, SPARROW integration (judge-plan Phases 3, 6, 7).
- ResearchRubrics fidelity evaluation (structural bipartite matching against expert criteria)
  and the RAG25 shared-component transfer study (judge-plan Phase 8).

See the two plan documents' "Implementation phases" / "Recommended implementation sequence"
sections for the full remaining order.
