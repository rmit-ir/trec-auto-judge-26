# TREC RAG 2026 Judge Implementation Plan

## Purpose

Implement a track-specialist AutoJudge for **TREC RAG 2026** whose design is aligned with the public 2026 evaluation information rather than with a generic RAG-quality rubric.

The core public signals available as of 2026-09-12 are unusually concrete:

1. TREC RAG 2026 evaluates submitted responses using **system-vs-system battles** with hidden identities, randomized presentation order, and A/B/tie decisions.
2. Each response also receives **individualized nugget/rubric scoring**.
3. Evidence support is evaluated with **weighted citation precision** and **weighted citation recall**.
4. The public RAGDoll repository implements closely related support evaluation, nugget/rubric assignment, and Arena-style pairwise ranking.
5. The official TREC RAG 2026 development-data repository publishes:
   - RAG 2025 development nuggets;
   - UMBRELA-style qrels;
   - ResearchRubrics-style development rubrics;
   - 2026 development topics.
6. This implementation may assume access to a **provided ClimbMix-compatible search index/service**, similar to the `trec-rag-26/tasks/search_serve` stack used elsewhere in our work. Treat this index as a first-class input for topic research, rubric generation, evidence discovery, and document resolution rather than as an optional enhancement.

The judge should therefore be optimized for the same high-level decomposition:

- answer preference;
- topic-specific rubric fulfillment;
- citation/evidence support;
- efficient system ranking.

This plan intentionally does **not** assume that the organizer will combine these components with a particular hidden weighted formula. The AutoJudge should expose separate measures. Use 2025 labels only for components whose semantics genuinely carry over (especially citation support, nugget/coverage behavior, robustness, and ranking mechanics); do not use 2025 report-level rankings to optimize the 2026-specific Arena/ResearchRubrics objective.

---

# 1. Scope and naming

Create a new track-specialist judge:

```text
judges/rag26_arena/
```

Recommended user-facing name:

```text
RAGree-RAG26
```

Recommended judge class:

```python
RAG26ArenaJudge
```

Recommended workflow:

```text
judges/rag26_arena/workflow.yml
```

The implementation should reuse mature code from `judges/ragree_choice/` where appropriate, especially:

- deterministic response ordering;
- sparse pairwise graph construction;
- Davidson/Bradley-Terry fitting;
- adaptive comparison selection;
- position-reversal checks;
- Condorcet/cycle/Kemeny diagnostics;
- bounded retry/failure handling.

Do **not** simply rename `ragree_choice`. The semantics of the track-specific pointwise and pairwise prompts must be changed to reflect the RAG 2026 public evaluation.

---

# 2. Data-handling constraints

Before any work involving evaluation data, Claude Code must read:

```text
AGENTS.md
CLAUDE.md
.claude/skills/autojudge-develop/SKILL.md
```

and the canonical TREC AutoJudge data-handling policy.

Important boundary:

- 2025 pilot data (`rag25`, `ragtime25`, `dragun25`) is unrestricted and may be used for development, shared-component tuning, ablations, and transfer diagnostics.
- TREC RAG 2025 did **not** use the same report-quality evaluation as the public 2026 specification: 2025 centered on nugget/sub-narrative coverage plus citation support, whereas 2026 adds system battles and individualized nugget/rubric scoring.
- 2026 evaluation reports outside the permitted window are restricted.
- The judge may process all 2026 reports, but the coding agent may not inspect restricted report contents or tune using restricted 2026 scores.
- Do not attempt to deanonymize run IDs.

Select 2026-specific Arena/rubric semantics from the public 2026 guidance, RAGDoll behavior, ResearchRubrics benchmark/development artifacts, and response-independent index-grounded validation. Use unrestricted 2025 data only where the metric semantics match, or explicitly label the result as a transfer diagnostic.

---

# 3. Public evaluator behavior to mirror

## 3.1 Pairwise preference

RAGDoll exposes an Arena-style pairwise evaluator whose public prompt asks which answer the user would rather receive, considering:

- intent match;
- directness;
- usefulness;
- completeness;
- specificity;
- plausibility/accuracy;
- uncertainty calibration;
- explanation quality;
- recommendation quality when applicable;
- organization;
- noise control.

It explicitly avoids rewarding verbosity, brevity, polished prose, or formatting by themselves.

The pairwise evaluator supports:

```text
A
B
Tie
Tie (Both Bad)
```

RAGDoll also exposes a rubric-conditioned version where topic-specific criteria inform the pairwise decision without becoming a rigid checklist.

The RAG26 AutoJudge should imitate this behavior closely.

## 3.2 Rubric scoring

The public 2026 ResearchRubrics examples include:

- positively weighted criteria;
- negatively weighted criteria;
- explicit and implicit requirements;
- instruction-following dimensions;
- synthesis requirements;
- communication-quality dimensions;
- citation/reference-quality dimensions.

The judge should therefore support richer criteria than simple factual nuggets.

## 3.3 Citation support

The public RAGDoll implementation exposes support labels equivalent to:

```text
Full Support    -> 1.0
Partial Support -> 0.5
No Support      -> 0.0
Missing/failure -> 0.0
```

and computes:

```text
weighted_precision_first_citation
weighted_recall_first_citation
weighted_precision_all_judged_citations
weighted_recall_all_judged_citations
hard_precision
hard_recall
```

The track guidelines state that an answer object without citations:

- is omitted from citation-precision scoring;
- receives support 0 for citation-recall scoring.

This asymmetry must be reproduced exactly in our RAG26 support measures.

---

# 4. High-level architecture

```text
                         Request / narrative
                                |
                +---------------+---------------+
                |                               |
                v                               v
      topic rubric / nugget bank       response collection
                |                               |
                |                  +------------+------------+
                |                  |                         |
                |                  v                         v
                |          isolated rubric score      citation support
                |                  |                         |
                +------------------+-------------------------+
                                   |
                                   v
                         pairwise Arena ballots
                                   |
                                   v
                         SPARROW acquisition
                                   |
                                   v
                      Davidson/Bradley-Terry fit
                                   |
                                   v
                              leaderboard
```

The components should remain observable as separate measures.

Do not collapse everything into one handcrafted weighted score during the initial implementation.

---

# 5. Proposed package structure

```text
judges/
├── common/
│   ├── __init__.py
│   ├── llm.py
│   ├── parsing.py
│   ├── retry.py
│   ├── citations.py
│   └── evidence.py
│
└── rag26_arena/
    ├── __init__.py
    ├── judge.py
    ├── models.py
    ├── prompts.py
    ├── rubric.py
    ├── support.py
    ├── arena.py
    ├── sparrow.py
    ├── workflow.yml
    └── README.md
```

Do not refactor existing working judges into `judges/common/` immediately. First implement the new judge with minimal duplication. Only extract shared helpers after both old and new tests are green.

---

# 6. Core data models

Implement internal Pydantic/dataclass structures similar to the following.

## 6.1 Rubric criterion

```python
class RubricCriterion(BaseModel):
    criterion_id: str
    text: str
    weight: float
    axis: Literal[
        "explicit",
        "implicit",
        "instruction_following",
        "synthesis",
        "communication",
        "citation_quality",
        "factual_hazard",
    ]
    polarity: Literal["positive", "negative"]
    critical: bool = False
```

## 6.2 Criterion assignment

```python
class CriterionAssignment(BaseModel):
    criterion_id: str
    verdict: Literal[
        "SATISFIED",
        "PARTIAL",
        "NOT_SATISFIED",
        "CONTRADICTED",
        "NOT_APPLICABLE",
    ]
    confidence: float
```

No free-form rationale should be required in leaderboard output.

A short structured evidence field may be retained internally on unrestricted development data, but must never be surfaced from restricted 2026 reports.

## 6.3 Citation support assignment

```python
class CitationSupport(BaseModel):
    sentence_index: int
    citation_index: int
    doc_id: str
    support: Literal["FULL", "PARTIAL", "NONE"]
```

## 6.4 Pairwise ballot

Reuse or adapt the current RAGree Choice representation:

```python
class ArenaBallot(BaseModel):
    candidate_a: str
    candidate_b: str
    vote: Literal["A", "B", "TIE", "BOTH_BAD", "ABSTAIN"]
    judge_id: str
    weight: float = 1.0
```

`BOTH_BAD` should be preserved as a distinct diagnostic label even if it maps to a tie for Bradley-Terry fitting.

---

# 7. Rubric-generation strategy

## 7.1 Primary design

Generate rubrics from:

- the request/narrative;
- **the provided ClimbMix-compatible search index and document store**;
- organizer-style development rubrics/nuggets as structural examples.

The default production rubric generator should be evidence-grounded. A prompt-only generator remains useful only as an ablation and fallback.

Do not derive the rubric from relative quality differences among submitted candidate responses. Do not use candidate answer text, candidate citations, candidate retrieval traces, or candidate-selected documents as search seeds when constructing the topic rubric.

This avoids candidate-pool leakage and produces a stable topic-level evaluation target.

## 7.1.1 Search-index abstraction

Implement a small index client behind an interface such as:

```python
class ClimbMixIndex(Protocol):
    def search(self, query: str, *, k: int, with_text: bool = True) -> list[SearchHit]: ...
    def search_batch(self, queries: list[str], *, k: int, with_text: bool = True) -> list[list[SearchHit]]: ...
    def get_document(self, docid: str) -> str | None: ...
    def get_documents(self, docids: list[str]) -> dict[str, str | None]: ...
```

Normalize search hits internally to:

```python
class SearchHit(BaseModel):
    docid: str
    rank: int
    score: float
    text: str | None = None
```

The adapter must not leak transport-specific details into rubric-generation or support-scoring code.

## 7.1.2 Expected REST contract

The provided index may expose an API similar to the existing `trec-rag-26/tasks/search_serve` service. Support this shape directly where possible:

```text
POST /search
  {"query": "...", "k": 20, "with_text": true,
   "complexity": 64, "beam_width": 2}

POST /search/batch
  {"queries": ["...", "..."], "k": 20, "with_text": true,
   "complexity": 64, "beam_width": 2}

GET /doc/{docid}
POST /doc/batch
GET /health
```

Expected search hit fields:

```json
{"docid": "shard_00042_1337", "rank": 1, "score": 0.8123, "text": "..."}
```

Keep `complexity` and `beam_width` transport-level tuning knobs optional in the abstract interface so a future BM25/Pyserini/other backend can implement the same logical client without exposing ANN-specific settings.

Configure the base URL/path through settings or environment, never hardcode it. Suggested names:

```text
RAG26_SEARCH_BASE_URL
RAG26_SEARCH_TIMEOUT_SECONDS
RAG26_SEARCH_API_KEY   # only if the provided service requires one
```

If the index is provided as a local library/path instead of HTTP, implement a second adapter with the same interface rather than changing downstream code.

## 7.1.3 Search usage policy

Use the index for **topic research** and **evidence resolution**, with strict separation from candidate responses:

```text
topic narrative
   -> query planner
   -> index search
   -> deduplicated evidence pool
   -> topic research pack
   -> rubric generation
```

Never:

```text
candidate answer/citations
   -> search
   -> rubric construction
```

Candidate citations may of course be resolved later by the citation-support lane because that lane is evaluating the candidate rather than defining the topic rubric.

## 7.1.4 Research-pack defaults

Start with approximately:

```text
8-12 topic-derived search queries
k=20 per query
batch search when available
union + deduplicate by docid
retain 30-50 high-value documents after reranking/selection
```

The research pack should capture explicit subquestions, implicit context, counterarguments, quantitative evidence, stakeholder impacts, historical/temporal context, and evidence that would invalidate simplistic conclusions.

Cache the research pack separately from rubric prompts so rubric-generation experiments do not repeat index retrieval.

## 7.2 Rubric content

Target approximately 15-30 criteria per topic.

The generator should explicitly consider:

1. required information;
2. important subquestions;
3. important distinctions/trade-offs;
4. explicit user constraints;
5. synthesis requirements;
6. common factual hazards;
7. irrelevant-but-tempting material;
8. evidence/citation requirements;
9. clarity/organization only where materially tied to the request.

## 7.3 Positive and negative criteria

Support both:

```text
positive criterion -> reward when satisfied
negative criterion -> penalty when violated
```

Negative criteria are particularly useful for:

- factual reversals;
- incorrect numerical claims;
- unsupported causal conclusions;
- irrelevant tangents;
- violating requested scope;
- fabricated references;
- contradictory conclusions.

## 7.4 Critical criteria

Some criteria should be flagged `critical=true`.

Examples:

- central factual correctness;
- explicit safety/legal constraint in the request;
- answer states the opposite of the evidence;
- central requested comparison entirely omitted.

Do not make all high-weight criteria critical.

## 7.5 Determinism

Rubric generation must be deterministic under the same:

- topic;
- model;
- prompt;
- settings;
- corpus input.

Use temperature 0 where supported.

Sort any evidence/doc lists before prompt construction.

Prompt caching must produce stable cache keys.

---

# 8. Isolated rubric scoring

Score each response independently against the topic rubric before any pairwise comparison.

## 8.1 Verdict mapping

Recommended numeric mapping for positive criteria:

```text
SATISFIED      1.0
PARTIAL        0.5
NOT_SATISFIED  0.0
CONTRADICTED   0.0
NOT_APPLICABLE excluded
```

For negative criteria:

```text
NOT_SATISFIED / NOT_APPLICABLE -> 0 penalty
PARTIAL                        -> 0.5 penalty
SATISFIED / CONTRADICTED*      -> full penalty
```

Define exact semantics carefully in code and tests.

Prefer naming negative assignments as `VIOLATED` internally if the generic verdict vocabulary becomes confusing.

## 8.2 Measures

Compute at least:

```text
RAG26_RUBRIC_SCORE
RAG26_RUBRIC_COVERAGE
RAG26_CRITICAL_COVERAGE
RAG26_NEGATIVE_VIOLATION_RATE
RAG26_CRITICAL_ERROR
```

`RAG26_RUBRIC_SCORE` should be normalized to `[0,1]`.

Keep the underlying formula documented in `MeasureSpec.description`.

## 8.3 No hard veto initially

The first implementation should **not** automatically exclude a response from pairwise voting because its rubric score or citation score is low.

The public evaluator separates preference, rubric, and citation-support signals.

Implement evidence gating only as an ablation/variant after baseline evaluation on RAG25.

---

# 9. Citation-support lane

## 9.1 Input normalization

Use:

```python
Report.get_sentences_with_citations()
```

Never manually assume a track-specific citation representation.

Resolve cited doc IDs to document text using the following priority order:

1. document text already available in the normalized `Report`/framework object;
2. the provided ClimbMix index `get_document(s)` adapter;
3. another framework-supported document-fetch mechanism.

The index document endpoint should therefore be reusable by both rubric research and citation-support evaluation. Do not issue search queries merely to resolve a known `docid`; use direct document fetch whenever available.

## 9.2 Support prompt

Judge each `(sentence, cited-document)` pair independently.

Output exactly one of:

```text
FULL
PARTIAL
NONE
```

The prompt should ask whether the cited document supports the **specific answer sentence**, not whether the document is relevant to the topic generally.

## 9.3 Strongest citation first

The RAG 2026 guideline tells generators to order citations strongest to weakest.

Therefore calculate both:

- first-citation support metrics;
- all-citation support metrics.

## 9.4 Public-RAGDoll-compatible calculations

Implement:

```text
weighted(FULL)    = 1.0
weighted(PARTIAL) = 0.5
weighted(NONE)    = 0.0
```

For first-citation precision:

```text
sum(first citation weighted support)
------------------------------------
number of judged sentences with citations
```

For first-citation recall:

```text
sum(first citation weighted support)
------------------------------------
number of answer objects/sentences
```

Thus uncited answer objects lower recall but do not enter precision.

For all-citation variants, use the mean support across valid citations attached to each answer object before aggregation.

Also expose hard variants where only `FULL=1`, everything else 0.

## 9.5 Measures

```text
RAG26_CIT_PREC_FIRST
RAG26_CIT_REC_FIRST
RAG26_CIT_PREC_ALL
RAG26_CIT_REC_ALL
RAG26_CIT_HARD_PREC
RAG26_CIT_HARD_REC
```

Optionally expose:

```text
RAG26_UNCITED_CLAIM_RATE
```

only if the implementation can distinguish factual content from purely structural headings without introducing unstable heuristics.

---

# 10. Pairwise Arena lane

## 10.1 Default semantics

The pairwise judge should answer:

> Which answer would the user rather receive for this request?

The decision should account for:

- actual intent;
- usefulness;
- completeness;
- specificity;
- plausibility;
- calibration;
- organization;
- important rubric criteria.

Do not ask the pairwise judge to mechanically recompute the isolated rubric score.

## 10.2 Prompt structure

Default inputs:

```text
request/narrative
rubric summary
answer A plain text
answer B plain text
```

Do not include:

- system names;
- run metadata;
- current BT scores;
- pointwise scores;
- citation support labels;
- citation passages.

The public RAGDoll Arena path compares answer text independently from citation-support evaluation; we should preserve that separation in the primary variant.

## 10.3 Verdicts

Accept:

```text
A
B
TIE
BOTH_BAD
```

If malformed after bounded retries:

```text
ABSTAIN
```

Do not silently convert malformed output to a winner.

## 10.4 Position reversal

Reuse current RAGree Choice reversal behavior:

- reverse only high-value already-cast ballots;
- map swapped A/B result back to original orientation;
- agreement keeps original ballot;
- genuine direction reversal collapses to `ABSTAIN`;
- failed reversal leaves original ballot untouched;
- record disagreement rate.

Also record `BOTH_BAD` independently from `TIE` for diagnostics.

---

# 11. SPARROW / sparse battle acquisition

Use the current sparse comparison machinery as the production ranking backend.

## 11.1 Candidate pool

Sort responses by `run_id` before any pair construction.

## 11.2 Seed graph

Create a connected graph with configurable degree:

```yaml
initial_graph_degree: 4
```

## 11.3 Adaptive comparisons

Use fitted uncertainty/top-k boundary sensitivity to add comparisons.

Default exploratory value:

```yaml
adaptive_comparisons_per_topic: 4
```

Do not tune this on 2026 data.

## 11.4 Small pools

For small candidate pools, support full round robin:

```yaml
full_round_robin_max_candidates: 5
```

## 11.5 Fitting

Reuse weighted Davidson-Bradley-Terry where ties are modeled explicitly.

Treat `BOTH_BAD` as a tie for the utility fit but retain its distinct diagnostic count.

## 11.6 Audits

Keep:

```text
CONDORCET_WINNER
RANKING_STABLE
CYCLE_RATE
BALLOT_COVERAGE
POSITION_REVERSAL_CHECKED
POSITION_REVERSAL_DISAGREEMENT_RATE
```

Add:

```text
BOTH_BAD_RATE
```

---

# 12. Leaderboard measures

Primary candidate measures:

```text
RAG26_ARENA_SCORE
RAG26_RUBRIC_SCORE
RAG26_RUBRIC_COVERAGE
RAG26_CRITICAL_COVERAGE
RAG26_CRITICAL_ERROR
RAG26_NEGATIVE_VIOLATION_RATE
RAG26_CIT_PREC_FIRST
RAG26_CIT_REC_FIRST
RAG26_CIT_PREC_ALL
RAG26_CIT_REC_ALL
RAG26_CIT_HARD_PREC
RAG26_CIT_HARD_REC
RAG26_PAIRWISE_WIN_RATE
RAG26_BOTH_BAD_RATE
```

Plus ranking diagnostics from existing RAGree Choice.

Every measure must have a precise `MeasureSpec.description` including:

- numeric range;
- aggregation semantics;
- whether higher/lower is better;
- caveats.

Do not make a combined weighted score the default initially.

---

# 13. Workflow configuration

Create a `workflow.yml` with all tunable settings outside Python code.

Suggested starting point:

```yaml
judge_class: "judges.rag26_arena.judge:RAG26ArenaJudge"

create_nuggets: false
create_qrels: false
judge: true
uses_llm: true

settings:
  filebase: "rag26-arena"

judge_settings:
  rubric_source: "generated"
  rubric_max_criteria: 24
  rubric_include_negative: true
  rubric_include_critical: true

  # Provided ClimbMix-compatible index. Prefer a pre-generated frozen rubric
  # artifact for final 2026 runs, but keep runtime generation supported.
  search_base_url: null
  search_timeout_seconds: 30
  search_k_per_query: 20
  search_query_count: 10
  search_research_doc_limit: 40
  search_complexity: 64
  search_beam_width: 2
  search_batch: true

  full_round_robin_max_candidates: 5
  initial_graph_degree: 4
  adaptive_comparisons_per_topic: 4
  top_k: 1
  position_reversal_budget: 2

  support_max_doc_chars: 12000
  arena_max_answer_chars: 12000
  llm_max_attempts: 3
  llm_retry_backoff_seconds: 0.25
```

Do not copy these values blindly if framework/API constraints differ; preserve the intent.

---

# 14. Required variants

Add named workflow variants for clean ablation.

## 14.1 `arena-only`

Pairwise Arena ranking without rubric conditioning or citation gating.

Purpose:

- baseline closest to generic RAGDoll Arena behavior.

## 14.2 `rubric-arena`

Pairwise Arena with topic rubric in the prompt.

Purpose:

- test whether rubric-conditioned battles improve alignment.

## 14.3 `rubric-support-arena`

Same pairwise ranking plus isolated rubric and citation-support output measures.

Do **not** make these scores affect ballots yet.

Purpose:

- produce all track-aligned diagnostics without changing ranking semantics.

## 14.4 `evidence-gated`

Optional later variant.

Exclude only clearly catastrophic responses from preference wins using a predefined gate.

Do not implement until unrestricted RAG25 results justify it.

## 14.5 `exhaustive-small`

```yaml
full_round_robin_max_candidates: 999
adaptive_comparisons_per_topic: 0
```

Purpose:

- reference ranking to evaluate SPARROW efficiency.

## 14.6 `no-reversal`

```yaml
position_reversal_budget: 0
```

Purpose:

- quantify order-bias correction benefit/cost.

---

# 15. Testing strategy

Create track-specific tests.

Recommended files:

```text
tests/test_rag26_arena_judge.py
tests/test_rag26_rubric.py
tests/test_rag26_support.py
tests/test_rag26_arena.py
tests/test_rag26_search_index.py
```

## 15.0 Search-index adapter tests

Before rubric-generation tests, verify the provided-index integration against a dummy/local HTTP service with the same wire contract used by the existing `trec-rag-26` search server.

Test:

- `POST /search` request shape and hit normalization;
- `POST /search/batch` preserves query order;
- direct `/doc/{docid}` fetch;
- `/doc/batch` partial-miss handling;
- timeouts/retries;
- 4xx/5xx failures become typed availability failures rather than empty evidence;
- deterministic deduplication by `docid`;
- no credentials/base URL hardcoded in source;
- research queries are topic-derived and never candidate-derived.

## 15.1 Rubric tests

Test:

- positive criterion scoring;
- partial credit;
- negative criterion penalties;
- critical criterion flagging;
- empty/invalid rubric failure;
- normalization to `[0,1]`;
- deterministic criterion ordering.

## 15.2 Citation tests

Construct synthetic sentences for:

1. full first citation;
2. partial first citation;
3. no-support first citation;
4. uncited sentence;
5. multiple citations with mixed support;
6. failed/missing judgment.

Verify exact RAGDoll-style metric calculations.

## 15.3 Arena parsing tests

Test exact parsing of:

```text
A
B
TIE
BOTH_BAD
ABSTAIN
malformed
```

## 15.4 Position-reversal tests

Reuse/extending existing RAGree Choice cases.

## 15.5 Failure policy

Ensure:

- pointwise support call failure does not crash whole run;
- pairwise malformed replies retry;
- exhausted pairwise failures become unavailable/abstain, not false losses;
- empty reports receive defined low scores;
- expected topics remain present in leaderboard output.

## 15.6 Endpoint contract

The new judge must pass `tests/test_endpoint_contract.py`.

No hardcoded provider, model, URL, or credentials.

---

# 16. Prompt-cache requirements

All prompts must be deterministic.

Rules:

- sort runs by `run_id`;
- sort rubric criteria by `criterion_id`;
- sort doc IDs before constructing support requests unless citation order is semantically required;
- preserve answer A/B randomization seed deterministically;
- do not include timestamps, UUIDs, object reprs, temp paths, or unordered dict renderings in prompts.

Use the existing Minima LLM cache behavior.

Do not inspect restricted 2026 prompt-cache entries.

---

# 17. Cost and runtime logging

Every full evaluation run should log safe aggregate metrics:

```text
rubric-generation calls
rubric-assignment calls
citation-support calls
pairwise calls
reversal calls
cache hits/misses
input tokens
output tokens
elapsed seconds
estimated cost
```

Do not log restricted answer text, citations, evidence passages, prompts, or completions.

Use the repo's existing cost tooling where possible.

---

# 18. Development-data evaluation

Use different development evidence for different components. There is **no single 2025 optimization target for the full RAG26 judge** because the report-evaluation design changed between TREC RAG 2025 and 2026.

The existing `docs/evaluation-2025-plan.md` remains the source of truth for 2025 split discipline, but its results must be interpreted by component.

Recommended progression:

```text
ResearchRubrics benchmark + organizer dev rubrics
    -> tune/validate rubric schema, axes, criterion generation,
       weight semantics, rubric grading, and rubric-conditioned behavior

public RAGDoll / TREC RAG 2026 guidance
    -> anchor Arena prompt semantics, tie policy, and support metrics

provided ClimbMix-style index
    -> generate response-independent topic research packs and rubrics

RAG25 Generation / AugGen
    -> tune only shared-semantics components (citation support,
       coverage/nugget behavior, retrieval quality) and run transfer diagnostics

RAGTIME25 + DRAGUN25
    -> optional robustness / cross-track transfer checks

RAG26
    -> blind execution only; no tuning from restricted outputs
```

A 2025 system-ranking improvement is informative but **not sufficient evidence** to change a 2026-specific Arena/rubric design.

---

# 19. 2025 validation and transfer questions

The evaluation harness should answer these separately and label each result by whether the semantics transfer to 2026.

## Question A: Do citation-support measures reproduce the shared 2025/2026 support semantics?

Validate Full/Partial/No Support classification and weighted precision/recall against available 2025 citation-support truth.

This is a legitimate shared-semantics tuning target.

## Question B: Does generated rubric content recover 2025 nuggets/sub-narratives?

Compare generated rubric criteria and evidence-pack content semantically against 2025 nugget/sub-narrative assessments.

Use this to diagnose topic-research coverage, not to force ResearchRubrics axes/weights to imitate the 2025 nugget scheme.

## Question C: Does SPARROW preserve exhaustive ranking?

Compare sparse vs exhaustive using the **same cached ballot source** where possible.

Measure:

```text
winner agreement
Kendall tau between sparse and exhaustive rankings
top-3 overlap
comparisons/topic
token cost
wall-clock
```

This evaluates the acquisition/ranking algorithm independently of whether the ballot semantics are 2025 or 2026.

## Question D: Does position reversal improve robustness?

Compare with/without reversal using the same ballot prompt and candidate set.

Measure:

```text
ranking changes
reversal disagreement rate
exhaustive-reference agreement where available
extra cost
```

Do not use 2025 report-level human correlation alone to decide the 2026 reversal policy.

## Question E: How do 2026-style Arena/rubric measures transfer to 2025?

Compare:

```text
arena-only
rubric-arena
pointwise generated-rubric score
```

Report Kendall/Spearman/pairwise score-induced ordering against official 2025 truth, but label these results **transfer diagnostics**, not direct estimates of 2026 battle accuracy.

Do not combine components into a production score until their own target semantics have been validated.

---

# 20. Model strategy

Initial implementation should work with the single OpenAI-compatible endpoint injected by AutoJudge.

Do not hardcode Bedrock-specific assumptions into the judge.

The architecture may retain `judge_id` and per-judge weights to support a future heterogeneous jury.

Future experiment:

```text
model A only
model B only
model C only
majority jury
calibrated weighted jury
```

First establish that the track-aligned prompts are stable and internally consistent with the public 2026/RAGDoll semantics using one endpoint. Use 2025 only as a transfer diagnostic for the Arena/rubric prompts, not as their primary selection target.

---

# 21. Optional heterogeneous jury extension

After the baseline judge is stable:

1. add an external orchestration path that can call multiple independent model families;
2. ensure each model receives identical canonicalized prompts;
3. store only structured verdicts in normal outputs;
4. calibrate judge weights only on data whose preference semantics match the weighting objective; do not learn 2026 Arena jury weights merely from 2025 report-quality rankings;
5. compare majority vs learned/calibrated weighting;
6. retain family-level disagreement as an uncertainty measure.

Do not treat repeated samples from one model as independent jury members.

---

# 22. Error and uncertainty policy

## Rubric generation failure

If no valid rubric can be generated after bounded retries:

- mark topic rubric unavailable;
- fall back to arena-only pairwise prompt;
- do not create an empty rubric that appears valid.

## Rubric assignment failure

- set availability diagnostic to 0;
- do not treat failure as criterion failure.

## Citation-support failure

- distinguish unavailable judgment from `NONE` internally;
- public numeric score may use 0 placeholder only if accompanied by an evaluated/availability measure.

## Pairwise failure

- retry;
- then `ABSTAIN`;
- do not add a fake tie or loss.

## Candidate with empty answer

- define deterministic low pointwise score;
- permit pairwise comparison to identify it as worse;
- never crash.

---

# 23. Recommended availability diagnostics

Add:

```text
RAG26_RUBRIC_EVALUATED
RAG26_SUPPORT_EVALUATED
RAG26_ARENA_COVERAGE
```

These prevent transient endpoint failures from being interpreted as genuine quality judgments.

---

# 24. Implementation phases

## Phase 0 - inspect and freeze interfaces

Tasks:

- read `ragree_choice` implementation and tests;
- read current AutoJudge workflow/API docs;
- identify reusable SPARROW functions;
- avoid modifying existing behavior.

Deliverable:

- short implementation note in PR/commit description;
- no functional changes yet.

## Phase 1 - scaffold

Create:

```text
judges/rag26_arena/
```

with importable class and minimal workflow.

Acceptance:

- tracked workflow imports;
- endpoint-contract test reaches injected endpoint;
- smoke test can produce a leaderboard on kiddie.

## Phase 2 - provided search-index integration

Implement the ClimbMix index abstraction and adapter before rubric generation.

Use the existing `trec-rag-26/tasks/search_serve` REST contract as the reference shape where applicable, but keep downstream code transport-agnostic.

Acceptance:

- `/health`, search, batch search, single-doc fetch, and batch-doc fetch supported when exposed by the provided service;
- response normalization covered by dummy-API tests;
- base URL/credentials injected from settings/environment;
- timeouts and service failures produce explicit unavailable states;
- no candidate-response data is used for topic-research queries.

## Phase 3 - citation support

Implement RAGDoll-compatible support metrics using direct `docid` fetch from the index where framework-provided document text is unavailable.

Acceptance:

- exact synthetic metric tests;
- empty citations handled correctly;
- direct document fetch is used for known ids rather than search;
- no report text emitted to leaderboard.

## Phase 4 - rubric models/generation

Implement rubric structures, parser, validation, the topic-research pack, and evidence-grounded generator. Follow the detailed plan in `docs/trec-rag-2026-researchrubrics-generation-plan.md`.

Acceptance:

- query plan derived only from topic/narrative;
- index retrieval cached by topic/query/config;
- deterministic research pack on cached replay;
- deterministic rubric on repeated cached runs;
- invalid/empty rubrics fail safely;
- positive/negative/critical criteria supported.

## Phase 5 - isolated rubric scoring

Implement criterion assignment and aggregate measures.

Acceptance:

- synthetic tests;
- score normalized;
- failures distinguish unavailable from bad.

## Phase 6 - Arena pairwise prompt

Implement RAGDoll-inspired preference semantics.

Acceptance:

- A/B/tie/both-bad parser;
- response ordering deterministic before randomized presentation;
- pairwise outputs contain no free text.

## Phase 7 - SPARROW integration

Adapt existing sparse election machinery.

Acceptance:

- sparse graph connected;
- exhaustive-small variant works;
- same ballots lead to deterministic fit;
- reversal tests pass.

## Phase 8 - ResearchRubrics + 2025 validation

Run two distinct validation streams:

1. **ResearchRubrics/public-2026 fidelity** for 2026-specific rubric/Arena semantics;
2. **RAG25 shared-component validation / transfer diagnostics** for citation support, coverage behavior, retrieval quality, and SPARROW mechanics.

Do not choose a 2026-specific Arena/rubric winner based solely on 2025 report-level correlation.

Deliver:

- ResearchRubrics structural/functional comparison;
- RAG25 shared-component result JSONL;
- transfer-correlation table clearly labelled as such;
- comparison-cost table;
- bootstrap CIs.

## Phase 9 - optimize and freeze by component

Only after Phase 8:

- adjust rubric-generation prompts using ResearchRubrics fidelity evidence;
- anchor Arena prompt/tie semantics to public RAGDoll/2026 behavior;
- tune citation-support/evidence components on matching 2025 labels;
- tune graph degree/adaptive comparisons against exhaustive rankings from the same ballot source;
- assess evidence-gate ablation without conflating 2025 ranking with 2026 battles;
- select and document the frozen RAG26 configuration from this combined evidence.

## Phase 10 - frozen 2026 execution

- freeze code/config;
- run on 2026 as allowed by AutoJudge policy;
- verify completeness;
- do not use 2026 outputs for tuning.

---

# 25. Acceptance criteria

The first production-ready RAG26 judge is complete when:

1. all repository tests pass;
2. endpoint-contract test passes;
3. kiddie smoke test passes;
4. ResearchRubrics validation and RAG25 shared-component/transfer evaluation complete;
5. all expected run/topic cells are present;
6. citation metrics reproduce synthetic RAGDoll semantics exactly;
7. pairwise ranking is deterministic under cached ballots;
8. position reversal is tested;
9. sparse ranking can be compared against exhaustive-small;
10. all measure descriptions are explicit;
11. restricted report text cannot appear in standard outputs/logs;
12. chosen production variant is selected only from permissible development evidence, with 2026-specific semantics justified by ResearchRubrics/public-2026 sources rather than 2025 ranking optimization.

---

# 26. Performance targets for development

These are engineering targets, not organizer guarantees.

Aim for:

- strong structural/functional fidelity to organizer-shared ResearchRubrics on held development prompts;
- useful RAG25 correlation for shared support/coverage components, without requiring 2026-specific Arena/rubric scores to optimize 2025 rankings;
- rubric-conditioned Arena to remain stable and useful under ResearchRubrics/public-2026 validation; treat RAG25 arena-vs-rubric differences as transfer evidence only;
- sparse SPARROW winner agreement >= 95% with exhaustive comparison on development topics where exhaustive is feasible;
- sparse SPARROW Kendall tau >= 0.95 vs exhaustive ranking where candidate counts permit;
- position-reversal disagreements low enough that reversal correction is worth its extra calls;
- citation-support measures substantially correlated with official citation-related truth measures where available;
- no surprising collapse on RAG25 AugGen transfer checks relative to Generation.

Do not force these targets by tuning on 2026, and do not force 2026-specific rubric/Arena semantics to fit the different 2025 evaluation objective.

---

# 27. Things explicitly not to do

Do not:

- optimize directly on restricted RAG26 scores;
- inspect restricted RAG26 reports;
- identify anonymous systems;
- hardcode model/provider credentials;
- create one opaque weighted quality score before component validation;
- make pointwise evidence failure equivalent to factual failure;
- let answer length alone determine preference;
- include current leaderboard rank in pairwise prompts;
- treat multiple samples from one model as a heterogeneous jury;
- log prompts/completions containing restricted report text;
- rewrite working `ragree_choice` until the new judge has independent tests.

---

# 28. Public references to keep beside implementation

These are external design references, not dependencies that must be imported.

## TREC RAG

```text
https://trec-rag.github.io/
https://github.com/TREC-RAG/trec-rag-skills
https://github.com/TREC-RAG/trec-rag-data
```

Important public 2026 task reference:

```text
TREC-RAG/trec-rag-skills
skills/trec-rag-2026-track-guidelines/references/rag-task.md
```

## RAGDoll

```text
https://github.com/castorini/RAGDoll
```

Relevant modules:

```text
src/ragdoll/arena/prompts.py
src/ragdoll/arena/stages.py
src/ragdoll/support/metrics.py
src/ragdoll/nuggetizer/metrics.py
```

## Development data

```text
TREC-RAG/trec-rag-data/trec-rag-2026/development-data/
```

contains public examples of:

```text
rag25-dev-nuggets
rag25-dev-umbrela-qrels
researchrubrics-dev-rubrics
topics
```

---

# 29. Suggested first Claude Code task

When this plan is handed to an implementation agent, start with:

1. inspect `judges/ragree_choice/` and identify reusable ranking components;
2. scaffold `judges/rag26_arena/` without changing old judge behavior;
3. implement and fully test the RAGDoll-compatible citation support metric layer;
4. implement Arena verdict parsing and a minimal `arena-only` variant;
5. run tests and kiddie smoke test;
6. only then implement rubric generation/scoring.

This order produces a usable baseline early and isolates the most track-specific components behind tests before adding expensive rubric generation.

---

# 30. Final design principle

The RAG26 judge should predict the **observable organizer evaluation process**, not an abstract universal notion of RAG quality.

The working hypothesis for the 2026 track is:

```text
Arena-style human preference
        +
topic-specific rubric fulfillment
        +
citation support precision/recall
        +
efficient sparse pairwise ranking
```

is a better predictor of TREC RAG 2026 evaluation than a single pointwise relevance/factuality prompt.

The implementation must preserve these components separately. Combination decisions should use matching development evidence: ResearchRubrics/public-2026 semantics for Arena/rubric components, 2025 truth for shared citation/coverage components, and exhaustive-vs-sparse comparisons for SPARROW mechanics. Do not infer the final 2026 combination from 2025 report-level rankings alone.
