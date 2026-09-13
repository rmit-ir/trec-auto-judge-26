# RAGTIME 2026 Judge Implementation Plan

## Purpose

Implement a track-specialist AutoJudge for **RAGTIME 2026** whose design is optimized for multilingual report-generation evaluation, especially the balance between:

- coverage of the important information requested by the topic;
- support of report claims by cited evidence;
- citation relevance and citation correctness;
- whole-report alignment with the request.

Unlike the TREC RAG 2026 judge, this judge should **not** use pairwise Arena ranking as its primary mechanism.

The best public prior is the **ARGUE / Auto-ARGUE** evaluation family, which evaluates generated reports sentence-by-sentence and combines information coverage with support/citation correctness. RAGTIME 2026 also makes auto-nuggetization/request decomposition a first-class task, reinforcing the importance of a question/nugget-based evaluator.

This plan therefore proposes a new judge rather than a thin variant of `ragree_choice`.

Recommended user-facing name:

```text
RAGree-RAGTIME26
```

Recommended package:

```text
judges/ragtime26_argue/
```

Recommended judge class:

```python
RAGTIME26ArgueJudge
```

---

# 1. Confirmed public information vs informed inference

## 1.1 Confirmed public information

As of 2026-09-12, public RAGTIME 2026 information establishes that:

- the task is report generation from multilingual news/evidence;
- report requests and generated reports are in English;
- evidence sources include English, Spanish, Chinese, and Russian;
- evaluation is citation-based;
- RAGTIME 2026 includes an Auto-nuggetization / Request Decomposition task where systems produce single-sentence questions that should be answered by a good report.

## 1.2 Strong prior, but do not present as confirmed final 2026 scoring formula

The public Auto-ARGUE implementation and prior report-generation evaluations emphasize:

- `nugget_coverage` / nugget recall;
- `sentence_support`;
- F1 between nugget coverage and sentence support;
- importance-weighted nugget coverage;
- citation support;
- citation relevance;
- correctly cited sentences;
- missing citation detection;
- macro/micro variants.

These are the strongest available design targets, but the final hidden 2026 organizer formula may differ.

Therefore:

- implement these signals explicitly;
- expose them separately;
- use RAGTIME25 ground truth to select the best predictor;
- do not assert that `f1_weighted` is definitely the official 2026 score unless organizers publish that fact later.

---

# 2. Data-handling constraints

Before implementation, Claude Code must read:

```text
AGENTS.md
CLAUDE.md
.claude/skills/autojudge-develop/SKILL.md
```

and the full canonical TREC AutoJudge data-handling policy.

Important boundary:

- `ragtime25` is unrestricted pilot/development data and may be inspected/tuned against truth.
- RAGTIME26 evaluation reports outside the permitted window are restricted.
- The judge may process all RAGTIME26 reports.
- Coding agents must not inspect or derive development decisions from restricted report contents.
- Never attempt to deanonymize run IDs.

All prompt design, threshold setting, model selection, weighting, and ablation decisions must be made using unrestricted 2025 data or synthetic fixtures.

---

# 3. Core design hypothesis

The judge should estimate the same two properties that matter most for a high-quality evidence-grounded report:

```text
                  quality report
                        |
            +-----------+-----------+
            |                       |
            v                       v
    information coverage      evidential support
            |                       |
            v                       v
     nugget/question recall    sentence/citation precision
            +-----------+-----------+
                        |
                        v
                  balanced score
```

A report that is perfectly supported but omits most required information should not score highly.

A report that covers every requested point but contains unsupported or wrongly cited claims should not score highly either.

This is exactly why an F1-like balance is a strong design prior.

---

# 4. High-level architecture

```text
                        Report request
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
       multilingual evidence search   request analysis
                 |                       |
                 +-----------+-----------+
                             |
                             v
                    QA nugget generation
                             |
                    dedupe + importance
                             |
                             v
                       Nugget bank
                             |
                             v
                     Submitted report
                             |
              +--------------+--------------+
              |                             |
              v                             v
      sentence/citation judge        nugget-answer judge
              |                             |
              +--------------+--------------+
                             |
                             v
                 support / coverage metrics
                             |
                             v
                  request-alignment check
                             |
                             v
                        leaderboard
```

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
└── ragtime26_argue/
    ├── __init__.py
    ├── judge.py
    ├── models.py
    ├── prompts.py
    ├── nuggets.py
    ├── multilingual.py
    ├── citations.py
    ├── argue.py
    ├── scoring.py
    ├── workflow.yml
    └── README.md
```

Do not perform a broad common-code refactor as the first step.

First implement the new judge independently enough to keep existing RAGree tests stable. Extract helpers only after both systems are green.

---

# 6. Nugget model

The central object should be a QA-style nugget rather than a vague topic criterion.

Recommended internal model:

```python
class RAGTIMENugget(BaseModel):
    nugget_id: str
    question: str
    answers: list[str]
    aggregator: Literal["AND", "OR"]
    importance: Literal["VITAL", "OKAY"]
    supporting_documents: list[str]
    languages: list[str]
```

Optional development-only fields:

```python
supporting_passages: list[str]
source_confidence: float
```

Do not expose supporting passage text in normal leaderboard output.

---

# 7. Nugget-generation strategy

## 7.1 Primary design

Generate nugget questions from both:

- the report request;
- retrieved multilingual source evidence.

Do not generate nuggets solely from candidate participant responses.

The generator should identify the pieces of information a strong report should answer.

## 7.2 Retrieval before decomposition

Recommended flow:

```text
request
  -> multilingual retrieval
  -> candidate documents/passages
  -> event/entity/fact extraction
  -> candidate questions
  -> deduplication
  -> importance classification
  -> final nugget bank
```

The document-grounded step is important because the request alone may imply information needs whose actual answer structure is only visible after evidence exploration.

## 7.3 Nugget question quality rules

Each nugget should:

- be a single-sentence question;
- capture one coherent information need;
- be answerable from available evidence;
- avoid overlapping heavily with another nugget;
- avoid trivial background facts unless materially required;
- preserve important named entities, dates, quantities, or causal relationships;
- have explicit acceptable answers or answer conditions where possible.

## 7.4 AND vs OR nuggets

Support Auto-ARGUE-like semantics:

- `OR`: at least one acceptable answer satisfies the nugget;
- `AND`: all required subanswers must be present.

Example:

```text
Question: Which two agencies jointly announced the program?
Aggregator: AND
Answers: [Agency A, Agency B]
```

versus:

```text
Question: Which city hosted the meeting?
Aggregator: OR
Answers: [Kyiv, Kiev]
```

## 7.5 Nugget importance

Initial baseline:

```text
VITAL weight = 2.0
OKAY  weight = 1.0
```

matching the public Auto-ARGUE prior.

Importance assignment should consider:

- explicit emphasis in the request;
- whether omission prevents answering the central question;
- whether the fact resolves a key ambiguity;
- whether it is necessary for the requested chronology/comparison/causal explanation;
- whether multiple independent sources support its importance.

Do not use participant response frequency as an importance signal in the default judge.

---

# 8. Multilingual evidence strategy

## 8.1 Languages

RAGTIME 2026 evidence may come from:

```text
English
Spanish
Chinese
Russian
```

The report itself is English.

## 8.2 Preserve native text

Whenever possible retain:

```text
native source passage
normalized English translation/representation
```

Do not throw away the native evidence after translation.

Named entities, dates, amounts, hedging, attribution, and negation are particularly vulnerable to translation errors.

## 8.3 Support-judge prompt

For non-English evidence, present:

```text
English report sentence
native source passage
English translation/summary if available
```

and ask the judge to decide support.

The prompt must explicitly instruct the model to resolve conflicts in favor of the native/source wording where translation appears lossy.

## 8.4 Translation strategy

Do not translate the entire corpus eagerly.

Prefer:

1. retrieve in native collection;
2. select evidence passage;
3. translate only selected evidence when needed;
4. cache translations by stable document/passage key.

This limits cost and reduces unnecessary transformation.

---

# 9. Sentence-level ARGUE logic

Use `Report.get_sentences_with_citations()` to normalize report sentences and citations.

For every sentence, apply different logic depending on whether it has citations.

---

# 10. Sentences with citations

For every cited sentence:

## Step 1 - citation relevance

Ask whether the cited document contains information relevant to the report request / nugget bank.

Output:

```text
RELEVANT
NOT_RELEVANT
UNAVAILABLE
```

This is not the same as sentence support.

A source can be relevant to the topic but fail to support the attached claim.

## Step 2 - sentence attestation/support

For every citation attached to the sentence, determine whether it supports the sentence.

Recommended label set:

```text
FULL
PARTIAL
NONE
UNAVAILABLE
```

## Step 3 - nugget answering

Only after evidence support is assessed, determine which nugget questions the sentence correctly answers.

A nugget should receive answer credit only when:

- the sentence actually supplies an answer;
- the answer is correct;
- required evidence is sufficiently supported.

This guards against rewarding unsupported but plausible prose.

---

# 11. Sentences without citations

For every uncited sentence:

## Step 1 - requires citation

Judge whether the sentence makes a factual claim that should be supported.

Output:

```text
YES
NO
UNAVAILABLE
```

Structural prose such as headings, transitions, and purely organizational text should generally be `NO`.

## Step 2 - first instance

If citation is required, determine whether the factual content is novel or merely repeats a fact that was properly cited earlier.

Output:

```text
FIRST_INSTANCE
REPEATED
UNAVAILABLE
```

Only first-instance unsupported factual claims should receive the strongest missing-citation penalty.

## Step 3 - negative assertions

Implement negative-assertion handling as an optional later variant.

Examples:

```text
"No evidence was found that ..."
"The identity has not been confirmed."
```

These can be valid report content but require careful evidence treatment.

Do not block initial implementation on this feature.

---

# 12. Nugget-answer assignment

For each nugget, determine whether the report contains a correct answer.

Do not simply compare the whole report semantically to the question.

Preferred evaluation unit:

```text
nugget x report sentences
```

The system should identify one or more supporting report sentences for each matched nugget.

Recommended verdicts:

```text
FULL
PARTIAL
NONE
CONTRADICTED
UNAVAILABLE
```

## 12.1 Strict coverage

```text
FULL = 1
PARTIAL = 0
NONE = 0
CONTRADICTED = 0
```

## 12.2 Soft coverage

```text
FULL = 1.0
PARTIAL = 0.5
NONE = 0.0
CONTRADICTED = 0.0
```

Evaluate both against RAGTIME25 truth before choosing a primary production measure.

## 12.3 AND semantics

For `AND` nuggets, all required answer components must be present for FULL credit.

Partial subanswer may receive PARTIAL under the soft variant.

## 12.4 OR semantics

For `OR` nuggets, one correct acceptable answer is enough for FULL credit, provided no conflicting answer is presented.

---

# 13. Core metrics

Implement Auto-ARGUE-compatible metrics first.

## 13.1 Counts

```text
sentences
character_count
sentences_with_citations_or_requiring_citations
correctly_cited_sentences
first_instance_sentences_missing_citation
nuggets
nuggets_weighted
correct_nuggets
correct_nuggets_weighted
citations
relevant_citations
supporting_citations
```

## 13.2 Citation metrics

```text
citation_relevance
citation_support
```

Definitions should be explicit in code and measure descriptions.

## 13.3 Sentence support

Core support signal:

```text
sentence_support
```

Operationally this should estimate the proportion of scored factual sentences that are correctly supported.

Match Auto-ARGUE behavior as closely as practical for the initial baseline.

## 13.4 Nugget coverage

```text
nugget_coverage
nugget_coverage_weighted
```

Unweighted:

```text
correct unique nuggets / total nuggets
```

Weighted:

```text
sum(weights of correctly answered nuggets)
------------------------------------------
sum(weights of all nuggets)
```

with starting weights:

```text
VITAL = 2
OKAY = 1
```

## 13.5 F1

```text
f1 = harmonic_mean(sentence_support, nugget_coverage)
```

and:

```text
f1_weighted = harmonic_mean(
    sentence_support,
    nugget_coverage_weighted,
)
```

Handle `0/0` deterministically as 0.

## 13.6 Macro and micro variants

Support run-level:

```text
citation_relevance_micro
citation_relevance_macro
citation_support_micro
citation_support_macro
sentence_support_micro
sentence_support_macro
nugget_coverage_micro
nugget_coverage_macro
nugget_coverage_weighted_micro
nugget_coverage_weighted_macro
f1_micro
f1_macro
f1_weighted_micro
f1_weighted_macro
```

Only expose variants AutoJudge can aggregate correctly and clearly document.

---

# 14. Additional RAGTIME-specific metrics

Add these separately rather than altering the core Auto-ARGUE-compatible measures.

## 14.1 Request alignment

```text
RAGTIME_REQUEST_ALIGNMENT
```

Question:

> Does the report as a whole address the actual information need and requested scope?

This catches reports that contain many correct facts but answer the wrong question.

Output normalized `[0,1]` or a structured ordinal scale mapped deterministically.

## 14.2 Contradiction rate

```text
RAGTIME_CONTRADICTION_RATE
```

Measure the fraction of factual/scored report sentences that conflict with cited evidence or strongly supported corpus evidence.

Keep contradiction separate from mere missing support.

## 14.3 Missing citation rate

```text
RAGTIME_MISSING_CITATION_RATE
```

Use only first-instance sentences judged to require citation.

Do not penalize headings or repeated already-cited information.

## 14.4 Soft nugget coverage

```text
RAGTIME_NUGGET_COVERAGE_SOFT
RAGTIME_NUGGET_COVERAGE_WEIGHTED_SOFT
```

Useful as an ablation if partial credit correlates better with human assessments.

---

# 15. Leaderboard measure names

Recommended public AutoJudge measure names:

```text
RAGTIME_NUGGET_COVERAGE
RAGTIME_NUGGET_COVERAGE_WEIGHTED
RAGTIME_NUGGET_COVERAGE_SOFT
RAGTIME_NUGGET_COVERAGE_WEIGHTED_SOFT
RAGTIME_SENTENCE_SUPPORT
RAGTIME_F1
RAGTIME_F1_WEIGHTED
RAGTIME_CITATION_SUPPORT
RAGTIME_CITATION_RELEVANCE
RAGTIME_CORRECTLY_CITED_SENTENCES_RATE
RAGTIME_MISSING_CITATION_RATE
RAGTIME_REQUEST_ALIGNMENT
RAGTIME_CONTRADICTION_RATE
```

Add availability/evaluation diagnostics:

```text
RAGTIME_NUGGETS_EVALUATED
RAGTIME_SUPPORT_EVALUATED
RAGTIME_REQUEST_ALIGNMENT_EVALUATED
```

Every `MeasureSpec` must explain:

- range;
- direction;
- denominator;
- behavior on unavailable LLM calls.

---

# 16. Workflow configuration

Suggested initial workflow:

```yaml
nugget_class: "judges.ragtime26_argue.judge:RAGTIME26ArgueJudge"
judge_class: "judges.ragtime26_argue.judge:RAGTIME26ArgueJudge"

create_nuggets: true
create_qrels: false
judge: true
nugget_depends_on_responses: false
judge_uses_nuggets: true
uses_llm: true

settings:
  filebase: "ragtime26-argue"

nugget_settings:
  max_nuggets: 24
  vital_weight: 2.0
  okay_weight: 1.0
  use_multilingual_evidence: true

judge_settings:
  partial_nugget_credit: 0.5
  max_document_chars: 12000
  max_report_chars: 30000
  llm_max_attempts: 3
  llm_retry_backoff_seconds: 0.25
```

Adjust schema to actual AutoJudge API requirements when implementing.

Important lifecycle intent:

- nugget creation should depend on topic/request/corpus evidence, not participant responses;
- generated nugget banks should be reusable across runs;
- judging consumes the same bank for all candidate reports for a topic.

---

# 17. Required workflow variants

## 17.1 `support-only`

No nugget coverage contribution to the primary experimental score.

Expose sentence/citation measures.

Purpose:

- establish how much human alignment comes from grounding alone.

## 17.2 `coverage-only`

Nugget generation and coverage scoring, ignoring support in the headline measure.

Purpose:

- quantify coverage's independent value.

## 17.3 `argue-f1`

```text
nugget_coverage + sentence_support -> f1
```

Purpose:

- direct Auto-ARGUE-style baseline.

## 17.4 `argue-f1-weighted`

Use importance-weighted coverage.

Purpose:

- test whether VITAL/OKAY weighting improves RAGTIME25 correlation.

## 17.5 `soft-coverage`

Give PARTIAL nugget matches 0.5 credit.

Purpose:

- test whether graded completeness better matches human assessments.

## 17.6 `request-aligned`

Expose/use request alignment alongside F1.

Do not choose a final fusion weight until validated on unrestricted 2025 folds.

## 17.7 `query-only-nuggets`

Generate nuggets from request only, without corpus evidence.

Purpose:

- isolate the value/cost of document-grounded nugget generation.

---

# 18. Prompt design

Keep judgment prompts narrow rather than using one giant rubric prompt.

Separate prompts for:

```text
nugget_generation
nugget_importance
citation_relevance
sentence_attested
requires_citation
first_instance
sentence_answers_nugget
request_alignment
```

This mirrors Auto-ARGUE's decomposed decision approach and makes errors diagnosable.

## 18.1 Output format

Prefer strict structured labels or JSON.

Do not require verbose model rationales.

Examples:

```json
{"support":"FULL"}
```

```json
{"requires_citation":true}
```

```json
{"nugget":"n7","verdict":"PARTIAL"}
```

This improves parse reliability and reduces cost.

---

# 19. Batching strategy

LLM volume can become large because the naïve Cartesian product is:

```text
reports x sentences x citations x nuggets
```

Avoid blindly evaluating all combinations.

Recommended cascade:

## Stage 1 - cheap deterministic filtering

- normalize citations;
- skip structural headings;
- match cited docs by ID;
- reuse document/nugget relevance cache;
- dedupe identical evidence judgments.

## Stage 2 - candidate nugget routing

For each sentence, shortlist likely nuggets using a cheap semantic/router call or deterministic retrieval over nugget questions.

Then run expensive `sentence_answers_nugget` only on top candidates.

## Stage 3 - full judge

Use the main LLM only for ambiguous semantic checks.

The routing mechanism itself must be evaluated on unrestricted data to ensure it does not hurt nugget recall.

---

# 20. Cache architecture

Cache by stable semantic keys.

Examples:

```text
nugget-bank:<topic-id>:<generator-config-hash>
translation:<doc-id>:<passage-id>:<translation-model>
citation-support:<sentence-hash>:<doc-id>:<judge-model>
nugget-match:<sentence-hash>:<nugget-id>:<judge-model>
request-alignment:<report-hash>:<topic-id>:<judge-model>
```

Within AutoJudge/Minima, prompts still need deterministic rendering for normal cache reuse.

Do not inspect prompt-cache entries derived from restricted 2026 reports.

---

# 21. Failure policy

Failure handling must distinguish **unavailable** from **bad**.

## Nugget generation failure

After bounded retries:

- fail the topic loudly rather than silently producing an empty nugget bank;
- the AutoJudge runner already treats empty nugget banks as invalid.

## Citation relevance/support failure

- mark unavailable internally;
- do not convert provider failure into `NONE` without an availability diagnostic.

## Nugget assignment failure

- leave nugget unjudged/unavailable;
- do not automatically count it as missing information unless scoring policy explicitly documents that fallback.

## Request alignment failure

- do not treat unavailable as 0 quality;
- expose evaluated flag.

## Empty report

- coverage = 0;
- support should use defined safe semantics;
- final F1 = 0;
- no crash.

---

# 22. Testing strategy

Create:

```text
tests/test_ragtime26_nuggets.py
tests/test_ragtime26_argue.py
tests/test_ragtime26_scoring.py
tests/test_ragtime26_multilingual.py
tests/test_ragtime26_judge.py
```

---

# 23. Nugget tests

Test:

- OR nugget matching;
- AND nugget matching;
- duplicate question removal;
- vital/okay weights;
- empty nugget bank rejection;
- deterministic nugget IDs/order;
- partial credit behavior;
- contradictory answer behavior.

---

# 24. Sentence-support tests

Synthetic fixtures should include:

1. cited sentence fully supported;
2. cited sentence partially supported;
3. cited sentence unsupported;
4. relevant document but unsupported attached sentence;
5. irrelevant document;
6. uncited factual first instance;
7. uncited repeated fact;
8. heading/structural sentence;
9. mixed-support multiple citations;
10. unavailable judge call.

Verify exact counts and denominators.

---

# 25. Scoring tests

Create deterministic score fixtures and verify:

```text
nugget_coverage
nugget_coverage_weighted
sentence_support
citation_support
citation_relevance
f1
f1_weighted
```

Test 0 denominators explicitly.

Test macro/micro aggregation independently.

Use Auto-ARGUE public definitions as the baseline reference.

---

# 26. Multilingual tests

Do not depend on external services in unit tests.

Create small synthetic multilingual fixtures containing:

- English;
- Spanish;
- Russian;
- Chinese.

Include:

- matching entity/date;
- negation difference;
- numeric mismatch;
- attribution mismatch.

Mock translation/judge outputs and test pipeline semantics.

Add optional integration tests only when a model endpoint is available.

---

# 27. Endpoint contract

The judge must use the injected AutoJudge `llm_config`.

No hardcoded:

- API keys;
- URLs;
- Bedrock regions;
- model IDs.

It must pass the repository endpoint-contract test.

---

# 28. RAGTIME25 development evaluation

Use unrestricted RAGTIME25 as the primary development benchmark.

The existing `docs/evaluation-2025-plan.md` controls the split/evaluation discipline.

Recommended progression:

```text
RAGTIME25 development folds
    -> implement/tune prompts and scoring

RAGTIME25 held-out folds
    -> choose variants

DRAGUN25
    -> report-generation cross-domain sanity check

RAG25
    -> optional broader RAG sanity check

RAGTIME26
    -> frozen blind execution
```

---

# 29. Primary 2025 experiments

## Experiment A - support vs coverage

Compare:

```text
support-only
coverage-only
argue-f1
```

Question:

> Does balancing coverage and support improve system-ranking correlation over either component alone?

## Experiment B - weighted coverage

Compare:

```text
argue-f1
argue-f1-weighted
```

Question:

> Do VITAL/OKAY weights improve correlation with official/human RAGTIME25 measures?

## Experiment C - partial credit

Compare strict vs soft coverage.

Question:

> Does graded nugget completion better predict human judgments?

## Experiment D - document-grounded nuggets

Compare:

```text
query-only-nuggets
document-grounded-nuggets
```

Hold judge model constant.

Question:

> Does evidence-grounded decomposition improve coverage prediction enough to justify retrieval cost?

## Experiment E - request alignment

Compare core F1 with request-alignment diagnostic/fusion.

Do not introduce a fusion weight without cross-validation.

---

# 30. Evaluation metrics for the judge itself

Against RAGTIME25 truth, report:

- Kendall tau-b;
- Spearman rho;
- Pearson where magnitude is meaningful;
- tauAP-b/top-weighted ranking where available;
- bootstrap 95% confidence intervals by topic.

For individual human-derived measures, examine correlation with:

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

where these exist in the local truth.

Avoid picking whichever metric happens to yield the highest score after the fact.

Predefine primary candidates before the final held-out fold:

```text
RAGTIME_F1_WEIGHTED
RAGTIME_NUGGET_COVERAGE_WEIGHTED
RAGTIME_F1
```

---

# 31. Suggested primary endpoint

For local method selection, use:

> Kendall tau-b between the candidate judge measure and the designated RAGTIME25 official/human system ranking on held-out topics.

Secondary endpoints:

- Spearman rho;
- component-level correlation;
- support/coverage diagnostic agreement;
- cost;
- failure rate.

---

# 32. Cost and resource tracking

Log safe aggregate statistics:

```text
nugget-generation calls
translation calls
citation-support calls
citation-relevance calls
nugget-match calls
requires-citation calls
request-alignment calls
cache hits/misses
input tokens
output tokens
elapsed time
estimated cost
```

Do not log restricted report text or prompts/completions.

Reuse repository cost utilities.

---

# 33. Performance optimization

The largest likely cost is sentence × nugget matching.

Optimization order:

1. cache topic nugget banks;
2. cache document translation;
3. cache document relevance to nuggets;
4. shortlist candidate nuggets per sentence;
5. batch LLM judgments;
6. avoid evaluating repeated sentence/evidence pairs;
7. consider smaller routing model before larger semantic judge;
8. profile before introducing complex concurrency.

Do not optimize by reducing nugget recall without measuring the impact on held-out RAGTIME25 data.

---

# 34. Optional multi-model jury

Only after the single-endpoint baseline works.

Candidates:

```text
citation support -> highly factual model
nugget coverage -> strong semantic reasoning model
request alignment -> preference/instruction-following model
```

A heterogeneous jury may be used for ambiguous judgments.

Do not use three samples of the same model and call them independent judges.

Calibrate model weights using unrestricted development folds only.

---

# 35. Optional confidence/escalation

For every semantic judgment retain a confidence estimate if reliably available.

Potential cascade:

```text
cheap judge
   |
high confidence -> accept
   |
low confidence
   v
stronger judge / second family
```

Escalation is most valuable for:

- PARTIAL vs FULL support;
- AND nugget completion;
- contradictions;
- cross-lingual attribution;
- request-alignment borderline cases.

Evaluate selective escalation for accuracy/cost tradeoff on RAGTIME25.

---

# 36. Suggested implementation phases

## Phase 0 - inspect interfaces

Read:

- current RAGree implementations;
- `auto-judge-base` data model docs;
- Auto-ARGUE public scoring code/docs;
- current RAGTIME25 local data manifest.

Do not modify existing judge behavior.

## Phase 1 - scaffold

Create:

```text
judges/ragtime26_argue/
```

with importable judge and workflow.

Acceptance:

- workflow parses;
- judge imports;
- endpoint-contract test sees configured endpoint.

## Phase 2 - implement deterministic scoring core

Before any complex prompts, implement pure scoring functions for:

```text
nugget coverage
weighted nugget coverage
sentence support
citation support
citation relevance
F1
weighted F1
```

Acceptance:

- extensive synthetic tests;
- formulas documented.

## Phase 3 - sentence/citation judgment pipeline

Implement:

```text
citation relevance
sentence attestation
requires citation
first instance
```

Acceptance:

- fixture-driven tests;
- unavailable != false quality judgment.

## Phase 4 - nugget data model and static-bank judging

First support a prebuilt nugget bank before implementing generation.

This lets the judge be tested against existing RAGTIME25 nuggets if available.

Acceptance:

- load/consume external `NuggetBanks` correctly;
- sentence-to-nugget assignment works;
- coverage metrics emitted.

## Phase 5 - nugget generation

Add request + corpus evidence grounded generation.

Acceptance:

- nonempty validated banks;
- stable IDs/order;
- AND/OR and VITAL/OKAY supported;
- cache reuse verified.

## Phase 6 - multilingual evidence

Implement document retrieval/translation adapter.

Acceptance:

- all four 2026 languages supported at interface level;
- unit tests cover multilingual examples;
- translations cached.

## Phase 7 - request alignment

Add independent whole-report alignment score.

Acceptance:

- parser/tests;
- evaluation availability flag;
- no hardcoded fusion.

## Phase 8 - RAGTIME25 meta-evaluation

Run variants on unrestricted data.

Produce:

- result JSONL;
- correlations;
- bootstrap CIs;
- component ablation table;
- cost table.

## Phase 9 - optimization

Based only on unrestricted results:

- choose strict vs soft coverage;
- choose weighted vs unweighted;
- tune nugget counts;
- tune candidate routing depth;
- decide whether request alignment should influence headline score.

## Phase 10 - freeze

Freeze production workflow before RAGTIME26 execution.

Do not tune from restricted 2026 outputs.

---

# 37. Acceptance criteria

The first production-ready RAGTIME26 judge is complete when:

1. full repo tests pass;
2. endpoint-contract test passes;
3. kiddie/basic smoke test completes;
4. deterministic scoring functions match expected fixtures;
5. prebuilt nugget-bank judging works;
6. generated nugget banks are valid and nonempty;
7. RAGTIME25 evaluation completes with all expected cells;
8. all primary measures have explicit descriptions;
9. multilingual evidence adapter has test coverage;
10. transient model failures are distinguishable from bad quality;
11. no restricted report text is emitted in normal logs/artifacts;
12. final configuration is selected only from unrestricted development evidence.

---

# 38. Engineering targets

These are development targets, not claims about official scoring.

Aim for:

- `RAGTIME_F1_WEIGHTED` or another preregistered primary measure to outperform simple relevance on held-out RAGTIME25 ranking correlation;
- document-grounded nugget generation to improve coverage correlation relative to query-only generation;
- citation-support judgments to correlate strongly with official citation support/sentence support measures;
- missing-citation and contradiction diagnostics to improve error analysis without destabilizing ranking;
- routing optimization to reduce sentence×nugget LLM calls substantially with minimal loss of nugget recall;
- no large degradation on DRAGUN25 report-generation sanity checks.

---

# 39. Things explicitly not to do

Do not:

- use pairwise SPARROW as the default RAGTIME scoring mechanism;
- treat relevance and support as the same judgment;
- count unsupported but semantically correct-looking sentences toward nugget coverage without evidence checks;
- generate nugget banks from whichever candidate systems happen to be present;
- assign zero quality when the LLM endpoint simply failed;
- translate the whole corpus eagerly;
- collapse all components into an arbitrary weighted score before evaluation;
- tune against restricted RAGTIME26 outputs;
- inspect restricted RAGTIME26 report contents;
- deanonymize run IDs;
- hardcode provider/model credentials;
- emit report text/rationales into leaderboard or logs.

---

# 40. Public references

## RAGTIME 2026

```text
https://trec-ragtime.github.io/
```

Use the latest public task/guideline material when implementing, but distinguish confirmed 2026 rules from prior-year assumptions.

## Auto-ARGUE

```text
https://github.com/hltcoe/auto-argue
```

Relevant implementation/reference areas:

```text
auto_argue/score.py
auto_argue/judge.py
auto_argue/prompts/
README.md
```

Key public metric family:

```text
nugget_coverage
nugget_coverage_weighted
sentence_support
f1
f1_weighted
citation_support
citation_relevance
```

---

# 41. Relationship to the RAG26 judge

The two judges should share infrastructure where useful but must preserve different inductive biases.

| Design dimension | RAG26 | RAGTIME26 |
|---|---|---|
| Main target | pairwise answer preference | coverage + support |
| Main mechanism | Arena / SPARROW | ARGUE-style pointwise scoring |
| Topic decomposition | weighted rubric | QA nugget bank |
| Citation evaluation | separate support lane | integral to sentence reward |
| Multilingual evidence | optional/track dependent | core requirement |
| Pairwise ranking | central | optional diagnostic |
| Coverage | rubric fulfillment | nugget recall |
| Primary failure | preference/rubric mismatch | missing important information |

Do not force code sharing if it makes these semantics less clear.

---

# 42. Suggested first Claude Code task

When starting implementation:

1. scaffold `judges/ragtime26_argue/`;
2. implement pure scoring functions mirroring Auto-ARGUE metrics;
3. write exhaustive tests for those formulas;
4. implement judging against a prebuilt nugget bank before attempting nugget generation;
5. add sentence-attestation / requires-citation logic;
6. run against kiddie/synthetic fixtures;
7. only then add document-grounded nugget generation and multilingual adapters.

This order minimizes uncertainty: the scoring semantics become stable before expensive generation/retrieval components are introduced.

---

# 43. Final design principle

The RAGTIME26 judge should evaluate a report as an **evidence-grounded answer set**, not as a polished block of prose.

The working hypothesis to test on RAGTIME25 is:

```text
high-quality report
    =
important requested information covered
    AND
factual report sentences supported
    AND
citations relevant/correct
    AND
report aligned with the request
```

The primary judge should therefore remain decomposed and diagnostic until unrestricted 2025 evaluation demonstrates which component or combination best predicts the official/human rankings.
