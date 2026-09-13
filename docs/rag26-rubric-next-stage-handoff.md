# Handoff: RAG26 rubric construction — axis-generation milestone

**Current base commit:** `8cfee44` or later  
**Canonical plan:** `docs/rag26-rubric-construction-execution-plan.md`

## Read first

Read in full:

1. `AGENTS.md`
2. `CLAUDE.md`
3. `.claude/skills/autojudge-develop/SKILL.md`
4. `docs/rag26-rubric-construction-execution-plan.md`
5. `docs/trec-rag-2026-researchrubrics-generation-plan.md`

The data policy is mandatory. Rubric construction remains entirely response-independent: only public Requests, ResearchRubrics development material, and ClimbMix corpus evidence may enter this pipeline. Never use candidate RAG26 answers/citations/scores to create or select rubric content.

---

# Verified current state

Milestones completed and live-proven:

```text
Stage A  topic characterization + research-intent planning      DONE
Stage B  dense+sparse ClimbMix retrieval + RRF                 DONE
Stage C  evidence condensation                                 DONE
```

Latest implementation commit:

```text
8cfee44 Rubric construction: Stage B/C -- ClimbMix retrieval + evidence condensation
```

Current tests:

```text
236 passed, 2 xfailed
```

Live infrastructure:

```text
Dense ClimbMix :8088  healthy
Sparse BM25    :8085  healthy
```

Current rubric-generation spend is approximately `$0.0024`, far below the `$25` workstream cap.

## Model decision for this milestone

Keep the pipeline simple:

```text
Primary construction/internal critic:
  Qwen3 235B A22B 2507
  qwen.qwen3-235b-a22b-2507-v1:0
  ap-southeast-2
  Flex

Independent final audit later:
  GPT-5.6 Luna
```

Do **not** introduce Qwen3 32B now. The 235B Flex model is cheap enough that adding a smaller model would increase implementation/reproducibility complexity for negligible savings. A 32B ablation may be tested only after the 30-topic development benchmark.

---

# Goal of this milestone

Implement the complete **candidate-generation layer** only:

```text
D1 Explicit criteria
D2 Implicit criteria
D3 Synthesis criteria
D4 Negative/failure criteria
D5 Low-volume axes:
   - References & Citation Quality
   - Communication Quality
   - Instruction Following
```

Do **not** yet implement:

```text
atomicity/dedupe critic
examples pass
weight assignment
final criterion selection
Luna audit
119-topic production generation
```

Those come after this candidate layer is stable and tested.

---

# Shared candidate model

Reuse or extend the existing typed generation model. Each generated candidate must retain provenance separate from the organizer compatibility schema.

Recommended internal shape:

```python
class RubricCandidate(BaseModel):
    candidate_id: str
    criterion: str
    axis: Literal[
        "Explicit Criteria",
        "Implicit Criteria",
        "Synthesis of Information",
        "References & Citation Quality",
        "Communication Quality",
        "Instruction Following",
    ]
    polarity: Literal["positive", "negative"]
    rationale: str
    narrative_spans: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    confidence: float
```

Do **not** assign final numeric weight in these generation stages.

Do **not** add examples yet.

Candidate IDs must be deterministic code-generated IDs, not model-generated arbitrary IDs.

All parsers must reject malformed/unknown axes, empty criteria, wrong polarity, out-of-range confidence, unknown evidence IDs, and model-invented narrative/evidence references.

---

# D1 — Explicit criteria generator

## Inputs

Use primarily:

```text
Request narrative
TopicCharacterization.explicit_subquestions
named constraints
decision/synthesis goal
```

Evidence notes may be visible only for disambiguation; they must not introduce new explicit asks.

## Target

Generate approximately:

```text
Simple topic:    5–7 explicit candidates
Moderate topic:  6–9
High topic:      8–11
```

Do not force exact counts. Every explicit ask should be covered once.

## Rules

Each candidate must:

- correspond to an explicit user requirement;
- start with `The response ...` or an equally observable formulation;
- describe one primary outcome;
- be judgeable from the answer alone;
- avoid hidden reasoning requirements;
- avoid duplicating two asks that can be evaluated separately;
- preserve named entities, periods, comparisons, stakeholders, requested recommendation/decision, etc.;
- use only positive polarity unless the Request itself contains a material explicit prohibition that is best represented as a negative failure criterion later.

Require model output to cite the relevant explicit-subquestion/narrative reference from the supplied structured characterization, then verify it deterministically.

## Tests

Use synthetic fixtures for:

- one explicit ask;
- several independent asks;
- comparison + recommendation;
- scope/time-period constraint;
- no duplicated asks;
- malformed JSON;
- model invents non-existing narrative ref;
- cache replay.

---

# D2 — Implicit criteria generator

This is the highest-value and highest-risk generator.

## Inputs

```text
Request narrative
TopicCharacterization
all EvidenceNote objects
current explicit candidates
```

## Target

Generate approximately:

```text
Simple topic:     7–10 implicit candidates
Moderate topic:  10–14
High topic:      12–17
```

These are pre-normalization candidate counts, not final quotas.

## Model prompt question

Anchor the model on:

> What would a knowledgeable expert expect a high-quality response to address even though the user did not literally enumerate it?

## Candidate classes

Prioritise:

- prerequisite context;
- important caveats;
- confounders;
- alternative explanations;
- stakeholder effects;
- evidence limitations;
- uncertainty;
- causal-vs-correlational distinctions;
- time sensitivity;
- missing decision-relevant quantities;
- meaningful counterarguments;
- second-order effects.

## Grounding rule

Every factual/evidence-dependent implicit criterion must cite at least one valid `evidence_id`.

Prefer:

```text
one highly authoritative/relevant evidence note
or
multiple independent evidence notes
```

If evidence is disputed, produce a criterion requiring calibrated treatment of disagreement instead of asserting one side as settled.

A purely reasoning/usability implicit criterion may have no evidence ID only when its rationale clearly derives from the Request structure itself.

## Scope-control rule

Require a one-line rationale explaining why the criterion is a reasonable expectation rather than arbitrary scope expansion.

Reject criteria that are merely interesting facts from the retrieved corpus.

## Tests

Synthetic evidence packs must cover:

- well-supported caveat;
- conflicting evidence;
- irrelevant-but-interesting evidence;
- unsupported factual criterion attempt;
- duplicate explicit criterion attempt;
- causal overclaim prevention;
- cache replay.

---

# D3 — Synthesis criteria generator

## Inputs

```text
Request
characterization
EvidenceNotes grouped by subtopic
explicit + implicit candidates (compact summaries only)
```

## Target

```text
Simple:    2–3 candidates
Moderate:  3–5
High:      4–7
```

## Rules

Every synthesis criterion must require a relationship that isolated fact copying cannot satisfy.

Prefer verbs such as:

```text
compare
reconcile
connect
distinguish
weigh
explain
assess
contextualize
```

Reject pseudo-synthesis such as:

```text
"The response mentions A, B, and C"
```

unless the actual requirement is to explain how A/B/C interact.

Where evidence conflicts, synthesis criteria should reward explaining disagreement/evidence strength rather than forcing a single conclusion.

Each evidence-grounded synthesis criterion must retain the relevant evidence IDs/subtopics.

---

# D4 — Negative/failure-mode generator

Negative criteria are generated separately but must still use one of the six canonical axes.

## Inputs

```text
Request
characterization
EvidenceNotes
positive candidates
```

## Target

```text
Simple:    2–4
Moderate:  3–6
High:      4–7
```

Do not generate negatives simply to hit a quota.

## Candidate failure classes

Prioritise material failures:

- factual reversal;
- causal overclaim;
- incorrect number/denominator;
- fabricated certainty;
- stale/out-of-scope information;
- misleading omission;
- one-sided treatment where balance is required;
- source/evidence misattribution;
- contradiction with strong evidence;
- scope violation;
- material audience/format violation.

## Avoid mirror penalties

Do not create a negative criterion that is merely the logical negation of an existing positive criterion unless the commission of the error is materially worse than omission.

Example:

```text
positive: response discusses uncertainty
bad mirror: response fails to discuss uncertainty
```

Instead reserve negative weight for something materially harmful:

```text
response presents a disputed/uncertain conclusion as established fact
```

---

# D5 — low-volume axes generator

Combine these three axes into **one Qwen3 235B call per topic** to reduce orchestration overhead while keeping typed output separated by axis.

## References & Citation Quality

Target:

```text
1–2 candidates
```

Keep this light because citation support is separately scored by `RAG26ArenaJudge`.

Good rubric-level criteria:

- important empirical/numerical claims are backed by specific evidence;
- conflicting claims are attributed rather than blended;
- sources are not misrepresented.

Do not duplicate sentence-level citation-support metrics.

## Communication Quality

Target:

```text
1–3 candidates
```

Make topic-specific usability criteria rather than universal essay-style rules.

Examples:

- trade-offs are organised so comparison is easy;
- specialist terminology needed for the decision is explained;
- requested bottom line is clear and not buried.

## Instruction Following

Target:

```text
1–2 candidates
```

Only use for actual scope/audience/format/decision constraints.

Do not move factual explicit requirements into this axis.

---

# Prompt architecture

Create separate prompt versions, for example:

```text
EXPLICIT_PROMPT_VERSION = "v1"
IMPLICIT_PROMPT_VERSION = "v1"
SYNTHESIS_PROMPT_VERSION = "v1"
NEGATIVE_PROMPT_VERSION = "v1"
LOW_VOLUME_PROMPT_VERSION = "v1"
```

Use the existing response-independent `RubricLlmCache` and cost tracker.

Each cache key must include:

```text
stage
prompt version
model ID
tier
region
messages
```

Do not include endpoint credentials.

Do not cache partially invalid outputs as successful results.

Malformed structured output should use bounded retry, with the same semantics already used by characterization/evidence condensation.

---

# Orchestration API

Add a single orchestration function that accepts already-completed Stage A/C outputs, e.g.:

```python
generate_axis_candidates(
    topic,
    characterization,
    evidence_notes,
    *,
    model_id,
    region,
    tier,
    fallback_tier,
    cache,
    cost_tracker,
) -> list[RubricCandidate]
```

Internally call the five generators in deterministic order:

```text
explicit
implicit
synthesis
negative
low-volume
```

The orchestration result must be deterministically ordered and candidate IDs deterministically assigned in code.

Do not perform deduplication/weighting/final selection in this function yet.

---

# Candidate-count diagnostics

After the full candidate layer, print safe diagnostics only:

```text
qid
stage
candidate count per axis
positive/negative counts
number evidence-grounded
cache hit/miss
input/output tokens
estimated cost
retry count
elapsed
```

Do not print full prompt/evidence bodies into routine logs.

For response-independent rubric construction it is safe and desirable to print actual token totals and actual estimated dollar spend. Do not hide those numbers; they are important operational diagnostics and are not restricted report-derived data.

Expected pre-normalization candidate total for a moderate topic is roughly:

```text
explicit       6–9
implicit      10–14
synthesis      3–5
negative       3–6
references     1–2
communication  1–3
instruction    1–2
------------------
rough total   25–41
```

This is intentionally broader than the final 24–28 target because the next milestone will remove redundancy and low-value candidates.

---

# Tests before any new paid multi-topic run

Create offline tests with mocked Bedrock for every generator.

At minimum test:

```text
valid parse
malformed JSON
empty list
unknown axis
wrong polarity
unknown evidence_id
invalid confidence
candidate count outside bounded range
cache miss -> cache hit
retry after invalid response
deterministic candidate IDs/order
no candidate-response input path exists
```

Full suite must remain green.

Do not make normal pytest perform real Bedrock calls.

---

# Live proof for this milestone

After offline tests pass, run exactly **one unrestricted/synthetic/kiddie topic** through all five axis generators using real Qwen3 235B Flex.

Report:

```text
calls by stage
actual input tokens
actual output tokens
actual estimated USD
candidate counts by axis
cache miss/hit counts
validation/retry counts
```

Then rerun with no valid AWS credentials and confirm every call replays from the response-independent cache at zero cost.

Do not proceed to a 30-topic ResearchRubrics benchmark from this milestone yet.

---

# Exit gate

This milestone is complete only when:

```text
all five axis generators implemented
shared typed candidate validation implemented
one orchestrator implemented
offline tests green
full pytest green
one-topic real Qwen235 Flex run succeeds
cache-only replay succeeds
safe token/cost telemetry printed
no candidate-response dependencies introduced
```

Commit as one or two logical commits, not five tiny prompt-only commits.

After this exit gate, stop and report back before implementing atomicity/deduplication, weighting, final selection, or Luna review.
