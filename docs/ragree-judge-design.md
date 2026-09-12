# Proposal: **RAGree**

**RAGree** is a **Verifiable, Preference-aligned, agent-aware RAG judge**.

Its central design principle is that **human preference, factual correctness, grounding, retrieval quality, and agent quality must be measured separately before they are combined**. It should not be implemented as one large model receiving one long rubric and returning a single 1–10 score.

The recommended architecture has three independent judging lanes:

1. **Evidence Judge** — claim accuracy, answer precision and recall, grounding, retrieval quality, and citations.
2. **Preference Judge** — probability that an evidence-informed human would prefer the answer.
3. **Trajectory Judge** — retrieval and tool-use quality for reasoning and agentic systems.

A calibrated decision layer then combines the three, applying **hard correctness constraints before preference optimisation**.

This separation is important. In RAGChecker’s 280-pair human meta-evaluation, its claim-level metrics achieved overall Pearson/Spearman correlations of approximately **0.619/0.609**, ahead of the tested RAGAS metrics at **0.483/0.572**, but still below human–human correlations of **0.701/0.689**. At the same time, JudgeBench shows that models that look competent as preference judges can perform close to chance on hard objective-correctness comparisons.

The more recent Auto-Judge work provides further evidence for a composite design: combining a relevance judge with sentence-level evidence attestation improved the predicted F1 compared with using relevance alone, and its cross-task framework explicitly separates leaderboard correlation, label agreement, and pairwise preference agreement.

---

## 1. What RAGree should report

The primary output should be a **score vector**, not only an overall score.

| Category | Metric | Interpretation |
|---|---|---|
| Human preference | `expert_preference_probability` | Probability that an evidence-informed human prefers this answer to a calibrated baseline or alternative |
| Answer accuracy | `task_correctness` | Direct correctness against a gold answer where one exists |
| Factual precision | `claim_precision` | Proportion of weighted factual claims that are supported by authoritative evidence |
| Answer recall | `claim_recall` | Proportion of required answer nuggets actually covered |
| Answer quality | `claim_f1` | Harmonic mean of claim precision and claim recall |
| Grounding | `context_faithfulness` | Proportion of claims supported specifically by the context supplied to the generator |
| Retrieval | `retrieval_precision` | Proportion of retrieved evidence that contributes to answering the question |
| Retrieval | `retrieval_recall` | Proportion of required information for which supporting evidence was retrieved |
| Citations | `citation_precision` | Proportion of citations that support the exact claim to which they are attached |
| Citations | `citation_recall` | Proportion of important factual claims that have a valid citation |
| Safety/reliability | `critical_error_rate` | Rate of major contradictions, fabricated entities, unsafe omissions, or materially stale claims |
| Agentic quality | `trajectory_score` | Quality of retrieval calls, tool choices, arguments, sequencing, and evidence use |
| Calibration | `judge_confidence` | Calibrated probability that the verdict itself is correct |

### A crucial point about recall

**True recall requires a denominator.** It is impossible to measure absolute answer recall without knowing the set of information that a complete answer should contain.

Therefore RAGree should distinguish:

- `gold_claim_recall`: measured against human-vetted required nuggets;
- `reference_claim_recall`: measured against one or more reference answers;
- `estimated_coverage`: measured against an automatically generated nugget bank.

The third is useful for large-scale monitoring, but it should never be presented as absolute recall without qualification.

---

## 2. Overall architecture

```text
Question
  + candidate answer
  + retrieved contexts
  + citations
  + optional reference/gold nuggets
  + optional agent trace
  + corpus access
             |
             v
   Input canonicalisation
   provenance and schema checks
             |
             v
   Claim and nugget decomposition
      /             |              \
     v              v               v
Evidence Judge  Preference Judge  Trajectory Judge
     |              |               |
claim support   pointwise-first   tool selection
claim coverage  pairwise-final    arguments/order
retrieval P/R   diverse jury      evidence discovery
citation P/R    human calibration efficiency/recovery
     \              |               /
      \             |              /
       v            v             v
       Calibration and uncertainty layer
                       |
                       v
       score vector + decision + audit trail
```

---

# 3. Evidence Judge

## 3.1 Atomic claim decomposition

The answer is decomposed into minimal, independently verifiable claims. For example:

> “The legislation begins in July 2027 and gives all Victorian employees two days of remote work.”

becomes:

1. The legislation begins in July 2027.
2. The legislation applies to all Victorian employees.
3. It creates an entitlement to two remote-working days.

Each claim receives:

- a salience weight;
- a claim type;
- a temporal scope;
- an evidence requirement;
- a severity level if incorrect.

Useful claim types include:

- factual;
- numerical;
- temporal;
- causal;
- comparative;
- procedural;
- recommendation;
- interpretation;
- explicitly uncertain or conditional.

FActScore demonstrated the usefulness of atomic facts for factual precision, while RAGChecker extended claim-level analysis to answer precision, recall, retrieval diagnostics, context utilisation, hallucination, and faithfulness.

## 3.2 Four-way claim verification

For each claim, the judge retrieves the strongest available evidence and produces:

- `SUPPORTED`
- `PARTIALLY_SUPPORTED`
- `CONTRADICTED`
- `NOT_ENOUGH_EVIDENCE`

Each verdict must include:

- exact supporting or contradicting spans;
- document identifiers;
- source date;
- source-quality classification;
- confidence;
- explanation limited to the evidence relationship.

A claim should be verified twice:

### Retrieved-context verification

Does the context that was supplied to the generator support the claim?

This produces **context faithfulness**.

### Authoritative-corpus verification

Does the best available source in the evaluation corpus support the claim?

This produces **factual correctness**.

These are not interchangeable. An answer may faithfully repeat an incorrect or outdated retrieved document. Conversely, it may state something true from parametric knowledge that was not supported by the supplied context.

Research on RAG attribution also distinguishes citation correctness from causal faithfulness: citations can superficially support a statement without having actually influenced its generation. One study found substantial post-rationalisation in the evaluated setup and recommends perturbation-based tests of whether changing the cited evidence changes the answer.

## 3.3 Claim precision and recall

Let \(C\) be the claims made by the answer and \(N\) the required answer nuggets.

Assign each answer claim a salience weight \(w_i\), and an evidence score:

- supported: 1
- partially supported: 0.5
- unsupported: 0
- contradicted: 0, with an additional error penalty

Then:

\[
P_{\text{claim}}
=
\frac{\sum_{i \in C} w_i \cdot \text{support}(i)}
     {\sum_{i \in C} w_i}
\]

For answer recall:

\[
R_{\text{claim}}
=
\frac{\sum_{j \in N} w_j \cdot \mathbf{1}[\text{nugget }j\text{ is covered}]}
     {\sum_{j \in N} w_j}
\]

And:

\[
F_{1,\text{claim}}
=
\frac{2P_{\text{claim}}R_{\text{claim}}}
     {P_{\text{claim}}+R_{\text{claim}}}
\]

For domains where omission is more dangerous than an extra minor claim, such as medicine, legal advice, or compliance, use \(F_2\) rather than \(F_1\) to place more weight on recall.

### Critical-error weighting

Not all claims should count equally. An incorrect effective date, dosage, eligibility rule, or numerical result can be much more consequential than a minor descriptive error.

Every claim should therefore have:

```json
{
  "importance": 0.0,
  "error_severity": "minor | major | critical",
  "claim_type": "temporal | numeric | factual | causal | other"
}
```

A critical contradiction should trigger a hard failure regardless of average claim precision.

---

## 3.4 Retrieval precision and recall

Retrieval evaluation should operate at **evidence-span or nugget level**, not just at whole-document level.

### Retrieval recall

For each required nugget, determine whether the top-\(k\) retrieval contains enough evidence to support it.

\[
R_{\text{retrieval}}
=
\frac{\text{required nuggets with supporting retrieved evidence}}
     {\text{all required nuggets}}
\]

### Retrieval precision

A retrieved chunk can contain both useful and irrelevant material. RAGree should therefore calculate a rank-weighted evidence density:

\[
P_{\text{retrieval}}
=
\frac{\sum_k d(k)\cdot \text{relevant evidence mass in chunk }k}
     {\sum_k d(k)\cdot \text{total evidence mass in chunk }k}
\]

where \(d(k)\) discounts lower-ranked chunks.

Also report:

- context redundancy;
- contradictory-context rate;
- stale-source rate;
- source diversity;
- context utilisation;
- relevant and irrelevant noise sensitivity.

This localises failures:

| Retrieval recall | Claim recall | Likely problem |
|---:|---:|---|
| Low | Low | Retriever did not find the necessary information |
| High | Low | Generator failed to use available evidence |
| High | High, but precision low | Generator added unsupported material |
| Low | High | Answer likely relied on parametric knowledge |
| High | High | Healthy retrieval-to-generation path |

---

## 3.5 Citation evaluation

For every answer–citation link, the judge should verify:

1. whether the cited source entails the associated claim;
2. whether the citation points to the correct passage;
3. whether every important externally verifiable claim has a citation;
4. whether the source is authoritative and temporally appropriate;
5. whether the citation is likely post-hoc rather than causally connected to generation.

ALCE formalised citation precision and recall and reported substantial human agreement for citation recall and moderate agreement for citation precision; its automated labels reached 85.1% and 77.6% accuracy respectively in the reported human comparison.

For a fully observable RAG pipeline, run a **causal attribution audit** on a sample:

- remove the claimed supporting passage;
- replace its central proposition with a controlled counterfactual;
- rerun generation;
- measure whether the answer or its confidence changes appropriately.

This is too expensive for every production response, but it is valuable for model certification and periodic audits.

---

# 4. Preference Judge

## 4.1 Optimise evidence-informed human preference

The primary target should be:

> “Which answer would an informed human prefer after considering correctness, completeness, usefulness, clarity, uncertainty, and evidence?”

This is better than asking which answer merely “sounds better.”

It is useful to collect two distinct human labels:

- **UX preference** — which response is easier and more pleasant to use;
- **expert preference** — which response is better after checking evidence and completeness.

The production quality gate should use expert preference. UX preference can optimise presentation after the answer has passed factual thresholds.

## 4.2 Pointwise first, pairwise second

Naive pairwise comparison often aligns better with human ranking than direct 1–5 scoring, and PairS shows that uncertainty-guided pairwise ranking can outperform direct scoring and produce more robust rankings. However, pairwise judges remain vulnerable to position and contextual biases.

RAGree should use a **PRePair-style procedure**:

1. Judge Answer A independently.
2. Judge Answer B independently.
3. Freeze both pointwise evidence tables.
4. Compare A and B using only those analyses and the original answers.
5. Repeat with A/B order reversed.
6. Return a win only when both orderings agree.
7. Otherwise return a tie or send the case to an arbiter.

PRePair was proposed specifically because ordinary pairwise comparison can amplify superficial preferences; it combines pointwise reasoning with a final pairwise decision and improved robustness on adversarial comparisons while retaining strong standard-benchmark performance.

## 4.3 Three-model heterogeneous jury

Use three independent judge roles:

1. **Open evaluator model** trained on direct and pairwise assessment.
2. **Open reasoning model** instructed to independently solve or verify the task.
3. **Evidence specialist** using claim-level NLI and retrieval.

Do not expose:

- generator model name;
- provider name;
- system identity;
- original answer ordering;
- irrelevant metadata such as latency or model size.

A heterogeneous panel is preferable to three samples from the same model. In the Panel-of-LLMs work, a diverse jury correlated better with humans than the single large judge used in the experiments, was reported as more than seven times cheaper, and reduced intra-model bias.

The system should explicitly test:

- order-swap consistency;
- verbosity attacks;
- authoritative-tone attacks;
- self-family preference;
- duplicated-content attacks;
- citation-density preference;
- formatting preference;
- preference for answers that merely repeat the reference wording.

Position, verbosity, self-enhancement, and limited-reasoning biases have all been documented in LLM judges.

## 4.4 Preference probability rather than a raw score

The model should return:

```json
{
  "p_a_better": 0.71,
  "p_tie": 0.18,
  "p_b_better": 0.11
}
```

These probabilities should be calibrated against human vote distributions rather than hard majority labels.

For system-level ranking, fit a Bradley–Terry or Plackett–Luce model over pairwise results. This produces:

- a latent system-quality estimate;
- confidence intervals;
- meaningful comparisons between systems;
- fewer arbitrary differences between 4/5 and 5/5 scores.

---

# 5. Trajectory Judge for reasoning and agentic RAG

The final answer and the trajectory must be scored separately.

A model can reach the correct answer after:

- unnecessary searches;
- incorrectly parameterised tools;
- reliance on an unrelated source;
- failed actions it never noticed;
- accidental leakage;
- excessive cost;
- a tool result that it misrepresented.

Conversely, a reasonable trajectory can fail because the underlying tool or corpus was incomplete.

## Trajectory metrics

RAGree should evaluate:

| Metric | Meaning |
|---|---|
| `tool_selection_precision` | Fraction of invoked tools that were appropriate |
| `tool_selection_recall` | Fraction of necessary tools that were invoked |
| `argument_correctness` | Validity and semantic correctness of tool parameters |
| `dependency_satisfaction` | Whether actions occurred in a valid causal order |
| `evidence_discovery_recall` | Required nuggets discovered by the agent |
| `evidence_use_rate` | Retrieved evidence actually used in the final answer |
| `source_diversity` | Whether independent sources were consulted where necessary |
| `recovery_score` | Whether the agent recognised and recovered from failed retrieval |
| `trajectory_efficiency` | Useful evidence acquired per call, token, cost, and unit time |
| `provenance_integrity` | Every final claim can be traced to an observable source or computation |

TRAJECT-Bench similarly argues that final-answer accuracy alone is insufficient and reports tool selection, argument correctness, inclusion, exact trajectory matching, and dependency/order satisfaction.

## Agent-as-judge escalation

Most evaluations should be handled by the static cascade. A true judging agent should be invoked only for ambiguous or high-risk cases. It may:

- search the corpus;
- open full documents;
- compare document versions;
- execute deterministic calculations;
- inspect structured traces;
- verify tool outputs;
- locate missing evidence;
- build a claim–source graph.

In a software-agent evaluation setting, Agent-as-a-Judge’s active inspection substantially improved alignment with human consensus over a black-box LLM judge; the result should not be assumed to transfer unchanged to RAG, but it is strong evidence that giving the judge targeted read, locate, graph, and retrieval abilities can outperform asking it to decide from one static prompt.

## Hidden reasoning is not required

RAGree must **not depend on access to private chain-of-thought**.

For open-weight reasoning models, the judge may use:

- visible final answer;
- citations;
- tool calls;
- retrieved observations;
- code execution results;
- public reasoning summaries;
- token probabilities where available.

It should not treat a plausible reasoning narrative as proof that the process was correct. Process scores should be based on observable actions and evidence dependencies.

---

# 6. How the final RAGree score should work

The full metric vector remains the authoritative result. A scalar score is useful for ranking and automation, but it should be **truth-constrained**.

## Step 1: factual eligibility gate

```text
eligible =
    lower_95_bound(claim_precision) >= precision_threshold
    AND lower_95_bound(claim_recall) >= recall_threshold
    AND critical_contradictions == 0
    AND citation_fabrication == false
    AND policy_violations == 0
```

Thresholds should differ by domain. A casual recommendation system can tolerate more uncertainty than a legal, medical, financial, or compliance RAG system.

## Step 2: human-preference score

Among eligible answers:

\[
\text{RAGree Score}
=
100 \times
P(\text{evidence-informed human prefers candidate over domain baseline})
\]

The baseline can be:

- the current production system;
- a human-written answer;
- a domain-specific acceptable-quality anchor;
- an ensemble of degraded, acceptable, and excellent anchor answers.

## Step 3: learned monotonic fusion

For offline system ranking, fit a monotonic calibration model using human comparisons:

\[
S =
\sigma\left(
\beta_0
+ \beta_p \operatorname{logit}(P_{\text{pref}})
+ \beta_f \operatorname{logit}(F_{\text{claim}})
+ \beta_c \operatorname{logit}(F_{\text{citation}})
+ \beta_t T_{\text{trajectory}}
- \beta_e E_{\text{critical}}
\right)
\]

Constrain all quality coefficients to be non-negative and the error coefficient to be non-positive. This prevents the training process from learning, for example, that lower factual precision is desirable merely because verbose incorrect answers occasionally received favourable user votes.

Do **not** start with hand-selected weights such as “40% relevance, 30% accuracy, 30% style.” Learn the fusion from human comparisons and validate it on completely hidden systems.

---

# 7. Concrete implementation

## 7.1 Evaluation record

```json
{
  "eval_id": "eval-000123",
  "domain": "legal",
  "query": "What rights does the proposed law create?",
  "query_time": "2026-08-29T00:00:00Z",

  "answer": "...",

  "retrieved_contexts": [
    {
      "context_id": "ctx-1",
      "document_id": "doc-17",
      "text": "...",
      "rank": 1,
      "retrieval_score": 0.87,
      "published_at": "2026-07-01",
      "source_type": "government"
    }
  ],

  "citations": [
    {
      "answer_start": 105,
      "answer_end": 188,
      "context_ids": ["ctx-1"]
    }
  ],

  "reference_answers": [],
  "gold_nuggets": [],
  "corpus_handle": "corpus-v42",

  "trace": [
    {
      "step": 1,
      "action": "search",
      "tool": "hybrid_search",
      "arguments": {"query": "...", "top_k": 10},
      "observation_ids": ["ctx-1", "ctx-2"],
      "latency_ms": 83,
      "cost": 0.0004
    }
  ]
}
```

## 7.2 Output contract

```json
{
  "ragree_score": 86.4,
  "decision": "auto_pass",

  "preference": {
    "expert_preference_probability": 0.83,
    "ux_preference_probability": 0.76,
    "pairwise_order_consistent": true
  },

  "answer": {
    "task_correctness": 0.90,
    "claim_precision": 0.94,
    "claim_recall": 0.82,
    "claim_f1": 0.88,
    "context_faithfulness": 0.91,
    "critical_error_rate": 0.0
  },

  "retrieval": {
    "precision": 0.61,
    "recall": 0.89,
    "context_utilisation": 0.72,
    "contradictory_context_rate": 0.0
  },

  "citations": {
    "precision": 0.96,
    "recall": 0.85,
    "causal_faithfulness_sampled": null
  },

  "trajectory": {
    "tool_selection_precision": 0.90,
    "tool_selection_recall": 1.0,
    "argument_correctness": 0.95,
    "dependency_satisfaction": 1.0,
    "efficiency": 0.71
  },

  "uncertainty": {
    "confidence": 0.91,
    "human_review_probability": 0.04,
    "jury_disagreement": 0.06
  },

  "findings": [
    {
      "claim_id": "claim-7",
      "verdict": "NOT_ENOUGH_EVIDENCE",
      "severity": "minor",
      "claim": "...",
      "evidence_ids": ["ctx-3"],
      "explanation": "The cited passage discusses eligibility but not the claimed effective date."
    }
  ]
}
```

## 7.3 Model cascade

### Tier 0: deterministic checks

No LLM required:

- citation IDs exist;
- answer spans are valid;
- sources are accessible;
- timestamps are parseable;
- tool calls satisfy schemas;
- URLs or identifiers were not fabricated;
- expected structured output is complete;
- numeric calculations can be recomputed.

### Tier 1: inexpensive local models

- Atomic claim extraction using a small or medium open instruction model.
- Embedding and lexical evidence matching.
- MiniCheck-like grounding verifier.
- Small relevance classifier.
- Citation-span entailment model.

MiniCheck’s 770M fact-checking model reached GPT-4-level accuracy on the paper’s unified grounding benchmark, making it a strong starting point for the inexpensive verifier tier.

### Tier 2: open preference judge

Fine-tune a current 7B–14B open-weight backbone for:

- direct pointwise scoring;
- PRePair comparison;
- evidence-informed preference;
- structured rubric output;
- uncertainty and abstention.

Prometheus 2 is an appropriate research baseline because it unifies direct assessment and pairwise ranking, and its 7B and 8x7B variants reported the strongest results among the open evaluator models tested in that work. It should be treated as a baseline rather than presumed to remain the best backbone in 2026.

### Tier 3: open reasoning arbiter

Invoke only when:

- the verifier and preference judge disagree;
- the claim is mathematically or causally complex;
- evidence conflicts;
- the source is long or multimodal;
- the claim is marked critical;
- the generator and judge share a model family;
- confidence falls below the selective-prediction threshold.

JudgeLRM provides evidence that open reasoning-oriented models can be trained specifically as judges, while ConsJudge suggests using consistency across rubric combinations to construct better preference data for judge training.

### Tier 4: judging agent

The arbiter gains corpus search, document reading, source comparison, and deterministic execution tools. Only the most uncertain cases reach this tier.

This cascade should make the majority of evaluations inexpensive while concentrating reasoning compute on the cases where it improves reliability.

---

# 8. Training data design

Public datasets are useful for bootstrapping:

- **RAGTruth** contains nearly 18,000 naturally generated RAG responses with case- and word-level hallucination annotation.
- **RAGBench** contains explainable labels across relevance, adherence, and completeness settings.
- **FActScore**, **RAGChecker**, and **ALCE** provide useful claim, coverage, and citation formulations.
- **JudgeBench** and **LLMBar-style adversarial data** are important for objective-correctness and superficial-bias testing.
- **TRAJECT-Bench** can seed trajectory and tool-use evaluation.

Public data alone will not produce the most human-correlated judge for a particular RAG product. A domain-specific human calibration set is necessary.

## Initial production-grade data target

I would begin with:

- **1,500 queries** across five materially different domains;
- **6–8 RAG configurations** per query;
- approximately **9,000–12,000 candidate responses**;
- **1,000 human-vetted nugget banks**;
- **20,000 claim–evidence labels**;
- **3,000 pairwise response comparisons**, with three human raters per pair;
- a separate set of at least **300 human-labelled responses per major deployment domain** for final calibration;
- a **20% query holdout**;
- at least one completely held-out generator family;
- a rolling temporal holdout made from newly collected queries and sources.

The pairwise labels should use a five-way scale:

- A substantially better;
- A slightly better;
- tie;
- B slightly better;
- B substantially better.

Store the full rater distribution rather than collapsing immediately to a majority vote.

## Controlled failure generation

For each correct answer, construct validated siblings containing one controlled defect:

- one unsupported addition;
- one omitted required nugget;
- one incorrect number;
- one incorrect date;
- one reversed causal claim;
- one irrelevant citation;
- one citation pointing to the wrong passage;
- one post-hoc citation;
- one stale source;
- one contradiction with retrieved evidence;
- one distractor-following answer;
- one unnecessarily verbose but semantically identical answer;
- one confident but wrong answer;
- one correct abstention;
- one unnecessary abstention;
- one prompt-injection-following answer;
- one incorrect agent-tool call;
- one correct final answer reached through an invalid trajectory.

These pairs make the judge’s error modes identifiable instead of relying exclusively on naturally occurring mistakes.

---

# 9. Training procedure

## Stage A: teacher labelling

Use a heterogeneous teacher jury to generate structured candidate labels. Retain a synthetic label only when:

- at least two independent model families agree;
- the evidence spans are valid;
- a deterministic verifier finds no structural contradiction;
- the example is not in the hidden human test set.

Synthetic labels should never replace human validation; they increase coverage.

## Stage B: supervised fine-tuning

Train the model to output:

1. atomic claims;
2. required nuggets;
3. evidence links;
4. four-way support verdicts;
5. severity;
6. pointwise rubric analysis;
7. final pairwise preference;
8. calibrated uncertainty.

Use concise structured rationales rather than exposing or requiring unrestricted chain-of-thought.

## Stage C: preference optimisation

Train on human pairwise comparisons using DPO, ORPO, or a comparable preference objective.

Create consistency pairs by:

- reversing answer order;
- changing answer labels;
- paraphrasing the rubric;
- permuting rubric dimensions;
- removing model names;
- varying irrelevant formatting;
- shortening verbose answers without changing content.

ConsJudge’s consistency-based construction is directly relevant here because RAG judge outputs can change substantially with prompt and rubric formulation.

## Stage D: objective-verification training

For automatically checkable tasks, use outcome-based training:

- numerical recomputation;
- exact entity verification;
- date consistency;
- citation-span entailment;
- tool-schema validity;
- required action ordering;
- controlled counterfactual evidence.

Do not use subjective preference rewards to train the claim verifier.

## Stage E: calibration

Fit:

- Platt scaling or isotonic calibration for binary labels;
- temperature scaling for multi-class claim verdicts;
- a calibrated pairwise model for human preference;
- bootstrap or prediction-powered confidence intervals at the system level.

ARES is especially relevant here: it combines lightweight learned judges with a comparatively small human validation set and prediction-powered inference, demonstrating how automatic judging can be statistically corrected rather than treated as perfect ground truth.

---

# 10. Meta-evaluating the judge

A judge should not be accepted because its explanations look plausible. It must be evaluated as a predictive system.

## Required meta-metrics

| Goal | Metrics |
|---|---|
| Human preference correlation | Pairwise agreement, cross-entropy, Brier score |
| System ranking | Kendall’s \(\tau\), Spearman’s \(\rho\), top-weighted rank correlation |
| Objective claim judging | Macro-F1, balanced accuracy, class-specific precision and recall |
| Hallucination detection | Unsupported-claim recall and contradiction recall |
| Citation judging | Citation precision, recall, and F1 |
| Calibration | Expected calibration error and reliability curves |
| Selective automation | Error rate as a function of automated coverage |
| Consistency | Order-swap agreement, rubric-paraphrase agreement, transitivity |
| Generalisation | Cross-domain, cross-generator-family, and temporal holdouts |
| Agentic evaluation | Tool precision/recall, argument accuracy, dependency satisfaction |

Auto-Judge’s framework is particularly appropriate for this layer because it distinguishes response-level label agreement, pairwise preference agreement, and system-level leaderboard correlation, while using fresh runs and delayed truth labels to reduce memorisation and leakage.

## Proposed go/no-go targets

These are engineering targets, not claims established by the cited literature:

- pairwise preference agreement within **five percentage points of human–human agreement**;
- system-level Kendall’s \(\tau \geq 0.80\) on a fresh hidden set;
- macro-F1 of at least **0.85** on claim-verification labels;
- at least **0.95 recall for critical contradictions**;
- expected calibration error below **0.05**;
- order-swap decision changes below **2%**, excluding legitimate ties;
- no material decrease on a completely unseen generator family;
- an automated-review threshold selected to meet a predetermined maximum error rate, rather than an arbitrary confidence score.

These targets should be tightened or loosened according to the consequences of a wrong automatic decision.

---

# 11. Human-review routing

RAGree should support three outcomes:

### `AUTO_PASS`

- high lower-bound claim precision;
- sufficient answer recall;
- no critical contradictions;
- adequate citation coverage;
- stable jury verdict;
- calibrated preference above the deployment threshold.

### `HUMAN_REVIEW`

- conflicting evidence;
- low-confidence recall estimate;
- judge disagreement;
- critical domain;
- suspected citation post-rationalisation;
- new source or model family;
- unfamiliar tool trajectory;
- answer near the decision boundary.

### `FAIL`

- fabricated citation;
- major contradiction;
- unsupported critical claim;
- prompt-injection compliance;
- missing mandatory information;
- irrecoverably invalid agent trace.

The threshold should be selected using a **risk–coverage curve**: choose the maximum percentage of cases that can be automated while holding the observed error rate below the organisation’s tolerance.

---

# 12. Implementation roadmap

## Weeks 1–2: baseline and schema

Deliver:

- canonical evaluation schema;
- claim and nugget representation;
- RAGChecker-style claim metrics;
- ALCE-style citation metrics;
- deterministic trace checks;
- baseline Prometheus 2 and MiniCheck runs;
- reproducible evaluation harness.

## Weeks 3–6: human benchmark

Deliver:

- query sampling;
- system-response pool;
- human annotation interface;
- required-nugget guidelines;
- claim-evidence annotations;
- evidence-informed pairwise preference labels;
- adversarial sibling generation;
- hidden model-family and temporal test splits.

## Weeks 7–10: train open-weight judges

Deliver:

- claim extractor;
- claim verifier;
- pointwise preference judge;
- PRePair comparator;
- heterogeneous jury;
- consistency training;
- confidence calibration.

## Weeks 11–13: agentic extension

Deliver:

- trace canonicaliser;
- claim–evidence–action graph;
- tool and argument checks;
- dependency and ordering metrics;
- retrieval-efficiency metrics;
- agent-as-judge escalation workflow.

## Weeks 14–16: meta-evaluation and shadow deployment

Deliver:

- judge-versus-human report;
- order, verbosity, self-family, and injection red-team results;
- system-ranking correlation;
- calibration and selective-risk curves;
- initial auto-pass/review thresholds;
- monitoring dashboard;
- frozen hidden test set;
- model card and evaluation card.

---

# 13. The most relevant lessons from the literature

| Research direction | What RAGree should borrow |
|---|---|
| FActScore and RAGChecker | Atomic claims, claim precision, answer coverage, and component-level diagnosis rather than one holistic grade |
| RAGAS | Reference-free relevance and faithfulness as useful baselines, but not sufficient as the sole judge |
| ARES | Learned lightweight judges, domain adaptation, human calibration, and statistically corrected system estimates |
| ALCE | Separate citation precision from citation recall and validate both against humans |
| PairS and PRePair | Pairwise human alignment, uncertainty-guided ranking, and pointwise-first debiasing |
| Panel of LLMs | A heterogeneous jury instead of repeated samples from one model family |
| Prometheus 2 and JudgeLRM | Open-weight judges can support direct assessment, pairwise ranking, and explicit judge reasoning |
| ConsJudge | Train for consistency across dimensions, prompts, and evaluation formats |
| JudgeBench | Human-like preference ability does not guarantee objective verification ability |
| Agent-as-a-Judge and TRAJECT-Bench | Let difficult judges inspect artifacts and evaluate observable trajectories, not only final prose |
| Auto-Judge | Continuous cross-task meta-evaluation on fresh system outputs, with separate label, preference, and leaderboard measures |

# Final recommendation

Build **RAGree as a modular evaluation system, not a monolithic judge prompt**.

The decisive architecture is:

- **claim-level objective verification** for accuracy, precision, and recall;
- **pointwise-first, pairwise-final preference judging** for human alignment;
- **a heterogeneous open-weight jury** to reduce family-specific bias;
- **trajectory evaluation** for reasoning and agentic RAG;
- **human-calibrated uncertainty** and statistical confidence intervals;
- **hard factuality gates before preference ranking**;
- **continuous meta-evaluation on fresh, hidden systems and topics**.

The first release should be the reproducible benchmark and meta-evaluation harness. The open-weight judge should only become an automated production gate after it demonstrably improves both evidence-informed human preference agreement and objective claim-level precision/recall on a fresh, model-family-held-out test set.

---

## Implementation status in this repo

This document is the full research-program proposal (16-week roadmap, human annotation
collection, fine-tuned open-weight judges). What actually ships as code in
`judges/ragree/` is scoped down to what a single coding session can responsibly build
inside this starter kit's `judge()`/`create_nuggets()`/`create_qrels()` protocol:

- **Built**: a scoped Evidence Judge lane — sentence-level claim proxies, four-way
  verification against retrieved context, `context_faithfulness`,
  `estimated_coverage`, `claim_f1`, citation precision/coverage, and evidence-coverage
  reporting — all via prompted LLM calls through `llm_config` (no fine-tuning or
  training pipeline). `estimated_coverage` is not gold-denominator claim recall, and
  sentence-level claims are not full atomic claim decomposition.
- **Not built**: the Preference Judge (PRePair pairwise procedure, heterogeneous jury,
  Bradley-Terry fusion), the Trajectory Judge (needs agent-trace input this starter kit's
  `Report`/`Request` model does not carry), the learned monotonic fusion layer, and all of
  Sections 8-12 (training data collection, fine-tuning, meta-evaluation harness beyond this
  repo's existing `auto-judge-evaluate meta-evaluate`). These need real human-annotation
  infrastructure and multi-week turnaround, not a coding change.
