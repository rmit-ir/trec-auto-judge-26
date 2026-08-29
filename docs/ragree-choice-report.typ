#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.3cm),
  numbering: "1",
  number-align: center + bottom,
)

#set text(size: 11pt, lang: "en")
#set par(justify: true, leading: 0.52em)
#set heading(numbering: "1.")

#show heading.where(level: 1): it => block(
  above: 0.8em,
  below: 0.35em,
)[
  #set text(size: 14pt, weight: "bold")
  #it
]

#show heading.where(level: 2): it => block(
  above: 0.55em,
  below: 0.25em,
)[
  #set text(size: 11pt, weight: "bold")
  #it
]

#let node(body) = box(
  width: 100%,
  height: 38pt,
  inset: 2.5pt,
  radius: 2pt,
  stroke: 0.6pt + rgb("#52677a"),
  fill: rgb("#edf3f7"),
  align(center + horizon, text(size: 6.8pt, weight: "semibold", body)),
)

#align(center)[
  #text(size: 18pt, weight: "bold")[RAGree Choice / SPARROW]
  #linebreak()
  #text(size: 11pt, weight: "semibold", fill: rgb("#405466"))[
    An Evidence-Gated Preferential AutoJudge
  ]
]

#v(0.4em)

#block(
  inset: 7pt,
  radius: 2pt,
  fill: rgb("#f4f6f8"),
  stroke: 0.5pt + rgb("#b4bec7"),
)[
  *Abstract.* RAGree Choice is a preferential AutoJudge for selecting a winner,
  top-$k$, or acceptable response from a candidate pool. Its SPARROW pipeline
  applies a schema-valid pointwise evidence gate before acquiring hard
  pairwise ballots, fits a weighted Davidson--Bradley--Terry model with explicit
  ties, directs additional comparisons toward an uncertain top-$k$ boundary,
  selectively reverses answer position, and audits the operational ranking
  using Condorcet, cycle, and Kemeny--Young diagnostics. The governing rule is:
  *every judge gets a vote; evidence gets a veto.* A live synthetic smoke run
  required 15 LLM calls rather than 20 for an ungated comparison baseline and
  produced stable, interpretable rankings. These measurements establish
  end-to-end mechanism execution, not real-world judging accuracy.
]

= Theoretical framework

*Preference models with ties.* Bradley--Terry models represent pairwise choice
through latent utilities; Davidson's extension assigns positive probability to
ties rather than treating them as half-observed wins. This is appropriate for
RAG evaluation, where materially indistinguishable answers are common and a
forced directional verdict can create spurious separation. SPARROW preserves
four hard ballot outcomes—`A`, `B`, `TIE`, and `ABSTAIN`—and excludes
abstentions from directional fitting.

*Pointwise-first, pairwise-final judging.* PairS and PRePair motivate separating
absolute inspection from comparative judgment. RAGree Choice first constructs
one evidence-grounded brief per candidate and only then presents eligible
briefs and answers pairwise. This reduces pairwise inconsistency caused by
judges discovering criteria opportunistically, while retaining the greater
discriminative power of direct comparison. Eligibility requires no critical
error and factual-support and citation-validity estimates at or above their
configured thresholds. Completeness contributes to
`POINTWISE_EVIDENCE_SCORE` but is not itself a gate threshold.

*Social choice and auditing.* A strict Condorcet winner defeats every other
eligible candidate in the observed weighted preference matrix. Condorcet
cycles expose collective inconsistency, while a Kemeny--Young order maximizes
agreement with all directed margins. Exact Kemeny optimization is NP-hard and
requires factorial enumeration in this implementation; it is therefore an
audit, not the live rule. Exact search is limited to at most eight candidates,
after which a Davidson--BT-seeded adjacent-swap local search is diagnostic
only. A Davidson--BT/Kemeny winner disagreement sets `RANKING_STABLE` to zero
and signals targeted acquisition or review.

*Judge reliability and effective panels.* Reliability-aware BT variants such
as BT-$sigma$ model judge-specific noise or confidence rather than assuming
exchangeable voters. Related panel-of-LLMs studies find that a nominal panel
often contains far fewer effective-independent votes than its raw judge count:
shared training data, model lineage, prompts, and systematic biases induce
correlated errors. Consequently, the ballot schema retains `judge_id` and a
positive `weight`, and aggregation is judge-weighted and multi-family-ready.
The current AutoJudge prototype nevertheless uses only its one injected LLM
endpoint; repeated calls are not misrepresented as an independent jury.

= Design and implementation

#figure(
  block(width: 100%)[
    #grid(
      columns: (
        1fr, 8pt, 1fr, 8pt, 1fr, 8pt, 1fr, 8pt, 1fr, 8pt, 1fr, 8pt,
        1fr, 8pt, 1fr, 8pt, 1fr,
      ),
      column-gutter: 1pt,
      node[Evidence brief],
      align(center + horizon)[→],
      node[Eligibility gate],
      align(center + horizon)[→],
      node[Sparse seed graph],
      align(center + horizon)[→],
      node[Pairwise ballots],
      align(center + horizon)[→],
      node[Davidson--BT fit],
      align(center + horizon)[→],
      node[Adaptive boundary selection],
      align(center + horizon)[→],
      node[Selective position reversal],
      align(center + horizon)[→],
      node[Condorcet / cycle / Kemeny audit],
      align(center + horizon)[→],
      node[Leaderboard],
    )
  ],
  caption: [SPARROW pipeline. Evidence eligibility precedes preference acquisition; social-choice rules independently audit the fitted live ranking.],
) <pipeline>

== Davidson--BT aggregation

For utilities $u_a, u_b$, let $delta = u_a-u_b$ and tie factor $tau > 0$.
The implementation uses

$ D = e^delta + 1 + tau e^(delta/2), $

$ P(A " beats " B) = e^delta / D, quad
  P(B " beats " A) = 1 / D, quad
  P("tie") = tau e^(delta/2) / D. $

The fit is weighted MAP gradient ascent with an L2 prior toward zero.
Utilities are identified only up to an additive constant, so every iteration
subtracts their mean. For numerical stability, the implemented utility
difference is clipped to the interval $[-30, 30]$. `CHOICE_SCORE` is 100 times
the mean fitted probability of beating a uniformly selected eligible
opponent; it is pool-relative and is not comparable across changing pools
without frozen anchors. `PAIRWISE_WIN_RATE` is the observed weighted fraction
of non-abstaining ballots won, with ties contributing one half.

== Acquisition, failure semantics, and reversal

Pools of at most five eligible candidates use a full round robin. Larger pools
begin with a deterministic cycle, guaranteeing connectivity, then greedily add
the not-yet-connected pair having minimum summed current degree until
$ceil(n d / 2)$ edges are present. Requested degrees below two mean the minimum
connected degree of two. An adaptive max-heap then ranks unseen pairs by

$ (2 I["crosses top-" k " boundary"] + 1/(1+abs(u_a-u_b))) / "expected cost", $

concentrating calls on uncertain comparisons capable of moving the decision.

A parsed `ABSTAIN` is a completed ballot and consumes an acquisition edge;
transport or schema failure creates no ballot and remains schedulable after
bounded retry. Likewise, a missing or malformed evidence brief yields
`EVIDENCE_EVALUATED = 0`, not an inferred factual score or gate failure.
`NO_ACCEPTABLE_ANSWER = 1` is emitted only when every candidate has a
schema-valid brief and every candidate fails eligibility.

Selective position reversal rechecks, within budget, directional ballots
crossing the top-$k$ boundary first and then `TIE` ballots; existing
`ABSTAIN`s are never reversed. The answer order is swapped and the returned
vote is mapped back to the original orientation. Agreement preserves the
single original ballot. Genuine disagreement replaces it with one `ABSTAIN`,
not a `TIE`, because position sensitivity is uncertainty about direction rather
than evidence of equality. A failed reversal leaves the ballot unchanged and
does not increment `POSITION_REVERSAL_CHECKED`.

= Empirical results and analysis

#figure(
  block[
    #set text(size: 7.5pt)
    #table(
      columns: (0.65fr, 0.65fr, 0.75fr, 0.9fr, 0.75fr, 0.9fr),
      align: (center, center, center, center, center, center),
      inset: 2.6pt,
      stroke: 0.35pt + rgb("#9daab4"),
      table.header(
        [*$n$*],
        [*Degree*],
        [*Edges*],
        [*Full RR*],
        [*% full*],
        [*Max spread*],
      ),
      [10], [2], [10], [45], [22.2%], [0],
      [10], [3], [15], [45], [33.3%], [2],
      [10], [4], [20], [45], [44.4%], [0],
      [10], [5], [25], [45], [55.6%], [2],
      [16], [2], [16], [120], [13.3%], [0],
      [16], [3], [24], [120], [20.0%], [0],
      [16], [4 (default)], [32], [120], [26.7%], [2],
      [16], [5], [40], [120], [33.3%], [2],
    )
  ],
  caption: [Measured sparse-seed acquisition. The $n=16$, degree-4 result reproduces the design proposal's 32-edge arithmetic.],
) <sparse-data>

@sparse-data confirms that the connected seed graph is substantially
smaller than a full round robin while remaining near-regular. Across the
separate sweep $n=6..29$ and degree $2..7$, the worst observed degree spread
was two. Exact regularity was intentionally rejected as cosmetic complexity:
connectivity is the correctness requirement, and it holds for every
nonnegative requested degree.

#figure(
  block[
    #set text(size: 7.1pt)
    #table(
      columns: (0.75fr, 0.55fr, 0.65fr, 0.9fr, 0.7fr, 0.75fr, 0.75fr),
      align: (left, center, center, center, center, center, center),
      inset: 2.2pt,
      stroke: 0.35pt + rgb("#9daab4"),
      table.header(
        [*Topic / run*],
        [*Eligible*],
        [*Pointwise*],
        [*Win rate*],
        [*Choice*],
        [*Condorcet*],
        [*Stable*],
      ),
      [cloud / run1], [1.0], [0.933], [1.0], [88.01], [1.0], [1.0],
      [cloud / run2], [1.0], [0.833], [0.5], [41.79], [0.0], [1.0],
      [cloud / run3], [0.0], [0.600], [0.0], [0.00], [0.0], [0.0],
      [cloud / run4], [1.0], [0.457], [0.0], [1.77], [0.0], [1.0],
      [leaf / run1], [1.0], [0.967], [1.0], [85.29], [1.0], [1.0],
      [leaf / run2], [1.0], [0.783], [0.0], [1.92], [0.0], [1.0],
      [leaf / run3], [0.0], [0.450], [0.0], [0.00], [0.0], [0.0],
      [leaf / run4], [0.0], [0.233], [0.0], [0.00], [0.0], [0.0],
    )
  ],
  caption: [Live RAGree Choice output on the synthetic `kiddie` smoke-test data. Choice is `CHOICE_SCORE`; Pointwise is `POINTWISE_EVIDENCE_SCORE`.],
) <leaderboard-data>

The live run contained two topics, four competing runs per topic, and eight
responses. It made eight unconditional evidence-brief calls, four initial
pairwise calls among eligible candidates, and three reversal calls, for 15
calls total. An ungated baseline would compare all six pairs per topic:
12 pairwise calls plus the same eight briefs, or 20 total. Thus gating reduced
pairwise volume by 66.7% and overall calls by 25% in this small case. No
adaptive calls were needed because each eligible pool was already exhausted by
its small-pool round robin.

@leaderboard-data illustrates the veto semantics. `run3` has nontrivial
pointwise scores on both topics but is ineligible and therefore receives no
preference wins. `run1` is the strict Condorcet winner and has the highest
pointwise score on both topics; Davidson--BT and Kemeny agree on each eligible
winner. `CYCLE_RATE` was 0.0 throughout. Position reversal checked two cloud
ballots and one leaf ballot, with disagreement rate 0.0, meaning no position
inconsistency was detected among the checked comparisons—not that reversal was
omitted.

The `kiddie` collection is a synthetic pipeline smoke test, not a real quality
benchmark. These values demonstrate that gate, ballot acquisition, fitting,
reconciliation, audit, and leaderboard emission execute end to end and produce
differentiated outputs. They do *not* demonstrate real-world judge accuracy or
human alignment. At report time, 39 regression tests passed across pure
ranking primitives, configuration and failure handling, and an offline
end-to-end integration test including eligible, ineligible, unavailable, and
reversal-reconciliation cases.

= Limitations

The prototype has one injected LLM endpoint and therefore no genuine
multi-model jury, despite its judge-weighted, multi-family-ready ballot model.
Its pointwise scores are LLM estimates, not gold claim-level precision or
recall; no authoritative-corpus or retrieval calibration is supplied. There is
also no human calibration of eligibility thresholds, utility scale, judge
reliability, correlated error, or position-bias rates. Pool-relative scores
require frozen reference anchors for longitudinal comparison, and large-pool
Kemeny output is only a local-search diagnostic.

= References

#set text(size: 9pt)
#set par(leading: 0.42em)

#enum(
  spacing: 0.5em,
  [Bradley, R. A., and Terry, M. E. *Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons.*],
  [Davidson, R. R. *On Extending the Bradley--Terry Model to Accommodate Ties in Paired Comparison Experiments.*],
  [*PairS* and *PRePair*: pointwise-first, pairwise-final methods for LLM-based response evaluation and ranking.],
  [Condorcet, M. de. Majority preference and the Condorcet winner criterion.],
  [Kemeny, J. G., and Young, H. P. Consensus ranking and maximum-agreement social choice.],
  [Reliability-aware Bradley--Terry extensions, including *BT-$sigma$*, for judge-specific uncertainty and heterogeneous evaluator quality.],
  [*Panel of LLM Evaluators (PoLL)* and related diverse-model jury studies: correlated model judgments and the gap between nominal and effective-independent panel size.],
  [RAGree Choice design summary and `sparrow.py` implementation: evidence gating, sparse acquisition, weighted Davidson--BT fitting, selective position reversal, and Condorcet/Kemeny auditing.],
)
