# Source ledger

Reviewed 2026-11-11. Paths below are relative to the implementation repository,
unless explicitly described otherwise. This ledger is editorial support, not paper content.

## Snapshots and adaptation

- Report style: `../trec-rag-26-report/main.tex`, section organization, tables,
  and bibliography. Reused ACM `sigconf` presentation, numbered pages, section
  files, and relevant references. No RAG system results were transferred.
- Authorship: the user confirmed Oleg Zendel is the sole author.
- Acknowledgments: adapted from `../trec-rag-26-report/main.tex` at the user's
  request, retaining the ADM+S grant and RACE computing acknowledgment and
  changing the track name to AutoJudge.
- The implementation repository is linked by URL in an introduction footnote,
  rather than included as a bibliography citation.
- Local implementation checkout: `../auto-judge-starter-kit`, fast-forwarded
  from `5435458` to `ed525ce` on 2026-11-11; now matches Sunfyre and origin/main.
- Newer remote implementation checkout: Sunfyre
  `/home/el7/E103037/repos/auto-judge-starter-kit`, HEAD
  `ed525ce3a0b37bd5da291b5c05905e235338a35c`, clean at review.
- Policy read: implementation `AGENTS.md`. No candidate report text, identifying
  information, response-derived prompt caches, or model completions were opened.
  Saved scored outputs were used only for format/completeness checks.

## Claims and sources

| Report content | Primary evidence |
| --- | --- |
| Four RAG26-only uploaded data runs, including `ragree-default` | Author-provided TIRA screenshot, retained as `evidence/tira-rag26-run-uploads.png`; author explicitly confirmed all data submissions were on RAG26 only |
| Three Arena uploads, primary runtime model, approximate costs, failed support lane, filter rejections | `judges/rag26_arena/README.md`, Submission status; remote version adds the four-run metadata pointer |
| RAGree sentence grounding, seven metrics, four nugget questions, six-word heuristic, missing-evidence handling | `judges/ragree/ragree_judge.py`, `judges/ragree/workflow.yml`, `judges/ragree/README.md` |
| Arena prompts and truncation; rubric sign/order rendering | `judges/rag26_arena/prompts.py` |
| Actual ranking score, retries, independent support, reversal behavior | `judges/rag26_arena/judge.py` |
| Sparse graph, Davidson tie model, tie factor, acquisition, exact/heuristic Kemeny, Condorcet/cycle limitations | `judges/ragree_choice/sparrow.py`; Arena adapter above |
| Support denominators, availability, hard-first convention | `judges/rag26_arena/support.py` and measure descriptions in `judge.py` |
| Shared production settings | All four saved remote `.config.yml` files listed below; `judges/rag26_arena/workflow.yml` |
| Offline pipeline, generation model, corpus backends, omitted stages | `judges/rag26_arena/README.md`; `rubrics/build.py`, `finalize.py`, `weights.py`, and `hybrid_retrieval.py` |
| V2 motivation and prompt revisions | `judges/rag26_arena/README.md`, Frozen rubric artifacts |
| Three-topic model-selection exercise | `evaluation/rubric-model-bakeoff/summary.json`; constructor/judge identities in the README |
| Synthetic end-to-end/cache checks and container dry-run | `judges/rag26_arena/README.md`; `docs/rag26-tira-dryrun-handoff.md`; no new judge execution performed |
| Two actual code submissions: RAGree and Choice | Explicit author confirmation on 2026-11-11; supersedes the older handoff describing proposed Arena code submissions |
| Choice evidence gate, preference ranking, and diagnostics | `judges/ragree_choice/README.md` and `judges/ragree_choice/ragree_choice_judge.py` |
| 2025 pilot-truth gap | `evaluation/README.md`; reported as a documented blocker, not a new filesystem verification |

The method-description document says every ballot is reversed in one early paragraph;
this conflicts with executable settings. The report follows the actual budget of two
selected reversals per topic. Design plans also propose pointwise rubric scoring and
larger architectures that were not implemented; those are not described as submitted.

## Saved remote execution artifacts

Under `output/rag26-generation/`:

| Directory | File prefix | Config timestamp (UTC) | Recorded commit | Dirty |
| --- | --- | --- | --- | --- |
| `ragree/default-all-all` | `ragree` | 2026-09-12 15:26:37 | `bd3083985b985cda380d21c8ece90d0e75508b46` | true |
| `rag26_arena/arena-support-all-all` | `rag26-arena` | 2026-10-14 09:09:16 | `84b2fb06a0e1a9ebbd1d6f4bccd9195550a8cd7c` | true |
| `rag26_arena/rubric-support-arena-all-all` | `rag26-arena` | 2026-10-14 09:06:53 | `84b2fb06a0e1a9ebbd1d6f4bccd9195550a8cd7c` | true |
| `rag26_arena/rubric-v2-support-arena-all-all` | `rag26-arena` | 2026-10-14 10:49:37 | `c912570a577081d36ca4dff2381ba5820264c0dd` | true |

Each prefix has `.config.yml`, `.eval.txt`, and `.eval.measures.yml` artifacts.
Only `.config.yml` and structural/numeric properties of `.eval.txt` were read.
All four configs specify `gpt-5.6-luna`.

Remote output checks: whitespace-delimited columns are run, topic, measure, value.
All four files have 83 run IDs, 119 non-aggregate topic IDs plus `all`, unique
(run, topic, measure) keys, and finite numeric values. Each Arena output has 16
measures and 159,360 rows (including 1,328 aggregate rows). RAGree has 7 measures
and 69,720 rows (including 581 aggregate rows). Counts saturate the Cartesian
product of identifiers and measures. No score distribution, system ranking,
response text, or quality comparison was surfaced or used.

## Frozen artifacts

Both files are under `judges/rag26_arena/rubrics/frozen/` and contain 119 topics:

| Artifact | Criteria | Min/max per topic | SHA-256 |
| --- | --- | --- | --- |
| `rag26-rubrics.jsonl` | 3,536 | 22 / 31 | `648a57ce23e85718f4b86e39f593152f7109b3efc8d34a90bee67e524be295c6` |
| `rag26-rubrics-v2.jsonl` | 3,547 | 22 / 31 | `029badb50dadb64b289dffae9edbb7198726b72edf38bdfcdbe5ccfdf9bcecc6` |

Counts and hashes were computed on Sunfyre from response-independent artifacts.
Mean counts are 3,536/119 = 29.71 and 3,547/119 = 29.81 (rounded).
The comparison budget is derived from the source graph implementation:
83*4/2 = 166 seed edges; 166*119 = 19,754; (166+4+2)*119 = 20,468;
83*82/2*119 = 404,957 exhaustive pairs. These are not provider billing counts.

## Outstanding verification

See `gap-analysis.md`. The author confirmed that official results will be released
only after the TREC conference and that this report is submitted beforehand.
The manuscript treats official meta-evaluation as post-conference work, not a
missing result needed for submission. Bibliographic records were reused from the reference
report; ResearchRubrics metadata was also checked against arXiv:2511.07685.

## Author-confirmed final submission record (2026-11-11)

Two code submissions: RAGree and Choice. No additional code submissions are
asserted. All four data submissions were RAG26 only. The retained screenshot
shows dataset `rag26-20260827_1-test` for every row and these upload timestamps
(timezone not shown):

| Data run | Displayed upload timestamp |
| --- | --- |
| `rag26_arena-rubric-v2-support-arena` | 2026-10-14-14-09-20 |
| `rag26_arena-rubric-support-arena` | 2026-10-14-12-43-40 |
| `rag26_arena-arena-support` | 2026-10-14-07-23-10 |
| `ragree-default` | 2026-09-13-02-30-13 |

The screenshot confirms data uploads; code submission evidence is the author's
confirmation. Neither is claimed as evidence of official evaluation results.

The pre-conference publication timing is based on the author's explicit
confirmation; no additional external schedule verification was needed.

## Documentation and ledger follow-up

See `documentation-review.md` for the document inventory, conflicting agent
notes, and reconciliation. `evidence/cost-ledger-review.json` preserves the
reviewed numeric evidence and its scope. The RAGree cost record in
`docs/cost-estimation.md` was initially overlooked: 209,646 calls, 128,904,481
input and 23,772,804 output tokens, estimated $54.31 over two phases.
The historical rate calculation returns $54.308261.

The bake-off `usage.json` files are itemized ledgers. Their summary totals
include prior-spend reserves; no cumulative grand total is inferred. Numeric
construction cache fields were queried using SQLite JSON extraction only.
The shared runtime cache was examined only for schema, row counts, and
aggregate numeric usage by UTC date, never response text or raw response JSON.
Its 375,403 retained rows cannot be attributed to individual judge runs or
models from timestamps alone, and are not treated as a billing ledger.

Git comparison found no changes to the RAGree runtime between its recorded
base commit and current HEAD, and none to Arena runtime preference, support,
or SPARROW source between the recorded October base and current HEAD. V2's
workflow stanza was added by `45ba179`, after the saved V2 config's base
commit `c912570`; this explains one subsequently committed configuration
change but does not establish the entire dirty snapshot.

## Verified manuscript cost figures (2026-11-12)

The cost paragraph now uses only $54.31 reconstructed from the recorded RAGree
token counts and historical price table, plus the independently summed itemized
model-selection ledgers: $0.1616000765 and $0.0629127623, reported as $0.16 and
$0.06. Approximate Arena figures are omitted from manuscript text.
A separate rough construction estimate ($5–6 for retained successful requests,
assuming production Luna rates) is documented in the review notes only.

## Construction estimate included (2026-11-12)

At the author's explicit request, the manuscript now includes the rough US$5–6
construction estimate, with the production-rate assumption and exclusions for
model selection and unrecorded failed/overwritten calls. Numeric basis remains
`evidence/cost-ledger-review.json`, `construction_rough_estimate_review`.

## RAGree development and estimated-coverage motivation (2026-11-12)

Expanded the Evidence Judge section using `docs/ragree-judge-design.md`
(especially the denominator discussion and implementation-status boundary)
and `judges/ragree/ragree_judge.py` (`RAGreeNuggetCreator`, coverage prompt,
coverage aggregation, `_parse_yes`, and `_harmonic_mean`). Verified that the
coverage prompt receives `get_report_text()` without an explicit character
slice, whereas Arena slices answers to 2,000 characters. Coverage questions
are shared per topic, equally weighted, and generated without candidate input.
Four questions are the default, not an unconditional guarantee: parser fallback
can produce fewer, and no parsed questions triggers a topic-title fallback.
Failed coverage calls receive no YES credit with the bank denominator retained.

The benefits/risks/constraints/recommendation illustration is a hypothetical
example, not restricted evaluation content. The expectation that estimated
coverage is the most promising standalone RAGree signal reflects the author's
explicit methodological judgment, not an empirical ranking or a claim derived
from hidden results. Official performance remains post-conference work.

## Rubric-construction motivation and citations (2026-11-12)

Added explicit citations to the organizers' ResearchRubrics development release
and public RAG task/evaluation guidelines, plus direct ResearchRubrics citations
in the construction section. The same Sharma et al. bibliography entry appears
in `../trec-rag-26-report/references.bib` and is used in that report's internal
evaluation section to describe the 30 organizer development topics. The local
`evaluation/researchrubrics/dev-rubric-stats.md` confirms the 30-topic dev-file
count. This is a shared development resource, not transfer of RAG system scores
or experiments into this report.

Primary web sources checked: the official `TREC-RAG/trec-rag-data` development
directory lists `researchrubrics-dev-rubrics`; `TREC-RAG/trec-rag-skills`
`references/rag-task.md`, Evaluation section, describes battles and individualized
nugget/rubric scoring; arXiv:2511.07685 confirms expert-authored ResearchRubrics.
The interpretation that this suggested a potential official evaluation
direction is explicitly presented as our design motivation, not a promise that
our generated criteria replicate the final organizer assessments.

## Title terminology and sparse preference citations (2026-11-12)

Retained the title and made its method terms explicit in the abstract:
evidence grounding checks support in cited documents; preference is the LLM's
prediction of which answer a user would prefer; sparse means comparing only
a small subset of possible answer pairs; rubric guidance supplies frozen,
response-independent assessment criteria to those comparisons.

The Arena section now defines the comparison graph and cites Negahban, Oh,
and Shah's *Rank Centrality: Ranking from Pairwise Comparisons* (2017,
DOI `10.1287/opre.2016.1534`) for the general incomplete-comparison ranking
problem and the relevance of graph structure, not as the algorithm implemented
by SPARROW or a guarantee for our degree-four graph. The authors' arXiv
record (`https://arxiv.org/abs/1209.1688`) supports that description; the
University of Illinois publication record confirms the journal metadata.
Added Davidson's original tie-model paper (1970,
DOI `10.1080/01621459.1970.10481082`) alongside the existing Bradley--Terry
citation. Its publisher search record confirms the title, author, journal,
volume, issue, pages, and tie-extension scope; direct publisher-page access
returned HTTP 403.

## Recent-literature review (2026-11-13)

Added six references from 2025--2026 after checking primary papers and
publication metadata. These support methodological rationale and limitations;
they do not establish the submitted judges' performance or validate their
particular thresholds, question count, comparison budget, or truncation limits.
The classical ranking-model and RRF references remain as algorithm provenance.

| Manuscript location | Added source and supported point | Boundary |
| --- | --- | --- |
| Estimated coverage | Pradeep et al., *The Great Nugget Recall* (SIGIR 2025), [author-hosted published paper](https://cs.uwaterloo.ca/~jimmylin/publications/3726302.3730090.pdf), DOI `10.1145/3726302.3730090`: separates nugget creation from assignment and validates automatic nugget evaluation against human assessments. | Their nuggets use relevant-document pools; RAGree uses request-derived questions. Their human agreement is not evidence for our question bank. |
| Grounding | Thakur et al., *Assessing Support for the TREC 2024 RAG Track* (SIGIR 2025), [author-hosted published paper](https://cs.uwaterloo.ca/~jimmylin/publications/3726302.3730165.pdf), DOI `10.1145/3726302.3730165`: compares sentence--document support judgments with human assessments. | Supports the evaluation target, not our weights or missing-evidence rules. |
| Sparse Arena | Daynauth et al., *Ranking Unraveled* (ACL 2025), [paper and metadata](https://aclanthology.org/2025.acl-long.1265/), DOI `10.18653/v1/2025.acl-long.1265`: ranking aggregation and matchup distributions affect ranking behavior. | No validation of SPARROW, degree four, or our adaptive budget is inferred. |
| Reversal checks | Shi et al., *Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge* (IJCNLP/AACL 2025), [paper and metadata](https://aclanthology.org/2025.ijcnlp-long.18/), DOI `10.18653/v1/2025.ijcnlp-long.18`: position sensitivity varies by judge and task. | Motivates checking sensitivity, not a claim that two reversals remove bias. |
| Rubric construction and limitations | Zhou et al., *RubricBench* (ACL 2026), [paper and metadata](https://aclanthology.org/2026.acl-long.1439/), DOI `10.18653/v1/2026.acl-long.1439`: Section 3.1 uses instruction-derived rubrics without candidate responses; experiments identify a gap between generated and human-authored criteria. | Supports response independence and the need for validation, not effectiveness of our V2 changes. |
| Construction-model selection | Chen et al., *Do LLM Evaluators Prefer Themselves for a Reason?* (2025), [arXiv record](https://arxiv.org/abs/2504.03846): distinguishes legitimate from harmful self-preference. | The paper studies preference for a model's own outputs; its limitations discuss same-family preference as broader future work. Our cross-model family effect remains untested. |

The two SIGIR bibliography records were reused from the RAG report after
checking the published author copies, including the additional NIST coauthors
absent from the earlier arXiv versions. ACL authors, years, page ranges, and
DOIs were checked against the Anthology and PDF title pages. Chen et al. is
cited as an arXiv preprint, with no unverified conference claim.

## Sparse preference elicitation and voting references (2026-11-13)

Selected two direct references for the Arena section:

- Liu et al., *Aligning with Human Judgement: The Role of Pairwise Preference
  in Large Language Model Evaluators* (COLM 2024), introduces PairS:
  uncertainty-guided LLM comparisons for ranking generated texts. Checked
  the [author manuscript](https://arxiv.org/html/2403.16950v5) and
  [coauthor's institutional publication record](https://research.monash.edu/en/publications/aligning-with-human-judgement-the-role-of-pairwise-preference-in-/).
- George and Dimitrakakis, *Eliciting Kemeny Rankings* (AAAI 2024), relates
  adaptive pairwise preference elicitation to Kemeny aggregation and voting.
  Checked the [published author copy](https://libra.unine.ch/server/api/core/bitstreams/a50b4d24-a527-4741-89c5-6da92cf2608b/content)
  and the AAAI volume contents (38(11), 12164--12171). DOI:
  `10.1609/aaai.v38i11.29105`.

These are related approaches, not asserted historical inspirations or
algorithms implemented by SPARROW. The operational fit remains Davidson--BT;
Kemeny is an audit with exact search only through eight candidates. Neither
source establishes guarantees for our graph or acquisition budget. Soft
Condorcet Optimization (Lanctot et al., AAMAS 2025) was considered but omitted:
its alternative rating objective is less directly matched to our implementation
than PairS acquisition and Kemeny elicitation. Direct OpenReview and AAAI
article access encountered bot protection; the accessible primary author
copies and institutional/proceedings metadata supplied the evidence.

## Selective reviewer revision and internal diagnostics (2026-11-13)

The attached review was adopted selectively: retain the TREC RAG report's ACM
layout and plain-language SPARROW description; consolidate the author-requested
extreme-budget caveat in Discussion; clarify runtime model identifiers, rubric
formatting, emitted measures, and nominal versus measured resource usage.
Do not restore equations merely for visual symmetry, invent an endpoint alias,
claim a public exact-replay archive, or erase incomplete execution snapshots.
The suggested table typos were already corrected in the source.

`evidence/analyze_saved_outputs.py` aggregates only the four saved runner
`.eval.txt` files listed above. It was run on Sunfyre with the implementation
checkout's existing SciPy 1.15.3 environment. `evidence/internal-diagnostics.json`
records source-file hashes, results, denominators, and limitations. Assertions
check complete finite score grids, matching candidate/topic sets, repeated
topic-level diagnostics, integral reconstructed reversal counts, and tie-aware
rank-correlation examples. A separate standard-library implementation of average
ranks, Pearson correlation, and Kendall tau-b independently matched all 357
topic/run-pair comparisons and winner-agreement totals.

Only the 119 individual topics are analyzed; existing `all` aggregates are not
added again. Repeated per-topic diagnostics are counted once, not 83 times.
The table's BOTH_BAD percentage is a mean of topic proportions; reversal rates
pool successful checks. Citation correlations retain only fully evaluated support
(`RAG26_SUPPORT_EVALUATED=1`); RAGree correlations use emitted scores and inherit their missing-data
limitations. No candidate text, response-derived prompts, caches, or raw ballots
were inspected. Full ballot-category rates are not recoverable from the scored
outputs alone. Results describe judge behavior, cannot separate rubric effects
from LLM variability without repeat-run controls, and do not validate human
agreement or select/tune a judge on test topics.

## Additional agreement analyses integrated (2026-11-13)

Reviewed implementation commit `89f38b1ab94e9d43a0101095be0f35f64ca2e45f`,
including `docs/judge-agreement-analysis.md`, `scripts/agreement_heatmap.py`,
`scripts/topic_alpha.py`, `scripts/agreement_table.py`, and the committed CSV/TSV
outputs. `evidence/additional-agreement-review.json` records file hashes,
selected coefficients, alpha summaries, validation, and interpretation limits.

The aggregate matrices compare rankings of the 83 systems' mean scores over
119 topics; they do not average correlations of topic-level rankings. All 288
heatmap cells were reproduced using the scripts' CSV parsing, and the 144
Kendall cells agreed with SciPy. The two CSV parsers differ slightly in floating
point ties for aggregate critical-error rates; those pair coefficients are not
used in the paper. The reported Arena pair comparisons are unaffected.

All 238 per-topic ordinal-alpha values (three Arena judges and twelve pooled
outputs) were recomputed from the saved scored files on Sunfyre. Each matched
the separately installed `krippendorff.alpha` ordinal implementation within
`5.56e-16`. Systems are units, metric columns are raters, and within-topic average
ranks handle ties and different score scales. The three-judge mean is 0.763315,
median 0.780020, range 0.403596--0.900204. The pooled mean is 0.243924; the
twelve outputs overlap and target different constructs, so this is exploratory
concordance rather than reliability of one quality variable. No 0.667 or other
reliability threshold is applied.

Cross-metric aggregate correlations use emitted scores without availability
adjustment; unavailable support and failed judgments can therefore affect them.
The older masked per-topic first-citation correlations must not be relabeled as
these unmasked aggregate mean-citation correlations. Also, a lower top-weighted
coefficient is not by itself proof that all disagreements occur near the top.
The analysis document's statement that Kemeny agreement is absent from scored
outputs is incorrect: `RANKING_STABLE` provides that diagnostic, retained here.
No judge settings or variant selection were changed using test-topic results.

Method references were verified against primary sources: Urbano and Marrero,
*The Treatment of Ties in AP Correlation*, ICTIR 2017, 321--324,
DOI `10.1145/3121050.3121106`, via the TU Delft institutional record; and
Krippendorff, *Computing Krippendorff's Alpha-Reliability* (2011), via the
University of Pennsylvania working paper and author-hosted Annenberg PDF.
These support the statistics, not accuracy or human agreement for our judges.

For the six-page revision, aggregate and top-weighted comparisons replace the
same-winner column in the diagnostic table (winner counts remain in prose).
The older per-topic Spearman paragraph is replaced by clearly labeled aggregate
Kendall associations. The nominal budget table is removed because its counts
are already present in the SPARROW figure, introduction, and resource-usage
paragraph. Both pipeline figures, the full extreme-budget caveat, and the ACM
typography used by the TREC RAG report are retained. No code or judge settings
in the implementation repository are changed.

## Publication figures for the seven-page report (2026-11-13)

The author permits extension to seven pages only for figures and text explaining
their analyses/results. The analysis checkout remains at `89f38b1`; all reviewed
files still match the hashes recorded in `evidence/additional-agreement-review.json`.
The two matrices and per-topic alpha CSV are copied byte-for-byte to
`figures/data/`. `figures/render_agreement.py` generates vector PDF assets from
these snapshots, without inspecting candidate answers or rerunning judges.
It checks source hashes, finite/aligned/symmetric matrices with unit diagonals,
and the independently verified 119-topic alpha summaries before plotting.

The agreement figure uses only the box plot, with the three Arena preference
judges and twelve heterogeneous outputs. Means, medians, Tukey whiskers, and
outliers are shown; the original topic-order panel and reliability-threshold
line are omitted. Both aggregate heatmaps share the same labels and symmetric
minus-one-to-one scale. Report-aligned names replace raw script aliases; Plain's
citation columns remain weighted mean-citation measures and critical-error
rates remain inverted. Captions retain the construct-mixing, missing-data,
and human-validation limits already established in the text.

Linux Libertine, the report's text family, is embedded in the plot PDFs rather
than falling back to Matplotlib's default fonts. The blue/lavender/teal colors
derive from the existing pipeline figures. Heatmap labels and cell annotations
are sized for the actual full-width figure, rather than shrinking the original
large plotting canvas. Existing methods, tables, and pipeline figures are
preserved; manuscript edits are limited to the new figures/captions and their
analysis cross-references.

## Combined heatmap and original box-plot styling (2026-11-13)

At the author's request, Figure 3 combines the two symmetric correlation
matrices into a single display: Kendall tau-b above the diagonal and
top-weighted tau-AP-b below it. Diagonal self-correlations are blank. Both
triangles retain the same source values and shared minus-one-to-one color scale;
this display is intentionally asymmetric because the triangles show different
statistics. Captions and plot labels identify the mapping explicitly.

Figure 4 returns to the original analysis script's unfilled black box outlines,
orange median lines, green mean triangles, and translucent blue/orange topic
points. The report's Linux Libertine font is retained. Only the box-plot panel
is used; the topic-order panel and unsupported reliability-threshold line remain
omitted. The input CSV snapshots and all reported statistics are unchanged.

## Figure-led interpretation of the agreement trends (2026-11-13)

The analysis text now explicitly links each figure's main observations to brief,
tentative explanations. The verified aggregate Kendall coefficients within
RAGree are 0.051516 for estimated coverage versus context faithfulness and
0.820746 for citation precision versus context faithfulness. These support
describing both distinct and overlapping measurement targets: addressing the
request and grounding claims need not track one another, while citation
precision and faithfulness both assess evidential support. Low rank correlation
alone does not establish statistical independence or orthogonality, and not all
RAGree measures are weakly associated.

The contrast between aggregate and per-topic Arena agreement may reflect
averaging of topic-specific differences. The box-plot gap between three
preference scores and twelve heterogeneous outputs may reflect their mixture
of measurement targets. These are plausible interpretations of the observed
patterns, not causal tests or validation of the measured constructs. Emitted
score/missing-support conventions and the absence of human validation retain
their existing qualifications. No data, plots, judge settings, or scores change.

## Compact box plot and reproducible PDF generation (2026-11-13)

The box plot now uses a wider paper footprint, tighter outer margins, broader
unfilled boxes, and stronger median/mean/whisker marks. Seeded topic points are
spread slightly further horizontally for visibility. All 119 values per group
and the common zero-to-one vertical scale are retained, as are the original
blue/orange point styling and Linux Libertine font. The topic count moves
inside the plot instead of consuming a separate title margin.

Both plots are generated as PDF files by `figures/render_agreement.py` from
the archived CSVs. Creation/modification timestamps are omitted so two runs
with the tested fonts and libraries produce identical output bytes.
`evidence/figure-reproduction.json` preserves both hash comparisons, input and
generator hashes, the command, and tested versions; the README documents how
to regenerate the assets. The heatmap's before/after rendering is pixel-identical
despite its metadata cleanup. No underlying score analyses or manuscript prose
are changed.

## Box-plot outlier annotations (2026-11-13)

All five observations beyond the existing 1.5-IQR whiskers are labeled by
topic number, with leader lines to the plotted hollow flier markers. Arena's
outliers are topics 10 and 77; the twelve-output outliers are topics 36, 84,
and 113. These identify low per-topic concordance observations, not system
producers or answer-quality errors. Only the already readable score-derived
alpha CSV and topic identifiers were inspected; no candidate content was read.

The generator selects annotations from Matplotlib's actual flier values and
checks both topic sets, all five labels, their bounds/nonoverlap, and unchanged
input hashes. Fixed annotation offsets retain deterministic PDF generation.
Both PDFs were regenerated twice with identical hashes; the heatmap remains
byte-identical. The reproduction evidence is refreshed for the annotated plot,
and the caption defines T labels using the original topic identifier format.

## Aligning outlier annotations with topic jitter (2026-11-13)

The five label leader lines and hollow outlier rings now share each topic's
actual jittered scatter coordinates, instead of using the category center.
Coordinates are selected by topic ID from the plotted scatter artist, preserving
the CSV's row order. The original outlier topic sets and vertical values are
unchanged. Generator checks compare the label anchors and ring positions with
those scatter coordinates and retain the text bounds/nonoverlap checks.

Two generator runs produced byte-identical PDFs; the heatmap remains unchanged.
The reproduction evidence records the corrected anchor coordinates and hashes.

## Inline floats and source organization (2026-11-13)

All figure TeX, including TikZ diagrams and PDF float wrappers, is now inline
in the relevant manuscript sections. Table files contain only their tabular or
tabularx environments; their float wrappers, captions, labels, and alignment
commands are inline in the sections. Expanded section source is identical to
the previous layout.

The figures directory contains only PDF assets. The reproducible generator is
now report-support/render_agreement.py, with archived inputs in docs/figures/.
README commands and current reproduction metadata use these paths; earlier
entries above record their historical locations. Both generated figure PDFs
retain exactly their previous hashes after two runs.

The fresh build remains seven pages with 25 references. Percent comments at
the two table input boundaries suppress incidental extra whitespace. All seven
rendered pages are byte-identical to the previously accepted PDF.

## Six-page revision and Table 2 interpretation (2026-11-13)

The author requested a six-page limit including references. Repeated caveats
are consolidated in the two-paragraph Discussion, preserving the distinction
between LLM proxy judgments and untested human agreement and the deliberately
extreme, unvalidated 172-call budget. Coverage, Davidson outright-win probability,
and boundary priority are expressed inline. Boundary priority retains crossing
the current top-k cutoff before smaller utility gaps at equal cost.

Section 5.3 now discusses the more-than-doubled both-bad rates alongside fewer
observed cycles and slightly more swap disagreements in the rubric runs.
Stricter judgments are a tentative interpretation, not an isolated causal
rubric effect; sparse-cycle denominators and the need for repeat runs remain
explicit. The lowest Arena topic concordance is rag2026-77 at
0.4035956554038705, reported as .404 rather than the suggested .41.
Only display scales change for the two analysis figures; their PDF assets and
underlying data are unchanged.

Reaching six pages required further concise rewrites of the introduction,
evidence descriptions, stages, related work, and diagnostic interpretation.
All six stages and 25 citation keys remain. Figure 3 uses .65 text width;
Figure 4 is a single-column float. Manual column balancing is omitted to allow
normal flow. The template, margins, and body font are unchanged. The final fresh
build is six pages including all references, with no overfull boxes or undefined
references; all six pages and enlarged formulas/figures were visually reviewed.
