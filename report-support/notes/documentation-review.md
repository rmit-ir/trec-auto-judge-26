# Documentation and cost-ledger review

Reviewed 2026-11-11 for the pre-conference AutoJudge participation report.

## Synchronization

The local implementation checkout `../auto-judge-starter-kit` was clean and
five commits behind Sunfyre. Fetched origin and fast-forwarded it from `5435458`
to `ed525ce3a0b37bd5da291b5c05905e235338a35c`. Sunfyre was already at that
commit; fetched origin and confirmed no further changes. Both implementation
checkouts are clean and match `origin/main`. No reset, stash, history rewrite,
judge run, paid model call, data download, or remote publication was performed.
The participation-report working files remain here; this sync refers to the
implementation repository, not an Overleaf publication.

## Evidence recovered

| Record | Finding | Interpretation |
| --- | --- | --- |
| `docs/cost-estimation.md`, Measurements and Judge multiplier | RAGree: 209,646 calls; 128,904,481 input tokens; 23,772,804 output tokens; $54.31 | Documented completed RAG26 execution, split $18.95 Mac and $35.36 Sunfyre resume. Historical rate calculation gives $54.308261. Not an invoice. |
| `judges/rag26_arena/README.md`, Submission status | Approximately $90 for Arena submission runs including reruns; V2 approximately $11 | V2 is already included; do not add $11 again. Scope differs from the earlier RAGree run. |
| `evaluation/rubric-model-bakeoff/usage.json` | 15 itemized records sum to $0.1616000765 | Its summary total of $0.6216000765 includes a $0.46 prior-spend reserve. |
| `evaluation/rubric-model-bakeoff-probe/usage.json` | 9 itemized records sum to $0.0629127623 | Summary total $0.4542410013 includes an implied $0.391328239 reserve. The code explicitly supports reserves for interrupted calls. |
| `evaluation/luna56-vs-luna6/judgments/*.usage.json` | Two judge usage records: 4,630/1,681 and 4,990/2,154 input/output tokens | Partial comparison accounting; not a complete model-selection invoice. |
| Sunfyre `rubric-work*/rubric_llm_cache.db` | Five construction cache databases found, including an empty smoke cache | Successful retained calls, not a durable complete spend ledger; records contain numeric usage but no recoverable model/tier attribution in the stored usage metadata. |
| Sunfyre `cache/minima_llm.db` | 375,403 retained runtime entries, dated September 12 and October 13–14 UTC | Shared across executions; timestamp grouping cannot isolate run/model spend, and replaced entries may be lost. No per-judge dollar total inferred. |

The author confirmed these usage ledgers and caches are the intended ledger
sources. No additional project ledger filename was identified. Numeric evidence
is preserved in `evidence/cost-ledger-review.json`.

### Why construction cost is still only partially recovered

The non-rehearsal construction cache copies contain 2,473 rows across three
nonempty directories, representing 2,351 unique keys. The 122 duplicate-key
entries have different token counts, consistent with copied caches and later
re-execution; deduplication does not reconstruct all paid attempts. Choosing
the later lexicographic directory's entry yields 6,190,391 input and 9,538,726
output tokens. If all are priced at the historical GPT-6 Luna rates, that
retained snapshot is equivalent to $5.3884021. This is **conditional price-equivalent
arithmetic, not actual spend**. Neither that number nor the raw per-directory
sum belongs in a participation grand total. Validation failures can be billed
before an entry is accepted into the cache, and replacement can erase prior
usage. No provider invoices or complete per-execution guard logs were found.

`CostTracker` records successful transport calls before parsing, including billed
parse failures, but keeps records in memory. `rubrics/build.py` prints the summary
rather than writing an append-only usage ledger. `cost_guard.py` isolates new
runtime cache writes using a starting row ID; its logs are required to reconstruct
an execution's scope from a shared cache. No surviving project guard-log file
was located in the searched repo or temporary paths.

## Conflicts in agent-written notes

| Document(s) | Stale/proposed claim | Reconciled interpretation |
| --- | --- | --- |
| `cost-estimation.md`, final submission-status table | RAGTIME data runs planned; code uploads not yet made | Historical September state. Author confirms exactly four RAG26-only data uploads and two code submissions (RAGree and Choice); screenshot confirms the data uploads. |
| `rag26-code-submission-handoff.md` | Submit three Arena code variants next | Instructions for proposed work, not an execution receipt. Do not describe these as the actual code submissions. |
| `rag26-tira-method-descriptions.md`, first Arena description | Every ballot re-asked with swapped order | Executable config and implementation cap selected reversals at two per topic. Later descriptions say key ballots. |
| `rag26-rubric-construction-execution-plan.md`, older body; `rag26-rubric-next-stage-handoff.md` | Qwen3 235B primary constructor; spend about $0.0024; pipeline unfinished | Early milestone. October header, frozen artifacts, build defaults, and latest README supersede it: GPT-6 Luna production construction and complete 119-topic artifacts. |
| `rag26-tira-submission-plan.md`, early current-state section | Arena, support, generation still missing | Historical implementation plan. Later code, artifacts, and README establish what was completed. |
| `ragree-judge-design.md` | Heterogeneous jury, trajectories, training, authoritative-corpus checks | Research-program proposal. Its final implementation-status section explicitly narrows the shipped Evidence Judge. Choice adds its own preference machinery; no independent jury is established. |
| `trec-rag-2026-judge-implementation-plan.md`, rubric scoring phases | Independent pointwise rubric scoring planned | Not implemented in the submitted rubric-conditioned Arena data runs. Rubric guidance affects pairwise prompts only. |
| `ragtime-2026-judge-implementation-plan.md` | RAGTIME specialist and evaluation program | Proposal, not evidence of a RAGTIME submission. |
| `evaluation-2025-plan.md` | Extensive baselines, ablations, uncertainty estimates | Plan; `evaluation/README.md` records a harness/smoke implementation blocked on truth. Do not infer completed validation. |
| `rag26-tira-dryrun-handoff.md`, earlier unverified statements | Dry-run not done | The appended dated Result section records both Arena dry-runs passing; later result takes precedence. |

## Reproducibility finding

No differences were found in `judges/ragree/ragree_judge.py` between its saved
base `bd3083985b985cda380d21c8ece90d0e75508b46` and synchronized HEAD. Likewise,
Arena `judge.py`, `prompts.py`, `support.py`, and SPARROW are unchanged from
`84b2fb06a0e1a9ebbd1d6f4bccd9195550a8cd7c` to synchronized HEAD. The saved configs
and frozen rubrics identify the methodological settings and artifacts.

V2's workflow stanza was committed in `45ba179`, following the config's recorded
base `c912570`; its values agree with the saved V2 config. Thus implementation
source is available in Git. However, all four saved run configs mark the tree
dirty and no saved execution patch was found. Later Git history cannot prove
there were no temporary source edits at the moment of execution. The gap is
an exact execution snapshot, not an absence of documented implementation.

## Document inventory and disposition

The review inventoried repository documentation and checked execution, provenance,
model-choice, cost, and submission claims against code, Git history, persisted
metadata, ledgers, and the author's corrections. Long design documents are
research/implementation plans; their proposed experiments are not execution
records. Example and dataset READMEs describe starter-kit contracts, not team
submissions. External references in plans were not independently re-reviewed
as a new literature survey.

| Documentation | Role in this review |
| --- | --- |
| Root `README.md`, `AGENTS.md`, `CLAUDE.md` | Scope, source links, input/cache conventions, mandatory data-handling boundary |
| `docs/bedrock.md` | Historical backend and pricing configuration, not spend evidence |
| `docs/cost-estimation.md` | Completed RAGree accounting; historical submission-status table qualified |
| `docs/evaluation-2025-plan.md` | Proposed validation program and component-specific evaluation boundaries |
| `docs/rag26-code-submission-handoff.md` | Proposed container uploads; superseded final submission status |
| `docs/rag26-rubric-construction-execution-plan.md` | Pipeline stages, scope, historical model/cost plan, October correction |
| `docs/rag26-rubric-next-stage-handoff.md` | Early Stage A–C checkpoint and D-stage requirements; not final spend |
| `docs/rag26-sonnet-handoff.md` | Ordered implementation instructions; not completion evidence |
| `docs/rag26-tira-dryrun-handoff.md` | Appended dated successful container/cache-replay record |
| `docs/rag26-tira-method-descriptions.md` | Four data-run identities and earlier method summaries; reversal wording corrected |
| `docs/rag26-tira-submission-plan.md` | Historical milestones, public semantics, comparison budgets, appended dry-run status |
| `docs/ragree-choice.md` | Implemented gate, sparse acquisition, availability and reversal semantics |
| `docs/ragree-choice-report.typ` | Existing method write-up and two-topic synthetic smoke-test results; not real-data accuracy |
| `docs/ragree-judge-design.md` | Broader proposal plus explicit implemented/unimplemented boundary |
| `docs/ragtime-2026-judge-implementation-plan.md` | Unsubmitted specialist proposal; keeps data scope honest |
| `docs/trec-rag-2026-judge-implementation-plan.md` | Arena/support/rubric architecture; plan checked against actual implementation |
| `docs/trec-rag-2026-researchrubrics-generation-plan.md` | Schema, methodological rationale, planned calibration/provenance; no assumed completion |
| `judges/ragree/README.md`, `judges/ragree_choice/README.md`, `judges/rag26_arena/README.md` | Method-specific implementation and operational records |
| `judges/complete_example/README.md`, `judges/naive/README.md`, `judges/tinyjudge/README.md` | Template/synthetic checks; no participant quality evidence |
| `evaluation/README.md`, `evaluation/researchrubrics/dev-rubric-stats.md` | Actual harness boundary and structural dev-file measurements |
| `data/README.md`, `data/*/README.md` | Dataset registration/admin contracts and synthetic example caveats |

## Changes to the report

Added the RAGree run's documented call/token counts, $54.31 estimate, and restart
scope. Clarified the Arena cost scope and why bake-off ledger totals cannot be
summed. Added Choice's configured evidence thresholds and all-rejected condition.
Retained the correct two-code/four-RAG26-data submission record. Official results
remain post-conference work, as confirmed by the author.

## Documentation fingerprints

Fingerprints identify the synchronized documentation reviewed, not an exact
snapshot of any past judge execution. Only source documents are hashed.

| Path | SHA-256 |
| --- | --- |
| `AGENTS.md` | `9cffee7bd9f02742be919dd2a4865dabbca2e4f211ca0f6085e2db34b31fb29a` |
| `CLAUDE.md` | `1fd196c50670eaf886681d1dd69ed7bb3e94f80f450e138303263b4d707eccd1` |
| `README.md` | `34ade0f5ed012720c76be9d91d812a04bb77cb3a09259d1352e57e660a858a86` |
| `data/README.md` | `7a6f1a1eb3aee21f01521c6fadb3717c003df32134d54bfba582927bd9a44d48` |
| `data/dragun-repgen/README.md` | `d83013037a2b369775939f5c38265a6f2547856127abd133b804a258750a662b` |
| `data/kiddie/README.md` | `6620219248524e740d0407fdeb2a7c82f0e77a8bc9e1e40773f48983198e5694` |
| `data/longeval26-rag/README.md` | `60c86e283235968796b1cd3fd0a7d9ab8890d91621062fc770c053f89bae72f1` |
| `data/rag24/README.md` | `4e3ab93b248720eff9f893d860811e332935b53b97279106c6e58ec90ca53c82` |
| `data/rag25-auggen/README.md` | `45093a9764e609f73780d2ee9ae739334e07c734c0c2f30f4968969e74350a55` |
| `data/rag25-gen/README.md` | `cc8bfd5f000c4e1771efd8697521bb1491fe2cef8f67949c503a2949adc6b6ab` |
| `data/rag26/README.md` | `d9a2e5f2a3a686eb88787aed20b9d0a02eabc6d8473ba13118a6ce86e91c643d` |
| `data/ragtime25/README.md` | `ab76319dca18c62315b08a28a7c08bd9548e8f7961f2952e44da424bf1046996` |
| `data/ragtime26/README.md` | `f0faf46be0f7c74a92e45c12f5e3fa2a4fb56deeb07420a4bb5160ae714c551b` |
| `data/tira-evaluator/README.md` | `ff3c672a3641a26fbea8383167e8db45a199a9cfb531e8a3437c2a1924421cd3` |
| `docs/bedrock.md` | `ac65d3f81d5dc200029885b403ea6790ea842e858e3447632522efd36d18ae20` |
| `docs/cost-estimation.md` | `1f286d4fc98cfdd724548272498ccb376ddd28a66ad9d13450b89779b5f01838` |
| `docs/evaluation-2025-plan.md` | `1f3b5aa6d259f54fab525f2aaf1f5d65ba7bc58ded3188c5ff58c6e60401e8fe` |
| `docs/rag26-code-submission-handoff.md` | `625e1fb217ba292d8f4af236dc05349cce6fb7ff5f2ab081dc17dc829b17033e` |
| `docs/rag26-rubric-construction-execution-plan.md` | `42f40b79db920d69f54afd6e37207c45f142a418f3186f2618bb60b2d9dd17c7` |
| `docs/rag26-rubric-next-stage-handoff.md` | `a4305b51448b03510ee0f2ff9020fb8fdd822881954c2f039beb747e3fccf93e` |
| `docs/rag26-sonnet-handoff.md` | `a16b01b0e23cd57036614c3af2b70c521967f1aec9f2c3335f9c7cf0dd186df7` |
| `docs/rag26-tira-dryrun-handoff.md` | `b53dd39dc5194e2422e521ff6cd5986d173bdc0db35df092b032864a8dd71ebe` |
| `docs/rag26-tira-method-descriptions.md` | `cc7189b543549a830095f5dfb2d8cb942696ef3a99d67d3d296eecc4f567a30b` |
| `docs/rag26-tira-submission-plan.md` | `add15f504dc315e4b111807c685a87685d63677e9caef3516469ac10f8afa70c` |
| `docs/ragree-choice-report.typ` | `5591d68835819e60aeb6271f146c3a387729a6db91597081125e9c061569e79a` |
| `docs/ragree-choice.md` | `b011b66725fe8a359da36e90f24b7fde36458046974359feb69134c17dd36eed` |
| `docs/ragree-judge-design.md` | `a23dc8b6d580d16fe186aeb1bbd0816b44a6955fcdfc471cffbd36606134e1f4` |
| `docs/ragtime-2026-judge-implementation-plan.md` | `54706c0878ba88ddfba09f5756094d0317717aa6feb91d024d57a02128b1abed` |
| `docs/trec-rag-2026-judge-implementation-plan.md` | `8962968c24e94093ae09338ba7918d6c4b939f33be0534f97cc142b40b9751fd` |
| `docs/trec-rag-2026-researchrubrics-generation-plan.md` | `d11a63d66dd47b4b42d857b373d899b7d1a1211e9400875d18f20a5c3a8ec466` |
| `evaluation/README.md` | `c4d55107044dd93cfb02cfafeb9d067b38d57f25bd3989e139a9b64907917dc2` |
| `evaluation/researchrubrics/dev-rubric-stats.md` | `ec1933c713fa9f8d979240ca3401afbd2a30911242d5ad852a41a847128b07fd` |
| `judges/complete_example/README.md` | `5124aa65f5521f0789528a773fb2738e821ea08d71f47a64271e12ad2ec46598` |
| `judges/naive/README.md` | `025ac108e8b870556dc2787f5c5025855664c7ece700562ed688e333e51f9409` |
| `judges/rag26_arena/README.md` | `6a5f12555300a6fcc8e2c4ca606e3272b4816be27d61e8bea9f84ed437a603b9` |
| `judges/ragree/README.md` | `94d3350d554002e4489a03b1a912013fbfec3c94fb2f675db07ad03d12a1977e` |
| `judges/ragree_choice/README.md` | `dc3930f7bd7936e9b3431c41f88de09e760f81cdeba3213bb7832d18b4a61247` |
| `judges/tinyjudge/README.md` | `256c012a9a451ab30bc984f9b7951af38fc8d5a1bdf94de2551fc468c881aeeb` |

## Cost-text revision (2026-11-12)

At the author's request, the manuscript now includes only the RAGree token-backed
$54.31 estimate and the itemized comparison/probe ledger sums ($0.16 and $0.06,
rounded separately). Removed the approximate $90 Arena and $11 V2 figures from
the manuscript; they remain historical notes above, not independently
reconstructed per-execution costs.

A further numeric-only query of the three non-rehearsal construction caches
found 2,473 distinct (request key, input tokens, output tokens) records. The
122 repeated request keys have different token counts, so these are different
observed completions, not byte-identical copied usage records. Summing all
observed numeric records gives 6,254,717 input and 9,783,729 output tokens.
At the recorded GPT-6 Luna rates this is $5.5173362. Keeping only one observed
record per request key gives $5.3761191–$5.3983661, depending on which completion
is retained. These are sensitivity scenarios, not a billing confidence interval.

A useful rough estimate for **retained successful rubric-construction calls**
is therefore **$5–6**. It assumes the documented production model's rates and
excludes failed/absent/overwritten attempts, Qwen rehearsal, model-selection
exercises, infrastructure, and runtime judging. There is no verified upper
bound on all construction spending. The estimate is recorded here and in
`evidence/cost-ledger-review.json`, not asserted as verified cost in the paper.

The author subsequently requested inclusion of the US$5–6 estimate in the
report. It is now included as a rough construction estimate with its rate
assumption and exclusions, rather than presented as verified billing.
