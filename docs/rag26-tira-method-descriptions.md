# TIRA method descriptions (one per code submission)

All three variants share one judge, `RAG26ArenaJudge` (`judges/rag26_arena/`), running
`gpt-5.6-luna` through the injected OpenAI-compatible endpoint with a deterministic prompt cache.

## arena-support

Pairwise "arena" judge. For each topic, pairs of responses are shown side by side (answer text only) and the
LLM votes A, B, tie, both-bad or abstain. Pairs are chosen by a sparse seed graph plus adaptive
acquisition; every ballot is also re-asked with the answer order reversed to measure position bias.
Votes are fitted with a Davidson-Bradley-Terry model (SPARROW) to give each run a topic-level
preference score (`RAG26_ARENA_SCORE`), with Condorcet/Kemeny/cycle audits reported as diagnostics.
An independent, RAGDoll-compatible citation-support lane checks whether each cited document
supports the sentence citing it and reports first/all-citation weighted and hard precision/recall.
Preference and support stay separate measures. No rubric is used.

## rubric-support-arena

Same arena and citation-support lanes, but each pairwise prompt also shows a compact,
response-independent rubric for the topic (desirable "+" and undesirable "-" criteria, strongest
first, no weights shown). The rubrics were generated once, offline, and frozen in the repository
(V1): GPT-6 Luna (high reasoning) planned each request, retrieved evidence from the ClimbMix corpus
(dense + sparse hybrid, RRF), generated candidate criteria per axis (explicit, implicit, synthesis,
negative, references, communication, instruction following), then critiqued, weighted and selected
22-31 criteria per topic. No candidate response, score or leaderboard information entered rubric
construction.

## rubric-v2-support-arena

As rubric-support-arena, with the V2 frozen rubrics. V2 re-runs the same pipeline with tightened
prompts so that retrieved studies, organizations and cases are not turned into hidden mandatory
answer outlines (less evidence-driven over-specification); everything else is identical.

---

# Earlier judges

## ragree (RAGree Evidence Judge)

Pointwise evidence-grounding judge. Each cited sentence is checked by an LLM against its cited
document (supported, partially supported, contradicted, not enough evidence), and coverage is
estimated against nugget questions generated from the topic alone. Reports faithfulness,
estimated coverage, citation precision and coverage, and critical-error rate.

## ragree_choice (RAGree Choice / SPARROW)

Preference judge. Responses with a critical evidence error are vetoed, then the rest are compared
with sparse pairwise LLM ballots (a seed graph plus adaptively chosen top-k boundary pairs) and
ranked with a Davidson-Bradley-Terry fit. Key ballots are re-asked with the answer order swapped
to remove position bias, and Condorcet, cycle and Kemeny audits check the ranking.
