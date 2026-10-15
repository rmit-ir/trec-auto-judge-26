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
