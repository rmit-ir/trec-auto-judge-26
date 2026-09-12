# Cost estimation

Two independent numbers combine into a pre-flight $ estimate for a judge run,
without needing a full paid run first:

1. **Corpus token floor** — `scripts/corpus_tokens.py` tokenizes every run's
   response text, cited sentences, and cited evidence documents for a dataset,
   using the same tokenizer as the model actually configured (`OPENAI_MODEL`).
   It's free (no API calls) and gives a lower bound on how many input tokens
   the judge will touch, before any prompt scaffolding or multiple-call
   overhead is added.
2. **Judge multiplier** — measured once from a real run via
   `scripts/cost_guard.py`'s exact token accounting (from the LLM response
   `usage` field, not the corpus): `billed_input_tokens / corpus_floor_tokens`.
   This captures everything the floor doesn't: system prompts, few-shot text,
   retries, and how many times the judge re-touches each piece of text (e.g.
   `questions_per_topic` nugget calls, or `full_round_robin_max_candidates`
   pairwise comparisons).

Once you have a multiplier for a given judge+workflow, you can estimate the
cost of changing a workflow knob (more nugget questions, more comparisons)
without another paid run: recompute how the knob changes call *count*, keep
the multiplier, and feed the result into `scripts/azure_cost.py`.

## Tokenizer selection

`scripts/corpus_tokens.py:resolve_encoding(model_name)` tries, in order:
1. `tiktoken.encoding_for_model(model_name)` — tiktoken's own model map.
2. The `tokenizer_encoding` column in `docs/azure-openai-pricing.csv`, if the
   model has a row there.
3. `o200k_base` (the encoding every modern GPT-4o/4.1/5-family/o-series model
   uses) as a labeled approximation.

The source is always recorded (`encoding_source` in the output JSON) and a
warning is printed to stderr whenever it isn't the direct tiktoken match, so
an approximation is never silent.

`gpt-5.6-luna` resolves via tiktoken's own model map to `o200k_base`
(tiktoken >= 0.14).

## Data-policy constraints on this tooling

- `corpus_tokens.py` never prints response text, document text, file paths,
  or per-file progress — only integers.
- It never computes or emits a **per-run or per-team** token count. Token/
  character length is a length-and-style fingerprint of a specific run —
  reporting one would itself be an anonymization breach. Only dataset-level
  totals and per-topic sums **combined across all runs** are computed; a
  topic-level sum spread over 50+ runs doesn't isolate any one team.
- Output JSON lands in `output/token-counts/` (gitignored) and is not
  committed; only the aggregate row below is recorded here.

## Measurements

| Dataset | Judge | Corpus floor input tokens† | Reports | Topics | Encoding | Date |
|---|---|---|---|---|---|---|
| ragtime26-repgen | ragree | 53,274,678 | 5,047 | 103 | o200k_base | 2026-09-12 |
| rag26-generation | ragree | *pending — deferred, see note* | 83 runs × 119 topics | 119 | o200k_base | — |

† `topic_tokens + report_text_tokens + cited_sentence_tokens + cited_evidence_tokens`,
`max_doc_chars=1500`. At $0.20/1M input (gpt-5.6-luna), ragtime26-repgen's
floor alone is **~$10.65** input-only — the real per-call cost (multiple
calls per sentence/topic, output tokens, prompt scaffolding) will be several
times this; see the multiplier below once a full run has completed.

*rag26-generation is ~3GB; running the tokenizer pass concurrently with a
live paid judge run on a memory-constrained machine risked another OOM kill,
so it's deferred until either finishes.*

### Judge multiplier (fill in once a run completes)

If `CACHE_DIR` is dedicated to one run, that run's totals are the whole
cache: `scripts.cost_guard.read_usage_totals(Path("<cache_dir>/minima_llm.db"))`.

**If `CACHE_DIR` is shared/reused across runs (our actual setup — one
`./cache` for all judge×dataset combos), that call sums every run's spend
together, not just one.** Get one run's own totals from its own
`cost_guard.py` invocation instead: each guard call records its own
`baseline_rowid` at startup (see its `--log` output) and only ever
accumulates rows written after that — so the last `calls=… in=… out=…` line
in a given run's guard log *is* that run's isolated total, even against a
shared cache. (Reusing `cost_guard.py`'s own `read_usage_since_rowid()`
against the baseline it logged also works, if you want it outside the log.)

```
input_multiplier    = billed_in / corpus_floor_input_tokens
avg_output_per_call = billed_out / calls
```

| Dataset | Judge | Variant | Calls | Billed in | Billed out | input_multiplier | avg_output_per_call | Date |
|---|---|---|---|---|---|---|---|---|
| rag26-generation | ragree | default | (in progress) | (in progress) | (in progress) | — | — | 2026-09-12 |

Re-estimating a workflow change: recompute expected call count for the new
config, multiply the new corpus-floor input by `input_multiplier`, multiply
new call count by `avg_output_per_call`, feed both into
`scripts/azure_cost.py <model> <input> <output>`.

## Resuming after a failure or crash

Two independent layers, with different persistence:

1. **LLM-call cache (`<CACHE_DIR>/minima_llm.db`)** — every individual LLM
   call is cached by minima-llm, keyed by the exact request. Restarting the
   same workflow against the same `CACHE_DIR` replays every already-answered
   call from cache instantly — no re-query, no re-billing (`cost_guard.py`'s
   rowid-baseline tracking depends on exactly this: a cache hit never calls
   `PromptCache.put()`, so it never advances rowid or gets billed twice).
2. **Judge output** — only partially checkpointed. Nugget generation writes
   its file (`<filebase>.nuggets.jsonl`) as soon as that phase finishes, but
   the final leaderboard/judgments file is written once, at the very end of
   the full judging loop (`autojudge_base.judge_runner.run_judge`). A crash
   mid-judging leaves no usable output file yet, regardless of how much of
   the cache is already populated.

Practical consequence: a crash and rerun does **not** redo or re-pay for any
call the cache already has — only genuinely new calls hit the API — but it
does still have to re-walk the whole judge loop from the top to reconstruct
results and reach the final write. Cheap and fast in $, not necessarily fast
in wall-clock if the loop itself (not just the LLM calls) does non-trivial
work per topic.
