# Handoff: RAG26 Arena code submissions to TIRA (run on the Mac with Docker)

**Why you, not the other session:** the judge, the data submissions and the frozen rubrics
are finished on the shared Linux host, which has no Docker/Podman. The remaining step is
`tira-cli code-submission`, which builds and uploads a container, and it must run on a machine
with a container runtime. Your job: do the dry-runs, then the real code submissions for the
variants below, report back, and stop.

## Read first, in full

1. `AGENTS.md`, `CLAUDE.md` (data policy is mandatory and unconditional)
2. `.claude/skills/autojudge-submit/SKILL.md` -- drive the work through it, step by step
3. `docs/rag26-tira-dryrun-handoff.md` -- the earlier successful dry-run on this Mac (same flow)
4. `judges/rag26_arena/README.md` -- "Submission status" section

You only touch the `kiddie` dataset locally (`--dataset kiddie-20260605-training`, public,
synthetic). The real submissions run on TIRA's side; never download, open or print RAG26
reports, and never print the contents of `.env` or any key value.

## State

- Repo `rmit-ir/trec-auto-judge-26`, branch `main`, at or after `d755293`; the tree must be
  clean (`git status --porcelain` empty) and `pytest` green (288 passed, 2 xfailed at that SHA).
- Data submissions are already uploaded to TIRA dataset `rag26-20260827_1-test` as three
  separate systems: `rag26_arena-arena-support`, `rag26_arena-rubric-support-arena` (V1
  rubrics), `rag26_arena-rubric-v2-support-arena` (V2 rubrics). Do not re-upload data.
- Workflow `judges/rag26_arena/workflow.yml`, judge `RAG26ArenaJudge`. The frozen rubric files
  under `judges/rag26_arena/rubrics/frozen/` ship in the image via `ADD judges`.
- Deadline was extended to roughly 2026-10-14 (+24h); check the TIRA task page for the exact time.

## Variants to submit (one code submission each)

| Variant | Rubrics | Notes |
|---|---|---|
| `rubric-v2-support-arena` | V2 | newest; recommended first |
| `rubric-support-arena` | V1 | |
| `arena-support` | none | baseline |

If time is short, submit in that order and stop when the deadline is near. Ask the user if
unsure which to prioritise.

## Setup

```bash
git pull origin main && git status --porcelain      # must print nothing
uv venv && source .venv/bin/activate
uv pip install -e '.[all]' && uv pip install --upgrade tira
docker version                                       # must work
source .env                                          # never cat it; OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
env | grep -c '^OPENAI_'                             # count only, never print values
pytest
tira-cli login --token "$TIRA_TOKEN"                 # if not already logged in
```

`OPENAI_MODEL` must be `gpt-5.6-luna`. The backup `gpt-6-luna` (ASE vars) is NOT used in
submissions and is not forwarded into the container.

## Dry run (per variant; uploads nothing)

```bash
tira-cli code-submission --dry-run --path . \
  --cache-behaviour deterministic --mount-cache '$CACHE_DIR=EMPTY_DIR' \
  --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
  --task trec-auto-judge --dataset kiddie-20260605-training \
  --command 'auto-judge run --workflow /auto-judge/judges/rag26_arena/workflow.yml --variant <VARIANT> --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```

Add `--allow-network` if the local test needs the endpoint. Confirm the dry-run is green for
the variant before the real submission. Rubric variants fall back to the plain prompt for
qids without a rubric (kiddie), so a kiddie dry-run proves packaging, not rubric use.

## Real submission (per variant; outward-facing -- confirm with the user first)

Same command without `--dry-run` and **without `--dataset`**. Submit one variant at a time and
wait for the upload confirmation. Note: TIRA will execute the whole judge on the real test set
against the forwarded endpoint (cold cache), so each submission makes a full-size LLM run
(order of $10-30 and 1-3 hours of endpoint time per variant on the user's account). Tell the
user the expected cost before starting each one.

## Known issues

- `tira-cli verify-installation --team ...` returned a server-side HTTP 500 for every team
  spelling earlier. Do not loop on it; if the dry-run/real submission works, ignore it, and
  mention it to the user for TIRA support.
- A few citation-support requests are rejected by the endpoint's content filter. The judge
  tolerates this (they stay "unavailable", not negative; `batch.max_failures` is disabled in
  `judges/rag26_arena/judge.py`). This is expected, not a failure.
- Docker image builds copy the whole repo; keep the tree clean and do not add large files.

## Report back

For each variant: dry-run result, real-submission result (TIRA message/run name), any error
text (error type and location only; never quote restricted values), and the cost/time notice
you gave. Then stop.
