# Handoff: RAG26 Arena TIRA dry-run (this machine has Docker/Podman)

**Why you're running this, not the other session:** the RAG26 Arena judge
(`judges/rag26_arena/`) is implemented, tested, and verified end-to-end on
kiddie with deterministic cache replay -- but that work happened on a shared
RHEL8 host with no `docker`/`podman` and no root/sudo, so the TIRA
`code-submission --dry-run` gate (which builds and runs a container) could
never be exercised there. That is the one remaining item before this judge
clears `docs/rag26-tira-submission-plan.md` Phase 2's exit gate. Your job is
narrow: get the dry-run green (or diagnose exactly why it isn't), report
back, and stop -- do not start Phase 4 (intent planner) or later phases here.

## Read first, in full

In this order, before running anything:

1. `AGENTS.md`
2. `CLAUDE.md`
3. `.claude/skills/autojudge-submit/SKILL.md` -- this is the skill that
   drives everything below; walk it interactively rather than skipping to
   the command block in this doc.
4. `docs/rag26-tira-submission-plan.md` Section 6 ("Phase 2 -- prove the
   TIRA vertical slice immediately") for the exact background/context.

The AutoJudge data-handling policy in `AGENTS.md`/`CLAUDE.md` is mandatory
and unconditional. Nothing about this being a dry-run or a different machine
changes it. This task only touches the `kiddie` dataset (synthetic, public,
unrestricted) -- you should not need to touch any RAG26 topic/report data at
all to complete it.

## Current state (git SHA `2d11dc2` on `main`, already pushed)

`judges/rag26_arena/` has `judge.py`, `models.py`, `prompts.py`, `support.py`,
`workflow.yml` with three real variants: `arena-only` (default), `arena-support`
(adds the independent citation-support lane), `exhaustive-small`, `no-reversal`.
178 passed, 2 xfailed on the full suite as of this commit. Already verified
on the original machine (no container runtime there):

- `arena-only` and `arena-support` both run end-to-end on kiddie via the real
  injected endpoint (`.env` has `OPENAI_BASE_URL`/`OPENAI_MODEL` already --
  do not print `.env`'s contents; source it, don't `cat` it).
- Re-running either variant against a broken/unreachable `OPENAI_BASE_URL`
  with the same warm `$CACHE_DIR` produced a byte-identical leaderboard --
  deterministic cache-only replay confirmed twice.

What is **not** yet verified anywhere: Docker/Podman build, and
`tira-cli code-submission --dry-run`.

## Setup

```bash
git pull origin main   # confirm you're on 2d11dc2 or later
uv venv && source .venv/bin/activate
uv pip install -e '.[all]'
uv pip install --upgrade tira
```

Confirm you actually have a container runtime before going further:

```bash
docker version || podman version
```

If neither works, stop and report that back -- do not attempt a root
install or a rootless workaround without asking first; the intent here is
to use a machine that already has this, not to debug container tooling.

## Step 1 -- local verification (skill Step 1)

```bash
env | grep -i OPENAI   # confirm OPENAI_BASE_URL / OPENAI_MODEL / OPENAI_API_KEY are set (never print the key's value)
bash run_kiddie.sh
pytest
```

Both must be green before continuing.

## Step 2 -- git hygiene (skill Step 2)

`git status --porcelain` should already be clean (this handoff was written
from a clean tree right after pushing). If it isn't clean here, that's a
sync problem worth reporting before proceeding, not something to
auto-resolve by discarding anything.

## Step 3 -- tira-cli auth (skill Step 3)

```bash
tira-cli login --token <AUTH-TOKEN>
tira-cli verify-installation --task trec-auto-judge --team <TEAM>
```

You'll need the TIRA auth token and team name from the user -- ask if not
already provided. Don't guess a team name.

## Step 4 -- the dry run itself

Run **both** variants, since they exercise different code paths
(`arena-support` additionally calls the citation-support lane):

```bash
tira-cli code-submission \
    --dry-run --path . \
    --cache-behaviour deterministic --mount-cache '$CACHE_DIR=EMPTY_DIR' \
    --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
    --task trec-auto-judge --dataset kiddie-20260605-training \
    --command 'auto-judge run --workflow /auto-judge/judges/rag26_arena/workflow.yml --variant arena-only --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```

```bash
tira-cli code-submission \
    --dry-run --path . \
    --cache-behaviour deterministic --mount-cache '$CACHE_DIR=EMPTY_DIR' \
    --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
    --task trec-auto-judge --dataset kiddie-20260605-training \
    --command 'auto-judge run --workflow /auto-judge/judges/rag26_arena/workflow.yml --variant arena-support --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```

Recall from the skill: `--cache-behaviour deterministic` re-executes the
judge with the endpoint disabled against the first run's cache, so the judge
must reproduce from cache alone. If you want a faster iteration loop first,
mount your own populated cache instead of `EMPTY_DIR`
(`--mount-cache '$CACHE_DIR=cache'`, same `OPENAI_MODEL` used to build it) --
but the version you report as "the dry run passed" should be the
`EMPTY_DIR` (cold-start-proof) run, since that's what the real submission
uses.

**Do not run without `--dry-run`.** Nothing here should upload anything to
TIRA -- that is explicitly out of scope for this handoff.

## What "done" looks like

Both dry runs build the container and complete without error, and the
`--cache-behaviour deterministic` re-execution reproduces from cache with no
network calls (no hard-failing startup probe against the disabled endpoint).

## Report back

When you're done (pass or fail), report:

- Docker or Podman, and version.
- Pass/fail for each variant's dry run, with the actual command output on
  failure (stderr tail is enough, don't paste megabytes of build log).
- Any code change you had to make to get it working (there should ideally be
  none -- if there is one, explain exactly what broke and why, since that's
  a real bug in `judges/rag26_arena/` worth knowing about on the other
  machine too).
- Whether you committed anything, and the resulting git SHA.

Then stop. Do not proceed to Phase 3 hardening extras, Phase 4 (intent
planner), or any other later-phase work from `docs/rag26-tira-submission-plan.md`
-- that continues on the original machine/session.

## Result (2026-09-13, SHA `b27b3ac`)

- Docker Desktop 4.88.1, engine 29.7.2 (darwin/arm64).
- `arena-only` dry run: PASS. `arena-support` dry run: PASS. Both hit all six
  `tira-cli` checks -- dataset local, git clean, image built, `$CACHE_DIR`
  used, valid kiddie output, and cache-only re-execution reproduced with
  `OPENAI_BASE_URL=EMPTY`/`OPENAI_API_KEY=EMPTY` (no network).
- No code change needed in `judges/rag26_arena/`.
- Nothing committed by this task besides this doc update; tree stayed clean
  at `b27b3ac` throughout.
- Unrelated snag: `tira-cli verify-installation --task trec-auto-judge --team
  <TEAM>` 500-errored server-side for every team-name spelling tried
  (`RMIT-ADMS`, `rmit-adms`, `tira_vm_rmit-adms`) -- login and Docker checks
  passed fine standalone. Did not block the dry run (which needs no team
  scoping) but flag it to TIRA support before the real submission.

Phase 2 exit gate is green. Continue on the original machine/session for
Phase 3+.
