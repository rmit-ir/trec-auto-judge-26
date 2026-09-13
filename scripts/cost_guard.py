#!/usr/bin/env python3
"""Track live LLM spend from a minima-llm prompt cache and enforce a hard budget.

Reads only the numeric `usage` field of each cached response (prompt/completion
token counts) -- never the response text itself, so this never surfaces judge
output. Polls the cache db for rows written since this guard started (so a
warm/shared cache dir's pre-existing spend is never counted against this run),
and if the running cost crosses --budget-usd, kills the whole process group
led by --pid -- --pid is typically an orchestrator (e.g. run_all_datasets.py)
that runs the real LLM-calling process as a subprocess, so a plain kill of
just --pid would leave that subprocess spending.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import sys
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import azure_cost  # noqa: E402
import bedrock_cost  # noqa: E402

PROVIDERS = {
    # (load_prices, estimate, supports_tier) -- Bedrock models can be invoked at
    # more than one service tier (Flex vs Standard) at different prices for the
    # same model ID; Azure deployments have no such distinction.
    "azure": (azure_cost.load_model_prices, azure_cost.estimate_cost, False),
    "bedrock": (bedrock_cost.load_model_prices, bedrock_cost.estimate_cost, True),
}


def read_usage_totals(db_path: Path) -> tuple[int, int, int]:
    """Return (call_count, total_input_tokens, total_output_tokens) for ALL rows in the cache db.

    A one-shot "how much did this cache dir bill in total" query -- for live budget
    tracking against a cache dir that may be reused across runs, use
    read_usage_since_rowid() instead so past runs' spend isn't counted against this one.
    Only touches data["usage"] -- never data["choices"] / response text.
    """
    if not db_path.exists():
        return (0, 0, 0)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
    try:
        rows = conn.execute("SELECT response_raw FROM cache").fetchall()
    finally:
        conn.close()
    return _sum_usage(raw for (raw,) in rows)


def read_max_rowid(db_path: Path) -> int:
    """Return the current max rowid, or 0 if the db doesn't exist yet (nothing written since)."""
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
    try:
        (max_rowid,) = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM cache").fetchone()
    finally:
        conn.close()
    return max_rowid


def read_usage_since_rowid(db_path: Path, since_rowid: int) -> tuple[int, int, int, int]:
    """Return (new_max_rowid, call_count, input_tokens, output_tokens) for rows written
    after since_rowid. INSERT OR REPLACE only happens on a real (billed) API call, not a
    cache hit, and gives the replaced row a fresh rowid, so this only ever counts new spend.
    """
    if not db_path.exists():
        return (since_rowid, 0, 0, 0)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
    try:
        rows = conn.execute(
            "SELECT rowid, response_raw FROM cache WHERE rowid > ? ORDER BY rowid", (since_rowid,)
        ).fetchall()
    finally:
        conn.close()
    max_rowid = max((rowid for rowid, _ in rows), default=since_rowid)
    calls, total_in, total_out = _sum_usage(raw for _, raw in rows)
    return (max_rowid, calls, total_in, total_out)


def _sum_usage(raw_rows) -> tuple[int, int, int]:
    calls = total_in = total_out = 0
    for raw in raw_rows:
        if not raw:
            continue
        try:
            usage = json.loads(raw).get("usage", {}) or {}
        except json.JSONDecodeError:
            continue
        total_in += usage.get("prompt_tokens", 0) or 0
        total_out += usage.get("completion_tokens", 0) or 0
        calls += 1
    return (calls, total_in, total_out)


def terminate(pid: int, *, grace_seconds: float = 5.0) -> None:
    """SIGTERM then SIGKILL the whole process group led by pid.

    pid is expected to be a group leader (e.g. launched as `cmd &` from a shell, or with
    start_new_session=True) whose subprocesses share its pgid -- a plain os.kill(pid, ...)
    would leave any subprocess (like the real LLM-calling process) running.
    """
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    time.sleep(grace_seconds)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass  # already gone, or the pgid was reaped/reassigned after SIGTERM -- SIGKILL is best-effort


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=PROVIDERS, default="azure")
    parser.add_argument("--tier", default="default", help="Bedrock service tier (ignored for --provider azure)")
    parser.add_argument("--pid", required=True, type=int, help="Process group to kill if budget is exceeded")
    parser.add_argument("--budget-usd", required=True, type=Decimal)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--log", type=Path, default=None, help="Status log path (default: stdout)")
    args = parser.parse_args()

    load_prices, estimate_raw, supports_tier = PROVIDERS[args.provider]
    price_key = bedrock_cost._price_key(args.model, args.tier) if supports_tier else args.model
    if price_key not in load_prices():
        print(f"error: unknown model {args.model!r} (tier {args.tier!r}) for provider {args.provider!r}", file=sys.stderr)
        return 2
    estimate = (
        (lambda model, i, o: estimate_raw(model, i, o, tier=args.tier))
        if supports_tier else estimate_raw
    )
    db_path = args.cache_dir / "minima_llm.db"
    log = args.log.open("a") if args.log else sys.stdout

    def emit(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=log, flush=True)

    last_rowid = read_max_rowid(db_path)  # ignore any spend already in the cache before we started
    total_calls = total_in = total_out = 0
    consecutive_errors = 0
    polls_without_db = 0
    emit(f"cost_guard watching pid={args.pid} db={db_path} budget=${args.budget_usd} baseline_rowid={last_rowid}")

    while True:
        if not db_path.exists():
            # A missing db forever (bad --cache-dir, or CACHE_DIR unset on the watched run)
            # makes every poll read as $0.00 -- a budget guard silently doing nothing is the
            # one failure mode worse than crashing, so give up loudly instead of no-op'ing.
            polls_without_db += 1
            if polls_without_db >= 5:
                emit(f"no cache db appeared at {db_path} after {polls_without_db} polls -- "
                     "guard cannot see any spend, giving up (not killing; check --cache-dir / CACHE_DIR)")
                return 4
        else:
            polls_without_db = 0

        try:
            last_rowid, d_calls, d_in, d_out = read_usage_since_rowid(db_path, last_rowid)
            consecutive_errors = 0
        except sqlite3.Error as e:
            consecutive_errors += 1
            emit(f"sqlite read error ({consecutive_errors}/5): {e}")
            if consecutive_errors >= 5:
                emit("too many consecutive read errors -- guard is giving up (not killing; this is a guard failure, not a budget breach)")
                return 3
            time.sleep(args.poll_seconds)
            continue

        total_calls += d_calls
        total_in += d_in
        total_out += d_out
        cost = estimate(args.model, total_in, total_out)
        emit(f"calls={total_calls} in={total_in:,} out={total_out:,} cost=${cost:.2f}")

        if cost >= args.budget_usd:
            emit(f"BUDGET EXCEEDED: ${cost:.2f} >= ${args.budget_usd} -- killing process group of pid {args.pid}")
            terminate(args.pid)
            emit("killed. exiting.")
            return 1

        try:
            os.kill(args.pid, 0)  # raises if the process is gone
        except ProcessLookupError:
            emit(f"pid {args.pid} no longer running, final cost ${cost:.2f}. exiting.")
            return 0
        except PermissionError:
            pass

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
