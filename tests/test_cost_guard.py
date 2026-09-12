import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.cost_guard import read_max_rowid, read_usage_since_rowid, read_usage_totals, terminate


def _make_db(db_path, rows):
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE cache (key TEXT PRIMARY KEY, response_raw TEXT)")
    for key, usage in rows:
        conn.execute(
            "INSERT INTO cache VALUES (?, ?)",
            (key, json.dumps({"usage": usage})),
        )
        conn.commit()  # separate commits so each row gets a distinct, ordered rowid
    conn.close()


def test_read_usage_totals_sums_across_rows(tmp_path):
    db_path = tmp_path / "minima_llm.db"
    _make_db(
        db_path,
        [
            ("a", {"prompt_tokens": 100, "completion_tokens": 20}),
            ("b", {"prompt_tokens": 50, "completion_tokens": 10}),
        ],
    )
    calls, total_in, total_out = read_usage_totals(db_path)
    assert calls == 2
    assert total_in == 150
    assert total_out == 30


def test_read_usage_totals_missing_db_is_zero(tmp_path):
    assert read_usage_totals(tmp_path / "missing.db") == (0, 0, 0)


def test_read_usage_since_rowid_excludes_preexisting_rows(tmp_path):
    """A guard started against a warm/shared cache must not bill prior runs' spend."""
    db_path = tmp_path / "minima_llm.db"
    _make_db(db_path, [("old", {"prompt_tokens": 1000, "completion_tokens": 1000})])

    baseline = read_max_rowid(db_path)

    _make_db_append(db_path, [("new", {"prompt_tokens": 5, "completion_tokens": 7})])

    new_rowid, calls, total_in, total_out = read_usage_since_rowid(db_path, baseline)
    assert calls == 1
    assert total_in == 5
    assert total_out == 7
    assert new_rowid > baseline


def _make_db_append(db_path, rows):
    conn = sqlite3.connect(db_path)
    for key, usage in rows:
        conn.execute(
            "INSERT INTO cache VALUES (?, ?)",
            (key, json.dumps({"usage": usage})),
        )
        conn.commit()
    conn.close()


def test_terminate_kills_process_group(tmp_path):
    """terminate() must reach a subprocess of the target pid, not just the pid itself --
    that's the whole point (run_all_datasets.py runs the real LLM caller as a subprocess).
    The leader is its own session/group leader (start_new_session=True); its plain
    (non-session-leader) grandchild inherits that pgid, matching how run_all_datasets.py's
    `subprocess.run(cmd)` child is launched.
    """
    pidfile = tmp_path / "grandchild.pid"
    leader_code = (
        "import subprocess, sys, time;"
        f"c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']);"
        f"open({str(pidfile)!r}, 'w').write(str(c.pid));"
        "time.sleep(30)"
    )
    leader = subprocess.Popen([sys.executable, "-c", leader_code], start_new_session=True)
    try:
        for _ in range(50):  # wait for the grandchild to start and report its pid
            if pidfile.exists() and pidfile.read_text():
                break
            time.sleep(0.1)
        grandchild_pid = int(pidfile.read_text())

        terminate(leader.pid, grace_seconds=0.2)
        leader.wait(timeout=2)

        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)
    finally:
        if leader.poll() is None:
            leader.kill()


def test_main_gives_up_if_cache_db_never_appears(tmp_path):
    """A guard that never sees the db must fail loudly, not report $0.00 forever."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.cost_guard",
            "--cache-dir",
            str(tmp_path / "never-created"),
            "--model",
            "gpt-5.6-luna",
            "--pid",
            str(os.getpid()),  # any live pid; the db-missing path returns before it matters
            "--budget-usd",
            "100",
            "--poll-seconds",
            "0.05",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 4
    assert "no cache db appeared" in result.stdout  # emit() defaults to stdout when --log is unset
