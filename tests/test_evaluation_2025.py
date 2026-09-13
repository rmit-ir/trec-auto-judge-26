"""Tests for the 2025 RAGree evaluation harness (docs/evaluation-2025-plan.md).

Split into: (a) fully synthetic unit tests (no dependency on ./local-data, always run) for the
reusable pieces -- truth parsing/verification, split generation, split resolution, summary
loading; and (b) one integration test against the real 2025 manifest, skipped when
./local-data hasn't been fetched (it's gitignored and password-protected; see
fetch_datasets.py) rather than failing.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load("prepare_eval_2025", "prepare_eval_2025.py")
run_eval = _load("run_eval_2025", "run_eval_2025.py")
summarize = _load("summarize_eval_2025", "summarize_eval_2025.py")

import run_all_datasets  # noqa: E402  (importable: run_eval's load put REPO on sys.path)


# --- manifest -----------------------------------------------------------------------------

def test_manifest_has_four_datasets_with_required_keys():
    entries = prepare.load_manifest()
    assert {e["name"] for e in entries} == {
        "rag25-generation", "rag25-auggen", "ragtime25-repgen", "dragun25-repgen",
    }
    for e in entries:
        for key in ("from_release", "truth", "truth_format", "role"):
            assert key in e, f"{e['name']}: missing '{key}'"
        assert e["role"] in {"development", "validation", "frozen-cross-track-test"}


# --- topic/run id extraction (no local-data needed) ----------------------------------------

def test_topic_ids_of(tmp_path):
    topics = tmp_path / "topics.jsonl"
    topics.write_text('{"request_id": 14, "title": "x"}\n{"request_id": "abc", "title": "y"}\n')
    assert prepare.topic_ids_of(topics) == ["14", "abc"]


def test_run_ids_of(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for name in ["fox", "goat", ".hidden"]:
        (runs / name).write_text("{}\n")
    assert prepare.run_ids_of(runs) == ["fox", "goat"]


# --- truth parsing / verification (no local-data needed) ------------------------------------

def _write_truth(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for run_id, topic_id, measure, value in rows:
            f.write(f"{run_id}\t{topic_id}\t{measure}\t{value}\n")


def test_check_truth_ok(tmp_path):
    truth = tmp_path / "truth.tsv"
    _write_truth(truth, [
        ("run1", "t1", "M", 0.5), ("run1", "t2", "M", 0.6),
        ("run2", "t1", "M", 0.4), ("run2", "t2", "M", 0.7),
    ])
    measures = prepare.check_truth("ds", truth, "ir_measures", ["run1", "run2"], ["t1", "t2"])
    assert measures == {"M"}


def test_check_truth_detects_duplicate_cell(tmp_path):
    truth = tmp_path / "truth.tsv"
    _write_truth(truth, [
        ("run1", "t1", "M", 0.5), ("run1", "t1", "M", 0.9),  # duplicate (run1, t1, M) cell
    ])
    with pytest.raises(prepare.Problem, match="duplicate truth cell"):
        prepare.check_truth("ds", truth, "ir_measures", ["run1"], ["t1"])


def test_check_truth_warns_on_unknown_run_and_topic(tmp_path, capsys):
    truth = tmp_path / "truth.tsv"
    _write_truth(truth, [("ghost_run", "ghost_topic", "M", 0.5)])
    prepare.check_truth("ds", truth, "ir_measures", ["run1"], ["t1"])
    err = capsys.readouterr().err
    assert "ghost_run" in err
    assert "ghost_topic" in err


# --- deterministic splits (no local-data needed) ---------------------------------------------

def _fake_reports():
    return [
        {"name": "dev-ds", "role": "development",
         "all_runs": [f"run{i}" for i in range(10)],
         "assessed_topics": [str(i) for i in range(8)]},
        {"name": "frozen-ds", "role": "frozen-cross-track-test",
         "all_runs": [f"run{i}" for i in range(5)],
         "assessed_topics": ["a", "b", "c"]},
    ]


def test_make_splits_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "SPLITS_DIR", tmp_path / "a")
    prepare.make_splits(_fake_reports())
    monkeypatch.setattr(prepare, "SPLITS_DIR", tmp_path / "b")
    prepare.make_splits(_fake_reports())

    a = (tmp_path / "a" / "dev-ds.yml").read_text()
    b = (tmp_path / "b" / "dev-ds.yml").read_text()
    assert a == b


def test_make_splits_development_has_four_folds_covering_all_topics(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "SPLITS_DIR", tmp_path)
    prepare.make_splits(_fake_reports())
    doc = yaml.safe_load((tmp_path / "dev-ds.yml").read_text())
    folds = doc["topic_folds"]["folds"]
    assert set(folds) == {"fold-0", "fold-1", "fold-2", "fold-3"}
    assert sorted(t for f in folds.values() for t in f) == [str(i) for i in range(8)]


def test_make_splits_run_holdout_is_80_20_and_disjoint(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "SPLITS_DIR", tmp_path)
    prepare.make_splits(_fake_reports())
    doc = yaml.safe_load((tmp_path / "dev-ds.yml").read_text())
    known, held_out = doc["run_holdout"]["known"], doc["run_holdout"]["held_out"]
    assert len(known) == 8 and len(held_out) == 2
    assert set(known).isdisjoint(held_out)
    assert set(known) | set(held_out) == {f"run{i}" for i in range(10)}


def test_make_splits_frozen_dataset_has_no_folds(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "SPLITS_DIR", tmp_path)
    prepare.make_splits(_fake_reports())
    doc = yaml.safe_load((tmp_path / "frozen-ds.yml").read_text())
    assert "topic_folds" not in doc
    assert doc["assessed_topics"] == ["a", "b", "c"]


# --- run_eval_2025 split resolution / unique dirs (no local-data needed) --------------------

def test_split_topics_resolves_fold_and_assessed(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "SPLITS_DIR", tmp_path)
    (tmp_path / "ds.yml").write_text(yaml.safe_dump({
        "assessed_topics": ["1", "2", "3"],
        "topic_folds": {"folds": {"fold-0": ["1"], "fold-1": ["2", "3"]}},
    }))
    assert run_eval.split_topics("ds", "all") is None
    assert run_eval.split_topics("ds", "assessed") == ["1", "2", "3"]
    assert run_eval.split_topics("ds", "fold-1") == ["2", "3"]
    with pytest.raises(SystemExit, match="unknown --split"):
        run_eval.split_topics("ds", "fold-99")


def test_split_topics_missing_split_file_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "SPLITS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="no split file"):
        run_eval.split_topics("nonexistent-ds", "fold-0")


def test_split_run_holdout_resolves_known_and_held_out(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "SPLITS_DIR", tmp_path)
    (tmp_path / "ds.yml").write_text(yaml.safe_dump({
        "run_holdout": {"known": ["r1", "r2"], "held_out": ["r3"]},
    }))
    assert run_eval.split_run_holdout("ds", "all") is None
    assert run_eval.split_run_holdout("ds", "known") == ["r1", "r2"]
    assert run_eval.split_run_holdout("ds", "held-out") == ["r3"]


def test_split_run_holdout_missing_split_file_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "SPLITS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="no split file"):
        run_eval.split_run_holdout("nonexistent-ds", "known")


def test_unique_dir_never_collides(tmp_path):
    base = tmp_path / "experiment"
    first = run_eval.unique_dir(base)
    first.mkdir()
    second = run_eval.unique_dir(base)
    assert second != first
    assert not second.exists()


# --- summarize (no local-data needed) --------------------------------------------------------

def test_load_experiments_reads_manifests(tmp_path):
    exp_dir = tmp_path / "20260101-000000-naive-ds-fold-0"
    exp_dir.mkdir()
    (exp_dir / "manifest.yml").write_text(yaml.safe_dump({
        "experiment_id": exp_dir.name, "commit_sha": "abc123", "judge": "judges/naive/workflow.yml",
        "dataset": "ds", "split": "fold-0", "variant": None, "wall_time_seconds": 1.5, "success": True,
    }))
    rows = summarize.load_experiments(tmp_path)
    assert len(rows) == 1
    assert rows[0]["dataset"] == "ds"
    assert rows[0]["judge"] == "naive"
    assert rows[0]["success"] is True


def test_load_experiments_empty_dir_returns_no_rows(tmp_path):
    assert summarize.load_experiments(tmp_path) == []


# --- run_meta_evaluate --output wiring (no local-data needed) -------------------------------

def test_run_meta_evaluate_adds_output_flag_only_when_given(tmp_path, monkeypatch):
    dataset_out = tmp_path
    (dataset_out / "run1.eval.txt").write_text("run1\tt1\tM\t0.5\n")
    dataset = run_all_datasets.Dataset(name="ds", responses="r", topics="t",
                                        truth=str(tmp_path / "truth.jsonl"))
    seen_cmds = []
    monkeypatch.setattr(run_all_datasets.subprocess, "run",
                         lambda cmd, **kw: seen_cmds.append(cmd) or run_all_datasets.subprocess.CompletedProcess(cmd, 0))

    run_all_datasets.run_meta_evaluate(dataset, dataset_out)
    assert "--output" not in seen_cmds[0]

    seen_cmds.clear()
    run_all_datasets.run_meta_evaluate(dataset, dataset_out, output=tmp_path / "correlations.jsonl")
    assert "--output" in seen_cmds[0]
    assert str(tmp_path / "correlations.jsonl") in seen_cmds[0]
    assert "--out-format" in seen_cmds[0] and "jsonl" in seen_cmds[0]


# --- integration against the real fetched 2025 data (skipped if not fetched) ----------------

PILOT_FETCHED = (REPO / "local-data" / "rag25" / "datasets.yml").is_file()


@pytest.mark.skipif(not PILOT_FETCHED, reason="local-data/rag25 not fetched (./fetch_datasets.py --pilot)")
def test_check_dataset_reports_missing_truth_for_real_manifest():
    entries = {e["name"]: e for e in prepare.load_manifest()}
    report = prepare.check_dataset(entries["rag25-generation"])
    assert report["runs"] > 0
    assert report["topics"] > 0
    # True at the time this harness was written (2026-09-12): no pilot 2025 truth bundle has
    # been published/fetched. If this starts failing, the blocker in evaluation/README.md is
    # resolved -- update truth_available in evaluation/datasets-2025.yml too.
    assert report["truth_status"] == "missing"
