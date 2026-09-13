"""Smoke test for scripts/analyze_researchrubrics.py against a tiny synthetic dev-file fixture
(no dependency on the real organizer file, which is public data not committed to this repo)."""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent


def _load_analyze_module():
    spec = importlib.util.spec_from_file_location("analyze_researchrubrics",
                                                    REPO / "scripts" / "analyze_researchrubrics.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO))
    spec.loader.exec_module(module)
    return module


analyze = _load_analyze_module()

FIXTURE_ROWS = [
    {"qid": "q1", "domain": "STEM", "conceptual_breadth": "Simple", "logical_nesting": "Shallow",
     "exploration": "Low", "rubrics": [
         {"criterion": "The response defines the term (e.g., photosynthesis).", "weight": 5.0,
          "axis": "Explicit Criteria"},
         {"criterion": "The response omits an unrelated tangent.", "weight": -2.0,
          "axis": "Implicit Criteria"},
     ]},
    {"qid": "q2", "domain": "Current Events", "conceptual_breadth": "High",
     "logical_nesting": "Deep", "exploration": "High", "rubrics": [
         {"criterion": "The response compares two positions.", "weight": 3.0,
          "axis": "Synthesis of Information"},
     ]},
]


def test_analyze_writes_json_and_markdown(tmp_path):
    input_file = tmp_path / "dev.jsonl"
    with open(input_file, "w") as f:
        for row in FIXTURE_ROWS:
            f.write(json.dumps(row) + "\n")
    out_dir = tmp_path / "out"

    # main() parses sys.argv; patch it rather than passing args directly.
    old_argv = sys.argv
    try:
        sys.argv = ["analyze_researchrubrics.py", "--input", str(input_file), "--out-dir", str(out_dir)]
        rc = analyze.main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    stats = json.loads((out_dir / "dev-rubric-stats.json").read_text())
    assert stats["n_topics"] == 2
    assert stats["n_criteria_total"] == 3
    md = (out_dir / "dev-rubric-stats.md").read_text()
    assert "ResearchRubrics organizer dev-file statistics" in md


def test_analyze_missing_input_reports_error(tmp_path, capsys):
    old_argv = sys.argv
    try:
        sys.argv = ["analyze_researchrubrics.py", "--input", str(tmp_path / "nope.jsonl"),
                    "--out-dir", str(tmp_path / "out")]
        rc = analyze.main()
    finally:
        sys.argv = old_argv
    assert rc == 1
    assert "not found" in capsys.readouterr().err
