"""Synthetic, network-free end-to-end tests of RAGreeChoiceJudge.judge().

Unlike test_ragree_choice_failures.py (which exercises individual parsing/
validation helpers in isolation), these drive the whole judge() method through
a fake LLM backend and inspect the actual built Leaderboard, so a bug in the
wiring between the pieces -- not just in one piece -- would show up here.
"""

from autojudge_base import Request
from autojudge_base.report import Report, ReportMetaData
from autojudge_base.report_sentences import NeuclirReportSentence

import judges.ragree_choice.ragree_choice_judge as choice


class _Response:
    def __init__(self, text: str):
        self.text = text


def _report(run_id: str, topic_id: str, text: str) -> Report:
    return Report(
        metadata=ReportMetaData(team_id=run_id, run_id=run_id, topic_id=topic_id),
        responses=[NeuclirReportSentence(text=text, citations=None)],
    )


def _brief_json(*, critical_error=False, factual_support=0.9, citation_validity=0.9, completeness=0.8) -> str:
    return (
        "{"
        f'"critical_error": {str(critical_error).lower()}, '
        f'"factual_support": {factual_support}, "completeness": {completeness}, '
        f'"citation_validity": {citation_validity}, "confidence": 0.7}}'
    )


_BRIEF_PHASE_MARKER = "Assess this answer pointwise"
_PAIRWISE_PHASE_MARKER = "pointwise-first, pairwise-final"


class _ScriptedBackend:
    """Routes each request to a canned response by first identifying the
    prompt's phase (evidence brief vs. pairwise ballot -- request_id restarts
    at 0 for every batch and is not a reliable key), then matching a marker
    substring within that phase's table. Phase-scoping matters because the
    pairwise prompt echoes each answer's text too, so an unscoped "Answer A"
    marker could otherwise match a pairwise prompt and return brief-shaped
    JSON to a vote request.
    """

    def __init__(self, brief_script: dict[str, str], pairwise_script: dict[str, str] | None = None):
        self.brief_script = brief_script
        self.pairwise_script = pairwise_script or {}
        self.seen_content: list[str] = []

    async def run_batched(self, requests):
        results = []
        for request in requests:
            content = request.messages[-1]["content"]
            self.seen_content.append(content)
            if _BRIEF_PHASE_MARKER in content:
                table = self.brief_script
            elif _PAIRWISE_PHASE_MARKER in content:
                table = self.pairwise_script
            else:
                raise AssertionError(f"unrecognized prompt phase: {content[:200]!r}")
            for marker, response_text in table.items():
                if marker in content:
                    # None simulates a transport/parse failure for just this one
                    # scripted entry (a plain object() is not a MinimaLlmResponse).
                    results.append(_Response(response_text) if response_text is not None else object())
                    break
            else:
                raise AssertionError(f"no scripted response for prompt: {content[:300]!r}")
        return results


def _run_judge(monkeypatch, reports, topics, brief_script, pairwise_script=None, **judge_settings):
    monkeypatch.setattr(choice, "MinimaLlmResponse", _Response)
    backend = _ScriptedBackend(brief_script, pairwise_script)
    monkeypatch.setattr(choice, "_backend", lambda _config: backend)
    judge = choice.RAGreeChoiceJudge()
    settings = {
        "adaptive_comparisons_per_topic": 0,  # keep the call graph deterministic
        "position_reversal_budget": 0,  # ditto -- covered separately below
        "llm_max_attempts": 2,
        "llm_retry_backoff_seconds": 0.0,
        **judge_settings,
    }
    leaderboard = judge.judge(reports, topics, llm_config=object(), **settings)
    return leaderboard, backend


def _row(leaderboard, run_id: str, topic_id: str) -> dict[str, float]:
    for entry in leaderboard.entries:
        if entry.run_id == run_id and entry.topic_id == topic_id:
            return dict(entry.values)
    raise AssertionError(f"no leaderboard row for run={run_id!r} topic={topic_id!r}")


def test_mixed_unavailable_ineligible_eligible_topic(monkeypatch):
    topics = [Request(request_id="t1", title="What is X?")]
    reports = [
        _report("run-unavailable", "t1", "This run's evidence brief never parses."),
        _report("run-ineligible", "t1", "This run is evaluated but too weak."),
        _report("run-a", "t1", "Answer A, evidence-backed."),
        _report("run-b", "t1", "Answer B, evidence-backed."),
    ]
    brief_script = {
        "never parses": "not json, every attempt",
        "too weak": _brief_json(factual_support=0.1, citation_validity=0.1),
        "Answer A": _brief_json(),
        "Answer B": _brief_json(),
    }
    pairwise_script = {"": '{"vote": "A"}'}  # only one pair is ever cast here
    leaderboard, backend = _run_judge(monkeypatch, reports, topics, brief_script, pairwise_script)

    unavailable = _row(leaderboard, "run-unavailable", "t1")
    assert unavailable["EVIDENCE_EVALUATED"] == 0.0
    assert unavailable["EVIDENCE_ELIGIBLE"] == 0.0
    assert unavailable["NO_ACCEPTABLE_ANSWER"] == 0.0  # not everyone was evaluated

    ineligible = _row(leaderboard, "run-ineligible", "t1")
    assert ineligible["EVIDENCE_EVALUATED"] == 1.0
    assert ineligible["EVIDENCE_ELIGIBLE"] == 0.0
    assert ineligible["NO_ACCEPTABLE_ANSWER"] == 0.0

    run_a = _row(leaderboard, "run-a", "t1")
    run_b = _row(leaderboard, "run-b", "t1")
    assert run_a["EVIDENCE_EVALUATED"] == run_b["EVIDENCE_EVALUATED"] == 1.0
    assert run_a["EVIDENCE_ELIGIBLE"] == run_b["EVIDENCE_ELIGIBLE"] == 1.0
    # A single "A" ballot between the only two eligible candidates.
    assert run_a["PAIRWISE_WIN_RATE"] == 1.0
    assert run_b["PAIRWISE_WIN_RATE"] == 0.0
    assert run_a["CHOICE_SCORE"] > run_b["CHOICE_SCORE"]
    assert run_a["BALLOT_COVERAGE"] == run_b["BALLOT_COVERAGE"] == 1.0

    # The ineligible/unavailable candidates' answer text must never reach a
    # pairwise-ballot prompt -- only evidence-eligible candidates are compared.
    pairwise_prompts = [c for c in backend.seen_content if "Pointwise evidence brief A" in c]
    assert len(pairwise_prompts) == 1
    assert "This run's evidence brief" not in pairwise_prompts[0]
    assert "This run is evaluated but too weak" not in pairwise_prompts[0]


def test_all_evaluated_and_rejected_yields_no_acceptable_answer(monkeypatch):
    topics = [Request(request_id="t1", title="What is X?")]
    reports = [
        _report("run-1", "t1", "Weak answer one."),
        _report("run-2", "t1", "Weak answer two."),
    ]
    brief_script = {
        "Weak answer one": _brief_json(factual_support=0.1, citation_validity=0.1),
        "Weak answer two": _brief_json(factual_support=0.2, citation_validity=0.2),
    }
    leaderboard, _backend = _run_judge(monkeypatch, reports, topics, brief_script)

    for run_id in ("run-1", "run-2"):
        row = _row(leaderboard, run_id, "t1")
        assert row["EVIDENCE_EVALUATED"] == 1.0
        assert row["EVIDENCE_ELIGIBLE"] == 0.0
        assert row["NO_ACCEPTABLE_ANSWER"] == 1.0


def test_position_reversal_reconciliation(monkeypatch):
    # Three independent topics: consistent reversal (keeps the ballot),
    # disagreeing reversal (collapses to ABSTAIN), and a reversal call that
    # fails every retry (leaves the original ballot untouched, not checked).
    topics = [Request(request_id=t, title="What is X?") for t in ("agree", "disagree", "unavailable")]
    reports = [
        _report("alpha-agree", "agree", "Alpha content agree."),
        _report("beta-agree", "agree", "Beta content agree."),
        _report("alpha-disagree", "disagree", "Alpha content disagree."),
        _report("beta-disagree", "disagree", "Beta content disagree."),
        _report("alpha-unavailable", "unavailable", "Alpha content unavailable."),
        _report("beta-unavailable", "unavailable", "Beta content unavailable."),
    ]
    brief_script = {text: _brief_json() for text in (
        "Alpha content agree", "Beta content agree",
        "Alpha content disagree", "Beta content disagree",
        "Alpha content unavailable", "Beta content unavailable",
    )}
    pairwise_script = {
        # Initial call for "agree": Alpha presented as A, Beta as B -> Alpha (A) wins.
        "Answer A:\nAlpha content agree.": '{"vote": "A"}',
        # Reversal call for "agree": Beta presented as A, Alpha as B. Voting "B"
        # (Alpha) again is consistent with the original "A" (Alpha) decision.
        "Answer A:\nBeta content agree.": '{"vote": "B"}',
        # Initial call for "disagree": Alpha (A) wins.
        "Answer A:\nAlpha content disagree.": '{"vote": "A"}',
        # Reversal call for "disagree": voting "A" this time means Beta wins,
        # contradicting the original Alpha win -> disagreement.
        "Answer A:\nBeta content disagree.": '{"vote": "A"}',
        # Initial call for "unavailable": Alpha (A) wins.
        "Answer A:\nAlpha content unavailable.": '{"vote": "A"}',
        # Reversal call for "unavailable": every attempt fails to parse.
        "Answer A:\nBeta content unavailable.": None,
    }
    leaderboard, _backend = _run_judge(
        monkeypatch, reports, topics, brief_script, pairwise_script,
        position_reversal_budget=1, top_k=1,
    )

    agree_alpha = _row(leaderboard, "alpha-agree", "agree")
    assert agree_alpha["POSITION_REVERSAL_CHECKED"] == 1.0
    assert agree_alpha["POSITION_REVERSAL_DISAGREEMENT_RATE"] == 0.0
    assert agree_alpha["PAIRWISE_WIN_RATE"] == 1.0  # original "A" ballot kept as-is

    disagree_alpha = _row(leaderboard, "alpha-disagree", "disagree")
    disagree_beta = _row(leaderboard, "beta-disagree", "disagree")
    assert disagree_alpha["POSITION_REVERSAL_CHECKED"] == 1.0
    assert disagree_alpha["POSITION_REVERSAL_DISAGREEMENT_RATE"] == 1.0
    # The contradictory ballot was replaced with ABSTAIN, not a directional win.
    assert disagree_alpha["PAIRWISE_WIN_RATE"] == 0.0
    assert disagree_beta["PAIRWISE_WIN_RATE"] == 0.0

    unavailable_alpha = _row(leaderboard, "alpha-unavailable", "unavailable")
    assert unavailable_alpha["POSITION_REVERSAL_CHECKED"] == 0.0
    assert unavailable_alpha["POSITION_REVERSAL_DISAGREEMENT_RATE"] == 0.0
    # Reversal was unavailable, so the original "A" ballot is untouched.
    assert unavailable_alpha["PAIRWISE_WIN_RATE"] == 1.0
