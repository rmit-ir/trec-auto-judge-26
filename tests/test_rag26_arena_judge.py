"""Synthetic, network-free end-to-end tests of RAG26ArenaJudge.judge().

Mirrors tests/test_ragree_choice_judge.py's pattern (fake LLM backend, drive
the whole judge() method, inspect the built Leaderboard), adapted for the
arena-only judge: no evidence gate, and an extra BOTH_BAD vote.
"""

from autojudge_base import Request
from autojudge_base.report import Report, ReportMetaData
from autojudge_base.report_sentences import NeuclirReportSentence

import judges.rag26_arena.judge as arena


class _Response:
    def __init__(self, text: str):
        self.text = text


def _report(run_id: str, topic_id: str, text: str) -> Report:
    return Report(
        metadata=ReportMetaData(team_id=run_id, run_id=run_id, topic_id=topic_id),
        responses=[NeuclirReportSentence(text=text, citations=None)],
    )


class _ScriptedBackend:
    """Routes each pairwise/reversal request to a canned vote by matching a
    marker substring against the "Answer A:\\n..." line (unambiguous: every
    arena prompt embeds the literal answer text once, in that position)."""

    def __init__(self, script: dict[str, str]):
        self.script = script
        self.seen_content: list[str] = []

    async def run_batched(self, requests):
        results = []
        for request in requests:
            content = request.messages[-1]["content"]
            self.seen_content.append(content)
            for marker, response_text in self.script.items():
                if marker in content:
                    # None simulates a transport/parse failure for this one entry.
                    results.append(_Response(response_text) if response_text is not None else object())
                    break
            else:
                raise AssertionError(f"no scripted response for prompt: {content[:300]!r}")
        return results


def _run_judge(monkeypatch, reports, topics, script, **judge_settings):
    monkeypatch.setattr(arena, "MinimaLlmResponse", _Response)
    backend = _ScriptedBackend(script)
    monkeypatch.setattr(arena, "_backend", lambda _config: backend)
    judge = arena.RAG26ArenaJudge()
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


def test_a_wins_gives_higher_score_and_full_win_rate(monkeypatch):
    topics = [Request(request_id="t1", title="What is X?")]
    reports = [_report("run-a", "t1", "Answer A text."), _report("run-b", "t1", "Answer B text.")]
    leaderboard, _backend = _run_judge(monkeypatch, reports, topics, {"": '{"vote": "A"}'})

    run_a, run_b = _row(leaderboard, "run-a", "t1"), _row(leaderboard, "run-b", "t1")
    assert run_a["PAIRWISE_WIN_RATE"] == 1.0
    assert run_b["PAIRWISE_WIN_RATE"] == 0.0
    assert run_a["RAG26_ARENA_SCORE"] > run_b["RAG26_ARENA_SCORE"]
    assert run_a["CONDORCET_WINNER"] == 1.0
    assert run_b["CONDORCET_WINNER"] == 0.0
    assert run_a["BALLOT_COVERAGE"] == run_b["BALLOT_COVERAGE"] == 1.0
    assert run_a["BOTH_BAD_RATE"] == run_b["BOTH_BAD_RATE"] == 0.0


def test_tie_splits_win_rate_evenly(monkeypatch):
    topics = [Request(request_id="t1", title="What is X?")]
    reports = [_report("run-a", "t1", "Answer A text."), _report("run-b", "t1", "Answer B text.")]
    leaderboard, _backend = _run_judge(monkeypatch, reports, topics, {"": '{"vote": "TIE"}'})

    run_a, run_b = _row(leaderboard, "run-a", "t1"), _row(leaderboard, "run-b", "t1")
    assert run_a["PAIRWISE_WIN_RATE"] == run_b["PAIRWISE_WIN_RATE"] == 0.5
    assert run_a["RAG26_ARENA_SCORE"] == run_b["RAG26_ARENA_SCORE"]
    assert run_a["BOTH_BAD_RATE"] == 0.0


def test_both_bad_is_a_tracked_diagnostic_not_a_loss(monkeypatch):
    """BOTH_BAD must fit like a TIE (never a loss for either side) while still
    surfacing as its own topic-level diagnostic (plan Section 10.3/11.6)."""
    topics = [Request(request_id="t1", title="What is X?")]
    reports = [_report("run-a", "t1", "Answer A text."), _report("run-b", "t1", "Answer B text.")]
    leaderboard, _backend = _run_judge(monkeypatch, reports, topics, {"": '{"vote": "BOTH_BAD"}'})

    run_a, run_b = _row(leaderboard, "run-a", "t1"), _row(leaderboard, "run-b", "t1")
    assert run_a["PAIRWISE_WIN_RATE"] == run_b["PAIRWISE_WIN_RATE"] == 0.5
    assert run_a["BOTH_BAD_RATE"] == run_b["BOTH_BAD_RATE"] == 1.0


def test_malformed_vote_never_becomes_a_loss(monkeypatch):
    """Every retry returns unparseable text: no ballot is ever cast for this
    pair, so it must not look like a loss for either candidate."""
    topics = [Request(request_id="t1", title="What is X?")]
    reports = [_report("run-a", "t1", "Answer A text."), _report("run-b", "t1", "Answer B text.")]
    leaderboard, backend = _run_judge(monkeypatch, reports, topics, {"": "not json, every attempt"})

    run_a, run_b = _row(leaderboard, "run-a", "t1"), _row(leaderboard, "run-b", "t1")
    assert run_a["PAIRWISE_WIN_RATE"] == run_b["PAIRWISE_WIN_RATE"] == 0.0
    # No ballot at all -> equal, un-fitted utilities (Davidson win prob at
    # utility 0 vs 0 is 1/3, not 1/2: the tie mass also gets a third).
    assert run_a["RAG26_ARENA_SCORE"] == run_b["RAG26_ARENA_SCORE"]
    assert run_a["BALLOT_COVERAGE"] == run_b["BALLOT_COVERAGE"] == 0.0
    # Bounded retries: llm_max_attempts=2, one call per attempt for this single pair.
    assert len(backend.seen_content) == 2


def test_one_candidate_topic_gets_neutral_score_and_no_llm_call(monkeypatch):
    topics = [Request(request_id="t1", title="What is X?")]
    reports = [_report("run-solo", "t1", "The only answer.")]
    leaderboard, backend = _run_judge(monkeypatch, reports, topics, {})

    row = _row(leaderboard, "run-solo", "t1")
    assert row["RAG26_ARENA_SCORE"] == 50.0
    assert row["BALLOT_COVERAGE"] == 1.0
    assert row["PAIRWISE_WIN_RATE"] == 0.0
    assert not backend.seen_content


def test_empty_answer_text_does_not_crash_and_can_still_lose(monkeypatch):
    topics = [Request(request_id="t1", title="What is X?")]
    reports = [_report("run-empty", "t1", ""), _report("run-full", "t1", "A full, useful answer.")]
    leaderboard, _backend = _run_judge(monkeypatch, reports, topics, {"": '{"vote": "B"}'})

    empty_row = _row(leaderboard, "run-empty", "t1")
    full_row = _row(leaderboard, "run-full", "t1")
    assert empty_row["PAIRWISE_WIN_RATE"] == 0.0
    assert full_row["PAIRWISE_WIN_RATE"] == 1.0


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
    script = {
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
        monkeypatch, reports, topics, script,
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


def test_small_pool_is_full_round_robin(monkeypatch):
    """Below full_round_robin_max_candidates every pair is compared at least once."""
    topics = [Request(request_id="t1", title="What is X?")]
    reports = [_report(f"run-{i}", "t1", f"Answer {i}.") for i in range(4)]
    leaderboard, backend = _run_judge(
        monkeypatch, reports, topics, {"": '{"vote": "TIE"}'},
        full_round_robin_max_candidates=5,
    )
    # C(4, 2) = 6 distinct pairs, one ballot each.
    assert len(backend.seen_content) == 6
    for row in (_row(leaderboard, f"run-{i}", "t1") for i in range(4)):
        assert row["BALLOT_COVERAGE"] == 1.0


def test_larger_pool_uses_sparse_graph_not_exhaustive(monkeypatch):
    """Above full_round_robin_max_candidates, SPARROW's sparse seed graph is
    used instead of the C(n, 2) exhaustive comparison count."""
    topics = [Request(request_id="t1", title="What is X?")]
    reports = [_report(f"run-{i}", "t1", f"Answer {i}.") for i in range(8)]
    leaderboard, backend = _run_judge(
        monkeypatch, reports, topics, {"": '{"vote": "TIE"}'},
        full_round_robin_max_candidates=5, initial_graph_degree=4,
    )
    exhaustive_pairs = 8 * 7 // 2
    assert 0 < len(backend.seen_content) < exhaustive_pairs
    for row in (_row(leaderboard, f"run-{i}", "t1") for i in range(8)):
        assert row["BALLOT_COVERAGE"] > 0.0
