"""End-to-end tests of RAG26ArenaJudge's citation-support lane wiring.

Complements tests/test_rag26_support.py (pure metrics) by driving the whole
judge() method with citation_support_enabled=True through a fake LLM backend,
so a wiring bug between judge.py and support.py would show up here.
"""

from autojudge_base import Request
from autojudge_base.report import Document, Report, ReportMetaData
from autojudge_base.report_sentences import NeuclirReportSentence

import judges.rag26_arena.judge as arena


class _Response:
    def __init__(self, text: str):
        self.text = text


def _doc(doc_id: str, text: str) -> Document:
    return Document(id=doc_id, text=text)


def _report(run_id: str, topic_id: str, sentences, documents) -> Report:
    return Report(
        metadata=ReportMetaData(team_id=run_id, run_id=run_id, topic_id=topic_id),
        responses=sentences, documents=documents,
    )


class _ScriptedBackend:
    """Routes each request by prompt shape: arena ballots contain "Answer A:",
    citation-support batches contain "Cited document:"."""

    def __init__(self, arena_vote: str, support_reply: str | None):
        self.arena_vote = arena_vote
        self.support_reply = support_reply
        self.support_calls = 0
        self.seen_content: list[str] = []

    async def run_batched(self, requests):
        results = []
        for request in requests:
            content = request.messages[-1]["content"]
            self.seen_content.append(content)
            if "Cited document:" in content:
                self.support_calls += 1
                results.append(_Response(self.support_reply) if self.support_reply else object())
            else:
                results.append(_Response(f'{{"vote": "{self.arena_vote}"}}'))
        return results


def _run_judge(monkeypatch, reports, topics, arena_vote="TIE", support_reply=None, **judge_settings):
    monkeypatch.setattr(arena, "MinimaLlmResponse", _Response)
    backend = _ScriptedBackend(arena_vote, support_reply)
    monkeypatch.setattr(arena, "_backend", lambda _config: backend)
    judge = arena.RAG26ArenaJudge()
    settings = {
        "adaptive_comparisons_per_topic": 0,
        "position_reversal_budget": 0,
        "llm_max_attempts": 2,
        "llm_retry_backoff_seconds": 0.0,
        "citation_support_enabled": True,
        **judge_settings,
    }
    leaderboard = judge.judge(reports, topics, llm_config=object(), **settings)
    return leaderboard, backend


def _row(leaderboard, run_id: str, topic_id: str) -> dict[str, float]:
    for entry in leaderboard.entries:
        if entry.run_id == run_id and entry.topic_id == topic_id:
            return dict(entry.values)
    raise AssertionError(f"no leaderboard row for run={run_id!r} topic={topic_id!r}")


def test_citation_support_scores_independently_of_arena_ballots(monkeypatch):
    topics = [Request(request_id="t1", title="What is X?")]
    reports = [
        _report(
            "run-a", "t1",
            [NeuclirReportSentence(text="A claim.", citations=["d1"])],
            {"d1": _doc("d1", "Fully supports the claim.")},
        ),
        _report("run-b", "t1", [NeuclirReportSentence(text="B claim.", citations=None)], {}),
    ]
    leaderboard, backend = _run_judge(
        monkeypatch, reports, topics, arena_vote="A",
        support_reply='[{"index": 0, "support": "FULL"}]',
    )

    run_a = _row(leaderboard, "run-a", "t1")
    run_b = _row(leaderboard, "run-b", "t1")
    assert run_a["RAG26_CIT_PREC_FIRST"] == 1.0
    assert run_a["RAG26_SUPPORT_EVALUATED"] == 1.0
    # No citations at all: vacuously evaluated, zero score.
    assert run_b["RAG26_SUPPORT_EVALUATED"] == 1.0
    assert run_b["RAG26_CIT_PREC_FIRST"] == 0.0
    # Citation support must never influence the Arena ballot outcome.
    assert run_a["PAIRWISE_WIN_RATE"] == 1.0
    assert backend.support_calls == 1


def test_citation_support_disabled_by_default_gives_unevaluated_placeholders(monkeypatch):
    topics = [Request(request_id="t1", title="What is X?")]
    reports = [
        _report(
            "run-a", "t1",
            [NeuclirReportSentence(text="A claim.", citations=["d1"])],
            {"d1": _doc("d1", "Fully supports the claim.")},
        ),
        _report("run-b", "t1", [NeuclirReportSentence(text="B claim.", citations=None)], {}),
    ]
    monkeypatch.setattr(arena, "MinimaLlmResponse", _Response)
    backend = _ScriptedBackend("TIE", None)
    monkeypatch.setattr(arena, "_backend", lambda _config: backend)
    judge = arena.RAG26ArenaJudge()
    leaderboard = judge.judge(
        reports, topics, llm_config=object(),
        adaptive_comparisons_per_topic=0, position_reversal_budget=0,
    )
    row = _row(leaderboard, "run-a", "t1")
    assert row["RAG26_SUPPORT_EVALUATED"] == 0.0
    assert row["RAG26_CIT_PREC_FIRST"] == 0.0
    assert backend.support_calls == 0  # citation lane never called when disabled


def test_malformed_support_reply_leaves_it_unevaluated_not_zero_scored(monkeypatch):
    topics = [Request(request_id="t1", title="What is X?")]
    reports = [
        _report(
            "run-a", "t1",
            [NeuclirReportSentence(text="A claim.", citations=["d1"])],
            {"d1": _doc("d1", "Some text.")},
        ),
    ]
    leaderboard, backend = _run_judge(monkeypatch, reports, topics, support_reply="not json")
    row = _row(leaderboard, "run-a", "t1")
    assert row["RAG26_SUPPORT_EVALUATED"] == 0.0
    assert row["RAG26_CIT_PREC_FIRST"] == 0.0
    # Bounded retries: llm_max_attempts=2, one call per attempt for this single chunk.
    assert backend.support_calls == 2
