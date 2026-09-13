"""Unit tests for RAG26ArenaJudge vote parsing, config validation, and retries."""

import judges.rag26_arena.judge as arena
from judges.rag26_arena.models import BALLOT_VOTE, RAW_VOTES


class _Response:
    def __init__(self, text: str):
        self.text = text


def test_parse_vote_accepts_all_five_raw_votes(monkeypatch):
    monkeypatch.setattr(arena, "MinimaLlmResponse", _Response)
    judge = arena.RAG26ArenaJudge()
    for vote in RAW_VOTES:
        assert judge._parse_vote(_Response(f'{{"vote": "{vote}"}}')) == vote
        assert judge._parse_vote(_Response(f'{{"vote": "{vote.lower()}"}}')) == vote


def test_parse_vote_rejects_malformed_or_unknown(monkeypatch):
    monkeypatch.setattr(arena, "MinimaLlmResponse", _Response)
    judge = arena.RAG26ArenaJudge()
    assert judge._parse_vote(object()) is None  # not a MinimaLlmResponse: transport failure
    assert judge._parse_vote(_Response("not json")) is None
    assert judge._parse_vote(_Response('{"vote": "C"}')) is None
    assert judge._parse_vote(_Response('{"vote": "WIN"}')) is None
    assert judge._parse_vote(_Response("{}")) is None


def test_both_bad_maps_to_tie_for_the_fit():
    assert BALLOT_VOTE["BOTH_BAD"] == "TIE"
    assert BALLOT_VOTE["A"] == "A" and BALLOT_VOTE["B"] == "B"
    assert BALLOT_VOTE["ABSTAIN"] == "ABSTAIN"


def test_retry_recovers_a_valid_vote_after_malformed_attempt(monkeypatch):
    monkeypatch.setattr(arena, "MinimaLlmResponse", _Response)
    judge = arena.RAG26ArenaJudge()

    class _Backend:
        calls = 0

        async def run_batched(self, _requests):
            self.calls += 1
            return [_Response("not json") if self.calls == 1 else _Response('{"vote": "A"}')]

    backend = _Backend()
    monkeypatch.setattr(arena, "_backend", lambda _config: backend)
    monkeypatch.setattr(arena.time, "sleep", lambda _seconds: None)
    results = judge._run_parsed_with_retries(
        object(), [object()], judge._parse_vote,
        max_attempts=2, retry_backoff_seconds=0.0, operation="test vote",
    )
    assert backend.calls == 2
    assert results == ["A"]


def test_exhausted_retries_yield_none_not_a_fabricated_vote(monkeypatch):
    monkeypatch.setattr(arena, "MinimaLlmResponse", _Response)
    judge = arena.RAG26ArenaJudge()
    monkeypatch.setattr(arena, "_backend", lambda _config: _RaisingBackend())
    monkeypatch.setattr(arena.time, "sleep", lambda _seconds: None)
    results = judge._run_parsed_with_retries(
        object(), [object()], judge._parse_vote,
        max_attempts=2, retry_backoff_seconds=0.0, operation="test vote",
    )
    assert results == [None]


class _RaisingBackend:
    async def run_batched(self, _requests):
        raise RuntimeError("endpoint unavailable")


def test_config_validation_rejects_unsafe_values():
    judge = arena.RAG26ArenaJudge()
    good = dict(
        full_round_robin_max_candidates=5, initial_graph_degree=4,
        adaptive_comparisons_per_topic=4, top_k=1, arena_max_answer_chars=2000,
        judge_weight=1.0, exact_kemeny_limit=8, llm_max_attempts=3,
        llm_retry_backoff_seconds=0.25, position_reversal_budget=2,
        support_batch_size=10, support_max_doc_chars=1500,
    )
    judge._validate_config(**good)  # sanity: the defaults themselves are valid

    for bad_field, bad_value in (
        ("full_round_robin_max_candidates", 0),
        ("top_k", 0),
        ("arena_max_answer_chars", -1),
        ("initial_graph_degree", -1),
        ("position_reversal_budget", -1),
        ("judge_weight", 0.0),
        ("llm_retry_backoff_seconds", -0.1),
        ("support_batch_size", 0),
        ("support_max_doc_chars", -1),
    ):
        settings = {**good, bad_field: bad_value}
        try:
            judge._validate_config(**settings)
        except ValueError as error:
            assert bad_field in str(error)
        else:
            raise AssertionError(f"{bad_field}={bad_value!r} should fail fast")
