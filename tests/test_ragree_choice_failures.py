"""Regression tests for RAGree Choice unavailable-versus-verdict handling."""

import judges.ragree_choice.ragree_choice_judge as choice


class _Response:
    def __init__(self, text: str):
        self.text = text


def _valid_brief(*, critical_error=False) -> _Response:
    return _Response(
        "{"
        f'\"critical_error\": {str(critical_error).lower()}, '
        '\"factual_support\": 0.9, \"completeness\": 0.8, '
        '\"citation_validity\": 0.85, \"confidence\": 0.75}'
    )


def test_evidence_schema_failure_is_unavailable_not_a_zero_verdict(monkeypatch):
    monkeypatch.setattr(choice, "MinimaLlmResponse", _Response)
    judge = choice.RAGreeChoiceJudge()
    valid = judge._parse_brief(_valid_brief())
    assert valid is not None
    assert judge._eligible(valid, factual_support_threshold=0.65, citation_validity_threshold=0.65)
    assert judge._parse_brief(object()) is None
    assert judge._parse_brief(_Response("not json")) is None
    assert judge._parse_brief(_Response(
        '{"critical_error": "true", "factual_support": 0.9, "completeness": 0.8, '
        '"citation_validity": 0.85, "confidence": 0.75}'
    )) is None
    assert judge._parse_brief(_Response(
        '{"critical_error": false, "factual_support": "0.9", "completeness": 0.8, '
        '"citation_validity": 0.85, "confidence": 0.75}'
    )) is None
    assert judge._no_acceptable_answer([None, valid], []) == 0.0
    rejected = choice.EvidenceBrief(False, 0.1, 0.1, 0.1, 0.9)
    assert judge._no_acceptable_answer([rejected], []) == 1.0
    critical = judge._parse_brief(_valid_brief(critical_error=True))
    assert critical is not None
    assert not judge._eligible(critical, factual_support_threshold=0.65, citation_validity_threshold=0.65)


def test_retry_recovers_a_valid_evidence_brief(monkeypatch):
    monkeypatch.setattr(choice, "MinimaLlmResponse", _Response)
    judge = choice.RAGreeChoiceJudge()

    class _Backend:
        calls = 0

        async def run_batched(self, _requests):
            self.calls += 1
            return [_Response("not json") if self.calls == 1 else _valid_brief()]

    backend = _Backend()
    monkeypatch.setattr(choice, "_backend", lambda _config: backend)
    monkeypatch.setattr(choice.time, "sleep", lambda _seconds: None)
    results = judge._run_parsed_with_retries(
        object(), [object()], judge._parse_brief,
        max_attempts=2, retry_backoff_seconds=0.0, operation="test brief",
    )
    assert backend.calls == 2
    assert results[0] is not None


def test_pairwise_failure_is_distinct_from_explicit_abstention(monkeypatch):
    monkeypatch.setattr(choice, "MinimaLlmResponse", _Response)
    judge = choice.RAGreeChoiceJudge()
    assert judge._parse_vote(object()) is None
    assert judge._parse_vote(_Response("not json")) is None
    assert judge._parse_vote(_Response('{"vote": "ABSTAIN"}')) == "ABSTAIN"


def test_config_validation_rejects_unsafe_values():
    judge = choice.RAGreeChoiceJudge()
    try:
        judge._validate_config(
            factual_support_threshold=1.2,
            citation_validity_threshold=0.65,
            full_round_robin_max_candidates=5,
            initial_graph_degree=4,
            adaptive_comparisons_per_topic=4,
            top_k=1,
            max_answer_chars=2000,
            max_evidence_chars=4000,
            judge_weight=1.0,
            exact_kemeny_limit=8,
            llm_max_attempts=3,
            llm_retry_backoff_seconds=0.25,
        )
    except ValueError as error:
        assert "factual_support_threshold" in str(error)
    else:
        raise AssertionError("invalid threshold should fail fast")
