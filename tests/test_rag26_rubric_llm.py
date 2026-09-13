"""Offline tests for judges.rag26_arena.rubrics.llm. No network: boto3.client
and minima_llm's OpenAIMinimaLlm are monkeypatched/faked."""

from decimal import Decimal

import pytest

import judges.rag26_arena.rubrics.llm as llm_module
from judges.rag26_arena.rubrics.generation_models import UsageRecord
from judges.rag26_arena.rubrics.llm import (
    BedrockUnavailableError,
    CostTracker,
    LlmResult,
    call_bedrock,
    print_budget_banner,
)


class _FakeBedrockClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _ok_response(text: str, *, input_tokens=100, output_tokens=20):
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
    }


def test_call_bedrock_success_first_try(monkeypatch):
    client = _FakeBedrockClient([_ok_response('{"status": "READY"}')])
    monkeypatch.setattr(llm_module.boto3, "client", lambda *a, **k: client)
    result = call_bedrock("qwen.qwen3-235b-a22b-2507-v1:0", [{"role": "user", "content": "hi"}],
                           region="ap-southeast-2", tier="flex")
    assert result.text == '{"status": "READY"}'
    assert result.input_tokens == 100
    assert result.tier == "flex"
    assert client.requests[0]["serviceTier"] == {"type": "flex"}


def test_call_bedrock_splits_system_message(monkeypatch):
    client = _FakeBedrockClient([_ok_response("ok")])
    monkeypatch.setattr(llm_module.boto3, "client", lambda *a, **k: client)
    call_bedrock("m", [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
                 region="us-west-2", tier="default")
    request = client.requests[0]
    assert request["system"] == [{"text": "sys"}]
    assert request["messages"] == [{"role": "user", "content": [{"text": "usr"}]}]
    assert "serviceTier" not in request  # default tier omits the field


def test_call_bedrock_retries_then_succeeds(monkeypatch):
    from botocore.exceptions import ClientError
    error = ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "Converse")
    client = _FakeBedrockClient([error, _ok_response("recovered")])
    monkeypatch.setattr(llm_module.boto3, "client", lambda *a, **k: client)
    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)
    result = call_bedrock("m", [{"role": "user", "content": "hi"}], region="us-west-2",
                          tier="flex", max_attempts=2)
    assert result.text == "recovered"
    assert len(client.requests) == 2


def test_call_bedrock_falls_back_to_standard_tier_explicitly(monkeypatch, capsys):
    from botocore.exceptions import ClientError
    error = ClientError({"Error": {"Code": "ThrottlingException", "Message": "flex unavailable"}}, "Converse")
    client = _FakeBedrockClient([error, error, _ok_response("standard worked")])
    monkeypatch.setattr(llm_module.boto3, "client", lambda *a, **k: client)
    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)
    result = call_bedrock("m", [{"role": "user", "content": "hi"}], region="us-west-2",
                          tier="flex", fallback_tier="default", max_attempts=2)
    assert result.text == "standard worked"
    assert result.tier == "default"
    assert "falling back to tier='default'" in capsys.readouterr().out


def test_call_bedrock_never_falls_back_beyond_configured_tiers(monkeypatch):
    from botocore.exceptions import ClientError
    error = ClientError({"Error": {"Code": "ThrottlingException", "Message": "down"}}, "Converse")
    client = _FakeBedrockClient([error, error])
    monkeypatch.setattr(llm_module.boto3, "client", lambda *a, **k: client)
    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)
    with pytest.raises(BedrockUnavailableError):
        call_bedrock("m", [{"role": "user", "content": "hi"}], region="us-west-2",
                     tier="flex", fallback_tier=None, max_attempts=2)


def test_call_bedrock_empty_content_is_a_failure(monkeypatch):
    client = _FakeBedrockClient([
        {"output": {"message": {"content": []}}, "usage": {}},
        {"output": {"message": {"content": []}}, "usage": {}},
    ])
    monkeypatch.setattr(llm_module.boto3, "client", lambda *a, **k: client)
    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)
    with pytest.raises(BedrockUnavailableError):
        call_bedrock("m", [{"role": "user", "content": "hi"}], region="us-west-2",
                     tier="flex", max_attempts=2)


def test_cost_tracker_records_zero_cost_on_cache_hit():
    tracker = CostTracker(hard_cap_usd=Decimal("25"))
    result = LlmResult(text="x", input_tokens=1_000_000, output_tokens=1_000_000,
                        model_id="qwen.qwen3-235b-a22b-2507-v1:0", tier="flex", region="ap-southeast-2")
    record = tracker.record_bedrock(result, stage="characterize", cache_hit=True, qid="t1")
    assert record.estimated_usd == Decimal("0")
    assert tracker.total_usd() == Decimal("0")


def test_cost_tracker_estimates_real_spend_on_a_miss():
    tracker = CostTracker(hard_cap_usd=Decimal("25"))
    result = LlmResult(text="x", input_tokens=1_000_000, output_tokens=1_000_000,
                        model_id="qwen.qwen3-235b-a22b-2507-v1:0", tier="flex", region="ap-southeast-2")
    record = tracker.record_bedrock(result, stage="characterize", cache_hit=False, qid="t1")
    assert record.estimated_usd == Decimal("0.1133") + Decimal("0.4532")


def test_cost_tracker_by_stage_and_budget_check():
    tracker = CostTracker(hard_cap_usd=Decimal("0.0001"))
    result = LlmResult(text="x", input_tokens=1_000_000, output_tokens=1_000_000,
                        model_id="qwen.qwen3-235b-a22b-2507-v1:0", tier="flex", region="ap-southeast-2")
    tracker.record_bedrock(result, stage="characterize", cache_hit=False, qid="t1")
    tracker.record_bedrock(result, stage="explicit", cache_hit=False, qid="t1")
    stages = tracker.by_stage()
    assert set(stages) == {"characterize", "explicit"}
    assert tracker.total_usd() > tracker.hard_cap_usd
    with pytest.raises(llm_module.BudgetExceededError):
        tracker.check_budget()


def test_print_budget_banner_smoke(capsys):
    print_budget_banner(
        model_id="qwen.qwen3-235b-a22b-2507-v1:0", tier="flex", region="ap-southeast-2",
        topic_count=30, stage="characterize", ceiling_input_tokens=1_000, ceiling_output_tokens=200,
        ceiling_usd=Decimal("1.00"), cache_status="empty",
    )
    out = capsys.readouterr().out
    assert "characterize" in out and "30" in out and "$1.00" in out
