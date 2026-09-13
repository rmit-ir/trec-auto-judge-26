"""Offline unit tests for scripts/smoke_rag26_rubric_model.py's parsing/cost
logic. No network: boto3.client is monkeypatched. This must stay free -- the
actual live smoke call against Bedrock is a manual, explicitly-approved step,
not part of the normal test suite."""

from scripts import smoke_rag26_rubric_model as smoke


class _FakeClient:
    def __init__(self, response, *, captured):
        self._response = response
        self._captured = captured

    def converse(self, **kwargs):
        self._captured.update(kwargs)
        return self._response


def _response(text: str, *, input_tokens=10, output_tokens=8, served_tier="flex"):
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
        "serviceTier": {"type": served_tier},
    }


def test_valid_structured_output_passes_and_estimates_cost(monkeypatch, capsys):
    captured = {}
    monkeypatch.setattr(smoke.boto3, "client", lambda *a, **k: _FakeClient(
        _response('{"status": "READY", "n": 2}'), captured=captured,
    ))
    code = smoke.run_smoke(
        "qwen.qwen3-235b-a22b-2507-v1:0", region="ap-southeast-2", tier="flex",
    )
    assert code == 0
    assert captured["serviceTier"] == {"type": "flex"}
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "USD" in out


def test_default_tier_omits_service_tier_field(monkeypatch):
    captured = {}
    monkeypatch.setattr(smoke.boto3, "client", lambda *a, **k: _FakeClient(
        _response('{"status": "READY", "n": 2}', served_tier="default"), captured=captured,
    ))
    smoke.run_smoke("some-model", region="us-west-2", tier="default")
    assert "serviceTier" not in captured


def test_malformed_json_output_is_a_failure(monkeypatch, capsys):
    monkeypatch.setattr(smoke.boto3, "client", lambda *a, **k: _FakeClient(
        _response("not json"), captured={},
    ))
    code = smoke.run_smoke("qwen.qwen3-235b-a22b-2507-v1:0", region="ap-southeast-2", tier="flex")
    assert code == 1
    assert "PARTIAL" in capsys.readouterr().out


def test_wrong_json_shape_is_a_failure(monkeypatch):
    monkeypatch.setattr(smoke.boto3, "client", lambda *a, **k: _FakeClient(
        _response('{"status": "NOT_READY"}'), captured={},
    ))
    assert smoke.run_smoke("m", region="us-west-2", tier="flex") == 1


def test_empty_content_is_a_failure(monkeypatch, capsys):
    monkeypatch.setattr(smoke.boto3, "client", lambda *a, **k: _FakeClient(
        {"output": {"message": {"content": []}}, "usage": {}}, captured={},
    ))
    code = smoke.run_smoke("m", region="us-west-2", tier="flex")
    assert code == 1
    assert "FAIL" in capsys.readouterr().err


def test_unpriced_model_reports_unknown_cost_but_still_passes(monkeypatch, capsys):
    monkeypatch.setattr(smoke.boto3, "client", lambda *a, **k: _FakeClient(
        _response('{"status": "READY", "n": 2}'), captured={},
    ))
    code = smoke.run_smoke("not-in-price-table", region="ap-southeast-2", tier="flex")
    assert code == 0
    assert "UNKNOWN" in capsys.readouterr().out
