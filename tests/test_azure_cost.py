from decimal import Decimal

import pytest

from scripts.azure_cost import estimate_cost


def test_estimate_cost_gpt_5_6_luna():
    cost = estimate_cost("gpt-5.6-luna", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == Decimal("0.20") + Decimal("1.20")


def test_estimate_cost_rejects_negative_tokens():
    with pytest.raises(ValueError):
        estimate_cost("gpt-5.6-luna", input_tokens=-1, output_tokens=0)


def test_estimate_cost_unknown_model_raises():
    with pytest.raises(KeyError):
        estimate_cost("not-a-real-model", input_tokens=1, output_tokens=1)
