from decimal import Decimal

import pytest

from scripts.bedrock_cost import estimate_cost, load_model_prices


def test_estimate_cost_default_tier():
    cost = estimate_cost("zai.glm-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == Decimal("1.00") + Decimal("3.20")


def test_qwen_235b_flex_and_standard_are_distinct_prices_for_the_same_model_id():
    """Regression test: the price table used to be keyed by model_id alone, so
    a Flex and a Standard row for the same Bedrock model ID silently collided
    (the later CSV row won, the earlier one vanished with no error)."""
    flex = estimate_cost(
        "qwen.qwen3-235b-a22b-2507-v1:0", input_tokens=1_000_000, output_tokens=1_000_000, tier="flex",
    )
    standard = estimate_cost(
        "qwen.qwen3-235b-a22b-2507-v1:0", input_tokens=1_000_000, output_tokens=1_000_000, tier="default",
    )
    assert flex == Decimal("0.1133") + Decimal("0.4532")
    assert standard == Decimal("0.2266") + Decimal("0.9064")
    assert flex < standard


def test_qwen_32b_flex_price():
    cost = estimate_cost(
        "qwen.qwen3-32b-v1:0", input_tokens=1_000_000, output_tokens=1_000_000, tier="flex",
    )
    assert cost == Decimal("0.0773") + Decimal("0.3090")


def test_estimate_cost_rejects_negative_tokens():
    with pytest.raises(ValueError):
        estimate_cost("zai.glm-5", input_tokens=-1, output_tokens=0)


def test_estimate_cost_unknown_model_raises():
    with pytest.raises(KeyError):
        estimate_cost("not-a-real-model", input_tokens=1, output_tokens=1)


def test_estimate_cost_wrong_tier_for_known_model_raises():
    """A model that exists but not at the requested tier must fail loudly,
    not silently fall back to a different tier's price."""
    with pytest.raises(KeyError):
        estimate_cost("qwen.qwen3-32b-v1:0", input_tokens=1, output_tokens=1, tier="default")


def test_price_table_has_no_ambiguous_model_id_tier_pairs():
    prices = load_model_prices()
    assert prices  # sanity: the table isn't empty
    # Every (model_id, tier) pair in the CSV must map to exactly one row --
    # this guards against a future duplicate row silently overwriting another.
    assert len(prices) == len({(p.model_id, p.tier) for p in prices.values()})
