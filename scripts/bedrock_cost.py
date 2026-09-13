#!/usr/bin/env python3
"""Estimate Amazon Bedrock invocation costs from the local price table."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


PRICE_TABLE = (
    Path(__file__).resolve().parents[1] / "docs" / "bedrock-pricing-us-west-2.csv"
)
TOKENS_PER_MILLION = Decimal("1000000")


@dataclass(frozen=True)
class ModelPrice:
    """One Standard on-demand Bedrock Runtime price pair."""

    name: str
    model_id: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    region: str
    tier: str


def _price_key(model_id: str, tier: str) -> str:
    """Same Bedrock model ID can be invoked at more than one service tier
    (e.g. Qwen3 235B Flex vs Standard) at different prices, so the model ID
    alone is not a unique price key -- (model_id, tier) is."""
    return f"{model_id}#{tier}"


def load_model_prices() -> dict[str, ModelPrice]:
    """Load the local, versioned pricing table keyed by `model_id#tier`."""
    with PRICE_TABLE.open(newline="") as price_file:
        reader = csv.DictReader(price_file)
        return {
            _price_key(row["model_id"], row["tier"]): ModelPrice(
                name=row["model_name"],
                model_id=row["model_id"],
                input_usd_per_million=Decimal(row["input_usd_per_million"]),
                output_usd_per_million=Decimal(row["output_usd_per_million"]),
                region=row["region"],
                tier=row["tier"],
            )
            for row in reader
        }


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int, *, tier: str = "default") -> Decimal:
    """Return the USD estimate for a single request's input and output tokens."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative")
    price = load_model_prices()[_price_key(model_id, tier)]
    return (
        Decimal(input_tokens) * price.input_usd_per_million
        + Decimal(output_tokens) * price.output_usd_per_million
    ) / TOKENS_PER_MILLION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("input_tokens", type=int)
    parser.add_argument("output_tokens", type=int)
    parser.add_argument("--tier", default="default")
    args = parser.parse_args()
    print(f"${estimate_cost(args.model_id, args.input_tokens, args.output_tokens, tier=args.tier):.6f} USD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
