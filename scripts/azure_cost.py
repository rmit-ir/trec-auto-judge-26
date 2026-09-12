#!/usr/bin/env python3
"""Estimate Azure OpenAI invocation costs from the local price table."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


PRICE_TABLE = (
    Path(__file__).resolve().parents[1] / "docs" / "azure-openai-pricing.csv"
)
TOKENS_PER_MILLION = Decimal("1000000")


@dataclass(frozen=True)
class ModelPrice:
    """One Azure OpenAI Global Standard price row."""

    name: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    tier: str
    region: str
    tokenizer_encoding: str | None = None


def load_model_prices() -> dict[str, ModelPrice]:
    """Load the local, versioned pricing table keyed by Azure model name."""
    with PRICE_TABLE.open(newline="") as price_file:
        reader = csv.DictReader(price_file)
        return {
            row["model_name"]: ModelPrice(
                name=row["model_name"],
                input_usd_per_million=Decimal(row["input_usd_per_million"]),
                output_usd_per_million=Decimal(row["output_usd_per_million"]),
                tier=row["tier"],
                region=row["region"],
                tokenizer_encoding=row.get("tokenizer_encoding") or None,
            )
            for row in reader
        }


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> Decimal:
    """Return the USD estimate for a single request's input and output tokens.

    # ponytail: uncached input rate applied to all input tokens (no cache-hit
    # discount modeled). Switch to the cached_input_usd_per_million column if
    # prompt-cache hit ratio starts mattering for cost accuracy.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative")
    price = load_model_prices()[model_name]
    return (
        Decimal(input_tokens) * price.input_usd_per_million
        + Decimal(output_tokens) * price.output_usd_per_million
    ) / TOKENS_PER_MILLION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_name", choices=load_model_prices())
    parser.add_argument("input_tokens", type=int)
    parser.add_argument("output_tokens", type=int)
    args = parser.parse_args()
    print(f"${estimate_cost(args.model_name, args.input_tokens, args.output_tokens):.6f} USD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
