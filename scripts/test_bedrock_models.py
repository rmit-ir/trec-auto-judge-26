#!/usr/bin/env python3
"""Make one minimal Amazon Bedrock Converse request to each development model.

Load temporary AWS credentials from .env before running this script. It prints
no credentials and does not write prompts or responses to disk.
"""

from __future__ import annotations

import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from bedrock_cost import ModelPrice, estimate_cost, load_model_prices

PROMPT = "Reply with exactly: READY"


def test_model(price: ModelPrice) -> bool:
    """Run a short response and return whether Bedrock produced content."""
    try:
        client = boto3.client("bedrock-runtime", region_name=price.region)
        request = {
            "modelId": price.model_id,
            "messages": [{"role": "user", "content": [{"text": PROMPT}]}],
            "inferenceConfig": {"maxTokens": 32, "temperature": 0},
        }
        # These models reject serviceTier, including an explicit "default" value.
        # Omitting it selects Bedrock's Standard on-demand tier.
        if price.tier != "default":
            request["serviceTier"] = {"type": price.tier}
        response = client.converse(**request)
    except (BotoCoreError, ClientError) as error:
        print(f"FAIL {price.name}: {error}", file=sys.stderr)
        return False
    output = response.get("output", {}).get("message", {}).get("content", [])
    if not output:
        print(f"FAIL {price.name}: Bedrock returned no content blocks", file=sys.stderr)
        return False
    usage = response.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    cost = estimate_cost(price.model_id, input_tokens, output_tokens)
    served_tier = response.get("serviceTier", {}).get("type", price.tier)
    print(
        f"PASS {price.name} ({price.region}, {served_tier}): "
        f"{input_tokens} input + {output_tokens} output tokens "
        f"(estimated ${cost:.6f} USD)"
    )
    return True


def main() -> int:
    try:
        prices = load_model_prices().values()
    except (OSError, ValueError) as error:
        print(f"Unable to load Bedrock price table: {error}", file=sys.stderr)
        return 1
    try:
        results = [test_model(price) for price in prices]
    except (BotoCoreError, NoCredentialsError) as error:
        print(f"Bedrock smoke test failed: {error}", file=sys.stderr)
        return 1
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
