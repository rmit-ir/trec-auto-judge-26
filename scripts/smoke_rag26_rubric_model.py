#!/usr/bin/env python3
"""One targeted Bedrock Converse smoke request against a single selected model.

Unlike test_bedrock_models.py (which calls every model in the price table),
this tests exactly one model/region/tier -- for vetting a candidate model
before committing to it for the RAG26 rubric-construction workstream
(docs/rag26-rubric-construction-execution-plan.md Section 5) without paying
for the whole historic model list. The prompt requests structured (JSON)
output because that is the actual capability rubric generation depends on,
not just free-text completion.

Load temporary AWS credentials from .env before running this script. Prints
no credentials and does not write prompts or responses to disk.
"""

from __future__ import annotations

import argparse
import json
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

try:
    from bedrock_cost import estimate_cost  # running as `python scripts/smoke_rag26_rubric_model.py`
except ImportError:
    from scripts.bedrock_cost import estimate_cost  # imported as scripts.smoke_rag26_rubric_model

# Minimal structured-output check: the rubric generator needs reliable JSON
# object output, not just free-text completion.
PROMPT = 'Reply with ONLY this JSON object, no other text: {"status": "READY", "n": 2}'


def _extract_text(response: dict) -> str:
    content = response.get("output", {}).get("message", {}).get("content", [])
    return "".join(block.get("text", "") for block in content)


def run_smoke(model_id: str, *, region: str, tier: str) -> int:
    try:
        client = boto3.client("bedrock-runtime", region_name=region)
        request = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": PROMPT}]}],
            "inferenceConfig": {"maxTokens": 64, "temperature": 0},
        }
        # Some Bedrock models reject an explicit serviceTier, including "default";
        # omitting it selects Bedrock's Standard on-demand tier (see test_bedrock_models.py).
        if tier != "default":
            request["serviceTier"] = {"type": tier}
        response = client.converse(**request)
    except NoCredentialsError as error:
        print(f"FAIL {model_id}: no AWS credentials ({error})", file=sys.stderr)
        return 1
    except (BotoCoreError, ClientError) as error:
        print(f"FAIL {model_id}: {error}", file=sys.stderr)
        return 1

    text = _extract_text(response)
    if not text:
        print(f"FAIL {model_id}: Bedrock returned no content blocks", file=sys.stderr)
        return 1

    try:
        parsed = json.loads(text)
        json_ok = isinstance(parsed, dict) and parsed.get("status") == "READY" and parsed.get("n") == 2
    except json.JSONDecodeError:
        json_ok = False

    usage = response.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    served_tier = response.get("serviceTier", {}).get("type", tier)
    try:
        cost_text = f"${estimate_cost(model_id, input_tokens, output_tokens, tier=tier):.6f} USD"
    except KeyError:
        cost_text = "UNKNOWN (model/tier not in local price table)"

    status = "PASS" if json_ok else "PARTIAL"
    print(
        f"{status} {model_id} (region={region}, requested_tier={tier}, served_tier={served_tier}): "
        f"{input_tokens} input + {output_tokens} output tokens (estimated {cost_text})"
    )
    if not json_ok:
        print(f"  structured output did not parse as expected JSON: {text[:200]!r}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="Bedrock Runtime model ID")
    parser.add_argument("--region", required=True, help="e.g. ap-southeast-2")
    parser.add_argument("--tier", default="flex", help="Bedrock service tier (default: flex)")
    args = parser.parse_args()
    return run_smoke(args.model_id, region=args.region, tier=args.tier)


if __name__ == "__main__":
    raise SystemExit(main())
