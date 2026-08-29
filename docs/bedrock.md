# Amazon Bedrock model smoke test

RAGree uses Amazon Bedrock for development experiments with open-weight models.
This is a direct AWS Bedrock test, separate from the TIRA endpoint contract: a
submitted judge must still construct its LLM client from the injected
`llm_config` values.

## Credentials

Keep these values in the local, gitignored `.env` file. Do not commit the file
or print its contents:

```bash
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
```

`AWS_SESSION_TOKEN` is short lived under RMIT policy. When the smoke test
returns an expired-token or invalid-token error, renew the credentials in
`.env`, then rerun the same command. No source-code change is needed.

The smoke test does not read `AWS_REGION`: each model's `Converse` client uses
the `region` recorded for it in the local price table instead. Add
`AWS_REGION=...` to `.env` only if some other tool you run needs a default
Region.

## Run the test

Install the optional Bedrock extra (included by `.[all]`), load the local
environment, then invoke the standalone test:

```bash
set -a; source .env; set +a
.venv/bin/python scripts/test_bedrock_models.py
```

The script makes one short `Converse` request to each model and prints a
pass/fail summary, Bedrock's input/output token counts, and an estimated USD
cost. A non-empty response is sufficient: reasoning models can legitimately
emit a reasoning block before plain text. It does not read evaluation data or
log credentials. It is
deliberately not collected by `pytest`, preventing routine test runs from
spending money or depending on temporary AWS credentials.

## Current open-weight representatives and prices

The five models below are a deliberately broad, current selection—one model
from each open-weight family—not a benchmark leaderboard. All use the Bedrock
Runtime `Converse` API. The script does not compare prices or regions itself,
it just uses whatever the CSV says — the CSV is the versioned,
machine-readable source of truth:
[`bedrock-pricing-us-west-2.csv`](bedrock-pricing-us-west-2.csv). Live probes
on 2026-08-29 found that all five reject the `serviceTier` request field,
including an explicit `default` value, so the script omits it and Bedrock
uses Standard on-demand routing.

`us-west-2` is confirmed the cheapest available region for GLM 5, DeepSeek
V3.2, and Nemotron 3 Super 120B — checked against the AWS pricing page on
2026-08-29, each has higher Standard on-demand rates in every other Region it
is offered in (e.g. GLM 5 is $1.20/$3.84 in most Asia Pacific/South America
Regions and $1.55/$4.96 in London, versus $1.00/$3.20 in the `us-east-1`
/`us-east-2`/`us-west-2` group). gpt-oss-20b and Qwen3 Coder 480B A35B, by
contrast, appear to be priced identically in every Region Bedrock offers them
in — no cheaper Region was found for either, so `us-west-2` is simply a
convenient pick among equals, not a discount.

| Provider | Model | Bedrock Runtime model ID | Region | Tier | Input / 1M | Output / 1M |
| --- | --- | --- | --- | --- | --- | --- |
| Z.AI | GLM 5 (latest GLM) | `zai.glm-5` | us-west-2 | Standard (implicit) | $1.00 | $3.20 |
| OpenAI | gpt-oss-20b | `openai.gpt-oss-20b-1:0` | us-west-2 | Standard (implicit) | $0.07 | $0.30 |
| DeepSeek | DeepSeek V3.2 | `deepseek.v3.2` | us-west-2 | Standard (implicit) | $0.62 | $1.85 |
| Qwen | Qwen3 Coder 480B A35B | `qwen.qwen3-coder-480b-a35b-v1:0` | us-west-2 | Standard (implicit) | $0.22 | $1.80 |
| NVIDIA | Nemotron 3 Super 120B A12B | `nvidia.nemotron-super-3-120b` | us-west-2 | Standard (implicit) | $0.15 | $0.65 |

Prices are snapshots checked 2026-08-29 (the gpt-oss-20b and Qwen3 Coder
output prices were corrected on re-check — the original figures did not match
the AWS pricing page). They change by region and service tier; refresh the
CSV from the [Amazon Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/)
before relying on it for a budget.

## Estimate a particular invocation

Use Bedrock's `usage.inputTokens` and `usage.outputTokens` values:

```bash
.venv/bin/python scripts/bedrock_cost.py \
  zai.glm-5 1200 300
```

The calculator reads the local CSV and outputs the USD estimate. The smoke test
does this automatically for every live probe.

Before changing model IDs, check the current AWS model cards for
[GLM 5](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-zai-glm-5.html),
[gpt-oss-20b](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-20b.html),
[DeepSeek V3.2](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html),
[Qwen3 Coder 480B](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-qwen-qwen3-coder-480b-a35b-instruct.html),
and [Nemotron 3 Super 120B](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-nvidia-nemotron-super-3-120b.html).
