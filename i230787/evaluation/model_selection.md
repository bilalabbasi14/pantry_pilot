# Model selection rationale

Evidence: `results/model_comparison_summary.md`, `results/model_comparison_raw.json`, `results/model_comparison.csv`
(produced by `model_comparison.py`, run 2026-10-04, same 10 inputs per model, provider fallback disabled,
45 s pause between cases, temperature 0, max 1024 output tokens).

## Candidates and why

| Candidate | Why it was considered |
|---|---|
| `gemini:gemini-3.5-flash-lite` | Small, fast Gemini tier with JSON response mode; free-tier quota probe accepted 16 back-to-back requests without a 429. |
| `groq:openai/gpt-oss-120b` | Strong open-weight model on low-latency Groq hardware with JSON mode; cheap paid pricing. |
| `gemini:gemini-2.5-flash` | Established structured-output model; included as a third reference point. |

Larger "pro" tiers were not considered: the task is a short structured decision per step, and the
40 s run budget rewards latency over reasoning depth.

## Measured results (10 identical inputs)

| Model | Task success | Structured validity | Correct first action | Median / P90 latency | Avg tokens in / out per run | Est. cost, 10 runs |
|---|---|---|---|---|---|---|
| gemini-3.5-flash-lite | 10/10 | 17/17 decisions | 10/10 | 3654 / 5360 ms | 2988 / 138 | $0.0124 |
| groq gpt-oss-120b | 10/10 | 18/18 decisions | 10/10 | 3032 / 4608 ms | 3152 / 282 | $0.0068 |
| gemini-2.5-flash | 4/10 | 8/8 decisions | 3/10 | 6175 / 7743 ms | 3462 / 166 (4 runs) | $0.0058 (4 runs) |

Costs are estimates from token counts × paid list prices (`app/llm.py: PRICING_PER_MILLION`); the keys
used are free tier, so nothing was actually billed. Gemini 3.5 Flash-Lite pricing comes from third-party
price trackers, not an official Google page, so treat that row as approximate.

## Observed operational limits (from provider responses, not assumptions)

- Groq free tier: response headers report `x-ratelimit-limit-tokens: 8000` per minute per model. One agent
  run uses about 1.6k–5.6k input tokens, so sustained throughput is about 1–3 runs per minute. An earlier unpaced
  smoke test hit HTTP 429 on the 2nd and 3rd consecutive runs.
- Gemini 2.5 Flash free tier: the 429 body reports `generate_content_free_tier_requests, limit: 5` (per minute).
  In the experiment, 6 of 10 runs failed with `model_rate_limited` (the agent stopped safely with status `failed`).
- Gemini 3.5 Flash-Lite: no 429 in the experiment or in a 16-request burst probe.

## Decision

**Default: `gemini:gemini-3.5-flash-lite`; fallback/alternate: `groq:openai/gpt-oss-120b`.**

- Quality was tied: both had perfect task success, contract validity and first-action choice on the same inputs.
- Groq was about 17% faster (median) and about 45% cheaper at paid list prices, and would be the choice on a paid Groq plan.
- With the free-tier keys this deployment actually has, Flash-Lite's throughput headroom matters most. The Arena
  evaluator sends bursts of requests, and an 8k tokens/min cap would turn a correct agent into `failed` runs.
  Reliability under load outweighs a 0.6 s median latency gain.
- Flash-Lite also used about half the output tokens (no hidden reasoning tokens with `thinkingLevel: minimal`).
- Keeping Groq as an automatic fallback (`ENABLE_MODEL_FALLBACK=true`) gives a second, independent quota if Gemini
  returns 429 or 5xx. The model that actually answered is recorded in the `model_call` event.

The choice is configuration, not code: `MODEL_PROVIDER`, `MODEL_NAME`, `ALTERNATE_MODELS`.
