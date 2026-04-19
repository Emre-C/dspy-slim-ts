# LongCoT bridge (Python)

This folder is a tiny [uv](https://docs.astral.sh/uv/) project that depends on the upstream [LongCoT](https://github.com/LongHorizonReasoning/longcot) package so we can load questions and run deterministic `verify()` from the benchmark.

## Setup

From this directory:

```bash
uv sync
```

## Cost-safe order

1. **`pnpm run bench:longcot:preflight`** (repo root) — one ~48-token Hugging Face completion; confirms `HF_TOKEN` / router / model id. Uses repo-root `.env` if `HF_TOKEN` is not exported.
2. **`pnpm run bench:longcot:smoke`** — exactly **one** LongCoT question end-to-end through RLM with **hard caps** (summarise plan, ≤32 oracle calls, ≤2k completion tokens, no Gemini fallback in scoring).
3. **Scale up** — increase `--max`, `--max-completion-tokens`, `--max-oracle-calls` deliberately. Runs with `--max > 20` or very high limits require **`--i-accept-cost`** so a typo does not launch a huge job.

## Scripts

- `export_questions.py` — writes one JSON object per line (stdout) with the fields needed to reconstruct a `Question` and to feed RLM (`prompt`, `problem`, `answer`, …).
- `score_responses.py` — reads a JSONL file produced by `tools/bench_longcot_rlm.ts` and prints aggregate accuracy (same notion as LongCoT’s `run_eval.py`).

## Environment

- `HF_TOKEN` — Hugging Face token for the OpenAI-compatible router (used by the TypeScript runner, not these scripts). The runner also reads repo-root `.env` if `HF_TOKEN` is not already set in the environment.
- For math/chemistry **fallback** judges inside `verify()`, LongCoT may use Gemini; set `GEMINI_API_KEY` or `GOOGLE_API_KEY`, or pass `--no-fallback` to `score_responses.py`.
