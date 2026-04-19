# LongCoT bridge (Python)

This folder is a tiny [uv](https://docs.astral.sh/uv/) project that depends on the upstream [LongCoT](https://github.com/LongHorizonReasoning/longcot) package so we can load questions and run deterministic `verify()` from the benchmark.

## Setup

From this directory:

```bash
uv sync
```

## Cost-safe order

1. **`pnpm run bench:longcot:preflight`** (repo root) — one ~48-token Hugging Face completion; confirms `HF_TOKEN` / router / model id. Uses repo-root `.env` if `HF_TOKEN` is not exported.
2. **`pnpm run bench:longcot:smoke`** — one **easy** LongCoT question end-to-end with **hard caps** and **`--runner predict`** by default (LongCoT expects free-text `solution = …`; full **RLM** oracle leaves need structured JSON). Use `--smoke --runner rlm` only if your LM reliably emits the effect-oracle wire format.
3. **Scale up** — e.g. `pnpm run bench:longcot -- --runner predict --max 5 --domain logic --difficulty easy`. LongCoT-Mini is the **easy** slice (~100 per domain × 5 domains); this repo’s exporter is per `--domain` / `--difficulty`, so a full mini sweep is five domains × `--difficulty easy` (or extend `export_questions.py`). Runs with `--max > 20` or very high limits require **`--i-accept-cost`**.

**Hugging Face router:** long-horizon LongCoT prompts can run for many minutes. The hosted gateway may return **504 HTML error pages** under load; that shows up as a row `error` in the JSONL, not as a TypeScript stack trace. Retry with a smaller `--max`, a lighter domain first, or a provider with a higher server-side timeout.

## Scripts

- `export_questions.py` — writes one JSON object per line (stdout) with the fields needed to reconstruct a `Question` and to feed RLM (`prompt`, `problem`, `answer`, …).
- `score_responses.py` — reads a JSONL file produced by `tools/bench_longcot_rlm.ts` and prints aggregate accuracy (same notion as LongCoT’s `run_eval.py`).

## Environment

- `HF_TOKEN` — Hugging Face token for the OpenAI-compatible router (used by the TypeScript runner, not these scripts). The runner also reads repo-root `.env` if `HF_TOKEN` is not already set in the environment.
- For math/chemistry **fallback** judges inside `verify()`, LongCoT may use Gemini; set `GEMINI_API_KEY` or `GOOGLE_API_KEY`, or pass `--no-fallback` to `score_responses.py`.
