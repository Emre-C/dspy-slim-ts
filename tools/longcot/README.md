# LongCoT bridge (Python)

This folder is a tiny [uv](https://docs.astral.sh/uv/) project that depends on the upstream [LongCoT](https://github.com/LongHorizonReasoning/longcot) package so we can load questions and run deterministic `verify()` from the benchmark.

## Setup

From this directory:

```bash
uv sync
```

## Cost-safe order

1. **`pnpm run bench:longcot:preflight`** (repo root) — one ~48-token Hugging Face completion; confirms `HF_TOKEN` / router / model id. Uses repo-root `.env` if `HF_TOKEN` is not exported.
2. **`pnpm run bench:longcot:smoke`** — one **easy** LongCoT question end-to-end with **hard caps** and **`--runner predict`** by default. For **`--runner rlm`**, the bench configures **`ChatAdapter`** so the RLM’s final `answer` string stays in natural language (what `verify()` parses), not JSON scaffolding.
3. **Scale up** — e.g. `pnpm run bench:longcot -- --runner predict --max 5 --domain logic --difficulty easy`. LongCoT-Mini is the **easy** slice (~100 per domain × 5 domains); this repo’s exporter is per `--domain` / `--difficulty`, so a full mini sweep is five domains × `--difficulty easy` (or extend `export_questions.py`). Runs with `--max > 20` or very high limits require **`--i-accept-cost`**.
4. **Predict vs our RLM (same LM, same keys)** — `pnpm run bench:longcot:compare -- --domain logic --difficulty easy --max 3` runs **single-shot `LM.acall`** and **RLM v2** on identical questions, writes two JSONLs, scores both with `score_responses.py`, and prints a JSON summary (`delta_correct`, accuracy delta). Add **`--i-accept-cost`** under the same limits as the main bench (each question does **two** pipelines: predict + RLM).

### Versus upstream LongCoT `run_inference.py`

The [official repo](https://github.com/LongHorizonReasoning/longcot) runs **`uv run python run_inference.py`** with YAML configs — that is the **vendor baseline** (their prompts, parallelism, configs). This TypeScript repo measures **our** `LM` + optional **RLM v2** against the **same** `longcot.verify()` via `score_responses.py`. To compare “TS RLM vs vendor single-model completion,” align **model/provider** and **question slice** (e.g. same `--domain` / `--difficulty easy` as `--difficulty longcot-mini` upstream), then compare **accuracy** numbers from `run_inference.py` outputs vs our JSON summary — not line-for-line outputs, unless you wire the same model string on both sides.

**Hugging Face router:** long-horizon LongCoT prompts can run for many minutes. The hosted gateway may return **504 HTML error pages** under load; that shows up as a row `error` in the JSONL, not as a TypeScript stack trace. Retry with a smaller `--max`, a lighter domain first, or a provider with a higher server-side timeout.

## Scripts

- `export_questions.py` — writes one JSON object per line (stdout) with the fields needed to reconstruct a `Question` and to feed RLM (`prompt`, `problem`, `answer`, …).
- `score_responses.py` — reads a JSONL file produced by `tools/bench_longcot_rlm.ts` or `tools/compare_longcot_predict_rlm.ts` and prints aggregate accuracy (same notion as LongCoT’s packaged eval).

## Environment

- `HF_TOKEN` — Hugging Face token for the OpenAI-compatible router (used by the TypeScript runner, not these scripts). The runner also reads repo-root `.env` if `HF_TOKEN` is not already set in the environment.
- For math/chemistry **fallback** judges inside `verify()`, LongCoT may use Gemini; set `GEMINI_API_KEY` or `GOOGLE_API_KEY`, or pass `--no-fallback` to `score_responses.py`.
