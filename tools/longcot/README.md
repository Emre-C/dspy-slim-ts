# LongCoT bridge (Python)

This folder is a tiny [uv](https://docs.astral.sh/uv/) project that depends on the upstream [LongCoT](https://github.com/LongHorizonReasoning/longcot) package so we can load questions and run deterministic `verify()` from the benchmark.

## Setup

From this directory:

```bash
uv sync
```

## Scripts

- `export_questions.py` — writes one JSON object per line (stdout) with the fields needed to reconstruct a `Question` and to feed RLM (`prompt`, `problem`, `answer`, …).
- `score_responses.py` — reads a JSONL file produced by `tools/bench_longcot_rlm.ts` and prints aggregate accuracy (same notion as LongCoT’s `run_eval.py`).

## Environment

- `HF_TOKEN` — Hugging Face token for the OpenAI-compatible router (used by the TypeScript runner, not these scripts).
- For math/chemistry **fallback** judges inside `verify()`, LongCoT may use Gemini; set `GEMINI_API_KEY` or `GOOGLE_API_KEY`, or pass `--no-fallback` to `score_responses.py`.
