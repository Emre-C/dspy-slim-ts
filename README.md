# dspy-slim-ts

A craftsmanship-first TypeScript port of `dspy-slim`.

All four spec tiers are implemented and enforced in CI. The formal
specification in `../spec/abstractions.md` is the source of truth.
The product decision records under `docs/product/` explain why the
implementation is shaped the way it is. The test suite in `tests/` is
fixture-driven and documents the runtime contract in executable form.

## What This Repo Optimizes For

- Small, readable abstractions
- Strict type safety over convenient looseness
- File-per-concept layout
- Stable names for public concepts and internal boundaries
- Spec fidelity over compatibility theater

## Spec Tier Status

| Tier | Scope | Status |
|------|-------|--------|
| 1 — Minimal viable port | Signature, Field, Example, Prediction, Module, Predict, ChainOfThought, Adapters, LM, Settings | ✅ Complete |
| 2 — Agent workflows | Tool, ToolCalls, ReAct, History, native function calling | ✅ Complete |
| 3 — Optimization | Parallel executor, Callback system, GEPA wrapper, Evaluate | ✅ Complete |
| 4 — Full parity | RLM scaffold, NodeCodeInterpreter, budget vectors | ✅ Complete |

## Primary Source Files

- `src/field.ts` — immutable field construction and validation
- `src/signature.ts` — signature value objects, parsing, and ops
- `src/example.ts` — data container with input/label splitting
- `src/prediction.ts` — completions container with numeric protocol
- `src/module.ts` — runtime module traversal and predictor discovery
- `src/predict.ts` — orchestration, defaults, LM resolution, and traces
- `src/chain_of_thought.ts` — CoT wrapper over Predict
- `src/adapter.ts` — message formatting and LM response parsing
- `src/lm.ts` — OpenAI-compatible LM client
- `src/settings.ts` — async-local configuration state
- `src/tool.ts` — tool wrappers and tool-call normalization
- `src/react.ts` — the ReAct agent loop
- `src/parallel.ts` — bounded concurrent executor
- `src/evaluate.ts` — typed evaluation with metric scoring
- `src/callback.ts` — callback dispatch and nested call tracking
- `src/rlm.ts` — recursive language model scaffold
- `src/gepa.ts` — GEPA optimization wrapper

## Reading Order

1. `../spec/abstractions.md`
2. `docs/product/README.md`
3. `tests/*.test.ts`
4. `src/index.ts`

## Development

```sh
npm run build      # compile TypeScript
npm test           # run all tests (vitest)
npm run typecheck  # type-check without emitting
npm run release:gate  # verify release artifacts + npm publish surface
```

Keep changes small, explicit, and grounded in the spec.

## Release Gate

Public release readiness is machine-checked in CI by combining:

- `npm run build`
- `npm run typecheck`
- `npm test`
- `./benchmarks/run_benchmarks.sh --all`
- `uv run benchmarks/release_gate.py`

The release gate enforces:

- replay fixture presence and validity (`react_replay`, `react_recorded`, `rlm_replay`)
- a 3-dataset golden corpus with minimum counts (`gsm8k=50`, `squad=25`, `hotpotqa=25`) and `100+` total examples
- Node runtime matrix coverage (20/22/24) in CI
- a curated npm tarball surface (dist artifacts + package metadata/docs only)
