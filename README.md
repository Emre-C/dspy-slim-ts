# dspy-slim-ts

An opinionated TypeScript port of `dspy-slim`, which itself is a slimmed down version of the upstream infamous DSPy framework.

## Breaking changes (v0.2.0)

`RLM` is rewritten end-to-end. The REPL / `eval` / `node:vm` runtime is gone; in its place is a typed functional runtime built on four first-class pillars. Consumers of any `v0.1.x` tarball will see compilation errors on their first install after upgrading; migration is mechanical and worth it.

What changed:

- `RLM.forward()` **throws** `RuntimeError('RLM is async-only. Use acall() or aforward() instead.')`, matching `LM`. Use `await rlm.aforward({ ... })`.
- `NodeCodeInterpreter`, `createNodeCodeInterpreter`, and the REPL types (`REPLHistory`, `REPLEntry`, `REPLEntryKind`, `REPLVariable`, `CodeSession`, `CodeInterpreter`, `ExecuteRequest`, `ExecuteResult`, `InterpreterPatch`, `CodeInterpreterError`, `BudgetVector`, `LLMQueryRequest`, `LLMQueryResult`, `RLMConfig`) are **removed** from the public surface.
- The new public surface is the typed combinator runtime (`split`, `peek`, `map`, `filter`, `reduce`, `concat`, `cross`, `vote`, `ensemble`, `oracle`, `lit`, `vref`, `fn`, `bop`), the evaluator (`evaluate`, `buildEvaluationContext`), the planner (`resolvePlan`, `classifyTask`, `STATIC_PLANS`), the effects runtime (`Effect`, `EffectHandler`, built-in handlers, `parseOracleResponse`), and the typed memory primitives (`MemorySchema`, `applyMemoryWrite`, `defaultMemoryInjector`).
- `RLMOptions` is reshaped: `{ budget, taskType, subLm, lmRegistry, handlers, plans, trackTrace }`. No REPL-era fields remain.
- `GEPA`'s `PredictorTrace.history: REPLHistory | null` is replaced with `executionTrace: readonly EvaluationTrace[] | null`. GEPA is no longer RLM-coupled at the type level.

Why:

The v1 contract embedded a REPL-shaped execution model that conflated synchronous and asynchronous LM usage, required an `eval` / `vm` sandbox, leaked string-parsed control flow into every integration, and allowed unbounded free-form memory. The v0.2.0 contract is: typed AST, deterministic planner, structured effects, typed memory, async-only. The vision is in [docs/RLM_V2_VISION.md](docs/RLM_V2_VISION.md); the architecture decision record is [docs/product/rlm-v2-architecture.md](docs/product/rlm-v2-architecture.md); the implementation plan is in [docs/RLM_V2_IMPLEMENTATION_PLAN.md](docs/RLM_V2_IMPLEMENTATION_PLAN.md).

Migration sketch:

```ts
// v0.1.x
const rlm = new RLM(sig);
const pred = rlm.forward({ context: ctx, question: q });

// v0.2.x
const rlm = new RLM(sig);
const pred = await rlm.aforward({ context: ctx, question: q });
```

No migration shim is provided; the v1 surface is deleted, not deprecated.

## What this repo optimizes for

- Small, readable abstractions
- Strict type safety over convenient looseness
- File-per-concept layout
- Stable names for public concepts and internal boundaries
- Spec fidelity over compatibility theater

## Spec tiers

Each tier is complete. Scope by tier:

- **Tier 1 — Minimal viable port:** Signature, Field, Example, Prediction, Module, Predict, ChainOfThought, Adapters, LM, Settings
- **Tier 2 — Agent workflows:** Tool, ToolCalls, ReAct, History, native function calling
- **Tier 3 — Optimization:** Parallel executor, Callback system, GEPA wrapper, Evaluate
- **Tier 4 — Full parity:** RLM scaffold, NodeCodeInterpreter, budget vectors

## Primary source files

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

## Reading order

1. `../spec/abstractions.md`
2. `docs/product/README.md`
3. `tests/*.test.ts`
4. `src/index.ts`

## Development

This package uses pnpm (`pnpm-lock.yaml`).

```sh
pnpm run build         # compile TypeScript
pnpm test              # run all tests (vitest)
pnpm run typecheck     # type-check without emitting
pnpm run pack:dry-run  # tarball file list (pnpm)
pnpm run release:gate  # full gate via ../benchmarks/release_gate.py
```

Keep changes small, explicit, and grounded in the spec.

## Release gate

CI combines these steps:

- `pnpm run build`
- `pnpm run typecheck`
- `pnpm test`
- `./benchmarks/run_benchmarks.sh --all`
- `pnpm run release:gate` (runs `uv run ../benchmarks/release_gate.py`)

The gate checks replay fixtures (`react_replay`, `react_recorded`, `rlm_replay`), a golden corpus with minimum dataset sizes (`gsm8k=50`, `squad=25`, `hotpotqa=25`, 100+ examples total), Node 20/22/24 in CI, and a curated publish tarball (`dist` plus package metadata and docs).

The automated pack step uses `npm pack --dry-run --json` while local inspection uses pnpm; the JSON shapes differ, so the split is intentional. Details: [docs/product/release-gate-pack-tooling.md](docs/product/release-gate-pack-tooling.md).
