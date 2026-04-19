# dspy-slim-ts

An opinionated TypeScript port of `dspy-slim`, which itself is a slimmed down version of the upstream infamous DSPy framework.

## RLM v2 — 60-second quickstart

RLM v2 is a typed functional runtime with the LLM restricted to the role of a bounded leaf oracle. Four pillars: a typed combinator AST, a deterministic planner and task router, an effects-handler tool-use protocol, and a typed memory schema reinjected into the oracle system message. `aforward` is the only entry point; the REPL / `eval` / `node:vm` runtime is gone.

```ts
import { RLM, settings, LM } from 'dspy-slim-ts';

settings.configure({
  lm: new LM({
    model: 'openrouter/openai/gpt-4o-mini',
    apiKey: process.env.OPENROUTER_API_KEY,
    apiBase: 'https://openrouter.ai/api/v1',
  }),
});

const rlm = new RLM('question: str, context: str -> answer: str', {
  taskType: 'search', // skip the classifier when the caller knows the family
});

const prediction = await rlm.aforward({
  question: 'Where is the secret item?',
  context: 'The shelf in the blue room holds a vase of roses and a small owl figurine.',
});

console.log(prediction.getOr('answer', ''));
```

End-to-end runnable examples: [`examples/rlm_quickstart.ts`](examples/rlm_quickstart.ts) and [`examples/rlm_custom_effect.ts`](examples/rlm_custom_effect.ts). Both fall back to a scripted LM when `OPENROUTER_API_KEY` is unset, so `npx tsx examples/rlm_quickstart.ts` runs from a clean checkout.

## Breaking changes (v0.2.0)

`RLM` is rewritten end-to-end. The REPL / `eval` / `node:vm` runtime is gone; in its place is the typed functional runtime described above. Consumers of any `v0.1.x` tarball will see compilation errors on their first install after upgrading; migration is mechanical and worth it.

What changed:

- `RLM.forward()` **throws** `RuntimeError('RLM is async-only. Use acall() or aforward() instead.')`, matching `LM`. Use `await rlm.aforward({ ... })`.
- The v1 REPL runtime (`NodeCodeInterpreter`, `createNodeCodeInterpreter`) and its associated types (`REPLHistory`, `REPLEntry`, `REPLEntryKind`, `REPLVariable`, `CodeSession`, `CodeInterpreter`, `ExecuteRequest`, `ExecuteResult`, `InterpreterPatch`, `CodeInterpreterError`, `BudgetVector`, `LLMQueryRequest`, `LLMQueryResult`, `RLMConfig`) are **removed** from the public surface. `pnpm run deps:rlm-legacy --strict` fails the release gate if any of them reappear under `src/`.
- The new public surface is the typed combinator runtime (`split`, `peek`, `map`, `filter`, `reduce`, `concat`, `cross`, `vote`, `ensemble`, `oracle`, `lit`, `vref`, `fn`, `bop`), the evaluator (`evaluate`, `buildEvaluationContext`), the planner (`resolvePlan`, `classifyTask`, `STATIC_PLANS`), the effects runtime (`Effect`, `EffectHandler`, built-in handlers, `parseOracleResponse`), and the typed memory primitives (`MemorySchema`, `applyMemoryWrite`, `defaultMemoryInjector`).
- `RLMOptions` is reshaped: `{ budget, taskType, subLm, lmRegistry, handlers, plans, trackTrace, routeThreshold, routeBeamTopK, classifier }`. No REPL-era fields remain.
- `GEPA`'s `PredictorTrace.history` is replaced with `executionTrace: readonly EvaluationTrace[] | null`. GEPA is no longer RLM-coupled at the type level.

Why:

The v1 contract embedded a REPL-shaped execution model that conflated synchronous and asynchronous LM usage, required an `eval` / `vm` sandbox, leaked string-parsed control flow into every integration, and allowed unbounded free-form memory. The v0.2.0 contract is: typed AST, deterministic planner, structured effects, typed memory, async-only. Rationale, module map, and API index: [docs/product/rlm-v2-architecture.md](docs/product/rlm-v2-architecture.md).

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
- **Tier 4 — Full parity:** RLM v2: typed combinator runtime; async-only; structured tool use; typed memory

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
- `src/rlm.ts` — RLM v2 public facade (async-only `aforward`)
- `src/rlm_combinators.ts` — typed AST: `split`, `map`, `oracle`, `vote`, `ensemble`, `concat`, `cross`, `lit`, `vref`, `fn`, `bop`
- `src/rlm_evaluator.ts` — deterministic walker; effects loop; trace capture
- `src/rlm_planner.ts` + `src/rlm_planner_quality.ts` — `(k, n, budget) → resolved plan` with task-family quality curves
- `src/rlm_task_router.ts` — classifier, six static plans, beam routing
- `src/rlm_effects.ts` — `Effect` union, oracle signature, built-in handlers
- `src/rlm_memory.ts` — `MemorySchema`, `applyMemoryWrite`, system-reinjection banner
- `src/gepa.ts` — GEPA optimization wrapper (decoupled from RLM internals via `PredictorTrace.executionTrace`)

## Reading order

1. `examples/rlm_quickstart.ts` — the 60-second path for RLM v2
2. `docs/product/rlm-v2-architecture.md` — RLM v2 decision record, layers, and §2 contract index
3. `../spec/abstractions.md`
4. `docs/product/README.md`
5. `tests/*.test.ts`
6. `src/index.ts`

## Development

This package uses pnpm (`pnpm-lock.yaml`).

```sh
pnpm run build         # compile TypeScript
pnpm test              # run all tests (vitest)
pnpm run typecheck     # type-check without emitting
pnpm run pack:dry-run  # tarball file list (pnpm)
pnpm run release:gate  # full gate via ../benchmarks/release_gate.py
pnpm run bench:longcot:preflight  # tiny HF ping only (.env HF_TOKEN); run first
pnpm run bench:longcot:smoke      # one LongCoT question, hard caps; then scale up
pnpm run bench:longcot -- --domain logic --difficulty easy --max 5 --i-accept-cost  # larger (see tools/longcot/README.md)
```

Keep changes small, explicit, and grounded in the spec.

## Release gate

CI combines these steps:

- `pnpm run build`
- `pnpm run typecheck`
- `pnpm test`
- `pnpm run deps:rlm-legacy:strict` (fails if any v1 REPL symbol reappears under `src/`)
- `./benchmarks/run_benchmarks.sh --all`
- `pnpm run release:gate` (runs `uv run ../benchmarks/release_gate.py`)

The gate checks replay fixtures (`react_replay`, `react_recorded`, `rlm_v2_replay`), a golden corpus with minimum dataset sizes (`gsm8k=50`, `squad=25`, `hotpotqa=25`, 100+ examples total), Node 20/22/24 in CI, and a curated publish tarball (`dist` plus package metadata and docs). It also runs `tools/deps_rlm_legacy.mjs --strict` inside the package to enforce that the v1 REPL surface stays deleted.

The automated pack step uses `npm pack --dry-run --json` while local inspection uses pnpm; the JSON shapes differ, so the split is intentional. Details: [docs/product/release-gate-pack-tooling.md](docs/product/release-gate-pack-tooling.md).
