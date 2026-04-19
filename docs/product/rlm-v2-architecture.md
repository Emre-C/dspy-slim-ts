# RLM v2 Architecture

## Decision

`RLM` is a typed functional runtime with the LLM restricted to the role of
a bounded leaf oracle. It ships async-only, under the existing `RLM` name,
with four first-class runtime pillars:

1. a typed combinator AST evaluated by a deterministic TypeScript walker
2. a deterministic planner and task router that compute decomposition math
   and pick a plan before any LLM call
3. an effects-handler pattern that replaces string-parsed tool use with
   typed `Effect` intents dispatched by named handlers
4. a typed memory schema reinjected into the oracle system message at
   every call, replacing free-form scratchpad files

`eval`, `node:vm`, and the REPL runtime are gone. `forward()` throws
`RuntimeError`; `aforward({ ... })` is the only public entrypoint.

## Why

The v1 RLM gave the LLM a blank terminal and asked it to write recursive
Python-style code to decompose tasks. The paradigm was inherited from the
Python port and led to three classes of problem in the TS port:

- A synthetic sync/async split driven entirely by the REPL's choice of
  running `eval` on generated code. Every real LLM is async, but the
  REPL's imperative-code paradigm wanted a synchronous bridge, which is
  exactly what `docs/product/sync-lm-removal.md` removed from `LM` and
  left dangling on `RLM`.
- String-parsed control flow that the host TypeScript runtime could not
  verify, optimize, or reason about.
- Unconstrained file-based memory (`skill_file.txt` in upstream DSPy's
  documentation) that bloated, plateaued, and overfit to the judge
  rather than improving task accuracy.

The four pillars address one category each, by design rather than by
prompt engineering. Three papers converged on the same answers:

- *The Y-Combinator for LLMs: Solving Long-Context Rot with
  λ-Calculus* (arXiv:2603.20105) — typed combinators with deterministic
  planning.
- *Enabling RLM Inference with Shared Program State* (Cheng) — the
  effects-handler pattern.
- *Sparse Signal Loop* (Stochi) — opinionated memory reinjection over
  free-form logs.

Quality is the objective function; cost is a constraint, not a target.
Self-consistency is on by default. Ensemble oracles are supported
natively. The library spends calls to improve quality; callers tighten
budgets when they need to.

## What This Means In Code

- `src/rlm.ts` is a thin facade. `aforward({ ... })` classifies, plans,
  evaluates, and returns a `Prediction`. `forward()` throws
  `RuntimeError('RLM is async-only. Use acall() or aforward() instead.')`.
- `src/rlm_combinators.ts` defines the `CombinatorNode` AST. Plans are
  data, not closures. The planner substitutes literal values into
  `vref()` placeholders.
- `src/rlm_evaluator.ts` walks the AST. `Map`, `Vote`, and `Ensemble`
  run via `Promise.all` bounded by `maxParallelism`. `Reduce` is
  sequential `await` accumulation. `Oracle` drives the effects loop.
- `src/rlm_planner.ts` computes `k*`, `d`, and `N` per task type with a
  quality-shaped loss surface. Pure function; no network.
- `src/rlm_task_router.ts` classifies the task (at most one LM call) and
  picks a plan from the static registry. Low classifier confidence
  triggers beam routing via `Cross` over the top-K plans.
- `src/rlm_effects.ts` defines the `Effect` union, the `EffectHandler`
  protocol, and the built-in handlers (`ReadContext`, `WriteMemory`,
  `QueryOracle`, `Search`, `Yield`, `Custom`). The oracle-leaf loop is
  bounded by `maxEffectTurns`.
- `src/rlm_memory.ts` defines `MemorySchema` and the system-reinjection
  injector. Writes go through `applyMemoryWrite`, which validates type
  and length against the schema.
- `src/node_code_interpreter.ts` is deleted. `src/` contains zero imports
  of `node:vm`. `pnpm run deps:rlm-legacy` enforces this from Phase 9
  onwards.

## Revisit If

- Self-consistency or ensemble strategies are found to degrade quality
  on a meaningful fraction of real workloads (benchmarks would signal).
- Structured-output parsing of `OracleResponse` becomes a bottleneck
  relative to native function-calling adapters — likely a local upgrade
  inside `src/rlm_effects.ts`, not a runtime redesign.
- Typed memory proves too narrow for a task family we want to support —
  add a new `MemorySchema`, do not reintroduce free-form memory.
- `Promise.all` fan-out fails to saturate rate-limited LMs under
  realistic workloads — likely needs a smarter scheduler inside the
  evaluator, not a sync fallback on the facade.
- Nested self-recursive RLM becomes a requirement — v0.3.0 target; the
  `QueryOracle` effect becomes a first-class recursive entrypoint rather
  than a single-call delegate.
