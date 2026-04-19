# RLM v2 Architecture

Single canonical document for RLM v2: motivation, product decision, module map, and numbered sections (**§0.6**, **§2.1**–**§2.6**, **§4**) that source comments reference. The formal runtime clause lives in `../../spec/abstractions.md §0.5`; TypeScript shapes in `src/` are the source of truth for APIs.

---

## Context

The friction around sync `forward` vs async `aforward` came from **open-ended, string-based REPL execution**: when orchestration is LLM-generated imperative code run under `eval()` / `vm`, the host cannot map that model cleanly onto a native async runtime.

RLM v2 abandons that paradigm. Research lines that converge here:

- *The Y-Combinator for LLMs: Solving Long-Context Rot with λ-Calculus* (arXiv:2603.20105) — typed combinators and deterministic planning instead of LLM-written control code.
- *Enabling RLM Inference with Shared Program State* (Cheng) — effects-handler structured tool use instead of string-parsed REPL tools.
- *Sparse Signal Loop* (Stochi) — opinionated system reinjection and constrained memory instead of unbounded scratch files.

---

## Decision

`RLM` is a typed functional runtime with the LLM restricted to the role of a bounded leaf oracle. It ships async-only, under the existing `RLM` name, with four first-class runtime pillars:

1. A typed combinator AST evaluated by a deterministic TypeScript walker.
2. A deterministic planner and task router that compute decomposition parameters and pick a plan before neural execution (aside from an optional one-shot classifier).
3. An effects-handler pattern: oracle leaves yield typed `Effect` values; named `EffectHandler`s dispatch and re-prompt.
4. A typed memory schema reinjected into the oracle system message on every turn, replacing free-form scratchpads.

`eval`, `node:vm`, and the REPL runtime are gone. `forward()` throws `RuntimeError`; `aforward({ ... })` is the public entrypoint.

---

## Why

The v1 RLM exposed a blank terminal and relied on LLM-generated imperative code for recursion. In the TypeScript port that led to:

- A synthetic sync/async split driven by the REPL bridge.
- Control flow the host could not verify or optimize (string parsing).
- Unconstrained file-style memory that could bloat context and overfit evaluators.

The four pillars address those categories by design, not by prompt engineering. **Quality is the objective function; cost is a constraint.** Self-consistency and ensemble paths are first-class; default budgets are generous; callers tighten when they need to.

---

## Architecture layers

Every layer is async-native. Control flow below is deterministic TypeScript except where noted.

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │ RLM (src/rlm.ts) — facade; aforward only                               │
 └─────────────┬──────────────────────────────────────────────────────────┘
               │
 ┌─────────────▼──────────────────────────────────────────────────────────┐
 │ Task router (src/rlm_task_router.ts) — classify → plan(s); beam        │
 └─────────────┬──────────────────────────────────────────────────────────┘
               │
 ┌─────────────▼──────────────────────────────────────────────────────────┐
 │ Planner (src/rlm_planner.ts, src/rlm_planner_quality.ts) — k*, d, N   │
 └─────────────┬──────────────────────────────────────────────────────────┘
               │
 ┌─────────────▼──────────────────────────────────────────────────────────┐
 │ Evaluator (src/rlm_evaluator.ts) + combinators (src/rlm_combinators.ts)│
 └─────────────┬──────────────────────────────────────────────────────────┘
               │
 ┌─────────────▼──────────────────────────────────────────────────────────┐
 │ Effects (src/rlm_effects.ts) + memory (src/rlm_memory.ts)              │
 └────────────────────────────────────────────────────────────────────────┘
```

The LLM participates at most in: (1) optional task classification, (2) oracle leaves (`oracle` / `vote` / `ensemble`), and (3) effect re-prompts inside the oracle loop.

---

## What this means in code

**Facade — `src/rlm.ts`.** Resolves `BaseLM`, builds the prompt, classifies or uses `taskType`, routes plans (including beam), runs `resolvePlan` then `evaluate`, returns `Prediction` with optional diagnostics (`_rlm_*` fields). `forward()` throws the async-only `RuntimeError`.

**Pillar 1 — `src/rlm_combinators.ts`, `src/rlm_evaluator.ts`.** `CombinatorNode` is a closed sum type; plans are data. The evaluator walks the AST with `Promise.all` for parallel nodes, bounded by `maxParallelism`; `Reduce` is sequential; `Oracle` runs the bounded effect loop and emits trace rows (including virtual `'effect'` steps — see `src/rlm_trace_types.ts`).

**Pillar 2 — `src/rlm_planner.ts`, `src/rlm_planner_quality.ts`, `src/rlm_task_router.ts`.** Planner is pure over `(taskType, prompt length, budget)` and quality curves. Router performs the optional classifier call and maps to a single plan or a beam. Overrides: `routeThreshold`, `routeBeamTopK`, `classifier` on `RLMOptions`.

**Pillar 3 — `src/rlm_types.ts` (`Effect`, `EffectHandler`, `OracleResponse`), `src/rlm_effects.ts`.** Oracle wire format and built-in handler factories; handler registry merges user handlers. Loop capped by `budget.maxEffectTurns`.

**Pillar 4 — `src/rlm_memory.ts`.** `MemorySchema`, validated writes, `defaultMemoryInjector` banner into the oracle system message.

**Removed surfaces.** No `node:vm`. `pnpm run deps:rlm-legacy --strict` (and the release gate) guard against resurrecting v1 REPL symbols.

---

### 0.6 Quality objective and default budget

RLM v2 does not optimize a uniform cost-per-leaf λ-RLM paper objective. It targets **expected quality under a generous ceiling**: self-consistency width, ensemble-capable registry, beam routing when classification is uncertain, trace retention by default, and high default caps. The numeric defaults and their intent are documented on `DEFAULT_BUDGET` in `src/rlm_types.ts` (see **§2.2**).

---

### 2. Type contracts (comment index)

Subsections **§2.1**–**§2.6** are stable anchors for `src/` doc comments. **Do not treat this markdown as a verbatim mirror of types** — use the referenced modules.

#### 2.1 Combinator AST (`src/rlm_combinators.ts`)

Typed constructors (`split`, `map`, `oracle`, `lit`, `vref`, …) build a serializable `CombinatorNode` tree. `CombinatorValue` includes `JsonObject` for structured payloads flowing through the evaluator.

#### 2.2 Budget and execution context (`src/rlm_types.ts`, `src/rlm_trace_types.ts`)

`RLMBudget`, `DEFAULT_BUDGET`, `mergeBudget`. `EvaluationContext` threads `budget`, LMs, `signature`, `scope`, `depth`, `callsUsed`, `trace`, `handlers`, `memorySchema`, and a **`memoryCell`** holding the current `TypedMemoryState` snapshot for in-run writes. `EvaluationTrace` is defined in `rlm_trace_types.ts` so GEPA can depend on traces without importing the full RLM substrate.

#### 2.3 Task routing and planning (`src/rlm_task_router.ts`, `src/rlm_planner.ts`)

`TaskType`, `StaticPlan`, `ClassifierResult`, route/beam resolution, and `resolvePlan` inputs/outputs. Planner quality curves live in `rlm_planner_quality.ts`.

#### 2.4 Effects protocol (`src/rlm_types.ts`, `src/rlm_effects.ts`)

`Effect` union, `EffectResult`, `OracleResponse`, `EffectHandler` protocol; `parseOracleResponse` and `EFFECT_ORACLE_SIGNATURE`; built-in handler factories and registration helpers in `rlm_effects.ts`.

#### 2.5 Typed memory (`src/rlm_memory.ts`)

`MemorySchema`, `MemoryFieldSchema`, `TypedMemoryState`, `initialMemoryState`, `applyMemoryWrite`, `defaultMemoryInjector`.

#### 2.6 RLM public surface (`src/rlm.ts`)

`RLMOptions` (budget, `taskType`, `subLm`, `lmRegistry`, `handlers`, `plans`, `trackTrace`, routing/classifier knobs) and the `RLM` `Module` subclass with `aforward` / throwing `forward`.

---

### 4. Execution model (comment index)

#### 4.1 Combinator evaluation (`src/rlm_evaluator.ts`)

`evaluate(plan, ctx)` async-reduces the AST, enforces budgets, records `EvaluationTrace`, runs the oracle effect loop, and routes `vote` / `ensemble` / verifier reducers. See also `spec/abstractions.md §0.5`.

#### 4.2 Planning (`src/rlm_planner.ts`)

Pure resolution of template plans: partition size, depth, self-consistency width, and literal substitution into planner placeholders — no network.

#### 4.3 Task routing (`src/rlm_task_router.ts`)

Optional classifier call, registry of `StaticPlan`s, `resolveRoute` (single plan vs beam), and integration with the planner output.

---

## Out of scope (v0.2.x)

- Nested self-recursive `RLM` entry from inside plans (future: elevate `QueryOracle` or similar).
- Streaming partial outputs from `aforward`.
- Cross-call durable memory (memory is per `aforward` unless the caller layers persistence).
- Adaptive mid-run re-planning after execution starts.
- Auto-discovery for `lmRegistry` (callers supply the map).

---

## Verification

- Unit and integration tests: `tests/rlm_*.test.ts`, `tests/gepa.test.ts`.
- Release gate: `pnpm run release:gate` (includes legacy-symbol guard for v1 REPL surface).
- Runnable examples: `examples/rlm_quickstart.ts`, `examples/rlm_custom_effect.ts`.

---

## Revisit if

- Self-consistency or ensemble defaults measurably hurt quality on representative benchmarks.
- Structured parsing of oracle completions becomes a bottleneck vs native tool-calling — likely a localized change in `rlm_effects.ts` / adapters.
- Typed memory is too narrow for a major task family — extend schemas, do not reintroduce free-form REPL memory.
- Parallel `Promise.all` fan-out routinely hits rate limits — may need scheduler logic in the evaluator, not a sync `forward` path.
- Nested recursive RLM becomes a requirement — treat as a versioned feature with an explicit entrypoint from effects or plans.
