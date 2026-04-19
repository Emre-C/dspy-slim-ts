# RLM v2 — Implementation Plan

Companion to `docs/RLM_V2_VISION.md`. The vision argues *what* RLM v2 is and *why*.
This document specifies *how exactly* we get there, in the order we do it, with
the contracts we commit to and the tests that prove each commitment.

The plan is written against the current repo: `src/rlm.ts` (623 LOC),
`src/node_code_interpreter.ts` (600 LOC), `src/rlm_types.ts`, the shared spec
at `../spec/abstractions.md §0.5`, the fixture at `../spec/fixtures/rlm_replay.json`,
and the tests in `tests/rlm_gepa.test.ts` and `tests/rlm_gepa_automata.test.ts`.

## Product directives (authoritative for every decision below)

1. **The TS port is the canonical source of truth for its spec.** The Python
   port conforms or declares non-conformance in its own docs. We do not
   degrade the TS design to preserve symmetry with an upstream we do not
   control.
2. **Quality is the objective function; cost is a constraint, not a target.**
   Self-consistency on by default. Ensemble oracles wherever plans benefit.
   Beam search over plans when the classifier is uncertain. Budget defaults
   are generous; users tighten them if they need to.
3. **All four runtime pillars ship in v0.2.0.** No multi-release stagger.
   Effects-handler is a runtime, not a type stub. System reinjection is a
   runtime, not a schema placeholder. The v0.2.0 facade is the complete
   vision, minus self-recursive nested RLM which remains explicitly out of
   scope for v0.2.0.

Everything that follows is implementation detail in service of those three
directives.

---

## 0. Scope and defaults

### 0.1 v2 replaces v1 under the existing `RLM` name

The vision is explicit: **"No Backwards Compatibility."** The package is
still at `0.1.0` on disk and unpublished. A clean break is cheap. V2 takes
the `RLM` identifier; there is no `RLMv1`, `LegacyRLM`, or re-exported alias.
The release is cut as `0.2.0` with a BREAKING section in the README and the
changelog.

### 0.2 Async-only, following the `LM` precedent

`RLM.forward` throws `RuntimeError('RLM is async-only. Use acall() or aforward() instead.')`,
matching the behavior `LM.forward` already has per
`docs/product/sync-lm-removal.md`. `aforward({ ... })` is the single public
entrypoint. `call` / `acall` inherit async semantics from `Module.acall`.

### 0.3 `node:vm` and the code interpreter are deleted, not gated

`src/node_code_interpreter.ts` is removed. `vm` is not imported anywhere in
production code. The REPL-shaped types in `src/rlm_types.ts` (`REPLHistory`,
`REPLEntry`, `REPLVariable`, `CodeSession`, `CodeInterpreter`, `ExecuteRequest`,
`ExecuteResult`, `InterpreterPatch`) are removed from the public surface.

The one cross-module leak — `PredictorTrace.history: REPLHistory | null` in
`src/gepa_types.ts` — is closed in Phase 9 by replacing the field with a
generic `executionTrace: readonly EvaluationTrace[] | null` that is
abstracted across modules. GEPA doesn't need RLM runtime internals in its
trace shape; the current shape is a leaky abstraction from v1.

### 0.4 The spec is ours; we rewrite §0.5 in place

`spec/abstractions.md §0.5` currently reads like the v1 contract (REPL,
`llm_query`, `SUBMIT`). V2 rewrites this contract. Per the product
directive, the TS port is the canonical source of truth; the Python port
conforms or declares non-conformance in its own docs. No numbered sub-section
fallback, no parallel "functional-runtime clause" wrapping the REPL clause.
`§0.5` is rewritten end-to-end with the v2 combinator contract.

### 0.5 All four runtime pillars ship in v0.2.0

The vision names five pillars; pillar 5 is "no backwards compatibility,"
which is a release-level decision (§0.1–§0.3), not a runtime feature. The
four runtime pillars are:

1. **Combinator Runtime.** Typed AST; `Map → Promise.all`; `evaluate()`
   walks the tree with budget, parallelism, and depth bounds.
2. **Deterministic Planner + Task Router.** Quality-shaped `k*`, depth `d`,
   self-consistency width `N`; pre-built plans per task type; the LLM
   participates only as a leaf oracle (and, optionally, a classifier).
3. **Effects-Handler Pattern.** Oracle leaves emit structured intents; the
   runtime handles them and re-prompts. Structured tool use instead of string
   parsing.
4. **Opinionated System Reinjection.** Each plan declares a typed memory
   schema; the planner reinjects the current memory into system messages
   at every oracle call. Oracle leaves update memory via `WriteMemory`
   effects.

All four ship in v0.2.0. Pillars 3 and 4 are not value-add luxuries — they
are quality multipliers. Shipping without them leaves the runtime relying
on prompt hacks for tool use and free-form memory, which is exactly what
v1's REPL was doing badly.

### 0.6 Quality is the objective function

The λ-RLM paper's cost model minimizes
`L(k) = ⌈log_k(n/τ*)⌉ × cost_per_leaf(k)` with a uniform `cost_per_leaf`.
RLM v2 does not use this objective.

RLM v2's objective is expected quality at the ceiling of a generous budget.
Concretely:

- **Self-consistency is on by default.** Every oracle leaf runs `N = 5`
  completions with non-zero temperature and reduces by majority vote. The
  planner treats extra oracle calls as free until the budget ceiling; quality
  is the argmax variable.
- **Ensemble oracles are supported natively.** `modelHint` on an oracle
  node resolves via an LM registry to a different backing LM. Static plans
  can hint different models at different nodes (e.g., a small fast model
  on `Filter`, a large deep model on `Reduce`).
- **Beam search over plans.** When the classifier's top-1 plan has low
  confidence, the router `Cross`es the top-K plans in parallel and `Reduce`s
  by verifier oracle score. Confidence threshold is a planner knob, not a
  user knob.
- **Full trace retention by default.** `trackTrace` defaults to `true`.
  Every intermediate result is retained for post-hoc debugging, GEPA
  optimization, and re-evaluation.
- **Generous budget defaults.** `maxOracleCalls = 200`,
  `maxParallelism = 16`, `maxDepth = 6`, `selfConsistencyN = 5`. Users
  tighten; the library does not apologize for spending calls to improve
  quality.

When we compute `k*`, we pick the `k` that maximizes expected quality per
task, subject to a budget ceiling. The math is still deterministic; the loss
surface is quality-shaped, not cost-shaped.

### 0.7 Combinator library shipped in v0.2.0

All seven combinators from the λ-RLM paper plus two quality primitives plus
the neural oracle:

| Combinator | Role |
|---|---|
| `Split` | Partition a string into `k` chunks |
| `Peek` | Slice a window `[start, end)` out of the input |
| `Map` | Lift an async function over a list; `Promise.all` batched by `maxParallelism` |
| `Filter` | Retain list elements satisfying a predicate |
| `Reduce` | Fold a list into an aggregate; sequential `await` accumulation |
| `Concat` | Join a list of strings with a separator |
| `Cross` | Combine two lists into pairs (Cartesian product) |
| `Vote` | Run an oracle `N` times, majority-reduce (**new; self-consistency primitive**) |
| `Ensemble` | Run an oracle across multiple registered LMs, confidence-reduce (**new**) |
| `Oracle` (`M`) | Single neural-leaf call; thin wrapper over `Predict` |

`Vote` and `Ensemble` are not novel combinators in the research sense —
they are patterns the evaluator already supports via `Map + Reduce` — but
naming them as first-class nodes makes plans readable and makes the planner's
math tractable (it can substitute `N` directly without rewriting a
`Map/Reduce` subtree).

### 0.8 Phase ordering invariant: the facade lands early

The `RLM` facade lands in Phase 5, after pillars 1+2 are functional, and
before pillars 3+4. Later phases extend the facade behavior; they do not
replace it. This gives us a working end-to-end v2 against a real async LM
verified in CI at Phase 5, and each subsequent phase adds quality layers
on top. The phases are still sequential (each merges green or not at all),
but the facade's call signature and exit behavior are stable from Phase 5
onwards.

---

## 1. Architecture overview

V2 is five layers, composed top-down. Every layer is async-native.

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │ RLM (src/rlm.ts)                                                       │
 │   Signature-driven entrypoint. Owns aforward. Thin façade.             │
 └─────────────┬──────────────────────────────────────────────────────────┘
               │
 ┌─────────────▼──────────────────────────────────────────────────────────┐
 │ Task Router (src/rlm_task_router.ts)                                   │
 │   Classify task → select one or more CombinatorPlans from the registry.│
 │   Cross multiple candidates when classifier confidence is low.         │
 └─────────────┬──────────────────────────────────────────────────────────┘
               │
 ┌─────────────▼──────────────────────────────────────────────────────────┐
 │ Deterministic Planner (src/rlm_planner.ts)                             │
 │   Compute k*, depth d, self-consistency N, leaf threshold τ*.          │
 │   Quality-shaped objective under budget ceiling. Pure function.        │
 └─────────────┬──────────────────────────────────────────────────────────┘
               │
 ┌─────────────▼──────────────────────────────────────────────────────────┐
 │ Combinator Evaluator (src/rlm_evaluator.ts + src/rlm_combinators.ts)   │
 │   Walks the typed AST. Map → Promise.all. Oracle leaves → Predict.     │
 │   Enforces budget, propagates errors, collects traces.                 │
 └─────────────┬──────────────────────────────────────────────────────────┘
               │
 ┌─────────────▼──────────────────────────────────────────────────────────┐
 │ Effects Runtime (src/rlm_effects.ts) + Typed Memory (src/rlm_memory.ts)│
 │   Oracle leaves yield Effects; runtime dispatches via EffectHandlers   │
 │   and re-prompts. Memory reinjected into system messages per call.     │
 └────────────────────────────────────────────────────────────────────────┘
```

The LLM participates in exactly three places:

1. **At most one classification call** at the top of `aforward` to pick
   the plan (skipped if the caller passes `taskType`).
2. **Oracle leaf calls** inside `Map(M, ...)` / `Vote(M, ...)` / `Ensemble(M, ...)`
   nodes, parallelized via `Promise.all`, with `N`-way self-consistency by
   default.
3. **Effect re-prompts** when an oracle leaf yields an effect and the
   runtime re-calls the LLM with the effect result.

All other control flow is deterministic TypeScript. There is no `eval`,
no `vm`, no sync-LM bridge, no "the model writes control code" step.

---

## 2. Type contracts (canonical interfaces)

This section is the single source of truth for the shape of v2. Every other
section cites it.

### 2.1 Combinator AST

```typescript
// src/rlm_combinators.ts

export type CombinatorList = readonly CombinatorValue[];
export type CombinatorValue =
  | string
  | number
  | boolean
  | CombinatorList
  | Record<string, unknown>;

/**
 * A typed AST of operations. `evaluate()` lowers this into awaited Promises.
 * Every node is a plain object; no methods. Plans are serializable,
 * auditable, and planner-rewritable.
 */
export type CombinatorNode =
  | { readonly tag: 'literal'; readonly value: CombinatorValue }
  | { readonly tag: 'var'; readonly name: string }
  | { readonly tag: 'split'; readonly input: CombinatorNode; readonly k: CombinatorNode }
  | { readonly tag: 'peek'; readonly input: CombinatorNode; readonly start: CombinatorNode; readonly end: CombinatorNode }
  | { readonly tag: 'map'; readonly fn: CombinatorFn; readonly items: CombinatorNode }
  | { readonly tag: 'filter'; readonly pred: CombinatorFn; readonly items: CombinatorNode }
  | { readonly tag: 'reduce'; readonly op: CombinatorBinary; readonly items: CombinatorNode; readonly init?: CombinatorNode }
  | { readonly tag: 'concat'; readonly items: CombinatorNode; readonly separator?: CombinatorNode }
  | { readonly tag: 'cross'; readonly left: CombinatorNode; readonly right: CombinatorNode }
  | { readonly tag: 'vote'; readonly oracle: CombinatorNode; readonly n: CombinatorNode; readonly reducer?: VoteReducer }
  | { readonly tag: 'ensemble'; readonly oracle: CombinatorNode; readonly models: readonly string[]; readonly reducer?: EnsembleReducer }
  | { readonly tag: 'oracle'; readonly prompt: CombinatorNode; readonly modelHint?: string; readonly effectHandlers?: readonly string[] };

/**
 * Function body for `map` / `filter`. A sub-AST with a bound parameter name.
 * Lambda-calculus, not JS closure. Serializable.
 */
export interface CombinatorFn {
  readonly param: string;
  readonly body: CombinatorNode;
}

export interface CombinatorBinary {
  readonly left: string;
  readonly right: string;
  readonly body: CombinatorNode;
}

export type VoteReducer = 'majority' | 'mode' | 'verifier';
export type EnsembleReducer = 'majority' | 'confidence' | 'verifier';
```

The AST is a data structure, not a set of closures. The deterministic
planner introspects and rewrites plans (substitute `k*` into a `split`,
set `N` on a `vote`, narrow `models` on an `ensemble`). Closures are
opaque; an AST is transparent.

### 2.2 Budget and execution context

```typescript
// src/rlm_types.ts (rewritten end-to-end)

export interface RLMBudget {
  /** Max oracle (neural leaf) calls across the entire run. */
  readonly maxOracleCalls: number;
  /** Max parallel fan-out at any single `Map`. */
  readonly maxParallelism: number;
  /** Max recursion depth of self-referential plans. */
  readonly maxDepth: number;
  /** Leaf threshold τ*: strings at or below this length call the oracle directly. */
  readonly leafThreshold: number;
  /** Default self-consistency width N; 1 disables self-consistency at oracle leaves. */
  readonly selfConsistencyN: number;
  /** Max effect re-prompt turns inside a single oracle call. */
  readonly maxEffectTurns: number;
}

export const DEFAULT_BUDGET: RLMBudget = Object.freeze({
  maxOracleCalls: 200,
  maxParallelism: 16,
  maxDepth: 6,
  leafThreshold: 1000,
  selfConsistencyN: 5,
  maxEffectTurns: 8,
});

export interface EvaluationContext {
  readonly budget: RLMBudget;
  readonly lm: BaseLM;                                   // primary async LM
  readonly lmRegistry: ReadonlyMap<string, BaseLM>;     // for modelHint routing
  readonly signature: Signature;                         // input/output shape
  readonly scope: ReadonlyMap<string, CombinatorValue>; // bound variables
  readonly depth: number;                                // for recursion tracking
  readonly callsUsed: { current: number };              // mutable counter
  readonly trace: EvaluationTrace[];                    // append-only
  readonly handlers: ReadonlyMap<string, EffectHandler>;
  readonly memory: TypedMemoryState;                    // bound per-call
}

export interface EvaluationTrace {
  readonly step: number;
  readonly nodeTag: CombinatorNode['tag'];
  readonly startedAt: string;      // ISO
  readonly durationMs: number;
  readonly ok: boolean;
  readonly cause?: unknown;
  readonly extras?: Readonly<Record<string, unknown>>;  // per-node metadata (e.g., effect turn count, vote winner)
}
```

### 2.3 Task routing and planning

```typescript
// src/rlm_task_router.ts

export type TaskType =
  | 'search'
  | 'classify'
  | 'aggregate'
  | 'pairwise'
  | 'summarise'
  | 'multi_hop'
  | 'unknown';

export interface PlanningInputs {
  readonly taskType: TaskType;
  readonly promptLength: number;
  readonly budget: RLMBudget;
  readonly preferredK?: number;
}

export interface ResolvedPlan {
  readonly plan: CombinatorNode;
  readonly partitionK: number;
  readonly depth: number;
  readonly selfConsistencyN: number;
  readonly estimatedOracleCalls: number;
  readonly memorySchema: MemorySchema | null;
}

export interface StaticPlan {
  readonly taskType: TaskType;
  readonly template: CombinatorNode;           // unresolved; holds vref() placeholders
  readonly memorySchema: MemorySchema | null;  // optional per-plan memory
}

export interface ClassifierResult {
  readonly primary: TaskType;
  readonly confidence: number;                 // 0..1
  readonly candidates: readonly TaskType[];    // sorted, primary first
}
```

### 2.4 Effects

```typescript
// src/rlm_effects.ts

export type Effect =
  | { readonly kind: 'ReadContext'; readonly name: string; readonly start?: number; readonly end?: number }
  | { readonly kind: 'WriteMemory'; readonly key: string; readonly value: unknown }
  | { readonly kind: 'QueryOracle'; readonly prompt: string; readonly modelHint?: string }
  | { readonly kind: 'Search'; readonly query: string; readonly topK?: number }
  | { readonly kind: 'Yield' }
  | { readonly kind: 'Custom'; readonly name: string; readonly args: Readonly<Record<string, unknown>> };

export type EffectResult =
  | { readonly ok: true; readonly value: unknown }
  | { readonly ok: false; readonly error: string };

export interface EffectHandler {
  readonly name: Effect['kind'] | string;     // name identifies which Effect this handles
  handle(effect: Effect, ctx: EvaluationContext): Promise<EffectResult>;
}

/** Oracle-leaf response shape; parsed out of LM output. */
export type OracleResponse =
  | { readonly kind: 'value'; readonly value: string }
  | { readonly kind: 'effect'; readonly effect: Effect };
```

The oracle leaf drives a bounded loop:

```
for turn in 0..ctx.budget.maxEffectTurns:
    response = await predictOracle(prompt, memory, ctx)
    if response.kind == 'value':
        return response.value
    handler = ctx.handlers.get(response.effect.kind) ?? ctx.handlers.get('Custom')
    result = await handler.handle(response.effect, ctx)
    prompt = appendEffectResult(prompt, response.effect, result)
raise BudgetError('oracle exceeded maxEffectTurns')
```

### 2.5 Typed memory

```typescript
// src/rlm_memory.ts

export interface MemoryFieldSchema {
  readonly name: string;
  readonly type: TypeTag;
  readonly description: string;
  readonly initial?: unknown;
  readonly maxLength?: number;                 // applies to string values
}

export interface MemorySchema {
  readonly name: string;
  readonly fields: readonly MemoryFieldSchema[];
  readonly maxBytesSerialized: number;         // hard cap; default 2048
}

export type TypedMemoryState = ReadonlyMap<string, unknown>;

export interface MemoryInjector {
  render(schema: MemorySchema, state: TypedMemoryState): string;
}

export function applyMemoryWrite(
  state: TypedMemoryState,
  schema: MemorySchema,
  write: { readonly key: string; readonly value: unknown },
): TypedMemoryState;

export function initialMemoryState(schema: MemorySchema): TypedMemoryState;
```

The injector's `render()` produces a terse, typed header appended to the
oracle system message:

```
[[RLM_MEMORY schema=failure_diagnostic]]
failure_pattern: str = "oracle returned unrelated answer on first chunk"
next_check: str = "verify chunk boundary alignment"
prevented_action: str = "do not re-submit without reviewing boundary"
[[/RLM_MEMORY]]
```

Writes flow through `WriteMemory` effects; values are validated against the
schema (type and `maxLength`), serialized length is bounded by
`maxBytesSerialized`. Over-long writes are rejected with a structured error
that the LLM sees on the next turn.

### 2.6 RLM public surface

```typescript
// src/rlm.ts

export interface RLMOptions {
  readonly budget?: Partial<RLMBudget>;
  readonly taskType?: TaskType;                        // optional classifier override
  readonly subLm?: BaseLM | null;
  readonly lmRegistry?: ReadonlyMap<string, BaseLM>;   // for modelHint routing; default: empty
  readonly handlers?: readonly EffectHandler[];        // user-added effect handlers
  readonly plans?: readonly StaticPlan[];              // user-added static plans (merged with defaults)
  readonly trackTrace?: boolean;                        // default: true
}

export class RLM<TSig extends SignatureInput = Signature, ...>
  extends Module<InferInputs<TSig>, InferOutputs<TSig>> {

  constructor(signature: TSig, options?: RLMOptions);

  override forward(): never;   // throws RuntimeError, matches LM
  override async aforward(kwargs: InferInputs<TSig>): Promise<Prediction<InferOutputs<TSig>>>;
}
```

This is the entire public API. Combinator constructors, the evaluator,
effect handlers, and the memory primitives are exported from `src/index.ts`
for power users who want to author their own plans.

---

## 3. File layout

Every new module is listed with purpose and target LOC. Targets are not
ceilings; deviations of >2× trigger a design review.

| File | Purpose | Target LOC |
|---|---|---|
| `src/rlm.ts` | Public `RLM` class. Thin façade over the evaluator. | ~200 |
| `src/rlm_types.ts` | `RLMBudget`, `DEFAULT_BUDGET`, `EvaluationContext`, `EvaluationTrace`. | ~120 |
| `src/rlm_combinators.ts` | `CombinatorNode` union + typed constructors. | ~250 |
| `src/rlm_evaluator.ts` | `evaluate(plan, ctx)`: async walker. `Map → Promise.all`. Budget + trace. Oracle branch drives effect loop. | ~320 |
| `src/rlm_planner.ts` | `resolvePlan(...)`; quality-shaped `k*`, `d`, `N` math. | ~160 |
| `src/rlm_task_router.ts` | `classifyTask(...)` + static plan registry. | ~220 |
| `src/rlm_effects.ts` | `Effect` union + `EffectHandler` protocol + built-in handlers (`ReadContext`, `WriteMemory`, `QueryOracle`, `Yield`). | ~180 |
| `src/rlm_memory.ts` | `MemorySchema`, `TypedMemoryState`, `applyMemoryWrite`, `MemoryInjector`. | ~140 |
| `tests/rlm_combinators.test.ts` | Per-combinator unit tests (happy / empty / malformed). | ~320 |
| `tests/rlm_evaluator.test.ts` | End-to-end plan evaluation with a mock oracle. | ~300 |
| `tests/rlm_planner.test.ts` | Quality-math; `k*`, `d`, `N` invariants. | ~180 |
| `tests/rlm_task_router.test.ts` | Classifier + plan registry + beam routing. | ~220 |
| `tests/rlm_effects.test.ts` | Effect loop, re-prompt, max-turns, handler dispatch. | ~220 |
| `tests/rlm_memory.test.ts` | Schema validation, reinjection rendering, `applyMemoryWrite`. | ~160 |
| `tests/rlm_integration.test.ts` | Real-async-LM happy path (the engineer's original ask). | ~180 |
| `tests/rlm_property.test.ts` | Property / fuzz (termination, depth bound, determinism). | ~180 |
| `tests/rlm_replay.test.ts` | v2 fixture-driven replay. | ~120 |
| `tests/rlm_automata.test.ts` | v2 automata validation. | ~100 |
| `tests/gepa.test.ts` | GEPA over v2 `RLM` (rewritten from the old gepa section). | ~240 |

### Files deleted at Phase 9

| File | Size | Reason |
|---|---|---|
| `src/node_code_interpreter.ts` | 600 LOC | REPL runtime, no longer referenced. |
| `tests/rlm_gepa.test.ts` | ~350 LOC | Tests the REPL contract; GEPA section survives as `tests/gepa.test.ts`. |
| `tests/rlm_gepa_automata.test.ts` | 124 LOC | `rlm_control` and `interpreter_session` automata are REPL-shaped. Replaced by `tests/rlm_automata.test.ts`. |

### Net LOC

| | v1 | v2 | Δ |
|---|---|---|---|
| `src/` RLM surface | ~1,330 | ~1,590 | +260 (+19%) |
| `tests/` RLM | ~740 | ~2,220 | +1,480 (+200%) |

v2 src grows 19%. That's not a regression: v1 offloaded three of four pillars
into REPL-parsed strings and `eval()`/`vm` sandboxes, which is uncountable
LOC hidden inside string prompts plus hundreds of LOC of sandbox plumbing.
v2 surfaces each pillar as a typed, audited runtime. Tests roughly triple
because every new runtime (effects, memory, self-consistency, ensemble) gets
a proper test file. We get a system that is larger but understandable,
versus one that was smaller but relied on runtime string parsing.

---

## 4. Phase roadmap

Each phase is one PR. Phases are sequential; a phase merges green or it
doesn't merge. Each PR is small enough to review in one sitting.

Exit gates are literal: each phase lists the commands whose green is
required. "Exit" always implies `pnpm test && pnpm run typecheck && pnpm run build`
green, plus any phase-specific commands listed.

### Phase 0 — Spec and decision record (no code)

**Branch:** `rlm-v2/phase-0-spec`

**Deliverables:**

1. Rewrite `../spec/abstractions.md §0.5. RLM Contract Boundary` in place.
   The new §0.5 commits to:
   - signature-driven module interface (kept)
   - bounded recursive execution via `maxOracleCalls`, `maxParallelism`,
     `maxDepth`, `leafThreshold`, `selfConsistencyN`, `maxEffectTurns`
     (replaces `max_iterations`, `max_llm_calls`, `max_output_chars`)
   - typed combinator runtime: `Split`, `Peek`, `Map`, `Filter`, `Reduce`,
     `Concat`, `Cross`, `Vote`, `Ensemble`, and a neural-oracle leaf
     primitive (replaces "sandboxed execution loop with typed `SUBMIT`")
   - deterministic task router + planner; the LLM participates only as a
     bounded leaf oracle and, optionally, a task-type classifier (replaces
     `llm_query` / `llm_query_batched`)
   - effects-handler pattern for oracle-leaf tool use: oracle leaves emit
     structured `Effect`s, the runtime dispatches via named `EffectHandler`s
     and re-prompts with the result (replaces string-parsed REPL tool use)
   - typed memory with system reinjection: each plan declares a typed
     memory schema, the planner reinjects the current memory state into the
     system message at every oracle call (replaces free-form skill_file
     scratchpads)
   - sub-LM resolution order `options.subLm ?? settings.lm` (kept); plus an
     `lmRegistry` for `modelHint`-based ensemble routing (new)
   - async-only execution; `forward()` throws `RuntimeError` (new; parallels
     §6 `LM` semantics)
2. Add `docs/product/rlm-v2-architecture.md` as the permanent decision
   record, following the four-heading template (`Decision` / `Why` /
   `What This Means In Code` / `Revisit If`). This is the durable
   narrative; the plan you are reading now is superseded by the DR after
   implementation lands.
3. Add a `BREAKING CHANGES` section to `README.md` describing the v1 → v2
   move for anyone who pulled a v0.1.x tarball.
4. Verify that no consumer-facing code depends on the v1 RLM contract by
   running `pnpm run deps:rlm-legacy` (a new script that greps for the
   deprecated symbols and fails if any production import remains; see §5.7).

**Exit gate:** `pnpm test` green. `pnpm run typecheck` green. PR reviewed
and merged. Spec diff is self-contained and does not require any code
change. `docs/product/rlm-v2-architecture.md` passes the four-heading DR
template check.

### Phase 1 — Combinator primitives (no network)

**Branch:** `rlm-v2/phase-1-combinators`

**Deliverables:**

1. Rewrite `src/rlm_types.ts` per §2.2. `DEFAULT_BUDGET` is exported
   frozen. Old REPL types remain in the file (still imported by `rlm.ts`
   and `node_code_interpreter.ts`); Phase 9 deletes them.
2. Create `src/rlm_combinators.ts` per §2.1. Typed constructors:

   ```typescript
   export const split    = (input: CombinatorNode, k: CombinatorNode): CombinatorNode => ({ tag: 'split', input, k });
   export const peek     = (input: CombinatorNode, start: CombinatorNode, end: CombinatorNode): CombinatorNode => ({ tag: 'peek', input, start, end });
   export const map      = (fn: CombinatorFn, items: CombinatorNode): CombinatorNode => ({ tag: 'map', fn, items });
   export const filter   = (pred: CombinatorFn, items: CombinatorNode): CombinatorNode => ({ tag: 'filter', pred, items });
   export const reduce   = (op: CombinatorBinary, items: CombinatorNode, init?: CombinatorNode): CombinatorNode => ({ tag: 'reduce', op, items, ...(init ? { init } : {}) });
   export const concat   = (items: CombinatorNode, separator?: CombinatorNode): CombinatorNode => ({ tag: 'concat', items, ...(separator ? { separator } : {}) });
   export const cross    = (left: CombinatorNode, right: CombinatorNode): CombinatorNode => ({ tag: 'cross', left, right });
   export const vote     = (oracle: CombinatorNode, n: CombinatorNode, reducer?: VoteReducer): CombinatorNode => ({ tag: 'vote', oracle, n, ...(reducer ? { reducer } : {}) });
   export const ensemble = (oracle: CombinatorNode, models: readonly string[], reducer?: EnsembleReducer): CombinatorNode => ({ tag: 'ensemble', oracle, models, ...(reducer ? { reducer } : {}) });
   export const oracle   = (prompt: CombinatorNode, modelHint?: string, effectHandlers?: readonly string[]): CombinatorNode => ({ tag: 'oracle', prompt, ...(modelHint ? { modelHint } : {}), ...(effectHandlers ? { effectHandlers } : {}) });
   export const lit      = (value: CombinatorValue): CombinatorNode => ({ tag: 'literal', value });
   export const vref     = (name: string): CombinatorNode => ({ tag: 'var', name });
   export const fn       = (param: string, body: CombinatorNode): CombinatorFn => ({ param, body });
   export const bop      = (left: string, right: string, body: CombinatorNode): CombinatorBinary => ({ left, right, body });
   ```
3. Create `src/rlm_evaluator.ts` with the offline evaluator:
   - `evaluate(plan, ctx)` returning `Promise<CombinatorValue>`.
   - `Map` is `Promise.all` after chunking by `maxParallelism`.
   - `Reduce` is sequential `await` accumulation.
   - `Vote` and `Ensemble` throw `RuntimeError` at Phase 1 (LM not wired
     yet; they land fully at Phase 2).
   - `Oracle` throws `RuntimeError` at Phase 1 (same reason).
   - Every node pushes an entry to `ctx.trace`.
4. Create `tests/rlm_combinators.test.ts` with happy / empty / malformed
   cases per combinator.
5. Create `tests/rlm_evaluator.test.ts`: plans wired by hand; verify that
   `split → map → concat` round-trips a string when the map is identity;
   verify that `reduce` with missing `init` on empty list errors cleanly.

**Exit gate:** `pnpm test` green. `pnpm run typecheck` green. No network
calls. Public exports from `src/index.ts` unchanged.

### Phase 2 — Oracle leaf, self-consistency, ensemble, Predict integration

**Branch:** `rlm-v2/phase-2-oracle`

**Deliverables:**

1. Oracle branch in `rlm_evaluator.ts`:
   - `prompt` evaluated to a string.
   - Single-field `Signature` `'prompt: str -> answer: str'` cached at
     module scope.
   - A short-lived `Predict` instance is called per oracle invocation via
     `new Predict(oracleSig).acall({ prompt }, { lm: routedLm })`.
   - `routedLm = ctx.lmRegistry.get(modelHint) ?? ctx.lm`.
   - Budget enforcement: `ctx.callsUsed.current += 1` before the call;
     throws `BudgetError` when `current > ctx.budget.maxOracleCalls`.
2. Vote branch in `rlm_evaluator.ts`:
   - Runs the inner `oracle` node `N` times via `Promise.all` (respecting
     `maxParallelism`).
   - `reducer = 'majority'` (default) takes the mode of parsed answers.
     `reducer = 'verifier'` routes results through a verifier oracle and
     picks the highest-scoring.
3. Ensemble branch in `rlm_evaluator.ts`:
   - Runs the inner `oracle` node once per `modelHint` in `models`,
     overriding the routed LM for each.
   - `reducer = 'confidence'` (default) uses self-reported confidence
     fields from the oracle signature; `reducer = 'majority'` is equivalent
     to `Vote` across models; `reducer = 'verifier'` uses a separate
     verifier oracle.
4. Helper `buildEvaluationContext({ lm, signature, budget, lmRegistry, handlers, memory })`.
5. Extend `tests/rlm_evaluator.test.ts` with:
   - `QueueLM`-backed happy path: `oracle(lit('hello'))` returns the
     queued response.
   - Budget exhaustion: 5 oracle calls with `maxOracleCalls = 3` throws
     `BudgetError` after the third call with trace state preserved.
   - Parallelism cap: `Map` over 10 items with `maxParallelism = 2` runs
     exactly 5 batches (verified by timing assertions against a mock LM
     with a configurable delay).
   - `vote(oracle(...), 5, 'majority')` with 3 queued agreeing responses
     and 2 disagreeing returns the agreed-on value.
   - `ensemble(oracle(...), ['gpt-fast', 'gpt-deep'])` uses the expected
     registry entries and reduces by default.

**Exit gate:** `pnpm test` green. The evaluator can now run real async
LMs end-to-end at the primitive level. This is the earliest point at
which the engineer's "test what you ship" ask is satisfied, even before
the full `RLM` class exists.

### Phase 3 — Deterministic planner (quality-shaped)

**Branch:** `rlm-v2/phase-3-planner`

**Deliverables:**

1. `src/rlm_planner.ts`:
   - `computeOptimalPartitionSize({ promptLength, budget, taskType })`
     returns `k*` that maximizes expected-quality over integer `k ≥ 2`
     under the budget ceiling. The quality model is task-type-conditional
     (see §4.2 below for the initial quality curves).
   - `computeRecursionDepth({ promptLength, leafThreshold, k })` returns
     `⌈log_k(n / τ*)⌉`.
   - `computeSelfConsistencyN({ budget, estimatedOracleCalls })` returns
     the largest `N` that keeps the total call estimate within the budget
     ceiling. Floors at `1`; ceilings at `budget.selfConsistencyN`.
   - `resolvePlan({ plan, planningInputs })` walks the AST and substitutes
     literal `k`, `n`, and `modelHint` values into the template, leaving
     every other node alone.
2. Initial quality curves in `src/rlm_planner_quality.ts`
   (co-located helper):
   - `search`: quality grows monotonically with `k`; recommend `k = 4..8`.
   - `aggregate`: quality peaks at `k = 3..5`; too-large `k` loses signal.
   - `summarise`: quality grows monotonically with `k`; depth matters more
     than `k`.
   - `pairwise`: `k = 2` is the only meaningful value; depth = 1.
   - `multi_hop`: quality grows with both `k` and `N`; `d ≥ 2` often needed.
   - `classify`: quality saturates at `k = 2..3`; large `N` matters more.
   - `unknown`: conservative defaults (`k = 3`, `d = 2`, `N = 5`).

   Curves are seeded with literature-derived priors and become the
   calibration target of Phase 8 benchmark runs. Each curve is a pure
   function `(k, n) -> expectedQuality`.
3. `tests/rlm_planner.test.ts`:
   - Per-task-type math: given `n`, `budget`, the planner picks the
     documented-optimal `k` from the seeded quality curve.
   - Bounds: `k ≥ 2`; `depth ≥ 1`; `selfConsistencyN ≥ 1`.
   - Idempotency: `resolvePlan(resolvePlan(plan, inputs), inputs)` is
     structurally equal to the first call.
   - Property test (fast-check): for any `n ∈ [100, 100_000]`,
     `τ* ∈ [50, 5000]`, `budget.maxOracleCalls ∈ [10, 1000]`, the planner
     terminates and returns a resolvable plan.

**Exit gate:** `pnpm test` green. The planner is a pure function with no
dependency on the evaluator or the LM.

### Phase 4 — Task router and static plan registry

**Branch:** `rlm-v2/phase-4-router`

**Deliverables:**

1. `src/rlm_task_router.ts`:
   - `STATIC_PLANS: ReadonlyMap<TaskType, StaticPlan>` with six entries
     mapping to the Table 1 plans from the λ-RLM paper (`search`, `classify`,
     `aggregate`, `pairwise`, `summarise`, `multi_hop`). Each plan is
     parameterized on `k*`, `τ*`, `N`, and `modelHint` via `vref()`
     placeholders resolved by the planner. Where the plan benefits from
     memory, the `StaticPlan` attaches a `MemorySchema`.
   - `classifyTask(prompt, signature, lm)`: a single async call to a one-shot
     `Predict('context: str -> primary: str, confidence: float, candidates: list[str]')`.
     Returns a `ClassifierResult`. On no-match, returns `'unknown'` with
     confidence `0`.
   - Beam routing: `resolveRoute(classifierResult, threshold = 0.7)`
     returns either a single `StaticPlan` (high confidence) or a `Cross`-wrapped
     set of the top-K plans (low confidence), which the planner then
     resolves into a single top-level `Cross` node whose branches are
     each fully resolved.
2. `tests/rlm_task_router.test.ts`:
   - Each static plan resolves to a valid AST when composed with the planner.
   - The classifier returns the expected tag for a handful of seed prompts.
   - Unknown task type falls back to `summarise`.
   - Explicit `taskType` override skips the classifier (verified by LM
     call count).
   - Low-confidence classification produces a beam-routed `Cross` plan;
     high-confidence produces a single plan.

**Exit gate:** `pnpm test` green. The router is the last piece needed
before the public `RLM` class can be assembled.

### Phase 5 — Public `RLM` class (end-to-end MVP)

**Branch:** `rlm-v2/phase-5-facade`

**Deliverables:**

1. Rewrite `src/rlm.ts` end-to-end per §2.6:
   - Constructor validates the signature, the optional `budget` override,
     and registers built-in plans plus any user-supplied plans.
   - `forward()` throws
     `RuntimeError('RLM is async-only. Use acall() or aforward() instead.')`.
   - `aforward({ ...inputs })`:
     1. Resolve the `lm` via `options.subLm ?? settings.lm` (throws on
        absent).
     2. Build the prompt string from `inputs` via the active `Adapter`.
     3. Resolve the task type:
        `options.taskType ?? (await classifyTask(prompt, signature, lm))`.
     4. Look up (or beam-route) the static plan(s).
     5. `resolvePlan` with `promptLength`, `budget`, and the classifier's
        result.
     6. Build an `EvaluationContext` with default handlers (registered
        in Phase 6, but the context shape is already there), the
        user-supplied registry, and an initial `TypedMemoryState` (empty
        in Phase 5; populated in Phase 7).
     7. `evaluate(resolved.plan, ctx)`.
     8. Wrap the result in `Prediction.create<TOutputs>({ answer, trace })`
        per the existing `Prediction` contract.
2. Update `src/index.ts` exports per §6.
3. `tests/rlm_integration.test.ts`:
   - Real-async-LM happy path using a minimal `BaseLM` subclass backed
     by a scripted async fetch. Answers a `search`-typed question over a
     chunked context. Answers a `summarise`-typed question over a chunked
     context. Verifies the ordered trace-tag sequence matches the plan's
     AST walk.
   - No `ReplayLM`; no sync stubs. This is the promise the engineer
     asked for, at the earliest responsible point in the roadmap.

**Exit gate:** `pnpm test` green. `pnpm run typecheck` green. Every
public import path in `src/index.ts` resolves; `pnpm run pack:dry-run`
produces a tarball whose type declarations are consumable from a clean
TS project (tested by the consumer smoke test in Phase 11).

### Phase 6 — Effects runtime

**Branch:** `rlm-v2/phase-6-effects`

**Deliverables:**

1. `src/rlm_effects.ts`:
   - `Effect` union and `EffectHandler` protocol per §2.4.
   - Built-in handlers:
     - `ReadContextHandler` — reads a chunk from the input context by
       index or byte range; bounds-checked against the original prompt.
     - `WriteMemoryHandler` — applies a `WriteMemory` effect; validates
       the write against the active `MemorySchema`; returns an error
       result on schema violation (the error is surfaced to the LLM on
       the next turn so it can retry).
     - `QueryOracleHandler` — delegates a prompt to a freshly-built
       oracle leaf; still single-call at the handler level, no recursive
       `rlm.aforward`.
     - `YieldHandler` — no-op success; surfaces a control signal for
       cooperative partitioning.
     - `CustomHandler` dispatch path for user-added handlers.
   - `parseOracleResponse(completion: string): OracleResponse` — a
     structured-output parser (delegates to the active `Adapter`'s JSON
     parse path against a purpose-built signature).
2. Oracle loop in `rlm_evaluator.ts`:
   - For each oracle call, loop up to `ctx.budget.maxEffectTurns`:
     - Parse the completion as an `OracleResponse`.
     - If `kind === 'value'`, return `value`.
     - Else look up the handler by `effect.kind` (or `'Custom'` for
       named custom effects), call it, append the result to the prompt
       buffer, and continue the loop.
   - On loop exhaustion, throw `BudgetError('oracle exceeded maxEffectTurns')`.
   - Every loop turn pushes an `EvaluationTrace` entry tagged `'effect'`
     with extras `{ turn, effectKind, ok }`.
3. Oracle signature extended:
   `'prompt: str -> kind: literal["value", "effect"], value: optional[str], effect_name: optional[str], effect_args: optional[dict]'`.
4. `tests/rlm_effects.test.ts`:
   - Value immediately: no loop iteration, one oracle call.
   - Effect then value: two oracle calls, correct handler invoked.
   - Unknown handler: the runtime returns a structured error to the LLM
     on the next turn; the LLM can retry with a valid effect.
   - Loop exhaustion: `maxEffectTurns = 2` with three effect-type
     responses throws `BudgetError` with the partial trace preserved.
   - Custom handler dispatch: a user-supplied handler with a custom name
     is invoked via the `Custom` effect kind.

**Exit gate:** `pnpm test` green. End-to-end effect loop demonstrated
against a `QueueLM` in `tests/rlm_effects.test.ts` and via an extension
to `tests/rlm_integration.test.ts` that exercises a single `Search`
effect round-trip with a mocked search handler.

### Phase 7 — Typed memory and system reinjection

**Branch:** `rlm-v2/phase-7-memory`

**Deliverables:**

1. `src/rlm_memory.ts`:
   - `MemorySchema`, `MemoryFieldSchema`, `TypedMemoryState` types per §2.5.
   - `initialMemoryState(schema)`: returns a frozen `Map` seeded from each
     field's `initial`.
   - `applyMemoryWrite(state, schema, write)`: validates type and length
     against `schema`; returns a new `Map` or throws `ValueError` on
     violation.
   - `defaultMemoryInjector`: renders a deterministic, bounded
     `[[RLM_MEMORY ...]] ... [[/RLM_MEMORY]]` block suitable for
     prepending to the oracle's system message.
2. Wire the memory into the evaluator:
   - The `EvaluationContext.memory` is threaded through every oracle call.
   - Before each oracle invocation, the memory block is prepended to the
     system message via the active adapter's system-message hook.
   - `WriteMemoryHandler` (from Phase 6) is now fully functional: writes
     produce a new `TypedMemoryState` that is propagated through the
     context for subsequent oracle calls within the same plan tree.
3. Task router adjustments:
   - Static plans for `search`, `aggregate`, `multi_hop`, and `summarise`
     attach a default `MemorySchema` with 2–3 tightly-bounded fields
     (failure_pattern, next_check, prevented_action).
   - The `RLM` facade reads the memory schema from the resolved plan
     and builds the initial memory state accordingly.
4. `tests/rlm_memory.test.ts`:
   - Schema validation: too-long string, wrong type, unknown field name
     are all rejected.
   - `initialMemoryState(schema)` honors `initial` values.
   - `defaultMemoryInjector` renders deterministically and does not
     exceed `maxBytesSerialized`.
   - End-to-end: a plan with memory reinjects on the second oracle call
     the write that the first oracle emitted (verified via the captured
     system-message sequence on `QueueLM`).
5. Extension of `tests/rlm_integration.test.ts`: a real-async-LM run on
   a `search`-typed task verifies that a `WriteMemory` effect on oracle
   turn 1 changes the observed system message on oracle turn 2.

**Exit gate:** `pnpm test` green. All four vision pillars now live in
the facade's `aforward` path.

### Phase 8 — Replace fixtures and benchmarks; calibrate quality curves

**Branch:** `rlm-v2/phase-8-fixtures`

**Deliverables:**

1. Replace `../spec/fixtures/rlm_replay.json` with
   `../spec/fixtures/rlm_v2_replay.json`. Each case:
   - `id`, `task_type`, `inputs`, `budget`, `lm_script` (array of queued
     async responses), `expected.trace_tag_sequence`, `expected.final`,
     `expected.memory_transitions` (optional).
   - Update `../benchmarks/release_gate.py` `_REQUIRED_FIXTURE_CASES`
     from `rlm_replay.json` to `rlm_v2_replay.json` in the same PR so
     the gate keeps passing.
2. Record 12 seed cases across the six task types plus two effect-driven
   cases via `tools/record_rlm_v2.ts`, which runs `RLM` end-to-end
   against a canned `BaseLM` subclass and dumps the trace.
3. Create `tests/rlm_replay.test.ts` consuming the new fixture.
4. Update `../spec/fixtures/rlm_gepa_automata.json` with three new machines,
   replacing the old `rlm_control` and `interpreter_session`:
   - `rlm_v2_control`: states `INIT`, `CLASSIFY`, `PLAN`, `EVALUATE`,
     `FINAL_DONE`, `FINAL_BUDGET_ERROR`, `FINAL_ORACLE_ERROR`.
   - `combinator_eval`: states `PENDING`, `MAP_PARALLEL`, `REDUCE_SERIAL`,
     `VOTE_AGGREGATING`, `ENSEMBLE_FANOUT`, `ORACLE_INVOKING`, `DONE`,
     `FAILED`.
   - `oracle_effect_loop`: states `TURN_START`, `PREDICT`, `PARSE`,
     `HANDLE_EFFECT`, `MEMORY_WRITE`, `DONE`, `FAILED_BUDGET`.
5. Create `tests/rlm_automata.test.ts` against the new machines.
6. Benchmark calibration:
   - Run `pnpm run bench:rlm-v2` against `gsm8k`, `squad`, `hotpotqa`
     (already in the release gate corpus).
   - Feed the accuracy measurements back into the quality curves in
     `src/rlm_planner_quality.ts`. The curves are seeded from literature
     priors; this phase narrows them with observed data.
   - Acceptance target: **v2 accuracy >= v1 accuracy on all three
     datasets**, with latency and call-count reported but not
     acceptance-gated (quality is the objective).

**Exit gate:** `pnpm test` green. `./benchmarks/run_benchmarks.sh --all`
green. Accuracy regression vs. v1 is zero or positive on every dataset.
Calibrated quality curves are committed.

### Phase 9 — Delete legacy surface

**Branch:** `rlm-v2/phase-9-deleting`

**Deliverables:**

1. `rm src/node_code_interpreter.ts`.
2. Remove all now-unused imports of `NodeCodeInterpreter`,
   `SyncCodeInterpreter`, `SyncCodeSession`, `createNodeCodeInterpreter`,
   `REPLHistory`, `REPLEntry`, `REPLVariable`, `CodeInterpreter`,
   `CodeSession`, `ExecuteRequest`, `ExecuteResult`, `InterpreterPatch`,
   `CodeInterpreterError`, `BudgetVector`, `LLMQueryRequest`,
   `LLMQueryResult`, `RLMConfig` from `src/index.ts` and every
   consuming module.
3. Replace `PredictorTrace.history: REPLHistory | null` in
   `src/gepa_types.ts` with
   `executionTrace: readonly EvaluationTrace[] | null` (generic across
   modules). Update GEPA trace-capture helpers (`capturePredictorTraces`)
   accordingly. GEPA is no longer RLM-coupled at the type level.
4. `rm tests/rlm_gepa.test.ts` and `tests/rlm_gepa_automata.test.ts`.
   Keep `tests/gepa.test.ts` (the GEPA-over-RLM coverage).
5. Confirm `pnpm run deps:circular` passes and `pnpm run knip` reports
   no unused exports.
6. Confirm no file in `src/` imports `node:vm`. Automated check added to
   the release gate.

**Exit gate:** `pnpm test` green. Full release gate
(`pnpm run release:gate`) green. `pnpm run deps:rlm-legacy` confirms no
legacy symbols remain.

### Phase 10 — Consumer smoke tests, README, examples

**Branch:** `rlm-v2/phase-10-docs`

**Deliverables:**

1. `README.md`:
   - Replace the tier-4 line `"RLM scaffold"` with
     `"RLM v2: typed combinator runtime; async-only; structured tool use; typed memory"`.
   - Add a 60-second quickstart for `RLM` alongside the existing
     `Predict` / `ChainOfThought` / `ReAct` sections.
   - Retain the `BREAKING CHANGES` section added in Phase 0; update it
     with the final v0.2.0 → migration-from-v0.1 call-outs.
2. `docs/product/rlm-v2-architecture.md` (DR) — final pass: add
   `What This Means In Code` anchors to each concrete file; confirm the
   four-heading structure.
3. `examples/rlm_quickstart.ts` — runs against `OPENROUTER_API_KEY` when
   present, falls back to a bundled scripted LM otherwise. Exercises one
   `search` task and one `aggregate` task with memory.
4. `examples/rlm_custom_effect.ts` — shows how to register a custom
   `EffectHandler` and use it from a custom plan.
5. Consumer smoke test: `pnpm pack`; install the tarball into a sibling
   temp directory via the existing consumer-smoke harness; run both
   examples. The test fails if any type declaration is unresolvable or
   any import path is broken.

**Exit gate:** `pnpm run release:gate` green. External reader can run
the quickstart without reading the source. README conveys what v2 is in
under 200 words.

---

## 5. Test strategy

The release gate (`../benchmarks/release_gate.py`) enforces the durable
bar: golden corpus minimums (`gsm8k=50`, `squad=25`, `hotpotqa=25`, total
`>=100`), fixture-file presence, CI workflow shape, and the `dist` +
`README.md` tarball allowlist. RLM v2 hits that bar across the following
layers.

### 5.1 Unit tests

Per Phase 1–7. One test file per module. Invariants:

- Every combinator has happy / empty / malformed tests.
- Every planner function is property-tested with `fast-check` (add as
  a dev dep): `k ≥ 2`, `depth ≥ 1`, `N ≥ 1`, `resolvePlan` idempotence.
- Every evaluator branch is covered by a test asserting both the returned
  value and the trace emission.
- Every effect handler has a happy / error / malformed-args test.
- Every memory schema transition path is tested.

### 5.2 Fixture tests

Per Phase 8. `../spec/fixtures/rlm_v2_replay.json` drives deterministic
replay cases. The v2 fixture schema is a superset of the v1 schema (task
type, expected trace tag sequence, expected final, memory transitions),
so the Python port has a shape to conform to if and when it catches up.

### 5.3 Property / fuzz tests

`tests/rlm_property.test.ts`:

- **Termination:** for any well-formed plan and budget, `aforward`
  either resolves or rejects; it never hangs. Implemented via a `5s`
  `Promise.race` wrapper.
- **Depth bound:** for any `n ∈ [100, 100_000]`,
  `τ* ∈ [50, 5000]`, resulting `depth` never exceeds
  `⌈log_2(n/τ*)⌉ + 1`.
- **Budget monotonicity:** `callsUsed.current` is monotonically
  non-decreasing during a run.
- **Determinism under scripted LM:** two runs against the same
  `lm_script` produce the same final answer and the same trace-tag
  sequence (modulo timing).
- **Memory type safety:** a `WriteMemory` effect with a value that
  violates the schema never mutates the memory state.

### 5.4 Integration tests

Per Phases 5 and 7. Exactly the gap the engineer called out: happy
paths that run `RLM.aforward` against a real (OpenRouter-backed) or
mocked-but-realistic async LM. Lives in `tests/rlm_integration.test.ts`.
Extended at Phase 7 to assert memory reinjection on multi-turn oracle
sequences.

### 5.5 Consumer smoke tests

Per Phase 10. Runs from a separate sibling directory. Ensures types
resolve from the published tarball and both examples run end-to-end.

### 5.6 Benchmark tests

`bench:rlm-v2` under `package.json` compares v2 on `gsm8k`, `squad`,
`hotpotqa` (the three datasets in the release gate). Measures: final
accuracy, total oracle calls, total latency, total memory writes. The
launch acceptance bar is **v2 accuracy ≥ v1 accuracy** on every dataset;
latency and call counts are reported but not gated (quality is the
objective).

### 5.7 Legacy-surface guard

New script `pnpm run deps:rlm-legacy` greps `src/` for any of
`NodeCodeInterpreter`, `REPLHistory`, `REPLEntry`, `REPLVariable`,
`CodeSession`, `CodeInterpreter`, `ExecuteRequest`, `ExecuteResult`,
`InterpreterPatch`, `CodeInterpreterError`, `BudgetVector`,
`LLMQueryRequest`, `LLMQueryResult`, `RLMConfig`, or `node:vm`. Fails
the release gate if any production import remains. Added in Phase 0;
enforced from Phase 9.

---

## 6. Public API diff (`src/index.ts`)

Reviewable diff at the end of Phase 5, then extended at Phase 6/7 and
finalized at Phase 9.

### 6.1 Removed exports (at Phase 9)

```diff
-export type {
-  BudgetVector,
-  CodeInterpreterError,
-  REPLVariable,
-  REPLEntryKind,
-  REPLEntry,
-  REPLHistory,
-  InterpreterPatch,
-  ExecuteRequest,
-  FinalOutput,
-  ExecuteResult,
-  CodeSession,
-  CodeInterpreter,
-  LLMQueryRequest,
-  LLMQueryResult,
-  RLMConfig,
-  RLMRunResult,
-} from './rlm_types.js';
-export { NodeCodeInterpreter, createNodeCodeInterpreter } from './node_code_interpreter.js';
-export type { NodeCodeInterpreterOptions, SyncCodeInterpreter, SyncCodeSession } from './node_code_interpreter.js';
-export type { RLMOptions } from './rlm.js';
-export { RLM } from './rlm.js';
```

### 6.2 Added exports (phased: Phase 5 MVP, Phase 6/7 extensions)

```diff
+export type {
+  RLMBudget,
+  EvaluationContext,
+  EvaluationTrace,
+  RLMRunResult,
+} from './rlm_types.js';
+export { DEFAULT_BUDGET } from './rlm_types.js';
+
+export type {
+  CombinatorNode,
+  CombinatorFn,
+  CombinatorBinary,
+  CombinatorValue,
+  CombinatorList,
+  VoteReducer,
+  EnsembleReducer,
+} from './rlm_combinators.js';
+export {
+  split, peek, map, filter, reduce, concat, cross,
+  vote, ensemble, oracle, lit, vref, fn, bop,
+} from './rlm_combinators.js';
+
+export { evaluate, buildEvaluationContext } from './rlm_evaluator.js';
+
+export type { TaskType, PlanningInputs, ResolvedPlan, StaticPlan, ClassifierResult } from './rlm_task_router.js';
+export { resolvePlan, classifyTask, resolveRoute, STATIC_PLANS } from './rlm_task_router.js';
+
+export type { Effect, EffectHandler, EffectResult, OracleResponse } from './rlm_effects.js';
+export {
+  ReadContextHandler,
+  WriteMemoryHandler,
+  QueryOracleHandler,
+  YieldHandler,
+  DEFAULT_EFFECT_HANDLERS,
+  parseOracleResponse,
+} from './rlm_effects.js';
+
+export type { MemorySchema, MemoryFieldSchema, TypedMemoryState, MemoryInjector } from './rlm_memory.js';
+export {
+  initialMemoryState,
+  applyMemoryWrite,
+  defaultMemoryInjector,
+} from './rlm_memory.js';
+
+export type { RLMOptions } from './rlm.js';
+export { RLM } from './rlm.js';
```

---

## 7. Risks and mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Task classifier picks the wrong plan | Medium: accuracy regression | Medium | `taskType` override on `RLMOptions`; unknown falls back to `summarise`; low-confidence triggers beam routing (`Cross` the top-K). |
| Static plan registry is too narrow | Medium: some task shapes can't be expressed | Low | `options.plans` lets users register custom plans; custom combinator nodes authorable via the exported constructors. |
| Quality-curve seeding is wrong for small inputs | Low: degenerate decomposition | Low | Property tests clamp `k ≥ 2`, `depth ≥ 1`, `N ≥ 1`; Phase 8 calibrates curves against real datasets. |
| GEPA integration breaks after trace refactor | Medium: optimization flow no longer compiles with v2 `RLM` | Medium | `Predict` instances under oracle nodes remain GEPA-optimizable; `tests/gepa.test.ts` exercises the RLM+GEPA path end-to-end. |
| `Promise.all` fan-out overwhelms a rate-limited LM | High: production rate-limit outages | Medium | `maxParallelism` budget enforced at every `Map`, `Vote`, `Ensemble`. Default 16. Callers tighten. |
| Effects loop diverges (LLM keeps emitting effects) | Medium: latency spike | Medium | `maxEffectTurns` budget (default 8); exhaustion throws `BudgetError` with full turn trace. |
| Memory schema drift in LLM writes | Medium: invalid memory rejected mid-run | Low | `WriteMemoryHandler` returns a structured error result to the LLM on the next turn so it can retry with a valid write. |
| Runtime trace inflates memory for long runs | Low: observability regression | Low | `trackTrace: false` drops traces; default budget caps prevent unbounded growth. |
| Self-consistency amplifies rare LLM failure modes (majority is wrong) | Medium: lower accuracy in edge cases | Low | `reducer: 'verifier'` available on both `Vote` and `Ensemble`; benchmark calibration in Phase 8 catches regressions. |

---

## 8. Success criteria

V2 is considered shipped when all of these are true simultaneously:

1. `pnpm test` green on Node 20, 22, 24.
2. `pnpm run typecheck` green.
3. `pnpm run release:gate` green.
4. `./benchmarks/run_benchmarks.sh --all` green.
5. `pnpm run deps:rlm-legacy` green — no legacy symbols in `src/`.
6. `src/` contains zero imports of `node:vm`.
7. `examples/rlm_quickstart.ts` and `examples/rlm_custom_effect.ts` run
   end-to-end against a real async LM without any sync-stub fallback.
8. v2 accuracy >= v1 accuracy on all three datasets (gsm8k, squad,
   hotpotqa).
9. `docs/product/rlm-v2-architecture.md` exists and passes the DR
   four-heading template.
10. `README.md` explains v2 in fewer than 200 words.
11. `../spec/abstractions.md` §14.4 checklist item is rewritten from
    "RLM scaffold contract (§0.5)" to
    "RLM v2 functional runtime contract (§0.5)" and the sub-items
    enumerated in Phase 0's spec edit are all ticked.
12. All four runtime pillars are exercised by at least one integration
    test against a real async LM (combinator runtime, deterministic
    planner, effects handler, memory reinjection).

---

## 9. Out of scope for v0.2.0 (intentional)

- **Self-recursive nested RLM.** A plan may invoke `oracle(...)`, not
  `rlm.aforward(...)`. Recursive RLM trees are a v0.3.0 target,
  alongside the `QueryOracle` effect becoming a first-class recursive
  entrypoint rather than a single-call delegate.
- **Streaming outputs.** Every `aforward` resolves to a `Prediction`;
  incremental streaming of partial answers from `Map` is a v0.3.0
  consideration.
- **Cross-plan memory persistence.** Memory is per-`aforward` call in
  v0.2.0; persisting memory across calls (e.g., keyed on a thread id)
  is a v0.3.0 consideration.
- **Adaptive mid-run re-planning.** In v0.2.0 the planner runs once at
  the top of `aforward`; re-planning based on partial results (e.g.,
  memory-signaled "stuck" state) is a v0.3.0 feature.
- **LM registry auto-discovery.** `lmRegistry` is a user-provided Map
  in v0.2.0; a plugin-style auto-register mechanism is out of scope.

These deferrals are deliberate and the above list is exhaustive. The
v0.2.0 bar is: **eliminate the sync/async debt by design, ship all
four runtime pillars as first-class typed runtimes, and meet or beat
v1 accuracy on the full benchmark corpus.** Pillars 3 and 4 shipped
as runtimes (not stubs) is the difference between v0.2.0 being a
half-measure and being the actual vision.

---

## 10. What this plan commits us to

In one sentence per pillar:

1. **Combinator Runtime (§1, §2.1, §4 phases 1–2)** — a typed, data-
   oriented AST and async walker ship in v0.2.0; `eval` and `vm` are
   gone; `Map → Promise.all`; `Vote` and `Ensemble` are first-class
   quality primitives.
2. **Deterministic Planning + Task Routing (§2.3, §4 phases 3–4)** —
   quality-shaped math for `k*`, `d`, `N`; static plan registry with
   beam routing for low-confidence classification; the LLM picks the
   plan at most once per call.
3. **Effects-Handler (§2.4, §4 phase 6)** — oracle leaves yield typed
   `Effect`s; the runtime dispatches via named `EffectHandler`s and
   re-prompts; bounded by `maxEffectTurns`.
4. **Opinionated System Reinjection (§2.5, §4 phase 7)** — each plan
   declares a typed `MemorySchema`; the planner reinjects the current
   memory state into system messages at every oracle call; writes are
   schema-validated.
5. **No Backwards Compatibility (§0.1–§0.3, §4 phase 9, §6)** — v1 is
   deleted, not deprecated; the README and the DR state this plainly.

When v2 is shipped, the engineer's original question — `RLM.forward` vs
`RLM.aforward` — no longer exists. There is one entrypoint, and it is
unapologetically async, because the underlying runtime is a data graph
whose natural reduction is `Promise.all`, whose tool use is typed
structured output, and whose memory is a bounded typed schema reinjected
into the system prompt — and that is the only way this library earns
the claim on its `README.md` of being "tighter than the Python original."
