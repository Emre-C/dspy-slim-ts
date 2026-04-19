/**
 * RLM v2 substrate contracts.
 *
 * The canonical types for the typed functional runtime per
 * `docs/product/rlm-v2-architecture.md` §2.2 and
 * `../../spec/abstractions.md §0.5`. Legacy v1 interpreter types stay
 * out via `tools/deps_rlm_legacy.mjs --strict`.
 */

import { ValueError } from './exceptions.js';
import type { BaseLM } from './lm.js';
import type { Signature } from './signature.js';
import type { JsonObject, JsonValue } from './json_value.js';
import type { CombinatorValue } from './rlm_combinators.js';
import type { MemorySchema, TypedMemoryState } from './rlm_memory.js';
import type { EvaluationTrace } from './rlm_trace_types.js';

export type { EvaluationTrace } from './rlm_trace_types.js';

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

/**
 * Execution budget for a single `aforward` call. All fields are hard bounds.
 * Budget exhaustion throws `BudgetError` with the partial `EvaluationTrace`
 * attached to the surrounding context.
 *
 * Defaults (see `DEFAULT_BUDGET`) are intentionally generous: RLM v2's
 * objective function is quality, not cost. Users tighten when they need to.
 */
export interface RLMBudget {
  /** Max oracle (neural leaf) calls across the entire run. */
  readonly maxOracleCalls: number;
  /** Max parallel fan-out at any single `Map` / `Vote` / `Ensemble`. */
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

/**
 * The canonical default budget. Frozen to guard against accidental mutation.
 * See `docs/product/rlm-v2-architecture.md` §0.6 for the rationale behind each
 * number.
 */
export const DEFAULT_BUDGET: RLMBudget = Object.freeze({
  maxOracleCalls: 200,
  maxParallelism: 16,
  maxDepth: 6,
  leafThreshold: 1000,
  selfConsistencyN: 5,
  maxEffectTurns: 8,
});

function mergeBudgetInt(
  value: number | undefined,
  fallback: number,
  fieldName: string,
  min: number,
): number {
  if (value === undefined) {
    return fallback;
  }
  if (!Number.isInteger(value) || value < min) {
    throw new ValueError(
      `RLM ${fieldName} must be an integer >= ${min}, got ${String(value)}.`,
    );
  }
  return value;
}

/**
 * Merge partial user overrides onto {@link DEFAULT_BUDGET}. Explicit
 * numeric fields are validated per field (`maxOracleCalls` / `maxDepth` /
 * `maxEffectTurns` allow 0; parallelism and sampling floors stay ≥ 1).
 * `undefined` fields fall back to defaults. Used by the RLM facade and
 * `buildEvaluationContext`.
 */
export function mergeBudget(override?: Partial<RLMBudget>): RLMBudget {
  if (override === undefined) return DEFAULT_BUDGET;
  return Object.freeze({
    maxOracleCalls: mergeBudgetInt(
      override.maxOracleCalls,
      DEFAULT_BUDGET.maxOracleCalls,
      'budget.maxOracleCalls',
      0,
    ),
    maxParallelism: mergeBudgetInt(
      override.maxParallelism,
      DEFAULT_BUDGET.maxParallelism,
      'budget.maxParallelism',
      1,
    ),
    maxDepth: mergeBudgetInt(
      override.maxDepth,
      DEFAULT_BUDGET.maxDepth,
      'budget.maxDepth',
      0,
    ),
    leafThreshold: mergeBudgetInt(
      override.leafThreshold,
      DEFAULT_BUDGET.leafThreshold,
      'budget.leafThreshold',
      1,
    ),
    selfConsistencyN: mergeBudgetInt(
      override.selfConsistencyN,
      DEFAULT_BUDGET.selfConsistencyN,
      'budget.selfConsistencyN',
      1,
    ),
    maxEffectTurns: mergeBudgetInt(
      override.maxEffectTurns,
      DEFAULT_BUDGET.maxEffectTurns,
      'budget.maxEffectTurns',
      0,
    ),
  });
}

// ---------------------------------------------------------------------------
// Effects
// ---------------------------------------------------------------------------

/**
 * Structured tool-use intent emitted by an oracle leaf. The LLM returns an
 * `Effect` (instead of a plain answer) whenever it wants to read from the
 * prompt context, persist structured memory, delegate to another oracle,
 * search an external source, pause, or invoke a user-provided custom tool.
 *
 * This union is the **source of truth** for the effect protocol; the
 * parser, `EFFECT_ORACLE_SIGNATURE`, and all built-in handlers in
 * `src/rlm_effects.ts` are implementations of the contract declared here.
 *
 * Design notes (see `docs/product/rlm-v2-architecture.md` §2.4):
 *
 * - `Custom` is the open-world escape hatch. Its `name` selects the
 *   handler (with fall-through to any handler registered under the literal
 *   `'Custom'` key); `args` is an arbitrary JSON object that each custom
 *   handler validates internally.
 * - `QueryOracle` intentionally nests an oracle call inside an oracle
 *   call. The handler uses a **plain** (non-effect-loop) oracle path so
 *   sub-queries cannot recursively spawn another nested effect loop.
 */
export type Effect =
  | {
      readonly kind: 'ReadContext';
      readonly name: string;
      readonly start?: number;
      readonly end?: number;
    }
  | {
      readonly kind: 'WriteMemory';
      readonly key: string;
      readonly value: JsonValue;
    }
  | {
      readonly kind: 'QueryOracle';
      readonly prompt: string;
      readonly modelHint?: string;
    }
  | {
      readonly kind: 'Search';
      readonly query: string;
      readonly topK?: number;
    }
  | { readonly kind: 'Yield' }
  | {
      readonly kind: 'Custom';
      readonly name: string;
      readonly args: JsonObject;
    };

/**
 * Outcome of dispatching an `Effect` through an `EffectHandler`.
 *
 * The tagged shape — as opposed to a thrown exception — is deliberate: the
 * evaluator formats `{ ok: false, error }` results as a structured block in
 * the next-turn prompt so the LLM can see **what** went wrong and retry
 * with a corrected effect. Thrown errors bypass that recovery path and
 * terminate the whole `aforward` run.
 *
 * Handlers should therefore prefer `{ ok: false }` over `throw` for
 * recoverable failures (validation, unknown key, etc.) and let exceptions
 * escape only for genuinely fatal conditions (e.g., `BudgetError`).
 */
export type EffectResult =
  | { readonly ok: true; readonly value: JsonValue }
  | { readonly ok: false; readonly error: string };

/**
 * Response shape returned by `parseOracleResponse` for every oracle
 * completion. Either the LLM emitted a terminal answer (`kind: 'value'`)
 * or requested an effect (`kind: 'effect'`); in the latter case the
 * evaluator dispatches the effect through the handler registry and
 * re-enters the loop.
 */
export type OracleResponse =
  | { readonly kind: 'value'; readonly value: string }
  | { readonly kind: 'effect'; readonly effect: Effect };

/**
 * Dispatcher protocol for structured tool use. Implementations live in
 * `rlm_effects.ts`; the evaluator routes by `handler.name` without reshaping
 * `EvaluationContext`.
 */
export interface EffectHandler {
  readonly name: string;
  handle(effect: Effect, ctx: EvaluationContext): Promise<EffectResult>;
}

// ---------------------------------------------------------------------------
// Evaluation context
// ---------------------------------------------------------------------------

/**
 * Runtime context threaded through every evaluator call. Fields are grouped
 * by responsibility:
 *
 * - `budget`, `lm`, `lmRegistry`, `signature`, `memorySchema` are
 *   configuration; immutable for the lifetime of the run.
 * - `scope`, `depth` are lexical state; copy-on-descent.
 * - `callsUsed`, `trace`, `memoryCell` are mutable aggregates; shared
 *   across the entire eval tree via object identity so `BudgetError`
 *   propagation preserves the partial trace AND so effect handlers can
 *   update memory visible to every subsequent oracle call inside the
 *   same plan tree.
 * - `handlers` is the dispatch table (built-ins merged with
 *   `RLMOptions.handlers`, last write wins on duplicate names).
 */
export interface EvaluationContext {
  readonly budget: RLMBudget;
  readonly lm: BaseLM;
  readonly lmRegistry: ReadonlyMap<string, BaseLM>;
  readonly signature: Signature;
  readonly scope: ReadonlyMap<string, CombinatorValue>;
  readonly depth: number;
  readonly callsUsed: { current: number };
  readonly trace: EvaluationTrace[];
  readonly handlers: ReadonlyMap<string, EffectHandler>;
  /**
   * Mutable memory cell. The `current` reference holds the active
   * `TypedMemoryState` (immutable snapshot). `WriteMemory`
   * replaces `.current` with a new snapshot on every successful write
   * so subsequent oracle calls observe the update. When no memory
   * schema is in play the cell still exists but `current` is a shared
   * immutable empty snapshot.
   */
  readonly memoryCell: { current: TypedMemoryState };
  /**
   * Active `MemorySchema` for this plan tree, or `null` if undeclared.
   * `WriteMemory` handlers use it to validate writes.
   */
  readonly memorySchema: MemorySchema | null;
}

/**
 * Late-bound oracle callback for `QueryOracleHandler`. Lives here (not in
 * `rlm_effects.ts`) so `rlm_evaluator` can depend on the shape without
 * importing the effects implementation module at type-only boundaries.
 */
export type QueryOracleCallFn = (
  prompt: string,
  modelHint: string | undefined,
  ctx: EvaluationContext,
) => Promise<string>;

/**
 * Options accepted by `buildEvaluationContext` (`rlm_evaluator.ts`). Mirrors
 * the mutable slices of {@link EvaluationContext} that callers may supply
 * before defaults are filled in.
 */
export interface BuildEvaluationContextOptions {
  readonly lm: BaseLM;
  readonly signature: Signature;
  readonly budget?: Partial<RLMBudget>;
  readonly lmRegistry?: ReadonlyMap<string, BaseLM>;
  readonly scope?: ReadonlyMap<string, CombinatorValue>;
  readonly handlers?: ReadonlyMap<string, EffectHandler>;
  readonly memory?: TypedMemoryState;
  readonly memorySchema?: MemorySchema | null;
}
