/**
 * Shared RLM substrate contracts.
 *
 * This file currently carries two eras of contracts:
 *
 * 1. The **legacy REPL contract** (RLM v1). Types in the `// --- Legacy (v1) ---`
 *    block below are imported by `src/rlm.ts` and `src/node_code_interpreter.ts`
 *    to keep the v1 runtime compiling during the v2 rollout. Phase 9 deletes
 *    this block wholesale along with `node_code_interpreter.ts`.
 * 2. The **v2 typed functional runtime** (RLM v2). Types in the `// --- v2 ---`
 *    block below define the canonical contract per
 *    `docs/RLM_V2_IMPLEMENTATION_PLAN.md §2.2` and `spec/abstractions.md §0.5`.
 *
 * The v2 types are intentionally not re-exported from `src/index.ts` until
 * Phase 5 lands the public `RLM` facade.
 */

import type { BaseLM } from './lm.js';
import type { Signature } from './signature.js';
import type { CombinatorNode, CombinatorValue } from './rlm_combinators.js';

// ============================================================================
// --- Legacy (v1) ---
// The REPL-shaped runtime. Scheduled for deletion in Phase 9 per
// docs/RLM_V2_IMPLEMENTATION_PLAN.md §0.3 and §4 phase 9.
// ============================================================================

export interface BudgetVector {
  readonly maxIterations: number;
  readonly maxLlmCalls: number;
  readonly maxBatchWidth: number;
}

export interface CodeInterpreterError {
  readonly kind: 'runtime' | 'budget' | 'serialization' | 'protocol';
  readonly fatal: boolean;
  readonly step: number | null;
  readonly cause: unknown;
}

export interface REPLVariable<TValue = unknown> {
  readonly symbol: string;
  readonly value: TValue;
  readonly mutable: boolean;
}

export type REPLEntryKind =
  | 'exec'
  | 'query'
  | 'batch_query'
  | 'submit'
  | 'extract'
  | 'fault';

export interface REPLEntry {
  readonly step: number;
  readonly kind: REPLEntryKind;
  readonly ok: boolean;
}

export interface REPLHistory {
  readonly entries: readonly REPLEntry[];
  readonly liveSymbols: readonly string[];
}

export interface InterpreterPatch<TValue = unknown> {
  readonly bindings: Readonly<Record<string, TValue>>;
}

export interface ExecuteRequest {
  readonly step: number;
  readonly source: unknown;
  readonly budget: BudgetVector;
  readonly allowSubmit: boolean;
}

export interface FinalOutput<TOutput> {
  readonly value: TOutput;
  readonly via: 'submit' | 'extract';
}

export type ExecuteResult<TOutput> =
  | {
      readonly tag: 'continue';
      readonly historyDelta: readonly REPLEntry[];
    }
  | {
      readonly tag: 'submit';
      readonly historyDelta: readonly REPLEntry[];
      readonly output: FinalOutput<TOutput>;
    }
  | {
      readonly tag: 'fault';
      readonly historyDelta: readonly REPLEntry[];
      readonly liveVariables: readonly REPLVariable[];
      readonly error: CodeInterpreterError;
    };

export interface CodeSession<TSnapshot = unknown, TValue = unknown> {
  readonly execute: <TOutput>(
    request: ExecuteRequest,
  ) => Promise<ExecuteResult<TOutput>>;
  readonly inspectGlobals: () => Promise<readonly REPLVariable<TValue>[]>;
  readonly snapshotGlobals: () => Promise<TSnapshot>;
  readonly patchGlobals: (patch: InterpreterPatch<TValue>) => Promise<void>;
  readonly close: () => Promise<void>;
}

export interface CodeInterpreter<TSnapshot = unknown, TValue = unknown> {
  readonly createSession: () => Promise<CodeSession<TSnapshot, TValue>>;
}

export interface LLMQueryRequest<TInput = unknown> {
  readonly requestId: number;
  readonly payload: TInput;
}

export interface LLMQueryResult<TOutput = unknown> {
  readonly requestId: number;
  readonly ok: boolean;
  readonly output: TOutput | null;
}

export interface RLMConfig {
  readonly budget: BudgetVector;
  readonly trackTrace: boolean;
  readonly reservedToolNames: readonly string[];
  readonly subLmResolution: 'instance_then_settings';
}

export interface RLMRunResult<TOutput> {
  readonly output: FinalOutput<TOutput> | null;
  readonly history: REPLHistory;
  readonly error: CodeInterpreterError | null;
}

// ============================================================================
// --- v2 ---
// The typed functional runtime. Canonical per
// docs/RLM_V2_IMPLEMENTATION_PLAN.md §2.2 and spec/abstractions.md §0.5.
// ============================================================================

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
 * See `docs/RLM_V2_IMPLEMENTATION_PLAN.md §0.6` for the rationale behind each
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

/**
 * Single trace record emitted by the evaluator after each node completes
 * (success or failure). Append-only; surviving `BudgetError`s so partial
 * failures are debuggable.
 *
 * `step` equals the index of the entry in the owning `trace` array at the
 * moment it was pushed. Because `Map` / `Vote` / `Ensemble` run in parallel
 * via `Promise.all`, sibling entries may interleave; the contract is that
 * `step` always reflects completion order, not invocation order.
 */
export interface EvaluationTrace {
  readonly step: number;
  readonly nodeTag: CombinatorNode['tag'];
  readonly startedAt: string;
  readonly durationMs: number;
  readonly ok: boolean;
  readonly cause?: unknown;
  readonly extras?: Readonly<Record<string, unknown>>;
}

/**
 * Structured tool-use intent emitted by an oracle leaf. The typed contract
 * (`ReadContext`, `WriteMemory`, `QueryOracle`, `Search`, `Yield`, `Custom`)
 * lands in `src/rlm_effects.ts` at Phase 6. The `unknown` at this layer is a
 * deliberate forward declaration that lets the evaluator thread handlers
 * through the context without coupling to the still-to-ship Effect union.
 */
export type Effect = unknown;

/**
 * Outcome of dispatching an `Effect` through an `EffectHandler`. Phase 6
 * narrows this to a tagged result shape
 * `{ ok: true; value: unknown } | { ok: false; error: string }`; at Phase 1
 * the forward declaration keeps the context interface stable.
 */
export type EffectResult = unknown;

/**
 * Dispatcher protocol for structured tool use. Phase 6 ships the built-in
 * handlers (`ReadContextHandler`, `WriteMemoryHandler`, `QueryOracleHandler`,
 * `YieldHandler`) and the parser path. The shape here is the minimum the
 * evaluator needs to route effects without reshaping `EvaluationContext`.
 */
export interface EffectHandler {
  readonly name: string;
  handle(effect: Effect, ctx: EvaluationContext): Promise<EffectResult>;
}

/**
 * Typed per-`aforward` memory state. Each plan attaches a `MemorySchema`
 * (declared in `src/rlm_memory.ts` at Phase 7) that governs legal keys,
 * types, and length bounds; this map is the runtime projection of a schema
 * instance. The shape — a readonly Map keyed by field name — is stable
 * across phases.
 */
export type TypedMemoryState = ReadonlyMap<string, unknown>;

/**
 * Runtime context threaded through every evaluator call. Fields are grouped
 * by responsibility:
 *
 * - `budget`, `lm`, `lmRegistry`, `signature` are configuration; immutable
 *   for the lifetime of the run.
 * - `scope`, `depth`, `memory` are lexical state; copy-on-descent.
 * - `callsUsed`, `trace` are mutable aggregates; shared across the entire
 *   eval tree via object identity so `BudgetError` propagation preserves
 *   the partial trace.
 * - `handlers` is the dispatch table populated from the resolved plan plus
 *   `RLMOptions.handlers`; Phase 6 fills it in, Phase 1 ships an empty map.
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
  readonly memory: TypedMemoryState;
}
