/**
 * RLM v2 evaluation trace types (extracted module).
 *
 * Lives in a dedicated file so GEPA and other optimizers can depend on the
 * structural trace contract (`EvaluationTrace`) without importing the full
 * `rlm_types` substrate (budget, effects, evaluation context). This keeps the
 * dependency graph shallow: `rlm_combinators` → `rlm_trace_types` ←
 * `gepa_types`, with no back-edge into GEPA.
 */

import type { CombinatorNode } from './rlm_combinators.js';
import type { JsonObject } from './json_value.js';

// ---------------------------------------------------------------------------
// Trace
// ---------------------------------------------------------------------------

/**
 * Virtual trace tags for evaluator events that are not combinator nodes.
 * Today: `'effect'` once per effect-loop turn inside an oracle leaf.
 */
export type VirtualTraceTag = 'effect';

/**
 * Full set of tags an `EvaluationTrace` entry may carry. Either a real
 * `CombinatorNode` tag (literal/var/split/...) or a `VirtualTraceTag`.
 */
export type TraceTag = CombinatorNode['tag'] | VirtualTraceTag;

/**
 * Single trace record emitted by the evaluator after each node completes
 * (success or failure). Append-only; surviving `BudgetError`s so partial
 * failures are debuggable.
 *
 * `step` equals the index of the entry in the owning `trace` array at the
 * moment it was pushed. Because `Map` / `Vote` / `Ensemble` run in parallel
 * via `Promise.all`, sibling entries may interleave; the contract is that
 * `step` always reflects completion order, not invocation order.
 *
 * `nodeTag === 'effect'` entries are emitted once per effect-loop turn
 * inside an oracle leaf. Their `extras` field carries
 * `{ turn, effectKind, ok }` so the trace-replay tooling can reconstruct
 * the tool-use timeline without parsing raw completions.
 */
export interface EvaluationTrace {
  readonly step: number;
  readonly nodeTag: TraceTag;
  readonly startedAt: string;
  readonly durationMs: number;
  readonly ok: boolean;
  readonly cause?: unknown;
  readonly extras?: JsonObject;
}
