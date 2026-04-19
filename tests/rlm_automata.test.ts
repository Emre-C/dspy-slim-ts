/**
 * Finite-state fixtures vs runtime: `RLM.aforward`, combinator dispatch,
 * oracle effect loop (`WriteMemory` → `applyMemoryWrite`), plus unchanged
 * `query_scheduler` / `gepa_compile`. Missing `(state, event)` rows fail
 * with the unreachable pair (fixture and code must stay aligned).
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

type MachineName =
  | 'rlm_v2_control'
  | 'combinator_eval'
  | 'oracle_effect_loop'
  | 'query_scheduler'
  | 'gepa_compile';

interface AutomataFixtureCase {
  readonly id: string;
  readonly machine: MachineName;
  readonly start: string;
  readonly events: readonly string[];
  readonly expected_end: string;
}

type TransitionKey = `${string}:${string}`;

function key(state: string, event: string): TransitionKey {
  return `${state}:${event}`;
}

// ---------------------------------------------------------------------------
// rlm_v2_control — top-level facade lifecycle
// ---------------------------------------------------------------------------
//
// States:
// - INIT        — `RLM.aforward` entered; inputs validated.
// - CLASSIFY    — the async classifier call is in flight.
// - PLAN        — `resolvePlan` is substituting k/n into the template.
// - EVALUATE    — the combinator evaluator is walking the resolved AST.
// - FINAL_DONE  — `aforward` returns a `Prediction` cleanly.
// - FINAL_BUDGET_ERROR — any `BudgetError` surfaced from the evaluator.
// - FINAL_ORACLE_ERROR — any non-budget runtime error surfaced from the
//                         oracle call path (malformed LM payloads, handler
//                         exceptions, parse errors past `maxEffectTurns`).

const rlmV2ControlTransitions = new Map<TransitionKey, string>([
  [key('INIT', 'start'), 'CLASSIFY'],
  // taskType override or cached classifier result short-circuits the
  // classifier Predict call.
  [key('INIT', 'skip_classifier'), 'PLAN'],
  // Classifier path:
  [key('CLASSIFY', 'classify_ok'), 'PLAN'],
  // `options.taskType` was set; skip straight to plan resolution.
  [key('CLASSIFY', 'skip_classifier'), 'PLAN'],
  // Plan path:
  [key('PLAN', 'plan_resolved'), 'EVALUATE'],
  // Evaluate path:
  [key('EVALUATE', 'evaluate_ok'), 'FINAL_DONE'],
  [key('EVALUATE', 'evaluate_budget_err'), 'FINAL_BUDGET_ERROR'],
  [key('EVALUATE', 'evaluate_oracle_err'), 'FINAL_ORACLE_ERROR'],
  // Terminal → terminal self-loops for easy path composition:
  [key('FINAL_DONE', 'finish_ok'), 'FINAL_DONE'],
  [key('FINAL_BUDGET_ERROR', 'finish_budget'), 'FINAL_BUDGET_ERROR'],
  [key('FINAL_ORACLE_ERROR', 'finish_oracle'), 'FINAL_ORACLE_ERROR'],
  // `start` from INIT without a classifier: callers may pre-resolve the
  // classifier and enter the machine at `PLAN` (see `skip_classifier`).
  [key('INIT', 'plan_resolved'), 'EVALUATE'],
]);

// ---------------------------------------------------------------------------
// combinator_eval — dispatch table of `evaluate`
// ---------------------------------------------------------------------------
//
// The evaluator's `evaluateInner` picks exactly one handler per combinator
// node. Successful execution returns to `DONE`; any failure bubbles to
// `FAILED`. Leaves (`literal`, `var`, `split`, `peek`) complete inline so
// we collapse them into a single `dispatch_inline` → `complete` pair.

const combinatorEvalTransitions = new Map<TransitionKey, string>([
  [key('PENDING', 'dispatch_map'), 'MAP_PARALLEL'],
  [key('PENDING', 'dispatch_reduce'), 'REDUCE_SERIAL'],
  [key('PENDING', 'dispatch_vote'), 'VOTE_AGGREGATING'],
  [key('PENDING', 'dispatch_ensemble'), 'ENSEMBLE_FANOUT'],
  [key('PENDING', 'dispatch_oracle'), 'ORACLE_INVOKING'],
  [key('PENDING', 'dispatch_inline'), 'ORACLE_INVOKING'],
  [key('MAP_PARALLEL', 'map_leaves_ok'), 'DONE'],
  [key('MAP_PARALLEL', 'map_leaves_err'), 'FAILED'],
  [key('REDUCE_SERIAL', 'reduce_step_ok'), 'REDUCE_SERIAL'],
  [key('REDUCE_SERIAL', 'reduce_err'), 'FAILED'],
  [key('REDUCE_SERIAL', 'complete'), 'DONE'],
  [key('VOTE_AGGREGATING', 'vote_lanes_ok'), 'DONE'],
  [key('VOTE_AGGREGATING', 'vote_lanes_err'), 'FAILED'],
  [key('ENSEMBLE_FANOUT', 'ensemble_models_ok'), 'DONE'],
  [key('ENSEMBLE_FANOUT', 'ensemble_models_err'), 'FAILED'],
  [key('ORACLE_INVOKING', 'oracle_ok'), 'DONE'],
  [key('ORACLE_INVOKING', 'oracle_err'), 'FAILED'],
  // Terminal completion for any path that produces its final value inline.
  [key('MAP_PARALLEL', 'complete'), 'DONE'],
  [key('VOTE_AGGREGATING', 'complete'), 'DONE'],
  [key('ENSEMBLE_FANOUT', 'complete'), 'DONE'],
  [key('ORACLE_INVOKING', 'complete'), 'DONE'],
  [key('DONE', 'noop'), 'DONE'],
  [key('FAILED', 'noop'), 'FAILED'],
]);

// ---------------------------------------------------------------------------
// oracle_effect_loop — per-turn oracle lifecycle
// ---------------------------------------------------------------------------
//
// Each turn inside `callOracleLeafWithEffects` does:
// 1. `TURN_START` → call the Predict wrapper → `PREDICT`.
// 2. `PREDICT` → parse response as `OracleResponse` → `PARSE`.
// 3. `PARSE` → either return a value (terminal) or dispatch the emitted
//              effect through the handler registry.
// 4. Handlers fall into two tracked families:
//    - `WriteMemory` takes a dedicated edge because it also mutates the
//      typed memory cell and alters the next turn's system message.
//    - Every other handler (`ReadContext`, `QueryOracle`, `Search`,
//      `Yield`, `Custom`) collapses into `dispatch_handler_generic`.
// 5. A turn ends by looping back to `TURN_START` until either a `value`
//    resolves the oracle (`DONE`) or the effect-turn budget is exhausted
//    (`FAILED_BUDGET`).

const oracleEffectLoopTransitions = new Map<TransitionKey, string>([
  [key('TURN_START', 'call_predict'), 'PREDICT'],
  [key('PREDICT', 'parse_ok_value'), 'PARSE'],
  [key('PREDICT', 'parse_ok_effect'), 'PARSE'],
  [key('PREDICT', 'parse_err'), 'PARSE'],
  [key('PARSE', 'parse_ok_value'), 'PARSE'],
  [key('PARSE', 'parse_ok_effect'), 'PARSE'],
  [key('PARSE', 'finish'), 'DONE'],
  [key('PARSE', 'dispatch_handler_write_memory'), 'HANDLE_EFFECT'],
  [key('PARSE', 'dispatch_handler_generic'), 'HANDLE_EFFECT'],
  [key('HANDLE_EFFECT', 'memory_write_ok'), 'MEMORY_WRITE'],
  [key('HANDLE_EFFECT', 'memory_write_err'), 'MEMORY_WRITE'],
  [key('MEMORY_WRITE', 'next_turn'), 'TURN_START'],
  [key('HANDLE_EFFECT', 'next_turn'), 'TURN_START'],
  [key('PARSE', 'budget_exhausted'), 'FAILED_BUDGET'],
  [key('HANDLE_EFFECT', 'budget_exhausted'), 'FAILED_BUDGET'],
  [key('TURN_START', 'budget_exhausted'), 'FAILED_BUDGET'],
  // Convenience terminal passthroughs.
  [key('DONE', 'noop'), 'DONE'],
  [key('FAILED_BUDGET', 'noop'), 'FAILED_BUDGET'],
]);

// ---------------------------------------------------------------------------
// query_scheduler / gepa_compile — substrate machines kept from v1
// ---------------------------------------------------------------------------

const querySchedulerTransitions = new Map<TransitionKey, string>([
  [key('IDLE', 'enqueue'), 'QUEUED'],
  [key('QUEUED', 'dispatch'), 'RUNNING'],
  [key('RUNNING', 'resolve_ok'), 'SUCCEEDED'],
  [key('RUNNING', 'resolve_err'), 'FAILED'],
  [key('QUEUED', 'abort'), 'CANCELLED'],
  [key('RUNNING', 'abort'), 'CANCELLED'],
  [key('SUCCEEDED', 'drain'), 'IDLE'],
  [key('FAILED', 'drain'), 'IDLE'],
  [key('CANCELLED', 'drain'), 'IDLE'],
]);

const gepaCompileTransitions = new Map<TransitionKey, string>([
  [key('INIT', 'gate'), 'GATED'],
  [key('INIT', 'project'), 'PROJECT'],
  [key('PROJECT', 'trace'), 'TRACE'],
  [key('PROJECT', 'build_dataset'), 'DATASET'],
  [key('TRACE', 'build_dataset'), 'DATASET'],
  [key('DATASET', 'mutate'), 'MUTATE'],
  [key('MUTATE', 'dispatch_eval'), 'EVALUATE'],
  [key('EVALUATE', 'eval_ok'), 'FRONTIER'],
  [key('EVALUATE', 'eval_err'), 'FAILED'],
  [key('FRONTIER', 'update_frontier'), 'MUTATE'],
  [key('FRONTIER', 'commit'), 'COMMIT'],
  [key('COMMIT', 'stop'), 'DONE'],
]);

const transitionTables: Record<MachineName, ReadonlyMap<TransitionKey, string>> = {
  rlm_v2_control: rlmV2ControlTransitions,
  combinator_eval: combinatorEvalTransitions,
  oracle_effect_loop: oracleEffectLoopTransitions,
  query_scheduler: querySchedulerTransitions,
  gepa_compile: gepaCompileTransitions,
};

function runMachine(
  machine: MachineName,
  start: string,
  events: readonly string[],
): string {
  let state = start;
  const table = transitionTables[machine];
  for (const event of events) {
    const next = table.get(key(state, event));
    if (next === undefined) {
      throw new Error(
        `No transition defined for machine=${machine}, state=${state}, event=${event}.`,
      );
    }
    state = next;
  }
  return state;
}

const fixture = JSON.parse(
  readFileSync(
    new URL('../../spec/fixtures/rlm_gepa_automata.json', import.meta.url),
    'utf-8',
  ),
) as { readonly cases: readonly AutomataFixtureCase[] };

describe('RLM v2 automata fixtures', () => {
  for (const testCase of fixture.cases) {
    it(testCase.id, () => {
      const finalState = runMachine(
        testCase.machine,
        testCase.start,
        testCase.events,
      );
      expect(finalState).toBe(testCase.expected_end);
    });
  }
});
