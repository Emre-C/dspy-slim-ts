import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

type MachineName =
  | 'interpreter_session'
  | 'rlm_control'
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

const interpreterTransitions = new Map<TransitionKey, string>([
  [key('ABSENT', 'create'), 'OPEN'],
  [key('OPEN', 'execute_ok'), 'OPEN'],
  [key('OPEN', 'execute_fault'), 'FAULTED'],
  [key('OPEN', 'snapshot'), 'OPEN'],
  [key('OPEN', 'patch'), 'OPEN'],
  [key('OPEN', 'inspect'), 'OPEN'],
  [key('OPEN', 'close'), 'CLOSED'],
  [key('FAULTED', 'close'), 'CLOSED'],
  [key('CLOSED', 'close'), 'CLOSED'],
]);

const rlmControlTransitions = new Map<TransitionKey, string>([
  [key('INIT', 'boot'), 'STEP'],
  [key('STEP', 'choose_exec'), 'EXEC'],
  [key('STEP', 'choose_query'), 'QUERY'],
  [key('STEP', 'choose_extract'), 'EXTRACT'],
  [key('EXEC', 'exec_continue'), 'STEP'],
  [key('EXEC', 'exec_submit'), 'SUBMIT'],
  [key('EXEC', 'exec_fault'), 'FAULT'],
  [key('QUERY', 'query_return'), 'STEP'],
  [key('QUERY', 'query_fault'), 'FAULT'],
  [key('EXTRACT', 'extract_submit'), 'SUBMIT'],
  [key('EXTRACT', 'extract_fail'), 'FAULT'],
  [key('SUBMIT', 'commit'), 'HALT'],
  [key('FAULT', 'abort'), 'HALT'],
]);

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
  interpreter_session: interpreterTransitions,
  rlm_control: rlmControlTransitions,
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
    if (!next) {
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

describe('RLM/GEPA automata fixtures', () => {
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
