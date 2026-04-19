/**
 * Deterministic replay from `../../spec/fixtures/rlm_v2_replay.json`.
 * Each case scripts classifier + oracle turns; asserts trace tags and call
 * counts against `expected`. Planner/router/evaluator regressions show up
 * as fixture drift (`tools/record_rlm_v2.ts` refreshes observed blocks).
 */
import { readFileSync } from 'node:fs';
import { afterEach, describe, expect, it } from 'vitest';

import type { Message } from '../src/chat_message.js';
import { BaseLM, type LMOutput } from '../src/lm.js';
import { RLM, type RLMOptions } from '../src/rlm.js';
import { settings } from '../src/settings.js';
import {
  oracle,
  vref,
  type CombinatorNode,
} from '../src/rlm_combinators.js';
import type {
  EffectHandler,
  EvaluationTrace,
  RLMBudget,
} from '../src/rlm_types.js';
import type {
  MemorySchema,
} from '../src/rlm_memory.js';
import type { StaticPlan, TaskType } from '../src/rlm_task_router.js';

// ---------------------------------------------------------------------------
// Fixture schema
// ---------------------------------------------------------------------------

interface ClassifierPayload {
  readonly primary: string;
  readonly confidence: number;
  readonly candidates: readonly string[];
}

type OracleQueueEntry = string | Record<string, unknown>;

interface LMScript {
  readonly classifier: ClassifierPayload | null;
  readonly oracle_queue: readonly OracleQueueEntry[];
}

interface ExpectedMemoryTransition {
  readonly turn: number;
  readonly key: string;
  readonly value: string;
}

interface ReplayExpectation {
  readonly final_contains: string;
  readonly final_contains_all?: readonly string[];
  readonly classifier_calls: number;
  readonly oracle_calls_min: number;
  readonly oracle_calls_max: number;
  readonly route_kind: 'single' | 'beam';
  readonly route_task_types: readonly string[];
  readonly trace_has: readonly string[];
  readonly handler_search_queries?: readonly string[];
  readonly memory_transitions?: readonly ExpectedMemoryTransition[];
  readonly memory_banner_in_turn?: {
    readonly turn: number;
    readonly contains: string;
  };
}

interface CaseOptions {
  readonly taskType?: TaskType;
  readonly handlers?: readonly string[];
  readonly plan_id?: string;
}

interface ReplayCase {
  readonly id: string;
  readonly task_type: string;
  readonly signature: string;
  readonly inputs: Record<string, string>;
  readonly options?: CaseOptions;
  readonly budget?: Partial<RLMBudget>;
  readonly lm_script: LMScript;
  readonly expected: ReplayExpectation;
}

interface FixtureFile {
  readonly cases: readonly ReplayCase[];
}

// ---------------------------------------------------------------------------
// ScriptedReplayLM — replays the fixture's scripts verbatim
// ---------------------------------------------------------------------------

/**
 * Per-turn record of every LM invocation the RLM makes. The role
 * key lets tests filter `classifier` vs `oracle` calls. `systemContent`
 * is captured separately so tests can assert memory-banner reinjection
 * without scanning the concatenated prompt.
 */
interface CapturedCall {
  readonly role: 'classifier' | 'oracle';
  readonly userContent: string;
  readonly systemContent: string;
  readonly fullContent: string;
}

class ScriptedReplayLM extends BaseLM {
  readonly calls: CapturedCall[] = [];
  readonly classifier: ClassifierPayload | null;
  readonly oracleQueue: OracleQueueEntry[];

  constructor(script: LMScript) {
    super({ model: 'scripted-replay' });
    this.classifier = script.classifier;
    this.oracleQueue = [...script.oracle_queue];
  }

  protected override async agenerate(
    _prompt?: string,
    messages?: readonly Message[],
    _kwargs?: Record<string, unknown>,
  ): Promise<readonly LMOutput[]> {
    await Promise.resolve();
    const systemContent = extractSystemContent(messages);
    const userContent = extractUserContent(messages);
    const fullContent = allMessageContent(messages);
    const isClassifier = fullContent.includes('Classify the user request');
    this.calls.push({
      role: isClassifier ? 'classifier' : 'oracle',
      userContent,
      systemContent,
      fullContent,
    });
    if (isClassifier) {
      if (this.classifier === null) {
        throw new Error(
          'ScriptedReplayLM: classifier invoked but no payload scripted',
        );
      }
      return [JSON.stringify(this.classifier)];
    }
    const next = this.oracleQueue.shift();
    if (next === undefined) {
      throw new Error(
        `ScriptedReplayLM: oracle queue exhausted at call #${this.calls.length}`,
      );
    }
    return [formatOracleOutput(next, fullContent)];
  }

  get oracleCalls(): readonly CapturedCall[] {
    return this.calls.filter((c) => c.role === 'oracle');
  }

  get classifierCalls(): readonly CapturedCall[] {
    return this.calls.filter((c) => c.role === 'classifier');
  }
}

/**
 * Format an oracle-queue entry as a JSON completion matching the
 * output-field shape the current signature expects. A string entry is
 * the terminal "answer" payload; an object entry is passed through as
 * raw JSON (used for effect-emission entries and for exotic signatures
 * the fixture needs to mimic verbatim).
 */
function formatOracleOutput(
  entry: OracleQueueEntry,
  fullContent: string,
): string {
  if (typeof entry !== 'string') {
    return JSON.stringify(entry);
  }
  if (containsOutputField(fullContent, 'kind')) {
    return JSON.stringify({
      kind: 'value',
      value: entry,
      effect_name: null,
      effect_args: null,
    });
  }
  if (containsOutputField(fullContent, 'verdict')) {
    return JSON.stringify({ verdict: true });
  }
  if (containsOutputField(fullContent, 'confidence')) {
    return JSON.stringify({ answer: entry, confidence: 0.5 });
  }
  return JSON.stringify({ answer: entry });
}

function containsOutputField(content: string, name: string): boolean {
  const outputsBlockStart = content.indexOf('Your output fields are:');
  if (outputsBlockStart < 0) return false;
  const block = content.slice(outputsBlockStart);
  return block.includes(`- \`${name}\` (`);
}

function extractSystemContent(
  messages: readonly Message[] | undefined,
): string {
  if (!messages) return '';
  for (const msg of messages) {
    if (msg.role !== 'system') continue;
    if (typeof msg.content === 'string') return msg.content;
    return msg.content
      .map((part) => (part.type === 'text' ? (part.text ?? '') : ''))
      .join('\n');
  }
  return '';
}

function extractUserContent(
  messages: readonly Message[] | undefined,
): string {
  if (!messages) return '';
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i]!;
    if (msg.role === 'user' && typeof msg.content === 'string') {
      return msg.content;
    }
  }
  return typeof messages[0]?.content === 'string'
    ? (messages[0].content as string)
    : '';
}

function allMessageContent(
  messages: readonly Message[] | undefined,
): string {
  if (!messages) return '';
  return messages
    .map((m) => {
      if (typeof m.content === 'string') return m.content;
      return m.content
        .map((part) => (part.type === 'text' ? (part.text ?? '') : ''))
        .join('\n');
    })
    .join('\n');
}

// ---------------------------------------------------------------------------
// Handler and plan registries
// ---------------------------------------------------------------------------

/**
 * Per-case state that needs to be observable after the run (handler
 * invocation histories, memory observations). Keyed by handler stub
 * name in the fixture.
 */
interface HandlerFactoryContext {
  readonly searchQueries: string[];
}

function buildHandlerByName(
  name: string,
  hctx: HandlerFactoryContext,
): EffectHandler {
  switch (name) {
    case 'search_handler_stub':
      return {
        name: 'Search',
        async handle(effect) {
          if (effect.kind !== 'Search') {
            return { ok: false, error: 'Search handler: wrong kind' };
          }
          hctx.searchQueries.push(effect.query);
          return {
            ok: true,
            value: 'needle result for ' + effect.query,
          };
        },
      };
    default:
      throw new Error(`Unknown handler stub: ${name}`);
  }
}

const MEMORY_BEARING_PAIRWISE_SCHEMA: MemorySchema = Object.freeze({
  name: 'replay_memory',
  fields: Object.freeze([
    Object.freeze({
      name: 'note',
      type: 'string' as const,
      description: 'A scratch note written by the oracle.',
      maxLength: 120,
    }),
  ]),
  maxBytesSerialized: 512,
});

function buildPlanById(planId: string): StaticPlan {
  switch (planId) {
    case 'memory_bearing_pairwise':
      return Object.freeze({
        taskType: 'pairwise' as const,
        template: oracle(vref('input'), undefined) satisfies CombinatorNode,
        memorySchema: MEMORY_BEARING_PAIRWISE_SCHEMA,
      });
    default:
      throw new Error(`Unknown plan_id: ${planId}`);
  }
}

// ---------------------------------------------------------------------------
// Fixture loader
// ---------------------------------------------------------------------------

const FIXTURE: FixtureFile = JSON.parse(
  readFileSync(
    new URL('../../spec/fixtures/rlm_v2_replay.json', import.meta.url),
    'utf-8',
  ),
) as FixtureFile;

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

afterEach(() => {
  settings.reset();
});

describe('RLM v2 replay fixture', () => {
  for (const replayCase of FIXTURE.cases) {
    it(replayCase.id, async () => {
      const searchQueries: string[] = [];
      const hctx: HandlerFactoryContext = { searchQueries };

      const lm = new ScriptedReplayLM(replayCase.lm_script);
      settings.configure({ lm });

      const options: RLMOptions = buildRlmOptions(replayCase, hctx);
      const rlm = new RLM(replayCase.signature, options);
      const result = await rlm.aforward(
        replayCase.inputs as unknown as Record<string, unknown>,
      );

      const answerField = rlm.signature.outputFields.keys().next()
        .value as string;
      const answer = result.get(answerField);

      expect(typeof answer).toBe('string');
      expect(String(answer)).toContain(replayCase.expected.final_contains);
      if (replayCase.expected.final_contains_all !== undefined) {
        for (const needle of replayCase.expected.final_contains_all) {
          expect(String(answer)).toContain(needle);
        }
      }

      expect(lm.classifierCalls.length).toBe(
        replayCase.expected.classifier_calls,
      );
      expect(lm.oracleCalls.length).toBeGreaterThanOrEqual(
        replayCase.expected.oracle_calls_min,
      );
      expect(lm.oracleCalls.length).toBeLessThanOrEqual(
        replayCase.expected.oracle_calls_max,
      );

      const route = result.get('_rlm_route') as
        | { readonly kind: string; readonly taskTypes: readonly string[] }
        | undefined;
      expect(route).toBeDefined();
      expect(route?.kind).toBe(replayCase.expected.route_kind);
      expect([...(route?.taskTypes ?? [])]).toEqual(
        replayCase.expected.route_task_types,
      );

      const trace = (result.get('_rlm_trace') ??
        []) as readonly EvaluationTrace[];
      const traceTags = new Set<string>(trace.map((t) => t.nodeTag));
      for (const tag of replayCase.expected.trace_has) {
        expect(traceTags.has(tag)).toBe(true);
      }

      if (replayCase.expected.handler_search_queries !== undefined) {
        expect(hctx.searchQueries).toEqual(
          replayCase.expected.handler_search_queries,
        );
      }

      if (replayCase.expected.memory_transitions !== undefined) {
        // Effect trace entries use `extras.effectKind` to discriminate
        // handler dispatches. The memory write is observable as the
        // `i`th trace entry with `nodeTag === 'effect'` and
        // `extras.effectKind === 'WriteMemory'`.
        const writeMemoryEvents = trace.filter(
          (t) =>
            t.nodeTag === 'effect' &&
            (t.extras as { effectKind?: string } | undefined)?.effectKind ===
              'WriteMemory',
        );
        expect(writeMemoryEvents.length).toBe(
          replayCase.expected.memory_transitions.length,
        );
      }

      if (replayCase.expected.memory_banner_in_turn !== undefined) {
        const banner = replayCase.expected.memory_banner_in_turn;
        const oracleCalls = lm.oracleCalls;
        expect(oracleCalls.length).toBeGreaterThan(banner.turn);
        const systemOnTurn = oracleCalls[banner.turn]?.systemContent ?? '';
        expect(systemOnTurn).toContain(banner.contains);
      }
    });
  }
});

function buildRlmOptions(
  replayCase: ReplayCase,
  hctx: HandlerFactoryContext,
): RLMOptions {
  const options: { -readonly [K in keyof RLMOptions]: RLMOptions[K] } = {};
  const src = replayCase.options;
  if (src?.taskType !== undefined) {
    options.taskType = src.taskType;
  }
  if (replayCase.budget !== undefined) {
    options.budget = replayCase.budget;
  }
  if (src?.handlers !== undefined && src.handlers.length > 0) {
    options.handlers = src.handlers.map((name) =>
      buildHandlerByName(name, hctx),
    );
  }
  if (src?.plan_id !== undefined) {
    options.plans = [buildPlanById(src.plan_id)];
  }
  return options;
}
