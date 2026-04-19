#!/usr/bin/env -S npx tsx
/**
 * Replay harness for `spec/fixtures/rlm_v2_replay.json`: run each case through
 * `RLM` with scripted LM output, compare observed vs `expected`.
 *
 *   npx tsx tools/record_rlm_v2.ts --verify   # exit non-zero on drift
 *   npx tsx tools/record_rlm_v2.ts --dump     # print observed JSON
 *
 * Does not rewrite the fixture; pair with `tests/rlm_replay.test.ts`.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { argv, exit } from 'node:process';

import type { Message } from '../src/chat_message.js';
import { BaseLM, type LMOutput } from '../src/lm.js';
import { RLM, type RLMOptions } from '../src/rlm.js';
import { settings } from '../src/settings.js';
import { oracle, vref } from '../src/rlm_combinators.js';
import type {
  EffectHandler,
  EvaluationTrace,
  RLMBudget,
} from '../src/rlm_types.js';
import type { MemorySchema } from '../src/rlm_memory.js';
import type { StaticPlan, TaskType } from '../src/rlm_task_router.js';

// ---------------------------------------------------------------------------
// Fixture shapes — duplicated verbatim with tests/rlm_replay.test.ts to
// keep the recorder standalone and free of test-framework imports.
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
// ScriptedRecorderLM
// ---------------------------------------------------------------------------

interface CapturedCall {
  readonly role: 'classifier' | 'oracle';
  readonly userContent: string;
  readonly systemContent: string;
  readonly fullContent: string;
}

class ScriptedRecorderLM extends BaseLM {
  readonly calls: CapturedCall[] = [];
  readonly classifier: ClassifierPayload | null;
  readonly oracleQueue: OracleQueueEntry[];

  constructor(script: LMScript) {
    super({ model: 'scripted-recorder' });
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
          'Recorder classifier invoked but case has no classifier payload',
        );
      }
      return [JSON.stringify(this.classifier)];
    }
    const next = this.oracleQueue.shift();
    if (next === undefined) {
      throw new Error(
        `Recorder oracle queue exhausted at call #${this.calls.length}`,
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
// Handler and plan registries (shared with tests/rlm_replay.test.ts)
// ---------------------------------------------------------------------------

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
        template: oracle(vref('input'), undefined),
        memorySchema: MEMORY_BEARING_PAIRWISE_SCHEMA,
      });
    default:
      throw new Error(`Unknown plan_id: ${planId}`);
  }
}

// ---------------------------------------------------------------------------
// Recorder core
// ---------------------------------------------------------------------------

interface ObservedCaseReport {
  readonly id: string;
  readonly final: string;
  readonly classifier_calls: number;
  readonly oracle_calls: number;
  readonly route_kind: string;
  readonly route_task_types: readonly string[];
  readonly trace_tags: readonly string[];
  readonly handler_search_queries?: readonly string[];
  readonly memory_banner_turn?: {
    readonly turn: number;
    readonly system_excerpt: string;
  };
}

async function recordCase(
  replayCase: ReplayCase,
): Promise<ObservedCaseReport> {
  const searchQueries: string[] = [];
  const hctx: HandlerFactoryContext = { searchQueries };

  const lm = new ScriptedRecorderLM(replayCase.lm_script);
  settings.configure({ lm });
  try {
    const options: RLMOptions = buildRlmOptions(replayCase, hctx);
    const rlm = new RLM(replayCase.signature, options);
    const result = await rlm.aforward(
      replayCase.inputs as unknown as Record<string, unknown>,
    );
    const answerField = rlm.signature.outputFields.keys().next()
      .value as string;
    const answer = String(result.get(answerField) ?? '');
    const route = result.get('_rlm_route') as
      | { readonly kind: string; readonly taskTypes: readonly string[] }
      | undefined;
    const trace = (result.get('_rlm_trace') ??
      []) as readonly EvaluationTrace[];
    const traceTags = Array.from(new Set(trace.map((t) => t.nodeTag))).sort();

    const report: ObservedCaseReport = {
      id: replayCase.id,
      final: answer,
      classifier_calls: lm.classifierCalls.length,
      oracle_calls: lm.oracleCalls.length,
      route_kind: route?.kind ?? 'single',
      route_task_types: [...(route?.taskTypes ?? [])],
      trace_tags: traceTags,
      ...(searchQueries.length > 0
        ? { handler_search_queries: [...searchQueries] }
        : {}),
      ...(replayCase.expected.memory_banner_in_turn !== undefined
        ? buildMemoryBannerReport(replayCase, lm)
        : {}),
    };
    return report;
  } finally {
    settings.reset();
  }
}

function buildMemoryBannerReport(
  replayCase: ReplayCase,
  lm: ScriptedRecorderLM,
): Pick<ObservedCaseReport, 'memory_banner_turn'> {
  const banner = replayCase.expected.memory_banner_in_turn;
  if (banner === undefined) return {};
  const oracleCalls = lm.oracleCalls;
  const systemContent = oracleCalls[banner.turn]?.systemContent ?? '';
  const idx = systemContent.indexOf(banner.contains);
  const excerpt =
    idx < 0
      ? '<banner substring not found>'
      : systemContent.slice(Math.max(0, idx - 40), idx + banner.contains.length + 40);
  return {
    memory_banner_turn: {
      turn: banner.turn,
      system_excerpt: excerpt,
    },
  };
}

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

// ---------------------------------------------------------------------------
// Verification mode
// ---------------------------------------------------------------------------

interface VerificationResult {
  readonly ok: boolean;
  readonly problems: readonly string[];
}

function verifyCase(
  replayCase: ReplayCase,
  observed: ObservedCaseReport,
): VerificationResult {
  const problems: string[] = [];
  const exp = replayCase.expected;

  if (!observed.final.includes(exp.final_contains)) {
    problems.push(
      `final does not contain expected substring "${exp.final_contains}"`,
    );
  }
  if (exp.final_contains_all !== undefined) {
    for (const needle of exp.final_contains_all) {
      if (!observed.final.includes(needle)) {
        problems.push(`final missing expected substring "${needle}"`);
      }
    }
  }
  if (observed.classifier_calls !== exp.classifier_calls) {
    problems.push(
      `classifier_calls=${observed.classifier_calls}, expected=${exp.classifier_calls}`,
    );
  }
  if (
    observed.oracle_calls < exp.oracle_calls_min ||
    observed.oracle_calls > exp.oracle_calls_max
  ) {
    problems.push(
      `oracle_calls=${observed.oracle_calls} out of [${exp.oracle_calls_min}, ${exp.oracle_calls_max}]`,
    );
  }
  if (observed.route_kind !== exp.route_kind) {
    problems.push(
      `route_kind=${observed.route_kind}, expected=${exp.route_kind}`,
    );
  }
  if (
    JSON.stringify(observed.route_task_types) !==
    JSON.stringify(exp.route_task_types)
  ) {
    problems.push(
      `route_task_types=${JSON.stringify(observed.route_task_types)}, expected=${JSON.stringify(exp.route_task_types)}`,
    );
  }
  const traceSet = new Set(observed.trace_tags);
  for (const tag of exp.trace_has) {
    if (!traceSet.has(tag)) {
      problems.push(
        `trace tag "${tag}" missing (observed tags: ${observed.trace_tags.join(', ')})`,
      );
    }
  }
  if (exp.handler_search_queries !== undefined) {
    const obs = observed.handler_search_queries ?? [];
    if (JSON.stringify(obs) !== JSON.stringify(exp.handler_search_queries)) {
      problems.push(
        `handler_search_queries=${JSON.stringify(obs)}, expected=${JSON.stringify(exp.handler_search_queries)}`,
      );
    }
  }
  return { ok: problems.length === 0, problems };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const MODULE_DIR = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = resolve(
  MODULE_DIR,
  '..',
  '..',
  'spec',
  'fixtures',
  'rlm_v2_replay.json',
);

async function main(): Promise<void> {
  const verify = argv.includes('--verify');
  const dump = argv.includes('--dump');

  const fixture = JSON.parse(
    readFileSync(FIXTURE_PATH, 'utf-8'),
  ) as FixtureFile;

  const observations: ObservedCaseReport[] = [];
  const failures: string[] = [];

  for (const replayCase of fixture.cases) {
    const observed = await recordCase(replayCase);
    observations.push(observed);
    if (verify) {
      const verification = verifyCase(replayCase, observed);
      if (!verification.ok) {
        failures.push(
          `[${replayCase.id}] ${verification.problems.join('; ')}`,
        );
      }
    }
  }

  if (dump) {
    console.log(JSON.stringify(observations, null, 2));
  } else {
    console.log(
      `[record_rlm_v2] recorded ${observations.length} case(s) from ${FIXTURE_PATH}`,
    );
    for (const obs of observations) {
      console.log(
        `  ${obs.id}: oracle_calls=${obs.oracle_calls} classifier_calls=${obs.classifier_calls} route=${obs.route_kind}/${obs.route_task_types.join(',')} tags=${obs.trace_tags.join(',')}`,
      );
    }
  }

  if (verify) {
    if (failures.length > 0) {
      console.error('[record_rlm_v2] VERIFY FAIL:');
      for (const f of failures) {
        console.error('  ' + f);
      }
      exit(1);
    }
    console.log(
      `[record_rlm_v2] VERIFY OK — all ${observations.length} case(s) match the stored expectations.`,
    );
  }
}

main().catch((err) => {
  console.error('[record_rlm_v2] FAILED:', err);
  exit(1);
});
