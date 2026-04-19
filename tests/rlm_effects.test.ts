/**
 * Effects: parser/formatter, built-in handlers, evaluator oracle loop
 * (`EFFECT_ORACLE_SIGNATURE`, `'effect'` trace rows, budget). Hermetic —
 * scripted `QueueLM` only; `durationMs` from `performance.now()` is not
 * asserted for equality.
 */

import { describe, expect, it } from 'vitest';
import type { Message } from '../src/chat_message.js';
import { BudgetError, ValueError } from '../src/exceptions.js';
import { BaseLM, type LMOutput } from '../src/lm.js';
import { Prediction } from '../src/prediction.js';
import type { CombinatorValue } from '../src/rlm_combinators.js';
import { oracle, lit } from '../src/rlm_combinators.js';
import {
  EFFECT_KINDS,
  EFFECT_ORACLE_SIGNATURE,
  appendEffectResult,
  builtInEffectHandlers,
  isEffect,
  isEffectResult,
  parseOracleResponse,
  queryOracleHandler,
  readContextHandler,
  writeMemoryHandler,
  yieldHandler,
  type QueryOracleCallFn,
} from '../src/rlm_effects.js';
import {
  buildEvaluationContext,
  evaluate,
  type BuildEvaluationContextOptions,
} from '../src/rlm_evaluator.js';
import type { MemorySchema } from '../src/rlm_memory.js';
import type {
  Effect,
  EffectHandler,
  EffectResult,
  EvaluationContext,
} from '../src/rlm_types.js';
import { signatureFromString } from '../src/signature.js';

// ===========================================================================
// Helpers
// ===========================================================================

const DEFAULT_SIGNATURE = signatureFromString('prompt: str -> answer: str');

/**
 * Emit pre-scripted completions in first-in/first-out order. Each
 * `generate` call drains one payload; exhausting the queue throws with
 * the call index so test failures localize cleanly.
 *
 * Completions must be raw adapter payloads (JSON strings, since
 * `Predict` uses `JSONAdapter` at the oracle site). All calls, their
 * kwargs, and the last user message are recorded for assertion.
 */
interface RecordedCall {
  readonly messages: readonly Message[] | undefined;
  readonly kwargs: Record<string, unknown>;
}

class QueueLM extends BaseLM {
  readonly outputs: LMOutput[];
  readonly calls: RecordedCall[] = [];

  constructor(outputs: readonly LMOutput[]) {
    super({ model: 'queue-lm' });
    this.outputs = [...outputs];
  }

  protected override async agenerate(
    _prompt?: string,
    messages?: readonly Message[],
    kwargs?: Record<string, unknown>,
  ): Promise<readonly LMOutput[]> {
    await Promise.resolve();
    const next = this.outputs.shift();
    if (next === undefined) {
      throw new Error(`QueueLM exhausted at call #${this.calls.length + 1}`);
    }
    this.calls.push({ messages, kwargs: { ...(kwargs ?? {}) } });
    return [next];
  }
}

/** JSON payload for a terminal `{kind: 'value'}` response. */
function valuePayload(value: string): string {
  return JSON.stringify({
    kind: 'value',
    value,
    effect_name: null,
    effect_args: null,
  });
}

/** JSON payload for a `{kind: 'effect'}` request. */
function effectPayload(
  effectName: Effect['kind'],
  effectArgs: Readonly<Record<string, unknown>> = {},
): string {
  return JSON.stringify({
    kind: 'effect',
    value: null,
    effect_name: effectName,
    effect_args: effectArgs,
  });
}

function makeCtx(
  overrides: Partial<BuildEvaluationContextOptions> & {
    readonly lm: BaseLM;
  },
): EvaluationContext {
  return buildEvaluationContext({
    signature: DEFAULT_SIGNATURE,
    ...overrides,
  });
}

/**
 * Recover the text of the user message sent on a given `QueueLM`
 * call. The adapter serializes all oracle prompts as a plain-string
 * `user` message, so the last user message in each call is the
 * effect-loop prompt buffer.
 */
function pred(
  payload: Record<string, unknown>,
): Prediction<Record<string, unknown>> {
  return Prediction.create<Record<string, unknown>>(payload);
}

function lastUserText(call: RecordedCall): string {
  const messages = call.messages;
  if (messages === undefined) return '';
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i]!;
    if (msg.role !== 'user') continue;
    if (typeof msg.content === 'string') return msg.content;
    return msg.content
      .map((part) => (part.type === 'text' ? (part.text ?? '') : ''))
      .join('\n');
  }
  return '';
}

// ===========================================================================
// Pure: parseOracleResponse
// ===========================================================================

describe('parseOracleResponse', () => {
  it('parses a terminal value response', () => {
    expect(
      parseOracleResponse(
        pred({ kind: 'value', value: 'Paris', effect_name: null, effect_args: null }),
      ),
    ).toEqual({ kind: 'value', value: 'Paris' });
  });

  it('substitutes an empty string when value is missing', () => {
    expect(
      parseOracleResponse(
        pred({ kind: 'value', value: null, effect_name: null, effect_args: null }),
      ),
    ).toEqual({ kind: 'value', value: '' });
  });

  it('parses a ReadContext effect with start/end args', () => {
    expect(
      parseOracleResponse(
        pred({
          kind: 'effect',
          value: null,
          effect_name: 'ReadContext',
          effect_args: { name: 'input', start: 0, end: 40 },
        }),
      ),
    ).toEqual({
      kind: 'effect',
      effect: { kind: 'ReadContext', name: 'input', start: 0, end: 40 },
    });
  });

  it('parses a Yield effect with no args', () => {
    expect(
      parseOracleResponse(
        pred({ kind: 'effect', value: null, effect_name: 'Yield', effect_args: {} }),
      ),
    ).toEqual({ kind: 'effect', effect: { kind: 'Yield' } });
  });

  it('parses a Custom effect with nested args payload', () => {
    expect(
      parseOracleResponse(
        pred({
          kind: 'effect',
          value: null,
          effect_name: 'Custom',
          effect_args: { name: 'WeatherLookup', args: { city: 'Paris' } },
        }),
      ),
    ).toEqual({
      kind: 'effect',
      effect: {
        kind: 'Custom',
        name: 'WeatherLookup',
        args: { city: 'Paris' },
      },
    });
  });

  it('throws ValueError when kind is neither "value" nor "effect"', () => {
    expect(() =>
      parseOracleResponse(
        pred({ kind: 'garbage', value: null, effect_name: null, effect_args: null }),
      ),
    ).toThrow(ValueError);
  });

  it('throws ValueError when kind=effect but no effect_name is given', () => {
    expect(() =>
      parseOracleResponse(
        pred({ kind: 'effect', value: null, effect_name: '', effect_args: {} }),
      ),
    ).toThrow(ValueError);
  });

  it('throws ValueError when effect_name is unknown', () => {
    expect(() =>
      parseOracleResponse(
        pred({
          kind: 'effect',
          value: null,
          effect_name: 'NotAnEffect',
          effect_args: {},
        }),
      ),
    ).toThrow(ValueError);
  });

  it('throws when an effect is missing a required arg', () => {
    // ReadContext requires `name`
    expect(() =>
      parseOracleResponse(
        pred({
          kind: 'effect',
          value: null,
          effect_name: 'ReadContext',
          effect_args: { start: 0, end: 10 },
        }),
      ),
    ).toThrow(ValueError);
  });
});

// ===========================================================================
// Pure: appendEffectResult
// ===========================================================================

describe('appendEffectResult', () => {
  it('formats a successful result into a self-delimiting block', () => {
    const effect: Effect = { kind: 'ReadContext', name: 'input', start: 0, end: 5 };
    const result: EffectResult = { ok: true, value: 'hello' };
    const out = appendEffectResult('base prompt', effect, result, 0);
    expect(out).toContain('base prompt');
    expect(out).toContain('[[EFFECT turn=0 kind=ReadContext]]');
    expect(out).toContain('args: {"name":"input","start":0,"end":5}');
    expect(out).toContain('result: hello');
    expect(out).toContain('[[/EFFECT]]');
  });

  it('formats a failure as an error block so the LM can retry', () => {
    const effect: Effect = { kind: 'WriteMemory', key: 'bad', value: 1 };
    const result: EffectResult = { ok: false, error: 'no schema' };
    const out = appendEffectResult('', effect, result, 3);
    expect(out.startsWith('[[EFFECT turn=3 kind=WriteMemory]]')).toBe(true);
    expect(out).toContain('error: no schema');
  });

  it('stringifies non-string results as compact JSON', () => {
    const effect: Effect = {
      kind: 'Custom',
      name: 'Echo',
      args: { x: 1 },
    };
    const result: EffectResult = { ok: true, value: { seen: true, score: 0.5 } };
    const out = appendEffectResult('', effect, result, 1);
    expect(out).toContain('result: {"seen":true,"score":0.5}');
  });

  it('renders Yield args as `args: {}`', () => {
    const out = appendEffectResult(
      '',
      { kind: 'Yield' },
      { ok: true, value: null },
      7,
    );
    expect(out).toContain('args: {}');
    expect(out).toContain('result: ');
  });
});

// ===========================================================================
// Pure: EFFECT_KINDS, isEffect, isEffectResult
// ===========================================================================

describe('effect type guards', () => {
  it('EFFECT_KINDS is the canonical six-tag union order', () => {
    expect(EFFECT_KINDS).toEqual([
      'ReadContext',
      'WriteMemory',
      'QueryOracle',
      'Search',
      'Yield',
      'Custom',
    ]);
  });

  it('isEffect narrows real payloads', () => {
    expect(isEffect({ kind: 'Yield' })).toBe(true);
    expect(isEffect({ kind: 'Custom', name: 'X', args: {} })).toBe(true);
    expect(isEffect({ kind: 'NotReal' })).toBe(false);
    expect(isEffect(null)).toBe(false);
    expect(isEffect('effect')).toBe(false);
  });

  it('isEffect rejects malformed payloads', () => {
    // Missing required field `name`.
    expect(isEffect({ kind: 'ReadContext' })).toBe(false);
    // Wrong type for `key`.
    expect(isEffect({ kind: 'WriteMemory', key: 5, value: 'x' })).toBe(false);
  });

  it('isEffectResult narrows success and failure variants', () => {
    expect(isEffectResult({ ok: true, value: 1 })).toBe(true);
    expect(isEffectResult({ ok: false, error: 'nope' })).toBe(true);
    expect(isEffectResult({ ok: true })).toBe(false);
    expect(isEffectResult({ ok: false, error: 42 })).toBe(false);
  });
});

// ===========================================================================
// Built-in handler: readContextHandler
// ===========================================================================

describe('readContextHandler', () => {
  const handler = readContextHandler();

  it('returns the full scope binding when no range is given', async () => {
    const ctx = makeCtx({
      lm: new QueueLM([]),
      scope: new Map([['input', 'abcdef']]),
    });
    const effect: Effect = { kind: 'ReadContext', name: 'input' };
    await expect(handler.handle(effect, ctx)).resolves.toEqual({
      ok: true,
      value: 'abcdef',
    });
  });

  it('slices by [start, end) with standard clamping', async () => {
    const ctx = makeCtx({
      lm: new QueueLM([]),
      scope: new Map([['input', 'abcdef']]),
    });
    await expect(
      handler.handle(
        { kind: 'ReadContext', name: 'input', start: 1, end: 4 },
        ctx,
      ),
    ).resolves.toEqual({ ok: true, value: 'bcd' });
  });

  it('clamps out-of-range end and returns the trailing slice', async () => {
    const ctx = makeCtx({
      lm: new QueueLM([]),
      scope: new Map([['input', 'abc']]),
    });
    await expect(
      handler.handle(
        { kind: 'ReadContext', name: 'input', start: 1, end: 999 },
        ctx,
      ),
    ).resolves.toEqual({ ok: true, value: 'bc' });
  });

  it('returns empty string when the slice inverts', async () => {
    const ctx = makeCtx({
      lm: new QueueLM([]),
      scope: new Map([['input', 'abcdef']]),
    });
    await expect(
      handler.handle(
        { kind: 'ReadContext', name: 'input', start: 5, end: 2 },
        ctx,
      ),
    ).resolves.toEqual({ ok: true, value: '' });
  });

  it('returns a structured error when the binding is missing', async () => {
    const ctx = makeCtx({
      lm: new QueueLM([]),
      scope: new Map(),
    });
    await expect(
      handler.handle({ kind: 'ReadContext', name: 'nope' }, ctx),
    ).resolves.toMatchObject({ ok: false });
  });

  it('returns a structured error when the binding is not a string', async () => {
    const nonString: unknown = [1, 2, 3];
    // `scope` is typed against `CombinatorValue`, which excludes plain
    // objects like this sample array-of-numbers; the cast is a test
    // authoring convenience — production paths never bind such values.
    const ctx = makeCtx({
      lm: new QueueLM([]),
      scope: new Map([['input', nonString]]) as ReadonlyMap<
        string,
        CombinatorValue
      >,
    });
    await expect(
      handler.handle({ kind: 'ReadContext', name: 'input' }, ctx),
    ).resolves.toMatchObject({ ok: false });
  });
});

// ===========================================================================
// Built-in handler: writeMemoryHandler
// ===========================================================================

const NOTES_SCHEMA: MemorySchema = {
  name: 'notes',
  maxBytesSerialized: 2048,
  fields: [
    { name: 'title', type: 'string', description: 'current title', maxLength: 40 },
    { name: 'count', type: 'number', description: 'running counter' },
    { name: 'done', type: 'boolean', description: 'finished?' },
  ],
};

describe('writeMemoryHandler', () => {
  const handler = writeMemoryHandler();

  it('writes a valid field and mutates the memory cell', async () => {
    const ctx = makeCtx({ lm: new QueueLM([]), memorySchema: NOTES_SCHEMA });
    const result = await handler.handle(
      { kind: 'WriteMemory', key: 'title', value: 'hello' },
      ctx,
    );
    expect(result).toEqual({ ok: true, value: null });
    expect(ctx.memoryCell.current.get('title')).toBe('hello');
  });

  it('rejects writes when the plan declares no schema', async () => {
    const ctx = makeCtx({ lm: new QueueLM([]) });
    await expect(
      handler.handle({ kind: 'WriteMemory', key: 'title', value: 'x' }, ctx),
    ).resolves.toMatchObject({ ok: false });
    expect(ctx.memoryCell.current.size).toBe(0);
  });

  it('rejects unknown keys', async () => {
    const ctx = makeCtx({ lm: new QueueLM([]), memorySchema: NOTES_SCHEMA });
    await expect(
      handler.handle({ kind: 'WriteMemory', key: 'nope', value: 'x' }, ctx),
    ).resolves.toMatchObject({ ok: false });
  });

  it('rejects type mismatches per field schema', async () => {
    const ctx = makeCtx({ lm: new QueueLM([]), memorySchema: NOTES_SCHEMA });
    await expect(
      handler.handle({ kind: 'WriteMemory', key: 'count', value: 'nope' }, ctx),
    ).resolves.toMatchObject({ ok: false });
    await expect(
      handler.handle({ kind: 'WriteMemory', key: 'done', value: 1 }, ctx),
    ).resolves.toMatchObject({ ok: false });
    expect(ctx.memoryCell.current.size).toBe(0);
  });

  it('rejects strings longer than maxLength', async () => {
    const ctx = makeCtx({ lm: new QueueLM([]), memorySchema: NOTES_SCHEMA });
    await expect(
      handler.handle(
        { kind: 'WriteMemory', key: 'title', value: 'x'.repeat(41) },
        ctx,
      ),
    ).resolves.toMatchObject({ ok: false });
  });

  it('rejects writes that exceed the serialized byte budget', async () => {
    const schema: MemorySchema = {
      ...NOTES_SCHEMA,
      maxBytesSerialized: 10,
      fields: [
        {
          name: 'title',
          type: 'string',
          description: 'x',
          maxLength: 1000,
        },
      ],
    };
    const ctx = makeCtx({ lm: new QueueLM([]), memorySchema: schema });
    await expect(
      handler.handle(
        { kind: 'WriteMemory', key: 'title', value: 'this payload is too long' },
        ctx,
      ),
    ).resolves.toMatchObject({ ok: false });
    expect(ctx.memoryCell.current.size).toBe(0);
  });
});

// ===========================================================================
// Built-in handler: queryOracleHandler
// ===========================================================================

describe('queryOracleHandler', () => {
  it('delegates to the supplied callOracleFn and returns its answer', async () => {
    let seenPrompt = '';
    const fake: QueryOracleCallFn = async (prompt, modelHint, ctx) => {
      seenPrompt = prompt;
      expect(modelHint).toBeUndefined();
      expect(ctx).toBeDefined();
      return 'sub answer';
    };
    const handler = queryOracleHandler(fake);
    const ctx = makeCtx({ lm: new QueueLM([]) });
    await expect(
      handler.handle({ kind: 'QueryOracle', prompt: 'ask this' }, ctx),
    ).resolves.toEqual({ ok: true, value: 'sub answer' });
    expect(seenPrompt).toBe('ask this');
  });

  it('surfaces thrown handler errors as structured failures', async () => {
    const failing: QueryOracleCallFn = async () => {
      throw new Error('boom');
    };
    const handler = queryOracleHandler(failing);
    const ctx = makeCtx({ lm: new QueueLM([]) });
    await expect(
      handler.handle({ kind: 'QueryOracle', prompt: 'ask' }, ctx),
    ).resolves.toMatchObject({ ok: false });
  });

  it('propagates BudgetError instead of wrapping it', async () => {
    const budgetFail: QueryOracleCallFn = async () => {
      throw new BudgetError('out of calls');
    };
    const handler = queryOracleHandler(budgetFail);
    const ctx = makeCtx({ lm: new QueueLM([]) });
    await expect(
      handler.handle({ kind: 'QueryOracle', prompt: 'ask' }, ctx),
    ).rejects.toBeInstanceOf(BudgetError);
  });
});

// ===========================================================================
// Built-in handler: yieldHandler
// ===========================================================================

describe('yieldHandler', () => {
  it('always returns a success result with a null value', async () => {
    const ctx = makeCtx({ lm: new QueueLM([]) });
    await expect(
      yieldHandler().handle({ kind: 'Yield' }, ctx),
    ).resolves.toEqual({ ok: true, value: null });
  });
});

// ===========================================================================
// Registry assembly: builtInEffectHandlers
// ===========================================================================

describe('builtInEffectHandlers', () => {
  it('exposes all four default handlers in a stable order', () => {
    const fake: QueryOracleCallFn = async () => '';
    const names = builtInEffectHandlers(fake).map((h) => h.name);
    expect(names).toEqual([
      'ReadContext',
      'WriteMemory',
      'QueryOracle',
      'Yield',
    ]);
  });
});

// ===========================================================================
// End-to-end: effect loop integration
// ===========================================================================

describe('evaluate — oracle effect loop', () => {
  it('returns immediately on a kind=value response (one LM call, no effect trace)', async () => {
    const lm = new QueueLM([valuePayload('42')]);
    const ctx = makeCtx({ lm });
    await expect(
      evaluate(oracle(lit('ultimate question')), ctx),
    ).resolves.toBe('42');
    expect(lm.calls).toHaveLength(1);
    expect(ctx.trace.filter((t) => t.nodeTag === 'effect')).toHaveLength(0);
    expect(ctx.callsUsed.current).toBe(1);
  });

  it('dispatches ReadContext then terminates on the follow-up value', async () => {
    const lm = new QueueLM([
      effectPayload('ReadContext', { name: 'input', start: 0, end: 5 }),
      valuePayload('slice=abcde'),
    ]);
    const ctx = makeCtx({
      lm,
      scope: new Map([['input', 'abcdefghij']]),
    });
    await expect(
      evaluate(oracle(lit('fetch my slice please')), ctx),
    ).resolves.toBe('slice=abcde');

    expect(lm.calls).toHaveLength(2);
    const effectTrace = ctx.trace.filter((t) => t.nodeTag === 'effect');
    expect(effectTrace).toHaveLength(1);
    expect(effectTrace[0]!.ok).toBe(true);
    expect(effectTrace[0]!.extras).toMatchObject({
      turn: 0,
      effectKind: 'ReadContext',
      handlerName: 'ReadContext',
    });

    // Follow-up prompt contains the effect block with the result.
    const turn2Prompt = lastUserText(lm.calls[1]!);
    expect(turn2Prompt).toContain('[[EFFECT turn=0 kind=ReadContext]]');
    expect(turn2Prompt).toContain('result: abcde');
  });

  it('surfaces a structured error when no handler is registered for a Custom effect', async () => {
    const lm = new QueueLM([
      effectPayload('Custom', { name: 'Unregistered', args: { x: 1 } }),
      valuePayload('recovered'),
    ]);
    const ctx = makeCtx({ lm });
    await expect(
      evaluate(oracle(lit('trigger unknown tool')), ctx),
    ).resolves.toBe('recovered');

    const effectTrace = ctx.trace.filter((t) => t.nodeTag === 'effect');
    expect(effectTrace).toHaveLength(1);
    expect(effectTrace[0]!.ok).toBe(false);
    expect(effectTrace[0]!.extras).toMatchObject({
      turn: 0,
      effectKind: 'Custom',
    });

    const turn2Prompt = lastUserText(lm.calls[1]!);
    expect(turn2Prompt).toContain('[[EFFECT turn=0 kind=Custom]]');
    expect(turn2Prompt).toContain('error: No handler registered');
    expect(turn2Prompt).toContain('name="Unregistered"');
  });

  it('dispatches a user-supplied Custom handler by effect name', async () => {
    const seen: Effect[] = [];
    const lookup: EffectHandler = {
      name: 'WeatherLookup',
      async handle(effect) {
        seen.push(effect);
        return { ok: true, value: 'sunny, 22C' };
      },
    };
    const lm = new QueueLM([
      effectPayload('Custom', {
        name: 'WeatherLookup',
        args: { city: 'Paris' },
      }),
      valuePayload('final=sunny, 22C'),
    ]);
    const ctx = makeCtx({
      lm,
      handlers: new Map([['WeatherLookup', lookup]]),
    });
    await expect(
      evaluate(oracle(lit('ask the weather')), ctx),
    ).resolves.toBe('final=sunny, 22C');

    expect(seen).toHaveLength(1);
    expect(seen[0]).toEqual({
      kind: 'Custom',
      name: 'WeatherLookup',
      args: { city: 'Paris' },
    });

    const effectTrace = ctx.trace.filter((t) => t.nodeTag === 'effect');
    expect(effectTrace).toHaveLength(1);
    expect(effectTrace[0]!.ok).toBe(true);
    expect(effectTrace[0]!.extras).toMatchObject({
      effectKind: 'Custom',
      handlerName: 'WeatherLookup',
    });
  });

  it('user-supplied handler named "Yield" overrides the built-in', async () => {
    const spy: { called: boolean } = { called: false };
    const override: EffectHandler = {
      name: 'Yield',
      async handle() {
        spy.called = true;
        return { ok: true, value: 'user-yield' };
      },
    };
    const lm = new QueueLM([
      effectPayload('Yield'),
      valuePayload('after-yield'),
    ]);
    const ctx = makeCtx({
      lm,
      handlers: new Map([['Yield', override]]),
    });
    await expect(
      evaluate(oracle(lit('pause and resume')), ctx),
    ).resolves.toBe('after-yield');
    expect(spy.called).toBe(true);
    const turn2 = lastUserText(lm.calls[1]!);
    expect(turn2).toContain('result: user-yield');
  });

  it('recovers from a parse error by re-prompting with a structured error block', async () => {
    const lm = new QueueLM([
      // First turn: malformed response — effect without effect_name.
      JSON.stringify({
        kind: 'effect',
        value: null,
        effect_name: '',
        effect_args: {},
      }),
      valuePayload('recovered after parse error'),
    ]);
    const ctx = makeCtx({ lm });
    await expect(
      evaluate(oracle(lit('ask with bad effect')), ctx),
    ).resolves.toBe('recovered after parse error');

    const effectTrace = ctx.trace.filter((t) => t.nodeTag === 'effect');
    expect(effectTrace).toHaveLength(1);
    expect(effectTrace[0]!.ok).toBe(false);
    expect(effectTrace[0]!.extras).toMatchObject({
      turn: 0,
      effectKind: 'parse_error',
    });
    const turn2 = lastUserText(lm.calls[1]!);
    expect(turn2).toContain('[[EFFECT turn=0 kind=parse_error]]');
    expect(turn2).toContain('error:');
  });

  it('throws BudgetError when the effect loop exhausts maxEffectTurns, preserving the partial trace', async () => {
    const lm = new QueueLM([
      effectPayload('Yield'),
      effectPayload('Yield'),
    ]);
    const ctx = makeCtx({
      lm,
      budget: { maxEffectTurns: 2 },
    });
    await expect(evaluate(oracle(lit('loop forever')), ctx)).rejects.toBeInstanceOf(
      BudgetError,
    );
    const effectTrace = ctx.trace.filter((t) => t.nodeTag === 'effect');
    expect(effectTrace).toHaveLength(2);
    expect(effectTrace.every((t) => t.ok)).toBe(true);
    expect(effectTrace.map((t) => (t.extras as { turn: number } | undefined)?.turn)).toEqual([0, 1]);
  });

  it('drives a two-step effect chain with a WriteMemory then a value', async () => {
    const lm = new QueueLM([
      effectPayload('WriteMemory', { key: 'title', value: 'draft' }),
      valuePayload('ok'),
    ]);
    const ctx = makeCtx({
      lm,
      memorySchema: NOTES_SCHEMA,
    });
    await expect(
      evaluate(oracle(lit('save the title')), ctx),
    ).resolves.toBe('ok');
    expect(ctx.memoryCell.current.get('title')).toBe('draft');

    const turn2 = lastUserText(lm.calls[1]!);
    expect(turn2).toContain('[[EFFECT turn=0 kind=WriteMemory]]');
    // Successful WriteMemory returns `null`; renderer coerces to empty string.
    expect(turn2).toContain('result: ');
  });

  it('routes QueryOracle through the plain oracle signature (not a nested effect loop)', async () => {
    // Turn 0: outer oracle requests a QueryOracle.
    // Turn 1: sub-oracle responds in the plain signature shape ({answer}).
    // Turn 2: outer oracle resumes with a terminal value.
    const lm = new QueueLM([
      effectPayload('QueryOracle', { prompt: 'sub-question' }),
      JSON.stringify({ answer: 'sub result' }),
      valuePayload('final'),
    ]);
    const ctx = makeCtx({ lm });
    await expect(evaluate(oracle(lit('ask anything')), ctx)).resolves.toBe('final');
    expect(lm.calls).toHaveLength(3);

    // The sub-oracle's system message must describe the plain oracle
    // signature (a single `answer` output field), not the effect one.
    // If it used EFFECT_ORACLE_SIGNATURE, the sub-oracle would see
    // `- `kind` (...)` in its system message and a `{answer}` payload
    // would fail to parse.
    const turn2Prompt = lastUserText(lm.calls[2]!);
    expect(turn2Prompt).toContain('[[EFFECT turn=0 kind=QueryOracle]]');
    expect(turn2Prompt).toContain('result: sub result');
  });

  it('tags every effect-loop trace entry with nodeTag="effect"', async () => {
    const lm = new QueueLM([
      effectPayload('Yield'),
      effectPayload('Yield'),
      valuePayload('done'),
    ]);
    const ctx = makeCtx({ lm, budget: { maxEffectTurns: 5 } });
    await evaluate(oracle(lit('tag check')), ctx);
    const effectTrace = ctx.trace.filter((t) => t.nodeTag === 'effect');
    expect(effectTrace).toHaveLength(2);
    // The `oracle` node entry itself should also be present and non-effect.
    const oracleEntries = ctx.trace.filter((t) => t.nodeTag === 'oracle');
    expect(oracleEntries).toHaveLength(1);
    expect(oracleEntries[0]!.ok).toBe(true);
  });
});

// ===========================================================================
// Sanity: EFFECT_ORACLE_SIGNATURE shape
// ===========================================================================

describe('EFFECT_ORACLE_SIGNATURE', () => {
  it('exposes the four output fields the parser expects', () => {
    const outputs = [...EFFECT_ORACLE_SIGNATURE.outputFields.keys()];
    expect(outputs).toEqual(['kind', 'value', 'effect_name', 'effect_args']);
  });

  it('accepts `prompt` as the only input field', () => {
    const inputs = [...EFFECT_ORACLE_SIGNATURE.inputFields.keys()];
    expect(inputs).toEqual(['prompt']);
  });
});
