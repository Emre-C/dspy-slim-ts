/**
 * Typed memory: `initialMemoryState`, `applyMemoryWrite`,
 * `defaultMemoryInjector`, and oracle reinjection after `WriteMemory`.
 *
 * See `docs/product/rlm-v2-architecture.md` §2.5.
 *
 * 1. Schema validation — wrong type, unknown field, over-long string,
 *    byte budget.
 * 2. `initialMemoryState` seeding behavior.
 * 3. `defaultMemoryInjector` deterministic rendering and cap
 *    enforcement.
 * 4. End-to-end memory reinjection: a WriteMemory on oracle turn 1
 *    shows up in the system message of oracle turn 2.
 *
 * The end-to-end test uses a small recording LM that captures every
 * system message it observes, so the assertion is "the injected banner
 * appeared on turn N" rather than coupling to the effect-loop
 * implementation details.
 */

import { describe, expect, it } from 'vitest';

import type { Message } from '../src/adapter.js';
import { BaseLM, type LMOutput } from '../src/lm.js';
import { ValueError } from '../src/exceptions.js';
import { signatureFromString } from '../src/signature.js';

import {
  applyMemoryWrite,
  DEFAULT_MAX_MEMORY_BYTES,
  defaultMemoryInjector,
  initialMemoryState,
  type MemoryFieldSchema,
  type MemoryScalar,
  type MemorySchema,
} from '../src/rlm_memory.js';
import type { CombinatorValue } from '../src/rlm_combinators.js';
import { oracle, vref } from '../src/rlm_combinators.js';
import { buildEvaluationContext, evaluate } from '../src/rlm_evaluator.js';

// ===========================================================================
// Fixtures
// ===========================================================================

const TITLE_FIELD: MemoryFieldSchema = {
  name: 'title',
  type: 'string',
  description: 'short title',
  maxLength: 40,
};

const COUNT_FIELD: MemoryFieldSchema = {
  name: 'count',
  type: 'number',
  description: 'integer-like count',
};

const DONE_FIELD: MemoryFieldSchema = {
  name: 'done',
  type: 'boolean',
  description: 'done flag',
};

const NOTES_SCHEMA: MemorySchema = {
  name: 'notes',
  fields: [TITLE_FIELD, COUNT_FIELD, DONE_FIELD],
  maxBytesSerialized: 256,
};

const SEEDED_SCHEMA: MemorySchema = {
  name: 'seeded',
  fields: [
    { name: 'title', type: 'string', description: 'seed', initial: 'draft' },
    { name: 'count', type: 'number', description: 'seed', initial: 0 },
    { name: 'done', type: 'boolean', description: 'seed' },
  ],
  maxBytesSerialized: 256,
};

// ===========================================================================
// initialMemoryState
// ===========================================================================

describe('initialMemoryState', () => {
  it('returns an empty map when no field declares initial', () => {
    const state = initialMemoryState(NOTES_SCHEMA);
    expect(state.size).toBe(0);
  });

  it('returns a runtime-immutable snapshot', () => {
    const state = initialMemoryState(NOTES_SCHEMA) as unknown as Map<
      string,
      unknown
    >;
    expect(() => state.set('title', 'mutated')).toThrow(TypeError);
    expect(state.size).toBe(0);
  });

  it('seeds each field whose `initial` is defined', () => {
    const state = initialMemoryState(SEEDED_SCHEMA);
    expect(state.get('title')).toBe('draft');
    expect(state.get('count')).toBe(0);
    expect(state.has('done')).toBe(false);
  });

  it('rejects initial values that violate the declared type', () => {
    const schema: MemorySchema = {
      name: 'bad',
      fields: [{ name: 'n', type: 'number', description: 'x', initial: 'oops' }],
      maxBytesSerialized: 64,
    };
    expect(() => initialMemoryState(schema)).toThrow(ValueError);
  });

  it('rejects initial string values that exceed maxLength', () => {
    const schema: MemorySchema = {
      name: 'bad',
      fields: [
        {
          name: 'title',
          type: 'string',
          description: 'x',
          maxLength: 3,
          initial: 'long enough to trip the cap',
        },
      ],
      maxBytesSerialized: 64,
    };
    expect(() => initialMemoryState(schema)).toThrow(ValueError);
  });

  it('rejects seed state that exceeds maxBytesSerialized', () => {
    const schema: MemorySchema = {
      name: 'tiny',
      fields: [
        { name: 'title', type: 'string', description: 'x', initial: 'hello world' },
      ],
      maxBytesSerialized: 5,
    };
    expect(() => initialMemoryState(schema)).toThrow(ValueError);
  });
});

// ===========================================================================
// applyMemoryWrite
// ===========================================================================

describe('applyMemoryWrite', () => {
  it('returns a new map without mutating the input', () => {
    const before = initialMemoryState(NOTES_SCHEMA);
    const after = applyMemoryWrite(before, NOTES_SCHEMA, {
      key: 'title',
      value: 'hello',
    });
    expect(before.size).toBe(0);
    expect(after.get('title')).toBe('hello');
    expect(after).not.toBe(before);
  });

  it('returns a runtime-immutable updated snapshot', () => {
    const state = applyMemoryWrite(initialMemoryState(NOTES_SCHEMA), NOTES_SCHEMA, {
      key: 'title',
      value: 'hello',
    }) as unknown as Map<string, unknown>;
    expect(() => state.delete('title')).toThrow(TypeError);
    expect(state.get('title')).toBe('hello');
  });

  it('rejects unknown keys with a descriptive error', () => {
    const state = initialMemoryState(NOTES_SCHEMA);
    expect(() =>
      applyMemoryWrite(state, NOTES_SCHEMA, { key: 'nope', value: 'x' }),
    ).toThrow(/not declared in schema/);
  });

  it('rejects type mismatches per field', () => {
    const state = initialMemoryState(NOTES_SCHEMA);
    expect(() =>
      applyMemoryWrite(state, NOTES_SCHEMA, { key: 'count', value: 'x' }),
    ).toThrow(/expects finite number/);
    expect(() =>
      applyMemoryWrite(state, NOTES_SCHEMA, { key: 'done', value: 1 }),
    ).toThrow(/expects boolean/);
    expect(() =>
      applyMemoryWrite(state, NOTES_SCHEMA, { key: 'title', value: 5 }),
    ).toThrow(/expects string/);
  });

  it('rejects NaN and Infinity for number fields', () => {
    const state = initialMemoryState(NOTES_SCHEMA);
    expect(() =>
      applyMemoryWrite(state, NOTES_SCHEMA, { key: 'count', value: Number.NaN }),
    ).toThrow(/finite number/);
    expect(() =>
      applyMemoryWrite(state, NOTES_SCHEMA, {
        key: 'count',
        value: Number.POSITIVE_INFINITY,
      }),
    ).toThrow(/finite number/);
  });

  it('rejects string writes that exceed maxLength', () => {
    const state = initialMemoryState(NOTES_SCHEMA);
    expect(() =>
      applyMemoryWrite(state, NOTES_SCHEMA, {
        key: 'title',
        value: 'x'.repeat(41),
      }),
    ).toThrow(/exceeds maxLength/);
  });

  it('rejects writes that exceed the serialized byte budget', () => {
    const baseSchema: MemorySchema = {
      name: 'tiny',
      fields: [{ name: 'title', type: 'string', description: 'x', maxLength: 1000 }],
      maxBytesSerialized: 10_000,
    };
    const schema: MemorySchema = {
      ...baseSchema,
      maxBytesSerialized:
        Buffer.byteLength(defaultMemoryInjector.render(baseSchema, new Map()), 'utf8') +
        4,
    };
    const state = initialMemoryState(schema);
    expect(() =>
      applyMemoryWrite(state, schema, {
        key: 'title',
        value: 'this payload is way too long',
      }),
    ).toThrow(/maxBytesSerialized/);
  });

  it('rejects writes that overflow the rendered banner budget even when the raw payload would fit', () => {
    const baseSchema: MemorySchema = {
      name: 'tight',
      fields: [{ name: 'title', type: 'string', description: 'x', maxLength: 1000 }],
      maxBytesSerialized: 10_000,
    };
    const emptyBannerBytes = Buffer.byteLength(
      defaultMemoryInjector.render(baseSchema, new Map()),
      'utf8',
    );
    const schema: MemorySchema = {
      ...baseSchema,
      maxBytesSerialized: emptyBannerBytes,
    };
    const state = initialMemoryState(schema);
    expect(() =>
      applyMemoryWrite(state, schema, {
        key: 'title',
        value: 'ok',
      }),
    ).toThrow(/maxBytesSerialized/);
  });

  it('overwrites an existing value in-place semantically', () => {
    const state1 = applyMemoryWrite(initialMemoryState(NOTES_SCHEMA), NOTES_SCHEMA, {
      key: 'count',
      value: 1,
    });
    const state2 = applyMemoryWrite(state1, NOTES_SCHEMA, {
      key: 'count',
      value: 2,
    });
    expect(state2.get('count')).toBe(2);
    expect(state2.size).toBe(1);
  });
});

// ===========================================================================
// defaultMemoryInjector
// ===========================================================================

describe('defaultMemoryInjector', () => {
  it('renders the schema header/footer even when state is empty', () => {
    const banner = defaultMemoryInjector.render(NOTES_SCHEMA, new Map());
    expect(banner).toBe('[[RLM_MEMORY schema=notes]]\n[[/RLM_MEMORY]]');
  });

  it('renders fields in schema declaration order', () => {
    const state = new Map<string, MemoryScalar>([
      ['done', true],
      ['title', 'Greetings'],
      ['count', 3],
    ]);
    const banner = defaultMemoryInjector.render(NOTES_SCHEMA, state);
    const expected = [
      '[[RLM_MEMORY schema=notes]]',
      'title: string = "Greetings"',
      'count: number = 3',
      'done: boolean = true',
      '[[/RLM_MEMORY]]',
    ].join('\n');
    expect(banner).toBe(expected);
  });

  it('omits absent fields (undefined keys)', () => {
    const state = new Map<string, MemoryScalar>([['count', 7]]);
    const banner = defaultMemoryInjector.render(NOTES_SCHEMA, state);
    expect(banner).toContain('count: number = 7');
    expect(banner).not.toContain('title');
    expect(banner).not.toContain('done');
  });

  it('escapes quotes, newlines, and backslashes in string values', () => {
    const state = new Map<string, MemoryScalar>([
      ['title', 'quoted "x" and\nnewline and \\back'],
    ]);
    const banner = defaultMemoryInjector.render(NOTES_SCHEMA, state);
    expect(banner).toContain(
      'title: string = "quoted \\"x\\" and\\nnewline and \\\\back"',
    );
  });

  it('produces identical output across repeated renders (deterministic)', () => {
    const state = new Map<string, MemoryScalar>([
      ['title', 'a'],
      ['count', 2],
    ]);
    const first = defaultMemoryInjector.render(NOTES_SCHEMA, state);
    const second = defaultMemoryInjector.render(NOTES_SCHEMA, state);
    expect(first).toBe(second);
  });

  it('truncates the banner with a marker when it exceeds maxBytesSerialized', () => {
    const schema: MemorySchema = {
      name: 's',
      fields: [
        { name: 'a', type: 'string', description: 'x' },
        { name: 'b', type: 'string', description: 'x' },
        { name: 'c', type: 'string', description: 'x' },
      ],
      maxBytesSerialized: 80,
    };
    const state = new Map<string, MemoryScalar>([
      ['a', 'hello world with more words'],
      ['b', 'also a long value with plenty of chars'],
      ['c', 'third field value here'],
    ]);
    const banner = defaultMemoryInjector.render(schema, state);
    expect(Buffer.byteLength(banner, 'utf8')).toBeLessThanOrEqual(80);
    expect(banner).toContain('[[/RLM_MEMORY truncated]]');
    expect(banner).toContain('[[RLM_MEMORY schema=s]]');
    expect(banner).not.toContain('[[/RLM_MEMORY]]');
  });

  it('enforces maxBytesSerialized against UTF-8 bytes, not UTF-16 code units', () => {
    const generousSchema: MemorySchema = {
      name: 'wide',
      fields: [{ name: 'title', type: 'string', description: 'x' }],
      maxBytesSerialized: 10_000,
    };
    const state = new Map<string, MemoryScalar>([['title', '🙂🙂🙂🙂🙂']]);
    const fullBanner = defaultMemoryInjector.render(generousSchema, state);
    const fullChars = fullBanner.length;
    const fullBytes = Buffer.byteLength(fullBanner, 'utf8');

    expect(fullBytes).toBeGreaterThan(fullChars);

    const cappedBanner = defaultMemoryInjector.render(
      {
        ...generousSchema,
        maxBytesSerialized: fullChars,
      },
      state,
    );

    expect(Buffer.byteLength(cappedBanner, 'utf8')).toBeLessThanOrEqual(fullChars);
    expect(Buffer.byteLength(cappedBanner, 'utf8')).toBeLessThan(fullBytes);
  });
});

// ===========================================================================
// DEFAULT_MAX_MEMORY_BYTES
// ===========================================================================

describe('DEFAULT_MAX_MEMORY_BYTES', () => {
  it('is the documented 2048 bytes', () => {
    expect(DEFAULT_MAX_MEMORY_BYTES).toBe(2048);
  });
});

// ===========================================================================
// Evaluator wiring: memory banner visible to subsequent oracle turns
// ===========================================================================

interface Capture {
  readonly systemMessage: string;
  readonly userMessage: string;
}

/**
 * Minimal scripted LM that records every system message it sees and
 * emits scripted output per-call. Used to assert that the memory
 * banner lands in the system message text after a `WriteMemory`
 * effect runs.
 */
class RecordingLM extends BaseLM {
  readonly captured: Capture[] = [];
  private readonly script: readonly string[];

  constructor(script: readonly string[]) {
    super({ model: 'recording-lm' });
    this.script = script;
  }

  protected override async agenerate(
    _prompt?: string,
    messages?: readonly Message[],
    _kwargs?: Record<string, unknown>,
  ): Promise<readonly LMOutput[]> {
    await Promise.resolve();
    const msgs = messages ?? [];
    const system = extractMessage(msgs, 'system');
    const user = extractMessage(msgs, 'user');
    this.captured.push({ systemMessage: system, userMessage: user });
    const index = this.captured.length - 1;
    const completion =
      this.script[index] ?? this.script[this.script.length - 1] ?? '';
    return [completion];
  }
}

function extractMessage(
  messages: readonly Message[],
  role: 'system' | 'user',
): string {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i]!;
    if (msg.role !== role) continue;
    if (typeof msg.content === 'string') return msg.content;
    return msg.content
      .map((part) => (part.type === 'text' ? (part.text ?? '') : ''))
      .join('\n');
  }
  return '';
}

const PROMPT_SIGNATURE = signatureFromString('prompt: str -> answer: str');

describe('evaluator memory reinjection (end-to-end)', () => {
  it('is invisible to oracle calls when no memory schema is attached', async () => {
    const lm = new RecordingLM([
      JSON.stringify({
        kind: 'value',
        value: 'done',
        effect_name: null,
        effect_args: null,
      }),
    ]);
    const ctx = buildEvaluationContext({
      lm,
      signature: PROMPT_SIGNATURE,
      scope: new Map<string, CombinatorValue>([['input', 'hi']]),
    });
    const result = await evaluate(oracle(vref('input'), undefined), ctx);
    expect(result).toBe('done');
    expect(lm.captured).toHaveLength(1);
    expect(lm.captured[0]?.systemMessage).not.toContain('[[RLM_MEMORY');
  });

  it('renders the banner with seed state on turn 1 and the updated state on turn 2', async () => {
    const lm = new RecordingLM([
      // Turn 1: emit a WriteMemory effect.
      JSON.stringify({
        kind: 'effect',
        value: null,
        effect_name: 'WriteMemory',
        effect_args: { key: 'title', value: 'ready' },
      }),
      // Turn 2: return a value (effect result was appended to the prompt).
      JSON.stringify({
        kind: 'value',
        value: 'ok',
        effect_name: null,
        effect_args: null,
      }),
    ]);
    const ctx = buildEvaluationContext({
      lm,
      signature: PROMPT_SIGNATURE,
      scope: new Map<string, CombinatorValue>([['input', 'hi']]),
      memory: initialMemoryState(SEEDED_SCHEMA),
      memorySchema: SEEDED_SCHEMA,
    });
    const result = await evaluate(oracle(vref('input'), undefined), ctx);
    expect(result).toBe('ok');
    expect(lm.captured).toHaveLength(2);

    const firstSystem = lm.captured[0]?.systemMessage ?? '';
    expect(firstSystem).toContain('[[RLM_MEMORY schema=seeded]]');
    expect(firstSystem).toContain('title: string = "draft"');
    expect(firstSystem).toContain('count: number = 0');

    const secondSystem = lm.captured[1]?.systemMessage ?? '';
    expect(secondSystem).toContain('[[RLM_MEMORY schema=seeded]]');
    expect(secondSystem).toContain('title: string = "ready"');
    expect(secondSystem).not.toContain('title: string = "draft"');
  });

  it('surfaces a structured error to the LM when the write is rejected by the schema', async () => {
    const lm = new RecordingLM([
      // Turn 1: write an unknown key.
      JSON.stringify({
        kind: 'effect',
        value: null,
        effect_name: 'WriteMemory',
        effect_args: { key: 'unknown', value: 'x' },
      }),
      // Turn 2: bail out with a value.
      JSON.stringify({
        kind: 'value',
        value: 'bailed',
        effect_name: null,
        effect_args: null,
      }),
    ]);
    const ctx = buildEvaluationContext({
      lm,
      signature: PROMPT_SIGNATURE,
      scope: new Map<string, CombinatorValue>([['input', 'hi']]),
      memory: initialMemoryState(SEEDED_SCHEMA),
      memorySchema: SEEDED_SCHEMA,
    });
    const result = await evaluate(oracle(vref('input'), undefined), ctx);
    expect(result).toBe('bailed');

    const secondUser = lm.captured[1]?.userMessage ?? '';
    expect(secondUser).toMatch(/\[\[EFFECT turn=0 kind=WriteMemory\]\]/);
    expect(secondUser).toMatch(/not declared in schema "seeded"/);

    const secondSystem = lm.captured[1]?.systemMessage ?? '';
    expect(secondSystem).toContain('title: string = "draft"');
    expect(secondSystem).not.toContain('title: string = "x"');
  });
});
