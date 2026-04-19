import { describe, it, expect, expectTypeOf } from 'vitest';

import { Predict, ChainOfThought, Prediction, signatureFromString } from '../src/index.js';
import type { ParseSignature } from '../src/signature_types.js';

/**
 * Dual purpose: (1) static contract — `ParseSignature<'...'>` must match what we claim
 * the parser produces; (2) runtime parity — `signatureFromString` must agree with those
 * types so tests fail if implementation and type machinery drift apart.
 */

describe('ParseSignature — simple forms', () => {
  it('parses untyped fields as strings', () => {
    type Sig = ParseSignature<'question -> answer'>;
    expectTypeOf<Sig['inputs']>().toEqualTypeOf<{ question: string }>();
    expectTypeOf<Sig['outputs']>().toEqualTypeOf<{ answer: string }>();

    const sig = signatureFromString('question -> answer');
    expect([...sig.inputFields.keys()]).toEqual(['question']);
    expect([...sig.outputFields.keys()]).toEqual(['answer']);
  });

  it('maps scalar type tags to their native TypeScript types', () => {
    type Sig = ParseSignature<'q: str, n: int, f: float, b: bool -> answer: str'>;
    expectTypeOf<Sig['inputs']>().toEqualTypeOf<{
      q: string;
      n: number;
      f: number;
      b: boolean;
    }>();
    expectTypeOf<Sig['outputs']>().toEqualTypeOf<{ answer: string }>();

    const sig = signatureFromString('q: str, n: int, f: float, b: bool -> answer: str');
    expect([...sig.inputFields.keys()]).toEqual(['q', 'n', 'f', 'b']);
    expect(sig.inputFields.get('n')?.typeTag).toBe('int');
  });

  it('ignores surrounding whitespace', () => {
    type Sig = ParseSignature<'  q : str  ->   a : bool  ,  b  '>;
    expectTypeOf<Sig['inputs']>().toEqualTypeOf<{ q: string }>();
    expectTypeOf<Sig['outputs']>().toEqualTypeOf<{ a: boolean; b: string }>();
  });

  it('falls back to unknown for unrecognized type tags', () => {
    type Sig = ParseSignature<'input: mystery -> output'>;
    expectTypeOf<Sig['inputs']>().toEqualTypeOf<{ input: unknown }>();
    expectTypeOf<Sig['outputs']>().toEqualTypeOf<{ output: string }>();
  });
});

describe('ParseSignature — bracket-aware splitting', () => {
  it('treats commas inside brackets as part of the type, not a field separator', () => {
    // This is the key parity case. The runtime parser in src/split.ts walks
    // depth-tracking brackets; the type-level parser must do the same or it
    // will carve the signature into wrong-shaped fields.
    type Sig = ParseSignature<'q: dict[str, int] -> a: list[int]'>;
    expectTypeOf<Sig['inputs']>().toEqualTypeOf<{ q: Record<string, unknown> }>();
    expectTypeOf<Sig['outputs']>().toEqualTypeOf<{ a: readonly unknown[] }>();

    const sig = signatureFromString('q: dict[str, int] -> a: list[int]');
    expect([...sig.inputFields.keys()]).toEqual(['q']);
    expect([...sig.outputFields.keys()]).toEqual(['a']);
  });
});

describe('ParseSignature — degraded-input fallback', () => {
  it('returns a permissive record when the input is the widened string type', () => {
    // The constructor of Predict accepts `TSig extends string | Signature`.
    // If a caller passes a widened `string` variable instead of a literal,
    // `ParseSignature<string>` must fall through to a permissive shape so
    // non-literal callers are not flagged.
    type Sig = ParseSignature<string>;
    expectTypeOf<Sig['inputs']>().toEqualTypeOf<Record<string, unknown>>();
    expectTypeOf<Sig['outputs']>().toEqualTypeOf<Record<string, unknown>>();
  });
});

describe('Predictor generic propagation', () => {
  it('narrows Predict inputs and outputs from a literal signature string', () => {
    const predictor = new Predict('question -> answer');

    expectTypeOf<Parameters<typeof predictor.forward>[0]>()
      .toMatchTypeOf<{ question?: string } | undefined>();
    expectTypeOf<ReturnType<typeof predictor.forward>>()
      .toEqualTypeOf<Prediction<{ answer: string }>>();
  });

  it('injects the synthetic reasoning field into ChainOfThought outputs', () => {
    const chain = new ChainOfThought('question -> answer');
    expectTypeOf<ReturnType<typeof chain.forward>>()
      .toEqualTypeOf<Prediction<{ answer: string; reasoning: string }>>();
  });
});

describe('Prediction.getTyped', () => {
  it('returns the advisory output type without widening to undefined', () => {
    const prediction = Prediction.create<{ answer: string; score: number }>({
      answer: '42',
      score: 1,
    });

    expectTypeOf(prediction.getTyped('answer')).toEqualTypeOf<string>();
    expectTypeOf(prediction.getTyped('score')).toEqualTypeOf<number>();

    expect(prediction.getTyped('answer')).toBe('42');
    expect(prediction.getTyped('score')).toBe(1);
  });

  it('throws KeyError on a missing key, mirroring get()', () => {
    const prediction = Prediction.create<{ answer: string }>({});
    expect(() => prediction.getTyped('answer')).toThrow('Key "answer" not found');
  });
});

describe('Negative type assertions', () => {
  it('rejects object literals with unknown keys on a typed predict', () => {
    const predictor = new Predict('question -> answer');
    // The excess-property check on fresh object literals is the single most
    // important load-bearing property of the type-level signature work.
    // @ts-expect-error — `mystery` is not an input field of `question -> answer`
    void (() => predictor.forward({ question: 'hi', mystery: 'oops' }));
    // Sanity: a well-formed call must still compile.
    void (() => predictor.forward({ question: 'hi' }));
  });

  it('accepts control-plane overrides (config / demos / signature / lm) alongside inputs', () => {
    const predictor = new Predict('question -> answer');
    // These must not be flagged as excess properties even though they are
    // not declared in the signature's input fields.
    void (() => predictor.forward({ question: 'hi', config: { temperature: 0.2 } }));
    void (() => predictor.forward({ question: 'hi', demos: [] }));
  });
});
