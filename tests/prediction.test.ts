import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { Example } from '../src/example.js';
import { Prediction } from '../src/prediction.js';

interface FixtureCase {
  id: string;
  op: string;
  completions?: Record<string, unknown[]>;
  expected_store?: Record<string, unknown>;
  expected_completions_len?: number;
  store?: Record<string, unknown>;
  a_store?: Record<string, unknown>;
  b_value?: number;
  divisor?: number;
  expected?: unknown;
  expected_error?: string;
}

const fixture = JSON.parse(
  readFileSync(
    new URL(
      '../../spec/fixtures/prediction_ops.json',
      import.meta.url,
    ),
    'utf-8',
  ),
) as { cases: FixtureCase[] };

describe('Prediction (spec fixtures)', () => {
  for (const c of fixture.cases) {
    it(c.id, () => {
      switch (c.op) {
        case 'from_completions': {
          const pred = Prediction.fromCompletions(c.completions!);
          const dict = pred.toDict();
          for (const [key, val] of Object.entries(c.expected_store!)) {
            expect(dict[key]).toEqual(val);
          }
          expect(pred.completions!.length).toBe(c.expected_completions_len);
          break;
        }

        case 'float': {
          if (c.expected_error) {
            expect(() => Prediction.create(c.store!).toFloat()).toThrow(
              c.expected_error,
            );
          } else {
            expect(Prediction.create(c.store!).toFloat()).toBe(c.expected);
          }
          break;
        }

        case 'add': {
          expect(Prediction.create(c.a_store!).add(c.b_value!)).toBeCloseTo(
            c.expected as number,
          );
          break;
        }

        case 'div': {
          expect(Prediction.create(c.store!).div(c.divisor!)).toBeCloseTo(
            c.expected as number,
          );
          break;
        }

        case 'lt': {
          expect(Prediction.create(c.a_store!).lt(c.b_value!)).toBe(
            c.expected,
          );
          break;
        }

        case 'ge': {
          expect(Prediction.create(c.a_store!).ge(c.b_value!)).toBe(
            c.expected,
          );
          break;
        }

        case 'check_input_keys_absent': {
          expect(Prediction.create(c.store!).hasInputKeys()).toBe(false);
          break;
        }

        default:
          throw new Error(`Unknown op: ${c.op}`);
      }
    });
  }
});

describe('Prediction hardening', () => {
  it('uses a dedicated abstraction boundary instead of inheriting Example behavior', () => {
    const pred = Prediction.create({ answer: 'Paris' });

    expect(pred).not.toBeInstanceOf(Example);
    expect('withInputs' in pred).toBe(false);
    expect('inputs' in pred).toBe(false);
    expect('copy' in pred).toBe(false);
  });

  it('defensively copies store input on construction', () => {
    const store = { answer: 'Paris', score: 0.9 };
    const pred = Prediction.create(store);

    store.answer = 'Lyon';
    store.score = 0.1;

    expect(pred.toDict()).toEqual({ answer: 'Paris', score: 0.9 });
  });

  it('defensively copies completion arrays and returned completion views', () => {
    const completions = {
      answer: ['Paris', 'Lyon'],
      score: [0.9, 0.1],
    };
    const pred = Prediction.fromCompletions(completions);

    completions.answer[0] = 'Berlin';
    completions.score.push(0.05);

    expect(pred.toDict()).toEqual({ answer: 'Paris', score: 0.9 });
    expect(pred.completions?.get('answer')).toEqual(['Paris', 'Lyon']);

    const leaked = pred.completions?.get('answer') as unknown[];
    expect(() => leaked.push('Rome')).toThrow();
    expect(pred.completions?.get('answer')).toEqual(['Paris', 'Lyon']);
  });

  it('owns nested completion payloads and recursively serializes nested values', () => {
    const source = { meta: { confidence: 0.9 } };
    const pred = Prediction.fromCompletions({
      answer: [new Example({ text: 'Paris' })],
      details: [source],
    });

    source.meta.confidence = 0.1;

    expect(pred.toDict()).toEqual({
      answer: { text: 'Paris' },
      details: { meta: { confidence: 0.9 } },
    });
    expect(JSON.parse(JSON.stringify(pred))).toEqual(pred.toDict());
  });

  it('supports list-based completion input without accidental coercion', () => {
    const pred = Prediction.fromCompletions([
      { answer: 'Paris', score: 0.9 },
      { answer: 'Lyon', score: 0.1 },
    ]);

    expect(pred.toDict()).toEqual({ answer: 'Paris', score: 0.9 });
    expect(pred.completions?.get('answer')).toEqual(['Paris', 'Lyon']);
    expect(pred.completions?.length).toBe(2);
  });

  it('integrates its numeric protocol with native JavaScript coercion', () => {
    const pred = Prediction.create({ answer: 'Paris', score: 0.5 });

    expect(Number(pred)).toBe(0.5);
    expect(Number(pred) > 0.4).toBe(true);
    expect(Math.max(Number(pred), 0.4)).toBe(0.5);
    expect(`${pred}`).toBe('Prediction({"answer":"Paris","score":0.5})');
  });

  it('rejects non-numeric score values during numeric coercion', () => {
    const pred = Prediction.create({ score: 'not-a-number' });

    expect(() => Number(pred)).toThrow('score must be numeric');
  });
});
