import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { Example } from '../src/example.js';

interface FixtureCase {
  id: string;
  op: string;
  kwargs?: Record<string, unknown>;
  expected_keys?: string[];
  expected_len?: number;
  input_keys?: string[];
  expected_inputs_keys?: string[];
  expected_labels_keys?: string[];
  expected_result_has_input_keys?: boolean;
  expected_result_input_keys?: null;
  get_key?: string;
  default?: unknown;
  expected?: unknown;
  check_key?: string;
  overrides?: Record<string, unknown>;
  expected_store?: Record<string, unknown>;
  remove_keys?: string[];
  a?: Record<string, unknown>;
  b?: Record<string, unknown>;
  expected_error?: string;
}

const fixture = JSON.parse(
  readFileSync(
    new URL('../../dspy-slim/spec/fixtures/example_ops.json', import.meta.url),
    'utf-8',
  ),
) as { cases: FixtureCase[] };

describe('Example (spec fixtures)', () => {
  for (const c of fixture.cases) {
    it(c.id, () => {
      switch (c.op) {
        case 'create': {
          const ex = new Example(c.kwargs);
          expect([...ex.keys()].sort()).toEqual([...c.expected_keys!].sort());
          expect(ex.len()).toBe(c.expected_len);
          break;
        }

        case 'with_inputs': {
          const ex = new Example(c.kwargs).withInputs(...c.input_keys!);
          expect([...ex.inputs().keys()].sort()).toEqual(
            [...c.expected_inputs_keys!].sort(),
          );
          expect([...ex.labels().keys()].sort()).toEqual(
            [...c.expected_labels_keys!].sort(),
          );
          break;
        }

        case 'inputs': {
          const result = new Example(c.kwargs)
            .withInputs(...c.input_keys!)
            .inputs();
          expect(result.hasInputKeys()).toBe(c.expected_result_has_input_keys);
          break;
        }

        case 'labels': {
          const result = new Example(c.kwargs)
            .withInputs(...c.input_keys!)
            .labels();
          expect(result.hasInputKeys()).toBe(false);
          break;
        }

        case 'get': {
          expect(new Example(c.kwargs).get(c.get_key!)).toBe(c.expected);
          break;
        }

        case 'get_or': {
          expect(new Example(c.kwargs).getOr(c.get_key!, c.default)).toBe(
            c.expected,
          );
          break;
        }

        case 'contains': {
          expect(new Example(c.kwargs).has(c.check_key!)).toBe(c.expected);
          break;
        }

        case 'len': {
          expect(new Example(c.kwargs).len()).toBe(c.expected);
          break;
        }

        case 'copy': {
          expect(new Example(c.kwargs).copy(c.overrides).toDict()).toEqual(
            c.expected_store,
          );
          break;
        }

        case 'without': {
          expect(
            [...new Example(c.kwargs).without(...c.remove_keys!).keys()].sort(),
          ).toEqual([...c.expected_keys!].sort());
          break;
        }

        case 'to_dict': {
          expect(new Example(c.kwargs).toDict()).toEqual(c.expected);
          break;
        }

        case 'equals': {
          expect(new Example(c.a!).equals(new Example(c.b!))).toBe(c.expected);
          break;
        }

        case 'inputs_error': {
          expect(() => new Example(c.kwargs).inputs()).toThrow(
            c.expected_error!,
          );
          break;
        }

        default:
          throw new Error(`Unknown op: ${c.op}`);
      }
    });
  }
});

describe('Example hardening', () => {
  it('defensively copies construction inputs', () => {
    const data = { question: 'Why?', answer: 'Because.' };
    const inputKeys = new Set(['question']);
    const example = new Example(data, inputKeys);

    data.answer = 'Changed';
    inputKeys.clear();

    expect(example.get('answer')).toBe('Because.');
    expect(example.inputs().keys()).toEqual(['question']);
  });

  it('equality includes the input-key partition state', () => {
    const base = new Example({ question: 'Why?', answer: 'Because.' });
    const partitioned = base.withInputs('question');

    expect(base.equals(partitioned)).toBe(false);
  });

  it('owns and recursively serializes nested values instead of leaking internal shapes', () => {
    const nested = { meta: { source: 'web' } };
    const example = new Example({
      payload: nested,
      child: new Example({ answer: 'Paris' }),
      items: [new Example({ label: 'capital' }), { ok: true }],
    });

    nested.meta.source = 'mutated';

    expect(example.toDict()).toEqual({
      payload: { meta: { source: 'web' } },
      child: { answer: 'Paris' },
      items: [{ label: 'capital' }, { ok: true }],
    });
    expect(JSON.parse(JSON.stringify(example))).toEqual(example.toDict());
  });

  it('treats nested plain objects as owned values for equality', () => {
    const left = new Example({ payload: { score: 0.9 } });
    const right = new Example({ payload: { score: 0.9 } });

    expect(left.equals(right)).toBe(true);
  });
});
