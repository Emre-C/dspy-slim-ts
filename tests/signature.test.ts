import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import {
  createSignature,
  parseSignature,
  signatureFromString,
  appendField,
  prependField,
  deleteField,
  withInstructions,
  withUpdatedField,
  signatureEquals,
  signatureString,
  signatureFields,
} from '../src/signature.js';
import { createField, Field } from '../src/field.js';
import type { FieldKind, TypeTag } from '../src/types.js';

// ---------------------------------------------------------------------------
// Fixture types
// ---------------------------------------------------------------------------

interface ParseFixtureField {
  name: string;
  type_tag: string;
  is_type_undefined: boolean;
}

interface ParseCaseOk {
  id: string;
  input: string;
  expected: { inputs: ParseFixtureField[]; outputs: ParseFixtureField[] };
  expected_error?: undefined;
}

interface ParseCaseErr {
  id: string;
  input: string;
  expected_error: string;
  expected?: undefined;
}

type ParseCase = ParseCaseOk | ParseCaseErr;

interface OpsCase {
  id: string;
  base_sig: string;
  op: string;
  field_name?: string;
  field_kind?: string;
  field_type?: string;
  instructions?: string;
  expected_output_fields_ordered?: string[];
  expected_input_fields_ordered?: string[];
  expected_input_fields?: string[];
  expected_output_fields?: string[];
  expected_instructions?: string;
  expected_fields_ordered?: string[];
  expected?: string;
  expected_intersection?: string[];
}

// ---------------------------------------------------------------------------
// Load fixtures
// ---------------------------------------------------------------------------

const parseFixture: { cases: ParseCase[] } = JSON.parse(
  readFileSync(
    new URL('../../dspy-slim/spec/fixtures/signature_parse.json', import.meta.url),
    'utf-8',
  ),
);

const opsFixture: { cases: OpsCase[] } = JSON.parse(
  readFileSync(
    new URL('../../dspy-slim/spec/fixtures/signature_ops.json', import.meta.url),
    'utf-8',
  ),
);

// ---------------------------------------------------------------------------
// parseSignature
// ---------------------------------------------------------------------------

describe('parseSignature (fixture-driven)', () => {
  for (const c of parseFixture.cases) {
    if (c.expected_error != null) {
      it(`${c.id}: throws on "${c.input}"`, () => {
        expect(() => parseSignature(c.input)).toThrow(c.expected_error);
      });
    } else {
      it(`${c.id}: parses "${c.input}"`, () => {
        const result = parseSignature(c.input);

        expect(result.inputs.length).toBe(c.expected.inputs.length);
        for (let i = 0; i < c.expected.inputs.length; i++) {
          const exp = c.expected.inputs[i]!;
          const got = result.inputs[i]!;
          expect(got.name).toBe(exp.name);
          expect(got.typeTag).toBe(exp.type_tag);
          expect(got.isTypeUndefined).toBe(exp.is_type_undefined);
        }

        expect(result.outputs.length).toBe(c.expected.outputs.length);
        for (let i = 0; i < c.expected.outputs.length; i++) {
          const exp = c.expected.outputs[i]!;
          const got = result.outputs[i]!;
          expect(got.name).toBe(exp.name);
          expect(got.typeTag).toBe(exp.type_tag);
          expect(got.isTypeUndefined).toBe(exp.is_type_undefined);
        }
      });
    }
  }
});

// ---------------------------------------------------------------------------
// SignatureOps
// ---------------------------------------------------------------------------

describe('SignatureOps (fixture-driven)', () => {
  for (const c of opsFixture.cases) {
    it(c.id, () => {
      const sig = signatureFromString(c.base_sig);

      switch (c.op) {
        case 'append': {
          const result = appendField(
            sig,
            c.field_name!,
            c.field_kind! as FieldKind,
            c.field_type! as TypeTag,
          );
          if (c.expected_output_fields_ordered) {
            expect([...result.outputFields.keys()]).toEqual(
              c.expected_output_fields_ordered,
            );
          }
          if (c.expected_input_fields_ordered) {
            expect([...result.inputFields.keys()]).toEqual(
              c.expected_input_fields_ordered,
            );
          }
          break;
        }

        case 'prepend': {
          const result = prependField(
            sig,
            c.field_name!,
            c.field_kind! as FieldKind,
            c.field_type! as TypeTag,
          );
          if (c.expected_output_fields_ordered) {
            expect([...result.outputFields.keys()]).toEqual(
              c.expected_output_fields_ordered,
            );
          }
          break;
        }

        case 'delete': {
          const result = deleteField(sig, c.field_name!);
          expect([...result.inputFields.keys()]).toEqual(
            c.expected_input_fields,
          );
          expect([...result.outputFields.keys()]).toEqual(
            c.expected_output_fields,
          );
          break;
        }

        case 'with_instructions': {
          const result = withInstructions(sig, c.instructions!);
          expect(result.instructions).toBe(c.expected_instructions);
          break;
        }

        case 'check_default_instructions': {
          expect(sig.instructions).toBe(c.expected_instructions);
          break;
        }

        case 'check_fields_order': {
          const names = signatureFields(sig).map((f) => f.name);
          expect(names).toEqual(c.expected_fields_ordered);
          break;
        }

        case 'check_signature_string': {
          expect(signatureString(sig)).toBe(c.expected);
          break;
        }

        case 'check_disjoint': {
          const inputKeys = new Set(sig.inputFields.keys());
          const outputKeys = new Set(sig.outputFields.keys());
          const intersection = [...inputKeys].filter((k) => outputKeys.has(k));
          expect(intersection).toEqual(c.expected_intersection);
          break;
        }

        default:
          throw new Error(`Unknown op: ${c.op}`);
      }
    });
  }
});

describe('Signature hardening', () => {
  it('parses comma-bearing nested type expressions without inventing extra fields', () => {
    const result = parseSignature(
      'payload: dict[str, int], verdict: literal["yes, please", "no"] -> answer: union[str, bool]',
    );

    expect(result.inputs).toEqual([
      { name: 'payload', typeTag: 'dict', isTypeUndefined: false },
      { name: 'verdict', typeTag: 'literal', isTypeUndefined: false },
    ]);
    expect(result.outputs).toEqual([
      { name: 'answer', typeTag: 'union', isTypeUndefined: false },
    ]);
  });

  it('rejects duplicate names within the input side', () => {
    expect(() => parseSignature('question, question -> answer')).toThrow(
      'Duplicate input field "question"',
    );
  });

  it('rejects duplicate names within the output side', () => {
    expect(() => parseSignature('question -> answer, answer')).toThrow(
      'Duplicate output field "answer"',
    );
  });

  it('creates true Field value objects with defensive copies', () => {
    const typeArgs: TypeTag[] = ['str'];
    const constraints = ['required'];
    const field = createField({
      kind: 'output',
      name: 'answer',
      typeTag: 'list',
      typeArgs,
      constraints,
      description: 'Primary answer',
      isTypeUndefined: false,
    });

    typeArgs.push('int');
    constraints.push('nonempty');

    expect(field).toBeInstanceOf(Field);
    expect(field.typeArgs).toEqual(['str']);
    expect(field.constraints).toEqual(['required']);
  });

  it('createSignature rejects map keys that disagree with field names', () => {
    const badInputs = new Map([
      ['question', createField({ kind: 'input', name: 'prompt' })],
    ]);

    expect(() => createSignature(badInputs, new Map())).toThrow(
      'does not match field.name',
    );
  });

  it('createSignature rejects fields stored under the wrong section kind', () => {
    const badOutputs = new Map([
      ['answer', createField({ kind: 'input', name: 'answer' })],
    ]);

    expect(() => createSignature(new Map(), badOutputs)).toThrow(
      'stored in outputFields',
    );
  });

  it('signature maps are insulated from caller mutation on both ingress and egress', () => {
    const originalInputs = new Map([
      ['question', createField({ kind: 'input', name: 'question' })],
    ]);
    const sig = createSignature(originalInputs, new Map());

    originalInputs.set('extra', createField({ kind: 'input', name: 'extra' }));
    expect([...sig.inputFields.keys()]).toEqual(['question']);

    const leakedMap = sig.inputFields as Map<string, Field>;
    leakedMap.set('mutated', createField({ kind: 'input', name: 'mutated' }));
    expect([...sig.inputFields.keys()]).toEqual(['question']);
  });

  it('withUpdatedField rejects field/name mismatches', () => {
    const sig = signatureFromString('question -> answer');

    expect(() =>
      withUpdatedField(
        sig,
        'answer',
        createField({ kind: 'output', name: 'finalAnswer' }),
      ),
    ).toThrow('does not match field.name');
  });

  it('withUpdatedField rejects field kind mismatches', () => {
    const sig = signatureFromString('question -> answer');

    expect(() =>
      withUpdatedField(
        sig,
        'answer',
        createField({ kind: 'input', name: 'answer' }),
      ),
    ).toThrow('stored in outputFields');
  });

  it('appendField preserves the full metadata of an existing Field value object', () => {
    const sig = signatureFromString('question -> answer');
    const field = createField({
      kind: 'output',
      name: 'confidence',
      typeTag: 'float',
      description: 'Confidence score',
      prefix: 'Confidence Score',
      constraints: ['0 <= value <= 1'],
      isTypeUndefined: false,
    });

    const result = appendField(sig, field);
    const appended = result.outputFields.get('confidence');

    expect(appended?.description).toBe('Confidence score');
    expect(appended?.prefix).toBe('Confidence Score');
    expect(appended?.constraints).toEqual(['0 <= value <= 1']);
  });

  it('signatureEquals compares full field metadata rather than only names and type tags', () => {
    const base = signatureFromString('question -> answer');
    const enriched = withUpdatedField(
      base,
      'answer',
      createField({
        kind: 'output',
        name: 'answer',
        description: 'Final answer',
        prefix: 'Final Answer',
      }),
    );

    expect(signatureEquals(base, enriched)).toBe(false);
  });

  it('Field owns nested default values instead of aliasing caller objects', () => {
    const defaultValue = { thresholds: [0.2, 0.8] };
    const field = createField({
      kind: 'input',
      name: 'score',
      default: defaultValue,
    });

    defaultValue.thresholds[0] = 0.1;

    expect(field.default).toEqual({ thresholds: [0.2, 0.8] });
  });
});
