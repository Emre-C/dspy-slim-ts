/**
 * AX fixture oracle (§13 / §13.4): compare @ax-llm/ax to spec fixtures.
 *
 * This file is report-style: it documents where AX and our spec agree or diverge.
 * It is not a release gate on dspy-slim-ts — when AX rejects a fixture or behaves
 * differently, we log `[AX SKIP]` / `[AX DIVERGENCE]` instead of patching AX or the
 * spec here. Skips therefore record third-party limitations, not failures in our parser.
 *
 * No workarounds for AX; fixtures are fed raw.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { AxSignature } from '@ax-llm/ax';

const FIXTURES = resolve(import.meta.dirname, '../../spec/fixtures');

/** Sole catch site for AX calls (no retries). */
function tryAx<T>(fn: () => T): { ok: true; value: T } | { ok: false; error: string } {
  try {
    return { ok: true, value: fn() };
  } catch (e) {
    return { ok: false, error: (e as Error).message.split('\n')[0]! };
  }
}

describe('AX Oracle: Signature Parsing', () => {
  const fixtures = JSON.parse(
    readFileSync(resolve(FIXTURES, 'signature_parse.json'), 'utf-8'),
  );

  for (const c of fixtures.cases) {
    if (c.expected_error) {
      it(`[error case] ${c.id}: AX also rejects invalid input`, () => {
        const result = tryAx(() => AxSignature.create(c.input));
        expect(result.ok).toBe(false);
      });
      continue;
    }

    it(`${c.id}: cross-validate against AX`, () => {
      const result = tryAx(() => AxSignature.create(c.input));

      if (!result.ok) {
        console.warn(`[AX SKIP] ${c.id}: AX rejects "${c.input}": ${result.error}`);
        return;
      }

      const axInputs = result.value.getInputFields();
      const axOutputs = result.value.getOutputFields();

      expect(axInputs.length).toBe(c.expected.inputs.length);
      expect(axOutputs.length).toBe(c.expected.outputs.length);

      for (let i = 0; i < c.expected.inputs.length; i++) {
        expect(axInputs[i]!.name).toBe(c.expected.inputs[i].name);
      }
      for (let i = 0; i < c.expected.outputs.length; i++) {
        expect(axOutputs[i]!.name).toBe(c.expected.outputs[i].name);
      }
    });
  }
});

describe('AX Oracle: Title Inference (inferPrefix equivalent)', () => {
  const fixtures = JSON.parse(
    readFileSync(resolve(FIXTURES, 'infer_prefix.json'), 'utf-8'),
  );

  for (const c of fixtures.cases) {
    it(`${c.input}: compare AX title to spec "${c.expected}"`, () => {
      const dummyOut = c.input === 'dummy' ? 'result' : 'dummy';
      const result = tryAx(() => {
        const sig = AxSignature.create(`${c.input} -> ${dummyOut}`);
        return sig.getInputFields()[0]!.title;
      });

      if (!result.ok) {
        console.warn(`[AX SKIP] "${c.input}": AX rejects: ${result.error}`);
        return;
      }

      if (result.value === c.expected) {
        expect(result.value).toBe(c.expected);
      } else {
        console.warn(
          `[AX DIVERGENCE] "${c.input}": spec="${c.expected}", ax="${result.value}"`,
        );
        expect(result.value).not.toBe(c.expected);
      }
    });
  }
});

describe('AX Oracle: Signature Operations', () => {
  const fixtures = JSON.parse(
    readFileSync(resolve(FIXTURES, 'signature_ops.json'), 'utf-8'),
  );

  for (const c of fixtures.cases) {
    switch (c.op) {
      case 'append':
      case 'prepend':
      case 'delete':
      case 'with_instructions':
      case 'check_default_instructions': {
        it(`[AX SKIP] ${c.id}: no clean AX equivalent for "${c.op}"`, () => {});
        break;
      }

      case 'check_signature_string': {
        it(`${c.id}: AX toString contains expected field names`, () => {
          const result = tryAx(() => AxSignature.create(c.base_sig));
          if (!result.ok) {
            console.warn(`[AX SKIP] ${c.id}: AX rejects "${c.base_sig}"`);
            return;
          }

          const axStr = result.value.toString();
          expect(axStr).toContain('->');
          for (const name of c.expected.split(/\s*(?:->|,)\s*/)) {
            expect(axStr).toContain(name.trim());
          }
        });
        break;
      }

      case 'check_disjoint': {
        it(`${c.id}: AX enforces disjoint input/output keys`, () => {
          const result = tryAx(() => AxSignature.create(c.base_sig));
          if (!result.ok) {
            console.warn(`[AX SKIP] ${c.id}: AX rejects "${c.base_sig}"`);
            return;
          }

          const inputNames = new Set(
            result.value.getInputFields().map((f: { name: string }) => f.name),
          );
          const outputNames = result.value
            .getOutputFields()
            .map((f: { name: string }) => f.name);
          const intersection = outputNames.filter((n: string) => inputNames.has(n));
          expect(intersection).toEqual(c.expected_intersection);
        });
        break;
      }

      case 'check_fields_order': {
        it(`${c.id}: AX field ordering is inputs-before-outputs`, () => {
          const result = tryAx(() => AxSignature.create(c.base_sig));
          if (!result.ok) {
            console.warn(`[AX SKIP] ${c.id}: AX rejects "${c.base_sig}"`);
            return;
          }

          const allNames = [
            ...result.value.getInputFields().map((f: { name: string }) => f.name),
            ...result.value.getOutputFields().map((f: { name: string }) => f.name),
          ];
          expect(allNames).toEqual(c.expected_fields_ordered);
        });
        break;
      }

      default:
        break;
    }
  }
});
