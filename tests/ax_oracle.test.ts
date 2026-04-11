/**
 * AX Test Oracle — §13 of the spec.
 *
 * Cross-validates our spec fixtures against @ax-llm/ax to surface
 * agreements and divergences. This file does NOT test our code. It tests
 * AX against our fixtures and records the results as a living report.
 *
 * RULES (from §13.4 — preventing mediocrity seepage):
 *   1. NEVER work around AX bugs — if AX can't handle a case, skip it.
 *   2. NEVER translate our inputs to fit AX — feed raw or skip.
 *   3. NEVER define types to match AX's internal API.
 *   4. This file is a report, not a test suite we "must make green."
 *      Skips are expected and valuable — they document AX limitations.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { AxSignature } from '@ax-llm/ax';

const FIXTURES = resolve(import.meta.dirname, '../../spec/fixtures');

/**
 * Attempt to call AX and return its result, or null if AX rejects the input.
 * This is the ONLY place AX errors are caught — no workarounds, no retries.
 */
function tryAx<T>(fn: () => T): { ok: true; value: T } | { ok: false; error: string } {
  try {
    return { ok: true, value: fn() };
  } catch (e) {
    return { ok: false, error: (e as Error).message.split('\n')[0]! };
  }
}

// ---------------------------------------------------------------------------
// §13.2 — Signature Parsing
// ---------------------------------------------------------------------------

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
      // Feed our EXACT fixture input to AX — no translation.
      const result = tryAx(() => AxSignature.create(c.input));

      if (!result.ok) {
        // AX rejects our valid fixture input. Document and skip.
        console.warn(`[AX SKIP] ${c.id}: AX rejects "${c.input}": ${result.error}`);
        return;
      }

      const axInputs = result.value.getInputFields();
      const axOutputs = result.value.getOutputFields();

      // Field counts must agree.
      expect(axInputs.length).toBe(c.expected.inputs.length);
      expect(axOutputs.length).toBe(c.expected.outputs.length);

      // Field names must agree.
      for (let i = 0; i < c.expected.inputs.length; i++) {
        expect(axInputs[i]!.name).toBe(c.expected.inputs[i].name);
      }
      for (let i = 0; i < c.expected.outputs.length; i++) {
        expect(axOutputs[i]!.name).toBe(c.expected.outputs[i].name);
      }
    });
  }
});

// ---------------------------------------------------------------------------
// §13.2 — infer_prefix: AX's toTitle vs our spec
// ---------------------------------------------------------------------------

describe('AX Oracle: Title Inference (inferPrefix equivalent)', () => {
  const fixtures = JSON.parse(
    readFileSync(resolve(FIXTURES, 'infer_prefix.json'), 'utf-8'),
  );

  for (const c of fixtures.cases) {
    it(`${c.input}: compare AX title to spec "${c.expected}"`, () => {
      // Create a minimal signature to extract AX's inferred title.
      const dummyOut = c.input === 'dummy' ? 'result' : 'dummy';
      const result = tryAx(() => {
        const sig = AxSignature.create(`${c.input} -> ${dummyOut}`);
        return sig.getInputFields()[0]!.title;
      });

      if (!result.ok) {
        // AX rejects this field name entirely. Document the limitation.
        console.warn(`[AX SKIP] "${c.input}": AX rejects: ${result.error}`);
        return;
      }

      if (result.value === c.expected) {
        // Agreement — both produce the same title.
        expect(result.value).toBe(c.expected);
      } else {
        // Divergence — document it. Spec wins per §13.1 rule 2b.
        console.warn(
          `[AX DIVERGENCE] "${c.input}": spec="${c.expected}", ax="${result.value}"`,
        );
        expect(result.value).not.toBe(c.expected);
      }
    });
  }
});

// ---------------------------------------------------------------------------
// §13.2 — Signature Operations
// ---------------------------------------------------------------------------

describe('AX Oracle: Signature Operations', () => {
  const fixtures = JSON.parse(
    readFileSync(resolve(FIXTURES, 'signature_ops.json'), 'utf-8'),
  );

  for (const c of fixtures.cases) {
    // Only cross-validate ops that AX supports without workarounds.
    // AX's immutable append/prepend API is broken (its own parseField
    // rejects the AxFieldType its type declarations advertise).
    // We do NOT work around this — we skip and document.

    switch (c.op) {
      case 'append':
      case 'prepend':
      case 'delete':
      case 'with_instructions':
      case 'check_default_instructions': {
        it(`[AX SKIP] ${c.id}: no clean AX equivalent for "${c.op}"`, () => {
          // append/prepend: AX's immutable API is broken.
          // delete: AX has no deleteField API.
          // with_instructions/check_default_instructions: AX uses
          //   setDescription (different concept than DSPy instructions).
          // Skipping is the correct choice — zero workarounds.
        });
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
