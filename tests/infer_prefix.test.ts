import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { inferPrefix } from '../src/infer_prefix.js';

interface InferPrefixCase {
  input: string;
  expected: string;
}

const fixture: { cases: InferPrefixCase[] } = JSON.parse(
  readFileSync(
    new URL('../../dspy-slim/spec/fixtures/infer_prefix.json', import.meta.url),
    'utf-8',
  ),
);

describe('inferPrefix (fixture-driven)', () => {
  for (const c of fixture.cases) {
    it(`"${c.input}" → "${c.expected}"`, () => {
      expect(inferPrefix(c.input)).toBe(c.expected);
    });
  }
});
