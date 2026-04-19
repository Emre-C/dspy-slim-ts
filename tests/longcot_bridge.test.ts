/**
 * Smoke-test the LongCoT uv bridge (no HF_TOKEN; no RLM calls).
 */

import { randomUUID } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdirSync, unlinkSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const TEST_DIR = dirname(fileURLToPath(import.meta.url));
const LONGCOT_DIR = join(TEST_DIR, '..', 'tools', 'longcot');

describe('LongCoT uv bridge', () => {
  it('exports at least one question', () => {
    const r = spawnSync(
      'uv',
      ['run', 'python', 'export_questions.py', '--domain', 'logic', '--difficulty', 'easy', '--max', '1'],
      { cwd: LONGCOT_DIR, encoding: 'utf-8', maxBuffer: 32 * 1024 * 1024 },
    );
    expect(r.error, String(r.stderr)).toBeUndefined();
    expect(r.status, r.stderr).toBe(0);
    const line = r.stdout.trim().split('\n').find((l) => l.length > 0);
    expect(line).toBeDefined();
    const q = JSON.parse(line!) as { question_id?: string; prompt?: string; problem?: { template?: string } };
    expect(q.question_id).toBeTruthy();
    expect(q.prompt).toContain('solution');
    expect(q.problem?.template).toBeTruthy();
  });

  it('runs score_responses.py (LongCoT verify) on a stub answer', () => {
    const exp = spawnSync(
      'uv',
      ['run', 'python', 'export_questions.py', '--domain', 'logic', '--difficulty', 'easy', '--max', '1'],
      { cwd: LONGCOT_DIR, encoding: 'utf-8', maxBuffer: 32 * 1024 * 1024 },
    );
    expect(exp.status, exp.stderr).toBe(0);
    const line = exp.stdout.trim().split('\n').find((l) => l.length > 0);
    expect(line).toBeDefined();
    const q = JSON.parse(line!) as Record<string, unknown>;

    const runsDir = join(LONGCOT_DIR, 'runs');
    mkdirSync(runsDir, { recursive: true });
    const jsonlPath = join(runsDir, `vitest-longcot-${randomUUID()}.jsonl`);
    writeFileSync(
      jsonlPath,
      `${JSON.stringify({
        question: q,
        response_text: 'solution = []',
        error: null,
      })}\n`,
      'utf-8',
    );

    const score = spawnSync(
      'uv',
      ['run', 'python', 'score_responses.py', jsonlPath, '--no-fallback'],
      { cwd: LONGCOT_DIR, encoding: 'utf-8', maxBuffer: 4 * 1024 * 1024 },
    );
    try {
      expect(score.status, score.stderr).toBe(0);
      const summary = JSON.parse(score.stdout.trim()) as {
        total: number;
        failed: number;
        incorrect: number;
        correct: number;
      };
      expect(summary.total).toBe(1);
      expect(summary.failed).toBe(0);
      expect(summary.incorrect).toBe(1);
      expect(summary.correct).toBe(0);
    } finally {
      unlinkSync(jsonlPath);
    }
  });
});
