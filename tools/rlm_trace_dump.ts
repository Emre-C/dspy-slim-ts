/**
 * One-shot diagnostic: export the first LongCoT logic/easy question, run
 * the RLM against Fireworks with trace enabled, and dump a compact
 * per-turn summary of the oracle effect loop. Read-only; no fixture writes.
 */

import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { TaskType } from '../src/rlm_task_router.js';
import { ChatAdapter, LM, RLM, settings } from '../src/index.js';

const REPO_ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const LONGCOT_DIR = resolve(REPO_ROOT, 'tools', 'longcot');

function loadRootEnvFile(): void {
  try {
    const raw = readFileSync(resolve(REPO_ROOT, '.env'), 'utf-8');
    for (const line of raw.split('\n')) {
      const t = line.trim();
      if (t === '' || t.startsWith('#')) continue;
      const eq = t.indexOf('=');
      if (eq <= 0) continue;
      const k = t.slice(0, eq).trim();
      let v = t.slice(eq + 1).trim();
      if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
        v = v.slice(1, -1);
      }
      if (process.env[k] === undefined) process.env[k] = v;
    }
  } catch {
    /* absent */
  }
}

interface Question {
  readonly question_id: string;
  readonly prompt: string;
}

function firstLogicEasyQuestion(): Question {
  const result = spawnSync(
    'uv',
    ['run', 'python', 'export_questions.py', '--domain', 'logic', '--difficulty', 'easy', '--max', '1'],
    { cwd: LONGCOT_DIR, encoding: 'utf-8', maxBuffer: 64 * 1024 * 1024 },
  );
  if (result.status !== 0) throw new Error(result.stderr || 'export failed');
  const line = result.stdout.split('\n').find((l) => l.trim().length > 0);
  if (line === undefined) throw new Error('no questions');
  return JSON.parse(line) as Question;
}

interface TraceRow {
  readonly step: number;
  readonly nodeTag: string;
  readonly ok: boolean;
  readonly durationMs: number;
  readonly extras?: Readonly<Record<string, unknown>>;
}

async function main(): Promise<void> {
  loadRootEnvFile();
  const apiKey = process.env.FIREWORKS_API_KEY;
  if (apiKey === undefined) {
    console.error('FIREWORKS_API_KEY missing');
    process.exit(1);
  }

  const taskTypeArg = (process.argv[2] ?? 'multi_hop') as TaskType;
  const maxEffectTurns = Number(process.argv[3] ?? 8);
  const maxOracleCalls = Number(process.argv[4] ?? 32);

  const q = firstLogicEasyQuestion();
  console.error(`[trace] question_id=${q.question_id} promptLen=${q.prompt.length}`);
  console.error(`[trace] taskType=${taskTypeArg} maxEffectTurns=${maxEffectTurns} maxOracleCalls=${maxOracleCalls}`);

  settings.configure({
    lm: new LM({
      model: 'accounts/fireworks/routers/kimi-k2p5-turbo',
      apiKey,
      apiBase: 'https://api.fireworks.ai/inference/v1',
      kwargs: { max_completion_tokens: 8192 },
    }),
  });

  const rlm = new RLM('prompt: str -> answer: str', {
    taskType: taskTypeArg,
    budget: { maxOracleCalls, maxEffectTurns, selfConsistencyN: 1 },
    trackTrace: true,
  });

  const t0 = Date.now();
  let answer = '';
  let errorMessage: string | null = null;
  let trace: readonly TraceRow[] = [];
  try {
    const pred = await settings.context({ adapter: new ChatAdapter() }, async () =>
      rlm.aforward({ prompt: q.prompt }),
    );
    answer = String(pred.getOr('answer', '') ?? '');
    trace = pred.getOr('_rlm_trace', []) as readonly TraceRow[];
  } catch (err) {
    errorMessage = err instanceof Error ? err.message : String(err);
  }
  const dt = Date.now() - t0;

  // Summarise trace: tag histogram, per-effect turn sequence.
  const tagCounts = new Map<string, number>();
  for (const row of trace) {
    tagCounts.set(row.nodeTag, (tagCounts.get(row.nodeTag) ?? 0) + 1);
  }

  const effectTurns = trace
    .filter((r) => r.nodeTag === 'effect')
    .map((r) => ({
      step: r.step,
      effectKind: (r.extras as { effectKind?: string } | undefined)?.effectKind,
      ok: r.ok,
      durationMs: Math.round(r.durationMs),
    }));

  const oracleLeaves = trace.filter((r) => r.nodeTag === 'oracle');

  console.log(
    JSON.stringify(
      {
        elapsed_ms: dt,
        error: errorMessage,
        answer_preview: answer.slice(0, 240),
        answer_len: answer.length,
        trace_size: trace.length,
        tag_histogram: Object.fromEntries(tagCounts),
        oracle_count: oracleLeaves.length,
        effect_turns: effectTurns,
      },
      null,
      2,
    ),
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
