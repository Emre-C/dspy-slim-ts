/**
 * Local RLM smoke scripts (not part of the test suite).
 *
 *   npx tsx scratch/debug_rlm.ts [longcot | rlm-class]
 *
 *   longcot (default) — one LongCoT question (uv + export_questions.py),
 *     RLM v2 with task classification + routing (same stack as bench:longcot).
 *   rlm-class — fixed blocks prompt, `subLm` + tight budget, optional LM
 *     request logging (legacy debug_rlm2 behavior).
 *
 * Requires: `cd tools/longcot && uv sync` for longcot mode; FIREWORKS_API_KEY.
 */

import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { ChatAdapter, LM, RLM, settings } from '../src/index.js';

const REPO_ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const LONGCOT_DIR = resolve(REPO_ROOT, 'tools', 'longcot');

interface LongcotQuestion {
  readonly question_id: string;
  readonly prompt: string;
}

function exportOneQuestion(domain: string, difficulty: string): LongcotQuestion {
  const result = spawnSync(
    'uv',
    [
      'run',
      'python',
      'export_questions.py',
      '--domain',
      domain,
      '--difficulty',
      difficulty,
      '--max',
      '1',
    ],
    { cwd: LONGCOT_DIR, encoding: 'utf-8', maxBuffer: 64 * 1024 * 1024 },
  );
  if (result.status !== 0) {
    throw new Error(
      (result.stderr || result.stdout || 'export_questions failed') +
        '\nInstall uv and run: cd tools/longcot && uv sync',
    );
  }
  const line = result.stdout.split('\n').find((l) => l.trim().length > 0);
  if (line === undefined) throw new Error('no questions from export');
  return JSON.parse(line) as LongcotQuestion;
}

type Mode = 'longcot' | 'rlm-class';

function parseMode(argv: string[]): Mode {
  const m = argv[2];
  if (m === 'rlm-class') return 'rlm-class';
  return 'longcot';
}

function patchLmLogging(): void {
  const originalAforward = LM.prototype.aforward;
  LM.prototype.aforward = async function (prompt, messages, kwargs) {
    const t0 = Date.now();
    console.log(`[LM] Requesting...`);
    const response = await originalAforward.call(this, prompt, messages, kwargs);
    console.log(`[LM] Received response in ${Date.now() - t0}ms`);

    if (response.outputs && response.outputs.length > 0) {
      const text = response.outputs[0].text;
      console.log(
        `[LM] Text length: ${text.length}. Snippet:`,
        text.slice(0, 50).replace(/\n/g, ' '),
      );
    }
    return response;
  };
}

async function runLongcot(): Promise<void> {
  const question = exportOneQuestion('math', 'easy');
  console.log('Q:', question.question_id);

  const lm = new LM({
    model: 'fireworks/accounts/fireworks/routers/kimi-k2p5-turbo',
    apiKey: process.env.FIREWORKS_API_KEY,
    apiBase: process.env.FIREWORKS_API_BASE ?? 'https://api.fireworks.ai/inference/v1',
    kwargs: { max_tokens: 8192, stream: true },
  });

  settings.configure({ lm });
  const rlm = new RLM('prompt: str -> answer: str', {
    budget: { maxOracleCalls: 48, maxEffectTurns: 32, selfConsistencyN: 1 },
  });

  const t0 = Date.now();
  try {
    const pred = await settings.context({ adapter: new ChatAdapter() }, async () =>
      rlm.aforward({ prompt: question.prompt }),
    );
    console.log(`Finished in ${Date.now() - t0}ms`);
    const answer = String(pred.getOr('answer', '') ?? '');
    console.log('Final answer:', answer);
  } catch (e: unknown) {
    const err = e as { message?: string };
    console.error('Error/Exception:', err.message);
  }
}

async function runRlmClass(): Promise<void> {
  patchLmLogging();

  const lm = new LM({
    model: 'accounts/fireworks/routers/kimi-k2p5-turbo',
    apiKey: process.env.FIREWORKS_API_KEY,
    apiBase: process.env.FIREWORKS_API_BASE ?? 'https://api.fireworks.ai/inference/v1',
    kwargs: { max_tokens: 8192, stream: true },
  });

  const prompt = `Initial state: [[1, 2], [3]]
Goal state: [[1, 2, 3], []]
Number of blocks: 3
Number of stacks: 2

Find a sequence of moves that will transform the initial state into the goal state.
Format your solution as:
solution = [move0, move1, ..., movek].`;

  const rlm = new RLM('prompt: str -> answer: str', {
    subLm: lm,
    taskType: 'solve',
    budget: { maxOracleCalls: 10, maxEffectTurns: 5, selfConsistencyN: 1 },
  });

  const t0 = Date.now();
  try {
    const result = await rlm.aforward({ prompt });
    console.log(`Finished in ${Date.now() - t0}ms`);
    console.log('Final answer:', result);
  } catch (e: unknown) {
    const err = e as { message?: string };
    console.error('Error/Exception:', err.message);
  }
}

async function main(): Promise<void> {
  const mode = parseMode(process.argv);
  if (mode === 'rlm-class') {
    await runRlmClass();
  } else {
    await runLongcot();
  }
}

main().catch(console.error);
