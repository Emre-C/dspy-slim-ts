/**
 * Run LongCoT questions through RLM with an OpenAI-compatible LM (e.g. Hugging Face router).
 *
 * Prerequisites:
 *   cd tools/longcot && uv sync
 *
 * Usage:
 *   HF_TOKEN=... pnpm run bench:longcot -- --domain logic --difficulty easy --max 2
 *
 *   # Export + score only (no API calls; responses are empty — expect 0 accuracy)
 *   pnpm run bench:longcot -- --dry-run --max 1
 */

import { spawnSync } from 'node:child_process';
import { createWriteStream } from 'node:fs';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  LM,
  RLM,
  isTaskType,
  settings,
  type TaskType,
} from '../src/index.js';

const REPO_ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const LONGCOT_DIR = resolve(REPO_ROOT, 'tools', 'longcot');

interface LongCoTQuestion {
  readonly question_id: string;
  readonly domain: string;
  readonly difficulty: string;
  readonly prompt: string;
  readonly problem: Record<string, unknown> | null;
  readonly answer: unknown;
}

interface CliOptions {
  readonly domain: string;
  readonly difficulty: string;
  readonly max: number;
  readonly taskType: TaskType;
  readonly dryRun: boolean;
  readonly noFallbackScore: boolean;
  readonly model: string;
  readonly apiBase: string;
  readonly maxCompletionTokens: number;
  readonly outDir: string;
  readonly maxOracleCalls: number;
}

function parseArgs(argv: string[]): CliOptions {
  let domain = 'logic';
  let difficulty = 'easy';
  let max = 3;
  let taskType: TaskType = 'multi_hop';
  let dryRun = false;
  let noFallbackScore = false;
  let model = process.env.HF_MODEL ?? 'MiniMaxAI/MiniMax-M2.7:novita';
  let apiBase = process.env.HF_API_BASE ?? 'https://router.huggingface.co/v1';
  let maxCompletionTokens = Number(process.env.LONGCOT_MAX_COMPLETION_TOKENS ?? 16384);
  let outDir = resolve(REPO_ROOT, 'tools', 'longcot', 'runs');
  let maxOracleCalls = Number(process.env.LONGCOT_RLM_MAX_ORACLE_CALLS ?? 400);

  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i]!;
    if (a === '--domain' && argv[i + 1]) {
      domain = argv[++i]!;
    } else if (a === '--difficulty' && argv[i + 1]) {
      difficulty = argv[++i]!;
    } else if (a === '--max' && argv[i + 1]) {
      max = Math.max(0, Number(argv[++i]!));
    } else if (a === '--task-type' && argv[i + 1]) {
      const t = argv[++i]!;
      if (!isTaskType(t)) {
        throw new Error(
          `Invalid --task-type ${t}. Expected one of: search, classify, aggregate, pairwise, summarise, multi_hop, unknown`,
        );
      }
      taskType = t;
    } else if (a === '--dry-run') {
      dryRun = true;
    } else if (a === '--no-fallback-score') {
      noFallbackScore = true;
    } else if (a === '--model' && argv[i + 1]) {
      model = argv[++i]!;
    } else if (a === '--api-base' && argv[i + 1]) {
      apiBase = argv[++i]!;
    } else if (a === '--max-completion-tokens' && argv[i + 1]) {
      maxCompletionTokens = Math.max(1, Number(argv[++i]!));
    } else if (a === '--out-dir' && argv[i + 1]) {
      outDir = resolve(argv[++i]!);
    } else if (a === '--max-oracle-calls' && argv[i + 1]) {
      maxOracleCalls = Math.max(1, Number(argv[++i]!));
    } else if (a === '--help' || a === '-h') {
      console.log(`bench_longcot_rlm.ts

Environment:
  HF_TOKEN              Hugging Face API token (required unless --dry-run)
  HF_MODEL              Default: MiniMaxAI/MiniMax-M2.7:novita
  HF_API_BASE           Default: https://router.huggingface.co/v1
  LONGCOT_MAX_COMPLETION_TOKENS  Default: 16384
  LONGCOT_RLM_MAX_ORACLE_CALLS   Default: 400

Flags:
  --domain logic|cs|chemistry|chess|math
  --difficulty easy|medium|hard
  --max N
  --task-type search|classify|aggregate|pairwise|summarise|multi_hop|unknown
  --dry-run
  --no-fallback-score   Pass through to Python verify (no Gemini fallback)
  --model ...
  --api-base ...
  --max-completion-tokens N
  --out-dir PATH
  --max-oracle-calls N
`);
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${a}`);
    }
  }

  return {
    domain,
    difficulty,
    max,
    taskType,
    dryRun,
    noFallbackScore,
    model,
    apiBase,
    maxCompletionTokens,
    outDir,
    maxOracleCalls,
  };
}

function exportQuestions(opts: Pick<CliOptions, 'domain' | 'difficulty' | 'max'>): LongCoTQuestion[] {
  const result = spawnSync(
    'uv',
    [
      'run',
      'python',
      'export_questions.py',
      '--domain',
      opts.domain,
      '--difficulty',
      opts.difficulty,
      '--max',
      String(opts.max),
    ],
    {
      cwd: LONGCOT_DIR,
      encoding: 'utf-8',
      maxBuffer: 256 * 1024 * 1024,
    },
  );

  if (result.error) {
    throw new Error(
      `Failed to spawn uv: ${result.error.message}\nInstall uv and run: cd tools/longcot && uv sync`,
    );
  }

  if (result.status !== 0) {
    throw new Error(
      `export_questions.py failed (exit ${String(result.status)}):\n${result.stderr || result.stdout}`,
    );
  }

  const lines = result.stdout
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  return lines.map((line) => JSON.parse(line) as LongCoTQuestion);
}

async function main(): Promise<void> {
  const raw = process.argv.slice(2).filter((a) => a !== '--');
  const opts = parseArgs(raw);

  if (!opts.dryRun && !process.env.HF_TOKEN) {
    console.error('Missing HF_TOKEN. Set it for Hugging Face router access, or pass --dry-run.');
    process.exit(1);
  }

  mkdirSync(opts.outDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const responsesPath = resolve(
    opts.outDir,
    `longcot_rlm_${opts.domain}_${opts.difficulty}_${stamp}.jsonl`,
  );

  const questions = exportQuestions(opts);
  if (questions.length === 0) {
    console.error('No questions exported. Check --domain / --difficulty / LongCoT install.');
    process.exit(1);
  }

  console.error(`Exported ${String(questions.length)} question(s) → ${responsesPath}`);

  if (!opts.dryRun) {
    settings.configure({
      lm: new LM({
        model: opts.model,
        apiKey: process.env.HF_TOKEN,
        apiBase: opts.apiBase,
        kwargs: {
          max_completion_tokens: opts.maxCompletionTokens,
        },
      }),
    });
  }

  const rlm = new RLM('prompt: str -> answer: str', {
    taskType: opts.taskType,
    budget: {
      maxOracleCalls: opts.maxOracleCalls,
    },
  });

  const out = createWriteStream(responsesPath, { flags: 'w' });

  for (const q of questions) {
    const row: Record<string, unknown> = {
      question: q,
      response_text: '',
      error: null as string | null,
      latency_ms: 0,
    };

    if (opts.dryRun) {
      row.response_text = '';
      row.error = 'dry-run';
    } else {
      const t0 = Date.now();
      try {
        const pred = await rlm.aforward({ prompt: q.prompt });
        row.response_text = pred.getOr('answer', '');
        row.latency_ms = Date.now() - t0;
      } catch (e) {
        row.error = e instanceof Error ? e.message : String(e);
        row.latency_ms = Date.now() - t0;
      }
    }

    out.write(`${JSON.stringify(row)}\n`);
  }

  await new Promise<void>((resolvePromise, reject) => {
    out.end((err: Error | undefined) => (err ? reject(err) : resolvePromise()));
  });

  console.error(`Wrote responses. Scoring with LongCoT verify()...`);

  const scoreArgs = ['run', 'python', 'score_responses.py', responsesPath];
  if (opts.noFallbackScore) {
    scoreArgs.push('--no-fallback');
  }

  const score = spawnSync('uv', scoreArgs, {
    cwd: LONGCOT_DIR,
    encoding: 'utf-8',
    stdio: ['inherit', 'inherit', 'inherit'],
  });

  if (score.status !== 0) {
    process.exit(score.status ?? 1);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
