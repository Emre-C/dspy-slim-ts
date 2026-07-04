/**
 * LongCoT A/B: same questions, same LM — **single-shot (`LM.acall`)** vs **RLM v2**.
 *
 * Scores both JSONLs with `tools/longcot/score_responses.py` (same `verify()` as upstream
 * [LongCoT](https://github.com/LongHorizonReasoning/longcot)). Use this to measure whether
 * our RLM stack improves on one-shot completion on identical keys and model.
 *
 * Does **not** invoke Python `run_inference.py`; that baseline is a separate pipeline
 * (see `tools/longcot/README.md`).
 *
 * Usage:
 *   pnpm run bench:longcot:compare -- --domain logic --difficulty easy --max 3
 *   pnpm run bench:longcot:compare -- --dry-run --max 1
 */

import { spawnSync } from 'node:child_process';
import { createWriteStream, mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  ChatAdapter,
  LM,
  RLM,
  isTaskType,
  settings,
  type TaskType,
} from '../src/index.js';

import type { LMOutput } from '../src/lm.js';

type BenchLmBackend = 'together' | 'huggingface' | 'fireworks';

const REPO_ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const LONGCOT_DIR = resolve(REPO_ROOT, 'tools', 'longcot');

const TOGETHER_OPENAI_BASE = 'https://api.together.xyz/v1';
const DEFAULT_TOGETHER_MODEL = 'MiniMaxAI/MiniMax-M2.7';
const DEFAULT_HF_MODEL = 'MiniMaxAI/MiniMax-M2.7:together';
const FIREWORKS_OPENAI_BASE = 'https://api.fireworks.ai/inference/v1';
const DEFAULT_FIREWORKS_MODEL = 'accounts/fireworks/routers/kimi-k2p5-turbo';

interface LongCoTQuestion {
  readonly question_id: string;
  readonly domain: string;
  readonly difficulty: string;
  readonly prompt: string;
  readonly problem: Record<string, unknown> | null;
  readonly answer: unknown;
}

interface CompareCliOptions {
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
  readonly maxEffectTurns: number;
  readonly modelFromCli: boolean;
  readonly apiBaseFromCli: boolean;
  readonly taskTypeFromCli: boolean;
}

/** Domains that require `solve` routing — see bench_longcot_rlm.ts. */
const STATE_TRACKING_DOMAINS: ReadonlySet<string> = new Set([
  'logic',
  'chess',
  'cs',
]);

interface ScoreSummary {
  readonly total: number;
  readonly correct: number;
  readonly incorrect: number;
  readonly failed: number;
  readonly accuracy: number;
  readonly overall_accuracy: number;
}

function togetherApiKey(): string | undefined {
  return process.env.TOGETHER_API_KEY ?? process.env.TOGETHERAI_API_KEY;
}

function fireworksApiKey(): string | undefined {
  return process.env.FIREWORKS_API_KEY;
}

function fireworksOpenAiBaseUrl(): string {
  const raw = process.env.FIREWORKS_API_BASE ?? FIREWORKS_OPENAI_BASE;
  return raw.replace(/\/$/, '');
}

function resolveLongcotStream(backend: BenchLmBackend): boolean {
  const v = process.env.LONGCOT_STREAM?.trim().toLowerCase();
  if (v === '0' || v === 'false' || v === 'off') {
    return false;
  }
  if (v === '1' || v === 'true' || v === 'on') {
    return true;
  }
  return backend === 'huggingface' || backend === 'fireworks';
}

function resolveLmBackend(): BenchLmBackend {
  const override = process.env.LONGCOT_LM_BACKEND?.toLowerCase();
  if (override === 'huggingface' || override === 'hf') {
    return 'huggingface';
  }
  if (override === 'together') {
    return 'together';
  }
  if (override === 'fireworks' || override === 'fw') {
    return 'fireworks';
  }
  if (togetherApiKey()) {
    return 'together';
  }
  if (fireworksApiKey()) {
    return 'fireworks';
  }
  return 'huggingface';
}

function loadRootEnvFile(): void {
  const path = resolve(REPO_ROOT, '.env');
  let raw: string;
  try {
    raw = readFileSync(path, 'utf-8');
  } catch {
    return;
  }
  for (const line of raw.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.length === 0 || trimmed.startsWith('#')) {
      continue;
    }
    const eq = trimmed.indexOf('=');
    if (eq <= 0) {
      continue;
    }
    const key = trimmed.slice(0, eq).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
      continue;
    }
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (process.env[key] === undefined) {
      process.env[key] = value;
    }
  }
}

function lmOutputText(out: LMOutput | undefined): string {
  if (out === undefined) {
    return '';
  }
  return typeof out === 'string' ? out : out.text;
}

function parseCompareArgs(argv: string[]): CompareCliOptions {
  let domain = 'logic';
  let difficulty = 'easy';
  let max = 5;
  let taskType: TaskType = 'solve';
  let taskTypeFromCli = false;
  let dryRun = false;
  let noFallbackScore = false;
  let model = process.env.HF_MODEL ?? DEFAULT_HF_MODEL;
  let apiBase = process.env.HF_API_BASE ?? 'https://router.huggingface.co/v1';
  let modelFromCli = false;
  let apiBaseFromCli = false;
  let maxCompletionTokens = Number(process.env.LONGCOT_MAX_COMPLETION_TOKENS ?? 16384);
  let outDir = resolve(REPO_ROOT, 'tools', 'longcot', 'runs');
  let maxOracleCalls = Number(process.env.LONGCOT_RLM_MAX_ORACLE_CALLS ?? 400);
  let maxEffectTurns = Number(process.env.LONGCOT_RLM_MAX_EFFECT_TURNS ?? 64);

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
        throw new Error(`Invalid --task-type ${t}`);
      }
      taskType = t;
      taskTypeFromCli = true;
    } else if (a === '--max-effect-turns' && argv[i + 1]) {
      maxEffectTurns = Math.max(1, Number(argv[++i]!));
    } else if (a === '--dry-run') {
      dryRun = true;
    } else if (a === '--no-fallback-score') {
      noFallbackScore = true;
    } else if (a === '--model' && argv[i + 1]) {
      model = argv[++i]!;
      modelFromCli = true;
    } else if (a === '--api-base' && argv[i + 1]) {
      apiBase = argv[++i]!;
      apiBaseFromCli = true;
    } else if (a === '--max-completion-tokens' && argv[i + 1]) {
      maxCompletionTokens = Math.max(1, Number(argv[++i]!));
    } else if (a === '--out-dir' && argv[i + 1]) {
      outDir = resolve(argv[++i]!);
    } else if (a === '--max-oracle-calls' && argv[i + 1]) {
      maxOracleCalls = Math.max(1, Number(argv[++i]!));
    } else if (a === '--help' || a === '-h') {
      console.log(`compare_longcot_predict_rlm.ts

Runs the same LongCoT questions twice: baseline LM vs RLM v2, then scores both with verify().

Environment matches bench: LONGCOT_LM_BACKEND, keys, LONGCOT_STREAM, etc.

Flags:
  --domain, --difficulty, --max
  --task-type       RLM router (default multi_hop)
  --dry-run         No API; both JSONLs marked dry-run
  --no-fallback-score
  --model, --api-base
  --max-completion-tokens N
  --max-oracle-calls N
  --out-dir
  --i-accept-cost   Required when --max > 20 or high token/oracle limits (same gate as bench)
`);
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${a}`);
    }
  }

  // Domain-based auto-routing (mirrors bench_longcot_rlm.ts).
  if (!taskTypeFromCli && STATE_TRACKING_DOMAINS.has(domain)) {
    taskType = 'solve';
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
    maxEffectTurns,
    modelFromCli,
    apiBaseFromCli,
    taskTypeFromCli,
  };
}

function isExpensiveRun(opts: CompareCliOptions): boolean {
  return (
    opts.max > 20 ||
    opts.maxCompletionTokens > 50_000 ||
    opts.maxOracleCalls > 500
  );
}

function exportQuestions(
  opts: Pick<CompareCliOptions, 'domain' | 'difficulty' | 'max'>,
): LongCoTQuestion[] {
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
    throw new Error(`Failed to spawn uv: ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(
      `export_questions.py failed: ${result.stderr || result.stdout}`,
    );
  }

  const lines = result.stdout
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  return lines.map((line) => JSON.parse(line) as LongCoTQuestion);
}

function runScore(jsonlPath: string, noFallback: boolean): ScoreSummary {
  const args = ['run', 'python', 'score_responses.py', jsonlPath];
  if (noFallback) {
    args.push('--no-fallback');
  }
  const score = spawnSync('uv', args, {
    cwd: LONGCOT_DIR,
    encoding: 'utf-8',
    maxBuffer: 4 * 1024 * 1024,
  });
  if (score.status !== 0) {
    throw new Error(score.stderr || score.stdout || 'score_responses failed');
  }
  return JSON.parse(score.stdout.trim()) as ScoreSummary;
}

function writeJsonl(
  path: string,
  rows: ReadonlyArray<Record<string, unknown>>,
): Promise<void> {
  return new Promise((resolvePromise, reject) => {
    const out = createWriteStream(path, { flags: 'w' });
    for (const row of rows) {
      out.write(`${JSON.stringify(row)}\n`);
    }
    out.end((err: Error | undefined) => (err ? reject(err) : resolvePromise()));
  });
}

async function main(): Promise<void> {
  loadRootEnvFile();

  const raw = process.argv.slice(2).filter((a) => a !== '--');
  const acceptsCost = raw.includes('--i-accept-cost');
  const argv = raw.filter((a) => a !== '--i-accept-cost');

  let opts = parseCompareArgs(argv);

  const lmBackend = resolveLmBackend();
  if (lmBackend === 'together' && !opts.modelFromCli) {
    opts = { ...opts, model: process.env.TOGETHER_MODEL ?? DEFAULT_TOGETHER_MODEL };
  }
  if (lmBackend === 'fireworks' && !opts.modelFromCli) {
    opts = { ...opts, model: process.env.FIREWORKS_MODEL ?? DEFAULT_FIREWORKS_MODEL };
  }

  if (!opts.dryRun && lmBackend === 'together' && !togetherApiKey()) {
    console.error('Missing TOGETHER_API_KEY. Use --dry-run or set the key.');
    process.exit(1);
  }
  if (!opts.dryRun && lmBackend === 'fireworks' && !fireworksApiKey()) {
    console.error('Missing FIREWORKS_API_KEY.');
    process.exit(1);
  }
  if (!opts.dryRun && lmBackend === 'huggingface' && !process.env.HF_TOKEN) {
    console.error('Missing HF_TOKEN.');
    process.exit(1);
  }

  if (!opts.dryRun && isExpensiveRun(opts) && !acceptsCost) {
    console.error(
      '[compare:longcot] Potentially expensive (predict + RLM per question). Add --i-accept-cost to proceed, or lower --max / caps.',
    );
    process.exit(2);
  }

  const questions = exportQuestions(opts);
  if (questions.length === 0) {
    console.error('No questions exported.');
    process.exit(1);
  }

  mkdirSync(opts.outDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const predictPath = resolve(
    opts.outDir,
    `longcot_compare_predict_${opts.domain}_${opts.difficulty}_${stamp}.jsonl`,
  );
  const rlmPath = resolve(
    opts.outDir,
    `longcot_compare_rlm_${opts.domain}_${opts.difficulty}_${stamp}.jsonl`,
  );

  console.error(
    `[compare:longcot] ${String(questions.length)} question(s); predict → ${predictPath}`,
  );
  console.error(`[compare:longcot] RLM → ${rlmPath}`);
  console.error(`[compare:longcot] lmBackend=${lmBackend}`);

  const predictRows: Record<string, unknown>[] = [];
  const rlmRows: Record<string, unknown>[] = [];

  if (opts.dryRun) {
    for (const q of questions) {
      const base = {
        question: q,
        response_text: '',
        error: 'dry-run',
        latency_ms: 0,
      };
      predictRows.push({ ...base });
      rlmRows.push({ ...base });
    }
  } else {
    const fireworksBase = opts.apiBaseFromCli ? opts.apiBase : fireworksOpenAiBaseUrl();
    const lmOpts =
      lmBackend === 'together'
        ? {
            model: opts.model,
            apiKey: togetherApiKey()!,
            apiBase: TOGETHER_OPENAI_BASE,
          }
        : lmBackend === 'fireworks'
          ? {
              model: opts.model,
              apiKey: fireworksApiKey()!,
              apiBase: fireworksBase,
            }
          : {
              model: opts.model,
              apiKey: process.env.HF_TOKEN!,
              apiBase: opts.apiBase,
            };
    const useStream = resolveLongcotStream(lmBackend);
    settings.configure({
      lm: new LM({
        ...lmOpts,
        kwargs: {
          max_tokens: opts.maxCompletionTokens,
          ...(useStream ? { stream: true as const } : {}),
        },
      }),
    });

    const rlm = new RLM('prompt: str -> answer: str', {
      taskType: opts.taskType,
      budget: {
        maxOracleCalls: opts.maxOracleCalls,
        maxEffectTurns: opts.maxEffectTurns,
      },
    });

    const lm = settings.lm as LM;

    for (const q of questions) {
      let pText = '';
      let pErr: string | null = null;
      let pMs = 0;
      const t0 = Date.now();
      try {
        const out = await lm.acall(q.prompt);
        pText = lmOutputText(out[0]);
      } catch (e) {
        pErr = e instanceof Error ? e.message : String(e);
      }
      pMs = Date.now() - t0;
      predictRows.push({
        question: q,
        response_text: pText,
        error: pErr,
        latency_ms: pMs,
      });

      let rText = '';
      let rErr: string | null = null;
      let rMs = 0;
      let rTraceTurns = 0;
      const t1 = Date.now();
      try {
        const pred = await settings.context({ adapter: new ChatAdapter() }, async () =>
          rlm.aforward({ prompt: q.prompt }),
        );
        rText = String(pred.getOr('answer', '') ?? '');
        const trace = pred.getOr('_rlm_trace', []);
        if (Array.isArray(trace)) {
          rTraceTurns = trace.length;
        }
      } catch (e) {
        rErr = e instanceof Error ? e.message : String(e);
      }
      rMs = Date.now() - t1;
      rlmRows.push({
        question: q,
        response_text: rText,
        error: rErr,
        latency_ms: rMs,
        trace_turns: rTraceTurns,
      });
    }
  }

  await writeJsonl(predictPath, predictRows);
  await writeJsonl(rlmPath, rlmRows);

  console.error('[compare:longcot] Scoring with LongCoT verify()...');

  const predictScore = runScore(predictPath, opts.noFallbackScore);
  const rlmScore = runScore(rlmPath, opts.noFallbackScore);

  const out = {
    compare: 'predict_vs_rlm',
    lmBackend,
    predict_jsonl: predictPath,
    rlm_jsonl: rlmPath,
    predict: predictScore,
    rlm: rlmScore,
    delta_correct: rlmScore.correct - predictScore.correct,
    delta_accuracy_on_graded: (() => {
      const pGraded = predictScore.correct + predictScore.incorrect;
      const rGraded = rlmScore.correct + rlmScore.incorrect;
      const pAcc = pGraded ? predictScore.correct / pGraded : 0;
      const rAcc = rGraded ? rlmScore.correct / rGraded : 0;
      return rAcc - pAcc;
    })(),
  };

  console.log(JSON.stringify(out, null, 2));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
