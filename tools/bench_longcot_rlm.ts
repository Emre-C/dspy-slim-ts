/**
 * Run LongCoT questions through RLM with an OpenAI-compatible LM (e.g. Hugging Face router).
 *
 * Prerequisites:
 *   cd tools/longcot && uv sync
 *
 * Usage:
 *   pnpm run bench:longcot -- --domain logic --difficulty easy --max 2
 *
 *   If `HF_TOKEN` is unset, the runner loads repo-root `.env` (same directory as
 *   `package.json`). Explicit environment variables always win.
 *
 *   # Export + score only (no API calls; responses are empty — expect 0 accuracy)
 *   pnpm run bench:longcot -- --dry-run --max 1
 *
 * Cost safety (use in order):
 *   1. --preflight     One tiny HF chat completion (~tens of tokens); no LongCoT / RLM.
 *   2. --smoke         One LongCoT question with hard caps (summarise, ≤48 oracle calls, ≤8k completion tokens).
 *   3. Larger runs      Require --i-accept-cost when --max > 20, completion tokens > 50k, or oracle cap > 500.
 */

import { spawnSync } from 'node:child_process';
import { createWriteStream, mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { LMOutput } from '../src/lm.js';
import {
  LM,
  Predict,
  RLM,
  isTaskType,
  settings,
  type TaskType,
} from '../src/index.js';

type BenchRunner = 'rlm' | 'predict';

const REPO_ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const LONGCOT_DIR = resolve(REPO_ROOT, 'tools', 'longcot');

/**
 * Minimal `.env` loader (no dependency on `dotenv`). Does not override existing
 * `process.env` entries.
 */
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
  /** Extra-aggressive caps for a single cheap end-to-end probe */
  readonly smoke: boolean;
  /**
   * `predict` — plain `prompt -> answer` (matches LongCoT free-text `solution = …`).
   * `rlm` — full RLM v2 (oracle leaves expect structured effect-oracle JSON from the LM).
   */
  readonly runner: BenchRunner;
  /** True when `--runner` was passed; `--smoke` only overrides runner when this is false. */
  readonly runnerFromCli: boolean;
}

/** Reasoning-heavy models (e.g. MiniMax on HF) may need room for `reasoning_content` plus answer text per oracle call. */
const SMOKE_MAX_COMPLETION_TOKENS = 8192;
const SMOKE_MAX_ORACLE_CALLS = 48;

function lmOutputText(out: LMOutput | undefined): string {
  if (out === undefined) {
    return '';
  }
  return typeof out === 'string' ? out : out.text;
}

function applySmokeCaps(opts: CliOptions): CliOptions {
  return {
    ...opts,
    max: 1,
    taskType: 'summarise',
    maxCompletionTokens: Math.min(opts.maxCompletionTokens, SMOKE_MAX_COMPLETION_TOKENS),
    maxOracleCalls: Math.min(opts.maxOracleCalls, SMOKE_MAX_ORACLE_CALLS),
    noFallbackScore: true,
    smoke: true,
    runner: opts.runnerFromCli ? opts.runner : 'predict',
    runnerFromCli: opts.runnerFromCli,
  };
}

function isExpensiveRun(opts: CliOptions): boolean {
  return (
    opts.max > 20 ||
    opts.maxCompletionTokens > 50_000 ||
    opts.maxOracleCalls > 500
  );
}

async function runPreflight(model: string, apiBase: string): Promise<void> {
  const hfToken = process.env.HF_TOKEN;
  if (!hfToken) {
    console.error(
      'Missing HF_TOKEN. Add it to .env in the repo root, export it, or pass --dry-run on the full bench.',
    );
    process.exit(1);
  }

  const lm = new LM({
    model,
    apiKey: hfToken,
    apiBase,
    kwargs: {
      // Reasoning models (e.g. MiniMax on HF) may spend the first chunk of the
      // budget in `reasoning_content`; keep this high enough for a visible answer.
      max_completion_tokens: 512,
    },
  });

  const t0 = Date.now();
  const outputs = await lm.acall(
    'Reply with exactly the single capital letter A and nothing else.',
    undefined,
    {},
  );
  const dt = Date.now() - t0;
  let text = lmOutputText(outputs[0]).trim();

  if (text.length === 0 && lm.history.length > 0) {
    const snap = lm.history[lm.history.length - 1]!;
    const raw = snap.response;
    console.error(
      'preflight: empty parsed text; last response (truncated, for debugging):',
      JSON.stringify(raw).slice(0, 1200),
    );
  }

  if (text.length === 0) {
    console.error(
      'preflight failed: empty completion. Check HF model id, HF_TOKEN, and that the router returns message.content as a string.',
    );
    process.exit(1);
  }

  console.error(
    `preflight OK in ${String(dt)}ms (completion length ${String(text.length)}, first 80 chars quoted below)`,
  );
  console.error(JSON.stringify(text.slice(0, 80)));
}

function parseArgs(argv: string[]): CliOptions {
  let domain = 'logic';
  let difficulty = 'easy';
  let max = 1;
  let taskType: TaskType = 'multi_hop';
  let dryRun = false;
  let noFallbackScore = false;
  let model = process.env.HF_MODEL ?? 'MiniMaxAI/MiniMax-M2.7:novita';
  let apiBase = process.env.HF_API_BASE ?? 'https://router.huggingface.co/v1';
  let maxCompletionTokens = Number(process.env.LONGCOT_MAX_COMPLETION_TOKENS ?? 16384);
  let outDir = resolve(REPO_ROOT, 'tools', 'longcot', 'runs');
  let maxOracleCalls = Number(process.env.LONGCOT_RLM_MAX_ORACLE_CALLS ?? 400);
  let smoke = false;
  let runner: BenchRunner = 'rlm';
  let runnerFromCli = false;

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
    } else if (a === '--smoke') {
      smoke = true;
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
    } else if (a === '--runner' && argv[i + 1]) {
      const r = argv[++i]!;
      if (r !== 'rlm' && r !== 'predict') {
        throw new Error(`Invalid --runner ${r}. Use "rlm" or "predict".`);
      }
      runner = r;
      runnerFromCli = true;
    } else if (a === '--help' || a === '-h') {
      console.log(`bench_longcot_rlm.ts

Environment:
  HF_TOKEN              Hugging Face API token (required unless --dry-run)
  HF_MODEL              Default: MiniMaxAI/MiniMax-M2.7:novita
  HF_API_BASE           Default: https://router.huggingface.co/v1
  LONGCOT_MAX_COMPLETION_TOKENS  Default: 16384
  LONGCOT_RLM_MAX_ORACLE_CALLS   Default: 400

Cost safety:
  --preflight           One tiny HF completion only (no LongCoT / RLM). Run this first.
  --smoke               One question with tight caps + --no-fallback-score. Defaults to --runner predict
                        (LongCoT wants free-text "solution = …"). Use --smoke --runner rlm only if the LM
                        reliably emits RLM effect-oracle JSON.
  --i-accept-cost       Required when --max > 20, completion tokens > 50k, or oracle cap > 500.

Flags:
  --runner rlm|predict  Default: rlm. Use predict for LongCoT (and for --smoke unless overridden).
  --domain logic|cs|chemistry|chess|math
  --difficulty easy|medium|hard
  --max N               Default: 1 (accidental full-benchmark protection)
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
    smoke,
    runner,
    runnerFromCli,
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
  loadRootEnvFile();

  const raw = process.argv.slice(2).filter((a) => a !== '--');
  const wantsPreflight = raw.includes('--preflight');
  const acceptsCost = raw.includes('--i-accept-cost');
  const argvForParse = raw.filter((a) => a !== '--preflight' && a !== '--i-accept-cost');

  let opts = parseArgs(argvForParse);
  if (opts.smoke) {
    opts = applySmokeCaps(opts);
    console.error(
      '[bench:longcot] --smoke: forcing max=1, task-type=summarise, ' +
        `max-oracle-calls<=${String(SMOKE_MAX_ORACLE_CALLS)}, ` +
        `max-completion-tokens<=${String(SMOKE_MAX_COMPLETION_TOKENS)}, --no-fallback-score, ` +
        `runner=${opts.runner}` +
        (opts.runnerFromCli ? ' (from --runner)' : ' (default predict for LongCoT text)'),
    );
  }

  if (wantsPreflight) {
    if (!process.env.HF_TOKEN) {
      console.error(
        'Missing HF_TOKEN. Add it to .env in the repo root or export it before --preflight.',
      );
      process.exit(1);
    }
    await runPreflight(opts.model, opts.apiBase);
    return;
  }

  if (!opts.dryRun && !process.env.HF_TOKEN) {
    console.error(
      'Missing HF_TOKEN. Add it to .env in the repo root, export it, or pass --dry-run.',
    );
    process.exit(1);
  }

  if (!opts.dryRun && isExpensiveRun(opts) && !acceptsCost) {
    console.error(
      '[bench:longcot] Refusing a potentially expensive run (large --max and/or high limits).\n' +
        '  Use --smoke or lower --max / limits first. To proceed anyway, add --i-accept-cost.',
    );
    process.exit(2);
  }

  mkdirSync(opts.outDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const responsesPath = resolve(
    opts.outDir,
    `longcot_${opts.runner}_${opts.domain}_${opts.difficulty}_${stamp}.jsonl`,
  );

  const questions = exportQuestions(opts);
  if (questions.length === 0) {
    console.error('No questions exported. Check --domain / --difficulty / LongCoT install.');
    process.exit(1);
  }

  console.error(`Exported ${String(questions.length)} question(s) → ${responsesPath}`);

  if (!opts.dryRun) {
    const hfToken = process.env.HF_TOKEN;
    if (!hfToken) {
      throw new Error('HF_TOKEN disappeared after loadRootEnvFile (internal error)');
    }
    settings.configure({
      lm: new LM({
        model: opts.model,
        apiKey: hfToken,
        apiBase: opts.apiBase,
        kwargs: {
          max_completion_tokens: opts.maxCompletionTokens,
        },
      }),
    });
  }

  const predictor =
    !opts.dryRun && opts.runner === 'predict'
      ? new Predict('prompt: str -> answer: str')
      : null;
  const rlm =
    !opts.dryRun && opts.runner === 'rlm'
      ? new RLM('prompt: str -> answer: str', {
          taskType: opts.taskType,
          budget: {
            maxOracleCalls: opts.maxOracleCalls,
          },
        })
      : null;

  console.error(`[bench:longcot] runner=${opts.runner}`);

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
        if (predictor !== null) {
          const pred = await predictor.aforward({ prompt: q.prompt });
          row.response_text = pred.getOr('answer', '');
        } else if (rlm !== null) {
          const pred = await rlm.aforward({ prompt: q.prompt });
          row.response_text = pred.getOr('answer', '');
        } else {
          row.error = 'internal: no predictor or rlm';
        }
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
