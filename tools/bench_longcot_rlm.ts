/**
 * Run LongCoT questions through RLM with an OpenAI-compatible LM (e.g. Hugging Face router).
 *
 * Prerequisites:
 *   cd tools/longcot && uv sync
 *
 * Usage:
 *   pnpm run bench:longcot -- --domain logic --difficulty easy --max 2
 *
 *   If `HF_TOKEN` / Together API keys are unset, the runner loads repo-root `.env`
 *   (same directory as `package.json`). Explicit environment variables always win.
 *
 *   When `TOGETHER_API_KEY` (or `TOGETHERAI_API_KEY`) is set, requests go **directly**
 *   to Together's OpenAI-compatible API — same contract as
 *   `new OpenAI({ apiKey, baseURL: "https://api.together.xyz/v1" })` — bypassing the
 *   Hugging Face router (helps avoid HF gateway 504s on long runs).
 *   Set `LONGCOT_LM_BACKEND=huggingface` to force HF even if a Together key exists.
 *   On the HF backend, `LM` uses **SSE streaming** by default (`stream: true`) so long
 *   generations are less likely to hit idle **504** timeouts; override with `LONGCOT_STREAM=0`.
 *
 *   **Fireworks (Fire Pass / serverless):** `FIREWORKS_API_KEY` with
 *   `LONGCOT_LM_BACKEND=fireworks` uses the OpenAI-compatible endpoint
 *   `https://api.fireworks.ai/inference/v1` (see https://docs.fireworks.ai/firepass).
 *   Default model: `accounts/fireworks/routers/kimi-k2p5-turbo` (Fire Pass Kimi K2.5 Turbo).
 *
 *   # Export + score only (no API calls; responses are empty — expect 0 accuracy)
 *   pnpm run bench:longcot -- --dry-run --max 1
 *
 * Cost safety (use in order):
 *   1. --preflight     One tiny chat completion (~tens of tokens); no LongCoT / RLM.
 *   2. --smoke         One LongCoT question with hard caps (summarise, ≤48 oracle calls, ≤8k completion tokens).
 *   3. Larger runs      Require --i-accept-cost when --max > 20, completion tokens > 50k, or oracle cap > 500.
 */

import { spawnSync } from 'node:child_process';
import { createWriteStream, mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { LMOutput } from '../src/lm.js';
import {
  ChatAdapter,
  LM,
  RLM,
  isTaskType,
  settings,
  type TaskType,
} from '../src/index.js';

type BenchRunner = 'rlm' | 'predict';
type BenchLmBackend = 'together' | 'huggingface' | 'fireworks';

const REPO_ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const LONGCOT_DIR = resolve(REPO_ROOT, 'tools', 'longcot');

/**
 * Together OpenAI-compatible baseURL (same as the official `openai` npm package and
 * Together's docs: Bearer token + `/v1/chat/completions`).
 */
const TOGETHER_OPENAI_BASE = 'https://api.together.xyz/v1';
const DEFAULT_TOGETHER_MODEL = 'MiniMaxAI/MiniMax-M2.7';
const DEFAULT_HF_MODEL = 'MiniMaxAI/MiniMax-M2.7:together';

/** Fireworks OpenAI-compatible API (chat completions). @see https://docs.fireworks.ai/firepass */
const FIREWORKS_OPENAI_BASE = 'https://api.fireworks.ai/inference/v1';
/** Fire Pass / docs default for Kimi K2.5 Turbo router */
const DEFAULT_FIREWORKS_MODEL = 'accounts/fireworks/routers/kimi-k2p5-turbo';

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

/**
 * Streaming keeps long generations alive through gateways that time out idle HTTP connections.
 * Default: on for Hugging Face router only; set LONGCOT_STREAM=1 for all backends or 0 to disable.
 */
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
  readonly maxEffectTurns: number;
  /** Extra-aggressive caps for a single cheap end-to-end probe */
  readonly smoke: boolean;
  /**
   * `predict` — raw `LM.acall(prompt)` (LongCoT free-text `solution = …`, not JSON).
   * `rlm` — full RLM v2 (uses `ChatAdapter` so final `answer` is natural language for `verify()`).
   */
  readonly runner: BenchRunner;
  /** True when `--runner` was passed; `--smoke` only overrides runner when this is false. */
  readonly runnerFromCli: boolean;
  /** True when `--model` was explicitly passed. */
  readonly modelFromCli: boolean;
  /** True when `--api-base` was explicitly passed (any backend). */
  readonly apiBaseFromCli: boolean;
  /** True when `--task-type` was explicitly passed (disables domain-based auto-selection). */
  readonly taskTypeFromCli: boolean;
}

/**
 * Domains whose LongCoT questions are sequential state-tracking puzzles
 * (BlocksWorld, Sudoku, Dungeon, chess move sequences, stateful CS
 * problems). These MUST route to the `solve` task type because every
 * chunking/fan-out plan (search / aggregate / summarise / multi_hop)
 * destroys sequential state by splitting the prompt.
 */
const STATE_TRACKING_DOMAINS: ReadonlySet<string> = new Set([
  'logic',
  'chess',
  'cs',
]);

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
    // Smoke inherits the already-resolved taskType (e.g. `solve` for
    // state-tracking domains) unless the caller pinned one explicitly.
    // This used to force `summarise`, which is catastrophic for
    // BlocksWorld-style puzzles; the bench no longer overrides it.
    taskType: opts.taskTypeFromCli ? opts.taskType : opts.taskType,
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

async function runPreflight(
  model: string,
  apiBase: string,
  apiKey: string,
  useStream: boolean,
): Promise<void> {
  const lm = new LM({
    model,
    apiKey,
    apiBase,
    kwargs: {
      // Reasoning models (e.g. MiniMax on HF) may spend the first chunk of the
      // budget in `reasoning_content`; keep this high enough for a visible answer.
      max_completion_tokens: 512,
      ...(useStream ? { stream: true as const } : {}),
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
      'preflight failed: empty completion. Check model id and API key; for MiniMax-style models ensure '
        + 'the provider returns non-empty message.content or reasoning fields (see src/lm.ts).',
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
          `Invalid --task-type ${t}. Expected one of: search, classify, aggregate, pairwise, summarise, multi_hop, solve, unknown`,
        );
      }
      taskType = t;
      taskTypeFromCli = true;
    } else if (a === '--max-effect-turns' && argv[i + 1]) {
      maxEffectTurns = Math.max(1, Number(argv[++i]!));
    } else if (a === '--dry-run') {
      dryRun = true;
    } else if (a === '--smoke') {
      smoke = true;
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
  HF_TOKEN              Hugging Face API token (required for HF backend unless --dry-run)
  TOGETHER_API_KEY      Together API key (OpenAI-compat; if set, bench uses baseURL ${TOGETHER_OPENAI_BASE})
                        Also accepts TOGETHERAI_API_KEY. Override with LONGCOT_LM_BACKEND=huggingface.
  TOGETHER_MODEL        When using Together backend: default model id (${DEFAULT_TOGETHER_MODEL})
  FIREWORKS_API_KEY     Fireworks API key — use with LONGCOT_LM_BACKEND=fireworks (OpenAI base ${FIREWORKS_OPENAI_BASE})
  FIREWORKS_MODEL       Default when using Fireworks backend: ${DEFAULT_FIREWORKS_MODEL}
  FIREWORKS_API_BASE    Override Fireworks OpenAI base (default: ${FIREWORKS_OPENAI_BASE})
  LONGCOT_LM_BACKEND    together | huggingface | fireworks (aliases: hf, fw)
                        Default: together if TOGETHER_API_KEY set; else fireworks if FIREWORKS_API_KEY set; else huggingface
  HF_MODEL              Default on HF backend: ${DEFAULT_HF_MODEL}
  HF_API_BASE           Default: https://router.huggingface.co/v1
  LONGCOT_MAX_COMPLETION_TOKENS  Default: 16384
  LONGCOT_RLM_MAX_ORACLE_CALLS   Default: 400
  LONGCOT_STREAM        1|true|on = force SSE streaming for all backends; 0|false|off = disable.
                        When unset: streaming is enabled only for huggingface (mitigates router 504).

Cost safety:
  --preflight           One tiny chat completion only (provider per LONGCOT_LM_BACKEND / keys).
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

  // Domain-based auto-routing: state-tracking puzzle domains must NOT
  // use chunking plans; route them to `solve` unless the caller pinned
  // `--task-type` explicitly.
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
    smoke,
    runner,
    runnerFromCli,
    modelFromCli,
    apiBaseFromCli,
    taskTypeFromCli,
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

  const lmBackend = resolveLmBackend();
  if (lmBackend === 'together' && !opts.modelFromCli) {
    opts = {
      ...opts,
      model: process.env.TOGETHER_MODEL ?? DEFAULT_TOGETHER_MODEL,
    };
  }

  if (lmBackend === 'fireworks' && !opts.modelFromCli) {
    opts = {
      ...opts,
      model: process.env.FIREWORKS_MODEL ?? DEFAULT_FIREWORKS_MODEL,
    };
  }

  if (wantsPreflight) {
    const useStream = resolveLongcotStream(lmBackend);
    if (lmBackend === 'together') {
      const key = togetherApiKey();
      if (!key) {
        console.error(
          'Missing TOGETHER_API_KEY (or TOGETHERAI_API_KEY). Add it to .env or export it before --preflight.',
        );
        process.exit(1);
      }
      await runPreflight(opts.model, TOGETHER_OPENAI_BASE, key, useStream);
    } else if (lmBackend === 'fireworks') {
      const key = fireworksApiKey();
      if (!key) {
        console.error(
          'Missing FIREWORKS_API_KEY. Add it to .env or export it before --preflight.',
        );
        process.exit(1);
      }
      const base = opts.apiBaseFromCli ? opts.apiBase : fireworksOpenAiBaseUrl();
      await runPreflight(opts.model, base, key, useStream);
    } else {
      if (!process.env.HF_TOKEN) {
        console.error(
          'Missing HF_TOKEN. Add it to .env in the repo root or export it before --preflight.',
        );
        process.exit(1);
      }
      await runPreflight(opts.model, opts.apiBase, process.env.HF_TOKEN, useStream);
    }
    return;
  }

  if (!opts.dryRun && lmBackend === 'together' && !togetherApiKey()) {
    console.error(
      'Missing TOGETHER_API_KEY (or TOGETHERAI_API_KEY). Add it to .env, export it, or pass --dry-run.',
    );
    process.exit(1);
  }

  if (!opts.dryRun && lmBackend === 'fireworks' && !fireworksApiKey()) {
    console.error(
      'Missing FIREWORKS_API_KEY. Add it to .env, export it, or pass --dry-run.',
    );
    process.exit(1);
  }

  if (!opts.dryRun && lmBackend === 'huggingface' && !process.env.HF_TOKEN) {
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
    console.error(
      `[bench:longcot] lmBackend=${lmBackend} stream=${useStream ? 'on' : 'off'} (LONGCOT_STREAM)`,
    );
  }

  const rlm =
    !opts.dryRun && opts.runner === 'rlm'
      ? new RLM('prompt: str -> answer: str', {
          taskType: opts.taskType,
          budget: {
            maxOracleCalls: opts.maxOracleCalls,
            maxEffectTurns: opts.maxEffectTurns,
          },
        })
      : null;

  console.error(
    `[bench:longcot] runner=${opts.runner}` +
      (opts.runner === 'rlm' && !opts.dryRun ? ' (RLM uses ChatAdapter during aforward for verify()-friendly text)' : ''),
  );

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
        if (opts.runner === 'predict') {
          const lm = settings.lm;
          if (lm === null) {
            row.error = 'internal: settings.lm is not configured';
          } else {
            const outputs = await (lm as LM).acall(q.prompt);
            row.response_text = lmOutputText(outputs[0]);
          }
        } else if (rlm !== null) {
          const rlmT0 = Date.now();
          console.log(`[LM] Starting RLM inference for ${q.question_id}...`);
          try {
            const pred = await settings.context({ adapter: new ChatAdapter() }, async () =>
              rlm.aforward({ prompt: q.prompt }),
            );
            console.log(`[LM] RLM inference finished in ${Date.now() - rlmT0}ms.`);
            row.response_text = String(pred.getOr('answer', '') ?? '');
          } catch (e: any) {
            console.error(`[LM] RLM inference failed after ${Date.now() - rlmT0}ms: ${e.message}`);
            row.error = e.message;
          }
        } else {
          row.error = 'internal: no RLM (unexpected runner)';
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
