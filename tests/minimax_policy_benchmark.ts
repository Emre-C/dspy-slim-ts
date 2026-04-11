import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  AdapterParseError,
  ChainOfThought,
  LM,
  Predict,
  settings,
  type GoldenTranscriptEntry,
  type HistoryEntry,
} from '../src/index.js';

const CURRENT_FILE = fileURLToPath(import.meta.url);
const TESTS_DIR = dirname(CURRENT_FILE);
const REPO_ROOT = resolve(TESTS_DIR, '..');
const WORKSPACE_ROOT = resolve(REPO_ROOT, '..');
const BENCHMARKS_ROOT = resolve(WORKSPACE_ROOT, 'benchmarks');
const GOLDEN_ROOT = resolve(BENCHMARKS_ROOT, 'golden');
const REGISTRY_PATH = resolve(BENCHMARKS_ROOT, 'registry.json');
const APPROVED_MODEL = 'openrouter/minimax/minimax-m2.7';
const DEFAULT_PER_DATASET = 8;
const DEFAULT_MAX_TOKENS = 8192;
const NUMBER_WORDS = new Map<string, string>([
  ['zero', '0'],
  ['one', '1'],
  ['two', '2'],
  ['three', '3'],
  ['four', '4'],
  ['five', '5'],
  ['six', '6'],
  ['seven', '7'],
  ['eight', '8'],
  ['nine', '9'],
  ['ten', '10'],
  ['eleven', '11'],
  ['twelve', '12'],
  ['thirteen', '13'],
  ['fourteen', '14'],
  ['fifteen', '15'],
  ['sixteen', '16'],
  ['seventeen', '17'],
  ['eighteen', '18'],
  ['nineteen', '19'],
  ['twenty', '20'],
]);

type ModuleType = 'cot' | 'predict';
type FailureKind = 'none' | 'empty' | 'malformed' | 'runtime';
type PolicyName = 'current-global-minimal' | 'hidden-default' | 'quality-first-fallback';

interface RegistryDatasetConfig {
  readonly input_fields: readonly string[];
  readonly signature: string;
  readonly module: string;
  readonly metric?: string;
}

interface RegistryPayload {
  readonly datasets: Record<string, RegistryDatasetConfig>;
}

interface BenchmarkCase {
  readonly datasetName: string;
  readonly exampleIndex: number;
  readonly moduleType: ModuleType;
  readonly signature: string;
  readonly inputs: Record<string, unknown>;
  readonly goldAnswers: readonly string[];
  readonly metric: string;
}

interface PolicyDefinition {
  readonly name: PolicyName;
  readonly description: string;
  readonly callConfig: Record<string, unknown>;
}

interface CaseResult {
  readonly policyName: PolicyName;
  readonly datasetName: string;
  readonly exampleIndex: number;
  readonly parseSuccess: boolean;
  readonly accurate: boolean;
  readonly fallbackUsed: boolean;
  readonly attempts: number;
  readonly failureKind: FailureKind;
  readonly answer: string | null;
  readonly errorMessage: string | null;
  readonly latencyMs: number;
  readonly promptTokens: number;
  readonly completionTokens: number;
  readonly reasoningTokens: number;
  readonly totalTokens: number;
}

interface AggregateSummary {
  readonly totalCases: number;
  readonly parseSuccesses: number;
  readonly accurateResponses: number;
  readonly fallbackCases: number;
  readonly emptyResponses: number;
  readonly malformedResponses: number;
  readonly runtimeFailures: number;
  readonly promptTokens: number;
  readonly completionTokens: number;
  readonly reasoningTokens: number;
  readonly totalTokens: number;
  readonly totalLatencyMs: number;
  readonly parseSuccessRate: number;
  readonly accuracyRate: number;
  readonly accuracyGivenParseRate: number;
  readonly fallbackRate: number;
  readonly emptyResponseRate: number;
  readonly malformedResponseRate: number;
  readonly averageLatencyMs: number;
  readonly averagePromptTokens: number;
  readonly averageCompletionTokens: number;
  readonly averageReasoningTokens: number;
  readonly averageTotalTokens: number;
}

interface BenchmarkReport {
  readonly generatedAt: string;
  readonly model: string;
  readonly perDataset: number;
  readonly maxTokens: number;
  readonly datasets: readonly string[];
  readonly policies: readonly PolicyDefinition[];
  readonly overall: Record<PolicyName, AggregateSummary>;
  readonly byDataset: Record<PolicyName, Record<string, AggregateSummary>>;
  readonly caseResults: readonly CaseResult[];
}

function loadDotenv(): void {
  let currentDir = dirname(CURRENT_FILE);
  for (;;) {
    const envPath = resolve(currentDir, '.env');
    if (existsSync(envPath)) {
      const raw = readFileSync(envPath, 'utf-8');
      for (const line of raw.split(/\r?\n/)) {
        const trimmed = line.trim();
        if (trimmed === '' || trimmed.startsWith('#') || !trimmed.includes('=')) {
          continue;
        }
        const [rawKey, ...rawValue] = trimmed.split('=');
        const key = rawKey?.trim();
        if (!key || process.env[key] !== undefined) {
          continue;
        }
        const value = rawValue.join('=').trim().replace(/^['"]|['"]$/g, '');
        process.env[key] = value;
      }
      return;
    }

    const parentDir = dirname(currentDir);
    if (parentDir === currentDir) {
      return;
    }
    currentDir = parentDir;
  }
}

function parseArgs(argv: readonly string[]): {
  readonly datasets: readonly string[];
  readonly perDataset: number;
  readonly maxTokens: number;
  readonly outputPath?: string;
} {
  let datasets: readonly string[] | null = null;
  let perDataset = DEFAULT_PER_DATASET;
  let maxTokens = DEFAULT_MAX_TOKENS;
  let outputPath: string | undefined;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case '--datasets': {
        const value = argv[index + 1];
        if (!value) {
          throw new Error('Missing value for --datasets');
        }
        datasets = value.split(',').map((item) => item.trim()).filter((item) => item !== '');
        index += 1;
        break;
      }
      case '--per-dataset': {
        const value = argv[index + 1];
        if (!value) {
          throw new Error('Missing value for --per-dataset');
        }
        perDataset = Number.parseInt(value, 10);
        if (!Number.isInteger(perDataset) || perDataset <= 0) {
          throw new Error(`Invalid --per-dataset value: ${value}`);
        }
        index += 1;
        break;
      }
      case '--max-tokens': {
        const value = argv[index + 1];
        if (!value) {
          throw new Error('Missing value for --max-tokens');
        }
        maxTokens = Number.parseInt(value, 10);
        if (!Number.isInteger(maxTokens) || maxTokens <= 0) {
          throw new Error(`Invalid --max-tokens value: ${value}`);
        }
        index += 1;
        break;
      }
      case '--output': {
        const value = argv[index + 1];
        if (!value) {
          throw new Error('Missing value for --output');
        }
        outputPath = resolve(REPO_ROOT, value);
        index += 1;
        break;
      }
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  const registry = loadRegistry();
  const datasetNames = datasets ?? Object.keys(registry.datasets);
  for (const datasetName of datasetNames) {
    if (!(datasetName in registry.datasets)) {
      throw new Error(`Unknown dataset: ${datasetName}`);
    }
  }

  return {
    datasets: datasetNames,
    perDataset,
    maxTokens,
    ...(outputPath ? { outputPath } : {}),
  };
}

function loadRegistry(): RegistryPayload {
  return JSON.parse(readFileSync(REGISTRY_PATH, 'utf-8')) as RegistryPayload;
}

function moduleType(moduleName: string): ModuleType {
  if (moduleName === 'ChainOfThought') {
    return 'cot';
  }
  if (moduleName === 'Predict') {
    return 'predict';
  }
  throw new Error(`Unsupported module type: ${moduleName}`);
}

function entryGoldAnswers(entry: GoldenTranscriptEntry): readonly string[] {
  const candidates = entry.gold_answers
    ?.map((value) => String(value))
    .filter((value) => value.length > 0);
  if (candidates && candidates.length > 0) {
    return candidates;
  }
  return [String(entry.gold_answer ?? '')];
}

function entryInputs(
  entry: GoldenTranscriptEntry,
  inputFields: readonly string[],
): Record<string, unknown> {
  if (entry.inputs) {
    return entry.inputs;
  }
  return Object.fromEntries(inputFields.map((name) => [name, 'test']));
}

function loadCases(datasetNames: readonly string[], perDataset: number): BenchmarkCase[] {
  const registry = loadRegistry();
  const cases: BenchmarkCase[] = [];

  for (const datasetName of datasetNames) {
    const config = registry.datasets[datasetName]!;
    const raw = readFileSync(resolve(GOLDEN_ROOT, `${datasetName}.json`), 'utf-8');
    const entries = JSON.parse(raw) as GoldenTranscriptEntry[];
    const selected = entries.slice(0, perDataset);
    for (const entry of selected) {
      cases.push({
        datasetName,
        exampleIndex: entry.example_idx,
        moduleType: moduleType(String(entry.module ?? config.module)),
        signature: String(entry.signature ?? config.signature),
        inputs: entryInputs(entry, config.input_fields),
        goldAnswers: entryGoldAnswers(entry),
        metric: String(config.metric ?? 'exact_match'),
      });
    }
  }

  return cases;
}

function extractLastNumeric(value: unknown): string {
  const matches = String(value).match(/-?\d[\d,]*\.?\d*/g);
  if (!matches || matches.length === 0) {
    return String(value);
  }
  return matches.at(-1)!.replaceAll(',', '');
}

function normalizeMetricText(value: unknown): string {
  const normalized = String(value)
    .normalize('NFD')
    .toLowerCase()
    .replace(/[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g, '')
    .replace(/\b(a|an|the)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return normalized
    .split(' ')
    .filter((token) => token.length > 0)
    .map((token) => NUMBER_WORDS.get(token) ?? token)
    .join(' ');
}

function answersMatch(actual: unknown, expected: unknown, metric: string): boolean {
  if (metric === 'exact_match_numeric') {
    return extractLastNumeric(actual) === extractLastNumeric(expected);
  }
  if (metric === 'exact_match_normalized') {
    return normalizeMetricText(actual) === normalizeMetricText(expected);
  }
  if (metric === 'exact_match_normalized_contains') {
    const normalizedActual = normalizeMetricText(actual);
    const normalizedExpected = normalizeMetricText(expected);
    return normalizedActual === normalizedExpected
      || normalizedActual.includes(normalizedExpected)
      || normalizedExpected.includes(normalizedActual);
  }
  return String(actual) === String(expected);
}

function extractOutputText(entry: HistoryEntry | undefined): string {
  const first = entry?.outputs[0];
  if (!first) {
    return '';
  }
  return typeof first === 'string' ? first : first.text;
}

function classifyFailure(firstAttempt: HistoryEntry | undefined, error: unknown): FailureKind {
  if (!(error instanceof Error) && !firstAttempt) {
    return 'runtime';
  }

  const text = extractOutputText(firstAttempt).trim();
  if (text === '') {
    return 'empty';
  }

  if (error instanceof AdapterParseError) {
    return error.completion.trim() === '' ? 'empty' : 'malformed';
  }

  if (error instanceof Error) {
    if (error.message.toLowerCase().includes('empty or null response')) {
      return 'empty';
    }
    if (error.message.toLowerCase().includes('parse') || error.message.toLowerCase().includes('expected fields')) {
      return 'malformed';
    }
  }

  return firstAttempt ? 'malformed' : 'runtime';
}

function sumUsage(entries: readonly HistoryEntry[], key: string): number {
  return entries.reduce((total, entry) => total + (entry.usage[key] ?? 0), 0);
}

function summarize(results: readonly CaseResult[]): AggregateSummary {
  const totalCases = results.length;
  const parseSuccesses = results.filter((result) => result.parseSuccess).length;
  const accurateResponses = results.filter((result) => result.accurate).length;
  const fallbackCases = results.filter((result) => result.fallbackUsed).length;
  const emptyResponses = results.filter((result) => result.failureKind === 'empty').length;
  const malformedResponses = results.filter((result) => result.failureKind === 'malformed').length;
  const runtimeFailures = results.filter((result) => result.failureKind === 'runtime').length;
  const promptTokens = results.reduce((total, result) => total + result.promptTokens, 0);
  const completionTokens = results.reduce((total, result) => total + result.completionTokens, 0);
  const reasoningTokens = results.reduce((total, result) => total + result.reasoningTokens, 0);
  const totalTokens = results.reduce((total, result) => total + result.totalTokens, 0);
  const totalLatencyMs = results.reduce((total, result) => total + result.latencyMs, 0);

  return {
    totalCases,
    parseSuccesses,
    accurateResponses,
    fallbackCases,
    emptyResponses,
    malformedResponses,
    runtimeFailures,
    promptTokens,
    completionTokens,
    reasoningTokens,
    totalTokens,
    totalLatencyMs,
    parseSuccessRate: totalCases === 0 ? 0 : parseSuccesses / totalCases,
    accuracyRate: totalCases === 0 ? 0 : accurateResponses / totalCases,
    accuracyGivenParseRate: parseSuccesses === 0 ? 0 : accurateResponses / parseSuccesses,
    fallbackRate: totalCases === 0 ? 0 : fallbackCases / totalCases,
    emptyResponseRate: totalCases === 0 ? 0 : emptyResponses / totalCases,
    malformedResponseRate: totalCases === 0 ? 0 : malformedResponses / totalCases,
    averageLatencyMs: totalCases === 0 ? 0 : totalLatencyMs / totalCases,
    averagePromptTokens: totalCases === 0 ? 0 : promptTokens / totalCases,
    averageCompletionTokens: totalCases === 0 ? 0 : completionTokens / totalCases,
    averageReasoningTokens: totalCases === 0 ? 0 : reasoningTokens / totalCases,
    averageTotalTokens: totalCases === 0 ? 0 : totalTokens / totalCases,
  };
}

function formatRate(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function policyDefinitions(maxTokens: number): readonly PolicyDefinition[] {
  return [
    {
      name: 'current-global-minimal',
      description: 'Force hidden minimal reasoning on every request',
      callConfig: {
        max_tokens: maxTokens,
        extra_body: {
          reasoning: {
            exclude: true,
            effort: 'minimal',
          },
        },
      },
    },
    {
      name: 'hidden-default',
      description: 'Hide reasoning without forcing an effort override',
      callConfig: {
        max_tokens: maxTokens,
        extra_body: {
          reasoning: {
            exclude: true,
          },
        },
      },
    },
    {
      name: 'quality-first-fallback',
      description: 'Use the runtime default and only fall back to minimal reasoning after a structured-output failure',
      callConfig: {
        max_tokens: maxTokens,
      },
    },
  ];
}

function runCase(lm: LM, policy: PolicyDefinition, benchmarkCase: BenchmarkCase): CaseResult {
  settings.reset();
  settings.configure({ lm });

  const historyStart = lm.history.length;
  const startedAt = Date.now();

  try {
    const program = benchmarkCase.moduleType === 'cot'
      ? new ChainOfThought(benchmarkCase.signature)
      : new Predict(benchmarkCase.signature);
    const prediction = program.forward({
      ...benchmarkCase.inputs,
      config: policy.callConfig,
    });
    const latencyMs = Date.now() - startedAt;
    const entries = lm.history.slice(historyStart);
    const answer = prediction.get('answer');
    return {
      policyName: policy.name,
      datasetName: benchmarkCase.datasetName,
      exampleIndex: benchmarkCase.exampleIndex,
      parseSuccess: true,
      accurate: benchmarkCase.goldAnswers.some((gold) => answersMatch(answer, gold, benchmarkCase.metric)),
      fallbackUsed: entries.length > 1,
      attempts: entries.length,
      failureKind: entries.length > 1 ? classifyFailure(entries[0], null) : 'none',
      answer: answer === undefined || answer === null ? null : String(answer),
      errorMessage: null,
      latencyMs,
      promptTokens: sumUsage(entries, 'prompt_tokens'),
      completionTokens: sumUsage(entries, 'completion_tokens'),
      reasoningTokens: sumUsage(entries, 'reasoning_tokens'),
      totalTokens: sumUsage(entries, 'total_tokens'),
    };
  } catch (error) {
    const latencyMs = Date.now() - startedAt;
    const entries = lm.history.slice(historyStart);
    return {
      policyName: policy.name,
      datasetName: benchmarkCase.datasetName,
      exampleIndex: benchmarkCase.exampleIndex,
      parseSuccess: false,
      accurate: false,
      fallbackUsed: entries.length > 1,
      attempts: Math.max(entries.length, 1),
      failureKind: classifyFailure(entries[0], error),
      answer: null,
      errorMessage: error instanceof Error ? error.message : String(error),
      latencyMs,
      promptTokens: sumUsage(entries, 'prompt_tokens'),
      completionTokens: sumUsage(entries, 'completion_tokens'),
      reasoningTokens: sumUsage(entries, 'reasoning_tokens'),
      totalTokens: sumUsage(entries, 'total_tokens'),
    };
  }
}

function writeReport(outputPath: string, report: BenchmarkReport): void {
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`, 'utf-8');
}

function printSummary(report: BenchmarkReport): void {
  console.log(`MiniMax policy benchmark (${report.model})`);
  console.log(`Datasets: ${report.datasets.join(', ')} | cases per dataset: ${report.perDataset} | max_tokens: ${report.maxTokens}`);
  console.log('');

  for (const policy of report.policies) {
    const overall = report.overall[policy.name];
    console.log(`${policy.name}: ${policy.description}`);
    console.log(
      `  accuracy=${formatRate(overall.accuracyRate)} `
      + `parse=${formatRate(overall.parseSuccessRate)} `
      + `fallback=${formatRate(overall.fallbackRate)} `
      + `empty=${formatRate(overall.emptyResponseRate)} `
      + `malformed=${formatRate(overall.malformedResponseRate)} `
      + `avg_latency=${overall.averageLatencyMs.toFixed(0)}ms `
      + `avg_reasoning_tokens=${overall.averageReasoningTokens.toFixed(1)} `
      + `avg_completion_tokens=${overall.averageCompletionTokens.toFixed(1)}`,
    );
    for (const datasetName of report.datasets) {
      const datasetSummary = report.byDataset[policy.name][datasetName]!;
      console.log(
        `    ${datasetName}: accuracy=${formatRate(datasetSummary.accuracyRate)} `
        + `parse=${formatRate(datasetSummary.parseSuccessRate)} `
        + `fallback=${formatRate(datasetSummary.fallbackRate)} `
        + `avg_reasoning_tokens=${datasetSummary.averageReasoningTokens.toFixed(1)} `
        + `avg_completion_tokens=${datasetSummary.averageCompletionTokens.toFixed(1)}`,
      );
    }
    console.log('');
  }
}

function main(): void {
  loadDotenv();
  if (!process.env.OPENROUTER_API_KEY) {
    throw new Error('OPENROUTER_API_KEY not set (check .env)');
  }

  const options = parseArgs(process.argv.slice(2));
  const cases = loadCases(options.datasets, options.perDataset);
  const policies = policyDefinitions(options.maxTokens);
  const caseResults: CaseResult[] = [];

  for (const policy of policies) {
    const lm = new LM(APPROVED_MODEL, {
      apiKey: process.env.OPENROUTER_API_KEY,
      numRetries: 3,
      kwargs: {
        temperature: 0,
      },
    });

    for (const benchmarkCase of cases) {
      caseResults.push(runCase(lm, policy, benchmarkCase));
    }
  }

  const overall = Object.fromEntries(
    policies.map((policy) => [
      policy.name,
      summarize(caseResults.filter((result) => result.policyName === policy.name)),
    ]),
  ) as Record<PolicyName, AggregateSummary>;

  const byDataset = Object.fromEntries(
    policies.map((policy) => [
      policy.name,
      Object.fromEntries(
        options.datasets.map((datasetName) => [
          datasetName,
          summarize(
            caseResults.filter((result) => (
              result.policyName === policy.name && result.datasetName === datasetName
            )),
          ),
        ]),
      ),
    ]),
  ) as Record<PolicyName, Record<string, AggregateSummary>>;

  const report: BenchmarkReport = {
    generatedAt: new Date().toISOString(),
    model: APPROVED_MODEL,
    perDataset: options.perDataset,
    maxTokens: options.maxTokens,
    datasets: options.datasets,
    policies,
    overall,
    byDataset,
    caseResults,
  };

  printSummary(report);
  if (options.outputPath) {
    writeReport(options.outputPath, report);
    console.log(`Wrote report to ${options.outputPath}`);
  }
}

main();
