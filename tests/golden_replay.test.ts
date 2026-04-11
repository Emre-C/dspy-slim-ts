import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  ChainOfThought,
  Predict,
  ReAct,
  ReplayLM,
  settings,
  Tool,
  type GoldenTranscriptEntry,
} from '../src/index.js';

const GOLDEN_DIR = resolve(__dirname, '../../benchmarks/golden');
const REGISTRY_PATH = resolve(__dirname, '../../benchmarks/registry.json');
const REACT_REPLAY_PATHS = [
  resolve(__dirname, '../../spec/fixtures/react_replay.json'),
  resolve(__dirname, '../../spec/fixtures/react_recorded.json'),
] as const;
const GOLDEN_DATASET = process.env.GOLDEN_DATASET;
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

interface RegistryDatasetConfig {
  readonly input_fields: readonly string[];
  readonly signature: string;
  readonly module: string;
  readonly metric?: string;
}

interface GoldenFixtureCase {
  readonly fixtureName: string;
  readonly datasetName: string;
  readonly path: string;
  readonly transcripts: GoldenTranscriptEntry[];
  readonly moduleType: 'cot' | 'predict';
  readonly signature: string;
  readonly inputFields: readonly string[];
  readonly metric: string;
}

interface ReactReplayToolStep {
  readonly name: string;
  readonly args?: Record<string, unknown>;
  readonly observation?: string;
  readonly error_name?: string;
  readonly error_message?: string;
}

interface ReactReplayExpected {
  readonly answer: string;
  readonly reasoning: string;
  readonly trajectory_keys: readonly string[];
  readonly trajectory_subset: Record<string, unknown>;
  readonly observation_contains?: Record<string, readonly string[]>;
}

interface ReactReplayCase {
  readonly id: string;
  readonly fixture_name?: string;
  readonly signature: string;
  readonly inputs: Record<string, unknown>;
  readonly max_iters?: number;
  readonly tool_steps: readonly ReactReplayToolStep[];
  readonly lm_outputs: readonly Record<string, unknown>[];
  readonly expected: ReactReplayExpected;
}

function loadGolden(name: string): GoldenTranscriptEntry[] {
  const raw = readFileSync(resolve(GOLDEN_DIR, `${name}.json`), 'utf-8');
  return JSON.parse(raw);
}

function loadRegistry(): Record<string, RegistryDatasetConfig> {
  const raw = readFileSync(REGISTRY_PATH, 'utf-8');
  return JSON.parse(raw).datasets;
}

function loadReactReplayCases(): readonly ReactReplayCase[] {
  return REACT_REPLAY_PATHS.flatMap((path) => {
    if (!existsSync(path)) {
      return [];
    }
    const raw = readFileSync(path, 'utf-8');
    const fixtureName = path.split('/').at(-1)?.replace(/\.json$/, '') ?? 'react_replay';
    return (JSON.parse(raw) as { readonly cases: readonly ReactReplayCase[] }).cases.map((caseData) => ({
      ...caseData,
      fixture_name: fixtureName,
    }));
  });
}

function inferDatasetName(
  fixtureName: string,
  transcripts: readonly GoldenTranscriptEntry[],
  datasetNames: readonly string[],
): string {
  const dataset = transcripts[0]?.dataset;
  if (typeof dataset === 'string' && datasetNames.includes(dataset)) {
    return dataset;
  }

  for (const candidate of [...datasetNames].sort((a, b) => b.length - a.length)) {
    if (fixtureName === candidate || fixtureName.startsWith(`${candidate}_`)) {
      return candidate;
    }
  }

  throw new Error(`Could not infer dataset for golden transcript: ${fixtureName}.json`);
}

function moduleType(moduleName: string): 'cot' | 'predict' {
  if (moduleName === 'ChainOfThought') {
    return 'cot';
  }
  if (moduleName === 'Predict') {
    return 'predict';
  }
  throw new Error(`Unsupported module in registry: ${moduleName}`);
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

function entryExpectedAnswers(entry: GoldenTranscriptEntry): readonly string[] {
  const candidates = entry.gold_answers
    ?.map((value) => String(value))
    .filter((value) => value.length > 0);
  if (candidates && candidates.length > 0) {
    return candidates;
  }
  return [String(entry.gold_answer ?? '')];
}

function discoverGoldenCases(): GoldenFixtureCase[] {
  const registry = loadRegistry();
  if (GOLDEN_DATASET && !(GOLDEN_DATASET in registry)) {
    throw new Error(`Unknown GOLDEN_DATASET filter: ${GOLDEN_DATASET}`);
  }

  const datasetNames = Object.keys(registry);
  return readdirSync(GOLDEN_DIR)
    .filter((name) => name.endsWith('.json'))
    .sort()
    .map((fileName) => {
      const fixtureName = fileName.slice(0, -'.json'.length);
      const path = resolve(GOLDEN_DIR, fileName);
      const transcripts = loadGolden(fixtureName);
      if (transcripts.length === 0) {
        throw new Error(`Golden transcript is empty: ${fileName}`);
      }

      const datasetName = inferDatasetName(fixtureName, transcripts, datasetNames);
      const config = registry[datasetName]!;
      const moduleName = String(transcripts[0]?.module ?? config.module);
      const signature = String(transcripts[0]?.signature ?? config.signature);

      return {
        fixtureName,
        datasetName,
        path,
        transcripts,
        moduleType: moduleType(moduleName),
        signature,
        inputFields: config.input_fields,
          metric: config.metric ?? 'exact_match',
        } satisfies GoldenFixtureCase;
      })
    .filter((fixture) => !GOLDEN_DATASET || fixture.datasetName === GOLDEN_DATASET);
}

function entryInputs(
  entry: GoldenTranscriptEntry,
  inputFields: readonly string[],
): Record<string, unknown> {
  return entry.inputs ?? Object.fromEntries(inputFields.map((name) => [name, 'test']));
}

function payloadValidationEntry(): {
  readonly fixture: GoldenFixtureCase;
  readonly entry: GoldenTranscriptEntry;
} | null {
  for (const fixture of fixtures) {
    for (const entry of fixture.transcripts) {
      if (entry.inputs && (entry.prompt !== undefined && entry.prompt !== null || (entry.messages?.length ?? 0) > 0)) {
        return { fixture, entry };
      }
    }
  }
  return null;
}

const fixtures = discoverGoldenCases();
const cotFixtures = fixtures.filter((fixture) => fixture.moduleType === 'cot');
const predictFixtures = fixtures.filter((fixture) => fixture.moduleType === 'predict');
const reactReplayCases = loadReactReplayCases();

function createReplayTools(toolSteps: readonly ReactReplayToolStep[]): readonly Tool[] {
  const stepsByName = new Map<string, ReactReplayToolStep[]>();
  for (const step of toolSteps) {
    const existing = stepsByName.get(step.name) ?? [];
    existing.push(step);
    stepsByName.set(step.name, existing);
  }

  return [...stepsByName.entries()].map(([toolName, steps]) => {
    const argNames = new Set<string>();
    for (const step of steps) {
      for (const name of Object.keys(step.args ?? {})) {
        argNames.add(name);
      }
    }

    let callIndex = 0;
    return new Tool(({ ...args }: Record<string, unknown>) => {
      const step = steps[callIndex];
      callIndex += 1;
      expect(step).toBeDefined();
      expect(args).toEqual(step?.args ?? {});
      if (step?.error_name) {
        const error = new Error(step.error_message ?? 'scripted tool failure');
        error.name = step.error_name;
        throw error;
      }
      return step?.observation ?? `${toolName} observation`;
    }, {
      name: toolName,
      args: Object.fromEntries([...argNames].map((name) => [name, {}])),
    });
  });
}

afterEach(() => {
  settings.reset();
  vi.restoreAllMocks();
});

describe('Golden fixture discovery', () => {
  it('finds at least one matching fixture', () => {
    if (fixtures.length === 0) {
      throw new Error(`No golden fixtures discovered in ${GOLDEN_DIR} (filter=${String(GOLDEN_DATASET)})`);
    }
  });
});

describe('Golden ChainOfThought replay', () => {
  it('produces reasoning and answer for every discovered fixture', () => {
    if (cotFixtures.length === 0) {
      return;
    }

    for (const fixture of cotFixtures) {
      const lm = ReplayLM.fromTranscripts(fixture.transcripts);
      settings.configure({ lm });

      const cot = new ChainOfThought(fixture.signature);
      for (const entry of fixture.transcripts) {
        const result = cot.forward(entryInputs(entry, fixture.inputFields));
        expect(
          entryExpectedAnswers(entry).some((expected) => answersMatch(result.get('answer'), expected, fixture.metric)),
        ).toBe(true);
        expect(typeof result.get('reasoning')).toBe('string');
        expect((result.get('reasoning') as string).length).toBeGreaterThan(0);
      }

      expect(lm.exhausted).toBe(true);
    }
  });

  it('replays each transcript individually for every discovered fixture', () => {
    if (cotFixtures.length === 0) {
      return;
    }

    for (const fixture of cotFixtures) {
      for (let i = 0; i < fixture.transcripts.length; i += 1) {
        const entry = fixture.transcripts[i]!;
        const lm = ReplayLM.fromTranscripts([entry]);
        settings.configure({ lm });
        const cot = new ChainOfThought(fixture.signature);
        const result = cot.forward(entryInputs(entry, fixture.inputFields));
        expect(
          entryExpectedAnswers(entry).some((expected) => answersMatch(result.get('answer'), expected, fixture.metric)),
        ).toBe(true);
      }
    }
  });
});

describe('Golden Predict replay', () => {
  it('produces correct answers for every discovered fixture', () => {
    if (predictFixtures.length === 0) {
      return;
    }

    for (const fixture of predictFixtures) {
      const lm = ReplayLM.fromTranscripts(fixture.transcripts);
      settings.configure({ lm });

      const predict = new Predict(fixture.signature);
      for (const entry of fixture.transcripts) {
        const result = predict.call(entryInputs(entry, fixture.inputFields));
        expect(
          entryExpectedAnswers(entry).some((expected) => answersMatch(result.get('answer'), expected, fixture.metric)),
        ).toBe(true);
      }

      expect(lm.exhausted).toBe(true);
    }
  });

  it('does not include reasoning field in plain Predict', () => {
    if (predictFixtures.length === 0) {
      return;
    }

    for (const fixture of predictFixtures) {
      const entry = fixture.transcripts[0]!;
      const lm = ReplayLM.fromTranscripts([entry]);
      settings.configure({ lm });

      const predict = new Predict(fixture.signature);
      const result = predict.call(entryInputs(entry, fixture.inputFields));
      expect(result.toDict()).not.toHaveProperty('reasoning');
    }
  });
});

describe('Golden ReAct replay', () => {
  for (const fixture of reactReplayCases) {
    const fixtureLabel = `${fixture.fixture_name ?? 'react_replay'}:${fixture.id}`;

    it(`replays ${fixtureLabel}`, () => {
      const lm = new ReplayLM(fixture.lm_outputs.map((output) => JSON.stringify(output)));
      settings.configure({ lm });

      const react = new ReAct(
        fixture.signature,
        createReplayTools(fixture.tool_steps),
        fixture.max_iters ?? 20,
      );
      const result = react.forward(fixture.inputs);
      const trajectory = result.get('trajectory') as Record<string, unknown>;

      expect(result.get('answer')).toBe(fixture.expected.answer);
      expect(result.get('reasoning')).toBe(fixture.expected.reasoning);
      expect(Object.keys(trajectory)).toEqual([...fixture.expected.trajectory_keys]);

      for (const [key, value] of Object.entries(fixture.expected.trajectory_subset)) {
        expect(trajectory[key]).toEqual(value);
      }

      for (const [key, snippets] of Object.entries(fixture.expected.observation_contains ?? {})) {
        expect(typeof trajectory[key]).toBe('string');
        for (const snippet of snippets) {
          expect(trajectory[key]).toContain(snippet);
        }
      }

      expect(lm.exhausted).toBe(true);
    });
  }
});

describe('ReplayLM behavior', () => {
  it('throws when exhausted', () => {
    const lm = new ReplayLM(['{"answer": "x"}']);
    settings.configure({ lm });
    const predict = new Predict('q -> answer');
    predict.call({ q: 'first' });
    expect(() => predict.call({ q: 'second' })).toThrow(/exhausted/);
  });

  it('tracks remaining count', () => {
    const lm = new ReplayLM(['{"answer": "a"}', '{"answer": "b"}']);
    expect(lm.remaining).toBe(2);
    expect(lm.exhausted).toBe(false);
    settings.configure({ lm });
    new Predict('q -> answer').call({ q: 'first' });
    expect(lm.remaining).toBe(1);
    new Predict('q -> answer').call({ q: 'second' });
    expect(lm.remaining).toBe(0);
    expect(lm.exhausted).toBe(true);
  });

  it('loads from file', () => {
    const fixture = fixtures[0];
    if (!fixture) {
      throw new Error(`No golden fixtures discovered in ${GOLDEN_DIR}`);
    }

    const lm = ReplayLM.fromFile(fixture.path);
    expect(lm.remaining).toBe(fixture.transcripts.length);
  });

  it('throws on recorded payload mismatch', () => {
    const payloadEntry = payloadValidationEntry();
    if (!payloadEntry) {
      return;
    }

    const { fixture, entry } = payloadEntry;
    const lm = ReplayLM.fromTranscripts([entry]);
    settings.configure({ lm });

    const mismatchedInputs = Object.fromEntries(
      fixture.inputFields.map((name) => [name, 'not the recorded value']),
    );

    if (fixture.moduleType === 'cot') {
      expect(() => new ChainOfThought(fixture.signature).forward(mismatchedInputs)).toThrow(/payload mismatch/);
      return;
    }

    expect(() => new Predict(fixture.signature).call(mismatchedInputs)).toThrow(/payload mismatch/);
  });
});
