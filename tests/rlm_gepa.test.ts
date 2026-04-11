import { readFileSync } from 'node:fs';
import { afterEach, describe, expect, it } from 'vitest';
import {
  BaseLM,
  Example,
  GEPA,
  Module,
  NodeCodeInterpreter,
  Predict,
  Prediction,
  ReplayLM,
  RLM,
  capturePredictorTraces,
  createGatedGEPAEngine,
  createModuleGEPAAdapter,
  createStaticGEPAEngine,
  getOptimizationArtifact,
  materializeReflectiveDataset,
  normalizeMetricRecord,
  settings,
  type EvaluationRow,
  type LMOutput,
  type Message,
  type OptimizationArtifact,
} from '../src/index.js';

class QueueLM extends BaseLM {
  readonly queue: LMOutput[];
  calls = 0;
  readonly requests: Array<{
    readonly prompt: string | undefined;
    readonly messages: readonly Message[] | undefined;
    readonly kwargs: Record<string, unknown>;
  }> = [];

  constructor(queue: readonly LMOutput[]) {
    super({ model: 'queue-lm' });
    this.queue = [...queue];
  }

  protected override generate(
    prompt?: string,
    messages?: readonly Message[],
    kwargs: Record<string, unknown> = {},
  ): readonly LMOutput[] {
    this.requests.push({
      prompt,
      messages,
      kwargs: { ...kwargs },
    });
    const next = this.queue[this.calls];
    this.calls += 1;
    if (next === undefined) {
      throw new Error('QueueLM exhausted scripted outputs.');
    }
    return [next];
  }
}

class EchoLM extends BaseLM {
  readonly prefix: string;
  readonly prompts: string[] = [];

  constructor(prefix = 'echo') {
    super({ model: 'echo-lm' });
    this.prefix = prefix;
  }

  protected override generate(
    prompt?: string,
    _messages?: readonly Message[],
    _kwargs?: Record<string, unknown>,
  ): readonly LMOutput[] {
    const text = prompt ?? '';
    this.prompts.push(text);
    return [`${this.prefix}:${text}`];
  }
}

class TwoPredictorProgram extends Module {
  readonly left = new Predict('question -> answer');
  readonly right = new Predict('question -> verdict');

  override forward(_kwargs: Record<string, unknown>): Prediction {
    return Prediction.create({ answer: 'ok', verdict: 'ok' });
  }
}

function action(reasoning: string, code: string): string {
  return JSON.stringify({ reasoning, code });
}

function extractAnswer(answer: string): string {
  return JSON.stringify({ answer });
}

interface SharedRLMReplayBudget {
  readonly max_iterations?: number;
  readonly max_llm_calls?: number;
  readonly max_batch_width?: number;
}

interface SharedRLMReplayCase {
  readonly id: string;
  readonly signature: string;
  readonly inputs: Record<string, unknown>;
  readonly budget?: SharedRLMReplayBudget;
  readonly typescript: {
    readonly lm_outputs: readonly Record<string, unknown>[];
  };
  readonly expected: {
    readonly answer: string;
    readonly typescript_final_via: 'submit' | 'extract';
    readonly typescript_history_kinds: readonly string[];
    readonly typescript_repl_error_kind?: string;
  };
}

const sharedRlmReplayFixture = JSON.parse(
  readFileSync(
    new URL('../../spec/fixtures/rlm_replay.json', import.meta.url),
    'utf-8',
  ),
) as { readonly cases: readonly SharedRLMReplayCase[] };

afterEach(() => {
  settings.reset();
});

describe('NodeCodeInterpreter', () => {
  it('keeps session globals across execute calls and supports submit', () => {
    const interpreter = new NodeCodeInterpreter();
    const session = interpreter.createSessionSync();

    session.patchGlobalsSync({
      bindings: {
        counter: 1,
      },
    });

    const first = session.executeSync({
      step: 1,
      source: 'counter = counter + 1;',
      budget: { maxIterations: 3, maxLlmCalls: 3, maxBatchWidth: 3 },
      allowSubmit: true,
    });
    expect(first.tag).toBe('continue');

    const second = session.executeSync({
      step: 2,
      source: 'SUBMIT({ answer: counter });',
      budget: { maxIterations: 3, maxLlmCalls: 3, maxBatchWidth: 3 },
      allowSubmit: true,
    });
    expect(second.tag).toBe('submit');
    if (second.tag === 'submit') {
      expect(second.output.value).toEqual({ answer: 2 });
      expect(second.output.via).toBe('submit');
    }

    const inspected = session.inspectGlobalsSync();
    const counter = inspected.find((entry) => entry.symbol === 'counter');
    expect(counter?.value).toBe(2);

    const snapshot = session.snapshotGlobalsSync();
    expect(snapshot.counter).toBe(2);

    session.closeSync();
    const closed = session.executeSync({
      step: 3,
      source: 'counter = 99;',
      budget: { maxIterations: 3, maxLlmCalls: 3, maxBatchWidth: 3 },
      allowSubmit: true,
    });
    expect(closed.tag).toBe('fault');
    if (closed.tag === 'fault') {
      expect(closed.error.kind).toBe('protocol');
    }
  });

  it('faults the session after runtime errors', () => {
    const interpreter = new NodeCodeInterpreter();
    const session = interpreter.createSessionSync();
    const fault = session.executeSync({
      step: 1,
      source: 'throw new Error("boom");',
      budget: { maxIterations: 3, maxLlmCalls: 3, maxBatchWidth: 3 },
      allowSubmit: true,
    });
    expect(fault.tag).toBe('fault');

    const afterFault = session.executeSync({
      step: 2,
      source: 'SUBMIT({ answer: "x" });',
      budget: { maxIterations: 3, maxLlmCalls: 3, maxBatchWidth: 3 },
      allowSubmit: true,
    });
    expect(afterFault.tag).toBe('fault');
    if (afterFault.tag === 'fault') {
      expect(afterFault.error.kind).toBe('protocol');
    }
  });

  it('returns same-step history and recovered lexical bindings on fault', () => {
    const interpreter = new NodeCodeInterpreter();
    const session = interpreter.createSessionSync();

    session.patchGlobalsSync({
      bindings: {
        llmQuery: (prompt: string) => `echo:${prompt}`,
      },
    });

    const fault = session.executeSync({
      step: 1,
      source: 'const note = llmQuery("alpha"); throw new Error("boom");',
      budget: { maxIterations: 3, maxLlmCalls: 3, maxBatchWidth: 3 },
      allowSubmit: true,
    });
    expect(fault.tag).toBe('fault');
    if (fault.tag === 'fault') {
      expect(fault.historyDelta.map((entry) => entry.kind)).toEqual(['query', 'fault']);
      expect(fault.liveVariables).toEqual(expect.arrayContaining([
        expect.objectContaining({
          symbol: 'note',
          value: 'echo:alpha',
          mutable: false,
        }),
      ]));
    }
  });
});

describe('RLM', () => {
  it('follows EXEC -> SUBMIT with typed output', () => {
    const controller = new QueueLM([
      action('Ready to finish.', 'SUBMIT({ answer: "done" });'),
    ]);
    settings.configure({ lm: controller });

    const rlm = new RLM('question -> answer', {
      subLm: new EchoLM('sub'),
    });
    const result = rlm.forward({ question: 'What now?' });

    expect(result.get('answer')).toBe('done');
    expect(result.get('final_via')).toBe('submit');
    const history = result.get('repl_history') as { entries: Array<{ kind: string }> };
    expect(history.entries.some((entry) => entry.kind === 'submit')).toBe(true);
  });

  it('follows QUERY -> STEP and resolves sub-LM via instance first', () => {
    const controller = new QueueLM([
      action('Query helper model first.', 'const note = llmQuery("alpha"); SUBMIT({ answer: note });'),
    ]);
    const subLm = new EchoLM('sub');
    settings.configure({ lm: controller });

    const rlm = new RLM('question -> answer', { subLm });
    const result = rlm.forward({ question: 'unused' });

    expect(result.get('answer')).toBe('sub:alpha');
    expect(controller.calls).toBe(1);
    expect(subLm.prompts).toEqual(['alpha']);
  });

  it('uses extract fallback when max iterations are exhausted', () => {
    const controller = new QueueLM([
      action('Inspect only.', 'const x = 1 + 1; print(x);'),
      extractAnswer('fallback'),
    ]);
    settings.configure({ lm: controller });

    const rlm = new RLM('question -> answer', {
      budget: { maxIterations: 1, maxLlmCalls: 5, maxBatchWidth: 3 },
      subLm: new EchoLM('sub'),
    });
    const result = rlm.forward({ question: 'fallback?' });

    expect(result.get('answer')).toBe('fallback');
    expect(result.get('final_via')).toBe('extract');
    const history = result.get('repl_history') as { entries: Array<{ kind: string }> };
    expect(history.entries.some((entry) => entry.kind === 'extract')).toBe(true);
  });

  it('records runtime faults and still returns extract output', () => {
    const controller = new QueueLM([
      action('This will fail.', 'throw new Error("boom");'),
      extractAnswer('recovered'),
    ]);
    settings.configure({ lm: controller });

    const rlm = new RLM('question -> answer', {
      subLm: new EchoLM('sub'),
    });
    const result = rlm.forward({ question: 'fault?' });

    expect(result.get('answer')).toBe('recovered');
    const fault = result.get('repl_error') as { kind: string } | null;
    expect(fault?.kind).toBe('runtime');
  });

  it('preserves fault-step query and variable evidence for extract fallback', () => {
    const controller = new QueueLM([
      action('Query before crashing.', 'const note = llmQuery("alpha"); throw new Error("boom");'),
      extractAnswer('recovered'),
    ]);
    const subLm = new EchoLM('sub');
    settings.configure({ lm: controller });

    const rlm = new RLM('question -> answer', { subLm });
    const result = rlm.forward({ question: 'fault?' });

    expect(result.get('answer')).toBe('recovered');
    expect(subLm.prompts).toEqual(['alpha']);

    const history = result.get('repl_history') as {
      entries: Array<{ kind: string }>;
      liveSymbols: string[];
    };
    expect(history.entries.map((entry) => entry.kind)).toEqual(['query', 'fault', 'extract']);
    expect(history.liveSymbols).toContain('note');

    const extractMessage = controller.requests[1]?.messages?.at(-1)?.content;
    expect(typeof extractMessage).toBe('string');
    expect(extractMessage as string).toContain('note (string, mutable=false)');
    expect(extractMessage as string).toContain('step=1 kind=query ok=true');
  });

  it('rejects reserved tool names', () => {
    expect(() => {
      new RLM('question -> answer', {
        tools: [
          function llmQuery() {
            return 'not allowed';
          },
        ],
      });
    }).toThrow('reserved');
  });

  for (const fixture of sharedRlmReplayFixture.cases) {
    it(`replays shared fixture ${fixture.id}`, () => {
      const budget = fixture.budget;
      const controller = new ReplayLM(fixture.typescript.lm_outputs.map((output) => JSON.stringify(output)));
      settings.configure({ lm: controller });

      const rlm = new RLM(fixture.signature, {
        budget: {
          maxIterations: budget?.max_iterations ?? 20,
          maxLlmCalls: budget?.max_llm_calls ?? 50,
          maxBatchWidth: budget?.max_batch_width ?? 8,
        },
      });
      const result = rlm.forward(fixture.inputs);

      expect(result.get('answer')).toBe(fixture.expected.answer);
      expect(result.get('final_via')).toBe(fixture.expected.typescript_final_via);

      const history = result.get('repl_history') as { entries: Array<{ kind: string }> };
      expect(history.entries.map((entry) => entry.kind)).toEqual([...fixture.expected.typescript_history_kinds]);

      const replError = result.get('repl_error') as { kind: string } | null;
      if (fixture.expected.typescript_repl_error_kind) {
        expect(replError?.kind).toBe(fixture.expected.typescript_repl_error_kind);
      } else {
        expect(replError).toBeNull();
      }

      expect(controller.exhausted).toBe(true);
    });
  }
});

describe('GEPA facade and trace substrate', () => {
  it('normalizes scores and builds reflective datasets', () => {
    const metric = normalizeMetricRecord({
      score: 0.75,
      subscores: [0.5, 1],
      feedback: 'ok',
    });
    expect(metric.score).toBe(0.75);
    expect(metric.subscores).toEqual([0.5, 1]);
    expect(metric.feedback).toBe('ok');
    expect(metric.failed).toBe(false);

    const program = new TwoPredictorProgram();
    const example = new Example({ question: 'q', answer: 'a' }).withInputs('question');
    const rows: readonly EvaluationRow[] = Object.freeze([
      Object.freeze([
        example,
        Prediction.create({ answer: 'a' }),
        { score: 1, subscores: [1, 0.5], feedback: 'great' },
      ] as const satisfies EvaluationRow),
    ]);
    const traces = capturePredictorTraces(program, rows);
    expect(traces).toHaveLength(2);

    const dataset = materializeReflectiveDataset(traces);
    expect(dataset).toHaveLength(2);
    expect(dataset[0]?.datumId).toBe(0);
  });

  it('gates compile when no approved engine is configured', async () => {
    const adapter = createModuleGEPAAdapter<TwoPredictorProgram, string>();
    const gepa = new GEPA<TwoPredictorProgram, string>({
      tracked: true,
      adapter,
      engine: createGatedGEPAEngine(),
    });

    await expect(gepa.compile(new TwoPredictorProgram(), [], [])).rejects.toThrow('gated');
  });

  it('attaches optimization artifacts with the available engine', async () => {
    const program = new TwoPredictorProgram();
    const adapter = createModuleGEPAAdapter<TwoPredictorProgram, string>();
    const projection = adapter.project(program);
    const firstTarget = projection.targets[0]!;

    const artifact: OptimizationArtifact<string> = Object.freeze({
      tracked: true,
      selectedCandidateId: 7,
      frontier: Object.freeze([
        Object.freeze({
          candidateId: 7,
          objective: Object.freeze([0.9, 0.8]),
          feasible: true,
        }),
      ]),
      instructionMap: Object.freeze([
        Object.freeze({
          candidateId: 7,
          target: firstTarget,
          instruction: 'Use explicit evidence and concise synthesis.',
        }),
      ]),
    });

    const gepa = new GEPA<TwoPredictorProgram, string>({
      tracked: true,
      adapter,
      engine: createStaticGEPAEngine(artifact),
    });

    const result = await gepa.compile(program, [], []);
    const attached = getOptimizationArtifact<string>(result.program);
    expect(attached?.selectedCandidateId).toBe(7);

    const firstPredictor = result.program.namedPredictors()[0]?.[1] as unknown as Predict;
    expect(firstPredictor.signature.instructions).toBe('Use explicit evidence and concise synthesis.');
    expect([...firstPredictor.signature.inputFields.keys()]).toEqual(['question']);

    firstPredictor.lm = new QueueLM(['{"answer": "compiled"}']);
    expect(firstPredictor.forward({ question: 'Does this still run?' }).toDict()).toEqual({
      answer: 'compiled',
    });

    const copied = result.program.deepcopy();
    expect(getOptimizationArtifact<string>(copied)?.selectedCandidateId).toBe(7);
  });
});
