/** GEPA facade and trace helpers (`gepa.ts`, `gepa_trace.ts`, `gepa_types.ts`). */

import { afterEach, describe, expect, it } from 'vitest';

import {
  BaseLM,
  Example,
  GEPA,
  Module,
  Predict,
  Prediction,
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

// ---------------------------------------------------------------------------
// Test LMs and modules (shared across the GEPA cases below)
// ---------------------------------------------------------------------------

class QueueLM extends BaseLM {
  readonly queue: LMOutput[];
  calls = 0;

  constructor(queue: readonly LMOutput[]) {
    super({ model: 'queue-lm' });
    this.queue = [...queue];
  }

  protected override generate(
    _prompt?: string,
    _messages?: readonly Message[],
    _kwargs: Record<string, unknown> = {},
  ): readonly LMOutput[] {
    const next = this.queue[this.calls];
    this.calls += 1;
    if (next === undefined) {
      throw new Error('QueueLM exhausted scripted outputs.');
    }
    return [next];
  }
}

class TwoPredictorProgram extends Module {
  readonly left = new Predict('question -> answer');
  readonly right = new Predict('question -> verdict');

  override forward(_kwargs: Record<string, unknown>): Prediction {
    return Prediction.create({ answer: 'ok', verdict: 'ok' });
  }
}

afterEach(() => {
  settings.reset();
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
