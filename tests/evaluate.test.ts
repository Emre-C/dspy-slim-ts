import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  BaseCallback,
  BaseLM,
  Evaluate,
  EvaluationResult,
  Example,
  Predict,
  Prediction,
  currentCallID,
  settings,
  type LMOutput,
  type Message,
} from '../src/index.js';

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function newExample(question: string, answer: string): Example {
  return new Example({ question, answer }).withInputs('question');
}

function answerExactMatch(example: Example, prediction: Prediction): number {
  return prediction.get('answer') === example.get('answer') ? 1 : 0;
}

class QueueLM extends BaseLM {
  readonly outputs: LMOutput[];

  constructor(outputs: readonly LMOutput[]) {
    super({ model: 'evaluate-lm' });
    this.outputs = [...outputs];
  }

  protected override generate(
    _prompt?: string,
    _messages?: readonly Message[],
    _kwargs?: Record<string, unknown>,
  ): readonly LMOutput[] {
    const next = this.outputs.shift();
    if (next === undefined) {
      throw new Error('QueueLM ran out of outputs.');
    }

    return [next];
  }
}

afterEach(() => {
  settings.reset();
  vi.restoreAllMocks();
});

describe('Evaluate', () => {
  it('stores constructor configuration', () => {
    const devset = [newExample('What is 1+1?', '2')];
    const metric = vi.fn(answerExactMatch);
    const evaluate = new Evaluate({
      devset,
      metric,
      displayProgress: false,
    });

    expect(evaluate.devset).toBe(devset);
    expect(evaluate.metric).toBe(metric);
    expect(evaluate.numThreads).toBeNull();
    expect(evaluate.displayProgress).toBe(false);
  });

  it('evaluates a Predict program and returns aggregate score plus per-example rows', () => {
    settings.configure({
      lm: new QueueLM([
        '{"answer":"2"}',
        '{"answer":"4"}',
      ]),
    });
    const devset = [
      newExample('What is 1+1?', '2'),
      newExample('What is 2+2?', '4'),
    ];
    const program = new Predict('question -> answer');
    const evaluate = new Evaluate({
      devset,
      metric: answerExactMatch,
      displayProgress: false,
    });

    const result = evaluate.call(program);

    expect(result.score).toBe(100);
    expect(result.results).toHaveLength(2);
    expect(result.results[0]?.[1].toDict()).toEqual({ answer: '2' });
    expect(result.results[1]?.[1].toDict()).toEqual({ answer: '4' });
    expect(result.results.map((row) => row[2])).toEqual([1, 1]);
  });

  it('uses failureScore for failed examples while preserving row alignment', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const devset = [
      newExample('ok', 'OK'),
      newExample('boom', 'BOOM'),
    ];
    const program = {
      call(kwargs: Record<string, unknown>): Prediction {
        if (kwargs.question === 'boom') {
          throw new Error('boom');
        }
        return Prediction.create({ answer: String(kwargs.question).toUpperCase() });
      },
    };
    const evaluate = new Evaluate({
      devset,
      metric: answerExactMatch,
      displayProgress: false,
      failureScore: 0.25,
      maxErrors: 3,
    });

    const result = evaluate.call(program);

    expect(result.score).toBe(62.5);
    expect(result.results).toHaveLength(2);
    expect(result.results[0]?.[2]).toBe(1);
    expect(result.results[1]?.[1].toDict()).toEqual({});
    expect(result.results[1]?.[2]).toBe(0.25);
  });

  it('supports async programs and score-bearing Prediction metrics', async () => {
    let active = 0;
    let maxActive = 0;
    const devset = [
      newExample('a', 'A'),
      newExample('b', 'B'),
      newExample('c', 'C'),
    ];
    const program = {
      call(kwargs: Record<string, unknown>): Prediction {
        return Prediction.create({ answer: String(kwargs.question).toUpperCase() });
      },
      async acall(kwargs: Record<string, unknown>): Promise<Prediction> {
        active += 1;
        maxActive = Math.max(maxActive, active);
        await sleep(15);
        active -= 1;
        return Prediction.create({ answer: String(kwargs.question).toUpperCase() });
      },
    };
    const metric = async (example: Example, prediction: Prediction): Promise<Prediction> => {
      await sleep(1);
      return Prediction.create({
        score: prediction.get('answer') === example.get('answer') ? 1 : 0,
        feedback: `checked:${String(example.get('question'))}`,
      });
    };
    const evaluate = new Evaluate({
      devset,
      metric,
      displayProgress: false,
      numThreads: 1,
    });

    const result = await evaluate.acall(program);

    expect(result.score).toBe(100);
    expect(maxActive).toBe(1);
    expect(result.results).toHaveLength(3);
    expect(result.results.every((row) => row[2] instanceof Prediction)).toBe(true);
    expect((result.results[0]?.[2] as Prediction).toDict()).toEqual({
      score: 1,
      feedback: 'checked:a',
    });
  });

  it('throws when devset is empty', () => {
    const evaluate = new Evaluate({
      devset: [],
      metric: answerExactMatch,
    });
    const program = {
      call(): Prediction {
        return Prediction.create({ answer: 'x' });
      },
    };

    expect(() => evaluate.call(program)).toThrow('Evaluate requires a non-empty devset.');
  });

  it('throws when metric is missing', () => {
    const evaluate = new Evaluate({
      devset: [newExample('q', 'a')],
    });
    const program = {
      call(): Prediction {
        return Prediction.create({ answer: 'a' });
      },
    };

    expect(() => evaluate.call(program)).toThrow('Evaluate requires a metric.');
  });

  it('throws on deprecated returnOutputs option', () => {
    expect(() => new Evaluate({
      devset: [newExample('q', 'a')],
      returnOutputs: true,
    })).toThrow('`returnOutputs` is no longer supported');
  });

  it('extracts numeric scores from plain objects with a score property', () => {
    const devset = [
      newExample('q1', 'a1'),
      newExample('q2', 'a2'),
    ];
    const program = {
      call(kwargs: Record<string, unknown>): Prediction {
        return Prediction.create({ answer: String(kwargs.question).toUpperCase() });
      },
    };
    const metric = (_example: Example, _prediction: Prediction) => ({
      score: 0.5,
      reason: 'partial match',
    });
    const evaluate = new Evaluate({ devset, metric });

    const result = evaluate.call(program);

    expect(result.score).toBe(50);
    expect(result.results[0]?.[2]).toEqual({ score: 0.5, reason: 'partial match' });
  });

  it('produces a stable EvaluationResult via snapshot and toString', () => {
    const devset = [newExample('q', 'a')];
    const program = {
      call(): Prediction {
        return Prediction.create({ answer: 'a' });
      },
    };
    const evaluate = new Evaluate({ devset, metric: answerExactMatch });

    const result = evaluate.call(program);
    const snapshot = result.snapshot();

    expect(snapshot).toBeInstanceOf(EvaluationResult);
    expect(snapshot.score).toBe(result.score);
    expect(snapshot.results).toHaveLength(result.results.length);
    expect(result.toString()).toBe('EvaluationResult(score=100, results=<list of 1 results>)');
  });

  it('stores callbackMetadata and falls back from constructor to call site', () => {
    class MetadataCallback extends BaseCallback {
      captured: Record<string, unknown> | null = null;

      override onEvaluateStart(_callID: string, _instance: unknown, inputs: Record<string, unknown>): void {
        this.captured = inputs;
      }
    }

    const callback = new MetadataCallback();
    settings.configure({ callbacks: [callback] });

    const devset = [newExample('q', 'a')];
    const program = {
      call(): Prediction {
        return Prediction.create({ answer: 'a' });
      },
    };

    // Constructor metadata used as fallback
    const evaluate = new Evaluate({
      devset,
      metric: answerExactMatch,
      callbackMetadata: { source: 'constructor' },
    });
    evaluate.call(program);
    expect(evaluate.callbackMetadata).toEqual({ source: 'constructor' });
    expect((callback.captured as Record<string, unknown>).callbackMetadata).toEqual({ source: 'constructor' });

    // Call-site metadata overrides constructor metadata
    evaluate.call(program, { callbackMetadata: { source: 'call-site' } });
    expect((callback.captured as Record<string, unknown>).callbackMetadata).toEqual({ source: 'call-site' });
  });

  it('dispatches onEvaluateStart and onEvaluateEnd callbacks with correct call IDs', () => {
    const events: Array<{ name: string; callID: string; active: string | null }> = [];

    class RecordingCallback extends BaseCallback {
      override onEvaluateStart(callID: string): void {
        events.push({ name: 'start', callID, active: currentCallID() });
      }

      override onEvaluateEnd(callID: string): void {
        events.push({ name: 'end', callID, active: currentCallID() });
      }
    }

    settings.configure({ callbacks: [new RecordingCallback()] });
    const evaluate = new Evaluate({
      devset: [newExample('q', 'a')],
      metric: answerExactMatch,
    });
    const program = {
      call(): Prediction {
        return Prediction.create({ answer: 'a' });
      },
    };

    evaluate.call(program);

    expect(events).toHaveLength(2);
    expect(events[0]!.name).toBe('start');
    expect(events[1]!.name).toBe('end');
    expect(events[0]!.callID).toBe(events[1]!.callID);
    expect(typeof events[0]!.callID).toBe('string');
    expect(events[0]!.callID.length).toBeGreaterThan(0);
  });
});
