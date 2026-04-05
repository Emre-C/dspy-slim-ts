import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  BaseLM,
  ChainOfThought,
  settings,
  type LMOutput,
  type Message,
} from '../src/index.js';

class TestLM extends BaseLM {
  readonly outputs: readonly LMOutput[];

  constructor(outputs: readonly LMOutput[]) {
    super({ model: 'test-lm' });
    this.outputs = outputs;
  }

  protected override generate(
    _prompt?: string,
    _messages?: readonly Message[],
    _kwargs?: Record<string, unknown>,
  ): readonly LMOutput[] {
    return this.outputs;
  }
}

afterEach(() => {
  settings.reset();
  vi.restoreAllMocks();
});

describe('ChainOfThought', () => {
  it('prepends a reasoning output field ahead of the original outputs', () => {
    const chain = new ChainOfThought('question -> answer, confidence');

    expect([...chain.predict.signature.outputFields.keys()]).toEqual([
      'reasoning',
      'answer',
      'confidence',
    ]);
    expect(chain.predict.signature.outputFields.get('reasoning')?.typeTag).toBe('str');
    expect(chain.predict.signature.outputFields.get('reasoning')?.description).toBe('${reasoning}');
  });

  it('exposes the inner Predict instance through module traversal', () => {
    const chain = new ChainOfThought('question -> answer');

    expect(chain.namedPredictors().map(([name]) => name)).toEqual(['predict']);
  });

  it('delegates sync execution through Predict.call', () => {
    const chain = new ChainOfThought('question -> answer');
    chain.predict.lm = new TestLM(['{"reasoning":"Recall the known capital.","answer":"Paris"}']);

    const callSpy = vi.spyOn(chain.predict, 'call');
    const forwardSpy = vi.spyOn(chain.predict, 'forward');

    const prediction = chain.forward({ question: 'What is the capital of France?' });

    expect(callSpy).toHaveBeenCalledWith({ question: 'What is the capital of France?' });
    expect(forwardSpy).toHaveBeenCalledTimes(1);
    expect(prediction.toDict()).toEqual({
      reasoning: 'Recall the known capital.',
      answer: 'Paris',
    });
  });

  it('delegates async execution through Predict.acall', async () => {
    const chain = new ChainOfThought('question -> answer');
    chain.predict.lm = new TestLM(['{"reasoning":"Use the stored fact.","answer":"Paris"}']);

    const acallSpy = vi.spyOn(chain.predict, 'acall');
    const aforwardSpy = vi.spyOn(chain.predict, 'aforward');

    const prediction = await chain.aforward({ question: 'What is the capital of France?' });

    expect(acallSpy).toHaveBeenCalledWith({ question: 'What is the capital of France?' });
    expect(aforwardSpy).toHaveBeenCalledTimes(1);
    expect(prediction.toDict()).toEqual({
      reasoning: 'Use the stored fact.',
      answer: 'Paris',
    });
  });
});
