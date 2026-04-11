import { afterEach, describe, expect, it } from 'vitest';
import {
  BaseCallback,
  BaseLM,
  Evaluate,
  Example,
  JSONAdapter,
  Predict,
  Prediction,
  Tool,
  currentCallID,
  settings,
  type LMOutput,
  type Message,
} from '../src/index.js';

class CallbackLM extends BaseLM {
  readonly outputs: readonly LMOutput[];

  constructor(outputs: readonly LMOutput[], callbacks: readonly BaseCallback[] = []) {
    super({ model: 'callback-lm', callbacks });
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

class RecordingCallback extends BaseCallback {
  readonly events: Array<{ readonly name: string; readonly callID: string; readonly active: string | null }> = [];

  override onModuleStart(callID: string): void {
    this.events.push({ name: 'module:start', callID, active: currentCallID() });
  }

  override onModuleEnd(callID: string): void {
    this.events.push({ name: 'module:end', callID, active: currentCallID() });
  }

  override onEvaluateStart(callID: string): void {
    this.events.push({ name: 'evaluate:start', callID, active: currentCallID() });
  }

  override onEvaluateEnd(callID: string): void {
    this.events.push({ name: 'evaluate:end', callID, active: currentCallID() });
  }

  override onLmStart(callID: string): void {
    this.events.push({ name: 'lm:start', callID, active: currentCallID() });
  }

  override onLmEnd(callID: string): void {
    this.events.push({ name: 'lm:end', callID, active: currentCallID() });
  }

  override onAdapterFormatStart(callID: string): void {
    this.events.push({ name: 'adapter_format:start', callID, active: currentCallID() });
  }

  override onAdapterFormatEnd(callID: string): void {
    this.events.push({ name: 'adapter_format:end', callID, active: currentCallID() });
  }

  override onAdapterParseStart(callID: string): void {
    this.events.push({ name: 'adapter_parse:start', callID, active: currentCallID() });
  }

  override onAdapterParseEnd(callID: string): void {
    this.events.push({ name: 'adapter_parse:end', callID, active: currentCallID() });
  }

  override onToolStart(callID: string): void {
    this.events.push({ name: 'tool:start', callID, active: currentCallID() });
  }

  override onToolEnd(callID: string): void {
    this.events.push({ name: 'tool:end', callID, active: currentCallID() });
  }
}

afterEach(() => {
  settings.reset();
});

describe('callbacks', () => {
  it('dispatches global and instance callbacks across module, LM, and adapter boundaries', () => {
    const globalCallback = new RecordingCallback();
    const lmCallback = new RecordingCallback();

    settings.configure({
      callbacks: [globalCallback],
      adapter: new JSONAdapter(),
    });

    const predict = new Predict('question -> answer');
    predict.lm = new CallbackLM(['{"answer":"Paris"}'], [lmCallback]);

    const prediction = predict.call({ question: 'What is the capital of France?' });

    expect(prediction.toDict()).toEqual({ answer: 'Paris' });
    expect(globalCallback.events.map((event) => event.name)).toEqual([
      'module:start',
      'adapter_format:start',
      'adapter_format:end',
      'lm:start',
      'lm:end',
      'adapter_parse:start',
      'adapter_parse:end',
      'module:end',
    ]);

    const moduleCallID = globalCallback.events[0]!.callID;
    expect(globalCallback.events[1]?.active).toBe(moduleCallID);
    expect(globalCallback.events[3]?.active).toBe(moduleCallID);
    expect(lmCallback.events.map((event) => event.name)).toEqual(['lm:start', 'lm:end']);
  });

  it('dispatches tool callbacks for sync and async tool execution', async () => {
    const callback = new RecordingCallback();
    const syncTool = new Tool((query: string) => query, {
      callbacks: [callback],
    });
    const asyncTool = new Tool(async ({ query }: { query: string }) => query, {
      name: 'lookup',
      callbacks: [callback],
      args: { query: { type: 'string' } },
    });

    expect(syncTool.call({ query: 'alpha' })).toBe('alpha');
    await expect(asyncTool.acall({ query: 'beta' })).resolves.toBe('beta');

    expect(callback.events.map((event) => event.name)).toEqual([
      'tool:start',
      'tool:end',
      'tool:start',
      'tool:end',
    ]);
  });

  it('dispatches evaluate callbacks around evaluation execution', () => {
    const callback = new RecordingCallback();
    settings.configure({ callbacks: [callback] });

    const evaluate = new Evaluate({
      devset: [new Example({ question: 'What is 1+1?', answer: '2' }).withInputs('question')],
      metric: (example, prediction) => prediction.get('answer') === example.get('answer') ? 1 : 0,
      displayProgress: false,
    });
    const program = {
      call(): Prediction {
        return Prediction.create({ answer: '2' });
      },
    };

    const result = evaluate.call(program);

    expect(result.score).toBe(100);
    expect(callback.events.map((event) => event.name)).toEqual([
      'evaluate:start',
      'evaluate:end',
    ]);
  });
});
