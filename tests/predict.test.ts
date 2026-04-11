import { afterEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import {
  BaseLM,
  ChatAdapter,
  Predict,
  createField,
  createSignature,
  settings,
  signatureFromString,
  type LMOutput,
  type Message,
} from '../src/index.js';

interface SignatureFieldFixture {
  kind: 'input' | 'output';
  default?: unknown;
}

interface PredictPipelineFixtureCase {
  id: string;
  signature?: string;
  signature_fields?: Record<string, SignatureFieldFixture>;
  settings_lm?: 'settings_lm' | null;
  predict_lm?: 'predict_lm' | null;
  kwargs_lm?: 'kwargs_lm' | null;
  kwargs?: Record<string, unknown>;
  config?: Record<string, unknown>;
  predict_config?: Record<string, unknown>;
  call_config?: Record<string, unknown>;
  positional_args?: unknown[];
  expected_error?: string;
  expected_resolved_lm?: 'settings_lm' | 'predict_lm' | 'kwargs_lm';
  expected_config_temperature?: number | null;
  expected_kwargs_after_preprocess?: Record<string, unknown>;
  expected_warning?: string;
  expected_merged_config?: Record<string, unknown>;
}

class TestLM extends BaseLM {
  readonly label: string;
  readonly outputs: readonly LMOutput[];
  readonly calls: Array<{
    readonly messages: readonly Message[] | undefined;
    readonly kwargs: Record<string, unknown>;
  }> = [];

  constructor(
    label: string,
    outputs: readonly LMOutput[] = ['{"answer": "Paris"}'],
    kwargs: Record<string, unknown> = {},
  ) {
    super({ model: label, kwargs });
    this.label = label;
    this.outputs = outputs;
  }

  protected override generate(
    _prompt?: string,
    messages?: readonly Message[],
    kwargs: Record<string, unknown> = {},
  ): readonly LMOutput[] {
    this.calls.push({
      messages,
      kwargs: { ...kwargs },
    });

    return this.outputs;
  }
}

class SequenceLM extends BaseLM {
  readonly outputsByCall: readonly (readonly LMOutput[])[];
  readonly calls: Array<{
    readonly messages: readonly Message[] | undefined;
    readonly kwargs: Record<string, unknown>;
  }> = [];

  constructor(
    outputsByCall: readonly (readonly LMOutput[])[],
    model = 'openrouter/minimax/minimax-m2.7',
  ) {
    super({ model });
    this.outputsByCall = outputsByCall;
  }

  protected override generate(
    _prompt?: string,
    messages?: readonly Message[],
    kwargs: Record<string, unknown> = {},
  ): readonly LMOutput[] {
    this.calls.push({
      messages,
      kwargs: { ...kwargs },
    });

    const outputs = this.outputsByCall[this.calls.length - 1];
    if (!outputs) {
      throw new Error('SequenceLM ran out of scripted outputs');
    }

    return outputs;
  }
}

const fixture = JSON.parse(
  readFileSync(
    new URL('../../spec/fixtures/predict_pipeline.json', import.meta.url),
    'utf-8',
  ),
) as { cases: PredictPipelineFixtureCase[] };

function createFixtureSignature(c: PredictPipelineFixtureCase) {
  if (c.signature_fields) {
    const inputs = new Map<string, ReturnType<typeof createField>>();
    const outputs = new Map<string, ReturnType<typeof createField>>();

    for (const [name, field] of Object.entries(c.signature_fields)) {
      const created = createField({
        kind: field.kind,
        name,
        default: field.default,
      });

      if (field.kind === 'input') {
        inputs.set(name, created);
      } else {
        outputs.set(name, created);
      }
    }

    if (outputs.size === 0) {
      outputs.set('answer', createField({ kind: 'output', name: 'answer' }));
    }

    return createSignature(inputs, outputs);
  }

  return signatureFromString(c.signature ?? 'question -> answer');
}

afterEach(() => {
  settings.reset();
  vi.restoreAllMocks();
});

describe('Predict pipeline (spec fixtures)', () => {
  for (const c of fixture.cases) {
    it(c.id, () => {
      const signature = createFixtureSignature(c);
      const predict = new Predict(signature, c.predict_config ?? {});

      const settingsLm = c.settings_lm ? new TestLM(c.settings_lm) : null;
      const predictLm = c.predict_lm ? new TestLM(c.predict_lm) : null;
      const kwargsLm = c.kwargs_lm ? new TestLM(c.kwargs_lm) : null;

      if (settingsLm) {
        settings.configure({ lm: settingsLm });
      }

      if (predictLm) {
        predict.lm = predictLm;
      }

      if (c.expected_error) {
        const invoke = () => predict.forward(c.kwargs ?? {}, ...(c.positional_args ?? []));
        expect(invoke).toThrow(c.expected_error);
        return;
      }

      if (c.expected_resolved_lm) {
        const resolved = predict.resolveLm(kwargsLm ? { lm: kwargsLm } : {});
        expect((resolved as TestLM).label).toBe(c.expected_resolved_lm);
        return;
      }

      if (c.expected_merged_config) {
        expect(predict.resolveConfig({ config: c.call_config })).toEqual(c.expected_merged_config);
        return;
      }

      if (c.expected_config_temperature !== undefined) {
        predict.lm = new TestLM('fixture-lm');
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        const result = predict.preprocess({ question: 'Why?', config: c.config ?? {} });
        warn.mockRestore();
        expect(result.config.temperature ?? null).toBe(c.expected_config_temperature);
        return;
      }

      if (c.expected_kwargs_after_preprocess) {
        predict.lm = new TestLM('fixture-lm');
        const result = predict.preprocess(c.kwargs ?? {});
        expect(result.inputs).toEqual(c.expected_kwargs_after_preprocess);
        return;
      }

      if (c.expected_warning) {
        predict.lm = new TestLM('fixture-lm');
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        predict.preprocess(c.kwargs ?? {});
        expect(warn).toHaveBeenCalledWith(expect.stringContaining(c.expected_warning));
        return;
      }

      throw new Error(`Unhandled predict fixture case: ${c.id}`);
    });
  }
});

describe('Predict hardening', () => {
  it('runs the full forward pipeline through the default JSON adapter', () => {
    const lm = new TestLM('json-lm', ['{"answer": "Paris"}']);
    const predict = new Predict('question -> answer');
    predict.lm = lm;

    const prediction = predict.forward({ question: 'What is the capital of France?' });

    expect(prediction.toDict()).toEqual({ answer: 'Paris' });
    expect(lm.calls).toHaveLength(1);
    expect(lm.calls[0]?.messages?.at(-1)?.role).toBe('user');
    expect((lm.calls[0]?.messages?.at(-1)?.content as string)).toContain('question');
    expect(predict.traces).toHaveLength(1);
  });

  it('uses the configured adapter override instead of the default JSON adapter', () => {
    settings.configure({ adapter: new ChatAdapter() });

    const lm = new TestLM('chat-lm', ['[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]']);
    const predict = new Predict('question -> answer');
    predict.lm = lm;

    expect(predict.forward({ question: 'What is the capital of France?' }).toDict()).toEqual({
      answer: 'Paris',
    });
  });

  it('merges LM default kwargs with Predict config before invoking the LM', () => {
    const lm = new TestLM('json-lm', ['{"answer": "Paris"}'], {
      temperature: 0.2,
      max_tokens: 128,
    });
    const predict = new Predict('question -> answer', { temperature: 0.4 });
    predict.lm = lm;

    predict.forward({
      question: 'What is the capital of France?',
      config: { temperature: 0.9, n: 2 },
    });

    expect(lm.calls[0]?.kwargs).toEqual({
      temperature: 0.9,
      max_tokens: 128,
      n: 2,
    });
  });

  it('treats zero temperature as explicit when applying auto-adjust', () => {
    const predict = new Predict('question -> answer');
    predict.lm = new TestLM('fixture-lm', ['{"answer": "Paris"}'], { temperature: 0.9 });

    const result = predict.preprocess({
      question: 'Why?',
      config: { temperature: 0, n: 2 },
    });

    expect(result.config.temperature).toBe(0.7);
  });

  it('treats zero generations as explicit when applying auto-adjust', () => {
    const predict = new Predict('question -> answer');
    predict.lm = new TestLM('fixture-lm', ['{"answer": "Paris"}'], { n: 3 });

    const result = predict.preprocess({
      question: 'Why?',
      config: { temperature: null, n: 0 },
    });

    expect(result.config.temperature ?? null).toBeNull();
  });

  it('rejects reserved Predict control keys as input field names', () => {
    expect(() => new Predict('config -> answer')).toThrow('reserved for control overrides');

    const predict = new Predict('question -> answer');

    expect(() => predict.preprocess({ signature: 'lm -> answer' })).toThrow(
      'reserved for control overrides',
    );
  });

  it('retries OpenRouter Minimax structured-output failures with minimal hidden reasoning', () => {
    const lm = new SequenceLM([
      ['{}'],
      ['{"answer":"Paris"}'],
    ]);
    const predict = new Predict('question -> answer');
    predict.lm = lm;

    const result = predict.forward({ question: 'What is the capital of France?' });

    expect(result.toDict()).toEqual({ answer: 'Paris' });
    expect(lm.calls).toHaveLength(2);
    expect(lm.calls[0]?.kwargs.extra_body).toBeUndefined();
    expect(lm.calls[1]?.kwargs.extra_body).toEqual({
      reasoning: {
        exclude: true,
        effort: 'minimal',
      },
    });
  });

  it('does not override explicit MiniMax reasoning configuration during structured-output retries', () => {
    const lm = new SequenceLM([
      ['{}'],
    ]);
    const predict = new Predict('question -> answer');
    predict.lm = lm;

    expect(() => predict.forward({
      question: 'What is the capital of France?',
      config: {
        extra_body: {
          reasoning: {
            exclude: true,
            effort: 'high',
          },
        },
      },
    })).toThrow('Expected fields answer');
    expect(lm.calls).toHaveLength(1);
  });
});
