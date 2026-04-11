import { describe, expect, it } from 'vitest';
import {
  BaseLM,
  JSONAdapter,
  Prediction,
  Tool,
  ToolCalls,
  createField,
  createSignature,
  type LMOutput,
  type Message,
} from '../src/index.js';

class FunctionCallingLM extends BaseLM {
  readonly outputs: readonly LMOutput[];
  readonly calls: Array<{
    readonly messages: readonly Message[] | undefined;
    readonly kwargs: Record<string, unknown>;
  }> = [];

  constructor(outputs: readonly LMOutput[]) {
    super({ model: 'function-calling-lm' });
    this.outputs = outputs;
  }

  override get supportsFunctionCalling(): boolean {
    return true;
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

describe('native function calling', () => {
  it('routes Tool inputs through LM-native function calling and returns ToolCalls', () => {
    const signature = createSignature(
      new Map([
        ['question', createField({ kind: 'input', name: 'question' })],
        ['tools', createField({ kind: 'input', name: 'tools', typeTag: 'custom', isTypeUndefined: false })],
      ]),
      new Map([
        ['tool_calls', createField({ kind: 'output', name: 'tool_calls', typeTag: 'custom', isTypeUndefined: false })],
      ]),
    );

    const tool = new Tool((city: string) => city, {
      name: 'lookup',
      args: {
        city: { type: 'string' },
      },
      argTypes: {
        city: 'str',
      },
    });

    const lm = new FunctionCallingLM([
      {
        text: '',
        toolCalls: [
          {
            id: 'call_1',
            type: 'function',
            function: {
              name: 'lookup',
              arguments: '{"city":"Paris"}',
            },
          },
        ],
      },
    ]);

    const adapter = new JSONAdapter();
    const [result] = adapter.call(lm, {}, signature, [], {
      question: 'Where should we look?',
      tools: [tool],
    });

    expect(lm.calls).toHaveLength(1);
    expect(lm.calls[0]?.kwargs.tools).toEqual([
      {
        type: 'function',
        function: {
          name: 'lookup',
          description: '',
          parameters: {
            type: 'object',
            properties: {
              city: { type: 'string' },
            },
            required: ['city'],
          },
        },
      },
    ]);

    expect(String(lm.calls[0]?.messages?.at(-1)?.content)).not.toContain('tools');
    expect(result?.tool_calls).toBeInstanceOf(ToolCalls);
    expect((result?.tool_calls as ToolCalls).toolCalls).toEqual([
      { name: 'lookup', args: { city: 'Paris' } },
    ]);

    const prediction = Prediction.fromCompletions([result ?? {}], signature);
    expect(prediction.get('tool_calls')).toBeInstanceOf(ToolCalls);
    expect(prediction.toDict()).toEqual({
      tool_calls: {
        tool_calls: [
          { name: 'lookup', args: { city: 'Paris' } },
        ],
      },
    });
  });
});
