import { describe, expect, it } from 'vitest';
import { Tool, ToolCalls, ValueError } from '../src/index.js';

describe('Tool', () => {
  it('infers a usable tool shape from a plain function and executes named args in order', () => {
    function lookup(city: string, unit: string) {
      return `${city}:${unit}`;
    }

    const tool = new Tool(lookup);

    expect(tool.name).toBe('lookup');
    expect(Object.keys(tool.args)).toEqual(['city', 'unit']);
    expect(tool.argTypes).toEqual({ city: 'custom', unit: 'custom' });
    expect(tool.call({ unit: 'C', city: 'Paris' })).toBe('Paris:C');
  });

  it('supports explicit arg metadata for record-style tools and async execution', async () => {
    const tool = new Tool(async ({ city, days }: { city: string; days: number }) => (
      `${city}:${days}`
    ), {
      name: 'forecast',
      desc: 'Retrieve a short forecast.',
      args: {
        city: { type: 'string' },
        days: { type: 'integer' },
      },
      argTypes: {
        city: 'str',
        days: 'int',
      },
    });

    await expect(tool.acall({ city: 'Paris', days: '3' })).resolves.toBe('Paris:3');
    expect(tool.formatAsOpenAIFunctionCall()).toEqual({
      type: 'function',
      function: {
        name: 'forecast',
        description: 'Retrieve a short forecast.',
        parameters: {
          type: 'object',
          properties: {
            city: { type: 'string' },
            days: { type: 'integer' },
          },
          required: ['city', 'days'],
        },
      },
    });
  });

  it('rejects unexpected arguments', () => {
    const tool = new Tool((query: string) => query);

    expect(() => tool.call({ query: 'x', limit: 2 })).toThrow(ValueError);
  });
});

describe('ToolCalls', () => {
  it('normalizes flexible wire shapes into immutable tool calls', () => {
    const toolCalls = ToolCalls.from({
      tool_calls: [
        { name: 'search', args: { query: 'weather' } },
        { name: 'finish', args: {} },
      ],
    });

    expect(toolCalls.toolCalls).toEqual([
      { name: 'search', args: { query: 'weather' } },
      { name: 'finish', args: {} },
    ]);
    expect(toolCalls.format()).toEqual({
      tool_calls: [
        {
          type: 'function',
          function: {
            name: 'search',
            arguments: { query: 'weather' },
          },
        },
        {
          type: 'function',
          function: {
            name: 'finish',
            arguments: {},
          },
        },
      ],
    });
  });
});
