import { afterEach, describe, expect, it, vi } from 'vitest';
import { execFileSync } from 'node:child_process';
import {
  ContextWindowExceededError,
  LM,
  Module,
  Prediction,
  getGlobalHistory,
  resetGlobalHistory,
  settings,
  type Message,
} from '../src/index.js';

vi.mock('node:child_process', async () => {
  const actual = await vi.importActual<typeof import('node:child_process')>('node:child_process');
  return {
    ...actual,
    execFileSync: vi.fn(),
  };
});

class HistoryModule extends Module {
  override forward(): Prediction {
    return Prediction.create({ ok: true });
  }
}

afterEach(() => {
  settings.reset();
  resetGlobalHistory();
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

function requestBodyFromExecFileSyncCall(callIndex: number): Record<string, unknown> {
  const call = vi.mocked(execFileSync).mock.calls[callIndex];
  if (!call) {
    throw new Error(`Missing execFileSync mock call at index ${callIndex}.`);
  }
  const args = call[1];
  if (!Array.isArray(args)) {
    throw new Error('Expected mocked curl invocation arguments to be present.');
  }
  const bodyIndex = args.indexOf('-d');
  if (bodyIndex === -1) {
    throw new Error('Expected mocked curl invocation to include a -d request body flag.');
  }

  const body = args[bodyIndex + 1];
  if (typeof body !== 'string') {
    throw new Error('Expected mocked curl request body to be a string.');
  }

  return JSON.parse(body) as Record<string, unknown>;
}

describe('LM', () => {
  it('normalizes chat completions and updates global, LM, and module history', () => {
    vi.mocked(execFileSync)
      .mockReturnValueOnce(`${JSON.stringify({
        model: 'gpt-4.1-mini',
        choices: [
          {
            message: { content: 'hello' },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
      })}\n200`)
      .mockReturnValueOnce(`${JSON.stringify({
        model: 'gpt-4.1-mini',
        choices: [
          {
            message: {
              content: 'world',
              tool_calls: [
                {
                  id: 'call_1',
                  type: 'function',
                  function: { name: 'lookup', arguments: '{"city":"Paris"}' },
                },
              ],
            },
            finish_reason: 'stop',
            logprobs: { tokens: [] },
          },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
      })}\n200`);

    const lm = new LM('openai/gpt-4.1-mini', { apiKey: 'sk-test' });
    const module = new HistoryModule();
    settings.configure({ maxHistorySize: 1 });

    const first = settings.context({ callerModules: [module] }, () => lm.call(undefined, [
      { role: 'user', content: 'say hello' },
    ]));
    const second = settings.context({ callerModules: [module] }, () => lm.call(undefined, [
      { role: 'user', content: 'say world' },
    ], {
      temperature: 0.3,
      logprobs: true,
    }));

    expect(first).toEqual(['hello']);
    expect(second).toHaveLength(1);
    expect(typeof second[0]).not.toBe('string');
    expect((second[0] as { text: string }).text).toBe('world');

    expect(lm.history).toHaveLength(1);
    expect(module.history).toHaveLength(1);
    expect(getGlobalHistory()).toHaveLength(2);
    expect(lm.history[0]?.kwargs.temperature).toBe(0.3);

    const [, args] = vi.mocked(execFileSync).mock.calls[0]!;
    expect(args).toContain('https://api.openai.com/v1/chat/completions');
  });

  it('normalizes missing or null chat completion content to empty text', () => {
    vi.mocked(execFileSync)
      .mockReturnValueOnce(`${JSON.stringify({
        model: 'gpt-4.1-mini',
        choices: [
          {
            message: {},
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
      })}\n200`)
      .mockReturnValueOnce(`${JSON.stringify({
        model: 'gpt-4.1-mini',
        choices: [
          {
            message: { content: null },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
      })}\n200`);

    const lm = new LM('openai/gpt-4.1-mini', { apiKey: 'sk-test' });

    expect(lm.call(undefined, [{ role: 'user', content: 'first' }])).toEqual(['']);
    expect(lm.call(undefined, [{ role: 'user', content: 'second' }])).toEqual(['']);
  });

  it('records reasoning token usage from nested completion token details', () => {
    vi.mocked(execFileSync).mockReturnValueOnce(`${JSON.stringify({
      model: 'gpt-4.1-mini',
      choices: [
        {
          message: { content: 'hello' },
          finish_reason: 'stop',
        },
      ],
      usage: {
        prompt_tokens: 2,
        completion_tokens: 5,
        total_tokens: 7,
        completion_tokens_details: {
          reasoning_tokens: 3,
        },
      },
    })}\n200`);

    const lm = new LM('openai/gpt-4.1-mini', { apiKey: 'sk-test' });
    lm.call(undefined, [{ role: 'user', content: 'track usage' }]);

    expect(lm.history[0]?.usage.reasoning_tokens).toBe(3);
  });

  it('normalizes responses output and converts chat-style inputs for the responses endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({
        model: 'gpt-5',
        output: [
          {
            type: 'function_call',
            name: 'lookup',
            arguments: '{"city":"Paris"}',
            call_id: 'call_1',
          },
          {
            type: 'message',
            content: [
              { text: 'Use the lookup result.' },
            ],
          },
        ],
        usage: { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 },
      }),
    });

    const lm = new LM('openai/gpt-5', {
      apiKey: 'sk-test',
      modelType: 'responses',
      fetch: fetchMock,
    });

    const outputs = await lm.acall(undefined, [
      { role: 'system', content: 'Return structured output.' },
      { role: 'user', content: 'Find Paris.' },
    ], {
      response_format: { type: 'json_object' },
      max_tokens: 32,
    });

    expect(outputs).toHaveLength(1);
    expect(outputs[0]).toMatchObject({
      text: 'Use the lookup result.',
      toolCalls: [
        {
          id: 'call_1',
          type: 'function',
          function: { name: 'lookup', arguments: '{"city":"Paris"}' },
        },
      ],
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, { body: string }];
    const body = JSON.parse(init.body);
    expect(url).toBe('https://api.openai.com/v1/responses');
    expect(body.input).toEqual([
      { role: 'system', content: [{ type: 'input_text', text: 'Return structured output.' }] },
      { role: 'user', content: [{ type: 'input_text', text: 'Find Paris.' }] },
    ]);
    expect(body.text.format).toEqual({ type: 'json_object' });
    expect(body.max_output_tokens).toBe(32);
  });

  it('maps context-window failures to ContextWindowExceededError', () => {
    vi.mocked(execFileSync).mockImplementation(() => {
      throw new Error('maximum context length exceeded');
    });

    const lm = new LM('openai/gpt-4.1-mini', { apiKey: 'sk-test', numRetries: 0 });

    expect(() => lm.call(undefined, [{ role: 'user', content: 'too long' } as Message])).toThrow(
      ContextWindowExceededError,
    );
  });

  it('defaults OpenRouter Minimax calls to hidden reasoning without forcing effort', () => {
    vi.mocked(execFileSync).mockReturnValueOnce(`${JSON.stringify({
      model: 'minimax/minimax-m2.7',
      choices: [
        {
          message: { content: '{"answer":"The Port City"}' },
          finish_reason: 'stop',
        },
      ],
      usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
    })}\n200`);

    const lm = new LM('openrouter/minimax/minimax-m2.7', {
      apiKey: 'sk-test',
    });

    lm.call(undefined, [{ role: 'user', content: 'Answer as JSON.' }], { max_tokens: 2048 });

    const body = requestBodyFromExecFileSyncCall(0);
    expect(body.model).toBe('minimax/minimax-m2.7');
    expect(body.max_tokens).toBe(4096);
    expect(body.reasoning).toEqual({
      exclude: true,
    });
  });

  it('flattens extra_body and preserves explicit Minimax reasoning overrides', () => {
    vi.mocked(execFileSync).mockReturnValueOnce(`${JSON.stringify({
      model: 'minimax/minimax-m2.7',
      choices: [
        {
          message: { content: '{"answer":"The Port City"}' },
          finish_reason: 'stop',
        },
      ],
      usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
    })}\n200`);

    const lm = new LM('openrouter/minimax/minimax-m2.7', {
      apiKey: 'sk-test',
    });

    lm.call(undefined, [{ role: 'user', content: 'Answer as JSON.' }], {
      max_tokens: 2048,
      extra_body: {
        reasoning: {
          exclude: true,
          effort: 'high',
        },
        provider: {
          sort: 'latency',
        },
      },
    });

    const body = requestBodyFromExecFileSyncCall(0);
    expect(body.extra_body).toBeUndefined();
    expect(body.max_tokens).toBe(4096);
    expect(body.reasoning).toEqual({
      exclude: true,
      effort: 'high',
    });
    expect(body.provider).toEqual({
      sort: 'latency',
    });
  });
});
