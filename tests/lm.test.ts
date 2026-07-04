import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  ContextWindowExceededError,
  LM,
  Module,
  Prediction,
  RuntimeError,
  getGlobalHistory,
  resetGlobalHistory,
  settings,
  type Message,
} from '../src/index.js';

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

function mockFetch(
  ...responses: Array<{ status?: number; body: unknown }>
): ReturnType<typeof vi.fn> {
  const mock = vi.fn();
  for (const { status = 200, body } of responses) {
    mock.mockResolvedValueOnce({
      ok: status >= 200 && status < 300,
      status,
      text: async () => JSON.stringify(body),
    });
  }
  return mock;
}

function requestBodyFromFetchCall(
  fetchMock: ReturnType<typeof vi.fn>,
  callIndex: number,
): Record<string, unknown> {
  const call = fetchMock.mock.calls[callIndex] as [string, { body: string }] | undefined;
  if (!call) {
    throw new Error(`Missing fetch mock call at index ${callIndex}.`);
  }
  return JSON.parse(call[1].body) as Record<string, unknown>;
}

describe('LM', () => {
  it('LM.forward() throws async-only error', () => {
    const lm = new LM('openai/gpt-4.1-mini', { apiKey: 'sk-test' });
    expect(() => lm.forward(undefined, [{ role: 'user', content: 'hi' }])).toThrow(RuntimeError);
    expect(() => lm.forward(undefined, [{ role: 'user', content: 'hi' }])).toThrow(
      'LM is async-only',
    );
  });

  it('normalizes chat completions and updates global, LM, and module history', async () => {
    const fetchMock = mockFetch(
      {
        body: {
          model: 'gpt-4.1-mini',
          choices: [
            {
              message: { content: 'hello' },
              finish_reason: 'stop',
            },
          ],
          usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
        },
      },
      {
        body: {
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
        },
      },
    );

    const lm = new LM('openai/gpt-4.1-mini', { apiKey: 'sk-test', fetch: fetchMock });
    const module = new HistoryModule();
    settings.configure({ maxHistorySize: 1 });

    const first = await settings.context({ callerModules: [module] }, () => lm.acall(undefined, [
      { role: 'user', content: 'say hello' },
    ]));
    const second = await settings.context({ callerModules: [module] }, () => lm.acall(undefined, [
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

    const [url] = fetchMock.mock.calls[0] as [string, unknown];
    expect(url).toBe('https://api.openai.com/v1/chat/completions');
  });

  it('normalizes missing or null chat completion content to empty text', async () => {
    const fetchMock = mockFetch(
      {
        body: {
          model: 'gpt-4.1-mini',
          choices: [
            {
              message: {},
              finish_reason: 'stop',
            },
          ],
          usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
        },
      },
      {
        body: {
          model: 'gpt-4.1-mini',
          choices: [
            {
              message: { content: null },
              finish_reason: 'stop',
            },
          ],
          usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
        },
      },
    );

    const lm = new LM('openai/gpt-4.1-mini', { apiKey: 'sk-test', fetch: fetchMock });

    expect(await lm.acall(undefined, [{ role: 'user', content: 'first' }])).toEqual(['']);
    expect(await lm.acall(undefined, [{ role: 'user', content: 'second' }])).toEqual(['']);
  });

  it('uses message.reasoning when content is null (reasoning-only OpenAI-compatible)', async () => {
    const fetchMock = mockFetch({
      body: {
        model: 'MiniMaxAI/MiniMax-M2.7',
        choices: [
          {
            message: {
              role: 'assistant',
              content: null,
              reasoning: 'think step by step',
            },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 2, completion_tokens: 10, total_tokens: 12 },
      },
    });

    const lm = new LM('MiniMaxAI/MiniMax-M2.7', { apiKey: 'sk-test', fetch: fetchMock });
    expect(await lm.acall(undefined, [{ role: 'user', content: 'hi' }])).toEqual(['think step by step']);
  });

  it('prefers reasoning_content over reasoning when content is empty', async () => {
    const fetchMock = mockFetch({
      body: {
        model: 'deepseek/deepseek-r1',
        choices: [
          {
            message: {
              content: null,
              reasoning_content: 'rc',
              reasoning: 'r',
            },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 },
      },
    });

    const lm = new LM('deepseek/deepseek-r1', { apiKey: 'sk-test', fetch: fetchMock });
    expect(await lm.acall(undefined, [{ role: 'user', content: 'x' }])).toEqual(['rc']);
  });

  it('floors max_completion_tokens for minimaxai/* models (Together / HF ids)', async () => {
    const fetchMock = mockFetch({
      body: {
        model: 'MiniMaxAI/MiniMax-M2.7',
        choices: [
          {
            message: { content: '{"a":1}' },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
      },
    });

    const lm = new LM('MiniMaxAI/MiniMax-M2.7', {
      apiKey: 'sk-test',
      fetch: fetchMock,
    });

    await lm.acall(undefined, [{ role: 'user', content: 'Answer as JSON.' }], {
      max_completion_tokens: 2048,
    });

    const body = requestBodyFromFetchCall(fetchMock, 0);
    expect(body.model).toBe('MiniMaxAI/MiniMax-M2.7');
    expect(body.max_completion_tokens).toBe(4096);
  });

  it('concatenates chat completion content delivered as text parts (OpenAI-compatible)', async () => {
    const fetchMock = mockFetch({
      body: {
        model: 'gpt-4.1-mini',
        choices: [
          {
            message: {
              content: [
                { type: 'text', text: 'hel' },
                { type: 'text', text: 'lo' },
              ],
            },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
      },
    });

    const lm = new LM('openai/gpt-4.1-mini', { apiKey: 'sk-test', fetch: fetchMock });
    expect(await lm.acall(undefined, [{ role: 'user', content: 'hi' }])).toEqual(['hello']);
  });

  it('records reasoning token usage from nested completion token details', async () => {
    const fetchMock = mockFetch({
      body: {
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
      },
    });

    const lm = new LM('openai/gpt-4.1-mini', { apiKey: 'sk-test', fetch: fetchMock });
    await lm.acall(undefined, [{ role: 'user', content: 'track usage' }]);

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

  it('maps context-window failures to ContextWindowExceededError', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      text: async () => JSON.stringify({
        error: {
          message: 'maximum context length exceeded',
          code: 'context_length_exceeded',
        },
      }),
    });

    const lm = new LM('openai/gpt-4.1-mini', { apiKey: 'sk-test', numRetries: 0, fetch: fetchMock });

    await expect(lm.acall(undefined, [{ role: 'user', content: 'too long' } as Message])).rejects.toThrow(
      ContextWindowExceededError,
    );
  });

  it('defaults OpenRouter Minimax calls to hidden reasoning without forcing effort', async () => {
    const fetchMock = mockFetch({
      body: {
        model: 'minimax/minimax-m2.7',
        choices: [
          {
            message: { content: '{"answer":"The Port City"}' },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
      },
    });

    const lm = new LM('openrouter/minimax/minimax-m2.7', {
      apiKey: 'sk-test',
      fetch: fetchMock,
    });

    await lm.acall(undefined, [{ role: 'user', content: 'Answer as JSON.' }], { max_tokens: 2048 });

    const body = requestBodyFromFetchCall(fetchMock, 0);
    expect(body.model).toBe('minimax/minimax-m2.7');
    expect(body.max_tokens).toBe(4096);
    expect(body.reasoning).toEqual({
      exclude: true,
    });
  });

  it('aggregates OpenAI SSE streaming into a single completion (mitigates gateway timeouts)', async () => {
    const sse = [
      'data: {"id":"1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"He"},"finish_reason":null}]}',
      '',
      'data: {"id":"1","choices":[{"delta":{"content":"llo"},"finish_reason":null}]}',
      '',
      'data: {"id":"1","choices":[{"delta":{},"finish_reason":"stop"}]}',
      '',
      'data: [DONE]',
    ].join('\n');

    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve({
      ok: true,
      status: 200,
      body: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(sse));
          controller.close();
        },
      }),
      text: async () => {
        throw new Error('non-streaming path should not call response.text()');
      },
    }));

    const lm = new LM('openai/gpt-4.1-mini', { apiKey: 'sk-test', fetch: fetchMock });
    const out = await lm.acall(undefined, [{ role: 'user', content: 'hi' }], { stream: true });

    expect(out).toEqual(['Hello']);
    expect(requestBodyFromFetchCall(fetchMock, 0).stream).toBe(true);
    expect((fetchMock.mock.calls[0] as [string])[0]).toContain('/chat/completions');
  });

  it('aggregates reasoning deltas from SSE streaming when final content is empty', async () => {
    const sse = [
      'data: {"model":"MiniMaxAI/x","choices":[{"delta":{"reasoning_content":"step "}}]}',
      'data: {"choices":[{"delta":{"reasoning_content":"done"}}]}',
      'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
      'data: [DONE]',
    ].join('\n');

    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve({
      ok: true,
      status: 200,
      body: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(sse));
          controller.close();
        },
      }),
      text: async () => {
        throw new Error('non-streaming path should not call response.text()');
      },
    }));

    const lm = new LM('MiniMaxAI/MiniMax-M2.7', { apiKey: 'sk-test', fetch: fetchMock });
    const out = await lm.acall(undefined, [{ role: 'user', content: 'hi' }], { stream: true });

    expect(out).toEqual(['step done']);
  });

  it('flattens extra_body and preserves explicit Minimax reasoning overrides', async () => {
    const fetchMock = mockFetch({
      body: {
        model: 'minimax/minimax-m2.7',
        choices: [
          {
            message: { content: '{"answer":"The Port City"}' },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
      },
    });

    const lm = new LM('openrouter/minimax/minimax-m2.7', {
      apiKey: 'sk-test',
      fetch: fetchMock,
    });

    await lm.acall(undefined, [{ role: 'user', content: 'Answer as JSON.' }], {
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

    const body = requestBodyFromFetchCall(fetchMock, 0);
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
