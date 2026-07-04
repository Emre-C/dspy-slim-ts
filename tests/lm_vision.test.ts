import { describe, expect, it } from 'vitest';
import {
  LM,
  ReplayLM,
  type ContentPart,
  type Message,
} from '../src/index.js';

describe('LM.supportsVision', () => {
  it('returns true for openrouter/google/gemini-3-flash-preview', () => {
    const lm = new LM({
      model: 'openrouter/google/gemini-3-flash-preview',
      apiKey: 'sk-test',
    });
    expect(lm.supportsVision).toBe(true);
  });

  it('returns true for openai/gpt-4o', () => {
    const lm = new LM({ model: 'openai/gpt-4o', apiKey: 'sk-test' });
    expect(lm.supportsVision).toBe(true);
  });

  it('keeps the vision allowlist conservative', () => {
    const lm = new LM({ model: 'openai/gpt-4.1-mini', apiKey: 'sk-test' });
    expect(lm.supportsVision).toBe(false);
  });

  it('returns false for legacy text-only models', () => {
    const lm = new LM({ model: 'openai/gpt-3.5-turbo', apiKey: 'sk-test' });
    expect(lm.supportsVision).toBe(false);
  });

  it('returns false for openrouter/minimax/minimax-m2', () => {
    const lm = new LM({
      model: 'openrouter/minimax/minimax-m2',
      apiKey: 'sk-test',
    });
    expect(lm.supportsVision).toBe(false);
  });

  it('forceVisionCapable: true overrides the heuristic', () => {
    const lm = new LM({
      model: 'openai/some-future-vision-model',
      apiKey: 'sk-test',
      forceVisionCapable: true,
    });
    expect(lm.supportsVision).toBe(true);
  });

  it('forceVisionCapable: false overrides a recognized vision model', () => {
    const lm = new LM({
      model: 'openrouter/google/gemini-3-flash-preview',
      apiKey: 'sk-test',
      forceVisionCapable: false,
    });
    expect(lm.supportsVision).toBe(false);
  });

  it('vision-capable LM still preserves other capability flags', () => {
    const lm = new LM({
      model: 'openrouter/google/gemini-3-flash-preview',
      apiKey: 'sk-test',
    });
    expect(lm.supportsFunctionCalling).toBe(true);
    expect(lm.supportsVision).toBe(true);
  });

  it('acompletion returns the first output text for raw multimodal calls', async () => {
    const lm = new ReplayLM([{ text: 'raw answer' }]);
    const messages: readonly Message[] = [{
      role: 'user',
      content: [
        { type: 'text', text: 'Describe this image.' },
        {
          type: 'image_url',
          image_url: { url: 'data:image/png;base64,aGVsbG8=' },
        },
      ],
    }];

    await expect(lm.acompletion({ messages })).resolves.toBe('raw answer');
  });

  it('preserves image_url content parts in the OpenRouter request body', async () => {
    let body: unknown;
    const fetchImpl: typeof fetch = async (_url, init) => {
      body = JSON.parse(String(init?.body));
      return new Response(JSON.stringify({
        model: 'google/gemini-3-flash-preview',
        choices: [{ message: { content: 'ok' } }],
      }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    };
    const lm = new LM({
      model: 'openrouter/google/gemini-3-flash-preview',
      apiKey: 'sk-test',
      fetch: fetchImpl,
    });
    const content: readonly ContentPart[] = Object.freeze([
      { type: 'text', text: 'look' },
      {
        type: 'image_url',
        image_url: { url: 'data:image/png;base64,aGVsbG8=' },
      },
    ]);

    await lm.acall(undefined, [{ role: 'user', content }]);

    expect(body).toMatchObject({
      model: 'google/gemini-3-flash-preview',
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: 'look' },
          {
            type: 'image_url',
            image_url: { url: 'data:image/png;base64,aGVsbG8=' },
          },
        ],
      }],
    });
  });
});
