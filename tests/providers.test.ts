import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  BaseLM,
  JSONAdapter,
  ValueError,
  createField,
  createSignature,
  registerProfile,
  resolveProfile,
  type LMOutput,
  type Message,
  type ProviderProfile,
} from '../src/index.js';
import { openRouterMinimaxProfile } from '../src/providers/openrouter_minimax.js';
import { clearProfiles } from '../src/providers/profile.js';

class SyncStubLM extends BaseLM {
  readonly outputsByCall: readonly (readonly LMOutput[])[];
  readonly calls: Array<{ readonly kwargs: Record<string, unknown> }> = [];

  constructor(model: string, outputsByCall: readonly (readonly LMOutput[])[]) {
    super({ model });
    this.outputsByCall = outputsByCall;
  }

  protected override generate(
    _prompt?: string,
    _messages?: readonly Message[],
    kwargs: Record<string, unknown> = {},
  ): readonly LMOutput[] {
    const outputs = this.outputsByCall[this.calls.length];
    this.calls.push({ kwargs: { ...kwargs } });
    if (outputs === undefined) {
      throw new Error('SyncStubLM ran out of scripted outputs.');
    }
    return outputs;
  }
}

class AsyncStubLM extends BaseLM {
  readonly outputsByCall: readonly (readonly LMOutput[])[];
  readonly calls: Array<{ readonly kwargs: Record<string, unknown> }> = [];

  constructor(model: string, outputsByCall: readonly (readonly LMOutput[])[]) {
    super({ model });
    this.outputsByCall = outputsByCall;
  }

  protected override async agenerate(
    _prompt?: string,
    _messages?: readonly Message[],
    kwargs: Record<string, unknown> = {},
  ): Promise<readonly LMOutput[]> {
    const outputs = this.outputsByCall[this.calls.length];
    this.calls.push({ kwargs: { ...kwargs } });
    if (outputs === undefined) {
      throw new Error('AsyncStubLM ran out of scripted outputs.');
    }
    return outputs;
  }
}

function singleOutputSignature() {
  return createSignature(
    new Map([['question', createField({ kind: 'input', name: 'question' })]]),
    new Map([['answer', createField({ kind: 'output', name: 'answer' })]]),
  );
}

describe('Provider profile registry', () => {
  beforeEach(() => {
    clearProfiles();
    registerProfile(openRouterMinimaxProfile);
  });

  afterEach(() => {
    clearProfiles();
  });

  it('returns null for models matching no registered profile', () => {
    expect(resolveProfile('anthropic/claude-3.5')).toBeNull();
    expect(resolveProfile('openai/gpt-4.1-mini')).toBeNull();
  });

  it('resolves the built-in OpenRouter Minimax profile for matching models', () => {
    const profile = resolveProfile('openrouter/minimax/minimax-m2.7');
    expect(profile).not.toBeNull();
    expect(profile?.id).toBe('openrouter-minimax');
  });

  it('rejects duplicate registrations with a clear error', () => {
    expect(() => registerProfile(openRouterMinimaxProfile)).toThrow(ValueError);
    expect(() => registerProfile(openRouterMinimaxProfile)).toThrow(
      /'openrouter-minimax' is already registered/,
    );
  });

  it('resolves first-registered profile when multiple match (order contract)', () => {
    const earlier: ProviderProfile = {
      id: 'earlier',
      matches: (model) => model.startsWith('fictional/'),
    };
    const later: ProviderProfile = {
      id: 'later',
      matches: (model) => model.startsWith('fictional/'),
    };

    registerProfile(earlier);
    registerProfile(later);

    expect(resolveProfile('fictional/alpha')?.id).toBe('earlier');
  });
});

describe('Provider profile dispatch through core modules', () => {
  beforeEach(() => {
    clearProfiles();
  });

  afterEach(() => {
    clearProfiles();
  });

  it('routes Adapter.call retries through a custom profile', () => {
    registerProfile({
      id: 'acme-sync',
      matches: (model) => model.startsWith('fictional/'),
      adapterRetry: (_lm, lmKwargs, error) => {
        if (!(error instanceof Error) || error.name !== 'AdapterParseError') {
          return null;
        }
        return { ...lmKwargs, retry_pass: true };
      },
    });

    const lm = new SyncStubLM('fictional/acme-1', [['{}'], ['{"answer":"ok"}']]);
    const adapter = new JSONAdapter();
    const outputs = adapter.call(lm, {}, singleOutputSignature(), [], { question: 'Why?' });

    expect(outputs).toEqual([{ answer: 'ok' }]);
    expect(lm.calls).toHaveLength(2);
    expect(lm.calls[0]?.kwargs.retry_pass).toBeUndefined();
    expect(lm.calls[1]?.kwargs.retry_pass).toBe(true);
  });

  it('routes Adapter.acall retries through a custom profile', async () => {
    registerProfile({
      id: 'acme-async',
      matches: (model) => model.startsWith('fictional/'),
      adapterRetry: (_lm, lmKwargs, error) => {
        if (!(error instanceof Error) || error.name !== 'AdapterParseError') {
          return null;
        }
        return { ...lmKwargs, retry_pass: true };
      },
    });

    const lm = new AsyncStubLM('fictional/acme-2', [['{}'], ['{"answer":"ok"}']]);
    const adapter = new JSONAdapter();
    const outputs = await adapter.acall(lm, {}, singleOutputSignature(), [], { question: 'Why?' });

    expect(outputs).toEqual([{ answer: 'ok' }]);
    expect(lm.calls).toHaveLength(2);
    expect(lm.calls[1]?.kwargs.retry_pass).toBe(true);
  });

  it('rethrows the original error when the profile declines the retry', () => {
    registerProfile({
      id: 'acme-decline',
      matches: (model) => model.startsWith('fictional/'),
      adapterRetry: () => null,
    });

    const lm = new SyncStubLM('fictional/acme-3', [['{}']]);
    const adapter = new JSONAdapter();

    expect(() => adapter.call(lm, {}, singleOutputSignature(), [], { question: 'Why?' }))
      .toThrow(/Expected fields answer/);
    expect(lm.calls).toHaveLength(1);
  });

  it('skips retry when no profile matches the model', () => {
    const lm = new SyncStubLM('anthropic/claude-3.5', [['{}']]);
    const adapter = new JSONAdapter();

    expect(() => adapter.call(lm, {}, singleOutputSignature(), [], { question: 'Why?' }))
      .toThrow(/Expected fields answer/);
    expect(lm.calls).toHaveLength(1);
  });

  it('exposes mapRequest through resolveProfile for transport dispatch', () => {
    registerProfile({
      id: 'acme-map',
      matches: (model) => model.startsWith('fictional/'),
      mapRequest: (req) => ({ ...req, stamped_by: 'acme' }),
    });

    const mapped = resolveProfile('fictional/anything')?.mapRequest?.({ foo: 'bar' });

    expect(mapped).toEqual({ foo: 'bar', stamped_by: 'acme' });
  });
});
