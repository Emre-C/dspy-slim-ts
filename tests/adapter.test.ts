import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import {
  AdapterParseError,
  ChatAdapter,
  History,
  JSONAdapter,
  createField,
  createSignature,
  signatureFromString,
  type TypeTag,
} from '../src/index.js';

interface ChatParseFixtureCase {
  id: string;
  signature_outputs: string[];
  completion: string;
  expected?: Record<string, unknown>;
  expected_error?: string;
}

interface JsonParseFixtureCase extends ChatParseFixtureCase {
  output_types?: Record<string, TypeTag>;
}

interface MessageAssemblyFixtureCase {
  id: string;
  adapter?: 'chat' | 'json';
  signature: string;
  demos?: Record<string, unknown>[];
  demo_output?: Record<string, unknown>;
  inputs?: Record<string, unknown>;
  expected_message_roles?: string[];
  expected_system_contains?: string[];
  expected_last_user_contains?: string[];
  expected_user_content_contains?: string[];
  expected_assistant_content_contains?: string[];
  expected_assistant_content_is_valid_json?: boolean;
}

const chatParseFixture = JSON.parse(
  readFileSync(
    new URL('../../spec/fixtures/chat_adapter_parse.json', import.meta.url),
    'utf-8',
  ),
) as { cases: ChatParseFixtureCase[] };

const jsonParseFixture = JSON.parse(
  readFileSync(
    new URL('../../spec/fixtures/json_adapter_parse.json', import.meta.url),
    'utf-8',
  ),
) as { cases: JsonParseFixtureCase[] };

const messageAssemblyFixture = JSON.parse(
  readFileSync(
    new URL('../../spec/fixtures/message_assembly.json', import.meta.url),
    'utf-8',
  ),
) as { cases: MessageAssemblyFixtureCase[] };

function createOutputOnlySignature(
  outputNames: readonly string[],
  outputTypes: Readonly<Record<string, TypeTag>> = {},
) {
  const outputs = new Map(
    outputNames.map((name) => [
      name,
      createField({
        kind: 'output',
        name,
        ...(outputTypes[name] === undefined ? {} : { typeTag: outputTypes[name] }),
      }),
    ]),
  );

  return createSignature(new Map(), outputs);
}

describe('ChatAdapter parse (spec fixtures)', () => {
  const adapter = new ChatAdapter();

  for (const c of chatParseFixture.cases) {
    it(c.id, () => {
      const signature = createOutputOnlySignature(c.signature_outputs);

      if (c.expected_error) {
        expect(() => adapter.parse(signature, c.completion)).toThrow(AdapterParseError);
        return;
      }

      expect(adapter.parse(signature, c.completion)).toEqual(c.expected);
    });
  }
});

describe('JSONAdapter parse (spec fixtures)', () => {
  const adapter = new JSONAdapter();

  for (const c of jsonParseFixture.cases) {
    it(c.id, () => {
      const signature = createOutputOnlySignature(
        c.signature_outputs,
        c.output_types,
      );

      if (c.expected_error) {
        expect(() => adapter.parse(signature, c.completion)).toThrow(AdapterParseError);
        return;
      }

      expect(adapter.parse(signature, c.completion)).toEqual(c.expected);
    });
  }
});

describe('Adapter message assembly (spec fixtures)', () => {
  for (const c of messageAssemblyFixture.cases) {
    it(c.id, () => {
      const adapter = c.adapter === 'json' ? new JSONAdapter() : new ChatAdapter();
      const signature = signatureFromString(c.signature);
      const messages = adapter.format(signature, c.demos ?? [], c.inputs ?? {});

      if (c.expected_message_roles) {
        expect(messages.map((message) => message.role)).toEqual(c.expected_message_roles);
      }

      if (c.expected_system_contains) {
        const systemMessage = messages[0]?.content;
        expect(typeof systemMessage).toBe('string');
        for (const snippet of c.expected_system_contains) {
          expect(systemMessage as string).toContain(snippet);
        }
      }

      if (c.expected_last_user_contains) {
        const lastUser = messages.at(-1)?.content;
        expect(typeof lastUser).toBe('string');
        for (const snippet of c.expected_last_user_contains) {
          expect(lastUser as string).toContain(snippet);
        }
      }

      if (c.expected_user_content_contains) {
        const content = adapter.formatUserMessageContent(signature, c.inputs ?? {}, '', '', true);
        for (const snippet of c.expected_user_content_contains) {
          expect(content).toContain(snippet);
        }
      }

      if (c.expected_assistant_content_contains) {
        const content = adapter.formatAssistantMessageContent(signature, c.demo_output ?? {});
        for (const snippet of c.expected_assistant_content_contains) {
          expect(content).toContain(snippet);
        }
      }

      if (c.expected_assistant_content_is_valid_json) {
        const content = adapter.formatAssistantMessageContent(signature, c.demo_output ?? {});
        expect(() => JSON.parse(content)).not.toThrow();
      }
    });
  }
});

describe('Adapter hardening', () => {
  it('emits incomplete demos before complete demos when both are usable examples', () => {
    const adapter = new ChatAdapter();
    const signature = signatureFromString('question, context -> answer');

    const messages = adapter.format(
      signature,
      [
        { question: 'Complete question', context: 'ctx', answer: 'Complete answer' },
        { question: 'Incomplete question', answer: 'Incomplete answer' },
      ],
      { question: 'Current question', context: 'current context' },
    );

    const userMessages = messages
      .filter((message) => message.role === 'user')
      .map((message) => message.content as string);

    expect(userMessages[0]).toContain('Incomplete question');
    expect(userMessages[1]).toContain('Complete question');
    expect(userMessages[2]).toContain('Current question');
  });

  it('formats conversation history before the current request when a history payload is present', () => {
    const adapter = new ChatAdapter();
    const signature = createSignature(
      new Map([
        ['question', createField({ kind: 'input', name: 'question' })],
        ['history', createField({ kind: 'input', name: 'history', typeTag: 'custom', isTypeUndefined: false })],
      ]),
      new Map([
        ['answer', createField({ kind: 'output', name: 'answer' })],
      ]),
    );

    const messages = adapter.format(signature, [], {
      question: 'Are you sure?',
      history: new History([
        { question: 'What is the capital of France?', answer: 'Paris' },
      ]),
    });

    expect(messages.map((message) => message.role)).toEqual(['system', 'user', 'assistant', 'user']);
    expect(messages[1]?.content).toContain('What is the capital of France?');
    expect(messages[2]?.content).toContain('Paris');
    expect(messages[3]?.content).toContain('Are you sure?');
    expect(messages[3]?.content).not.toContain('history');
  });
});

// ---------------------------------------------------------------------------
// Optional output fields
//
// `optional[T]` output fields must be accepted when the LM omits them, must
// be rejected in the normal 'missing required field' and 'unexpected field'
// error paths, and must be announced to the LM in prompts so long-context
// callers do not have to guess. These tests pin all three behaviors for
// both `ChatAdapter` and `JSONAdapter`.
// ---------------------------------------------------------------------------

describe('Optional output fields', () => {
  const effectOracleSig = signatureFromString(
    'prompt: str -> ' +
      'kind: literal["value", "effect"], ' +
      'value: optional[str], ' +
      'effect_name: optional[str], ' +
      'effect_args: optional[dict]',
  );

  describe('ChatAdapter.parse', () => {
    const adapter = new ChatAdapter();

    it('accepts omitted trailing optional fields when required fields are present', () => {
      const completion =
        '[[ ## kind ## ]]\nvalue\n\n[[ ## value ## ]]\nhello\n\n[[ ## completed ## ]]';

      const parsed = adapter.parse(effectOracleSig, completion);

      expect(parsed).toEqual({ kind: 'value', value: 'hello' });
    });

    it('accepts all optional fields being omitted when only required is present', () => {
      const completion = '[[ ## kind ## ]]\nvalue\n\n[[ ## completed ## ]]';

      const parsed = adapter.parse(effectOracleSig, completion);

      expect(parsed).toEqual({ kind: 'value' });
    });

    it('still throws when a required field is missing', () => {
      const requiredSig = signatureFromString(
        'question: str -> answer: str, score: float',
      );
      const completion = '[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]';

      expect(() => adapter.parse(requiredSig, completion)).toThrowError(
        AdapterParseError,
      );
    });

    // NB: ChatAdapter intentionally ignores unknown `[[ ## ... ## ]]` headers
    // (see the `ignores_unknown_headers` fixture). The cross-adapter contract
    // is that the wire format is free to include extra sections. Declared
    // fields still have to obey order, which is the next case.

    it('rejects declared fields emitted out of declaration order', () => {
      const completion =
        '[[ ## value ## ]]\nhello\n\n[[ ## kind ## ]]\nvalue\n\n[[ ## completed ## ]]';

      expect(() => adapter.parse(effectOracleSig, completion)).toThrowError(
        AdapterParseError,
      );
    });
  });

  describe('JSONAdapter.parse', () => {
    const adapter = new JSONAdapter();

    it('accepts omitted optional fields in the JSON payload', () => {
      const parsed = adapter.parse(
        effectOracleSig,
        '{"kind": "value", "value": "hello"}',
      );

      expect(parsed).toEqual({ kind: 'value', value: 'hello' });
    });

    it('coerces a present optional value through its inner TypeTag', () => {
      const sig = signatureFromString('q: str -> count: optional[int]');

      const parsed = adapter.parse(sig, '{"count": "42"}');

      expect(parsed).toEqual({ count: 42 });
    });

    it('still throws when a required JSON field is missing', () => {
      const requiredSig = signatureFromString('q: str -> answer: str, score: float');

      expect(() => adapter.parse(requiredSig, '{"answer": "Paris"}')).toThrowError(
        AdapterParseError,
      );
    });
  });

  describe('Prompt formatting', () => {
    it('ChatAdapter marks optional fields in the system-message structure block', () => {
      const adapter = new ChatAdapter();
      const system = adapter.formatSystemMessage(effectOracleSig);

      // Structure block placeholder carries the `(optional)` marker so the
      // LM can tell required fields from omittable ones at a glance.
      expect(system).toContain('<string (optional)>');
      expect(system).toContain('<object (optional)>');
      // Required literal field is still rendered without the optional marker.
      expect(system).toContain('[[ ## kind ## ]]');
    });

    it('ChatAdapter announces optional output names in the user-message requirements', () => {
      const adapter = new ChatAdapter();
      const messages = adapter.format(effectOracleSig, [], { prompt: 'hi' });
      const lastUser = messages.at(-1)?.content as string;

      expect(lastUser).toContain('`value`');
      expect(lastUser).toContain('`effect_name`');
      expect(lastUser).toContain('`effect_args`');
      expect(lastUser).toContain('optional and may be omitted');
    });

    it('JSONAdapter announces optional fields in its user-message requirements', () => {
      const adapter = new JSONAdapter();
      const messages = adapter.format(effectOracleSig, [], { prompt: 'hi' });
      const lastUser = messages.at(-1)?.content as string;

      expect(lastUser).toContain('optional and may be omitted');
    });

    it('describeField annotates optional fields with the inner type and "optional"', () => {
      const adapter = new ChatAdapter();
      const system = adapter.formatSystemMessage(effectOracleSig);

      // `describeField` lines live in the "Your output fields are:" section.
      expect(system).toMatch(/`value` \(str, optional\)/);
      expect(system).toMatch(/`effect_args` \(dict, optional\)/);
    });
  });
});
