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
    new URL('../../dspy-slim/spec/fixtures/chat_adapter_parse.json', import.meta.url),
    'utf-8',
  ),
) as { cases: ChatParseFixtureCase[] };

const jsonParseFixture = JSON.parse(
  readFileSync(
    new URL('../../dspy-slim/spec/fixtures/json_adapter_parse.json', import.meta.url),
    'utf-8',
  ),
) as { cases: JsonParseFixtureCase[] };

const messageAssemblyFixture = JSON.parse(
  readFileSync(
    new URL('../../dspy-slim/spec/fixtures/message_assembly.json', import.meta.url),
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
        typeTag: outputTypes[name],
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
