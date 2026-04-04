/**
 * §5 — Adapter contract and message assembly.
 */

import { Example } from './example.js';
import type { Field } from './field.js';
import type { BaseLM, LMOutput } from './lm.js';
import {
  serializeOwnedValue,
  snapshotOwnedValue,
  snapshotRecord,
} from './owned_value.js';
import { Signature } from './signature.js';
import type { Role, TypeTag } from './types.js';

export interface ContentPart {
  readonly type: 'text' | 'image_url' | 'file';
  readonly text?: string;
  readonly image_url?: { readonly url: string };
  readonly file?: {
    readonly file_data?: string;
    readonly filename?: string;
    readonly file_id?: string;
  };
}

export interface Message {
  readonly role: Role;
  readonly content: string | readonly ContentPart[];
}

export type Demo = Example | Record<string, unknown>;

export interface AdapterOptions {
  readonly callbacks?: readonly unknown[];
  readonly useNativeFunctionCalling?: boolean;
}

const FIELD_HEADER_RE = /^\[\[ ## (\w+) ## \]\]/;

function isObjectLike(value: unknown): value is object {
  return typeof value === 'object' && value !== null;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!isObjectLike(value)) {
    return false;
  }

  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function describeField(field: Field): string {
  const description = field.description.trim();
  const suffix = description === '' ? field.prefix : description;
  return `- \`${field.name}\` (${field.typeTag}): ${suffix}`;
}

function fieldBlock(name: string, value: unknown): string {
  return `[[ ## ${name} ## ]]\n${formatValue(value)}`;
}

function placeholderForType(typeTag: TypeTag): string {
  switch (typeTag) {
    case 'str':
      return '<string>';
    case 'int':
      return '<integer>';
    case 'float':
      return '<float>';
    case 'bool':
      return '<boolean>';
    case 'list':
      return '<array>';
    case 'dict':
      return '<object>';
    case 'literal':
      return '<literal>';
    case 'enum':
      return '<enum>';
    case 'optional':
      return '<optional>';
    case 'union':
      return '<union>';
    case 'custom':
      return '<custom>';
  }
}

function formatValue(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  if (
    value === null
    || typeof value === 'number'
    || typeof value === 'boolean'
    || typeof value === 'bigint'
  ) {
    return String(value);
  }

  return JSON.stringify(serializeOwnedValue(value), null, 2);
}

function toDemoRecord(demo: Demo): Record<string, unknown> {
  return demo instanceof Example ? demo.toDict() : snapshotRecord(demo);
}

function coerceBoolean(value: unknown): boolean {
  if (typeof value === 'boolean') {
    return value;
  }

  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'true') {
      return true;
    }
    if (normalized === 'false') {
      return false;
    }
  }

  if (value === 1 || value === '1') {
    return true;
  }

  if (value === 0 || value === '0') {
    return false;
  }

  throw new Error(`Cannot coerce ${String(value)} to bool`);
}

function coerceNumber(value: unknown, kind: 'int' | 'float'): number {
  const numeric = typeof value === 'number' ? value : Number(String(value).trim());

  if (!Number.isFinite(numeric)) {
    throw new Error(`Cannot coerce ${String(value)} to ${kind}`);
  }

  if (kind === 'int' && !Number.isInteger(numeric)) {
    throw new Error(`Expected an integer, received ${String(value)}`);
  }

  return numeric;
}

function coerceJsonContainer(value: unknown, kind: 'list' | 'dict'): unknown {
  if (kind === 'list' && Array.isArray(value)) {
    return snapshotOwnedValue(value);
  }

  if (kind === 'dict' && isPlainObject(value)) {
    return snapshotOwnedValue(value);
  }

  if (typeof value !== 'string') {
    throw new Error(`Cannot coerce ${String(value)} to ${kind}`);
  }

  const parsed = JSON.parse(value);

  if (kind === 'list' && Array.isArray(parsed)) {
    return snapshotOwnedValue(parsed);
  }

  if (kind === 'dict' && isPlainObject(parsed)) {
    return snapshotOwnedValue(parsed);
  }

  throw new Error(`Cannot coerce ${value} to ${kind}`);
}

function parseFieldValue(field: Field, value: unknown): unknown {
  switch (field.typeTag) {
    case 'str':
      return typeof value === 'string' ? value : formatValue(value);
    case 'int':
      return coerceNumber(value, 'int');
    case 'float':
      return coerceNumber(value, 'float');
    case 'bool':
      return coerceBoolean(value);
    case 'list':
      return coerceJsonContainer(value, 'list');
    case 'dict':
      return coerceJsonContainer(value, 'dict');
    case 'literal':
    case 'enum':
    case 'optional':
    case 'union':
    case 'custom':
      return snapshotOwnedValue(value);
  }
}

function assertExactOutputKeys(
  actualKeys: readonly string[],
  expectedKeys: readonly string[],
): void {
  if (actualKeys.length !== expectedKeys.length) {
    throw new Error(`Expected fields ${expectedKeys.join(', ')}, received ${actualKeys.join(', ')}`);
  }

  for (let index = 0; index < expectedKeys.length; index += 1) {
    if (actualKeys[index] !== expectedKeys[index]) {
      throw new Error(`Expected fields ${expectedKeys.join(', ')}, received ${actualKeys.join(', ')}`);
    }
  }
}

function extractLmOutputText(output: LMOutput): string {
  if (typeof output === 'string') {
    return output;
  }

  return output.text;
}

function tryParseJson(candidate: string): unknown {
  try {
    return JSON.parse(candidate);
  } catch {
    return undefined;
  }
}

function repairJson(candidate: string): string {
  const withQuotedStrings = candidate.replace(
    /'([^'\\]*(?:\\.[^'\\]*)*)'/g,
    (_match, content: string) => JSON.stringify(content.replace(/\\'/g, "'")),
  );

  return withQuotedStrings.replace(
    /([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)/g,
    '$1"$2"$3',
  );
}

function extractFirstJsonObject(source: string): string | null {
  let start = -1;
  let depth = 0;
  let activeQuote: '"' | "'" | null = null;
  let escaping = false;

  for (let index = 0; index < source.length; index += 1) {
    const char = source[index]!;

    if (start === -1) {
      if (char === '{') {
        start = index;
        depth = 1;
      }
      continue;
    }

    if (activeQuote !== null) {
      if (escaping) {
        escaping = false;
        continue;
      }

      if (char === '\\') {
        escaping = true;
        continue;
      }

      if (char === activeQuote) {
        activeQuote = null;
      }

      continue;
    }

    if (char === '"' || char === "'") {
      activeQuote = char;
      continue;
    }

    if (char === '{') {
      depth += 1;
      continue;
    }

    if (char === '}') {
      depth -= 1;
      if (depth === 0) {
        return source.slice(start, index + 1);
      }
    }
  }

  return null;
}

function parseLooseJsonObject(source: string): Record<string, unknown> | null {
  const candidates = new Set<string>();
  const trimmed = source.trim();
  if (trimmed !== '') {
    candidates.add(trimmed);
  }

  const extracted = extractFirstJsonObject(source);
  if (extracted !== null) {
    candidates.add(extracted);
  }

  for (const candidate of candidates) {
    const parsed = tryParseJson(candidate);
    if (isPlainObject(parsed)) {
      return parsed;
    }

    const repaired = tryParseJson(repairJson(candidate));
    if (isPlainObject(repaired)) {
      return repaired;
    }
  }

  return null;
}

export class AdapterParseError extends Error {
  readonly adapterName: string;
  readonly signature: Signature;
  readonly completion: string;
  readonly parsedResult?: Record<string, unknown>;

  constructor(options: {
    readonly adapterName: string;
    readonly signature: Signature;
    readonly completion: string;
    readonly message?: string;
    readonly parsedResult?: Record<string, unknown>;
  }) {
    super(options.message ?? `${options.adapterName} could not parse the LM response`);
    this.name = 'AdapterParseError';
    this.adapterName = options.adapterName;
    this.signature = options.signature;
    this.completion = options.completion;
    if (options.parsedResult !== undefined) {
      this.parsedResult = options.parsedResult;
    }
  }
}

export abstract class Adapter {
  readonly callbacks: readonly unknown[];
  readonly useNativeFunctionCalling: boolean;

  protected constructor(options: AdapterOptions = {}) {
    this.callbacks = Object.freeze([...(options.callbacks ?? [])]);
    this.useNativeFunctionCalling = options.useNativeFunctionCalling ?? false;
  }

  call(
    lm: BaseLM,
    lmKwargs: Record<string, unknown>,
    signature: Signature,
    demos: readonly Demo[],
    inputs: Record<string, unknown>,
  ): Record<string, unknown>[] {
    const messages = this.format(signature, demos, inputs);
    const outputs = lm.call(undefined, messages, snapshotRecord(lmKwargs));

    return outputs.map((output) => this.parse(signature, extractLmOutputText(output)));
  }

  async acall(
    lm: BaseLM,
    lmKwargs: Record<string, unknown>,
    signature: Signature,
    demos: readonly Demo[],
    inputs: Record<string, unknown>,
  ): Promise<Record<string, unknown>[]> {
    const messages = this.format(signature, demos, inputs);
    const outputs = await lm.acall(undefined, messages, snapshotRecord(lmKwargs));

    return outputs.map((output) => this.parse(signature, extractLmOutputText(output)));
  }

  format(
    signature: Signature,
    demos: readonly Demo[],
    inputs: Record<string, unknown>,
  ): Message[] {
    return [
      { role: 'system', content: this.formatSystemMessage(signature) },
      ...this.formatDemos(signature, demos),
      {
        role: 'user',
        content: this.formatUserMessageContent(signature, snapshotRecord(inputs), '', '', true),
      },
    ];
  }

  formatSystemMessage(signature: Signature): string {
    return [
      this.formatFieldDescription(signature),
      this.formatFieldStructure(signature),
      this.formatTaskDescription(signature),
    ].join('\n\n');
  }

  formatFieldDescription(signature: Signature): string {
    const inputLines = [...signature.inputFields.values()].map(describeField);
    const outputLines = [...signature.outputFields.values()].map(describeField);

    return [
      'Your input fields are:',
      inputLines.length === 0 ? '- none' : inputLines.join('\n'),
      '',
      'Your output fields are:',
      outputLines.length === 0 ? '- none' : outputLines.join('\n'),
    ].join('\n');
  }

  abstract formatFieldStructure(signature: Signature): string;

  formatTaskDescription(signature: Signature): string {
    return `In adhering to this structure, your objective is: ${signature.instructions}`;
  }

  formatUserMessageContent(
    signature: Signature,
    inputs: Record<string, unknown>,
    prefix = '',
    suffix = '',
    mainRequest = false,
  ): string {
    const parts: string[] = [];

    if (prefix.trim() !== '') {
      parts.push(prefix.trim());
    }

    for (const [name] of signature.inputFields) {
      if (name in inputs) {
        parts.push(fieldBlock(name, inputs[name]));
      }
    }

    if (mainRequest) {
      parts.push(this.userMessageOutputRequirements(signature));
    }

    if (suffix.trim() !== '') {
      parts.push(suffix.trim());
    }

    return parts.join('\n\n').trim();
  }

  protected userMessageOutputRequirements(signature: Signature): string {
    const fields = [...signature.outputFields.keys()].map((name) => `\`[[ ## ${name} ## ]]\``);
    return `Respond with the corresponding output fields, starting with ${fields.join(', then ')}, and then ending with the marker for \`[[ ## completed ## ]]\`.`;
  }

  formatAssistantMessageContent(
    signature: Signature,
    outputs: Record<string, unknown>,
    missingFieldMessage?: string,
  ): string {
    const parts: string[] = [];

    for (const [name] of signature.outputFields) {
      if (name in outputs) {
        parts.push(fieldBlock(name, outputs[name]));
      } else if (missingFieldMessage !== undefined) {
        parts.push(fieldBlock(name, missingFieldMessage));
      }
    }

    parts.push('[[ ## completed ## ]]');
    return parts.join('\n\n').trim();
  }

  formatDemos(signature: Signature, demos: readonly Demo[]): Message[] {
    const complete: Record<string, unknown>[] = [];
    const incomplete: Record<string, unknown>[] = [];

    for (const rawDemo of demos) {
      const demo = toDemoRecord(rawDemo);
      const inputKeys = [...signature.inputFields.keys()];
      const outputKeys = [...signature.outputFields.keys()];

      const hasInput = inputKeys.some((name) => name in demo);
      const hasOutput = outputKeys.some((name) => name in demo);
      const isComplete = [...inputKeys, ...outputKeys].every((name) => demo[name] !== undefined && demo[name] !== null);

      if (isComplete) {
        complete.push(demo);
      } else if (hasInput && hasOutput) {
        incomplete.push(demo);
      }
    }

    const messages: Message[] = [];
    const incompletePrefix = 'This is an example of the task, though some input or output fields are not supplied.';

    for (const demo of incomplete) {
      messages.push({
        role: 'user',
        content: this.formatUserMessageContent(signature, demo, incompletePrefix),
      });
      messages.push({
        role: 'assistant',
        content: this.formatAssistantMessageContent(
          signature,
          demo,
          'Not supplied for this particular example.',
        ),
      });
    }

    for (const demo of complete) {
      messages.push({
        role: 'user',
        content: this.formatUserMessageContent(signature, demo),
      });
      messages.push({
        role: 'assistant',
        content: this.formatAssistantMessageContent(
          signature,
          demo,
          'Not supplied for this conversation history message.',
        ),
      });
    }

    return messages;
  }

  abstract parse(signature: Signature, completion: string): Record<string, unknown>;
}

export class ChatAdapter extends Adapter {
  constructor(options: AdapterOptions = {}) {
    super(options);
  }

  override formatFieldStructure(signature: Signature): string {
    const parts = [
      'All interactions will be structured in the following way, with the appropriate values filled in.',
      ...[...signature.inputFields.values()].map((field) => fieldBlock(field.name, placeholderForType(field.typeTag))),
      ...[...signature.outputFields.values()].map((field) => fieldBlock(field.name, placeholderForType(field.typeTag))),
      '[[ ## completed ## ]]',
    ];

    return parts.join('\n\n').trim();
  }

  override parse(signature: Signature, completion: string): Record<string, unknown> {
    const sections = new Map<string, string>();
    const expectedKeys = [...signature.outputFields.keys()];

    let currentHeader: string | null = null;
    let currentLines: string[] = [];

    const flushSection = (): void => {
      if (currentHeader === null) {
        return;
      }

      if (!signature.outputFields.has(currentHeader) || sections.has(currentHeader)) {
        return;
      }

      sections.set(currentHeader, currentLines.join('\n').trim());
    };

    for (const line of completion.split(/\r?\n/)) {
      const stripped = line.trim();
      const match = stripped.match(FIELD_HEADER_RE);
      if (match !== null) {
        flushSection();
        currentHeader = match[1] ?? null;
        currentLines = [];

        const remainder = stripped.slice(match[0].length).trim();
        if (remainder !== '') {
          currentLines.push(remainder);
        }
      } else {
        currentLines.push(line);
      }
    }

    flushSection();

    const actualKeys = [...sections.keys()];
    try {
      assertExactOutputKeys(actualKeys, expectedKeys);
    } catch (error) {
      throw new AdapterParseError({
        adapterName: 'ChatAdapter',
        signature,
        completion,
        parsedResult: Object.fromEntries(sections),
        message: error instanceof Error ? error.message : 'ChatAdapter could not parse the LM response',
      });
    }

    const parsed: Record<string, unknown> = {};
    try {
      for (const [name, rawValue] of sections) {
        parsed[name] = parseFieldValue(signature.outputFields.get(name)!, rawValue);
      }
    } catch (error) {
      throw new AdapterParseError({
        adapterName: 'ChatAdapter',
        signature,
        completion,
        parsedResult: parsed,
        message: error instanceof Error ? error.message : 'ChatAdapter could not parse the LM response',
      });
    }

    return parsed;
  }
}

export class JSONAdapter extends ChatAdapter {
  constructor(options: AdapterOptions = {}) {
    super({
      ...options,
      useNativeFunctionCalling: options.useNativeFunctionCalling ?? true,
    });
  }

  override formatFieldStructure(signature: Signature): string {
    const inputBlocks = [...signature.inputFields.values()].map((field) => (
      fieldBlock(field.name, placeholderForType(field.typeTag))
    ));

    const outputShape = Object.fromEntries(
      [...signature.outputFields.values()].map((field) => [
        field.name,
        placeholderForType(field.typeTag),
      ]),
    );

    return [
      'All interactions will be structured in the following way, with the appropriate values filled in.',
      'Inputs will have the following structure:',
      inputBlocks.join('\n\n').trim(),
      'Outputs will be a JSON object with the following fields.',
      JSON.stringify(outputShape, null, 2),
    ].join('\n\n').trim();
  }

  protected override userMessageOutputRequirements(signature: Signature): string {
    const fields = [...signature.outputFields.keys()].map((name) => `\`${name}\``);
    return `Respond with a JSON object in the following order of fields: ${fields.join(', then ')}.`;
  }

  override formatAssistantMessageContent(
    signature: Signature,
    outputs: Record<string, unknown>,
    missingFieldMessage?: string,
  ): string {
    const ordered: Record<string, unknown> = {};

    for (const [name] of signature.outputFields) {
      if (name in outputs) {
        ordered[name] = serializeOwnedValue(outputs[name]);
      } else if (missingFieldMessage !== undefined) {
        ordered[name] = missingFieldMessage;
      }
    }

    return JSON.stringify(ordered, null, 2);
  }

  override parse(signature: Signature, completion: string): Record<string, unknown> {
    const parsedObject = parseLooseJsonObject(completion);
    if (parsedObject === null) {
      throw new AdapterParseError({
        adapterName: 'JSONAdapter',
        signature,
        completion,
        message: 'LM response cannot be serialized to a JSON object.',
      });
    }

    const filtered: Record<string, unknown> = {};
    for (const [name, field] of signature.outputFields) {
      if (name in parsedObject) {
        try {
          filtered[name] = parseFieldValue(field, parsedObject[name]);
        } catch (error) {
          throw new AdapterParseError({
            adapterName: 'JSONAdapter',
            signature,
            completion,
            parsedResult: filtered,
            message: error instanceof Error ? error.message : 'JSONAdapter could not parse the LM response',
          });
        }
      }
    }

    const actualKeys = Object.keys(filtered);
    const expectedKeys = [...signature.outputFields.keys()];

    try {
      assertExactOutputKeys(actualKeys, expectedKeys);
    } catch (error) {
      throw new AdapterParseError({
        adapterName: 'JSONAdapter',
        signature,
        completion,
        parsedResult: filtered,
        message: error instanceof Error ? error.message : 'JSONAdapter could not parse the LM response',
      });
    }

    return filtered;
  }
}
