/**
 * §7.1–§7.2 — Tool and ToolCalls.
 *
 * TypeScript cannot recover source-level type annotations at runtime, so tool
 * metadata is best-effort inferred from parameter names and may be refined with
 * explicit schema/type overrides.
 */

import {
  type JSONSchemaType,
  coerceFieldValue,
  schemaTypeToTypeTag,
  typeTagToSchemaType,
} from './codec.js';
import { ValueError } from './exceptions.js';
import { isPlainObject } from './guards.js';
import { serializeOwnedValue, snapshotRecord } from './owned_value.js';
import type { TypeTag } from './types.js';
type ToolFunction = (...args: any[]) => unknown;
type InvokeMode = 'positional' | 'record';

export interface JSONSchema {
  readonly type?: JSONSchemaType;
  readonly description?: string;
  readonly default?: unknown;
  readonly enum?: readonly unknown[];
  readonly items?: JSONSchema;
  readonly properties?: Readonly<Record<string, JSONSchema>>;
}

export interface ToolOptions {
  readonly name?: string;
  readonly desc?: string;
  readonly args?: Readonly<Record<string, JSONSchema>>;
  readonly argTypes?: Readonly<Record<string, TypeTag>>;
}

export interface ToolCall {
  readonly name: string;
  readonly args: Record<string, unknown>;
}

interface ParameterSpec {
  readonly name: string;
}

function splitTopLevel(input: string): string[] {
  const parts: string[] = [];
  const stack: string[] = [];
  let activeQuote: string | null = null;
  let escaping = false;
  let start = 0;

  for (let index = 0; index < input.length; index += 1) {
    const char = input[index]!;

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

    if (char === '"' || char === "'" || char === '`') {
      activeQuote = char;
      continue;
    }

    if (char === '(' || char === '{' || char === '[') {
      stack.push(char);
      continue;
    }

    if (char === ')' || char === '}' || char === ']') {
      stack.pop();
      continue;
    }

    if (char === ',' && stack.length === 0) {
      parts.push(input.slice(start, index));
      start = index + 1;
    }
  }

  parts.push(input.slice(start));
  return parts;
}

function parameterListFromSource(source: string): string | null {
  const trimmed = source.trim();
  const arrowIndex = trimmed.indexOf('=>');

  if (arrowIndex !== -1) {
    let beforeArrow = trimmed.slice(0, arrowIndex).trim();
    if (beforeArrow.startsWith('async ')) {
      beforeArrow = beforeArrow.slice(6).trim();
    }

    if (beforeArrow.startsWith('(') && beforeArrow.endsWith(')')) {
      return beforeArrow.slice(1, -1);
    }

    return beforeArrow;
  }

  const openIndex = trimmed.indexOf('(');
  if (openIndex === -1) {
    return null;
  }

  let depth = 0;
  for (let index = openIndex; index < trimmed.length; index += 1) {
    const char = trimmed[index]!;
    if (char === '(') {
      depth += 1;
    } else if (char === ')') {
      depth -= 1;
      if (depth === 0) {
        return trimmed.slice(openIndex + 1, index);
      }
    }
  }

  return null;
}

function inferParameters(func: ToolFunction): readonly ParameterSpec[] {
  const parameterList = parameterListFromSource(func.toString());
  if (parameterList === null || parameterList.trim() === '') {
    return Object.freeze([]);
  }

  return Object.freeze(splitTopLevel(parameterList).flatMap((segment) => {
    const raw = segment.trim();
    if (raw === '') {
      return [];
    }

    const withoutRest = raw.replace(/^\.\.\./, '').trim();
    const name = withoutRest.split('=')[0]!.trim();
    if (!/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name)) {
      throw new Error(
        'Tool parameters must be simple identifiers unless explicit args metadata is provided.',
      );
    }

    return [{ name }];
  }));
}

function normalizeArgNames(
  inferred: readonly ParameterSpec[],
  args: Readonly<Record<string, JSONSchema>> | undefined,
  argTypes: Readonly<Record<string, TypeTag>> | undefined,
): readonly string[] {
  if (inferred.length > 0) {
    return Object.freeze(inferred.map((parameter) => parameter.name));
  }

  const explicitNames = new Set<string>([
    ...Object.keys(args ?? {}),
    ...Object.keys(argTypes ?? {}),
  ]);

  return Object.freeze([...explicitNames]);
}

function normalizeArgs(
  argNames: readonly string[],
  args: Readonly<Record<string, JSONSchema>> | undefined,
  argTypes: Readonly<Record<string, TypeTag>> | undefined,
): Record<string, JSONSchema> {
  const normalized: Record<string, JSONSchema> = {};
  for (const name of argNames) {
    const provided = args?.[name];
    const type = provided?.type === undefined && argTypes?.[name] !== undefined
      ? typeTagToSchemaType(argTypes[name]!)
      : undefined;

    normalized[name] = Object.freeze(type === undefined
      ? { ...(provided ?? {}) }
      : { ...(provided ?? {}), type });
  }

  return normalized;
}

function normalizeArgTypes(
  argNames: readonly string[],
  args: Readonly<Record<string, JSONSchema>>,
  argTypes: Readonly<Record<string, TypeTag>> | undefined,
): Record<string, TypeTag> {
  const normalized: Record<string, TypeTag> = {};
  for (const name of argNames) {
    normalized[name] = argTypes?.[name] ?? schemaTypeToTypeTag(args[name]?.type);
  }

  return normalized;
}

function validateSchema(name: string, schema: JSONSchema, value: unknown): void {
  if (schema.enum !== undefined && !schema.enum.some((candidate) => Object.is(candidate, value))) {
    throw new ValueError(`Arg ${name} must be one of: ${schema.enum.map(String).join(', ')}`);
  }

  switch (schema.type) {
    case undefined:
      return;
    case 'string':
      if (typeof value !== 'string') {
        throw new ValueError(`Arg ${name} must be a string.`);
      }
      return;
    case 'integer':
      if (!Number.isInteger(value)) {
        throw new ValueError(`Arg ${name} must be an integer.`);
      }
      return;
    case 'number':
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new ValueError(`Arg ${name} must be a number.`);
      }
      return;
    case 'boolean':
      if (typeof value !== 'boolean') {
        throw new ValueError(`Arg ${name} must be a boolean.`);
      }
      return;
    case 'array':
      if (!Array.isArray(value)) {
        throw new ValueError(`Arg ${name} must be an array.`);
      }
      return;
    case 'object':
      if (!isPlainObject(value)) {
        throw new ValueError(`Arg ${name} must be an object.`);
      }
      return;
  }
}

function formatArgSchemas(args: Readonly<Record<string, JSONSchema>>): string {
  return JSON.stringify(args, null, 2);
}

export class Tool {
  readonly func: ToolFunction;
  readonly name: string;
  readonly desc: string;
  readonly args: Readonly<Record<string, JSONSchema>>;
  readonly argTypes: Readonly<Record<string, TypeTag>>;
  readonly hasKwargs: boolean;

  readonly #argOrder: readonly string[];
  readonly #invokeMode: InvokeMode;

  constructor(func: ToolFunction, options: ToolOptions = {}) {
    this.func = func;

    let inferredParameters: readonly ParameterSpec[] = Object.freeze([]);
    try {
      inferredParameters = inferParameters(func);
    } catch {
      inferredParameters = Object.freeze([]);
    }

    const argNames = normalizeArgNames(inferredParameters, options.args, options.argTypes);
    const explicitArgMetadataCount = Object.keys(options.args ?? {}).length + Object.keys(options.argTypes ?? {}).length;
    if (argNames.length === 0 && explicitArgMetadataCount > 0) {
      throw new Error('Tool metadata must declare at least one argument name when args or argTypes are provided.');
    }

    const resolvedName = options.name ?? func.name ?? '';
    this.name = resolvedName.trim() === '' ? 'anonymous_tool' : resolvedName;

    this.desc = options.desc ?? '';
    this.args = Object.freeze(normalizeArgs(argNames, options.args, options.argTypes));
    this.argTypes = Object.freeze(normalizeArgTypes(argNames, this.args, options.argTypes));
    this.hasKwargs = false;
    this.#argOrder = argNames;
    this.#invokeMode = inferredParameters.length > 0 ? 'positional' : 'record';
  }

  call(kwargs: Record<string, unknown> = {}): unknown {
    const parsedKwargs = this.validateAndParseArgs(kwargs);
    const result = this.invoke(parsedKwargs);

    if (result instanceof Promise) {
      throw new ValueError(
        'You are calling `call` on an async tool; use `acall` instead.',
      );
    }

    return result;
  }

  async acall(kwargs: Record<string, unknown> = {}): Promise<unknown> {
    const parsedKwargs = this.validateAndParseArgs(kwargs);
    return this.invoke(parsedKwargs);
  }

  formatAsOpenAIFunctionCall(): {
    readonly type: 'function';
    readonly function: {
      readonly name: string;
      readonly description: string;
      readonly parameters: {
        readonly type: 'object';
        readonly properties: Readonly<Record<string, JSONSchema>>;
        readonly required: readonly string[];
      };
    };
  } {
    return {
      type: 'function',
      function: {
        name: this.name,
        description: this.desc,
        parameters: {
          type: 'object',
          properties: this.args,
          required: this.#argOrder,
        },
      },
    };
  }

  toString(): string {
    const description = this.desc === ''
      ? '.'
      : `, whose description is <desc>${this.desc}</desc>.`;
    return `${this.name}${description} It takes arguments ${formatArgSchemas(this.args)}.`;
  }

  private validateAndParseArgs(kwargs: Record<string, unknown>): Record<string, unknown> {
    if (!isPlainObject(kwargs)) {
      throw new ValueError('Tool kwargs must be a plain object.');
    }

    const parsedKwargs: Record<string, unknown> = {};
    for (const [name, value] of Object.entries(kwargs)) {
      if (!(name in this.args)) {
        throw new ValueError(`Arg ${name} is not in the tool's args.`);
      }

      const parsedValue = coerceFieldValue(this.argTypes[name]!, value);
      validateSchema(name, this.args[name]!, parsedValue);
      parsedKwargs[name] = parsedValue;
    }

    return snapshotRecord(parsedKwargs);
  }

  private invoke(kwargs: Record<string, unknown>): unknown {
    if (this.#invokeMode === 'record') {
      return this.func(snapshotRecord(kwargs));
    }

    const positionalArgs = this.#argOrder.map((name) => kwargs[name]);
    return this.func(...positionalArgs);
  }
}

function normalizeToolCall(value: unknown): ToolCall {
  if (!isPlainObject(value)) {
    throw new ValueError(`Received invalid value for ToolCalls: ${String(value)}`);
  }

  const name = value.name;
  const args = value.args;
  if (typeof name !== 'string' || !isPlainObject(args)) {
    throw new ValueError(`Received invalid value for ToolCalls: ${JSON.stringify(value)}`);
  }

  return Object.freeze({
    name,
    args: snapshotRecord(args),
  });
}

export class ToolCalls {
  readonly toolCalls: readonly ToolCall[];

  constructor(toolCalls: readonly ToolCall[]) {
    this.toolCalls = Object.freeze(toolCalls.map((toolCall) => normalizeToolCall(toolCall)));
  }

  static from(value: unknown): ToolCalls {
    if (value instanceof ToolCalls) {
      return value;
    }

    if (Array.isArray(value)) {
      return new ToolCalls(value.map((toolCall) => normalizeToolCall(toolCall)));
    }

    if (isPlainObject(value)) {
      if ('tool_calls' in value && Array.isArray(value.tool_calls)) {
        return new ToolCalls(value.tool_calls.map((toolCall) => normalizeToolCall(toolCall)));
      }

      if ('toolCalls' in value && Array.isArray(value.toolCalls)) {
        return new ToolCalls(value.toolCalls.map((toolCall) => normalizeToolCall(toolCall)));
      }

      if ('name' in value && 'args' in value) {
        return new ToolCalls([normalizeToolCall(value)]);
      }
    }

    throw new ValueError(`Received invalid value for ToolCalls: ${JSON.stringify(value)}`);
  }

  format(): { tool_calls: ReadonlyArray<{ readonly type: 'function'; readonly function: { readonly name: string; readonly arguments: Record<string, unknown> } }> } {
    return {
      tool_calls: this.toolCalls.map((toolCall) => ({
        type: 'function' as const,
        function: {
          name: toolCall.name,
          arguments: serializeOwnedValue(toolCall.args) as Record<string, unknown>,
        },
      })),
    };
  }

  toDict(): { tool_calls: readonly ToolCall[] } {
    return {
      tool_calls: this.toolCalls.map((toolCall) => ({
        name: toolCall.name,
        args: serializeOwnedValue(toolCall.args) as Record<string, unknown>,
      })),
    };
  }

  toJSON(): { tool_calls: readonly ToolCall[] } {
    return this.toDict();
  }
}
