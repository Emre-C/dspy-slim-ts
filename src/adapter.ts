/**
 * §5 — Adapter contract and message assembly.
 */

import { coerceBoolean, coerceJsonContainer, coerceNumber } from './codec.js';
import type { ContentPart, Message } from './chat_message.js';
import type { Callback } from './callback.js';
import { runWithCallbacks } from './callback.js';
import { ConfigurationError, RuntimeError, ValueError } from './exceptions.js';
import { Example } from './example.js';
import type { Field } from './field.js';
import { isPlainObject } from './guards.js';
import { isHistoryLike } from './history.js';
import { Image, isImage } from './image.js';
import type { BaseLM, LMOutput } from './lm.js';
import {
  serializeOwnedValue,
  snapshotOwnedValue,
  snapshotRecord,
} from './owned_value.js';
import { resolveProfile } from './providers/index.js';
import { deleteField, Signature, signatureString } from './signature.js';
import { Tool, ToolCalls } from './tool.js';
import type { TypeTag } from './types.js';

export type { ContentPart, Message } from './chat_message.js';

export type Demo = Example | Record<string, unknown>;

export interface AdapterOptions {
  readonly callbacks?: readonly Callback[];
  readonly useNativeFunctionCalling?: boolean;
}

interface AdapterCallPreprocessResult {
  readonly signature: Signature;
  readonly inputs: Record<string, unknown>;
  readonly lmKwargs: Record<string, unknown>;
  readonly toolOutputFieldName: string | null;
}

const FIELD_HEADER_RE = /^\[\[ ## (\w+) ## \]\]/;

/**
 * A field declared `optional[T]` may be omitted by the LM in its
 * response. `Field.typeTag === 'optional'` is the discriminator;
 * `Field.typeArgs[0]` carries the inner type when the signature
 * string used a bracketed form. Fields declared as bare `optional`
 * (no brackets) still read as optional at the adapter layer but are
 * coerced via the opaque `optional` branch (no inner coercion).
 */
function isOptionalField(field: Field): boolean {
  return field.typeTag === 'optional';
}

/**
 * The tag the adapter should use when rendering placeholders or
 * coercing parsed values for a field. Unwraps one level of
 * `optional[T]` so `<string (optional)>` / `coerceNumber` work as
 * callers expect; leaves every other tag unchanged.
 */
function effectiveTypeTag(field: Field): TypeTag {
  if (field.typeTag === 'optional' && field.typeArgs.length > 0) {
    return field.typeArgs[0]!;
  }
  return field.typeTag;
}

function describeField(field: Field): string {
  const description = field.description.trim();
  const suffix = description === '' ? field.prefix : description;
  const optionalSuffix = isOptionalField(field) ? ', optional' : '';
  const innerTag = isOptionalField(field) && field.typeArgs.length > 0
    ? field.typeArgs[0]!
    : field.typeTag;
  return `- \`${field.name}\` (${innerTag}${optionalSuffix}): ${suffix}`;
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
      return '<any>';
    case 'union':
      return '<union>';
    case 'custom':
      return '<custom>';
    case 'image':
      return '<image>';
  }
}

/**
 * Placeholder variant that takes a whole `Field` so it can surface
 * optionality to the LM (`<string (optional)>`). Prefer this over
 * `placeholderForType(field.typeTag)` at every site that renders the
 * signature-structure block.
 */
function placeholderForField(field: Field): string {
  const inner = placeholderForType(effectiveTypeTag(field));
  if (!isOptionalField(field)) {
    return inner;
  }
  // Strip trailing '>' so the optional marker ends up inside the
  // angle-bracket envelope: '<string (optional)>'.
  if (inner.startsWith('<') && inner.endsWith('>')) {
    return `${inner.slice(0, -1)} (optional)>`;
  }
  return `${inner} (optional)`;
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

function historyFieldName(
  signature: Signature,
  inputs: Record<string, unknown>,
): string | null {
  for (const [name] of signature.inputFields) {
    if (isHistoryLike(inputs[name])) {
      return name;
    }
  }

  return null;
}

function parseFieldValue(field: Field, value: unknown): unknown {
  // Null / undefined on an optional field is a valid "absent" value.
  // When present, recurse into the inner type so `optional[int]` coerces
  // a JSON string like "42" to 42, matching the inner-type contract.
  if (isOptionalField(field)) {
    if (value === null || value === undefined) {
      return null;
    }
    return coerceByTypeTag(effectiveTypeTag(field), value);
  }
  return coerceByTypeTag(field.typeTag, value);
}

function coerceByTypeTag(typeTag: TypeTag, value: unknown): unknown {
  switch (typeTag) {
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
    case 'image':
      return snapshotOwnedValue(value);
  }
}

/**
 * Validate a parsed LM response's output-field key set against the
 * signature. Semantics:
 *
 * - Every required (non-optional) output field must appear in
 *   `actualKeys`.
 * - Every key in `actualKeys` must be a declared output field.
 * - The relative order of `actualKeys` must match declaration order
 *   (a subsequence, not a prefix). Optional fields that are absent
 *   do not break ordering; when they *are* present, they must appear
 *   in their declared position relative to other declared fields.
 *
 * The error message quotes the full declared field list (with
 * required/optional markers) so both humans and the LM's retry can
 * see what shape was expected.
 */
function validateParsedOutputKeys(
  actualKeys: readonly string[],
  signature: Signature,
): void {
  const expectedOrder: string[] = [];
  const optional = new Set<string>();
  for (const [name, field] of signature.outputFields) {
    expectedOrder.push(name);
    if (isOptionalField(field)) {
      optional.add(name);
    }
  }

  const expectedSet = new Set(expectedOrder);
  const describe = (): string =>
    expectedOrder
      .map((name) => (optional.has(name) ? `${name} (optional)` : name))
      .join(', ');

  for (const actual of actualKeys) {
    if (!expectedSet.has(actual)) {
      throw new ValueError(
        `Unexpected output field "${actual}". Declared outputs: ${describe()}.`,
      );
    }
  }

  const actualSet = new Set(actualKeys);
  for (const name of expectedOrder) {
    if (!actualSet.has(name) && !optional.has(name)) {
      throw new ValueError(
        `Missing required output field "${name}". Declared outputs: ${describe()}; received: ${actualKeys.join(', ')}.`,
      );
    }
  }

  let cursor = 0;
  for (const actual of actualKeys) {
    let advance = cursor;
    while (advance < expectedOrder.length && expectedOrder[advance] !== actual) {
      advance += 1;
    }
    if (advance >= expectedOrder.length) {
      throw new ValueError(
        `Output field "${actual}" is out of declaration order. Declared: ${describe()}; received: ${actualKeys.join(', ')}.`,
      );
    }
    cursor = advance + 1;
  }
}

function extractLmOutputText(output: LMOutput): string {
  if (typeof output === 'string') {
    return output;
  }

  return output.text;
}

function isToolOutputEnvelope(output: LMOutput): output is Exclude<LMOutput, string> {
  return typeof output !== 'string';
}

function outputToolFieldName(signature: Signature): string | null {
  if (signature.outputFields.has('tool_calls')) {
    return 'tool_calls';
  }

  if (signature.outputFields.has('toolCalls')) {
    return 'toolCalls';
  }

  return null;
}

function normalizeNativeTools(value: unknown): readonly Tool[] | null {
  if (value instanceof Tool) {
    return Object.freeze([value]);
  }

  if (Array.isArray(value) && value.length > 0 && value.every((item) => item instanceof Tool)) {
    return Object.freeze([...value]);
  }

  return null;
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

function extractAllJsonObjects(source: string): string[] {
  const results: string[] = [];
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
        results.push(source.slice(start, index + 1));
        start = -1;
      }
    }
  }

  return results;
}

function extractMarkdownJsonBlocks(source: string): string[] {
  const blocks: string[] = [];
  const regex = /```(?:json)?\s*([\s\S]*?)\s*```/ig;
  let match;
  while ((match = regex.exec(source)) !== null) {
    if (match[1]) {
      blocks.push(match[1].trim());
    }
  }
  return blocks;
}

function parseLooseJsonObject(source: string): Record<string, unknown> | null {
  const candidates = new Set<string>();

  const markdownBlocks = extractMarkdownJsonBlocks(source);
  for (let i = markdownBlocks.length - 1; i >= 0; i -= 1) {
    const block = markdownBlocks[i]!;
    candidates.add(block);
    const objs = extractAllJsonObjects(block);
    for (let j = objs.length - 1; j >= 0; j -= 1) {
      candidates.add(objs[j]!);
    }
  }

  const extracted = extractAllJsonObjects(source);
  for (let i = extracted.length - 1; i >= 0; i -= 1) {
    candidates.add(extracted[i]!);
  }

  const trimmed = source.trim();
  if (trimmed !== '') {
    candidates.add(trimmed);
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

export class AdapterParseError extends RuntimeError {
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
  readonly callbacks: readonly Callback[];
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
    const processed = this.preprocessCall(lm, lmKwargs, signature, inputs);
    const messages = this.formatWithCallbacks(processed.signature, demos, processed.inputs);
    const parseOutputs = (currentLmKwargs: Record<string, unknown>): Record<string, unknown>[] => {
      const outputs = lm.call(undefined, messages, currentLmKwargs);
      return outputs.map((output) => this.postprocessOutput(
        processed.signature,
        signature,
        processed.toolOutputFieldName,
        output,
      ));
    };

    try {
      return parseOutputs(processed.lmKwargs);
    } catch (error) {
      const profile = resolveProfile(lm.model);
      const retryKwargs = profile?.adapterRetry?.(lm, processed.lmKwargs, error) ?? null;
      if (retryKwargs === null) {
        throw error;
      }
      return parseOutputs(retryKwargs);
    }
  }

  async acall(
    lm: BaseLM,
    lmKwargs: Record<string, unknown>,
    signature: Signature,
    demos: readonly Demo[],
    inputs: Record<string, unknown>,
  ): Promise<Record<string, unknown>[]> {
    const processed = this.preprocessCall(lm, lmKwargs, signature, inputs);
    const messages = this.formatWithCallbacks(processed.signature, demos, processed.inputs);
    const parseOutputs = async (currentLmKwargs: Record<string, unknown>): Promise<Record<string, unknown>[]> => {
      const outputs = await lm.acall(undefined, messages, currentLmKwargs);
      return outputs.map((output) => this.postprocessOutput(
        processed.signature,
        signature,
        processed.toolOutputFieldName,
        output,
      ));
    };

    try {
      return await parseOutputs(processed.lmKwargs);
    } catch (error) {
      const profile = resolveProfile(lm.model);
      const retryKwargs = profile?.adapterRetry?.(lm, processed.lmKwargs, error) ?? null;
      if (retryKwargs === null) {
        throw error;
      }
      return parseOutputs(retryKwargs);
    }
  }

  format(
    signature: Signature,
    demos: readonly Demo[],
    inputs: Record<string, unknown>,
  ): Message[] {
    const inputsCopy = snapshotRecord(inputs);
    const historyName = historyFieldName(signature, inputsCopy);
    const signatureWithoutHistory = historyName === null ? signature : deleteField(signature, historyName);

    return [
      { role: 'system', content: this.formatSystemMessage(signature) },
      ...this.formatDemos(signature, demos),
      ...this.formatConversationHistory(signatureWithoutHistory, historyName, inputsCopy),
      {
        role: 'user',
        content: this.formatUserMessageContent(signatureWithoutHistory, inputsCopy, '', '', true),
      },
    ];
  }

  /**
   * Returns true iff at least one input field is an `Image` instance.
   * The adapter uses this to decide between the legacy single-string
   * user-message path and the multimodal `ContentPart[]` path that
   * inlines `image_url` parts alongside the marker text.
   */
  protected hasImageInput(
    signature: Signature,
    inputs: Record<string, unknown>,
  ): boolean {
    for (const [name] of signature.inputFields) {
      if (name in inputs && isImage(inputs[name])) {
        return true;
      }
    }
    return false;
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

  /**
   * Build the user-turn content for a single Predict call.
   *
   * - If no input field is an `Image`, returns a single trimmed string
   *   (the historical behavior; preserved verbatim for non-vision flows).
   * - If at least one input field is an `Image`, returns a frozen
   *   `ContentPart[]` interleaving text markers and `image_url` parts.
   *   Images appear immediately after their declaration-order marker,
   *   matching OpenRouter's "text first, image right after its marker"
   *   guidance and keeping the existing structured-output parser intact
   *   (markers like `[[ ## name ## ]]` are preserved as text).
   *
   * `options.elideImages` collapses Image inputs into placeholder text
   * so the string-only path stays usable for demos and conversation
   * history (where mixing real image bytes into examples blows up
   * prompt cost without helping the model).
   */
  formatUserMessageContent(
    signature: Signature,
    inputs: Record<string, unknown>,
    prefix = '',
    suffix = '',
    mainRequest = false,
    options: { readonly elideImages?: boolean } = {},
  ): string | readonly ContentPart[] {
    const elideImages = options.elideImages ?? false;
    const useContentParts = !elideImages && this.hasImageInput(signature, inputs);

    if (!useContentParts) {
      const parts: string[] = [];

      if (prefix.trim() !== '') {
        parts.push(prefix.trim());
      }

      for (const [name] of signature.inputFields) {
        if (name in inputs) {
          const value = inputs[name];
          const rendered = isImage(value) ? '<image elided>' : value;
          parts.push(fieldBlock(name, rendered));
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

    // Per spec §3.2 / §7.1: emit content parts in three passes —
    //   1. one consolidated text block with prefix + every non-Image
    //      input field (in declaration order),
    //   2. for each Image input (in declaration order), a dedicated
    //      `text` marker block followed by its `image_url` part so
    //      images sit immediately after their marker,
    //   3. a final text block with output requirements + suffix.
    // The "text first" pass satisfies OpenRouter's preferred ordering;
    // the per-image marker/image pairing keeps multi-image attribution
    // unambiguous to the model.
    const contentParts: ContentPart[] = [];
    const pushTextIfNonEmpty = (segments: readonly string[]): void => {
      const joined = segments.join('\n\n').trim();
      if (joined === '') {
        return;
      }
      contentParts.push(Object.freeze({ type: 'text' as const, text: joined }));
    };

    const leadingText: string[] = [];
    if (prefix.trim() !== '') {
      leadingText.push(prefix.trim());
    }
    for (const [name] of signature.inputFields) {
      if (!(name in inputs)) {
        continue;
      }
      const value = inputs[name];
      if (!isImage(value)) {
        leadingText.push(fieldBlock(name, value));
      }
    }
    pushTextIfNonEmpty(leadingText);

    for (const [name] of signature.inputFields) {
      if (!(name in inputs)) {
        continue;
      }
      const value = inputs[name];
      if (!isImage(value)) {
        continue;
      }
      contentParts.push(Object.freeze({
        type: 'text' as const,
        text: `[[ ## ${name} ## ]]`,
      }));
      contentParts.push(Object.freeze({
        type: 'image_url' as const,
        image_url: Object.freeze({ url: (value as Image).toDataUri() }),
      }));
    }

    const trailingText: string[] = [];
    if (mainRequest) {
      trailingText.push(this.userMessageOutputRequirements(signature));
    }
    if (suffix.trim() !== '') {
      trailingText.push(suffix.trim());
    }
    pushTextIfNonEmpty(trailingText);

    return Object.freeze(contentParts);
  }

  protected userMessageOutputRequirements(signature: Signature): string {
    const fields = [...signature.outputFields.keys()].map((name) => `\`[[ ## ${name} ## ]]\``);
    const optionalNames = [...signature.outputFields.values()]
      .filter(isOptionalField)
      .map((field) => `\`${field.name}\``);
    const base = `Respond with the corresponding output fields, starting with ${fields.join(', then ')}, and then ending with the marker for \`[[ ## completed ## ]]\`.`;
    if (optionalNames.length === 0) {
      return base;
    }
    return `${base} ${optionalNames.join(', ')} ${optionalNames.length === 1 ? 'is' : 'are'} optional and may be omitted.`;
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
        content: this.formatUserMessageContent(
          signature,
          demo,
          incompletePrefix,
          '',
          false,
          { elideImages: true },
        ),
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
        content: this.formatUserMessageContent(
          signature,
          demo,
          '',
          '',
          false,
          { elideImages: true },
        ),
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

  formatConversationHistory(
    signature: Signature,
    historyFieldNameValue: string | null,
    inputs: Record<string, unknown>,
  ): Message[] {
    if (historyFieldNameValue === null) {
      return [];
    }

    const history = inputs[historyFieldNameValue];
    if (!isHistoryLike(history)) {
      return [];
    }

    const messages: Message[] = [];
    for (const entry of history.messages) {
      messages.push({
        role: 'user',
        content: this.formatUserMessageContent(
          signature,
          entry,
          '',
          '',
          false,
          { elideImages: true },
        ),
      });
      messages.push({
        role: 'assistant',
        content: this.formatAssistantMessageContent(signature, entry),
      });
    }

    delete inputs[historyFieldNameValue];
    return messages;
  }

  abstract parse(signature: Signature, completion: string): Record<string, unknown>;

  private formatWithCallbacks(
    signature: Signature,
    demos: readonly Demo[],
    inputs: Record<string, unknown>,
  ): Message[] {
    return runWithCallbacks({
      kind: 'adapter_format',
      instance: this,
      inputs: snapshotRecord({ signature: signatureString(signature), inputs, demos }),
      execute: () => this.format(signature, demos, inputs),
    });
  }

  private parseWithCallbacks(signature: Signature, completion: string): Record<string, unknown> {
    return runWithCallbacks({
      kind: 'adapter_parse',
      instance: this,
      inputs: { signature: signatureString(signature), completion },
      execute: () => this.parse(signature, completion),
    });
  }

  private preprocessCall(
    lm: BaseLM,
    lmKwargs: Record<string, unknown>,
    signature: Signature,
    inputs: Record<string, unknown>,
  ): AdapterCallPreprocessResult {
    const nextInputs = snapshotRecord(inputs);
    const nextLmKwargs = snapshotRecord(lmKwargs);
    let processedSignature = signature;
    const toolOutputFieldName = outputToolFieldName(signature);

    if (!this.useNativeFunctionCalling) {
      return {
        signature: processedSignature,
        inputs: nextInputs,
        lmKwargs: nextLmKwargs,
        toolOutputFieldName: null,
      };
    }

    let toolInputFieldName: string | null = null;
    let tools = normalizeNativeTools(nextLmKwargs.tools);

    if (tools === null) {
      for (const [name] of signature.inputFields) {
        const candidate = normalizeNativeTools(nextInputs[name]);
        if (candidate !== null) {
          toolInputFieldName = name;
          tools = candidate;
          break;
        }
      }
    }

    if (tools === null) {
      return {
        signature: processedSignature,
        inputs: nextInputs,
        lmKwargs: nextLmKwargs,
        toolOutputFieldName: null,
      };
    }

    if (!lm.supportsFunctionCalling) {
      throw new ConfigurationError('Native function calling requires an LM that supports function calling.');
    }

    if (toolOutputFieldName === null) {
      throw new ValueError('Native function calling requires an output field named tool_calls or toolCalls.');
    }

    nextLmKwargs.tools = tools.map((tool) => tool.formatAsOpenAIFunctionCall());

    if (toolInputFieldName !== null) {
      processedSignature = deleteField(processedSignature, toolInputFieldName);
      delete nextInputs[toolInputFieldName];
    }
    processedSignature = deleteField(processedSignature, toolOutputFieldName);

    return {
      signature: processedSignature,
      inputs: nextInputs,
      lmKwargs: nextLmKwargs,
      toolOutputFieldName,
    };
  }

  private postprocessOutput(
    processedSignature: Signature,
    originalSignature: Signature,
    toolOutputFieldName: string | null,
    output: LMOutput,
  ): Record<string, unknown> {
    const parsed: Record<string, unknown> = {};
    const text = extractLmOutputText(output).trim();
    const toolCalls = isToolOutputEnvelope(output) ? output.toolCalls : undefined;

    if (text !== '') {
      Object.assign(parsed, this.parseWithCallbacks(processedSignature, text));
    } else if (!(toolCalls && toolCalls.length > 0)) {
      throw new AdapterParseError({
        adapterName: this.constructor.name,
        signature: originalSignature,
        completion: text,
        message: 'The LM returned an empty or null response.',
      });
    }

    for (const [name] of originalSignature.outputFields) {
      if (!(name in parsed)) {
        parsed[name] = null;
      }
    }

    if (toolOutputFieldName !== null) {
      parsed[toolOutputFieldName] = toolCalls ? ToolCalls.from(toolCalls) : null;
    }

    if (isToolOutputEnvelope(output) && output.logprobs !== undefined) {
      parsed.logprobs = output.logprobs;
    }

    if (isToolOutputEnvelope(output) && output.citations !== undefined) {
      parsed.citations = snapshotOwnedValue(output.citations);
    }

    return Object.freeze({ ...parsed });
  }
}

export class ChatAdapter extends Adapter {
  constructor(options: AdapterOptions = {}) {
    super(options);
  }

  override formatFieldStructure(signature: Signature): string {
    const parts = [
      'All interactions will be structured in the following way, with the appropriate values filled in.',
      ...[...signature.inputFields.values()].map((field) => fieldBlock(field.name, placeholderForField(field))),
      ...[...signature.outputFields.values()].map((field) => fieldBlock(field.name, placeholderForField(field))),
      '[[ ## completed ## ]]',
    ];

    return parts.join('\n\n').trim();
  }

  override parse(signature: Signature, completion: string): Record<string, unknown> {
    const sections = new Map<string, string>();

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
      validateParsedOutputKeys(actualKeys, signature);
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
      fieldBlock(field.name, placeholderForField(field))
    ));

    const outputShape = Object.fromEntries(
      [...signature.outputFields.values()].map((field) => [
        field.name,
        placeholderForField(field),
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
    const optionalNames = [...signature.outputFields.values()]
      .filter(isOptionalField)
      .map((field) => `\`${field.name}\``);
    const base = `Respond with a JSON object in the following order of fields: ${fields.join(', then ')}.`;
    if (optionalNames.length === 0) {
      return base;
    }
    return `${base} ${optionalNames.join(', ')} ${optionalNames.length === 1 ? 'is' : 'are'} optional and may be omitted.`;
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

    try {
      validateParsedOutputKeys(actualKeys, signature);
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
