/**
 * §6 — Language model contract and OpenAI-compatible runtime.
 */

import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import type { Message } from './adapter.js';
import type { Callback } from './callback.js';
import { runWithCallbacks, runWithCallbacksAsync } from './callback.js';
import { ConfigurationError, ContextWindowExceededError, RuntimeError } from './exceptions.js';
import { isPlainObject } from './guards.js';
import { snapshotOwnedValue, snapshotRecord } from './owned_value.js';
import { settings } from './settings.js';
import type { ModelType } from './types.js';

const GLOBAL_HISTORY_MAX_SIZE = 10_000;
const REASONING_MODEL_PATTERN = /^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$/;
const RESPONSE_FORMAT_PARAMS = Object.freeze(new Set(['response_format']));
const RETRYABLE_STATUS_CODES = new Set([408, 409, 429, 500, 502, 503, 504]);
const RETRYABLE_ERROR_CODES = new Set(['rate_limit_exceeded', 'server_error', 'temporarily_unavailable']);
const GLOBAL_HISTORY: HistoryEntry[] = [];

export interface BaseLMOptions {
  readonly model: string;
  readonly modelType?: ModelType;
  readonly cache?: boolean;
  readonly kwargs?: Record<string, unknown>;
}

export interface ToolCallWire {
  readonly id?: string | undefined;
  readonly type?: 'function' | undefined;
  readonly function: {
    readonly name: string;
    readonly arguments: string;
  };
}

export interface LMOutputEnvelope {
  readonly text: string;
  readonly logprobs?: unknown;
  readonly citations?: readonly unknown[] | undefined;
  readonly toolCalls?: readonly ToolCallWire[] | undefined;
}

export type LMOutput = string | LMOutputEnvelope;

export interface ChatCompletionChoice {
  readonly message?: {
    readonly content?: string | null;
    readonly refusal?: string | null;
    readonly tool_calls?: readonly ToolCallWire[];
    readonly provider_specific_fields?: {
      readonly citations?: readonly unknown[] | readonly unknown[][];
    };
  };
  readonly text?: string | null;
  readonly finish_reason?: string | null;
  readonly logprobs?: unknown;
}

export interface ChatCompletionResponse {
  readonly choices: readonly ChatCompletionChoice[];
  readonly usage?: Record<string, number | undefined>;
  readonly model: string;
}

export interface ResponsesMessageContentItem {
  readonly text?: string;
  readonly annotations?: readonly unknown[];
}

export interface ResponsesOutputItem {
  readonly type: 'message' | 'function_call' | string;
  readonly content?: readonly ResponsesMessageContentItem[];
  readonly name?: string;
  readonly arguments?: string;
  readonly call_id?: string;
  readonly id?: string;
}

export interface ResponsesResponse {
  readonly output: readonly ResponsesOutputItem[];
  readonly usage?: Record<string, number | undefined>;
  readonly model: string;
}

export type LMResponse = ChatCompletionResponse | ResponsesResponse;

export interface HistoryEntry {
  readonly prompt: string | null;
  readonly messages: readonly Message[] | null;
  readonly kwargs: Record<string, unknown>;
  readonly response: unknown;
  readonly outputs: readonly LMOutput[];
  readonly usage: Record<string, number>;
  readonly cost: number | null;
  readonly timestamp: string;
  readonly uuid: string;
  readonly model: string;
  readonly responseModel: string;
  readonly modelType: ModelType;
}

export interface LMOptions extends BaseLMOptions {
  readonly apiKey?: string | undefined;
  readonly baseURL?: string | undefined;
  readonly apiBase?: string | undefined;
  readonly organization?: string | undefined;
  readonly project?: string | undefined;
  readonly headers?: Readonly<Record<string, string>> | undefined;
  readonly numRetries?: number;
  readonly useDeveloperRole?: boolean;
  readonly callbacks?: readonly Callback[] | undefined;
  readonly fetch?: typeof globalThis.fetch | undefined;
}

interface TransportOptions {
  readonly apiKey: string;
  readonly baseURL: string;
  readonly organization?: string | undefined;
  readonly project?: string | undefined;
  readonly headers: Record<string, string>;
  readonly request: Record<string, unknown>;
}

class OpenAIRequestError extends Error {
  readonly status: number | null;
  readonly code: string | null;
  readonly body: unknown;

  constructor(message: string, options: { readonly status?: number | null; readonly code?: string | null; readonly body?: unknown } = {}) {
    super(message);
    this.name = 'OpenAIRequestError';
    this.status = options.status ?? null;
    this.code = options.code ?? null;
    this.body = options.body;
  }
}

function providerNameFromModel(model: string): string {
  if (model.includes('/')) {
    return model.split('/', 1)[0]!.toLowerCase();
  }

  return 'openai';
}

function providerModelName(model: string): string {
  const provider = providerNameFromModel(model);
  if ((provider === 'openai' || provider === 'openrouter') && model.includes('/')) {
    return model.split('/').slice(1).join('/');
  }

  return model;
}

function providerEnvName(provider: string, suffix: string): string {
  return `${provider.toUpperCase()}_${suffix}`;
}

function isChatCompletionResponse(value: unknown): value is ChatCompletionResponse {
  return isPlainObject(value) && Array.isArray(value.choices) && typeof value.model === 'string';
}

function isResponsesResponse(value: unknown): value is ResponsesResponse {
  return isPlainObject(value) && Array.isArray(value.output) && typeof value.model === 'string';
}

function looksLikeLMOutputList(value: unknown): value is readonly LMOutput[] {
  return Array.isArray(value) && value.every((item) => (
    typeof item === 'string'
      || (isPlainObject(item) && typeof item.text === 'string')
  ));
}

function normalizeMessages(prompt?: string, messages?: readonly Message[]): readonly Message[] {
  if (messages !== undefined) {
    return messages;
  }

  return [{ role: 'user', content: prompt ?? '' }];
}

function usageRecord(value: unknown): Record<string, number> {
  if (!isPlainObject(value)) {
    return {};
  }

  const usage: Record<string, number> = {};
  for (const [key, item] of Object.entries(value)) {
    if (typeof item === 'number' && Number.isFinite(item)) {
      usage[key] = item;
    }
  }

  const completionTokenDetails = isPlainObject(value.completion_tokens_details)
    ? value.completion_tokens_details
    : null;
  if (
    completionTokenDetails !== null
    && typeof completionTokenDetails.reasoning_tokens === 'number'
    && Number.isFinite(completionTokenDetails.reasoning_tokens)
  ) {
    usage.reasoning_tokens = completionTokenDetails.reasoning_tokens;
  }

  return usage;
}

function flattenCitations(value: unknown): readonly unknown[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const flattened: unknown[] = [];
  for (const item of value) {
    if (Array.isArray(item)) {
      flattened.push(...item);
    } else {
      flattened.push(item);
    }
  }

  return flattened.length === 0 ? undefined : Object.freeze(flattened.map((item) => snapshotOwnedValue(item)));
}

function processChatCompletion(
  response: ChatCompletionResponse,
  mergedKwargs: Record<string, unknown>,
): readonly LMOutput[] {
  const outputs = response.choices.map((choice) => {
    const text = choice.message?.content ?? choice.message?.refusal ?? choice.text ?? '';
    const citations = flattenCitations(choice.message?.provider_specific_fields?.citations);
    const envelope: LMOutputEnvelope = {
      text,
      ...(mergedKwargs.logprobs ? { logprobs: choice.logprobs } : {}),
      ...(choice.message?.tool_calls ? { toolCalls: Object.freeze([...choice.message.tool_calls]) } : {}),
      ...(citations ? { citations } : {}),
    };

    return envelope;
  });

  if (outputs.every((output) => (
    output.logprobs === undefined
      && output.citations === undefined
      && output.toolCalls === undefined
  ))) {
    return Object.freeze(outputs.map((output) => output.text));
  }

  return Object.freeze(outputs.map((output) => Object.freeze(output)));
}

function processResponses(response: ResponsesResponse): readonly LMOutput[] {
  const textParts: string[] = [];
  const toolCalls: ToolCallWire[] = [];
  const citations: unknown[] = [];

  for (const output of response.output) {
    if (output.type === 'message') {
      for (const content of output.content ?? []) {
        if (typeof content.text === 'string') {
          textParts.push(content.text);
        }
        if (Array.isArray(content.annotations)) {
          citations.push(...content.annotations.map((annotation) => snapshotOwnedValue(annotation)));
        }
      }
      continue;
    }

    if (output.type === 'function_call' && typeof output.name === 'string') {
      toolCalls.push(Object.freeze({
        ...(output.call_id ?? output.id ? { id: output.call_id ?? output.id } : {}),
        type: 'function' as const,
        function: Object.freeze({
          name: output.name,
          arguments: output.arguments ?? '{}',
        }),
      }));
    }
  }

  const result: LMOutputEnvelope = {
    text: textParts.join(''),
    ...(toolCalls.length > 0 ? { toolCalls: Object.freeze(toolCalls) } : {}),
    ...(citations.length > 0 ? { citations: Object.freeze(citations) } : {}),
  };

  return Object.freeze([Object.freeze(result)]);
}

function processLMResponse(
  modelType: ModelType,
  response: LMResponse | readonly LMOutput[],
  mergedKwargs: Record<string, unknown>,
): readonly LMOutput[] {
  if (looksLikeLMOutputList(response)) {
    return Object.freeze(response.map((output) => (
      typeof output === 'string'
        ? output
        : Object.freeze({
          ...output,
          ...(output.toolCalls ? { toolCalls: Object.freeze([...output.toolCalls]) } : {}),
          ...(output.citations ? { citations: Object.freeze([...output.citations]) } : {}),
        })
    )));
  }

  if (modelType === 'responses' || isResponsesResponse(response)) {
    return processResponses(response as ResponsesResponse);
  }

  return processChatCompletion(response as ChatCompletionResponse, mergedKwargs);
}

function responseModelName(model: string, response: unknown): string {
  if ((isChatCompletionResponse(response) || isResponsesResponse(response)) && typeof response.model === 'string') {
    return response.model;
  }

  return model;
}

function scrubHistoryKwargs(kwargs: Record<string, unknown>): Record<string, unknown> {
  return snapshotRecord(Object.fromEntries(
    Object.entries(kwargs).filter(([key]) => !key.startsWith('api_') && key !== 'apiKey'),
  ));
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function errorCodeFromBody(body: unknown): string | null {
  if (!isPlainObject(body)) {
    return null;
  }

  const error = isPlainObject(body.error) ? body.error : body;
  return typeof error.code === 'string' ? error.code : null;
}

function errorMessageFromBody(body: unknown): string {
  if (!isPlainObject(body)) {
    return typeof body === 'string' ? body : 'OpenAI-compatible request failed.';
  }

  const error = isPlainObject(body.error) ? body.error : body;
  if (typeof error.message === 'string') {
    return error.message;
  }

  return 'OpenAI-compatible request failed.';
}

function retryableError(error: unknown): boolean {
  if (!(error instanceof OpenAIRequestError)) {
    return true;
  }

  if (error.status === null && error.code === null) {
    return true;
  }

  return (
    (error.status !== null && RETRYABLE_STATUS_CODES.has(error.status))
    || (error.code !== null && RETRYABLE_ERROR_CODES.has(error.code))
  );
}

function looksLikeContextWindowExceeded(error: unknown): boolean {
  if (error instanceof ContextWindowExceededError) {
    return true;
  }

  if (!(error instanceof Error)) {
    return false;
  }

  const code = error instanceof OpenAIRequestError ? error.code : null;
  if (code === 'context_length_exceeded' || code === 'string_above_max_length') {
    return true;
  }

  const message = error.message.toLowerCase();
  return [
    'context length',
    'context window',
    'maximum context length',
    'maximum context window',
    'prompt is too long',
    'too many tokens',
    'reduce the length',
  ].some((phrase) => message.includes(phrase));
}

function resolveApiKey(provider: string, request: Record<string, unknown>, defaults: LMOptions): string {
  const requestApiKey = request.apiKey ?? request.api_key;
  delete request.apiKey;
  delete request.api_key;

  const apiKey = typeof requestApiKey === 'string'
    ? requestApiKey
    : defaults.apiKey ?? process.env[providerEnvName(provider, 'API_KEY')];

  if (!apiKey) {
    throw new ConfigurationError(`Missing API key for provider ${provider}. Set ${providerEnvName(provider, 'API_KEY')} or pass apiKey.`);
  }

  return apiKey;
}

function resolveBaseURL(provider: string, request: Record<string, unknown>, defaults: LMOptions): string {
  const requestBaseURL = request.baseURL ?? request.baseUrl ?? request.base_url ?? request.apiBase ?? request.api_base;
  delete request.baseURL;
  delete request.baseUrl;
  delete request.base_url;
  delete request.apiBase;
  delete request.api_base;

  const configured = typeof requestBaseURL === 'string'
    ? requestBaseURL
    : defaults.baseURL
      ?? defaults.apiBase
      ?? process.env[providerEnvName(provider, 'BASE_URL')]
      ?? process.env[providerEnvName(provider, 'API_BASE')]
      ?? (provider === 'openrouter' ? 'https://openrouter.ai/api/v1' : 'https://api.openai.com/v1');

  return configured.replace(/\/$/, '');
}

function mergeExtraBody(request: Record<string, unknown>): Record<string, unknown> {
  const extraBody = request.extra_body ?? request.extraBody;
  const merged = { ...request };
  delete merged.extra_body;
  delete merged.extraBody;

  if (!isPlainObject(extraBody)) {
    return merged;
  }

  return {
    ...merged,
    ...snapshotRecord(extraBody),
  };
}

function isOpenRouterMinimaxModel(model: string): boolean {
  return providerNameFromModel(model) === 'openrouter'
    && providerModelName(model).toLowerCase().startsWith('minimax/');
}

function applyOpenRouterMinimaxReasoningDefaults(
  model: string,
  request: Record<string, unknown>,
): Record<string, unknown> {
  if (!isOpenRouterMinimaxModel(model)) {
    return request;
  }

  if (request.reasoning !== undefined) {
    return request;
  }

  return {
    ...request,
    reasoning: {
      exclude: true,
    },
  };
}

function applyOpenRouterMinimaxOutputFloor(request: Record<string, unknown>): Record<string, unknown> {
  const minimumOutputTokens = 4096;
  const normalized = { ...request };

  if (typeof normalized.max_tokens === 'number' && Number.isFinite(normalized.max_tokens)) {
    normalized.max_tokens = Math.max(normalized.max_tokens, minimumOutputTokens);
    return normalized;
  }

  if (typeof normalized.max_output_tokens === 'number' && Number.isFinite(normalized.max_output_tokens)) {
    normalized.max_output_tokens = Math.max(normalized.max_output_tokens, minimumOutputTokens);
    return normalized;
  }

  normalized.max_tokens = minimumOutputTokens;
  return normalized;
}

function applyOpenRouterMinimaxRequestDefaults(
  model: string,
  request: Record<string, unknown>,
): Record<string, unknown> {
  if (!isOpenRouterMinimaxModel(model)) {
    return request;
  }

  return applyOpenRouterMinimaxOutputFloor(
    applyOpenRouterMinimaxReasoningDefaults(model, request),
  );
}

function convertContentPartForResponses(part: Record<string, unknown>): Record<string, unknown> {
  if (part.type === 'image_url' && isPlainObject(part.image_url)) {
    return {
      type: 'input_image',
      image_url: part.image_url.url,
    };
  }

  if (part.type === 'file' && isPlainObject(part.file)) {
    return {
      type: 'input_file',
      file_data: part.file.file_data,
      filename: part.file.filename,
      file_id: part.file.file_id,
    };
  }

  return {
    type: 'input_text',
    text: typeof part.text === 'string' ? part.text : '',
  };
}

function convertChatRequestToResponsesRequest(
  request: Record<string, unknown>,
  useDeveloperRole: boolean,
): Record<string, unknown> {
  const converted = { ...request };

  if (Array.isArray(converted.messages)) {
    converted.input = converted.messages.map((message) => {
      if (!isPlainObject(message)) {
        return { role: 'user', content: [] };
      }

      const role = useDeveloperRole && message.role === 'system' ? 'developer' : message.role;
      const content = typeof message.content === 'string'
        ? [{ type: 'input_text', text: message.content }]
        : Array.isArray(message.content)
          ? message.content.map((part) => convertContentPartForResponses(part as Record<string, unknown>))
          : [];

      return { role, content };
    });

    delete converted.messages;
  }

  if (converted.max_completion_tokens !== undefined) {
    converted.max_output_tokens = converted.max_completion_tokens;
    delete converted.max_completion_tokens;
  } else if (converted.max_tokens !== undefined) {
    converted.max_output_tokens = converted.max_tokens;
    delete converted.max_tokens;
  }

  if (converted.response_format !== undefined) {
    const text = isPlainObject(converted.text) ? { ...converted.text } : {};
    text.format = converted.response_format;
    converted.text = text;
    delete converted.response_format;
  }

  return converted;
}

function buildTransportOptions(
  model: string,
  request: Record<string, unknown>,
  defaults: LMOptions,
): TransportOptions {
  const provider = providerNameFromModel(model);
  const requestCopy = { ...request };
  const apiKey = resolveApiKey(provider, requestCopy, defaults);
  const baseURL = resolveBaseURL(provider, requestCopy, defaults);
  const organization = typeof requestCopy.organization === 'string'
    ? requestCopy.organization
    : defaults.organization;
  const project = typeof requestCopy.project === 'string'
    ? requestCopy.project
    : defaults.project;
  delete requestCopy.organization;
  delete requestCopy.project;

  const requestHeaders = isPlainObject(requestCopy.headers)
    ? Object.fromEntries(Object.entries(requestCopy.headers).filter(([, value]) => typeof value === 'string')) as Record<string, string>
    : {};
  delete requestCopy.headers;
  delete requestCopy.rollout_id;

  return {
    apiKey,
    baseURL,
    ...(organization ? { organization } : {}),
    ...(project ? { project } : {}),
    headers: {
      ...(defaults.headers ? { ...defaults.headers } : {}),
      ...requestHeaders,
    },
    request: requestCopy,
  };
}

function parseCurlResponse(rawOutput: string): { readonly status: number; readonly body: unknown } {
  const separatorIndex = rawOutput.lastIndexOf('\n');
  if (separatorIndex === -1) {
    throw new OpenAIRequestError('Malformed curl response from OpenAI-compatible transport.');
  }

  const bodyText = rawOutput.slice(0, separatorIndex);
  const statusText = rawOutput.slice(separatorIndex + 1).trim();
  const status = Number(statusText);
  if (!Number.isInteger(status)) {
    throw new OpenAIRequestError(`Malformed HTTP status from OpenAI-compatible transport: ${statusText}`);
  }

  let body: unknown = bodyText;
  if (bodyText.trim() !== '') {
    try {
      body = JSON.parse(bodyText);
    } catch {
      body = bodyText;
    }
  }

  return { status, body };
}

function runSyncRequest(
  url: string,
  body: Record<string, unknown>,
  options: TransportOptions,
): unknown {
  const headers = {
    Authorization: `Bearer ${options.apiKey}`,
    'Content-Type': 'application/json',
    'User-Agent': 'dspy-slim-ts/0.1.0',
    ...(options.organization ? { 'OpenAI-Organization': options.organization } : {}),
    ...(options.project ? { 'OpenAI-Project': options.project } : {}),
    ...options.headers,
  };

  const args = ['--silent', '--show-error', '-X', 'POST', url];
  for (const [name, value] of Object.entries(headers)) {
    args.push('-H', `${name}: ${value}`);
  }
  args.push('-d', JSON.stringify(body), '-w', '\n%{http_code}');

  try {
    const output = execFileSync('curl', args, { encoding: 'utf8' });
    const { status, body: responseBody } = parseCurlResponse(output);
    if (status < 200 || status >= 300) {
      throw new OpenAIRequestError(errorMessageFromBody(responseBody), {
        status,
        code: errorCodeFromBody(responseBody),
        body: responseBody,
      });
    }

    return responseBody;
  } catch (error) {
    if (error instanceof OpenAIRequestError) {
      throw error;
    }

    if (error instanceof Error) {
      throw new OpenAIRequestError(error.message);
    }

    throw new OpenAIRequestError(String(error));
  }
}

async function runAsyncRequest(
  url: string,
  body: Record<string, unknown>,
  options: TransportOptions,
  fetchImpl: typeof globalThis.fetch,
): Promise<unknown> {
  const response = await fetchImpl(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${options.apiKey}`,
      'Content-Type': 'application/json',
      'User-Agent': 'dspy-slim-ts/0.1.0',
      ...(options.organization ? { 'OpenAI-Organization': options.organization } : {}),
      ...(options.project ? { 'OpenAI-Project': options.project } : {}),
      ...options.headers,
    },
    body: JSON.stringify(body),
  });

  const text = await response.text();
  const parsed = text.trim() === '' ? {} : (() => {
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  })();

  if (!response.ok) {
    throw new OpenAIRequestError(errorMessageFromBody(parsed), {
      status: response.status,
      code: errorCodeFromBody(parsed),
      body: parsed,
    });
  }

  return parsed;
}

export function getGlobalHistory(): readonly HistoryEntry[] {
  return Object.freeze([...GLOBAL_HISTORY]);
}

export function resetGlobalHistory(): void {
  GLOBAL_HISTORY.splice(0, GLOBAL_HISTORY.length);
}

function assertLMResponseShape(response: unknown): asserts response is LMResponse {
  if (isChatCompletionResponse(response) || isResponsesResponse(response)) {
    return;
  }

  throw new OpenAIRequestError('Malformed OpenAI-compatible response: missing choices/output payload.', {
    body: response,
  });
}

export abstract class BaseLM {
  readonly model: string;
  readonly modelType: ModelType;
  readonly cache: boolean;
  readonly callbacks: readonly Callback[];

  history: HistoryEntry[] = [];

  protected _kwargs: Record<string, unknown>;

  protected constructor(options: BaseLMOptions & { readonly callbacks?: readonly Callback[] | undefined }) {
    this.model = options.model;
    this.modelType = options.modelType ?? 'chat';
    this.cache = options.cache ?? true;
    this.callbacks = Object.freeze([...(options.callbacks ?? [])]);
    this._kwargs = snapshotRecord(options.kwargs);
  }

  get kwargs(): Readonly<Record<string, unknown>> {
    return Object.freeze({ ...this._kwargs });
  }

  get supportsFunctionCalling(): boolean {
    return false;
  }

  get supportsReasoning(): boolean {
    return false;
  }

  get supportsResponseSchema(): boolean {
    return false;
  }

  get supportedParams(): ReadonlySet<string> {
    return new Set<string>();
  }

  copy(overrides: Record<string, unknown> = {}): this {
    const clone = Object.assign(
      Object.create(Object.getPrototypeOf(this)) as this & BaseLM,
      this,
    );

    clone.history = [];
    clone._kwargs = snapshotRecord(this._kwargs);

    for (const [key, value] of Object.entries(overrides)) {
      if (key in clone && key !== '_kwargs') {
        Reflect.set(clone, key, value);
        continue;
      }

      if (value === undefined) {
        delete clone._kwargs[key];
      } else {
        clone._kwargs[key] = snapshotOwnedValue(value);
      }
    }

    return clone as this;
  }

  call(
    prompt?: string,
    messages?: readonly Message[],
    kwargs: Record<string, unknown> = {},
  ): readonly LMOutput[] {
    return runWithCallbacks({
      kind: 'lm',
      instance: this,
      inputs: snapshotRecord({
        ...(prompt === undefined ? {} : { prompt }),
        ...(messages === undefined ? {} : { messages }),
        ...snapshotRecord(kwargs),
      }),
      execute: () => {
        const response = this.forward(prompt, messages, kwargs);
        return this.processAndRecordResponse(response, prompt, messages, kwargs);
      },
    });
  }

  async acall(
    prompt?: string,
    messages?: readonly Message[],
    kwargs: Record<string, unknown> = {},
  ): Promise<readonly LMOutput[]> {
    return runWithCallbacksAsync({
      kind: 'lm',
      instance: this,
      inputs: snapshotRecord({
        ...(prompt === undefined ? {} : { prompt }),
        ...(messages === undefined ? {} : { messages }),
        ...snapshotRecord(kwargs),
      }),
      execute: async () => {
        const response = await this.aforward(prompt, messages, kwargs);
        return this.processAndRecordResponse(response, prompt, messages, kwargs);
      },
    });
  }

  forward(
    prompt?: string,
    messages?: readonly Message[],
    kwargs: Record<string, unknown> = {},
  ): LMResponse | readonly LMOutput[] {
    return this.generate(prompt, messages, this.mergeKwargs(kwargs));
  }

  async aforward(
    prompt?: string,
    messages?: readonly Message[],
    kwargs: Record<string, unknown> = {},
  ): Promise<LMResponse | readonly LMOutput[]> {
    return this.agenerate(prompt, messages, this.mergeKwargs(kwargs));
  }

  updateHistory(entry: HistoryEntry): void {
    if (settings.disableHistory) {
      return;
    }

    if (GLOBAL_HISTORY.length >= GLOBAL_HISTORY_MAX_SIZE) {
      GLOBAL_HISTORY.shift();
    }
    GLOBAL_HISTORY.push(entry);

    if (settings.maxHistorySize === 0) {
      return;
    }

    if (this.history.length >= settings.maxHistorySize) {
      this.history.shift();
    }
    this.history.push(entry);

    for (const module of settings.callerModules) {
      if (module.history.length >= settings.maxHistorySize) {
        module.history.shift();
      }
      module.history.push(entry);
    }
  }

  protected mergeKwargs(overrides: Record<string, unknown> = {}): Record<string, unknown> {
    return snapshotRecord({
      ...this._kwargs,
      ...snapshotRecord(overrides),
    });
  }

  protected generate(
    _prompt?: string,
    _messages?: readonly Message[],
    _kwargs?: Record<string, unknown>,
  ): readonly LMOutput[] {
    throw new RuntimeError('Subclasses must implement forward()/aforward() or generate().');
  }

  protected async agenerate(
    prompt?: string,
    messages?: readonly Message[],
    kwargs?: Record<string, unknown>,
  ): Promise<readonly LMOutput[]> {
    return this.generate(prompt, messages, kwargs);
  }

  private processAndRecordResponse(
    response: LMResponse | readonly LMOutput[],
    prompt: string | undefined,
    messages: readonly Message[] | undefined,
    kwargs: Record<string, unknown>,
  ): readonly LMOutput[] {
    const mergedKwargs = this.mergeKwargs(kwargs);
    const outputs = processLMResponse(this.modelType, response, mergedKwargs);

    if (settings.disableHistory) {
      return outputs;
    }

    const entry: HistoryEntry = Object.freeze({
      prompt: prompt ?? null,
      messages: messages ?? null,
      kwargs: scrubHistoryKwargs(mergedKwargs),
      response: snapshotOwnedValue(response),
      outputs,
      usage: usageRecord(isChatCompletionResponse(response) || isResponsesResponse(response) ? response.usage : undefined),
      cost: null,
      timestamp: new Date().toISOString(),
      uuid: randomUUID(),
      model: this.model,
      responseModel: responseModelName(this.model, response),
      modelType: this.modelType,
    });

    this.updateHistory(entry);
    return outputs;
  }
}

export class LM extends BaseLM {
  readonly apiKey: string | undefined;
  readonly baseURL: string | undefined;
  readonly apiBase: string | undefined;
  readonly organization: string | undefined;
  readonly project: string | undefined;
  readonly headers: Readonly<Record<string, string>>;
  readonly numRetries: number;
  readonly useDeveloperRole: boolean;

  readonly #fetchImpl: typeof globalThis.fetch;

  constructor(model: string, options?: Omit<LMOptions, 'model'>);
  constructor(options: LMOptions);
  constructor(modelOrOptions: string | LMOptions, maybeOptions: Omit<LMOptions, 'model'> = {}) {
    const options = typeof modelOrOptions === 'string'
      ? { ...maybeOptions, model: modelOrOptions }
      : modelOrOptions;

    super(options);
    this.apiKey = options.apiKey;
    this.baseURL = options.baseURL;
    this.apiBase = options.apiBase;
    this.organization = options.organization;
    this.project = options.project;
    this.headers = Object.freeze({ ...(options.headers ?? {}) });
    this.numRetries = options.numRetries ?? 3;
    this.useDeveloperRole = options.useDeveloperRole ?? false;
    this.#fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  override get supportsFunctionCalling(): boolean {
    const provider = providerNameFromModel(this.model);
    return (provider === 'openai' || provider === 'openrouter')
      && (this.modelType === 'chat' || this.modelType === 'responses');
  }

  override get supportsReasoning(): boolean {
    return providerNameFromModel(this.model) === 'openai'
      && REASONING_MODEL_PATTERN.test(providerModelName(this.model).toLowerCase());
  }

  override get supportsResponseSchema(): boolean {
    return providerNameFromModel(this.model) === 'openai'
      && (this.modelType === 'chat' || this.modelType === 'responses');
  }

  override get supportedParams(): ReadonlySet<string> {
    return this.supportsResponseSchema ? RESPONSE_FORMAT_PARAMS : new Set<string>();
  }

  override forward(
    prompt?: string,
    messages?: readonly Message[],
    kwargs: Record<string, unknown> = {},
  ): LMResponse {
    const mergedKwargs = this.mergeKwargs(kwargs);
    return this.requestWithRetries(false, prompt, messages, mergedKwargs) as LMResponse;
  }

  override async aforward(
    prompt?: string,
    messages?: readonly Message[],
    kwargs: Record<string, unknown> = {},
  ): Promise<LMResponse> {
    const mergedKwargs = this.mergeKwargs(kwargs);
    return this.requestWithRetries(true, prompt, messages, mergedKwargs) as Promise<LMResponse>;
  }

  private requestWithRetries(
    asyncMode: boolean,
    prompt: string | undefined,
    messages: readonly Message[] | undefined,
    mergedKwargs: Record<string, unknown>,
  ): LMResponse | Promise<LMResponse> {
    const normalizedMessages = normalizeMessages(prompt, messages);

    if (asyncMode) {
      return (async () => {
        let lastError: unknown = null;

        for (let attempt = 0; attempt <= this.numRetries; attempt += 1) {
          try {
            const response = await this.performRequest(true, normalizedMessages, mergedKwargs);
            assertLMResponseShape(response);
            return response;
          } catch (error) {
            lastError = error;
            if (looksLikeContextWindowExceeded(error)) {
              const message = error instanceof Error ? error.message : 'Context window exceeded';
              throw new ContextWindowExceededError({ model: this.model, message });
            }

            if (attempt >= this.numRetries || !retryableError(error)) {
              throw error;
            }

            await delay(100 * (2 ** attempt));
          }
        }

        throw lastError instanceof Error ? lastError : new Error(String(lastError));
      })();
    }

    let lastError: unknown = null;

    for (let attempt = 0; attempt <= this.numRetries; attempt += 1) {
      try {
        const response = this.performRequest(false, normalizedMessages, mergedKwargs);
        assertLMResponseShape(response);
        return response;
      } catch (error) {
        lastError = error;
        if (looksLikeContextWindowExceeded(error)) {
          const message = error instanceof Error ? error.message : 'Context window exceeded';
          throw new ContextWindowExceededError({ model: this.model, message });
        }

        if (attempt >= this.numRetries || !retryableError(error)) {
          throw error;
        }
      }
    }

    throw lastError instanceof Error ? lastError : new Error(String(lastError));
  }

  private performRequest(
    asyncMode: boolean,
    messages: readonly Message[],
    mergedKwargs: Record<string, unknown>,
  ): LMResponse | Promise<LMResponse> {
    const request = {
      model: providerModelName(this.model),
      messages,
      ...mergedKwargs,
    };

    const transport = buildTransportOptions(this.model, request, {
      model: this.model,
      modelType: this.modelType,
      cache: this.cache,
      kwargs: this.kwargs,
      apiKey: this.apiKey,
      baseURL: this.baseURL,
      apiBase: this.apiBase,
      organization: this.organization,
      project: this.project,
      headers: this.headers,
      numRetries: this.numRetries,
      useDeveloperRole: this.useDeveloperRole,
      callbacks: this.callbacks,
      fetch: this.#fetchImpl,
    });

    const endpoint = this.modelType === 'responses' ? '/responses' : '/chat/completions';
    const url = `${transport.baseURL}${endpoint}`;
    let body = this.modelType === 'responses'
      ? convertChatRequestToResponsesRequest(transport.request, this.useDeveloperRole)
      : transport.request;
    body = mergeExtraBody(body);
    body = applyOpenRouterMinimaxRequestDefaults(this.model, body);

    if (asyncMode) {
      return runAsyncRequest(url, body, transport, this.#fetchImpl) as Promise<LMResponse>;
    }

    return runSyncRequest(url, body, transport) as LMResponse;
  }
}
