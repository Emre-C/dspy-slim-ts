/**
 * OpenAI-style SSE streaming for `/v1/chat/completions` (see OpenAI streaming docs).
 * Used to avoid HTTP gateway timeouts on long single-shot responses (e.g. HF router 504).
 */

import { isPlainObject } from './guards.js';

/** Same rules as {@link stringifyAssistantContent} in `lm.ts` — keep delta text extraction aligned. */
function stringifyDeltaContent(content: unknown): string {
  if (content === null || content === undefined) {
    return '';
  }
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    let acc = '';
    for (const part of content) {
      if (typeof part === 'string') {
        acc += part;
      } else if (isPlainObject(part) && typeof part.text === 'string') {
        acc += part.text;
      }
    }
    return acc;
  }
  return '';
}

function appendAssistantDelta(
  delta: Record<string, unknown>,
  into: { content: string; reasoningContent: string; reasoning: string },
): void {
  into.content += stringifyDeltaContent(delta.content);
  into.reasoningContent += stringifyDeltaContent(
    (delta as { readonly reasoning_content?: unknown }).reasoning_content,
  );
  into.reasoning += stringifyDeltaContent(delta.reasoning);
}

interface StreamAgg {
  model: string;
  content: string;
  reasoningContent: string;
  reasoning: string;
  finishReason: string | null;
  usage: Record<string, number> | undefined;
  sawDataLine: boolean;
}

function createStreamAgg(fallbackModel: string): StreamAgg {
  return {
    model: fallbackModel,
    content: '',
    reasoningContent: '',
    reasoning: '',
    finishReason: null,
    usage: undefined,
    sawDataLine: false,
  };
}

function usageFromChunk(chunk: Record<string, unknown>): Record<string, number> | undefined {
  const usage = chunk.usage;
  if (!isPlainObject(usage)) {
    return undefined;
  }
  const out: Record<string, number> = {};
  for (const [key, value] of Object.entries(usage)) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      out[key] = value;
    }
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

function accumulateOpenAiChatChunk(chunk: unknown, agg: StreamAgg): void {
  if (!isPlainObject(chunk)) {
    return;
  }

  if (typeof chunk.model === 'string' && chunk.model.length > 0) {
    agg.model = chunk.model;
  }

  const usage = usageFromChunk(chunk);
  if (usage !== undefined) {
    agg.usage = usage;
  }

  const choices = chunk.choices;
  if (!Array.isArray(choices) || choices.length === 0) {
    return;
  }

  const first = choices[0];
  if (!isPlainObject(first)) {
    return;
  }

  const delta = first.delta;
  if (isPlainObject(delta)) {
    appendAssistantDelta(delta, agg);
  }

  if (typeof first.finish_reason === 'string') {
    agg.finishReason = first.finish_reason;
  }
}

function parseSseDataLine(payload: string, agg: StreamAgg): void {
  const trimmed = payload.trim();
  if (trimmed === '[DONE]') {
    return;
  }
  try {
    const parsed: unknown = JSON.parse(trimmed);
    agg.sawDataLine = true;
    accumulateOpenAiChatChunk(parsed, agg);
  } catch {
    // Ignore non-JSON lines (comments, malformed frames).
  }
}

function splitStreamBuffer(buffer: string, flush: boolean): { rest: string; lines: string[] } {
  const lines: string[] = [];
  let rest = buffer;
  while (true) {
    const nl = rest.indexOf('\n');
    if (nl < 0) {
      break;
    }
    const line = rest.slice(0, nl);
    rest = rest.slice(nl + 1);
    lines.push(line.replace(/\r$/, ''));
  }
  if (flush && rest.length > 0) {
    lines.push(rest.replace(/\r$/, ''));
    rest = '';
  }
  return { rest, lines };
}

function processSseLines(lines: string[], agg: StreamAgg): void {
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.length === 0 || trimmed.startsWith(':')) {
      continue;
    }
    if (!trimmed.startsWith('data:')) {
      continue;
    }
    parseSseDataLine(trimmed.slice(5).trimStart(), agg);
  }
}

/**
 * Aggregates an OpenAI-compatible `text/event-stream` body into a single non-streaming
 * `chat.completion`-shaped JSON object for reuse by `processChatCompletion`.
 */
export async function aggregateOpenAiChatCompletionStream(
  body: ReadableStream<Uint8Array>,
  fallbackModel: string,
): Promise<{
  readonly model: string;
  readonly choices: readonly unknown[];
  readonly usage?: Record<string, number | undefined>;
}> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const agg = createStreamAgg(fallbackModel);

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    if (value !== undefined) {
      buffer += decoder.decode(value, { stream: true });
      const { rest, lines } = splitStreamBuffer(buffer, false);
      buffer = rest;
      processSseLines(lines, agg);
    }
  }
  buffer += decoder.decode();
  processSseLines(splitStreamBuffer(buffer, true).lines, agg);

  if (!agg.sawDataLine) {
    throw new Error('Empty or unrecognized OpenAI-compatible SSE stream (no data events).');
  }

  const message: Record<string, unknown> = {};
  const hasContent = agg.content.length > 0;
  message.content = hasContent ? agg.content : null;
  if (agg.reasoningContent.length > 0) {
    message.reasoning_content = agg.reasoningContent;
  }
  if (agg.reasoning.length > 0) {
    message.reasoning = agg.reasoning;
  }

  return {
    model: agg.model || fallbackModel,
    choices: [
      {
        message,
        finish_reason: agg.finishReason,
      },
    ],
    ...(agg.usage !== undefined ? { usage: { ...agg.usage } } : {}),
  };
}
