/**
 * ReplayLM — deterministic LM that replays pre-recorded responses.
 *
 * Used for golden-transcript testing: record real LM responses once,
 * then replay them in CI to test the full pipeline without API keys.
 *
 * Supports loading transcripts from JSON files (the format written by
 * the Python `record_golden.py` script).
 */

import { readFileSync } from 'node:fs';
import type { Message } from './chat_message.js';
import { BaseLM, type LMOutput } from './lm.js';
import { RuntimeError } from './exceptions.js';
import { ownedValueEquals, snapshotOwnedValue } from './owned_value.js';

export interface GoldenTranscriptEntry {
  readonly inputs?: Record<string, unknown>;
  readonly prompt?: string | null;
  readonly messages?: readonly Message[] | null;
  readonly module?: string;
  readonly signature?: string;
  readonly typescript_prompt?: string | null;
  readonly typescript_messages?: readonly Message[] | null;
  readonly output: string;
  readonly dataset: string;
  readonly example_idx: number;
  readonly gold_answer?: string;
  readonly gold_answers?: readonly string[];
}

interface ReplayTurn {
  readonly output: LMOutput;
  readonly prompt: string | null;
  readonly messages: readonly Message[] | null;
  readonly validateCall: boolean;
}

function shouldValidateCall(entry: GoldenTranscriptEntry): boolean {
  const prompt = entry.typescript_prompt ?? entry.prompt ?? null;
  const messages = entry.typescript_messages ?? entry.messages ?? null;
  return prompt !== null
    ? true
    : Array.isArray(messages) && messages.length > 0;
}

function nonSystemMessages(messages: readonly Message[] | null | undefined): readonly Message[] {
  return (messages ?? []).filter((message) => message.role !== 'system');
}

function systemRoles(messages: readonly Message[] | null | undefined): readonly string[] {
  return (messages ?? [])
    .filter((message) => message.role === 'system')
    .map((message) => message.role);
}

function messagesMatch(
  expected: readonly Message[] | null | undefined,
  actual: readonly Message[] | null | undefined,
): boolean {
  if (ownedValueEquals(actual ?? null, expected ?? null)) {
    return true;
  }

  if (!ownedValueEquals(nonSystemMessages(actual), nonSystemMessages(expected))) {
    return false;
  }

  const expectedSystemRoles = systemRoles(expected);
  if (expectedSystemRoles.length === 0) {
    return true;
  }

  return ownedValueEquals(systemRoles(actual), expectedSystemRoles)
    && (actual?.length ?? 0) === (expected?.length ?? 0);
}

function describePayload(value: unknown): string {
  return JSON.stringify(value, null, 2) ?? String(value);
}

export class ReplayLM extends BaseLM {
  private readonly queue: readonly ReplayTurn[];
  private cursor = 0;

  constructor(outputs: readonly LMOutput[], queue?: readonly ReplayTurn[]) {
    super({ model: 'replay-lm' });
    this.queue = queue ?? outputs.map((output) => Object.freeze({
      output,
      prompt: null,
      messages: null,
      validateCall: false,
    }));
  }

  /**
   * Load a ReplayLM from a golden transcript JSON file.
   * The file should contain an array of GoldenTranscriptEntry objects.
   */
  static fromFile(path: string): ReplayLM {
    const raw = readFileSync(path, 'utf-8');
    const entries: readonly GoldenTranscriptEntry[] = JSON.parse(raw);
    return ReplayLM.fromTranscripts(entries);
  }

  /**
   * Load a ReplayLM from an array of GoldenTranscriptEntry objects.
   */
  static fromTranscripts(entries: readonly GoldenTranscriptEntry[]): ReplayLM {
    const queue = Object.freeze(entries.map((entry) => Object.freeze({
      output: entry.output,
      prompt: entry.typescript_prompt ?? entry.prompt ?? null,
      messages: (snapshotOwnedValue(
        entry.typescript_messages ?? entry.messages ?? null,
      ) as readonly Message[] | null),
      validateCall: shouldValidateCall(entry),
    })));
    return new ReplayLM([], queue);
  }

  /** Number of outputs remaining in the queue. */
  get remaining(): number {
    return this.queue.length - this.cursor;
  }

  /** Whether all outputs have been consumed. */
  get exhausted(): boolean {
    return this.cursor >= this.queue.length;
  }

  private assertExpectedCall(turn: ReplayTurn, prompt?: string, messages?: readonly Message[]): void {
    if (!turn.validateCall) {
      return;
    }

    const actualPrompt = prompt ?? null;
    const actualMessages = messages ?? null;
    if (actualPrompt === turn.prompt && messagesMatch(turn.messages, actualMessages)) {
      return;
    }

    throw new RuntimeError(
      `ReplayLM payload mismatch at turn ${this.cursor}: expected prompt=${describePayload(turn.prompt)}, `
      + `messages=${describePayload(turn.messages)}; got prompt=${describePayload(actualPrompt)}, `
      + `messages=${describePayload(actualMessages)}.`,
    );
  }

  protected override generate(
    prompt?: string,
    messages?: readonly Message[],
    _kwargs?: Record<string, unknown>,
  ): readonly LMOutput[] {
    if (this.cursor >= this.queue.length) {
      throw new RuntimeError(
        `ReplayLM exhausted: all ${this.queue.length} recorded outputs have been consumed.`,
      );
    }

    const turn = this.queue[this.cursor]!;
    this.assertExpectedCall(turn, prompt, messages);
    this.cursor += 1;
    return [turn.output];
  }
}
