/**
 * §6 — Language model runtime contract.
 *
 * We keep the LM abstraction intentionally small at this stage: adapters and
 * Predict only need runtime identity, default kwargs, and sync/async text
 * generation entrypoints. BaseLM owns default-kwarg merging so every concrete
 * LM gets the same call semantics without reimplementing that policy.
 */

import type { Message } from './adapter.js';
import { snapshotRecord } from './owned_value.js';
import type { ModelType } from './types.js';

export interface BaseLMOptions {
  readonly model: string;
  readonly modelType?: ModelType;
  readonly cache?: boolean;
  readonly kwargs?: Record<string, unknown>;
}

export interface LMOutputEnvelope {
  readonly text: string;
  readonly logprobs?: unknown;
  readonly citations?: readonly unknown[];
  readonly toolCalls?: readonly unknown[];
}

export type LMOutput = string | LMOutputEnvelope;

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

export abstract class BaseLM {
  readonly model: string;
  readonly modelType: ModelType;
  readonly cache: boolean;

  history: HistoryEntry[] = [];

  protected _kwargs: Record<string, unknown>;

  protected constructor(options: BaseLMOptions) {
    this.model = options.model;
    this.modelType = options.modelType ?? 'chat';
    this.cache = options.cache ?? true;
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
    clone._kwargs = this.mergeKwargs(overrides);

    return clone as this;
  }

  call(
    prompt?: string,
    messages?: readonly Message[],
    kwargs?: Record<string, unknown>,
  ): readonly LMOutput[] {
    return this.generate(prompt, messages, this.mergeKwargs(kwargs));
  }

  async acall(
    prompt?: string,
    messages?: readonly Message[],
    kwargs?: Record<string, unknown>,
  ): Promise<readonly LMOutput[]> {
    return this.agenerate(prompt, messages, this.mergeKwargs(kwargs));
  }

  protected mergeKwargs(overrides: Record<string, unknown> = {}): Record<string, unknown> {
    return snapshotRecord({
      ...this._kwargs,
      ...snapshotRecord(overrides),
    });
  }

  protected abstract generate(
    prompt?: string,
    messages?: readonly Message[],
    kwargs?: Record<string, unknown>,
  ): readonly LMOutput[];

  protected async agenerate(
    prompt?: string,
    messages?: readonly Message[],
    kwargs?: Record<string, unknown>,
  ): Promise<readonly LMOutput[]> {
    return this.generate(prompt, messages, kwargs);
  }
}
