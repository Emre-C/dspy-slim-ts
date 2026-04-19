/**
 * §10.3 — Parallel execution engine.
 *
 * Concurrency-bounded task runner with error-count cancellation and
 * straggler resubmission. Synchronous `execute` runs sequentially;
 * `executeAsync` runs a promise pool capped at `numThreads`.
 *
 * State machine (per spec):
 *   Idle → Running → Done | Cancelled | Error
 *   Per-task: Pending → Executing → Completed | Failed | Resubmitted
 */

import type { Demo } from './adapter.js';
import { Example } from './example.js';
import { RuntimeError, ValueError } from './exceptions.js';
import { isPlainObject } from './guards.js';
import { settings } from './settings.js';

const EMPTY_RESULTS = Object.freeze([]) as readonly null[];
const POLL_INTERVAL_MS = 10;

export interface ParallelOptions {
  readonly numThreads?: number;
  readonly maxErrors?: number;
  readonly accessExamples?: boolean;
  readonly returnFailedExamples?: boolean;
  readonly provideTraceback?: boolean;
  readonly disableProgressBar?: boolean;
  readonly timeout?: number;
  readonly stragglerLimit?: number;
}

export interface ParallelCallable<TResult> {
  readonly call?: ((kwargs: Record<string, unknown>) => TResult) | undefined;
  readonly acall?: ((kwargs: Record<string, unknown>) => Promise<TResult>) | undefined;
}

export type ParallelInput = Demo | readonly unknown[];
export type ParallelTarget<TResult> = ParallelCallable<TResult> | Parallel | ((...args: readonly unknown[]) => TResult | Promise<TResult>);
export type ParallelExecPair<TResult, TInput = ParallelInput> = readonly [ParallelTarget<TResult>, TInput];
export type ParallelResults<TResult> = readonly (TResult | null)[];
export type ParallelFailureBundle<TResult, TInput = ParallelInput> = readonly [
  ParallelResults<TResult>,
  readonly TInput[],
  readonly Error[],
];
export type ParallelForwardResult<TResult, TInput = ParallelInput> =
  | ParallelResults<TResult>
  | ParallelFailureBundle<TResult, TInput>;

interface AttemptResult<TResult> {
  readonly submissionID: number;
  readonly index: number;
  readonly status: 'fulfilled' | 'rejected';
  readonly value?: TResult | undefined;
  readonly error?: Error | undefined;
}

interface ActiveAttempt<TData, TResult> {
  readonly submissionID: number;
  readonly index: number;
  readonly item: TData;
  readonly startedAt: number;
  readonly promise: Promise<AttemptResult<TResult>>;
  hasResubmitted: boolean;
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function ensurePositiveInteger(name: string, value: number): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new ValueError(`${name} must be a positive integer.`);
  }

  return value;
}

function ensureNonNegativeInteger(name: string, value: number): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new ValueError(`${name} must be a non-negative integer.`);
  }

  return value;
}

function isFunctionTarget<TResult>(value: unknown): value is (...args: readonly unknown[]) => TResult | Promise<TResult> {
  return typeof value === 'function';
}

function isSyncCallTarget<TResult>(value: unknown): value is { call: (kwargs: Record<string, unknown>) => TResult } {
  return typeof value === 'object' && value !== null && 'call' in value && typeof value.call === 'function';
}

function isAsyncCallTarget<TResult>(value: unknown): value is { acall: (kwargs: Record<string, unknown>) => Promise<TResult> } {
  return typeof value === 'object' && value !== null && 'acall' in value && typeof value.acall === 'function';
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export class Parallel {
  readonly numThreads: number;
  readonly maxErrors: number;
  readonly accessExamples: boolean;
  readonly returnFailedExamples: boolean;
  readonly provideTraceback: boolean;
  readonly disableProgressBar: boolean;
  readonly timeout: number;
  readonly stragglerLimit: number;

  #errorCount = 0;
  #cancelJobs = false;
  #failedIndices: readonly number[] = Object.freeze([]);
  #failedItems: readonly unknown[] = Object.freeze([]);
  #exceptions: readonly Error[] = Object.freeze([]);

  get failedIndices(): readonly number[] {
    return this.#failedIndices;
  }

  get failedItems(): readonly unknown[] {
    return this.#failedItems;
  }

  get exceptions(): readonly Error[] {
    return this.#exceptions;
  }

  constructor(options: ParallelOptions = {}) {
    this.numThreads = ensurePositiveInteger('numThreads', options.numThreads ?? settings.numThreads);
    this.maxErrors = ensurePositiveInteger('maxErrors', options.maxErrors ?? settings.maxErrors);
    this.accessExamples = options.accessExamples ?? true;
    this.returnFailedExamples = options.returnFailedExamples ?? false;
    this.provideTraceback = options.provideTraceback ?? false;
    this.disableProgressBar = options.disableProgressBar ?? true;
    this.timeout = ensureNonNegativeInteger('timeout', options.timeout ?? 120_000);
    this.stragglerLimit = ensurePositiveInteger('stragglerLimit', options.stragglerLimit ?? 3);
  }

  execute<TData, TResult>(task: (item: TData) => TResult, data: readonly TData[]): ParallelResults<TResult> {
    const results: Array<TResult | null> = Array.from({ length: data.length }, () => null);
    const failures = new Map<number, Error>();
    if (data.length === 0) {
      this.resetRunState();
      return EMPTY_RESULTS as ParallelResults<TResult>;
    }

    this.resetRunState();

    for (const [index, item] of data.entries()) {
      if (this.#cancelJobs) {
        break;
      }

      try {
        results[index] = task(item);
      } catch (error) {
        this.recordFailure(failures, index, error);
      }
    }

    this.finalizeFailures(failures, data);

    if (this.#cancelJobs) {
      throw new RuntimeError('Execution cancelled due to errors or interruption.');
    }

    return Object.freeze([...results]);
  }

  async executeAsync<TData, TResult>(
    task: (item: TData) => TResult | Promise<TResult>,
    data: readonly TData[],
    options: { readonly numThreads?: number | null } = {},
  ): Promise<ParallelResults<TResult>> {
    if (data.length === 0) {
      this.resetRunState();
      return EMPTY_RESULTS as ParallelResults<TResult>;
    }

    this.resetRunState();
    const failures = new Map<number, Error>();
    const results: Array<TResult | null> = Array.from({ length: data.length }, () => null);
    const numThreads = this.resolveNumThreads(options.numThreads);
    const active = new Map<number, ActiveAttempt<TData, TResult>>();
    let nextIndex = 0;
    let submissionID = 0;

    const launch = (index: number, item: TData): void => {
      const currentSubmissionID = submissionID;
      submissionID += 1;
      const promise = Promise.resolve().then(() => (
        settings.withThread(currentSubmissionID + 1, () => task(item))
      )).then((value) => ({
        submissionID: currentSubmissionID,
        index,
        status: 'fulfilled' as const,
        value,
      })).catch((error: unknown) => ({
        submissionID: currentSubmissionID,
        index,
        status: 'rejected' as const,
        error: toError(error),
      }));

      active.set(currentSubmissionID, {
        submissionID: currentSubmissionID,
        index,
        item,
        startedAt: Date.now(),
        promise,
        hasResubmitted: false,
      });
    };

    while ((nextIndex < data.length || active.size > 0) && !this.#cancelJobs) {
      while (nextIndex < data.length && active.size < numThreads) {
        launch(nextIndex, data[nextIndex]!);
        nextIndex += 1;
      }

      if (active.size === 0) {
        break;
      }

      const settled = await Promise.race<AttemptResult<TResult> | null>([
        ...[...active.values()].map((attempt) => attempt.promise),
        sleep(POLL_INTERVAL_MS).then(() => null),
      ]);

      if (settled !== null) {
        active.delete(settled.submissionID);
        if (settled.status === 'fulfilled') {
          if (results[settled.index] === null) {
            results[settled.index] = settled.value!;
            failures.delete(settled.index);
            for (const [submissionIDKey, attempt] of active.entries()) {
              if (attempt.index === settled.index) {
                active.delete(submissionIDKey);
              }
            }
          }
        } else if (results[settled.index] === null) {
          const hasSiblingAttempt = [...active.values()].some((a) => a.index === settled.index);
          if (hasSiblingAttempt) {
            if (!failures.has(settled.index)) {
              failures.set(settled.index, settled.error!);
            }
          } else {
            this.recordFailure(failures, settled.index, settled.error!);
          }
        }
        continue;
      }

      if (this.timeout > 0) {
        const uniqueActiveIndices = new Set<number>();
        for (const attempt of active.values()) {
          uniqueActiveIndices.add(attempt.index);
        }
        if (uniqueActiveIndices.size <= this.stragglerLimit) {
          const now = Date.now();
          for (const attempt of active.values()) {
            if (!attempt.hasResubmitted && now - attempt.startedAt >= this.timeout) {
              attempt.hasResubmitted = true;
              launch(attempt.index, attempt.item);
            }
          }
        }
      }
    }

    this.finalizeFailures(failures, data);

    if (this.#cancelJobs) {
      throw new RuntimeError('Execution cancelled due to errors or interruption.');
    }

    return Object.freeze([...results]);
  }

  forward<TResult, TInput = ParallelInput>(
    execPairs: readonly ParallelExecPair<TResult, TInput>[],
    numThreads: number | null = null,
  ): ParallelForwardResult<TResult, TInput> {
    this.resolveNumThreads(numThreads);
    const task = (pair: ParallelExecPair<TResult, TInput>) => this.invokeSyncPair(pair);
    const results = this.execute(task, execPairs);
    return this.formatForwardResult(results, execPairs);
  }

  async aforward<TResult, TInput = ParallelInput>(
    execPairs: readonly ParallelExecPair<TResult, TInput>[],
    numThreads: number | null = null,
  ): Promise<ParallelForwardResult<TResult, TInput>> {
    const task = (pair: ParallelExecPair<TResult, TInput>) => this.invokeAsyncPair(pair);
    const results = await this.executeAsync(task, execPairs, { numThreads });
    return this.formatForwardResult(results, execPairs);
  }

  call<TResult, TInput = ParallelInput>(
    execPairs: readonly ParallelExecPair<TResult, TInput>[],
    numThreads: number | null = null,
  ): ParallelForwardResult<TResult, TInput> {
    return this.forward(execPairs, numThreads);
  }

  acall<TResult, TInput = ParallelInput>(
    execPairs: readonly ParallelExecPair<TResult, TInput>[],
    numThreads: number | null = null,
  ): Promise<ParallelForwardResult<TResult, TInput>> {
    return this.aforward(execPairs, numThreads);
  }

  private formatForwardResult<TResult, TInput>(
    results: ParallelResults<TResult>,
    execPairs: readonly ParallelExecPair<TResult, TInput>[],
  ): ParallelForwardResult<TResult, TInput> {
    if (!this.returnFailedExamples) {
      return results;
    }

    const failedExamples = Object.freeze(
      this.#failedIndices.map((index) => execPairs[index]![1]),
    ) as readonly TInput[];
    return Object.freeze([results, failedExamples, this.#exceptions] as const) as ParallelFailureBundle<
      TResult,
      TInput
    >;
  }

  private invokeSyncPair<TResult, TInput>(pair: ParallelExecPair<TResult, TInput>): TResult {
    const [invokable, example] = pair;
    return this.invokeSync(invokable, example);
  }

  private async invokeAsyncPair<TResult, TInput>(pair: ParallelExecPair<TResult, TInput>): Promise<TResult> {
    const [invokable, example] = pair;
    return this.invokeAsync(invokable, example);
  }

  private invokeSync<TResult>(
    invokable: ParallelTarget<TResult>,
    input: unknown,
  ): TResult {
    if (invokable instanceof Parallel) {
      if (!Array.isArray(input)) {
        throw new ValueError('Nested Parallel expects an array of execution pairs.');
      }
      return invokable.forward(input as readonly ParallelExecPair<TResult>[]) as TResult;
    }

    if (input instanceof Example) {
      if (!this.accessExamples) {
        throw new ValueError('Parallel accessExamples=false does not support Example inputs in TypeScript.');
      }
      return this.invokeSyncWithKwargs(invokable, input.inputs().toDict());
    }

    if (Array.isArray(input)) {
      if (isFunctionTarget<TResult>(invokable)) {
        const result = invokable(...input);
        if (result instanceof Promise) {
          throw new ValueError('Parallel forward received an async callable; use aforward instead.');
        }
        return result;
      }
      throw new ValueError('Parallel array inputs require a function target or nested Parallel.');
    }

    if (isPlainObject(input)) {
      return this.invokeSyncWithKwargs(invokable, input);
    }

    if (isFunctionTarget<TResult>(invokable)) {
      const result = invokable(input);
      if (result instanceof Promise) {
        throw new ValueError('Parallel forward received an async callable; use aforward instead.');
      }
      return result;
    }

    throw new ValueError('Parallel input type is not supported for the provided target.');
  }

  private async invokeAsync<TResult>(
    invokable: ParallelTarget<TResult>,
    input: unknown,
  ): Promise<TResult> {
    if (invokable instanceof Parallel) {
      if (!Array.isArray(input)) {
        throw new ValueError('Nested Parallel expects an array of execution pairs.');
      }
      return invokable.aforward(input as readonly ParallelExecPair<TResult>[]) as Promise<TResult>;
    }

    if (input instanceof Example) {
      if (!this.accessExamples) {
        throw new ValueError('Parallel accessExamples=false does not support Example inputs in TypeScript.');
      }
      return this.invokeAsyncWithKwargs(invokable, input.inputs().toDict());
    }

    if (Array.isArray(input)) {
      if (isFunctionTarget<TResult>(invokable)) {
        return Promise.resolve(invokable(...input));
      }
      throw new ValueError('Parallel array inputs require a function target or nested Parallel.');
    }

    if (isPlainObject(input)) {
      return this.invokeAsyncWithKwargs(invokable, input);
    }

    if (isFunctionTarget<TResult>(invokable)) {
      return Promise.resolve(invokable(input));
    }

    throw new ValueError('Parallel input type is not supported for the provided target.');
  }

  private invokeSyncWithKwargs<TResult>(
    invokable: Exclude<ParallelTarget<TResult>, Parallel>,
    kwargs: Record<string, unknown>,
  ): TResult {
    if (isSyncCallTarget<TResult>(invokable)) {
      const target: { call: (inputs: Record<string, unknown>) => TResult } = invokable;
      return target.call(kwargs);
    }

    if (isAsyncCallTarget<TResult>(invokable)) {
      throw new ValueError('Parallel forward received an async callable; use aforward instead.');
    }

    if (isFunctionTarget<TResult>(invokable)) {
      const result = invokable(kwargs);
      if (result instanceof Promise) {
        throw new ValueError('Parallel forward received an async callable; use aforward instead.');
      }
      return result;
    }

    throw new ValueError('Parallel target must expose call/acall or be a function.');
  }

  private async invokeAsyncWithKwargs<TResult>(
    invokable: Exclude<ParallelTarget<TResult>, Parallel>,
    kwargs: Record<string, unknown>,
  ): Promise<TResult> {
    if (isAsyncCallTarget<TResult>(invokable)) {
      const target: { acall: (inputs: Record<string, unknown>) => Promise<TResult> } = invokable;
      return target.acall(kwargs);
    }

    if (isSyncCallTarget<TResult>(invokable)) {
      const target: { call: (inputs: Record<string, unknown>) => TResult } = invokable;
      return target.call(kwargs);
    }

    if (isFunctionTarget<TResult>(invokable)) {
      return Promise.resolve(invokable(kwargs));
    }

    throw new ValueError('Parallel target must expose call/acall or be a function.');
  }

  private recordFailure(failures: Map<number, Error>, index: number, error: unknown): void {
    const normalized = toError(error);
    if (!failures.has(index)) {
      failures.set(index, normalized);
    }
    this.#errorCount += 1;
    if (this.provideTraceback) {
      console.error(normalized.stack ?? `${normalized.name}: ${normalized.message}`);
    } else {
      console.error(`${normalized.name}: ${normalized.message}. Set provideTraceback=true for traceback.`);
    }
    if (this.#errorCount >= this.maxErrors) {
      this.#cancelJobs = true;
    }
  }

  private finalizeFailures<TData>(failures: Map<number, Error>, data: readonly TData[]): void {
    const failedIndices = Object.freeze([...failures.keys()].sort((left, right) => left - right));
    this.#failedIndices = failedIndices;
    this.#failedItems = Object.freeze(failedIndices.map((index) => data[index]!));
    this.#exceptions = Object.freeze(failedIndices.map((index) => failures.get(index)!));
  }

  private resolveNumThreads(numThreads: number | null | undefined): number {
    return ensurePositiveInteger('numThreads', numThreads ?? this.numThreads);
  }

  private resetRunState(): void {
    this.#errorCount = 0;
    this.#cancelJobs = false;
    this.#failedIndices = Object.freeze([]);
    this.#failedItems = Object.freeze([]);
    this.#exceptions = Object.freeze([]);
  }
}
