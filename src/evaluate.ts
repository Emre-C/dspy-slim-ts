/**
 * §14.3 — Evaluate: metric-driven program evaluation over a devset.
 *
 * Runs a program against each example, scores with a user-supplied metric,
 * and returns an aggregate `EvaluationResult` (a Prediction subclass with
 * numeric protocol). Execution is delegated to `Parallel` for concurrency
 * and error-threshold control.
 */

import type { Callback } from './callback.js';
import { runWithCallbacks, runWithCallbacksAsync } from './callback.js';
import { Example } from './example.js';
import { ValueError } from './exceptions.js';
import type { Module } from './module.js';
import { snapshotRecord } from './owned_value.js';
import { Parallel } from './parallel.js';
import { Prediction } from './prediction.js';

export type EvaluationScore<TScore = unknown> = TScore | number;
export type EvaluationRow<TScore = unknown> = readonly [
  Example,
  Prediction,
  EvaluationScore<TScore>,
];
export type EvaluationMetric<TScore = unknown> = (
  example: Example,
  prediction: Prediction,
) => TScore | Promise<TScore>;

/**
 * Minimal duck-typed contract for a program that can be evaluated.
 *
 * `TInputs` matches each module’s kwargs (e.g. `{ question: string }` for a
 * typed `Predict`). Evaluation maps examples with {@link evalKwargs}; the
 * devset must align with the program’s input signature.
 */
export interface EvaluateProgram<TInputs extends Record<string, unknown> = Record<string, unknown>> {
  call(kwargs: TInputs, ...args: readonly unknown[]): Prediction;
  acall?(kwargs: TInputs, ...args: readonly unknown[]): Promise<Prediction>;
}

/** `Module` (e.g. `Predict`) or a duck-typed program object. */
export type EvaluableProgram<TInputs extends Record<string, unknown>> =
  | Module<TInputs>
  | EvaluateProgram<TInputs>;

/**
 * Bridges serialized example inputs to a program’s typed kwargs. This is the
 * single assertion point: runtime correctness depends on the devset matching
 * the program, same as the previous `any`-based contract.
 */
function evalKwargs<TInputs extends Record<string, unknown>>(example: Example): TInputs {
  return example.inputs().toDict() as TInputs;
}

export interface EvaluateOptions<TScore = unknown> {
  readonly devset: readonly Example[];
  readonly metric?: EvaluationMetric<TScore> | null;
  readonly numThreads?: number | null;
  readonly displayProgress?: boolean;
  readonly maxErrors?: number | null;
  readonly provideTraceback?: boolean | null;
  readonly failureScore?: number;
  readonly callbackMetadata?: Record<string, unknown> | null;
  readonly callbacks?: readonly Callback[];
}

export interface EvaluateCallOptions<TScore = unknown> {
  readonly metric?: EvaluationMetric<TScore> | null;
  readonly devset?: readonly Example[] | null;
  readonly numThreads?: number | null;
  readonly displayProgress?: boolean | null;
  readonly callbackMetadata?: Record<string, unknown> | null;
}

function roundToTwo(value: number): number {
  return Math.round(value * 100) / 100;
}

function ensureMetric<TScore>(metric: EvaluationMetric<TScore> | null): EvaluationMetric<TScore> {
  if (metric === null) {
    throw new ValueError('Evaluate requires a metric.');
  }

  return metric;
}

function ensureDevset(devset: readonly Example[] | null | undefined): readonly Example[] {
  if (devset === null || devset === undefined) {
    throw new ValueError('Evaluate requires a devset.');
  }

  if (devset.length === 0) {
    throw new ValueError('Evaluate requires a non-empty devset.');
  }

  return devset;
}

function ensurePrediction(value: unknown): Prediction {
  if (value instanceof Prediction) {
    return value;
  }

  throw new ValueError('Evaluation program must return a Prediction.');
}

function numericScoreOf(score: unknown): number {
  if (typeof score === 'number' && Number.isFinite(score)) {
    return score;
  }

  if (score instanceof Prediction) {
    return score.toFloat();
  }

  if (typeof score === 'object' && score !== null && 'score' in score) {
    const numeric = (score as { readonly score: unknown }).score;
    if (typeof numeric === 'number' && Number.isFinite(numeric)) {
      return numeric;
    }
  }

  throw new ValueError('Evaluation metrics must return a finite number or a score-bearing Prediction/object.');
}

function hasAsyncProgram<TInputs extends Record<string, unknown>>(
  program: EvaluableProgram<TInputs>,
): program is EvaluableProgram<TInputs> & {
  acall(kwargs: TInputs, ...args: readonly unknown[]): Promise<Prediction>;
} {
  return typeof program.acall === 'function';
}

export class EvaluationResult<TScore = unknown> extends Prediction {
  constructor(score: number, results: readonly EvaluationRow<TScore>[]) {
    super({ score, results }, null);
  }

  get score(): number {
    return this.toFloat();
  }

  get results(): readonly EvaluationRow<TScore>[] {
    return this.get('results') as readonly EvaluationRow<TScore>[];
  }

  override snapshot(): EvaluationResult<TScore> {
    return new EvaluationResult(this.score, this.results);
  }

  override toString(): string {
    return `EvaluationResult(score=${this.score}, results=<list of ${this.results.length} results>)`;
  }
}

export class Evaluate<TScore = unknown> {
  readonly devset: readonly Example[];
  readonly metric: EvaluationMetric<TScore> | null;
  readonly numThreads: number | null;
  readonly displayProgress: boolean;
  readonly maxErrors: number | null;
  readonly provideTraceback: boolean | null;
  readonly failureScore: number;
  readonly callbackMetadata: Record<string, unknown> | null;
  readonly callbacks: readonly Callback[];

  constructor(options: EvaluateOptions<TScore>) {
    this.devset = options.devset;
    this.metric = options.metric ?? null;
    this.numThreads = options.numThreads ?? null;
    this.displayProgress = options.displayProgress ?? false;
    this.maxErrors = options.maxErrors ?? null;
    this.provideTraceback = options.provideTraceback ?? null;
    this.failureScore = options.failureScore ?? 0;
    this.callbackMetadata = options.callbackMetadata ?? null;
    this.callbacks = Object.freeze([...(options.callbacks ?? [])]);
  }

  forward<TInputs extends Record<string, unknown>>(
    program: EvaluableProgram<TInputs>,
    options: EvaluateCallOptions<TScore> = {},
  ): EvaluationResult<TScore> {
    return this.call(program, options);
  }

  aforward<TInputs extends Record<string, unknown>>(
    program: EvaluableProgram<TInputs>,
    options: EvaluateCallOptions<TScore> = {},
  ): Promise<EvaluationResult<TScore>> {
    return this.acall(program, options);
  }

  call<TInputs extends Record<string, unknown>>(
    program: EvaluableProgram<TInputs>,
    options: EvaluateCallOptions<TScore> = {},
  ): EvaluationResult<TScore> {
    const metric = ensureMetric(options.metric ?? this.metric);
    const devset = ensureDevset(options.devset ?? this.devset);
    const numThreads = options.numThreads ?? this.numThreads;
    const displayProgress = options.displayProgress ?? this.displayProgress;
    const callbackMetadata = options.callbackMetadata ?? this.callbackMetadata;

    return runWithCallbacks({
      kind: 'evaluate',
      instance: this,
      inputs: snapshotRecord({
        program,
        metric,
        devset,
        numThreads,
        displayProgress,
        callbackMetadata,
      }),
      execute: () => {
        const executor = this.createExecutor(numThreads, displayProgress);
        const outputs = executor.execute((example) => {
          const prediction = ensurePrediction(program.call(evalKwargs<TInputs>(example)));
          const score = metric(example, prediction);
          if (score instanceof Promise) {
            throw new ValueError('Evaluate.call received an async metric; use acall instead.');
          }
          return [prediction, score] as const;
        }, devset);
        return this.buildResult(devset, outputs);
      },
    });
  }

  async acall<TInputs extends Record<string, unknown>>(
    program: EvaluableProgram<TInputs>,
    options: EvaluateCallOptions<TScore> = {},
  ): Promise<EvaluationResult<TScore>> {
    const metric = ensureMetric(options.metric ?? this.metric);
    const devset = ensureDevset(options.devset ?? this.devset);
    const numThreads = options.numThreads ?? this.numThreads;
    const displayProgress = options.displayProgress ?? this.displayProgress;
    const callbackMetadata = options.callbackMetadata ?? this.callbackMetadata;

    return runWithCallbacksAsync({
      kind: 'evaluate',
      instance: this,
      inputs: snapshotRecord({
        program,
        metric,
        devset,
        numThreads,
        displayProgress,
        callbackMetadata,
      }),
      execute: async () => {
        const executor = this.createExecutor(numThreads, displayProgress);
        const outputs = await executor.executeAsync(async (example) => {
          const prediction = ensurePrediction(
            hasAsyncProgram(program)
              ? await program.acall(evalKwargs<TInputs>(example))
              : program.call(evalKwargs<TInputs>(example)),
          );
          const score = await Promise.resolve(metric(example, prediction));
          return [prediction, score] as const;
        }, devset, { numThreads });
        return this.buildResult(devset, outputs);
      },
    });
  }

  private createExecutor(numThreads: number | null, displayProgress: boolean): Parallel {
    return new Parallel({
      ...(numThreads === null ? {} : { numThreads }),
      ...(this.maxErrors === null ? {} : { maxErrors: this.maxErrors }),
      ...(this.provideTraceback === null ? {} : { provideTraceback: this.provideTraceback }),
      disableProgressBar: !displayProgress,
    });
  }

  private buildResult(
    devset: readonly Example[],
    outputs: readonly ((readonly [Prediction, TScore]) | null)[],
  ): EvaluationResult<TScore> {
    const rows = Object.freeze(devset.map((example, index) => {
      const output = outputs[index];
      if (output === null || output === undefined) {
        return [example, Prediction.create({}), this.failureScore] as const;
      }

      return [example, output[0], output[1]] as const;
    }));
    const total = rows.reduce((sum, row) => sum + numericScoreOf(row[2]), 0);
    return new EvaluationResult(roundToTwo((100 * total) / rows.length), rows);
  }
}
