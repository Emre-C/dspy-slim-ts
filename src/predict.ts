/**
 * §4 — Predict pipeline.
 */

import { Adapter, type Demo, JSONAdapter } from './adapter.js';
import { ConfigurationError, RuntimeError, ValueError } from './exceptions.js';
import { Example } from './example.js';
import { createField } from './field.js';
import { isPlainObject } from './guards.js';
import { BaseLM } from './lm.js';
import { Module, markPredictor } from './module.js';
import { snapshotRecord } from './owned_value.js';
import { Prediction } from './prediction.js';
import { settings } from './settings.js';
import {
  Signature,
  createSignature,
  ensureSignature,
} from './signature.js';
import type { InferInputs, InferOutputs, SignatureInput } from './signature_types.js';

/**
 * Control-plane keys accepted alongside input fields in a single `.forward()`
 * call, matching the runtime's `resolveSignature` / `resolveDemos` /
 * `resolveConfig` / `resolveLm` branches. Callers with literal-string
 * signatures should still be able to pass these without the type system
 * treating them as excess input properties.
 */
export interface PredictForwardOverrides {
  readonly signature?: Signature | string;
  readonly demos?: readonly Demo[];
  readonly config?: Record<string, unknown>;
  readonly lm?: BaseLM;
}

/**
 * Kwargs accepted by `Predict.forward` / `.call` / `.acall` / `.aforward`:
 * the inferred inputs (possibly `Record<string, unknown>` for non-literal
 * signatures) plus the four control-plane overrides.
 */
export type PredictKwargs<TInputs extends Record<string, unknown>> =
  TInputs & PredictForwardOverrides;

export interface PredictTrace<TOutputs extends Record<string, unknown> = Record<string, unknown>> {
  readonly inputs: Record<string, unknown>;
  readonly prediction: Prediction<TOutputs>;
}

export interface PredictPreprocessResult {
  readonly adapter: Adapter;
  readonly lm: BaseLM;
  readonly signature: Signature;
  readonly demos: readonly Record<string, unknown>[];
  readonly config: Record<string, unknown>;
  readonly inputs: Record<string, unknown>;
}

const EMPTY_RECORD = Object.freeze({}) as Record<string, unknown>;
const RESERVED_PREDICT_INPUT_KEYS = new Set(['signature', 'demos', 'config', 'lm']);

function ensurePredictSignature(value: Signature | string): Signature {
  const signature = ensureSignature(value);
  const reservedInputs = [...signature.inputFields.keys()].filter((name) => (
    RESERVED_PREDICT_INPUT_KEYS.has(name)
  ));

  if (reservedInputs.length > 0) {
    throw new ValueError(
      `Predict input field names are reserved for control overrides: ${reservedInputs.join(', ')}.`,
    );
  }

  return signature;
}

function normalizeConfig(value: unknown): Record<string, unknown> {
  if (value === undefined) {
    return {};
  }

  if (!isPlainObject(value)) {
    throw new ValueError('Predict config must be a plain object');
  }

  return snapshotRecord(value);
}

function normalizeDemos(demos: readonly Demo[]): readonly Record<string, unknown>[] {
  return Object.freeze(demos.map((demo) => (
    demo instanceof Example
      ? snapshotRecord(demo.toDict())
      : snapshotRecord(demo)
  )));
}

function firstDefined(...values: readonly unknown[]): unknown {
  let last: unknown;

  for (const value of values) {
    last = value;
    if (value !== undefined && value !== null) {
      return value;
    }
  }

  return last;
}

/**
 * A DSPy predictor.
 *
 * The three generic parameters form a pipeline:
 *   - `TSig` is the constructor-argument shape (`string | Signature`). When a
 *     string literal is passed, TypeScript infers `TSig = 'literal string'`;
 *     when a `Signature` is passed, `TSig = Signature`; when a widened
 *     `string` variable is passed, `TSig = string`.
 *   - `TInputs` / `TOutputs` are derived from `TSig` via `InferInputs` /
 *     `InferOutputs` so that only literal-string constructions see narrowed
 *     records, and every other construction falls back to
 *     `Record<string, unknown>`.
 *
 * The runtime behavior is independent of these generics — the source of truth
 * remains the `Signature` object built by `ensurePredictSignature`.
 */
export class Predict<
  TSig extends SignatureInput = Signature,
  TInputs extends Record<string, unknown> = InferInputs<TSig>,
  TOutputs extends Record<string, unknown> = InferOutputs<TSig>,
> extends Module<PredictKwargs<TInputs>, TOutputs> {
  signature: Signature;
  config: Record<string, unknown>;
  lm: BaseLM | null = null;
  demos: readonly Record<string, unknown>[] = Object.freeze([]);
  traces: readonly PredictTrace<TOutputs>[] = Object.freeze([]);
  train: readonly unknown[] = Object.freeze([]);

  constructor(signature: TSig, config: Record<string, unknown> = {}) {
    super();
    this.signature = ensurePredictSignature(signature);
    this.config = normalizeConfig(config);
    markPredictor(this);
  }

  reset(): void {
    this.lm = null;
    this.demos = Object.freeze([]);
    this.traces = Object.freeze([]);
    this.train = Object.freeze([]);
  }

  override call(
    kwargs: PredictKwargs<TInputs> = EMPTY_RECORD as PredictKwargs<TInputs>,
    ...positionalArgs: unknown[]
  ): Prediction<TOutputs> {
    const normalizedKwargs = this.normalizeInvocation(kwargs, positionalArgs);
    return super.call(normalizedKwargs as PredictKwargs<TInputs>);
  }

  override async acall(
    kwargs: PredictKwargs<TInputs> = EMPTY_RECORD as PredictKwargs<TInputs>,
    ...positionalArgs: unknown[]
  ): Promise<Prediction<TOutputs>> {
    const normalizedKwargs = this.normalizeInvocation(kwargs, positionalArgs);
    return super.acall(normalizedKwargs as PredictKwargs<TInputs>);
  }

  preprocess(kwargs: Record<string, unknown> = EMPTY_RECORD): PredictPreprocessResult {
    const raw = snapshotRecord(kwargs);
    const signature = this.resolveSignature(raw);
    const demos = this.resolveDemos(raw);
    const config = this.resolveConfig(raw);
    const lm = this.resolveLm(raw);
    const adapter = this.resolveAdapter();
    this.applyTemperatureAutoAdjust(config, lm);

    const providedInputs = this.extractInputKwargs(raw);
    const filteredInputs = this.filterInputFields(signature, providedInputs);
    const inputs = this.populateDefaults(signature, filteredInputs);

    this.validateInputs(signature, providedInputs, inputs);

    return {
      adapter,
      lm,
      signature,
      demos,
      config,
      inputs,
    };
  }

  resolveSignature(kwargs: Record<string, unknown>): Signature {
    if (!(kwargs.signature === undefined)) {
      const provided = kwargs.signature;
      if (provided instanceof Signature || typeof provided === 'string') {
        return ensurePredictSignature(provided);
      }

      throw new ValueError('Predict signature override must be a Signature or string');
    }

    return this.signature;
  }

  resolveDemos(kwargs: Record<string, unknown>): readonly Record<string, unknown>[] {
    if (!(kwargs.demos === undefined)) {
      const provided = kwargs.demos;
      if (!Array.isArray(provided)) {
        throw new ValueError('Predict demos override must be an array');
      }

      return normalizeDemos(provided as readonly Demo[]);
    }

    return this.demos;
  }

  resolveConfig(kwargs: Record<string, unknown>): Record<string, unknown> {
    return snapshotRecord({
      ...this.config,
      ...normalizeConfig(kwargs.config),
    });
  }

  resolveLm(kwargs: Record<string, unknown>): BaseLM {
    const directLm = kwargs.lm;
    const candidate = directLm ?? this.lm ?? settings.lm;

    if (candidate === null || candidate === undefined) {
      throw new ConfigurationError('No LM is loaded.');
    }

    if (typeof candidate === 'string') {
      throw new ValueError('Predict LM must be an instance of BaseLM, not a string.');
    }

    if (!(candidate instanceof BaseLM)) {
      throw new ValueError(`Predict LM must be an instance of BaseLM, received ${typeof candidate}.`);
    }

    return candidate;
  }

  resolveAdapter(): Adapter {
    const configured = settings.adapter;
    if (configured === null || configured === undefined) {
      return new JSONAdapter();
    }

    if (!(configured instanceof Adapter)) {
      throw new RuntimeError('settings.adapter must be an Adapter instance');
    }

    return configured;
  }

  populateDefaults(
    signature: Signature,
    kwargs: Record<string, unknown>,
  ): Record<string, unknown> {
    const populated = snapshotRecord(kwargs);

    for (const [name, field] of signature.inputFields) {
      if (!(name in populated) && field.default !== undefined) {
        populated[name] = field.default;
      }
    }

    return populated;
  }

  validateInputs(
    signature: Signature,
    rawInputs: Record<string, unknown>,
    resolvedInputs: Record<string, unknown>,
  ): void {
    const expected = [...signature.inputFields.keys()];
    const extras = Object.keys(rawInputs).filter((key) => !signature.inputFields.has(key));
    if (extras.length > 0) {
      console.warn(
        `Input contains fields not in signature. These fields will be ignored: ${extras.join(', ')}. Expected fields: ${expected.join(', ')}.`,
      );
    }

    const missing = expected.filter((key) => !(key in resolvedInputs));
    if (missing.length > 0) {
      const present = expected.filter((key) => key in resolvedInputs);
      console.warn(
        `Not all input fields were provided to Predict. Present: ${present.join(', ')}. Missing: ${missing.join(', ')}.`,
      );
    }
  }

  buildPrediction(
    completions: readonly Readonly<Record<string, unknown>>[],
    signature: Signature,
  ): Prediction<TOutputs> {
    return Prediction.fromCompletions<TOutputs>(completions, signature);
  }

  appendTrace(inputs: Record<string, unknown>, prediction: Prediction<TOutputs>): void {
    this.traces = Object.freeze([
      ...this.traces,
      Object.freeze({
        inputs: snapshotRecord(inputs),
        prediction,
      }),
    ]);
  }

  updateConfig(kwargs: Record<string, unknown>): void {
    this.config = snapshotRecord({
      ...this.config,
      ...normalizeConfig(kwargs),
    });
  }

  getConfig(): Record<string, unknown> {
    return snapshotRecord(this.config);
  }

  override forward(
    kwargs: PredictKwargs<TInputs> = EMPTY_RECORD as PredictKwargs<TInputs>,
    ...positionalArgs: unknown[]
  ): Prediction<TOutputs> {
    const normalizedKwargs = this.normalizeInvocation(kwargs, positionalArgs);
    const { adapter, lm, signature, demos, config, inputs } = this.preprocess(normalizedKwargs);
    const completions = adapter.call(lm, config, signature, demos, inputs);
    const prediction = this.buildPrediction(completions, signature);
    this.appendTrace(inputs, prediction);
    return prediction;
  }

  override async aforward(
    kwargs: PredictKwargs<TInputs> = EMPTY_RECORD as PredictKwargs<TInputs>,
    ...positionalArgs: unknown[]
  ): Promise<Prediction<TOutputs>> {
    const normalizedKwargs = this.normalizeInvocation(kwargs, positionalArgs);
    const { adapter, lm, signature, demos, config, inputs } = this.preprocess(normalizedKwargs);
    const completions = await adapter.acall(lm, config, signature, demos, inputs);
    const prediction = this.buildPrediction(completions, signature);
    this.appendTrace(inputs, prediction);
    return prediction;
  }

  static withDefaultOutput(
    inputDefaults: Record<string, unknown>,
    outputName = 'answer',
  ): Predict {
    const inputs = new Map<string, ReturnType<typeof createField>>();
    for (const [name, value] of Object.entries(inputDefaults)) {
      inputs.set(name, createField({ kind: 'input', name, default: value }));
    }

    const outputs = new Map([
      [outputName, createField({ kind: 'output', name: outputName })],
    ]);

    return new Predict(createSignature(inputs, outputs));
  }

  private normalizeInvocation(
    kwargs: PredictKwargs<TInputs> | Record<string, unknown>,
    positionalArgs: readonly unknown[],
  ): Record<string, unknown> {
    if (positionalArgs.length > 0) {
      throw new ValueError(this.positionalArgsErrorMessage());
    }

    if (!isPlainObject(kwargs)) {
      throw new ValueError(this.positionalArgsErrorMessage());
    }

    return snapshotRecord(kwargs as Record<string, unknown>);
  }

  private positionalArgsErrorMessage(): string {
    return `Positional arguments are not allowed when calling Predict. Use a single object whose keys match the input fields: ${[...this.signature.inputFields.keys()].join(', ')}.`;
  }

  private extractInputKwargs(kwargs: Record<string, unknown>): Record<string, unknown> {
    const inputs: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(kwargs)) {
      if (RESERVED_PREDICT_INPUT_KEYS.has(key)) {
        continue;
      }

      inputs[key] = value;
    }

    return inputs;
  }

  private filterInputFields(
    signature: Signature,
    kwargs: Record<string, unknown>,
  ): Record<string, unknown> {
    const filtered: Record<string, unknown> = {};

    for (const [name] of signature.inputFields) {
      if (name in kwargs) {
        filtered[name] = kwargs[name];
      }
    }

    return filtered;
  }

  /**
   * Sampling multiple completions (`n > 1`) at a near-zero temperature produces
   * near-identical outputs, which defeats the purpose of sampling and wastes
   * tokens. When the caller asks for multiple generations without raising
   * temperature, we bump it to 0.7 so the generations actually differ. Mirrors
   * the same heuristic in upstream DSPy; see `dspy/predict/predict.py`.
   *
   * The 0.15 ceiling is intentionally generous: any temperature above it is
   * treated as "the caller knows what they're doing" and left alone.
   */
  private applyTemperatureAutoAdjust(config: Record<string, unknown>, lm: BaseLM): void {
    const effectiveTemperature = firstDefined(config.temperature, lm.kwargs.temperature);
    const effectiveGenerations = firstDefined(config.n, lm.kwargs.n, lm.kwargs.num_generations, 1);
    const generationCount = Number(effectiveGenerations);
    const temperatureValue = effectiveTemperature === undefined || effectiveTemperature === null
      ? null
      : Number(effectiveTemperature);

    if ((temperatureValue === null || temperatureValue <= 0.15) && generationCount > 1) {
      config.temperature = 0.7;
    }
  }
}
