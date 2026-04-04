/**
 * §4 — Predict pipeline.
 */

import { Adapter, ChatAdapter, type Demo, JSONAdapter } from './adapter.js';
import { Example } from './example.js';
import { createField } from './field.js';
import { BaseLM } from './lm.js';
import { Module, markPredictor } from './module.js';
import { snapshotRecord } from './owned_value.js';
import { Prediction } from './prediction.js';
import { settings } from './settings.js';
import {
  Signature,
  createSignature,
  signatureFromString,
} from './signature.js';

export interface PredictTrace {
  readonly inputs: Record<string, unknown>;
  readonly prediction: Prediction;
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

function ensureSignature(value: Signature | string): Signature {
  if (value instanceof Signature) {
    return value;
  }

  return signatureFromString(value);
}

function ensurePredictSignature(value: Signature | string): Signature {
  const signature = ensureSignature(value);
  const reservedInputs = [...signature.inputFields.keys()].filter((name) => (
    RESERVED_PREDICT_INPUT_KEYS.has(name)
  ));

  if (reservedInputs.length > 0) {
    throw new Error(
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
    throw new Error('Predict config must be a plain object');
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

export class Predict extends Module {
  signature: Signature;
  config: Record<string, unknown>;
  lm: BaseLM | null = null;
  demos: readonly Record<string, unknown>[] = Object.freeze([]);
  traces: readonly PredictTrace[] = Object.freeze([]);
  train: readonly unknown[] = Object.freeze([]);

  constructor(signature: Signature | string, config: Record<string, unknown> = {}) {
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
    kwargs: Record<string, unknown> = EMPTY_RECORD,
    ...positionalArgs: unknown[]
  ): Prediction {
    const normalizedKwargs = this.normalizeInvocation(kwargs, positionalArgs);
    return super.call(normalizedKwargs);
  }

  override async acall(
    kwargs: Record<string, unknown> = EMPTY_RECORD,
    ...positionalArgs: unknown[]
  ): Promise<Prediction> {
    const normalizedKwargs = this.normalizeInvocation(kwargs, positionalArgs);
    return super.acall(normalizedKwargs);
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

      throw new Error('Predict signature override must be a Signature or string');
    }

    return this.signature;
  }

  resolveDemos(kwargs: Record<string, unknown>): readonly Record<string, unknown>[] {
    if (!(kwargs.demos === undefined)) {
      const provided = kwargs.demos;
      if (!Array.isArray(provided)) {
        throw new Error('Predict demos override must be an array');
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
      throw new Error('No LM is loaded.');
    }

    if (typeof candidate === 'string') {
      throw new Error('Predict LM must be an instance of BaseLM, not a string.');
    }

    if (!(candidate instanceof BaseLM)) {
      throw new Error(`Predict LM must be an instance of BaseLM, received ${typeof candidate}.`);
    }

    return candidate;
  }

  resolveAdapter(): Adapter {
    const configured = settings.adapter;
    if (configured === null || configured === undefined) {
      return new JSONAdapter();
    }

    if (!(configured instanceof Adapter)) {
      throw new Error('settings.adapter must be an Adapter instance');
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
  ): Prediction {
    return Prediction.fromCompletions(completions, signature);
  }

  appendTrace(inputs: Record<string, unknown>, prediction: Prediction): void {
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
    kwargs: Record<string, unknown> = EMPTY_RECORD,
    ...positionalArgs: unknown[]
  ): Prediction {
    const normalizedKwargs = this.normalizeInvocation(kwargs, positionalArgs);
    const { adapter, lm, signature, demos, config, inputs } = this.preprocess(normalizedKwargs);
    const completions = adapter.call(lm, config, signature, demos, inputs);
    const prediction = this.buildPrediction(completions, signature);
    this.appendTrace(inputs, prediction);
    return prediction;
  }

  override async aforward(
    kwargs: Record<string, unknown> = EMPTY_RECORD,
    ...positionalArgs: unknown[]
  ): Promise<Prediction> {
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
    kwargs: Record<string, unknown>,
    positionalArgs: readonly unknown[],
  ): Record<string, unknown> {
    if (positionalArgs.length > 0) {
      throw new Error(this.positionalArgsErrorMessage());
    }

    if (!isPlainObject(kwargs)) {
      throw new Error(this.positionalArgsErrorMessage());
    }

    return snapshotRecord(kwargs);
  }

  private positionalArgsErrorMessage(): string {
    return `Positional arguments are not allowed when calling Predict. Use a single object whose keys match the input fields: ${[...this.signature.inputFields.keys()].join(', ')}.`;
  }

  private extractInputKwargs(kwargs: Record<string, unknown>): Record<string, unknown> {
    const inputs: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(kwargs)) {
      if (key === 'signature' || key === 'demos' || key === 'config' || key === 'lm') {
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

export { Adapter, ChatAdapter, JSONAdapter };
