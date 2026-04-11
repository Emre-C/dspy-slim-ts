/**
 * GEPA trace capture and metric normalization helpers.
 */

import type { EvaluationRow } from './evaluate.js';
import { ValueError } from './exceptions.js';
import { isPlainObject } from './guards.js';
import type { Module } from './module.js';
import { Prediction } from './prediction.js';
import type {
  MetricRecord,
  PredictorTarget,
  PredictorTrace,
  ReflectiveDatum,
} from './gepa_types.js';

function stableTokenId(token: string): number {
  let hash = 0;
  for (let index = 0; index < token.length; index += 1) {
    hash = ((hash * 31) + token.charCodeAt(index)) >>> 0;
  }
  return hash;
}

function parsePredictorPath(name: string, predictorIndex: number): readonly number[] {
  const path: number[] = [];
  const tokenPattern = /([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]|\['([^']+)'\]/g;
  let match: RegExpExecArray | null = tokenPattern.exec(name);
  while (match !== null) {
    if (match[2] !== undefined) {
      path.push(Number.parseInt(match[2], 10));
    } else {
      const token = match[1] ?? match[3];
      if (token) {
        path.push(stableTokenId(token));
      }
    }
    match = tokenPattern.exec(name);
  }

  if (path.length === 0) {
    path.push(predictorIndex);
  }

  return Object.freeze(path);
}

function normalizeSubscores(value: unknown): readonly number[] {
  if (Array.isArray(value)) {
    return Object.freeze(value.filter((entry): entry is number => (
      typeof entry === 'number' && Number.isFinite(entry)
    )));
  }

  if (isPlainObject(value)) {
    const numericValues = Object.values(value).filter((entry): entry is number => (
      typeof entry === 'number' && Number.isFinite(entry)
    ));
    return Object.freeze(numericValues);
  }

  return Object.freeze([]);
}

export function normalizeMetricRecord(
  value: unknown,
  failureScore = 0,
): MetricRecord {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Object.freeze({
      score: value,
      subscores: Object.freeze([]),
      feedback: null,
      failed: false,
    });
  }

  if (value instanceof Prediction) {
    let score: number | null = null;
    try {
      score = value.toFloat();
    } catch {
      score = null;
    }

    const feedback = value.has('feedback') ? value.get('feedback') : null;
    const subscores = value.has('subscores')
      ? normalizeSubscores(value.get('subscores'))
      : Object.freeze([]);

    return Object.freeze({
      score,
      subscores,
      feedback,
      failed: score === null,
    });
  }

  if (isPlainObject(value)) {
    const numericScore = typeof value.score === 'number' && Number.isFinite(value.score)
      ? value.score
      : null;

    const failed = typeof value.failed === 'boolean'
      ? value.failed
      : numericScore === null;

    return Object.freeze({
      score: numericScore ?? (failed ? failureScore : null),
      subscores: normalizeSubscores(value.subscores),
      feedback: value.feedback ?? null,
      failed,
    });
  }

  return Object.freeze({
    score: failureScore,
    subscores: Object.freeze([]),
    feedback: null,
    failed: true,
  });
}

export function projectPredictorTargets(
  program: Module,
): readonly PredictorTarget[] {
  return Object.freeze(program.namedPredictors().map(([name], predictorIndex) => (
    Object.freeze({
      targetId: predictorIndex,
      predictorIndex,
      path: parsePredictorPath(name, predictorIndex),
    })
  )));
}

export function capturePredictorTraces(
  program: Module,
  rows: readonly EvaluationRow[],
  options: {
    readonly failureScore?: number;
    readonly histories?: Readonly<Record<number, unknown>>;
  } = {},
): readonly PredictorTrace[] {
  const targets = projectPredictorTargets(program);
  const traces: PredictorTrace[] = [];
  const failureScore = options.failureScore ?? 0;

  for (const [exampleId, row] of rows.entries()) {
    const [example, prediction, score] = row;
    const exampleInput = example.inputs().toDict();
    const predictionOutput = prediction.toDict();
    const metric = normalizeMetricRecord(score, failureScore);
    const history = (options.histories?.[exampleId] ?? null) as PredictorTrace['history'];

    for (const target of targets) {
      traces.push(Object.freeze({
        exampleId,
        target,
        input: exampleInput,
        output: predictionOutput,
        metric,
        history,
      }));
    }
  }

  return Object.freeze(traces);
}

export function materializeReflectiveDataset(
  traces: readonly PredictorTrace[],
): readonly ReflectiveDatum[] {
  return Object.freeze(traces.map((trace, datumId) => Object.freeze({
    datumId,
    target: trace.target,
    trace,
  })));
}

export function ensureNonEmptyTargets(
  targets: readonly PredictorTarget[],
): void {
  if (targets.length === 0) {
    throw new ValueError(
      'GEPA requires at least one instruction-bearing predictor target.',
    );
  }
}
