/**
 * §4.3 — ChainOfThought: a thin Predict wrapper that prepends reasoning.
 */

import { createField } from './field.js';
import { Module } from './module.js';
import { Predict } from './predict.js';
import { type Prediction } from './prediction.js';
import { ensureSignature, prependField, type Signature } from './signature.js';

const EMPTY_RECORD = Object.freeze({}) as Record<string, unknown>;

export class ChainOfThought extends Module {
  predict: Predict;

  constructor(signature: Signature | string, config: Record<string, unknown> = {}) {
    super();

    const extendedSignature = prependField(
      ensureSignature(signature),
      createField({
        kind: 'output',
        name: 'reasoning',
        typeTag: 'str',
        description: '${reasoning}',
        isTypeUndefined: false,
      }),
    );

    this.predict = new Predict(extendedSignature, config);
  }

  override forward(kwargs: Record<string, unknown> = EMPTY_RECORD): Prediction {
    return this.predict.call(kwargs);
  }

  override aforward(kwargs: Record<string, unknown> = EMPTY_RECORD): Promise<Prediction> {
    return this.predict.acall(kwargs);
  }
}
