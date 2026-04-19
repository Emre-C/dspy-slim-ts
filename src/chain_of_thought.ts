/** §4.3 — Predict wrapper that prepends a `reasoning` output field; `TOutputs` includes `reasoning: string`. */

import { createField } from './field.js';
import { Module } from './module.js';
import { Predict, type PredictKwargs } from './predict.js';
import { type Prediction } from './prediction.js';
import { ensureSignature, prependField, type Signature } from './signature.js';
import type { Flatten, InferInputs, InferOutputs, SignatureInput } from './signature_types.js';

const EMPTY_RECORD = Object.freeze({}) as Record<string, unknown>;

/**
 * Inner `Predict` is built from the widened `Signature` (after `prependField`);
 * class generics still anchor inference to the caller’s `TSig`.
 */
export class ChainOfThought<
  TSig extends SignatureInput = Signature,
  TInputs extends Record<string, unknown> = InferInputs<TSig>,
  TOutputs extends Record<string, unknown> = Flatten<InferOutputs<TSig> & { reasoning: string }>,
> extends Module<PredictKwargs<TInputs>, TOutputs> {
  readonly predict: Predict<Signature, TInputs, TOutputs>;

  constructor(signature: TSig, config: Record<string, unknown> = {}) {
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

    this.predict = new Predict<Signature, TInputs, TOutputs>(extendedSignature, config);
  }

  override forward(kwargs: PredictKwargs<TInputs> = EMPTY_RECORD as PredictKwargs<TInputs>): Prediction<TOutputs> {
    return this.predict.call(kwargs);
  }

  override aforward(kwargs: PredictKwargs<TInputs> = EMPTY_RECORD as PredictKwargs<TInputs>): Promise<Prediction<TOutputs>> {
    return this.predict.acall(kwargs);
  }
}
