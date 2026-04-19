/**
 * §4.3 — ChainOfThought: a thin Predict wrapper that prepends reasoning.
 *
 * The generics mirror `Predict`'s: `TSig` accepts a signature literal string
 * or a runtime `Signature` object, and `TInputs`/`TOutputs` are derived so
 * that a literal-string construction gets autocompletion on both sides.
 * `TOutputs` is intersected with `{ reasoning: string }` because CoT always
 * prepends a reasoning output field at runtime.
 */

import { createField } from './field.js';
import { Module } from './module.js';
import { Predict, type PredictKwargs } from './predict.js';
import { type Prediction } from './prediction.js';
import { ensureSignature, prependField, type Signature } from './signature.js';
import type { Flatten, InferInputs, InferOutputs, SignatureInput } from './signature_types.js';

const EMPTY_RECORD = Object.freeze({}) as Record<string, unknown>;

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

    // The extended signature is a runtime `Signature`, not a literal string,
    // so we explicitly specialize the inner `Predict` with `Signature` as
    // `TSig`. `TInputs` and `TOutputs` still flow through from the caller's
    // literal-string inference.
    this.predict = new Predict<Signature, TInputs, TOutputs>(extendedSignature, config);
  }

  override forward(kwargs: PredictKwargs<TInputs> = EMPTY_RECORD as PredictKwargs<TInputs>): Prediction<TOutputs> {
    return this.predict.call(kwargs);
  }

  override aforward(kwargs: PredictKwargs<TInputs> = EMPTY_RECORD as PredictKwargs<TInputs>): Promise<Prediction<TOutputs>> {
    return this.predict.acall(kwargs);
  }
}
