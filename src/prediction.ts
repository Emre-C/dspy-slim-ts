/**
 * §2.2 — Prediction: a completion-backed result container with numeric protocol.
 *
 * Invariants:
 *   - _inputKeys are always absent from the public API
 *   - Numeric protocol requires a "score" field
 *   - Completions: all arrays are the same length
 */

import { KeyError, RuntimeError, ValueError } from './exceptions.js';
import { serializeOwnedValue, snapshotOwnedValue, snapshotRecord } from './owned_value.js';
import { Signature } from './signature.js';

const DSPY_PREFIX_RE = /^dspy_/;

function cloneStore(data: Record<string, unknown>): Map<string, unknown> {
  return new Map(Object.entries(snapshotRecord(data)));
}

function cloneCompletionsMap(
  completions: ReadonlyMap<string, readonly unknown[]>,
): Map<string, readonly unknown[]> {
  const cloned = new Map<string, readonly unknown[]>();

  for (const [key, values] of completions) {
    cloned.set(
      key,
      Object.freeze(values.map((value) => snapshotOwnedValue(value))),
    );
  }

  return cloned;
}

function normalizeCompletionInput(
  completions:
    | Readonly<Record<string, readonly unknown[]>>
    | readonly Readonly<Record<string, unknown>>[],
): Map<string, readonly unknown[]> {
  if (Array.isArray(completions)) {
    const normalized = new Map<string, unknown[]>();

    for (const [index, completion] of completions.entries()) {
      if (completion === null || typeof completion !== 'object' || Array.isArray(completion)) {
        throw new ValueError(
          `Completion at index ${index} must be a plain object of field values`,
        );
      }

      for (const [key, value] of Object.entries(completion)) {
        const values = normalized.get(key);
        if (values) {
          values.push(value);
        } else {
          normalized.set(key, [value]);
        }
      }
    }

    return new Map(normalized);
  }

  const normalized = new Map<string, readonly unknown[]>();

  for (const [key, value] of Object.entries(completions)) {
    if (!Array.isArray(value)) {
      throw new ValueError(`Completion field "${key}" must be an array of values`);
    }

    normalized.set(key, value);
  }

  return normalized;
}

// ---------------------------------------------------------------------------
// Completions
// ---------------------------------------------------------------------------

export class Completions {
  private readonly _completions: ReadonlyMap<string, readonly unknown[]>;
  readonly signature: Signature | null;

  constructor(
    completions: ReadonlyMap<string, readonly unknown[]>,
    signature: Signature | null = null,
  ) {
    if (signature !== null && !(signature instanceof Signature)) {
      throw new ValueError('Completions.signature must be a Signature or null');
    }

    // Validate: all arrays same length.
    let expectedLen: number | null = null;
    const cloned = cloneCompletionsMap(completions);
    for (const [key, arr] of cloned) {
      if (expectedLen === null) {
        expectedLen = arr.length;
      } else if (arr.length !== expectedLen) {
        throw new ValueError(
          `Completions array length mismatch: "${key}" has ${arr.length}, expected ${expectedLen}`,
        );
      }
    }

    this._completions = cloned;
    this.signature = signature;
  }

  get length(): number {
    const first = this._completions.values().next();
    return first.done ? 0 : first.value.length;
  }

  get(key: string): readonly unknown[] | undefined {
    return this._completions.get(key);
  }

  keys(): IterableIterator<string> {
    return this._completions.keys();
  }

  toDict(): Record<string, readonly unknown[]> {
    const result: Record<string, readonly unknown[]> = {};

    for (const [key, values] of this._completions) {
      result[key] = values.map((value) => serializeOwnedValue(value));
    }

    return result;
  }

  toJSON(): Record<string, readonly unknown[]> {
    return this.toDict();
  }
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

/**
 * Completion-backed result container with numeric protocol and optional
 * advisory output typing.
 *
 * The `TOutputs` generic is a **type-level annotation only**. The runtime
 * store is untyped: whatever keys and values were passed to
 * `Prediction.create` or `Prediction.fromCompletions` are stored verbatim.
 * Callers with a string-literal `Signature` receive `TOutputs` propagated
 * from `Predict<'q -> a'>` so `getTyped('a')` returns `string` at compile
 * time, but nothing at runtime enforces this shape.
 */
export class Prediction<TOutputs extends Record<string, unknown> = Record<string, unknown>> {
  readonly completions: Completions | null;
  readonly #store: Map<string, unknown>;

  protected constructor(
    data: Record<string, unknown>,
    completions: Completions | null,
  ) {
    this.#store = cloneStore(data);
    this.completions = completions;
  }

  /**
   * Construct a `Prediction` with an advisory `TOut` type. The generic is
   * intentionally decoupled from `data`: LM completions arrive as
   * unstructured records, and forcing a runtime check would either duplicate
   * the adapter's parsing contract or produce false-positive type errors at
   * every call site. Pass `TOut` when you know (or assert) the data shape
   * matches the declared outputs of the originating `Signature`.
   */
  static create<TOut extends Record<string, unknown> = Record<string, unknown>>(
    data: Record<string, unknown>,
  ): Prediction<TOut> {
    return new Prediction<TOut>(data, null);
  }

  static fromCompletions<TOut extends Record<string, unknown> = Record<string, unknown>>(
    completionsData:
      | Readonly<Record<string, readonly unknown[]>>
      | readonly Readonly<Record<string, unknown>>[],
    signature: Signature | null = null,
  ): Prediction<TOut> {
    const completionsMap = normalizeCompletionInput(completionsData);
    const store: Record<string, unknown> = {};

    for (const [key, values] of completionsMap) {
      store[key] = values[0];
    }

    const completions = new Completions(completionsMap, signature);
    return new Prediction<TOut>(store, completions);
  }

  // --- Numeric protocol (requires "score" field) ---

  get(key: string): unknown {
    if (!this.#store.has(key)) {
      throw new KeyError(`Key "${key}" not found in Prediction`);
    }

    return this.#store.get(key);
  }

  /**
   * Typed counterpart to `get`. Throws `KeyError` on a missing key exactly
   * like `get`, so the caller does not have to distinguish "key absent" from
   * "value was undefined" at runtime. The returned type is the advisory
   * `TOutputs[K]`; remember that the store is not validated against
   * `TOutputs`, so the claim is structural, not load-bearing.
   */
  getTyped<K extends keyof TOutputs & string>(key: K): TOutputs[K] {
    if (!this.#store.has(key)) {
      throw new KeyError(`Key "${key}" not found in Prediction`);
    }

    return this.#store.get(key) as TOutputs[K];
  }

  getOr(key: string, defaultValue: unknown): unknown {
    return this.#store.has(key) ? this.#store.get(key) : defaultValue;
  }

  has(key: string): boolean {
    return this.#store.has(key);
  }

  keys(includeDspy = false): string[] {
    const keys: string[] = [];
    for (const key of this.#store.keys()) {
      if (includeDspy || !DSPY_PREFIX_RE.test(key)) {
        keys.push(key);
      }
    }

    return keys;
  }

  len(): number {
    let count = 0;
    for (const key of this.#store.keys()) {
      if (!DSPY_PREFIX_RE.test(key)) {
        count++;
      }
    }

    return count;
  }

  toDict(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const [key, value] of this.#store) {
      result[key] = serializeOwnedValue(value);
    }

    return result;
  }

  toJSON(): Record<string, unknown> {
    return this.toDict();
  }

  snapshot(): Prediction<TOutputs> {
    if (this.completions === null) {
      return Prediction.create<TOutputs>(this.toDict());
    }

    return Prediction.fromCompletions<TOutputs>(this.completions.toDict(), this.completions.signature);
  }

  toFloat(): number {
    if (!this.has('score')) {
      throw new RuntimeError(
        'Prediction does not have a \'score\' field — cannot convert to number',
      );
    }

    const numericScore = Number(this.get('score'));
    if (Number.isNaN(numericScore)) {
      throw new ValueError('Prediction score must be numeric to participate in numeric operations');
    }

    return numericScore;
  }

  valueOf(): number {
    return this.toFloat();
  }

  toString(): string {
    return `Prediction(${JSON.stringify(this.toDict())})`;
  }

  [Symbol.toPrimitive](hint: string): number | string {
    if (hint === 'string') {
      return this.toString();
    }

    return this.toFloat();
  }

  add(other: number | Prediction): number {
    const a = this.toFloat();
    const b = other instanceof Prediction ? other.toFloat() : other;
    return a + b;
  }

  div(divisor: number | Prediction): number {
    const a = this.toFloat();
    const b = divisor instanceof Prediction ? divisor.toFloat() : divisor;
    return a / b;
  }

  lt(other: number | Prediction): boolean {
    const a = this.toFloat();
    const b = other instanceof Prediction ? other.toFloat() : other;
    return a < b;
  }

  le(other: number | Prediction): boolean {
    const a = this.toFloat();
    const b = other instanceof Prediction ? other.toFloat() : other;
    return a <= b;
  }

  gt(other: number | Prediction): boolean {
    const a = this.toFloat();
    const b = other instanceof Prediction ? other.toFloat() : other;
    return a > b;
  }

  ge(other: number | Prediction): boolean {
    const a = this.toFloat();
    const b = other instanceof Prediction ? other.toFloat() : other;
    return a >= b;
  }

  /** Predictions never have input keys (spec S6). Duck-type compat with Example. */
  hasInputKeys(): false {
    return false;
  }
}
