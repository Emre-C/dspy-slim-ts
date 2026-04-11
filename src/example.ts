/**
 * §2.1 — Example: a keyed data container for training/evaluation data.
 *
 * Invariants:
 *   - inputs(e) ∪ labels(e) ⊇ {k ∈ keys(e) | k ∈ _inputKeys}
 *   - inputs(e).keys ∩ labels(e).keys == ∅
 *   - len(e) excludes keys matching ^dspy_
 *   - withInputs returns a copy; original is unmodified
 */

import {
  ownedValueEquals,
  serializeOwnedValue,
  snapshotRecord,
  snapshotOwnedValue,
} from './owned_value.js';
import { KeyError, RuntimeError } from './exceptions.js';

const DSPY_PREFIX_RE = /^dspy_/;

export class Example {
  private readonly _store: Map<string, unknown>;
  private readonly _inputKeys: Set<string> | null;

  constructor(
    data?: Record<string, unknown>,
    inputKeys?: ReadonlySet<string> | null,
  ) {
    this._store = new Map(Object.entries(snapshotRecord(data)));
    this._inputKeys = inputKeys ? new Set(inputKeys) : null;
  }

  get(key: string): unknown {
    if (!this._store.has(key)) {
      throw new KeyError(`Key "${key}" not found in Example`);
    }
    return this._store.get(key);
  }

  getOr(key: string, defaultValue: unknown): unknown {
    return this._store.has(key) ? this._store.get(key) : defaultValue;
  }

  set(key: string, value: unknown): void {
    this._store.set(key, snapshotOwnedValue(value));
  }

  has(key: string): boolean {
    return this._store.has(key);
  }

  keys(includeDspy = false): string[] {
    const result: string[] = [];
    for (const k of this._store.keys()) {
      if (includeDspy || !DSPY_PREFIX_RE.test(k)) {
        result.push(k);
      }
    }
    return result;
  }

  len(): number {
    let count = 0;
    for (const k of this._store.keys()) {
      if (!DSPY_PREFIX_RE.test(k)) count++;
    }
    return count;
  }

  withInputs(...keys: string[]): Example {
    const copy = this._copy();
    return new Example(copy, new Set(keys));
  }

  inputs(): Example {
    if (this._inputKeys === null) {
      throw new RuntimeError('Inputs have not been set — call withInputs() first');
    }
    const data: Record<string, unknown> = {};
    for (const k of this._inputKeys) {
      if (this._store.has(k)) {
        data[k] = this._store.get(k);
      }
    }
    return new Example(data, new Set(this._inputKeys));
  }

  labels(): Example {
    if (this._inputKeys === null) {
      throw new RuntimeError('Inputs have not been set — call withInputs() first');
    }
    const data: Record<string, unknown> = {};
    for (const [k, v] of this._store) {
      if (!this._inputKeys.has(k)) {
        data[k] = v;
      }
    }
    return new Example(data, null);
  }

  toDict(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const [k, v] of this._store) {
      result[k] = serializeOwnedValue(v);
    }
    return result;
  }

  toJSON(): Record<string, unknown> {
    return this.toDict();
  }

  snapshot(): Example {
    return new Example(this.toDict(), this._inputKeys);
  }

  copy(overrides?: Record<string, unknown>): Example {
    const data = this._copy();
    if (overrides) {
      for (const [k, v] of Object.entries(overrides)) {
        data[k] = v;
      }
    }
    return new Example(data, this._inputKeys);
  }

  without(...keys: string[]): Example {
    const removeSet = new Set(keys);
    const data: Record<string, unknown> = {};
    for (const [k, v] of this._store) {
      if (!removeSet.has(k)) {
        data[k] = v;
      }
    }
    return new Example(data, this._inputKeys);
  }

  equals(other: Example): boolean {
    if (this._store.size !== other._store.size) return false;
    for (const [k, v] of this._store) {
      if (!other._store.has(k)) return false;
      if (!ownedValueEquals(v, other._store.get(k))) return false;
    }

    if (this._inputKeys === null || other._inputKeys === null) {
      return this._inputKeys === other._inputKeys;
    }

    if (this._inputKeys.size !== other._inputKeys.size) {
      return false;
    }

    for (const key of this._inputKeys) {
      if (!other._inputKeys.has(key)) {
        return false;
      }
    }

    return true;
  }

  /** Whether _inputKeys is set (not null). */
  hasInputKeys(): boolean {
    return this._inputKeys !== null;
  }

  private _copy(): Record<string, unknown> {
    return this.toDict();
  }
}
