/**
 * §1.2 — Field: a single named, typed slot within a Signature.
 *
 * Invariants:
 *   - kind ∈ {"input", "output"}
 *   - prefix ≠ ""
 *   - name matches [a-zA-Z_][a-zA-Z0-9_]*
 */

import {
  isFieldKind,
  isTypeTag,
  type FieldKind,
  type TypeTag,
} from './types.js';
import { ValueError } from './exceptions.js';
import { inferPrefix } from './infer_prefix.js';
import { ownedValueEquals, snapshotOwnedValue } from './owned_value.js';

const FIELD_NAME_RE = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
const EMPTY_TYPE_ARGS = Object.freeze([] as TypeTag[]);
const EMPTY_CONSTRAINTS = Object.freeze([] as string[]);

export interface FieldInit {
  readonly kind: FieldKind;
  readonly name: string;
  readonly typeTag?: TypeTag;
  readonly typeArgs?: readonly TypeTag[];
  readonly description?: string;
  readonly prefix?: string;
  readonly constraints?: readonly string[];
  readonly default?: unknown;
  readonly isTypeUndefined?: boolean;
}

function freezeArray<T>(
  values: readonly T[] | undefined,
  emptyValue: readonly T[],
): readonly T[] {
  if (values === undefined) {
    return emptyValue;
  }

  return Object.freeze([...values]);
}

function arrayEquals<T>(a: readonly T[], b: readonly T[]): boolean {
  if (a.length !== b.length) {
    return false;
  }

  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }

  return true;
}

export class Field {
  readonly kind: FieldKind;
  readonly name: string;
  readonly typeTag: TypeTag;
  readonly description: string;
  readonly prefix: string;
  readonly default: unknown;
  readonly isTypeUndefined: boolean;

  readonly #typeArgs: readonly TypeTag[];
  readonly #constraints: readonly string[];

  private constructor(init: {
    readonly kind: FieldKind;
    readonly name: string;
    readonly typeTag: TypeTag;
    readonly typeArgs: readonly TypeTag[];
    readonly description: string;
    readonly prefix: string;
    readonly constraints: readonly string[];
    readonly default: unknown;
    readonly isTypeUndefined: boolean;
  }) {
    this.kind = init.kind;
    this.name = init.name;
    this.typeTag = init.typeTag;
    this.#typeArgs = init.typeArgs;
    this.description = init.description;
    this.prefix = init.prefix;
    this.#constraints = init.constraints;
    this.default = init.default;
    this.isTypeUndefined = init.isTypeUndefined;
  }

  static create(init: FieldInit): Field {
    if (!isFieldKind(init.kind)) {
      throw new ValueError(`Invalid field kind "${String(init.kind)}"`);
    }

    if (!FIELD_NAME_RE.test(init.name)) {
      throw new ValueError(
        `Invalid field name "${init.name}": must match [a-zA-Z_][a-zA-Z0-9_]*`,
      );
    }

    if (init.isTypeUndefined === true && init.typeTag !== undefined) {
      throw new ValueError(
        `Field "${init.name}" cannot declare both typeTag and isTypeUndefined=true`,
      );
    }

    if (init.isTypeUndefined === false && init.typeTag === undefined) {
      throw new ValueError(
        `Field "${init.name}" cannot set isTypeUndefined=false without an explicit typeTag`,
      );
    }

    const typeTag = init.typeTag ?? 'str';
    if (!isTypeTag(typeTag)) {
      throw new ValueError(`Invalid type tag "${String(typeTag)}" for field "${init.name}"`);
    }

    const typeArgs = freezeArray(init.typeArgs, EMPTY_TYPE_ARGS);
    for (const typeArg of typeArgs) {
      if (!isTypeTag(typeArg)) {
        throw new ValueError(
          `Invalid type argument "${String(typeArg)}" for field "${init.name}"`,
        );
      }
    }

    const constraints = freezeArray(init.constraints, EMPTY_CONSTRAINTS);
    const prefix = (init.prefix ?? inferPrefix(init.name)).trim();
    if (prefix === '') {
      throw new ValueError(`Field "${init.name}" resolved to empty prefix`);
    }

    return new Field({
      kind: init.kind,
      name: init.name,
      typeTag,
      typeArgs,
      description: init.description ?? '',
      prefix,
      constraints,
      default: snapshotOwnedValue(init.default),
      isTypeUndefined: init.isTypeUndefined ?? (init.typeTag === undefined),
    });
  }

  get typeArgs(): readonly TypeTag[] {
    return this.#typeArgs;
  }

  get constraints(): readonly string[] {
    return this.#constraints;
  }

  equals(other: Field): boolean {
    return (
      this.kind === other.kind
      && this.name === other.name
      && this.typeTag === other.typeTag
      && arrayEquals(this.#typeArgs, other.#typeArgs)
      && this.description === other.description
      && this.prefix === other.prefix
      && arrayEquals(this.#constraints, other.#constraints)
      && ownedValueEquals(this.default, other.default)
      && this.isTypeUndefined === other.isTypeUndefined
    );
  }
}

export function createField(init: FieldInit): Field {
  return Field.create(init);
}
