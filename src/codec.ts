/**
 * Unified type coercion and schema-type mapping.
 *
 * Every module that needs to coerce runtime values to typed slots (Field,
 * Tool arg, Adapter parse) imports from here.  One coercion path, one set
 * of error messages, one place to maintain.
 */

import { ValueError } from './exceptions.js';
import { isPlainObject } from './guards.js';
import { snapshotOwnedValue } from './owned_value.js';
import type { TypeTag } from './types.js';

// ---------------------------------------------------------------------------
// JSON-Schema ↔ TypeTag mapping
// ---------------------------------------------------------------------------

export type JSONSchemaType = 'string' | 'integer' | 'number' | 'boolean' | 'array' | 'object';

const TYPE_TAG_TO_SCHEMA: Readonly<Record<string, JSONSchemaType>> = {
  str: 'string',
  int: 'integer',
  float: 'number',
  bool: 'boolean',
  list: 'array',
  dict: 'object',
};

const SCHEMA_TO_TYPE_TAG: Readonly<Record<string, TypeTag>> = {
  string: 'str',
  integer: 'int',
  number: 'float',
  boolean: 'bool',
  array: 'list',
  object: 'dict',
};

export function typeTagToSchemaType(typeTag: TypeTag): JSONSchemaType | undefined {
  return TYPE_TAG_TO_SCHEMA[typeTag];
}

export function schemaTypeToTypeTag(schemaType: JSONSchemaType | undefined): TypeTag {
  if (schemaType === undefined) {
    return 'custom';
  }

  return SCHEMA_TO_TYPE_TAG[schemaType] ?? 'custom';
}

// ---------------------------------------------------------------------------
// Primitive coercion
// ---------------------------------------------------------------------------

export function coerceBoolean(value: unknown): boolean {
  if (typeof value === 'boolean') {
    return value;
  }

  if (typeof value === 'string') {
    const lower = value.trim().toLowerCase();
    if (lower === 'true' || lower === '1') {
      return true;
    }
    if (lower === 'false' || lower === '0') {
      return false;
    }
  }

  if (value === 1) {
    return true;
  }

  if (value === 0) {
    return false;
  }

  throw new ValueError(`Cannot coerce ${String(value)} to bool`);
}

export function coerceNumber(value: unknown, kind: 'int' | 'float'): number {
  const numeric = typeof value === 'number' ? value : Number(String(value).trim());

  if (!Number.isFinite(numeric)) {
    throw new ValueError(`Cannot coerce ${String(value)} to ${kind}`);
  }

  if (kind === 'int' && !Number.isInteger(numeric)) {
    throw new ValueError(`Expected integer for ${String(value)}`);
  }

  return numeric;
}

export function coerceJsonContainer(value: unknown, kind: 'list' | 'dict'): unknown {
  if (kind === 'list' && Array.isArray(value)) {
    return snapshotOwnedValue(value);
  }

  if (kind === 'dict' && isPlainObject(value)) {
    return snapshotOwnedValue(value);
  }

  if (typeof value !== 'string') {
    throw new ValueError(`Cannot coerce ${String(value)} to ${kind}`);
  }

  const parsed: unknown = JSON.parse(value);

  if (kind === 'list' && Array.isArray(parsed)) {
    return snapshotOwnedValue(parsed);
  }

  if (kind === 'dict' && isPlainObject(parsed)) {
    return snapshotOwnedValue(parsed);
  }

  throw new ValueError(`Cannot coerce ${String(value)} to ${kind}`);
}

// ---------------------------------------------------------------------------
// High-level: coerce any value to the slot described by a TypeTag
// ---------------------------------------------------------------------------

export function coerceFieldValue(typeTag: TypeTag, value: unknown): unknown {
  switch (typeTag) {
    case 'str':
      return typeof value === 'string' ? value : String(value);
    case 'int':
      return coerceNumber(value, 'int');
    case 'float':
      return coerceNumber(value, 'float');
    case 'bool':
      return coerceBoolean(value);
    case 'list':
      return coerceJsonContainer(value, 'list');
    case 'dict':
      return coerceJsonContainer(value, 'dict');
    case 'literal':
    case 'enum':
    case 'optional':
    case 'union':
    case 'custom':
      return snapshotOwnedValue(value);
  }
}
