/** §1.1 — Primitive scalars for the DSPy type universe. */

export const FIELD_KINDS = ['input', 'output'] as const;

export type FieldKind = (typeof FIELD_KINDS)[number];

export function isFieldKind(value: string): value is FieldKind {
  return (FIELD_KINDS as readonly string[]).includes(value);
}

export const TYPE_TAGS = [
  'str',
  'int',
  'float',
  'bool',
  'list',
  'dict',
  'literal',
  'enum',
  'optional',
  'union',
  'custom',
] as const;

export type TypeTag = (typeof TYPE_TAGS)[number];

export function isTypeTag(value: string): value is TypeTag {
  return (TYPE_TAGS as readonly string[]).includes(value);
}

export type Role = 'system' | 'user' | 'assistant' | 'developer';

export type ModelType = 'chat' | 'responses';

export type AdapterKind = 'chat' | 'json';

/** Structural minimum for objects stored as the LM in settings. */
export interface LMLike {
  readonly model: string;
}

/** Structural minimum for objects stored as the adapter in settings. */
export interface AdapterLike {
  readonly useNativeFunctionCalling: boolean;
}
