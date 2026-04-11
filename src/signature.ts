/**
 * §1.3 — Signature: a typed contract of input/output fields.
 * §1.4 — Signature parsing from string form.
 *
 * Immutable value objects. All mutation operations return new instances.
 *
 * Invariants:
 *   - input_fields ∩ output_fields keys are disjoint
 *   - fields = inputs (insertion order) then outputs (insertion order)
 */

import { Field, createField } from './field.js';
import { InvariantError, ValueError } from './exceptions.js';
import { splitTopLevel } from './split.js';
import { isTypeTag, type FieldKind, type TypeTag } from './types.js';

// ---------------------------------------------------------------------------
// Signature
// ---------------------------------------------------------------------------

export interface SignatureOptions {
  readonly name?: string;
  readonly instructions?: string;
}

function defaultInstructions(
  inputFields: ReadonlyMap<string, Field>,
  outputFields: ReadonlyMap<string, Field>,
): string {
  const inputs = [...inputFields.keys()].map((k) => `\`${k}\``).join(', ');
  const outputs = [...outputFields.keys()].map((k) => `\`${k}\``).join(', ');
  return `Given the fields ${inputs}, produce the fields ${outputs}.`;
}

function cloneValidatedFieldMap(
  fields: ReadonlyMap<string, Field>,
  expectedKind: FieldKind,
): Map<string, Field> {
  const cloned = new Map<string, Field>();

  for (const [key, field] of fields) {
    if (!(field instanceof Field)) {
      throw new InvariantError(`Field map entry "${key}" is not a Field instance`);
    }

    if (field.name !== key) {
      throw new InvariantError(
        `Field map key "${key}" does not match field.name "${field.name}"`,
      );
    }

    if (field.kind !== expectedKind) {
      throw new InvariantError(
        `Field "${key}" has kind "${field.kind}" but is stored in ${expectedKind}Fields`,
      );
    }

    cloned.set(key, field);
  }

  return cloned;
}

function assertDistinctFieldNames(
  fields: readonly ParsedField[],
  groupLabel: string,
  source: string,
): void {
  const seen = new Set<string>();

  for (const field of fields) {
    if (seen.has(field.name)) {
      throw new ValueError(
        `Duplicate ${groupLabel} field "${field.name}" in signature "${source}"`,
      );
    }

    seen.add(field.name);
  }
}

function parseFieldList(raw: string, kind: FieldKind): readonly ParsedField[] {
  if (raw.trim() === '') {
    return [];
  }

  return splitTopLevel(raw, ',', true).map((segment) => {
    const trimmed = segment.trim();
    if (trimmed === '') {
      throw new ValueError('Signature contains an empty field segment');
    }

    const fieldParts = splitTopLevel(trimmed, ':', true);
    if (fieldParts.length > 2) {
      throw new ValueError(`Field "${trimmed}" contains multiple type separators`);
    }

    if (fieldParts.length === 2) {
      const name = fieldParts[0]!.trim();
      const rawType = fieldParts[1]!.trim();
      if (rawType === '') {
        throw new ValueError(`Field "${name}" is missing a type annotation`);
      }

      const bracketIndex = rawType.indexOf('[');
      const baseType =
        bracketIndex === -1 ? rawType : rawType.slice(0, bracketIndex).trim();
      const typeTag = isTypeTag(baseType) ? baseType : 'custom';

      createField({
        kind,
        name,
        typeTag,
        isTypeUndefined: false,
      });

      return { name, typeTag, isTypeUndefined: false };
    }

    const name = trimmed;
    createField({ kind, name });
    return { name, typeTag: 'str', isTypeUndefined: true };
  });
}

function normalizeFieldForInsertion(
  fieldOrName: Field | string,
  kind?: FieldKind,
  typeTag?: TypeTag,
): Field {
  if (fieldOrName instanceof Field) {
    return fieldOrName;
  }

  if (kind === undefined || typeTag === undefined) {
    throw new ValueError('append/prepend requires either a Field or name+kind+typeTag');
  }

  if (!isTypeTag(typeTag)) {
    throw new ValueError(`Invalid type tag "${String(typeTag)}"`);
  }

  return createField({
    kind,
    name: fieldOrName,
    typeTag,
    isTypeUndefined: false,
  });
}

export class Signature {
  readonly name: string;
  readonly instructions: string;

  readonly #inputFields: ReadonlyMap<string, Field>;
  readonly #outputFields: ReadonlyMap<string, Field>;

  private constructor(
    name: string,
    instructions: string,
    inputFields: ReadonlyMap<string, Field>,
    outputFields: ReadonlyMap<string, Field>,
  ) {
    this.name = name;
    this.instructions = instructions;
    this.#inputFields = inputFields;
    this.#outputFields = outputFields;
  }

  static create(
    inputFields: ReadonlyMap<string, Field>,
    outputFields: ReadonlyMap<string, Field>,
    options?: SignatureOptions,
  ): Signature {
    const clonedInputs = cloneValidatedFieldMap(inputFields, 'input');
    const clonedOutputs = cloneValidatedFieldMap(outputFields, 'output');

    for (const key of clonedInputs.keys()) {
      if (clonedOutputs.has(key)) {
        throw new InvariantError(
          `Field "${key}" appears in both input and output fields (disjoint invariant violated)`,
        );
      }
    }

    return new Signature(
      options?.name ?? '',
      options?.instructions ?? defaultInstructions(clonedInputs, clonedOutputs),
      clonedInputs,
      clonedOutputs,
    );
  }

  get inputFields(): ReadonlyMap<string, Field> {
    return new Map(this.#inputFields);
  }

  get outputFields(): ReadonlyMap<string, Field> {
    return new Map(this.#outputFields);
  }
}

export function createSignature(
  inputFields: ReadonlyMap<string, Field>,
  outputFields: ReadonlyMap<string, Field>,
  options?: SignatureOptions,
): Signature {
  return Signature.create(inputFields, outputFields, options);
}

// ---------------------------------------------------------------------------
// SignatureOps — all return new Signature instances
// ---------------------------------------------------------------------------

export function appendField(sig: Signature, field: Field): Signature;
export function appendField(
  sig: Signature,
  name: string,
  kind: FieldKind,
  typeTag: TypeTag,
): Signature;
export function appendField(
  sig: Signature,
  fieldOrName: Field | string,
  kind?: FieldKind,
  typeTag?: TypeTag,
): Signature {
  const field = normalizeFieldForInsertion(fieldOrName, kind, typeTag);
  const newInputs = new Map(sig.inputFields);
  const newOutputs = new Map(sig.outputFields);

  if (field.kind === 'input') {
    newInputs.delete(field.name);
    newInputs.set(field.name, field);
  } else {
    newOutputs.delete(field.name);
    newOutputs.set(field.name, field);
  }

  return createSignature(newInputs, newOutputs, {
    name: sig.name,
    instructions: sig.instructions,
  });
}

export function prependField(sig: Signature, field: Field): Signature;
export function prependField(
  sig: Signature,
  name: string,
  kind: FieldKind,
  typeTag: TypeTag,
): Signature;
export function prependField(
  sig: Signature,
  fieldOrName: Field | string,
  kind?: FieldKind,
  typeTag?: TypeTag,
): Signature {
  const field = normalizeFieldForInsertion(fieldOrName, kind, typeTag);

  if (field.kind === 'input') {
    const newInputs = new Map<string, Field>();
    newInputs.set(field.name, field);
    for (const [key, value] of sig.inputFields) {
      if (key !== field.name) {
        newInputs.set(key, value);
      }
    }

    return createSignature(newInputs, new Map(sig.outputFields), {
      name: sig.name,
      instructions: sig.instructions,
    });
  }

  const newOutputs = new Map<string, Field>();
  newOutputs.set(field.name, field);
  for (const [key, value] of sig.outputFields) {
    if (key !== field.name) {
      newOutputs.set(key, value);
    }
  }

  return createSignature(new Map(sig.inputFields), newOutputs, {
    name: sig.name,
    instructions: sig.instructions,
  });
}

export function deleteField(sig: Signature, fieldName: string): Signature {
  const newInputs = new Map(sig.inputFields);
  const newOutputs = new Map(sig.outputFields);
  const deletedInput = newInputs.delete(fieldName);
  const deletedOutput = newOutputs.delete(fieldName);

  if (!deletedInput && !deletedOutput) {
    throw new ValueError(`Field "${fieldName}" not found in signature`);
  }

  return createSignature(newInputs, newOutputs, {
    name: sig.name,
    instructions: sig.instructions,
  });
}

export function withInstructions(
  sig: Signature,
  instructions: string,
): Signature {
  return createSignature(
    new Map(sig.inputFields),
    new Map(sig.outputFields),
    { name: sig.name, instructions },
  );
}

export function withUpdatedField(
  sig: Signature,
  fieldName: string,
  field: Field,
): Signature {
  const newInputs = new Map(sig.inputFields);
  const newOutputs = new Map(sig.outputFields);

  if (newInputs.has(fieldName)) {
    newInputs.set(fieldName, field);
  } else if (newOutputs.has(fieldName)) {
    newOutputs.set(fieldName, field);
  } else {
    throw new ValueError(`Field "${fieldName}" not found in signature`);
  }

  return createSignature(newInputs, newOutputs, {
    name: sig.name,
    instructions: sig.instructions,
  });
}

export function signatureEquals(a: Signature, b: Signature): boolean {
  if (a.name !== b.name) return false;
  if (a.instructions !== b.instructions) return false;
  if (a.inputFields.size !== b.inputFields.size) return false;
  if (a.outputFields.size !== b.outputFields.size) return false;

  for (const [key, fieldA] of a.inputFields) {
    const fieldB = b.inputFields.get(key);
    if (!fieldB) return false;
    if (!fieldA.equals(fieldB)) return false;
  }
  for (const [key, fieldA] of a.outputFields) {
    const fieldB = b.outputFields.get(key);
    if (!fieldB) return false;
    if (!fieldA.equals(fieldB)) return false;
  }

  return true;
}

/** Canonical string form: "input1, input2 -> output1, output2" */
export function signatureString(sig: Signature): string {
  const inputs = [...sig.inputFields.keys()].join(', ');
  const outputs = [...sig.outputFields.keys()].join(', ');
  return `${inputs} -> ${outputs}`;
}

/** All fields in insertion order: inputs first, then outputs. */
export function signatureFields(sig: Signature): readonly Field[] {
  return [...sig.inputFields.values(), ...sig.outputFields.values()];
}

// ---------------------------------------------------------------------------
// §1.4 — String parsing
// ---------------------------------------------------------------------------

export interface ParsedField {
  readonly name: string;
  readonly typeTag: TypeTag;
  readonly isTypeUndefined: boolean;
}

export interface ParseResult {
  readonly inputs: readonly ParsedField[];
  readonly outputs: readonly ParsedField[];
}

export function parseSignature(input: string): ParseResult {
  const arrowParts = splitTopLevel(input, '->', true);
  if (arrowParts.length !== 2) {
    throw new ValueError(
      `Signature must contain exactly one '->': got "${input}"`,
    );
  }

  const inputs = parseFieldList(arrowParts[0]!, 'input');
  const outputs = parseFieldList(arrowParts[1]!, 'output');

  assertDistinctFieldNames(inputs, 'input', input);
  assertDistinctFieldNames(outputs, 'output', input);

  // Enforce disjoint names.
  const inputNames = new Set(inputs.map((f) => f.name));
  for (const out of outputs) {
    if (inputNames.has(out.name)) {
      throw new ValueError(
        `Input and output fields must have distinct names: "${out.name}" appears in both`,
      );
    }
  }

  return { inputs, outputs };
}

/** Normalise a Signature-or-string to a Signature. */
export function ensureSignature(value: Signature | string): Signature {
  return value instanceof Signature ? value : signatureFromString(value);
}

/** Convenience: parse a string and produce a full Signature. */
export function signatureFromString(
  input: string,
  instructions?: string,
): Signature {
  const parsed = parseSignature(input);

  const inputFields = new Map<string, Field>();
  for (const pf of parsed.inputs) {
    inputFields.set(
      pf.name,
      pf.isTypeUndefined
        ? createField({ kind: 'input', name: pf.name })
        : createField({
            kind: 'input',
            name: pf.name,
            typeTag: pf.typeTag,
            isTypeUndefined: false,
          }),
    );
  }

  const outputFields = new Map<string, Field>();
  for (const pf of parsed.outputs) {
    outputFields.set(
      pf.name,
      pf.isTypeUndefined
        ? createField({ kind: 'output', name: pf.name })
        : createField({
            kind: 'output',
            name: pf.name,
            typeTag: pf.typeTag,
            isTypeUndefined: false,
          }),
    );
  }

  if (instructions === undefined) {
    return createSignature(inputFields, outputFields);
  }

  return createSignature(inputFields, outputFields, { instructions });
}
