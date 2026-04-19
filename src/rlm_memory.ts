/**
 * Typed memory schema and runtime for RLM v2 static plans.
 *
 * Declarations attach to static and resolved plans; runtime helpers seed
 * state, apply writes, and render the oracle banner. `WriteMemory` handlers
 * delegate to `applyMemoryWrite` so validation matches reinjection.
 *
 * Contract: `docs/product/rlm-v2-architecture.md` §2.5.
 */

import { ValueError } from './exceptions.js';
import type { JsonValue } from './json_value.js';
import { typeName } from './rlm_util.js';

// ---------------------------------------------------------------------------
// Type tags
// ---------------------------------------------------------------------------
//
// The set of permitted field types is intentionally narrow. Oracle
// reinjection is a terse, typed banner — not a free-form note — and every
// schema change should be a deliberate widening. Lists and nested objects
// would defeat the purpose; if a field needs structure, author two fields.

/** Scalar types allowed for a memory field — distinct from field `TypeTag` in `types.ts`. */
export type MemoryScalarType = 'string' | 'number' | 'boolean';

/** Runtime value stored for a schema field after validation. */
export type MemoryScalar = string | number | boolean;

// ---------------------------------------------------------------------------
// Field and schema shape
// ---------------------------------------------------------------------------

/**
 * One field in a typed memory schema.
 *
 * - `name`: stable identifier; the LM references it in `WriteMemory` effect
 *   payloads.
 * - `type`: the runtime type tag; `applyMemoryWrite` rejects values
 *   whose `typeof` mismatch.
 * - `description`: human-readable context. The default injector does
 *   not render descriptions (banner stays terse), but custom injectors
 *   may use them.
 * - `initial`: default value before any `WriteMemory` effect has run.
 *   `undefined` (or absent) means "absent until written" — the injector
 *   emits nothing for absent fields so prompts stay short.
 * - `maxLength`: bounds `string`-typed values. `applyMemoryWrite`
 *   throws `ValueError` when exceeded; the handler layer catches and
 *   surfaces the message as a structured effect error so the LM can
 *   retry with a shorter value.
 */
export interface MemoryFieldSchema {
  readonly name: string;
  readonly type: MemoryScalarType;
  readonly description: string;
  /** Authoring-time seed; may disagree with `type` until `initialMemoryState` validates. */
  readonly initial?: unknown;
  readonly maxLength?: number;
}

/**
 * A typed memory schema — the contract each plan attaches to govern what
 * the oracle may remember between calls. Flows from `StaticPlan` →
 * `ResolvedPlan` → `EvaluationContext` so `WriteMemory` handlers can validate.
 *
 * The serialized banner rendered by `defaultMemoryInjector` is hard-capped
 * at `maxBytesSerialized`. Writes are validated against the rendered
 * banner's UTF-8 byte size — not a proxy like JSON payload length — so the
 * cap matches what is actually reinjected into the system message. The
 * default cap is 2048 bytes — enough for ~6 tightly-bounded string fields.
 */
export interface MemorySchema {
  readonly name: string;
  readonly fields: readonly MemoryFieldSchema[];
  readonly maxBytesSerialized: number;
}

// ---------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------

/**
 * Typed per-`aforward` memory state. Each plan attaches a `MemorySchema`
 * that governs legal keys, types, and length bounds; this map is the
 * runtime projection of a schema instance.
 *
 * Stored in `EvaluationContext.memoryCell` (via `src/rlm_types.ts`) so
 * `WriteMemory` handlers can publish a new immutable snapshot and every
 * subsequent oracle call reads from the latest one. The shape — a
 * readonly `Map<string, MemoryScalar>` — is stable across phases; values are
 * type-validated on write against the schema so the element type is the
 * closed scalar union, not free-form data.
 */
export type TypedMemoryState = ReadonlyMap<string, MemoryScalar>;

/**
 * Canonical empty memory snapshot reused anywhere a plan does not
 * declare memory. The proxy wrapper makes the snapshot runtime-immutable:
 * mutating methods (`set`, `delete`, `clear`) throw instead of silently
 * modifying the backing `Map`.
 */
export const EMPTY_TYPED_MEMORY_STATE: TypedMemoryState =
  createTypedMemoryState();

// ---------------------------------------------------------------------------
// Injector protocol
// ---------------------------------------------------------------------------

/**
 * Render a `TypedMemoryState` as a banner suitable for prepending to the
 * oracle's system message. `defaultMemoryInjector` ships the
 * `[[RLM_MEMORY ...]] ... [[/RLM_MEMORY]]` layout documented in the spec.
 *
 * The protocol is a function, not a class, because stateless rendering is
 * easier to test and swap per plan if a task benefits from a different
 * serialization.
 */
export interface MemoryInjector {
  render(schema: MemorySchema, state: TypedMemoryState): string;
}

// ---------------------------------------------------------------------------
// Runtime: state construction and validated writes
// ---------------------------------------------------------------------------

/**
 * Recommended default for `MemorySchema.maxBytesSerialized`. Keeps the
 * reinjected banner under ~2KB so it does not dominate the oracle's
 * system message on every turn. Exposed so plan authors can reuse the
 * canonical cap without remembering the literal.
 */
export const DEFAULT_MAX_MEMORY_BYTES = 2048 as const;

/**
 * Build the initial memory state for a schema.
 *
 * Fields with a defined `initial` value are seeded; fields without one
 * (or with `initial === undefined`) are omitted so the rendered banner
 * only surfaces intentionally-populated data. Seed values are validated
 * against the field's type and `maxLength`, because an out-of-band
 * `initial` that the schema itself rejects would leave the plan in an
 * inconsistent state on turn 1.
 *
 * The returned snapshot is runtime-immutable: it is exposed through a
 * `ReadonlyMap` proxy that throws on mutation methods, so accidental
 * in-place writes fail loudly instead of silently leaking state across
 * oracle calls.
 */
export function initialMemoryState(schema: MemorySchema): TypedMemoryState {
  const state = new Map<string, MemoryScalar>();
  for (const field of schema.fields) {
    if (field.initial === undefined) continue;
    const validated = validateFieldValue(field, field.initial);
    if (!validated.ok) {
      throw new ValueError(
        `RLM memory schema "${schema.name}" has an invalid initial value ` +
          `for field "${field.name}": ${validated.error}`,
      );
    }
    state.set(field.name, validated.value);
  }
  if (!withinSerializedBudget(state, schema)) {
    throw new ValueError(
      `RLM memory schema "${schema.name}" initial state exceeds ` +
        `maxBytesSerialized=${schema.maxBytesSerialized}; either widen the ` +
        `budget or tighten field initials.`,
    );
  }
  return state.size === 0 ? EMPTY_TYPED_MEMORY_STATE : createTypedMemoryState(state);
}

/**
 * Shape of a single memory write produced by the LM via `WriteMemory`.
 */
export interface MemoryWrite {
  readonly key: string;
  /** Wire value from the LM; validated against the field before commit. */
  readonly value: JsonValue;
}

/**
 * Apply a validated write to `state`, returning a new immutable snapshot.
 *
 * Validation, in order:
 *
 * 1. The field name must be declared in `schema`.
 * 2. The value must match the field's `type` tag (`string`, `number`,
 *    or `boolean`). Non-finite numbers are rejected.
 * 3. For `string` fields, the value must respect `maxLength` if set.
 * 4. The resulting state's rendered memory banner must fit
 *    `schema.maxBytesSerialized` when encoded as UTF-8.
 *
 * Any rule violation throws `ValueError`. The input `state` is never
 * mutated; this function returns a fresh immutable snapshot. Handlers
 * should catch the error and surface it as a structured effect result so
 * the LM sees a clean diagnostic on the next turn rather than a stack
 * trace.
 */
export function applyMemoryWrite(
  state: TypedMemoryState,
  schema: MemorySchema,
  write: MemoryWrite,
): TypedMemoryState {
  const field = schema.fields.find((candidate) => candidate.name === write.key);
  if (field === undefined) {
    throw new ValueError(
      `RLM memory: key "${write.key}" is not declared in schema "${schema.name}" ` +
        `(known: ${schema.fields.map((f) => f.name).join(', ') || '(none)'})`,
    );
  }
  const validated = validateFieldValue(field, write.value);
  if (!validated.ok) {
    throw new ValueError(validated.error);
  }
  const next = new Map<string, MemoryScalar>(state);
  next.set(field.name, validated.value);
  if (!withinSerializedBudget(next, schema)) {
    throw new ValueError(
      `RLM memory: write to "${write.key}" would exceed ` +
        `maxBytesSerialized=${schema.maxBytesSerialized}`,
    );
  }
  return createTypedMemoryState(next);
}

// ---------------------------------------------------------------------------
// Runtime: default injector
// ---------------------------------------------------------------------------

/**
 * Render the default `[[RLM_MEMORY ...]] ... [[/RLM_MEMORY]]` banner.
 *
 * Layout example:
 *
 * ```
 * [[RLM_MEMORY schema=failure_diagnostic]]
 * failure_pattern: str = "oracle returned unrelated answer"
 * next_check: str = "verify chunk boundary alignment"
 * [[/RLM_MEMORY]]
 * ```
 *
 * - Fields are emitted in schema declaration order (not state
 *   iteration order) so the banner is deterministic across writes that
 *   touch the same fields.
 * - Absent fields (not in `state`) are omitted; the header/footer lines
 *   are always emitted so the LM can always see that memory exists and
 *   emit `WriteMemory` effects with confidence.
 * - Values are formatted per `MemoryScalarType`: strings via `JSON.stringify`
 *   so quotes and newlines are escaped; numbers and booleans as their
 *   canonical string form.
 * - The injector never returns a banner whose UTF-8 byte length exceeds
 *   `schema.maxBytesSerialized`: if the declaration-order banner is too
 *   large, it is truncated with a clear `[[/RLM_MEMORY truncated]]`
 *   marker. This is a safety net — the write path already rejects
 *   out-of-budget writes, so truncation only fires for states that were
 *   constructed outside the normal handler path.
 */
export const defaultMemoryInjector: MemoryInjector = Object.freeze({
  render(schema: MemorySchema, state: TypedMemoryState): string {
    const header = `[[RLM_MEMORY schema=${schema.name}]]`;
    const footer = `[[/RLM_MEMORY]]`;
    const bodyLines = collectBodyLines(schema, state);
    const banner = joinLines([header, ...bodyLines, footer]);
    if (utf8ByteLength(banner) <= schema.maxBytesSerialized) {
      return banner;
    }
    return truncateBanner(header, bodyLines, schema.maxBytesSerialized);
  },
});

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

interface ValidationOk {
  readonly ok: true;
  readonly value: MemoryScalar;
}
interface ValidationErr {
  readonly ok: false;
  readonly error: string;
}
type ValidationResult = ValidationOk | ValidationErr;

function validateFieldValue(
  field: MemoryFieldSchema,
  value: unknown,
): ValidationResult {
  if (field.type === 'string') {
    if (typeof value !== 'string') {
      return {
        ok: false,
        error: `RLM memory: field "${field.name}" expects string, got ${typeName(value)}`,
      };
    }
    if (field.maxLength !== undefined && value.length > field.maxLength) {
      return {
        ok: false,
        error:
          `RLM memory: field "${field.name}" length ${value.length} ` +
          `exceeds maxLength ${field.maxLength}`,
      };
    }
    return { ok: true, value };
  }
  if (field.type === 'number') {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return {
        ok: false,
        error:
          `RLM memory: field "${field.name}" expects finite number, ` +
          `got ${typeName(value)}`,
      };
    }
    return { ok: true, value };
  }
  if (typeof value !== 'boolean') {
    return {
      ok: false,
      error:
        `RLM memory: field "${field.name}" expects boolean, got ${typeName(value)}`,
    };
  }
  return { ok: true, value };
}

function withinSerializedBudget(
  state: ReadonlyMap<string, MemoryScalar>,
  schema: MemorySchema,
): boolean {
  return (
    utf8ByteLength(renderUnboundedBanner(schema, state)) <=
    schema.maxBytesSerialized
  );
}

function renderFieldValue(type: MemoryScalarType, value: MemoryScalar | undefined): string {
  if (type === 'string') {
    return JSON.stringify(typeof value === 'string' ? value : String(value));
  }
  if (type === 'boolean') {
    return value === true ? 'true' : 'false';
  }
  return typeof value === 'number' && Number.isFinite(value)
    ? String(value)
    : '0';
}

function renderUnboundedBanner(
  schema: MemorySchema,
  state: ReadonlyMap<string, MemoryScalar>,
): string {
  const header = `[[RLM_MEMORY schema=${schema.name}]]`;
  const footer = `[[/RLM_MEMORY]]`;
  return joinLines([header, ...collectBodyLines(schema, state), footer]);
}

function collectBodyLines(
  schema: MemorySchema,
  state: ReadonlyMap<string, MemoryScalar>,
): string[] {
  const bodyLines: string[] = [];
  for (const field of schema.fields) {
    if (!state.has(field.name)) continue;
    const value = state.get(field.name);
    bodyLines.push(
      `${field.name}: ${field.type} = ${renderFieldValue(field.type, value)}`,
    );
  }
  return bodyLines;
}

function truncateBanner(
  header: string,
  bodyLines: readonly string[],
  cap: number,
): string {
  const truncatedFooter = '[[/RLM_MEMORY truncated]]';
  const minimal = joinLines([header, truncatedFooter]);
  if (utf8ByteLength(minimal) > cap) {
    return truncateUtf8(minimal, cap);
  }
  const accumulator: string[] = [header];
  let best = minimal;
  for (const line of bodyLines) {
    accumulator.push(line);
    const candidate = joinLines([...accumulator, truncatedFooter]);
    if (utf8ByteLength(candidate) > cap) {
      accumulator.pop();
      break;
    }
    best = candidate;
  }
  return best;
}

function joinLines(lines: readonly string[]): string {
  return lines.join('\n');
}

function utf8ByteLength(value: string): number {
  return Buffer.byteLength(value, 'utf8');
}

function truncateUtf8(value: string, cap: number): string {
  let out = '';
  let size = 0;
  for (const ch of value) {
    const next = size + utf8ByteLength(ch);
    if (next > cap) break;
    out += ch;
    size = next;
  }
  return out;
}

function createTypedMemoryState(
  entries: Iterable<readonly [string, MemoryScalar]> = [],
): TypedMemoryState {
  const backing = new Map<string, MemoryScalar>(entries);
  const error = () => {
    throw new TypeError(
      'RLM memory: TypedMemoryState is immutable; create a new snapshot instead.',
    );
  };
  return new Proxy(backing, {
    get(target, prop, _receiver) {
      if (prop === 'set' || prop === 'delete' || prop === 'clear') {
        return error;
      }
      const value = Reflect.get(target, prop, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
    set() {
      error();
      return false;
    },
    defineProperty() {
      error();
      return false;
    },
    deleteProperty() {
      error();
      return false;
    },
  }) as TypedMemoryState;
}
