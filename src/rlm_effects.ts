/**
 * Effect shapes (re-exported from `rlm_types.ts`), oracle wire signature,
 * `parseOracleResponse` / `appendEffectResult`, and built-in `EffectHandler`
 * factories. `QueryOracle` does one plain-oracle call (no nested effect loop).
 * `WriteMemory` validates via `applyMemoryWrite` in `rlm_memory.ts`. Handlers
 * should return `{ ok: false }` for recoverable failures so the effect loop
 * can retry; reserve throws for fatal cases (`BudgetError`, programmer errors).
 *
 * Contract: `docs/product/rlm-v2-architecture.md` §2.4.
 */

import { BudgetError, ValueError } from './exceptions.js';
import { isPlainObject } from './guards.js';
import type { JsonObject, JsonValue } from './json_value.js';
import { type Signature, signatureFromString } from './signature.js';
import type { Prediction } from './prediction.js';
import type {
  Effect,
  EffectHandler,
  EffectResult,
  OracleResponse,
  QueryOracleCallFn,
} from './rlm_types.js';
import { applyMemoryWrite } from './rlm_memory.js';

// Re-export the protocol types declared in `rlm_types.ts`. Callers that
// only need the shapes (e.g. to build a `Custom` handler) can import
// from here alongside the parser and handler factories.
export type { Effect, EffectResult, OracleResponse, QueryOracleCallFn };

// ---------------------------------------------------------------------------
// Effect kinds
// ---------------------------------------------------------------------------

/**
 * Canonical effect kind ordering (oracle signature literals match this list).
 *
 * Effect semantics (full contracts in `rlm_types.ts`):
 *
 * - `ReadContext` — slice a scope binding (usually the user input) by
 *   character index. Bounds clamp; inverted ranges yield the empty
 *   string.
 * - `WriteMemory` — persist a typed value into the plan's memory.
 *   Rejected when the plan has no `MemorySchema` or when the value
 *   violates the field type / length / byte budget.
 * - `QueryOracle` — single-call delegation to a sub-oracle that
 *   speaks the plain (non-effect) signature. Does not recursively
 *   enter another effect loop.
 * - `Search` — out-of-plan retrieval; no built-in backend, users
 *   provide a `Custom` handler or a named `'Search'` override.
 * - `Yield` — cooperative no-op useful for partitioning trace turns.
 * - `Custom` — open-world escape hatch dispatched by name.
 */
export const EFFECT_KINDS: readonly Effect['kind'][] = Object.freeze([
  'ReadContext',
  'WriteMemory',
  'QueryOracle',
  'Search',
  'Yield',
  'Custom',
]);

const EFFECT_KIND_SET: ReadonlySet<string> = new Set(EFFECT_KINDS);

// ---------------------------------------------------------------------------
// Type guards
// ---------------------------------------------------------------------------

/**
 * Narrow `unknown` to `Effect` (structural `kind` + required fields only;
 * range validation is the handler's job).
 */
export function isEffect(value: unknown): value is Effect {
  if (value === null || typeof value !== 'object') return false;
  const record = value as { readonly kind?: unknown };
  if (typeof record.kind !== 'string') return false;
  if (!EFFECT_KIND_SET.has(record.kind)) return false;
  switch (record.kind as Effect['kind']) {
    case 'ReadContext': {
      const v = value as { readonly name?: unknown };
      return typeof v.name === 'string';
    }
    case 'WriteMemory': {
      const v = value as { readonly key?: unknown };
      return typeof v.key === 'string';
    }
    case 'QueryOracle': {
      const v = value as { readonly prompt?: unknown };
      return typeof v.prompt === 'string';
    }
    case 'Search': {
      const v = value as { readonly query?: unknown };
      return typeof v.query === 'string';
    }
    case 'Yield':
      return true;
    case 'Custom': {
      const v = value as {
        readonly name?: unknown;
        readonly args?: unknown;
      };
      return (
        typeof v.name === 'string' &&
        v.args !== null &&
        typeof v.args === 'object'
      );
    }
  }
}

/** Narrow `unknown` to `EffectResult`. */
export function isEffectResult(value: unknown): value is EffectResult {
  if (!isPlainObject(value)) return false;
  const record = value as { readonly ok?: unknown };
  if (typeof record.ok !== 'boolean') return false;
  if (record.ok === true) {
    return 'value' in value;
  }
  return 'error' in value && typeof (value as { readonly error?: unknown }).error === 'string';
}

// ---------------------------------------------------------------------------
// Oracle signature
// ---------------------------------------------------------------------------

/**
 * The extended oracle signature the effect loop speaks. Every oracle
 * leaf call goes through this signature so the LM can either answer
 * directly (`kind: 'value'`) or request a tool use
 * (`kind: 'effect'`). The field set mirrors the `OracleResponse`
 * union:
 *
 * - `kind`: literal discriminant.
 * - `value`: populated when `kind === 'value'`.
 * - `effect_name`: populated when `kind === 'effect'`; one of the six
 *   `EFFECT_KINDS`.
 * - `effect_args`: populated when `kind === 'effect'`; the payload
 *   fields of the chosen effect (e.g. `{ name: '…', start: 0, end: … }`
 *   for `ReadContext`).
 *
 * Parsing into `OracleResponse` is `parseOracleResponse`'s job; the
 * signature here is intentionally loose (all effect-side fields are
 * `optional`) so malformed LM outputs surface through the parser's
 * structured error path rather than an adapter-level coercion throw.
 */
export const EFFECT_ORACLE_SIGNATURE: Signature = signatureFromString(
  'prompt: str -> ' +
    'kind: literal["value", "effect"], ' +
    'value: optional[str], ' +
    'effect_name: optional[str], ' +
    'effect_args: optional[dict]',
);

// ---------------------------------------------------------------------------
// Response parser
// ---------------------------------------------------------------------------

/**
 * Coerce the `Prediction` produced by `Predict.acall(EFFECT_ORACLE_SIGNATURE)`
 * into a typed `OracleResponse`.
 *
 * The parser is deliberately permissive at edges but strict on
 * semantics:
 *
 * - A `kind: 'value'` response with a missing or empty `value` field
 *   is treated as an empty-string answer — the LM's loosest acceptable
 *   way to say "done, nothing to report".
 * - A `kind: 'effect'` response with an unknown or malformed
 *   `effect_name` throws `ValueError`. The evaluator wraps this in an
 *   `EffectResult` failure so the LM sees the structured error on the
 *   next turn and can retry.
 * - Absent `kind` — or `kind` outside the `{'value', 'effect'}` pair —
 *   is treated as a malformed response and throws. The adapter-level
 *   literal enforcement already normalizes this; the extra guard is
 *   defensive.
 */
export function parseOracleResponse(
  prediction: Prediction<Record<string, unknown>>,
): OracleResponse {
  const rawKind = prediction.getOr('kind', undefined);
  if (rawKind !== 'value' && rawKind !== 'effect') {
    throw new ValueError(
      `RLM effect: oracle returned unknown kind "${String(rawKind)}" (expected "value" or "effect")`,
    );
  }
  if (rawKind === 'value') {
    const rawValue = prediction.getOr('value', '');
    const value = typeof rawValue === 'string' ? rawValue : '';
    return { kind: 'value', value };
  }
  const rawName = prediction.getOr('effect_name', undefined);
  if (typeof rawName !== 'string' || rawName.length === 0) {
    throw new ValueError(
      'RLM effect: oracle returned kind="effect" without an effect_name.',
    );
  }
  const rawArgs = prediction.getOr('effect_args', undefined);
  const args =
    rawArgs !== null && typeof rawArgs === 'object' && !Array.isArray(rawArgs)
      ? (rawArgs as Readonly<Record<string, JsonValue>>)
      : EMPTY_ARGS;
  const effect = buildEffect(rawName, args);
  return { kind: 'effect', effect };
}

const EMPTY_ARGS: JsonObject = Object.freeze({});

function buildEffect(
  name: string,
  args: Readonly<Record<string, JsonValue>>,
): Effect {
  switch (name) {
    case 'ReadContext': {
      const boundName = stringArg(args, 'name');
      if (boundName === null) {
        throw new ValueError(
          'RLM effect: ReadContext requires effect_args.name (string)',
        );
      }
      const start = optionalFiniteIntegerArg(args, 'start');
      const end = optionalFiniteIntegerArg(args, 'end');
      const out: Effect = {
        kind: 'ReadContext',
        name: boundName,
        ...(start !== undefined ? { start } : {}),
        ...(end !== undefined ? { end } : {}),
      };
      return out;
    }
    case 'WriteMemory': {
      const key = stringArg(args, 'key');
      if (key === null) {
        throw new ValueError(
          'RLM effect: WriteMemory requires effect_args.key (string)',
        );
      }
      const rawValue = args.value;
      const value: JsonValue = rawValue === undefined ? null : rawValue as JsonValue;
      return { kind: 'WriteMemory', key, value };
    }
    case 'QueryOracle': {
      const prompt = stringArg(args, 'prompt');
      if (prompt === null) {
        throw new ValueError(
          'RLM effect: QueryOracle requires effect_args.prompt (string)',
        );
      }
      const modelHint = stringArg(args, 'modelHint');
      return {
        kind: 'QueryOracle',
        prompt,
        ...(modelHint !== null ? { modelHint } : {}),
      };
    }
    case 'Search': {
      const query = stringArg(args, 'query');
      if (query === null) {
        throw new ValueError(
          'RLM effect: Search requires effect_args.query (string)',
        );
      }
      const topK = optionalFiniteIntegerArg(args, 'topK');
      return {
        kind: 'Search',
        query,
        ...(topK !== undefined ? { topK } : {}),
      };
    }
    case 'Yield':
      return { kind: 'Yield' };
    case 'Custom': {
      const customName = stringArg(args, 'name');
      if (customName === null) {
        throw new ValueError(
          'RLM effect: Custom requires effect_args.name (string)',
        );
      }
      const innerArgs = args.args;
      const normalized =
        innerArgs !== null &&
        typeof innerArgs === 'object' &&
        !Array.isArray(innerArgs)
          ? (innerArgs as JsonObject)
          : EMPTY_ARGS;
      return { kind: 'Custom', name: customName, args: normalized };
    }
    default:
      throw new ValueError(
        `RLM effect: unknown effect_name "${name}" (expected one of: ${EFFECT_KINDS.join(', ')})`,
      );
  }
}

function stringArg(
  args: Readonly<Record<string, JsonValue>>,
  key: string,
): string | null {
  const value = args[key];
  return typeof value === 'string' ? value : null;
}

function optionalFiniteIntegerArg(
  args: Readonly<Record<string, JsonValue>>,
  key: string,
): number | undefined {
  const value = args[key];
  if (value === undefined || value === null) return undefined;
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValueError(
      `RLM effect: effect_args.${key} must be a finite number, got ${String(value)}`,
    );
  }
  return Math.trunc(value);
}

// ---------------------------------------------------------------------------
// Next-turn prompt assembly
// ---------------------------------------------------------------------------

/**
 * Build the oracle's next-turn prompt by appending the current turn's
 * effect request and its handler result to the previous prompt.
 *
 * The format is a compact, human-readable block delimited by
 * `[[EFFECT ...]] ... [[/EFFECT]]`. The `kind` and `turn` indices are
 * always present so the LM can reason about which request each block
 * answers; on success the rendered value sits between `result:` and
 * `[[/EFFECT]]`, on failure the rendered error sits between `error:`
 * and `[[/EFFECT]]`.
 *
 * Rendering rules:
 *
 * - Effect args are `JSON.stringify`'d, with a fallback to `String()`
 *   on cyclic objects. Compact JSON matches the LM's own output
 *   format and is the shortest faithful representation.
 * - Successful results are rendered via `renderValue` — strings pass
 *   through unchanged; objects and arrays are JSON-stringified; other
 *   primitives are `String()`'d. The `String()` fallback matches the
 *   RLM v2 evaluator's own coercion logic in `rlm.ts`.
 * - Errors are rendered verbatim; they always arrive from the handler
 *   as strings.
 */
export function appendEffectResult(
  prompt: string,
  effect: Effect,
  result: EffectResult,
  turn: number,
): string {
  const argsBlob = renderEffectArgs(effect);
  const header = `[[EFFECT turn=${turn} kind=${effect.kind}]]`;
  const request = argsBlob === '' ? 'args: {}' : `args: ${argsBlob}`;
  const body = result.ok
    ? `result: ${renderValue(result.value)}`
    : `error: ${result.error}`;
  const footer = '[[/EFFECT]]';
  const block = [header, request, body, footer].join('\n');
  return prompt.length === 0 ? block : `${prompt}\n\n${block}`;
}

function renderEffectArgs(effect: Effect): string {
  const { kind: _kind, ...rest } = effect;
  if (Object.keys(rest).length === 0) return '';
  return JSON.stringify(rest);
}

function renderValue(value: JsonValue): string {
  if (typeof value === 'string') return value;
  if (value === undefined || value === null) return '';
  if (typeof value === 'object') {
    return JSON.stringify(value);
  }
  return String(value);
}

// ---------------------------------------------------------------------------
// Built-in handler factories
// ---------------------------------------------------------------------------

/**
 * Read a chunk of text from a scope binding by name and optional
 * `[start, end)` character slice. Bounds are clamped to the string's
 * length; inverted or empty slices yield an empty string rather than
 * throwing so the LM can ask for a trailing slice without precise
 * length knowledge.
 *
 * The default `boundName` is `'input'` — the RLM facade binds the
 * original user prompt to that key so plans that want to re-read the
 * input just request `ReadContext(name='input', start=…, end=…)`.
 */
export function readContextHandler(): EffectHandler {
  return {
    name: 'ReadContext',
    handle: async (effect, ctx) => {
      if (!isEffect(effect) || effect.kind !== 'ReadContext') {
        return { ok: false, error: 'ReadContext: malformed effect payload' };
      }
      const raw = ctx.scope.get(effect.name);
      if (raw === undefined && !ctx.scope.has(effect.name)) {
        return {
          ok: false,
          error: `ReadContext: scope binding "${effect.name}" is not defined`,
        };
      }
      if (typeof raw !== 'string') {
        return {
          ok: false,
          error: `ReadContext: scope binding "${effect.name}" is not a string`,
        };
      }
      const start = clampIndex(effect.start, raw.length, 0);
      const end = clampIndex(effect.end, raw.length, raw.length);
      if (end <= start) {
        return { ok: true, value: '' };
      }
      return { ok: true, value: raw.slice(start, end) };
    },
  };
}

function clampIndex(
  raw: number | undefined,
  length: number,
  fallback: number,
): number {
  if (raw === undefined || !Number.isFinite(raw)) return fallback;
  const trunc = Math.trunc(raw);
  if (trunc < 0) return 0;
  if (trunc > length) return length;
  return trunc;
}

/**
 * Persist a typed value into the plan's memory.
 *
 * - If the plan did not declare a `MemorySchema`, the write is
 *   rejected with a structured error — plans without memory should
 *   not be using `WriteMemory` at all, and the LM's next turn gets a
 *   clear diagnostic.
 * - Field-level validation uses the schema's `type` (`string` /
 *   `number` / `boolean`) and optional `maxLength` (string-only).
 * - The rendered memory state's total byte budget is enforced against
 *   `schema.maxBytesSerialized` using the same deterministic banner
 *   shape the injector prepends to system messages, so write-time
 *   validation matches what later oracle turns will actually see.
 *
 * Successful writes mutate `ctx.memoryCell.current` so subsequent
 * oracle calls within the same plan tree observe the update via the
 * reinjected memory banner on later turns.
 */
export function writeMemoryHandler(): EffectHandler {
  return {
    name: 'WriteMemory',
    handle: async (effect, ctx) => {
      if (!isEffect(effect) || effect.kind !== 'WriteMemory') {
        return { ok: false, error: 'WriteMemory: malformed effect payload' };
      }
      const schema = ctx.memorySchema;
      if (schema === null) {
        return {
          ok: false,
          error:
            'WriteMemory: plan has no memory schema; declare one or avoid this effect',
        };
      }
      try {
        const next = applyMemoryWrite(ctx.memoryCell.current, schema, {
          key: effect.key,
          value: effect.value,
        });
        ctx.memoryCell.current = next;
        return { ok: true, value: null };
      } catch (err) {
        // Validation throws from `applyMemoryWrite` → structured effect error for the LM.
        if (err instanceof BudgetError) throw err;
        const message = err instanceof Error ? err.message : String(err);
        return { ok: false, error: message };
      }
    },
  };
}

/**
 * Delegate a fresh sub-prompt to the oracle. `QueryOracle` always uses
 * the plain `oracle` signature (no recursive effect loop) so it is a
 * true single-call helper. `modelHint` resolves against
 * `ctx.lmRegistry` exactly like a direct `oracle(prompt, hint)` node.
 *
 * The handler takes the `callOracleFn` as a late-bound dependency to
 * avoid a circular import: `rlm_evaluator.ts` imports this module for
 * `EFFECT_ORACLE_SIGNATURE` and `parseOracleResponse`, so it cannot
 * also provide a plain-oracle helper at module load time. The
 * evaluator constructs the handler with its own `callOracleLeaf`
 * bound in when it builds the default handler registry.
 *
 * {@link QueryOracleCallFn} is declared in `rlm_types.ts`.
 */
export function queryOracleHandler(
  callOracleFn: QueryOracleCallFn,
): EffectHandler {
  return {
    name: 'QueryOracle',
    handle: async (effect, ctx) => {
      if (!isEffect(effect) || effect.kind !== 'QueryOracle') {
        return { ok: false, error: 'QueryOracle: malformed effect payload' };
      }
      try {
        const answer = await callOracleFn(effect.prompt, effect.modelHint, ctx);
        return { ok: true, value: answer };
      } catch (err) {
        // Nested oracle I/O: surface failure as `EffectResult` so the outer effect loop can continue.
        if (err instanceof BudgetError) throw err;
        const message = err instanceof Error ? err.message : String(err);
        return {
          ok: false,
          error: `QueryOracle: sub-oracle failed: ${message}`,
        };
      }
    },
  };
}

/**
 * Cooperative no-op. The LM emits `Yield` when it has nothing more to
 * say this turn; the handler returns `ok: true` with a `null` value
 * and the evaluator treats the next iteration like a normal
 * follow-up. `Yield` is primarily useful to partition a long chain of
 * effects into distinct logical turns when introspecting traces.
 */
export function yieldHandler(): EffectHandler {
  return {
    name: 'Yield',
    handle: async () => ({ ok: true, value: null }),
  };
}

/**
 * Default registry of built-in handlers in dispatch order. The
 * evaluator merges user-supplied handlers on top so users can
 * override any of the defaults by name.
 */
export function builtInEffectHandlers(
  callOracleFn: QueryOracleCallFn,
): readonly EffectHandler[] {
  return Object.freeze([
    readContextHandler(),
    writeMemoryHandler(),
    queryOracleHandler(callOracleFn),
    yieldHandler(),
  ]);
}
