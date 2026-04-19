/**
 * §1.5 — Type-level signature parsing.
 *
 * Purely advisory type-level companion to the runtime parser in
 * `@/signature.ts`. Goal: map a literal signature string such as
 * `'question: str -> answer: str'` into `{ inputs: { question: string };
 * outputs: { answer: string } }` at compile time so that `Predict`,
 * `ChainOfThought`, `ReAct`, and `RLM` can surface input keys and output
 * field types to IDE tooling without changing runtime semantics.
 *
 * Intentional scope caps (see `docs/product/type-level-signature-inference.md`):
 *   - Field type tags: `str | int | float | bool | list | dict`. Tags we do
 *     not map fall through to `unknown`.
 *   - Parametric tags (`list[int]`, `dict[str, int]`) lose their inner type
 *     but are parsed correctly as a single field (commas inside brackets do
 *     not split field boundaries).
 *   - Quoted strings, escapes, and custom types are not modeled.
 *
 * Anything outside these caps should degrade to `Record<string, unknown>` so
 * that type errors from this parser never outrank the authoritative runtime
 * parser in `src/signature.ts`.
 */

import type { Signature } from './signature.js';

// ── Whitespace handling ─────────────────────────────────────────────────

type Whitespace = ' ' | '\n' | '\t' | '\r';

type TrimLeft<S extends string> = S extends `${Whitespace}${infer R}` ? TrimLeft<R> : S;
type TrimRight<S extends string> = S extends `${infer L}${Whitespace}` ? TrimRight<L> : S;
export type Trim<S extends string> = TrimRight<TrimLeft<S>>;

// ── Scalar type-tag mapping ─────────────────────────────────────────────

type MapTypeTag<T extends string> =
  T extends 'str' ? string :
  T extends 'int' | 'float' ? number :
  T extends 'bool' ? boolean :
  T extends 'list' ? readonly unknown[] :
  T extends 'dict' ? Record<string, unknown> :
  unknown;

/**
 * Strip one optional `[...]` parameter list from a type annotation so that
 * `list[int]` maps to the same base tag (`list`) as `list`. Inner type args
 * are intentionally discarded — we do not try to type the element type of a
 * parametric container at the signature-string level.
 */
type ParseTypeAnnotation<T extends string> =
  Trim<T> extends `${infer BaseType}[${string}]` ? MapTypeTag<Trim<BaseType>> :
  MapTypeTag<Trim<T>>;

// ── Bracket-aware field-list splitting ──────────────────────────────────

/**
 * Depth counter for nested `[ ]`, `( )`, and `{ }`. We only split on commas
 * that sit at depth zero, matching the runtime's `splitTopLevel` behavior in
 * `src/split.ts`. The counter is represented as a tuple of `unknown`s so
 * that "increment" and "decrement" are tuple append/rest operations, which
 * the TS type system evaluates cheaply.
 */
type Depth = readonly unknown[];
type Inc<D extends Depth> = [unknown, ...D];
type Dec<D extends Depth> = D extends readonly [unknown, ...infer Rest] ? Rest : D;
type IsZero<D extends Depth> = D extends readonly [] ? true : false;

/**
 * Split `S` on top-level commas, tracking bracket depth. `Current` is the
 * field accumulator; `Acc` is the list of finalized fields. Opening brackets
 * push depth, closing brackets pop, a comma at depth 0 flushes `Current`
 * into `Acc`.
 */
type SplitFields<
  S extends string,
  D extends Depth = [],
  Current extends string = '',
  Acc extends readonly string[] = [],
> =
  S extends `${infer C}${infer Rest}`
    ? C extends '[' | '(' | '{'
      ? SplitFields<Rest, Inc<D>, `${Current}${C}`, Acc>
      : C extends ']' | ')' | '}'
        ? SplitFields<Rest, Dec<D>, `${Current}${C}`, Acc>
        : C extends ','
          ? IsZero<D> extends true
            ? SplitFields<Rest, D, '', [...Acc, Current]>
            : SplitFields<Rest, D, `${Current}${C}`, Acc>
          : SplitFields<Rest, D, `${Current}${C}`, Acc>
    : Current extends '' ? Acc : [...Acc, Current];

// ── Field parsing ───────────────────────────────────────────────────────

/**
 * Parse a single field segment into a `{ name: type }` singleton record.
 * Segments without a `:` default to `str` (matching the runtime default in
 * `parseFieldList` for untagged names, before the full Signature object
 * resolves the final `TypeTag`).
 */
type ParseField<S extends string> =
  Trim<S> extends '' ? Record<never, never> :
  Trim<S> extends `${infer Name}:${infer Type}`
    ? { [K in Trim<Name>]: ParseTypeAnnotation<Type> }
    : { [K in Trim<S>]: string };

type MergeFieldRecords<Fields extends readonly string[]> =
  Fields extends readonly [infer Head extends string, ...infer Tail extends readonly string[]]
    ? ParseField<Head> & MergeFieldRecords<Tail>
    : Record<never, never>;

/**
 * Flatten a `{…} & {…} & …` intersection into a single object type so IDEs
 * surface a clean shape instead of the raw intersection. Also exported so
 * that predictor subclasses which combine inferred outputs with synthetic
 * fields (e.g. `ChainOfThought` adding `reasoning`) can present a single,
 * normalized record to downstream type assertions.
 */
export type Flatten<T> = { [K in keyof T]: T[K] };

type ParseFieldList<S extends string> = Flatten<MergeFieldRecords<SplitFields<S>>>;

// ── Public entry points ─────────────────────────────────────────────────

/**
 * Parse a DSPy signature string into input and output records. Returns the
 * permissive `Record<string, unknown>` fallback for any non-literal string
 * or malformed signature, which keeps non-literal callers (variables,
 * dynamically-built strings) from receiving spurious type errors.
 */
export type ParseSignature<S extends string> =
  Trim<S> extends `${infer Inputs}->${infer Outputs}`
    ? {
        inputs: ParseFieldList<Inputs>;
        outputs: ParseFieldList<Outputs>;
      }
    : { inputs: Record<string, unknown>; outputs: Record<string, unknown> };

/**
 * Extract the input record from any constructor-arg-like value. When `T` is
 * a string literal we run it through `ParseSignature`; when `T` is a
 * `Signature` object (or the widened `string` type) we fall back to the
 * permissive record so runtime behavior is unchanged.
 */
export type InferInputs<T> =
  T extends string
    ? string extends T
      ? Record<string, unknown>
      : ParseSignature<T>['inputs']
    : Record<string, unknown>;

export type InferOutputs<T> =
  T extends string
    ? string extends T
      ? Record<string, unknown>
      : ParseSignature<T>['outputs']
    : Record<string, unknown>;

/**
 * Accepted constructor-argument shape for every typed predictor class. We
 * widen to `string | Signature` at the class-generic level so that the
 * generic defaults give the right fallbacks for each case.
 */
export type SignatureInput = string | Signature;

export type IsStringLiteral<T> = T extends string ? string extends T ? false : true : false;
