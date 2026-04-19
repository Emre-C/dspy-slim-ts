# Type-Level Signature Inference

## Decision

DSPy signature strings are parsed twice: once at runtime by `parseSignature`
in `src/signature.ts`, and once at compile time by `ParseSignature<S>` in
`src/signature_types.ts`. The type-level parser is **advisory only** — it
never changes runtime behavior — but its result flows through `Predict`,
`ChainOfThought`, `ReAct`, and `RLM` as `TInputs` and `TOutputs` generic
parameters so IDE autocompletion and `tsc` excess-property checks reflect
the declared signature.

## Why

TypeScript's largest craftsmanship advantage over Python DSPy is the ability
to surface signature mistakes at edit time. A caller who writes
`new Predict('question -> answer').forward({ questoin: 'hi' })` should see
a red squiggle on `questoin`, not a runtime warning 500ms later after a
network call. Nothing else in the DSPy surface area gets meaningful safety
from the TS type system unless signatures are typed; everything downstream
(`Predict.forward` inputs, `Prediction.getTyped` outputs, `ChainOfThought`
reasoning augmentation) flows from this one parse.

Equally important: the feature must not leak. A caller building a signature
dynamically (`new Predict(signatureBuiltFromApi)`) should see the historical
permissive `Record<string, unknown>` shape and no spurious type errors. The
type-level parser is therefore gated behind `string` being a literal — non-
literal strings and `Signature` objects fall back to `Record<string, unknown>`
on both sides.

## What We Decided

### Runtime parser remains authoritative

`parseSignature` / `signatureFromString` in `src/signature.ts` is the single
source of truth for what a signature string means at runtime. The type-level
parser is a compile-time mirror whose only job is to feed IDE tooling. When
the two diverge, the runtime wins; when they agree, the user sees the same
input field keys and output field types at edit time as at runtime.

### Type-level parser tracks bracket depth

`SplitFields<S>` in `src/signature_types.ts` walks the signature character by
character, tracking `[`, `(`, `{` depth so that commas inside parametric
types (`dict[str, int]`, `list[str]`) do not split field boundaries. This
matches `splitTopLevel` in `src/split.ts` for the cases the type parser
supports, and it is the single largest correctness gap over a naive template
literal split.

### Scope caps

The type-level parser is intentionally narrower than the runtime:

- **Scalar tag mapping only.** `str → string`, `int | float → number`,
  `bool → boolean`, `list → readonly unknown[]`, `dict → Record<string, unknown>`.
  Unknown tags map to `unknown`.
- **Parametric type inner type is discarded.** `list[int]` and `list` both
  resolve to `readonly unknown[]` at the type level; the runtime still stores
  the inner `typeTag` on the `Field` for adapter-level behavior.
- **No quoted strings, escapes, or custom types.** If a signature uses these,
  the type-level parser falls through to `unknown`; the runtime still parses
  correctly.
- **Non-literal strings get `Record<string, unknown>`.** `ParseSignature<string>`
  deliberately returns the permissive shape, so dynamically built signatures
  carry their historical behavior.

These caps exist because the type-level parser is written in template
literal types and hits TS's recursion/instantiation depth quickly on rich
grammars. A narrower parser that is correct within its scope is more
valuable than an ambitious one that subtly differs from the runtime.

### `Prediction.getTyped` throws instead of returning `undefined`

The plan's initial sketch had `getTyped(key): TOutputs[K] | undefined`. The
ADR settled on `getTyped(key): TOutputs[K]` with `KeyError` on a missing
key, matching the existing `Prediction.get` contract. Two reasons:

1. **Consistency.** `get` throws; the typed counterpart should throw too.
   Mixing semantics in sibling methods violates the principle of least
   astonishment.
2. **No ambiguity.** `undefined` conflates "key absent" with "value was
   actually undefined". Throwing forces the caller to use `getOr` when they
   want a default, which is the explicit control-flow.

The generic is an **advisory annotation**, not a runtime check. Callers who
store non-conforming data under a typed `Prediction` will see a cast-through
at `getTyped`; they will not see a runtime validation error. This is a
deliberate trade: we do not duplicate the adapter's parsing contract.

## What This Means In Code

- `ParseSignature<'q: str, ctx: list -> a: str, score: float'>` evaluates to
  `{ inputs: { q: string; ctx: readonly unknown[] }; outputs: { a: string; score: number } }`.
- `new Predict('q -> a').forward({ q: 'x' })` compiles; same call with
  `{ z: 'x' }` is a TypeScript error on a fresh object literal.
- `new Predict(aSignatureObject).forward({ anythingAtAll: ... })` compiles,
  exactly as before — `TSig = Signature` falls through to the permissive
  shape.
- `Prediction.create<{ answer: string }>({ answer: 'Paris' }).getTyped('answer')`
  has compile-time type `string`; the same call on a Prediction without the
  `answer` key throws `KeyError` at runtime.
- `ChainOfThought<'q -> a'>`'s output type is
  `Flatten<{ a: string } & { reasoning: string }>`, reflecting the synthetic
  reasoning field the class prepends at runtime.

## How We Keep The Parsers In Sync

Drift between runtime and type-level parsers is a real risk. We mitigate it
with:

- **Shared fixture assertions.** Every non-trivial form in
  `tests/signature_types.test.ts` is parsed at both levels and the keys are
  asserted equal. New signature syntax must update both parsers and extend
  this fixture.
- **Negative tests.** The test file uses `@ts-expect-error` to prove that
  unknown input keys are rejected. If a future edit to `PredictKwargs`
  reopens the escape hatch, the negative test will fail to surface the
  expected error and the suite will break.
- **Scope-cap audits.** When a new signature feature ships in the runtime
  parser (a new type tag, a new delimiter), its type-level counterpart is
  added or the limitation is documented here before release.

## Revisit If

- The runtime parser grows grammar the type-level side cannot follow
  cleanly (e.g. user-defined type tags), and the scope-cap approach stops
  being honest.
- TypeScript's template-literal recursion limits raise or lower enough to
  change what is expressible (we are currently well inside them).
- A user-facing feature needs genuine runtime type validation of LM outputs
  (Zod-style), which would change the `Prediction` contract beyond an
  advisory generic.
