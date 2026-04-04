# Spec Authority And Oracle Boundary

## Status

Accepted.

## Decision

The authoritative behavior for this project comes from
`dspy-slim/spec/abstractions.md` and our spec fixtures, not from AX.

AX is used only as an oracle-style compatibility report in
`tests/ax_oracle.test.ts`.

## Why

We want a craftsman-quality TypeScript port of DSPy semantics, not a clone of
another JavaScript library's constraints or bugs.

AX is useful as an external reference point, but it is not the source of truth.
If we let oracle behavior drive implementation design, we would gradually erode
the product's correctness, semantics, and developer experience.

This is especially important in places where AX and the spec intentionally
diverge.

## What We Deliberately Preserve

- DSPy type names such as `str`, `int`, `float`, and `bool`
- our `inferPrefix` behavior from the spec instead of AX title formatting
- more permissive field name validation that matches DSPy Python semantics
- fixture-driven behavior from the formal spec even when AX cannot validate it

## What This Means In Code

- `@ax-llm/ax` is imported only in `tests/ax_oracle.test.ts`
- AX failures must never block CI
- AX limitations should be skipped with a visible warning rather than worked
  around in product code
- we do not translate our inputs to fit AX's internal preferences

## Revisit If

- the formal spec changes
- DSPy Python changes in a way that should change our semantics
- AX becomes useful for a new compatibility check that does not pressure the
  implementation boundary
