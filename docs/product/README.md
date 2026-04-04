# Product Decisions

This folder records product-specific engineering decisions that future
maintainers are likely to ask about.

The goal is not to duplicate the formal spec or the tests. The goal is to
capture the missing layer between them:

- what we decided
- why we decided it
- what trade-offs we accepted
- what would justify revisiting it later

## Scope

Documents here should focus on decisions that materially shape the product:

- API boundaries
- runtime semantics
- feature-specific behavior
- compatibility policy
- non-obvious ergonomics or DX choices

These are not generic meeting notes and not a changelog.

## How To Use This Folder

When you make a decision that a careful future maintainer might question, add a
small document here.

Prefer one file per decision area. Keep documents concrete and stable.

Good topics:

- why `Predict` rejects positional arguments
- why the spec is authoritative over AX oracle behavior
- why `Predict` is both a `Module` and a `Parameter`
- why LM formatting/parsing lives in `Adapter` rather than `BaseLM`

## Suggested Structure

Use short sections like these:

- `Status`
- `Decision`
- `Why`
- `What This Means In Code`
- `Revisit If`

## Current Decision Records

- [Spec Authority And Oracle Boundary](file:///Users/emre/codebase/minimal_dspy/dspy-slim-ts/docs/product/spec-authority-and-oracle-boundary.md)
- [Runtime Spine And Traversal Semantics](file:///Users/emre/codebase/minimal_dspy/dspy-slim-ts/docs/product/runtime-spine-and-traversal-semantics.md)
- [LM Adapter Predict Boundary](file:///Users/emre/codebase/minimal_dspy/dspy-slim-ts/docs/product/lm-adapter-predict-boundary.md)
