# Runtime Spine And Traversal Semantics

## Decision

The runtime spine is built around a few strict ideas:

- `Field` and `Signature` are immutable value objects
- `Example` and `Prediction` are separate concepts
- `Predict` is both a `Module` and a `Parameter`
- module traversal gives `Parameter` precedence over `Module`
- `Settings` uses async-local contextual overrides instead of mutable global snapshots

## Why

These choices keep the runtime durable under composition.

The main risk in a library like this is semantic drift caused by implicit state, fuzzy ownership, or traversal rules that change depending on inheritance shape. The current spine avoids that.

## What We Decided

### Predict Is A Terminal Parameter Leaf

`Predict` participates in module graphs, but when traversing parameters it must be treated as a parameter leaf rather than recursively descended into as a module.

This is why traversal checks `Parameter` before `Module`.

That preserves a stable identity for predictors inside composed modules and avoids surprising traversal behavior.

### Settings Context Stores Overrides, Not Frozen Snapshots

Contextual settings are async-local, but the stored data is the override set, not a frozen copy of the entire world.

That means later global `configure()` updates remain visible when computing a fresh snapshot, while context-local overrides still stay isolated.

### Owned Values Flow Through Shared Snapshot/Serialization Utilities

Nested records, arrays, and value-like objects should move through the shared owned-value layer rather than bespoke cloning rules in each module.

This keeps copy/serialization/equality semantics consistent across the runtime.

## What This Means In Code

- `Module.namedParameters()` must preserve `Parameter`-before-`Module` precedence
- predictor discovery relies on runtime branding, not only inheritance
- `Settings.context()` should merge override layers rather than materializing a disconnected frozen snapshot
- new runtime layers should reuse the shared owned-value utilities instead of inventing parallel copy semantics

## Revisit If

- we introduce a new category of runtime object that must participate in module traversal
- async-local settings stop being sufficient for concurrency requirements
- we find a concrete bug caused by predictor leaf semantics rather than a hypothetical one
