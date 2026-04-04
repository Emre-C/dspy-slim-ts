# LM Adapter Predict Boundary

## Status

Accepted.

## Decision

The runtime boundary is intentionally split into three layers:

- `BaseLM` is the model/client runtime contract
- `Adapter` owns prompt/message formatting and response parsing
- `Predict` owns orchestration, defaults resolution, and trace capture

## Why

These responsibilities change at different speeds and for different reasons.

Provider clients change because APIs, capabilities, and transport details vary.
Prompt formatting and parsing change because the product's structured prompt
grammar changes. Predict changes because invocation ergonomics and orchestration
rules evolve.

Keeping those concerns separated makes the code easier to reason about and less
fragile when one layer evolves.

## What We Decided

### BaseLM Is Not The Parsing Layer

`BaseLM` exists to provide runtime identity, model metadata, default kwargs,
capability predicates, and sync/async generation entrypoints.

It is not the place where structured outputs are parsed back into signature
fields.

### Adapter Owns The Prompt Grammar

Adapters format system/demo/user messages from signatures and demos, invoke the
LM, and parse raw text completions into typed output records.

This keeps prompt grammar and parsing policy in one place.

### Predict Owns Control-Plane Semantics

`Predict` resolves the effective signature, demos, config, LM, and adapter. It
also handles input defaults, warnings for missing or extra inputs, temperature
auto-adjust behavior, and trace storage.

Control keys such as `signature`, `demos`, `config`, and `lm` are reserved for
Predict overrides and should not double as user input field names.

## Durability Notes

- LM default kwargs are merged centrally by `BaseLM` so concrete providers do
  not drift on merge behavior
- Predict uses explicit nullish resolution for control-plane defaults instead of
  Python-style truthiness
- reserved control keys are rejected early rather than silently shadowing user
  inputs

## Revisit If

- we need a stable third-party LM plugin protocol that benefits from runtime
  branding without inheritance
- adapter responsibilities become too broad because of native tool calling or
  schema-specific transport behavior
- we add streaming or richer raw-response surfaces that require a clearer split
  between LM transport and parsed outputs
