# Provider Profile Boundary

## Decision

Provider-specific logic (request transformations, adapter retry strategies) is extracted
from `src/lm.ts` and `src/adapter.ts` into a typed plugin contract at
`src/providers/profile.ts`, with concrete implementations in `src/providers/`.

## Problem

`lm.ts` and `adapter.ts` both contained hard-coded references to `"minimax"` and
OpenRouter-specific request mutations.  This violates the single-responsibility principle:
core transport and adapter modules should not know about individual model families.

Adding support for a new provider (e.g., Anthropic-via-OpenRouter quirks) required editing
two core modules, increasing coupling and regression risk.

## Solution

### Contract

```ts
export interface ProviderProfile {
  readonly id: string;
  readonly matches: (model: string) => boolean;
  readonly mapRequest?: (req: Record<string, unknown>) => Record<string, unknown>;
  readonly adapterRetry?: (
    lm: BaseLM,
    lmKwargs: Record<string, unknown>,
    error: unknown,
  ) => Record<string, unknown> | null;
}
```

### Flow

1. `LM.performRequest` calls `resolveProfile(this.model)?.mapRequest?.(body)` to apply
   provider-specific request defaults.
2. `Adapter.call` / `Adapter.acall` calls `resolveProfile(lm.model)?.adapterRetry?.(…)`
   on parse failure to decide if a retry with modified kwargs is warranted.
3. Profiles are registered once at import time in `src/providers/index.ts`.

### Adding a new provider

Create `src/providers/my_provider.ts`, implement `ProviderProfile`, and register it in
`src/providers/index.ts`.  No changes to `lm.ts` or `adapter.ts` required.

## Verification

- `grep -rn "minimax" src/lm.ts src/adapter.ts` returns zero hits.
- All existing Minimax-specific tests pass unchanged (they use sync `SequenceLM` which
  implements `generate()` directly).
- A new provider can be tested by registering a fictional profile and asserting dispatch.
