# dspy-slim-ts Craftsmanship Refactor — Phase 1

Scope the TS port from "faithful slim port" to "tighter than the Python original" by removing the curl-based sync LM transport, containing provider-specific quirks behind a profile plugin, and raising the type-level ambition of signatures — in three tightly-scoped workstreams with verifiable exits.

## Why this plan looks the way it does

- You chose **Craftsmanship refactor** as the sole focus axis and **Deprecate sync LM entirely** as the sync-path policy. Everything else is out of scope for this cycle and called out explicitly at the end.
- The spec in `@/Users/emre/codebase/dspy-slim-project/spec/abstractions.md` already carries all four tiers. Tests in `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/tests` are green. The floor is intact; this plan raises the ceiling.
- I will not touch the spec surface or the fixture corpus. If any refactor requires a spec change, it becomes a separate plan.

## Current-state diagnosis

| Smell | Evidence | Why it matters |
|---|---|---|
| Sync LM spawns a `curl` subprocess | `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/lm.ts:5`, `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/lm.ts:637-680` | OS dependency, blocking, non-idiomatic, and leaks into `RLM.forward` via `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/rlm.ts:506-515` |
| Minimax/OpenRouter branches in core transport | `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/lm.ts:453-507` and `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/adapter.ts:303-353` | A core module that knows about one model family is not a core module |
| Signatures parsed at runtime only | `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/signature.ts:386-460` | TS's biggest craftsman advantage over Python is unused — users get no compile-time typing of `"q: string -> a: string"` |
| Tooling inconsistency | `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/package.json:23` and `@/Users/emre/codebase/dspy-slim-project/benchmarks/release_gate.py:208` shell out to `npm pack` while repo policy is pnpm | Minor, but visible; trust signal |

## Workstream A — Kill the curl transport, split sync from async cleanly

**Goal.** Remove `execFileSync('curl', …)` from the runtime and re-draw the sync/async boundary so that:

- `LM` (the real network client) is **async-only**. No `forward`; only `aforward`/`acall`.
- `BaseLM` keeps a sync `forward`/`call` surface, but a default `generate` that throws a clear "this LM does not support sync; implement `generate` or use `acall`" message. Test-only subclasses (`ReplayLM`, fixture LMs, user-stubbed LMs) can still implement sync `generate` and keep working.
- `Predict.forward` stays as a sync entrypoint *on paper* but is documented as "works only with sync-capable LMs (e.g. `ReplayLM`)". With a real async `LM`, it throws immediately from `BaseLM.call`.
- `RLM.forward` is explicitly removed from the public surface. `RLM.aforward` becomes the only public entry. Internally, `llmQuery`/`llmQueryBatched` inside the Node `vm` context are bridged to the host-thread async LM via a **worker-thread + `SharedArrayBuffer`/`Atomics.wait` shim** so the vm script can keep its synchronous grammar while the host thread drives true async I/O. The shim lives in a new `src/sync_await_bridge.ts` and is the only place where we reach for that pattern.

**Concrete edits (read-only until you approve the plan).**

- `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/lm.ts`: delete `execFileSync` import, `parseCurlResponse`, `runSyncRequest`, the sync branch of `requestWithRetries` and `performRequest`. `LM.forward` either gets removed or overridden to throw `RuntimeError('LM is async-only; use acall/aforward.')`.
- `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/rlm.ts`: remove `forward`, keep `aforward`; replace `querySubLm` with a worker-bridged resolver; move `runWithSyncSession` and all sync-only interpreter glue out of the RLM public surface.
- `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/node_code_interpreter.ts`: keep `executeSync` for deterministic unit tests, but surface `execute` as the primary path and wire the async bridge so `llmQuery` inside a vm script can `Atomics.wait` on the host-thread's async LM response.
- New `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/sync_await_bridge.ts`: the one place in the codebase that uses `node:worker_threads` + `SharedArrayBuffer` + `Atomics.wait` to turn an async call into a synchronous `llmQuery` return value inside a worker-hosted vm.
- Add a new product decision record at `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/docs/product/sync-lm-removal.md` documenting: why we removed curl, why `BaseLM` keeps a sync surface for test LMs, why RLM uses a worker bridge rather than Node's experimental `vm.SourceTextModule`.

**Verification.**

- All existing tests keep passing. The `predict.test.ts` cases that use `TestLM`/`SequenceLM` (sync) still work because they override `generate`.
- A new test asserts `new LM('…').forward(...)` throws a clear async-only error.
- A new `tests/rlm_async_bridge.test.ts` exercises `RLM.aforward` against a truly async mock LM (`async generate`) and verifies `llmQuery` inside the vm returns the resolved value.
- `rlm_replay.json` fixture tests keep passing; they use `ReplayLM` which remains sync.
- `release:gate`, `typecheck`, and Node matrix CI all green.

**Known risks and how they're contained.**

- Worker-thread + `Atomics.wait` adds a new Node-version-sensitive path. Mitigation: gate behind the existing `engines.node >= 20` requirement and add a `node-matrix` job assertion.
- `vm.Script` inside a worker has slightly different global plumbing than the main thread. Mitigation: keep the interpreter's public API identical; the worker is an implementation detail behind `createSession`.
- Some users' in-process subclasses may have assumed sync `LM.forward`. Mitigation: release note + `@deprecated` JSDoc for one minor version before hard-removal (but per your choice, the code path is removed now; the deprecation window is purely documentation).

## Workstream B — Extract provider quirks behind a profile plugin

**Goal.** `src/lm.ts` and `src/adapter.ts` stop knowing the string `"minimax"` entirely. Provider-specific transformations (request defaults, adapter retries, response massaging) live in a new `src/providers/` folder behind a small typed contract.

**Contract sketch (to be finalized in implementation).**

```ts
// src/providers/profile.ts
export interface ProviderProfile {
  readonly id: string;
  readonly matches: (model: string) => boolean;
  readonly mapRequest?: (req: Record<string, unknown>) => Record<string, unknown>;
  readonly adapterRetry?: (
    lm: BaseLM,
    kwargs: Record<string, unknown>,
    error: unknown,
  ) => Record<string, unknown> | null;
}
```

**Concrete edits.**

- New `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/providers/profile.ts` defining the contract and a tiny registry (`registerProfile`, `resolveProfile`).
- New `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/providers/openrouter_minimax.ts` moving `isOpenRouterMinimaxModel`, `applyOpenRouterMinimaxReasoningDefaults`, `applyOpenRouterMinimaxOutputFloor`, `shouldRetryWithOpenRouterMinimaxFallback`, `withOpenRouterMinimaxMinimalReasoning` out of `lm.ts`/`adapter.ts`.
- `src/lm.ts`: `performRequest` calls `resolveProfile(this.model)?.mapRequest?.(body) ?? body` instead of the hard-coded helpers.
- `src/adapter.ts`: the retry path calls `profile.adapterRetry(...)` instead of the `shouldRetryWithOpenRouterMinimaxFallback` hard-check.
- Profiles are registered once in `src/index.ts` (or a new `src/providers/index.ts` re-exported through `src/index.ts`) so core modules do not import concrete profiles.
- Add a decision record at `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/docs/product/provider-profile-boundary.md`.

**Verification.**

- Existing Minimax-specific tests (`predict.test.ts` retry case at `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/tests/predict.test.ts:292-332`) pass unchanged.
- New test adds a custom profile for a fictional provider and asserts core modules dispatch to it correctly.
- Grepping `"minimax"` in `src/lm.ts` and `src/adapter.ts` returns zero hits.

## Workstream C — Raise the type-level ambition of signatures

**Goal.** Users get compile-time typing for signature strings, so `new Predict('question: string -> answer: string')` produces a `Predict<{ question: string }, { answer: string }>` with IDE autocompletion on `.forward({...})` inputs and `.forward(...).get('answer')` outputs — without changing the spec or runtime semantics.

This is the single biggest TypeScript-native win we can claim over Python DSPy. AX has a partial version of this; we can do it better because our `TypeTag` universe is richer.

**Approach.**

- Add a new `src/signature_types.ts` with template-literal-string types that parse a signature string at the type level and yield `{ Inputs; Outputs }` records. The runtime parser in `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/signature.ts:386-460` stays authoritative; the type-level parser is purely advisory.
- `Predict`, `ChainOfThought`, `ReAct`, `RLM` grow generic parameters that flow from the signature string literal when one is used: `class Predict<TSig extends string, TInputs = Parse<TSig>['inputs'], TOutputs = Parse<TSig>['outputs']>`.
- When a `Signature` object (not string) is passed, the generics default to `Record<string, unknown>` — identical to today. This keeps the runtime API backward-compatible.
- `Prediction.get` stays `unknown` (we do not claim runtime type safety for LM outputs), but we expose a `Prediction.getTyped<K extends keyof TOutputs>(key: K): TOutputs[K] | undefined` variant wired through `Predict.forward`.

**Concrete edits.**

- New `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/src/signature_types.ts`: pure TS types, no runtime.
- `src/predict.ts`, `src/chain_of_thought.ts`, `src/react.ts`, `src/rlm.ts`: add optional generic parameters; default to existing behavior.
- `src/index.ts`: export the type-level utilities.
- New `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/tests/signature_types.test-d.ts` (or a `.test.ts` using `expectTypeOf`) that verifies type inference for the main signature forms.
- Decision record at `@/Users/emre/codebase/dspy-slim-project/dspy-slim-ts/docs/product/type-level-signature-inference.md` explaining the runtime/type split and why we did not try to typecheck LM outputs.

**Verification.**

- `tsc --noEmit -p tsconfig.typecheck.json` passes.
- Type-level tests assert inputs/outputs are inferred for representative signatures (`"q -> a"`, `"q: string, ctx: list -> a: string"`, a custom-type case).
- All existing runtime tests pass unchanged.

## Workstream D — Housekeeping

Small, verifiable cleanups that do not deserve their own workstream but are worth doing while the codebase is open.

- ✅ **Done.** `package.json:23` now runs `pnpm pack --dry-run` for human-facing surface inspection. `benchmarks/release_gate.py:208` intentionally keeps `npm pack --dry-run --json` because the two tools disagree on JSON shape (`npm` returns a list-of-objects, `pnpm` returns a single object), and the gate's parser was written against `npm`'s shape. The fallback documented in the open-questions section below is the resolved position.
- ⛔ **Struck.** There is no `dspy-slim-ts/AGENTS.md` to align. The workspace `.gitignore` deliberately excludes per-workspace `AGENTS.md` files as local, agent-specific instructions; the single authoritative reference is `@/Users/emre/codebase/dspy-slim-project/AGENTS.md`, which points at `dspy-slim/docs/FORK.md`.
- ✅ **Done.** `@ax-llm/ax` stays as a `devDependency` in `package.json`; `grep -rn "@ax-llm/ax" src/` returns zero hits, so the oracle-only role is preserved by the import graph itself. The rationale is captured in `docs/product/spec-authority-and-oracle-boundary.md`; no additional inline comment is needed.

## Sequencing and size

| Phase | Workstream | Rough size | Gate |
|---|---|---|---|
| 1 | A — kill curl, async-only LM, RLM worker bridge | Large; new file, three touched files, new tests | All existing tests green + new async-bridge test |
| 2 | B — provider profile plugin | Medium; two new files, two touched files | Grep `minimax` absent from `lm.ts`/`adapter.ts` |
| 3 | C — type-level signatures | Medium; one new file, four touched files, type-level tests | `typecheck` green, `.test-d.ts` green |
| 4 | D — housekeeping | Small | Existing gates |

Workstreams A and B can land together if convenient; C is independent; D is last and can piggyback on any phase.

## Out of scope for this plan

Called out explicitly so we do not accidentally drift:

- **Launch readiness** — no README rewrite, no runnable Predict/CoT/ReAct examples, no consumer-tarball smoke test, no TS-vs-Python benchmark story. These are real gaps per `@/Users/emre/codebase/dspy-slim-project/pre-release-checklist.md` but not what you picked.
- **RLM depth parity with upstream Python** — no `max_output_chars`, no history compaction, no paper-appendix prompt presets (`qwen_coder` / `qwen_small`), no concurrent `llm_query_batched`, no richer action instructions. Upstream Python RLM at `@/Users/emre/codebase/dspy-slim-project/dspy-slim/dspy/predict/rlm.py` remains richer. A future plan.
- **GEPA frontend / optimize_anything** — the shipped `createGatedGEPAEngine` stays gated. `@/Users/emre/codebase/dspy-slim-project/gepa.md`'s `optimize_anything` surface is not ported.
- **New research integration (Ramp RLM)** — `next_steps.md` mentions "new research for RLMs from Ramp" but the reference is ambiguous (likely not the already-implemented arXiv:2512.24601). Punting until you clarify.
- **Spec changes** — `@/Users/emre/codebase/dspy-slim-project/spec/abstractions.md` stays frozen for this cycle.

## Open questions (non-blocking for approval)

- For Workstream A's RLM worker bridge: is `node:worker_threads` + `SharedArrayBuffer`/`Atomics.wait` acceptable, or do you want me to prototype alternatives (e.g., Node's experimental `vm.SourceTextModule`, or a collect-and-resume RLM loop that avoids sync-await entirely)? The worker bridge is my default pick because it preserves the RLM paper's "code runs sync, llm_query returns synchronously" semantics without Node experimental flags.
- For Workstream C: how ambitious on type-level parsing? A minimal version types only the field names (strings always). A richer version types `typeTag` scalars. Going further into `literal`/`enum`/`union`/`custom` gets baroque and may not pay off. Default pick: types names + scalar tags (`str`/`int`/`float`/`bool`/`list`/`dict`), stop there.
- For Workstream D's `pnpm pack`: `npm pack --dry-run --json` output is consumed by `release_gate.py`; I want to verify the JSON shape is identical before switching. If not, I would leave `npm pack` there and add a comment; the repo is pnpm for install and `npm pack` only as a packaging inspection tool is defensible.
