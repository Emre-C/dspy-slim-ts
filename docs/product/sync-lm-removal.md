# Sync LM Removal

## Decision

The `execFileSync('curl', …)` sync HTTP transport has been removed from `LM`.
`LM.forward` now throws `RuntimeError('LM is async-only. Use acall() or aforward() instead.')`.

## Why curl was removed

- **OS dependency.** `curl` availability and behavior differ across macOS, Linux, and CI images.
- **Blocking I/O.** Synchronous subprocesses block the Node event loop, which violates the platform's
  concurrency model and can cause timeouts in long-running pipelines.
- **Non-idiomatic.** Every test had to mock `execFileSync`; the fetch-based async path already
  existed and was the only path used in production.

## Why `BaseLM` keeps a sync surface for test LMs

`BaseLM.call` → `BaseLM.forward` → `BaseLM.generate` remains synchronous and functional.
Test stubs (`TestLM`, `SequenceLM`, `ReplayLM`) override `generate()` and return data
synchronously without network I/O.  This preserves sync `Predict.forward` deterministic
tests without requiring an event loop.

The key distinction:

| Class     | `forward`              | `aforward`           |
|-----------|------------------------|----------------------|
| `BaseLM`  | calls `generate()`     | calls `agenerate()`  |
| `LM`      | **throws** async-only  | async HTTP via fetch  |
| `ReplayLM`| replays from fixture   | delegates to sync    |

## Why the RLM worker bridge is not in this PR

The plan's Workstream A originally called for a `worker_threads` + `SharedArrayBuffer` +
`Atomics.wait` bridge so that `RLM.forward` could drive a real async `LM` from inside the
synchronous vm interpreter.

This was descoped because:

1. All production RLM usage already goes through `aforward`.
2. `RLM.forward` works with sync-capable LMs (`ReplayLM`, test stubs).
3. The worker bridge introduces Node-version-sensitive complexity; it deserves its own
   focused PR with dedicated tests.

If a user needs `RLM.forward` with a real `LM` instance, they should use `RLM.aforward`
instead, or the worker bridge can be added as a follow-up.
