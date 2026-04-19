# Release Gate Pack Tooling

## Decision

- Local inspection uses **`pnpm pack --dry-run`** (`package.json` script `pack:dry-run`).
- The machine release gate in **`benchmarks/release_gate.py`** continues to run **`npm pack --dry-run --json`** with its stdout parsed as JSON.

## Why

`npm` and `pnpm` both implement pack dry-runs, but their **`--json` output shapes differ** (for example, `npm` emits a top-level JSON array of package metadata objects; `pnpm` emits a single object). The release gate was written against `npm`'s shape and validates the `files` list from that structure. Switching the gate to `pnpm pack` would require revisiting the parser and failure messages.

Using **pnpm** for day-to-day tarball inspection matches repo install policy. Using **npm** inside the gate is a deliberate, stable contract for automated checks—not a claim that the two tools are interchangeable at the JSON layer.

## What This Means In Code

- Do not change `release_gate.py` to `pnpm pack` without updating `_parse_npm_pack_json` and the downstream assumptions that the payload is a non-empty array.
- Do not assume `pnpm pack --json` output can be fed through the same parser as `npm pack --dry-run --json` without an adapter.

## Revisit If

- The gate is rewritten to consume a format both pack implementations share, or
- `pnpm`/`npm` converge on identical JSON for the fields the gate cares about (`files[].path`, etc.).
