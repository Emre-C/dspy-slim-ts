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

- `Decision`
- `Why`
- `What This Means In Code`
- `Revisit If`

The single `#` title is the decision name (e.g. `# Provider Profile Boundary`).
No separate date, status, or context headers — the commit history already tracks
those, and the document itself should be stable enough not to need them.

## Current Decision Records

- [Spec Authority And Oracle Boundary](./spec-authority-and-oracle-boundary.md)
- [Runtime Spine And Traversal Semantics](./runtime-spine-and-traversal-semantics.md)
- [LM Adapter Predict Boundary](./lm-adapter-predict-boundary.md)
- [Sync LM Removal](./sync-lm-removal.md)
- [Provider Profile Boundary](./provider-profile-boundary.md)
- [Type-Level Signature Inference](./type-level-signature-inference.md)
- [Release Gate Pack Tooling](./release-gate-pack-tooling.md) — why `pack:dry-run` uses pnpm while the automated gate uses `npm pack --json`
- [RLM v2 Architecture](./rlm-v2-architecture.md) — single canonical doc: motivation, decision record, layer diagram, module map, §2 type index, execution §4.x anchors, out-of-scope; supersedes v1 REPL and old vision/plan split
