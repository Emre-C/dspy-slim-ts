/**
 * RLM v2 combinator AST.
 *
 * Plans are pure data: a discriminated union of plain objects with a `tag`
 * field and tag-specific readonly payload. There are no closures and no
 * methods. The deterministic planner introspects and rewrites plans —
 * substitute `k*` into a `split`, set `N` on a `vote`, narrow `models` on an
 * `ensemble` — which requires that every node be structurally inspectable.
 *
 * See:
 * - `docs/RLM_V2_IMPLEMENTATION_PLAN.md §2.1` (canonical shape)
 * - `spec/abstractions.md §0.5` (typed combinator runtime clause)
 *
 * A plan is JSON-serializable. Deserialized plans round-trip identically
 * through `evaluate()`; the `CombinatorFn` / `CombinatorBinary` bodies are
 * sub-ASTs with named parameters (lambda-calculus, not JS closures).
 *
 * This module is evaluator-agnostic: it defines shapes and typed
 * constructors only. The walker lives in `src/rlm_evaluator.ts`.
 */

// ---------------------------------------------------------------------------
// Value domain
// ---------------------------------------------------------------------------

/**
 * Union of values that flow through the evaluator. The evaluator performs
 * explicit runtime type guards at each combinator boundary and throws
 * `ValueError` on mismatches; this union documents the closure set of
 * expected shapes.
 */
export type CombinatorValue =
  | string
  | number
  | boolean
  | CombinatorList
  | Record<string, unknown>;

export type CombinatorList = readonly CombinatorValue[];

// ---------------------------------------------------------------------------
// Node union
// ---------------------------------------------------------------------------

/**
 * The `CombinatorNode` discriminated union. Every plan is a tree of these.
 *
 * Deterministic leaves (`literal`, `var`) and deterministic internal nodes
 * (`split`, `peek`, `map`, `filter`, `reduce`, `concat`, `cross`) execute
 * without the LLM. Neural leaves (`oracle`) and neural fan-outs (`vote`,
 * `ensemble`) invoke an LLM via `Predict`; those branches land in Phase 2
 * and throw `RuntimeError` at Phase 1.
 */
export type CombinatorNode =
  | { readonly tag: 'literal'; readonly value: CombinatorValue }
  | { readonly tag: 'var'; readonly name: string }
  | {
      readonly tag: 'split';
      readonly input: CombinatorNode;
      readonly k: CombinatorNode;
    }
  | {
      readonly tag: 'peek';
      readonly input: CombinatorNode;
      readonly start: CombinatorNode;
      readonly end: CombinatorNode;
    }
  | {
      readonly tag: 'map';
      readonly fn: CombinatorFn;
      readonly items: CombinatorNode;
    }
  | {
      readonly tag: 'filter';
      readonly pred: CombinatorFn;
      readonly items: CombinatorNode;
    }
  | {
      readonly tag: 'reduce';
      readonly op: CombinatorBinary;
      readonly items: CombinatorNode;
      readonly init?: CombinatorNode;
    }
  | {
      readonly tag: 'concat';
      readonly items: CombinatorNode;
      readonly separator?: CombinatorNode;
    }
  | {
      readonly tag: 'cross';
      readonly left: CombinatorNode;
      readonly right: CombinatorNode;
    }
  | {
      readonly tag: 'vote';
      readonly oracle: CombinatorNode;
      readonly n: CombinatorNode;
      readonly reducer?: VoteReducer;
    }
  | {
      readonly tag: 'ensemble';
      readonly oracle: CombinatorNode;
      readonly models: readonly string[];
      readonly reducer?: EnsembleReducer;
    }
  | {
      readonly tag: 'oracle';
      readonly prompt: CombinatorNode;
      readonly modelHint?: string;
      readonly effectHandlers?: readonly string[];
    };

/**
 * Function body for `map` / `filter`. A sub-AST with a bound parameter name.
 * The evaluator extends the enclosing scope with `param ↦ element` before
 * evaluating `body`.
 */
export interface CombinatorFn {
  readonly param: string;
  readonly body: CombinatorNode;
}

/**
 * Binary function body for `reduce`. The evaluator extends the enclosing
 * scope with `left ↦ accumulator` and `right ↦ element` before evaluating
 * `body`.
 */
export interface CombinatorBinary {
  readonly left: string;
  readonly right: string;
  readonly body: CombinatorNode;
}

export type VoteReducer = 'majority' | 'mode' | 'verifier';
export type EnsembleReducer = 'majority' | 'confidence' | 'verifier';

// ---------------------------------------------------------------------------
// Typed constructors
// ---------------------------------------------------------------------------
//
// These are the only sanctioned way to construct plan nodes. They enforce
// `exactOptionalPropertyTypes` by omitting fields when their callers don't
// supply them, so round-tripping through `JSON.stringify` produces the same
// shape as the constructor output.

export const lit = (value: CombinatorValue): CombinatorNode => ({
  tag: 'literal',
  value,
});

export const vref = (name: string): CombinatorNode => ({ tag: 'var', name });

export const split = (input: CombinatorNode, k: CombinatorNode): CombinatorNode => ({
  tag: 'split',
  input,
  k,
});

export const peek = (
  input: CombinatorNode,
  start: CombinatorNode,
  end: CombinatorNode,
): CombinatorNode => ({ tag: 'peek', input, start, end });

export const map = (fn: CombinatorFn, items: CombinatorNode): CombinatorNode => ({
  tag: 'map',
  fn,
  items,
});

export const filter = (
  pred: CombinatorFn,
  items: CombinatorNode,
): CombinatorNode => ({ tag: 'filter', pred, items });

export const reduce = (
  op: CombinatorBinary,
  items: CombinatorNode,
  init?: CombinatorNode,
): CombinatorNode =>
  init === undefined
    ? { tag: 'reduce', op, items }
    : { tag: 'reduce', op, items, init };

export const concat = (
  items: CombinatorNode,
  separator?: CombinatorNode,
): CombinatorNode =>
  separator === undefined
    ? { tag: 'concat', items }
    : { tag: 'concat', items, separator };

export const cross = (left: CombinatorNode, right: CombinatorNode): CombinatorNode => ({
  tag: 'cross',
  left,
  right,
});

export const vote = (
  oracleNode: CombinatorNode,
  n: CombinatorNode,
  reducer?: VoteReducer,
): CombinatorNode =>
  reducer === undefined
    ? { tag: 'vote', oracle: oracleNode, n }
    : { tag: 'vote', oracle: oracleNode, n, reducer };

export const ensemble = (
  oracleNode: CombinatorNode,
  models: readonly string[],
  reducer?: EnsembleReducer,
): CombinatorNode =>
  reducer === undefined
    ? { tag: 'ensemble', oracle: oracleNode, models }
    : { tag: 'ensemble', oracle: oracleNode, models, reducer };

export const oracle = (
  prompt: CombinatorNode,
  modelHint?: string,
  effectHandlers?: readonly string[],
): CombinatorNode => {
  if (modelHint !== undefined && effectHandlers !== undefined) {
    return { tag: 'oracle', prompt, modelHint, effectHandlers };
  }
  if (modelHint !== undefined) {
    return { tag: 'oracle', prompt, modelHint };
  }
  if (effectHandlers !== undefined) {
    return { tag: 'oracle', prompt, effectHandlers };
  }
  return { tag: 'oracle', prompt };
};

export const fn = (param: string, body: CombinatorNode): CombinatorFn => ({
  param,
  body,
});

export const bop = (
  left: string,
  right: string,
  body: CombinatorNode,
): CombinatorBinary => ({ left, right, body });
