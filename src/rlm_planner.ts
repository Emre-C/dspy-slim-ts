/**
 * Deterministic, quality-shaped planner for RLM v2.
 *
 * The planner maps a task type, a prompt length, and a budget into a
 * fully-resolved combinator plan. Every decision is an explicit pure
 * function — no heuristic tuning, no hidden state — so the same inputs
 * produce the same plan on every run, and the reasoning is inspectable
 * by humans and by tests.
 *
 * Core API: `computeOptimalPartitionSize`, `computeRecursionDepth`,
 * `computeSelfConsistencyN`, and `resolvePlan` (full `k*` / `d` / `N` pick,
 * call estimate, AST walk substituting `vref('k')` / `vref('n')` with literals).
 * Spec: `docs/product/rlm-v2-architecture.md` §4.2.
 *
 * Substitution convention (intentionally matching the combinator AST
 * field names):
 *
 * - `vref('k')` → `lit(partitionK)` — `split.k` is the partition count.
 * - `vref('n')` → `lit(selfConsistencyN)` — `vote.n` is the self-consistency
 *   sample width.
 *
 * Plan templates are designed to place `vref('k')` where `split.k` is
 * expected and `vref('n')` where `vote.n` is expected, so the walk
 * produces a semantically correct tree without any additional rewiring.
 *
 * The planner is a pure function; it never calls the LM, never reads
 * files, never reads the evaluator's trace. This matters because the
 * planner is called before `evaluate` has seen a single token, and its
 * output is the runtime contract the evaluator enforces.
 *
 * Quality curves live in `./rlm_planner_quality.ts` so tuning does not
 * touch planner logic.
 *
 * See also:
 * - `./rlm_task_router.ts` for `TaskType`, `PlanningInputs`, `ResolvedPlan`.
 * - `./rlm_planner_quality.ts` for the per-task-type quality curves.
 */

import type { CombinatorNode, CombinatorValue } from './rlm_combinators.js';
import { lit } from './rlm_combinators.js';
import type { MemorySchema } from './rlm_memory.js';
import type { RLMBudget } from './rlm_types.js';
import type { PlanningInputs, ResolvedPlan } from './rlm_task_router.js';
import {
  K_SEARCH_MAX,
  K_SEARCH_MIN,
  QUALITY_CURVES,
} from './rlm_planner_quality.js';

// ===========================================================================
// Public re-exports of search bounds
// ===========================================================================
//
// Re-exported here so callers import bounds from the planner module only.

export { K_SEARCH_MIN, K_SEARCH_MAX } from './rlm_planner_quality.js';

// ===========================================================================
// computeOptimalPartitionSize
// ===========================================================================

/** Inputs for partition-size search: core {@link PlanningInputs} fields without `preferredK`. */
type PartitionSizeInputs = Pick<
  PlanningInputs,
  'promptLength' | 'budget' | 'taskType'
>;

/**
 * Pick the partition argmax `k*` for the task.
 *
 * Algorithm:
 *
 * 1. Clamp candidate `k` to `[K_SEARCH_MIN, K_SEARCH_MAX]`. `k = 1`
 *    degenerates `split` into the identity and is disallowed; `k > 16`
 *    is better served by recursing with a smaller `k`.
 * 2. Further ceiling the candidates by the budget — the number of leaf
 *    partitions is bounded by `budget.maxParallelism` (one parallel
 *    oracle call per top-level chunk) and by `budget.maxOracleCalls`
 *    (no single plan may ask for more top-level chunks than the call
 *    budget allows, even with `N = 1`).
 * 3. If the budget-ceiling drops below `K_SEARCH_MIN = 2`, the plan is
 *    under-budgeted; the pairwise task type legitimately needs only
 *    `k = 2`, so we still return 2 and let `computeSelfConsistencyN`
 *    floor `N` to 1.
 * 4. Evaluate the task-type quality curve at every surviving integer
 *    `k` and return the argmax.
 *
 * Ties are broken toward the **smaller** `k`, because the
 * quality-per-oracle-call gradient always favors the smaller split when
 * two splits tie in raw quality. No seed curve produces integer ties on
 * the `[2, 16]` grid; this tie-break is defensive.
 */
export function computeOptimalPartitionSize(
  inputs: PartitionSizeInputs,
): number {
  const budgetCeiling = budgetPartitionCeiling(inputs.budget);
  const curve = QUALITY_CURVES[inputs.taskType];
  let bestK = K_SEARCH_MIN;
  let bestQuality = -Infinity;
  for (let k = K_SEARCH_MIN; k <= budgetCeiling; k += 1) {
    const quality = curve(k, inputs.promptLength);
    if (quality > bestQuality) {
      bestQuality = quality;
      bestK = k;
    }
  }
  return bestK;
}

// ===========================================================================
// computeRecursionDepth
// ===========================================================================

interface RecursionDepthInputs {
  readonly promptLength: number;
  readonly leafThreshold: number;
  readonly k: number;
}

/**
 * `⌈log_k(n / τ*)⌉`, floored at `1`.
 *
 * This is the depth of a balanced `k`-ary partition tree that
 * guarantees every leaf is at most `τ*` characters. The formula is
 * exact when `n` is a power of `k`; it rounds up otherwise, which is
 * the conservative choice.
 *
 * Edge cases:
 *
 * - `promptLength ≤ leafThreshold`: a single oracle call suffices, but
 *   we return `1` — never `0` — because `d = 0` collapses the plan to
 *   the bare oracle node and skips the top-level `split`, which the
 *   evaluator's bookkeeping (`ctx.depth`) does not expect.
 * - `k ≤ 1`: mathematically ill-defined; we throw `RangeError` because
 *   this is a programmer error — the argmax planner never returns
 *   `k < 2`.
 * - `leafThreshold ≤ 0` or non-finite `promptLength`: also a RangeError.
 */
export function computeRecursionDepth(
  inputs: RecursionDepthInputs,
): number {
  const { promptLength, leafThreshold, k } = inputs;
  if (!Number.isFinite(promptLength) || promptLength < 0) {
    throw new RangeError(
      `computeRecursionDepth: promptLength must be a finite non-negative number, got ${promptLength}`,
    );
  }
  if (!Number.isFinite(leafThreshold) || leafThreshold <= 0) {
    throw new RangeError(
      `computeRecursionDepth: leafThreshold must be a finite positive number, got ${leafThreshold}`,
    );
  }
  if (!Number.isInteger(k) || k < 2) {
    throw new RangeError(
      `computeRecursionDepth: k must be an integer ≥ 2, got ${k}`,
    );
  }
  if (promptLength <= leafThreshold) return 1;
  const ratio = promptLength / leafThreshold;
  const depth = Math.ceil(Math.log(ratio) / Math.log(k));
  return Math.max(1, depth);
}

// ===========================================================================
// computeSelfConsistencyN
// ===========================================================================

interface SelfConsistencyInputs {
  readonly budget: RLMBudget;
  readonly estimatedOracleCalls: number;
}

/**
 * Pick `N` — the self-consistency sample width.
 *
 * Returns the largest `N ≤ budget.selfConsistencyN` such that
 * `N * estimatedOracleCalls ≤ budget.maxOracleCalls`.
 * Floors at `1`: if even a single sample exceeds the call budget we
 * still return `1`, and downstream budget enforcement in the evaluator
 * throws `BudgetError` the first time a leaf is reached.
 *
 * This keeps the planner's output structurally valid under every input —
 * no `-Infinity`, no `NaN`, no zero-sample plans — so `resolvePlan`'s
 * substitution never produces a corrupt AST.
 *
 * `estimatedOracleCalls` is the number of oracle calls the plan would
 * make at `N = 1`; it bakes in the depth and `k*` chosen by the other
 * planner functions. The planner treats it as the caller's best
 * estimate; underestimates blow the budget at run time and raise
 * `BudgetError`, not here.
 */
export function computeSelfConsistencyN(
  inputs: SelfConsistencyInputs,
): number {
  const { budget, estimatedOracleCalls } = inputs;
  const cap = Math.max(1, Math.trunc(budget.selfConsistencyN));
  if (!Number.isFinite(estimatedOracleCalls) || estimatedOracleCalls <= 0) {
    return cap;
  }
  const fit = Math.floor(budget.maxOracleCalls / estimatedOracleCalls);
  return Math.max(1, Math.min(cap, fit));
}

// ===========================================================================
// estimateOracleCalls
// ===========================================================================

/**
 * Upper-bound the oracle calls a plan issues, given `k*` and depth.
 * Static plans normalize to "one oracle leaf per partition at every
 * level", so a balanced `k`-ary tree at depth `d` has `k^d` leaves.
 *
 * Returns the call count at `N = 1`; callers multiply by `N` to get the
 * full-plan estimate used by the budget check.
 *
 * Defensive clamps: non-finite or negative inputs return `1` instead of
 * throwing because this is a best-effort estimator, not a validator;
 * the runtime evaluator is the authoritative budget enforcer and will
 * throw `BudgetError` if the estimate underestimates reality.
 */
export function estimateOracleCalls(k: number, depth: number): number {
  if (k < 1 || depth < 0 || !Number.isFinite(k) || !Number.isFinite(depth)) {
    return 1;
  }
  return Math.max(1, Math.floor(Math.pow(k, depth)));
}

// ===========================================================================
// resolvePlan
// ===========================================================================

/**
 * Arguments to `resolvePlan`.
 *
 * - `plan` is the unresolved template — typically the `template` field of
 *   a `StaticPlan`, but any AST with `vref('k')` / `vref('n')` placeholders
 *   works. The planner does not mutate it.
 * - `planningInputs` drives the k/d/N computation: task type, prompt
 *   length, budget, and the optional `preferredK` override.
 * - `memorySchema` propagates to `ResolvedPlan` (from `StaticPlan` when routed).
 * - `leafThreshold` overrides `budget.leafThreshold` for depth only (optional).
 */
export interface ResolvePlanArgs {
  readonly plan: CombinatorNode;
  readonly planningInputs: PlanningInputs;
  readonly memorySchema?: MemorySchema | null;
  readonly leafThreshold?: number;
}

/**
 * End-to-end plan resolution.
 *
 * Algorithm:
 *
 * 1. Pick `k*`:
 *    - If `planningInputs.preferredK` is set, clamp it to
 *      `[K_SEARCH_MIN, budgetCeiling]`.
 *    - Otherwise run the quality-curve argmax via
 *      `computeOptimalPartitionSize`.
 * 2. Compute `depth = ⌈log_k*(n / τ*)⌉`, floored at 1.
 * 3. Normalize every `vote.n` in the plan to `1` and structurally
 *    estimate the realized oracle calls of that per-sample tree.
 * 4. Compute `N = computeSelfConsistencyN(budget, estimatedOracleCalls_N=1)`.
 * 5. Walk the AST and replace every `vref('k')` with `lit(partitionK)`
 *    and every `vref('n')` with `lit(selfConsistencyN)`. Every other
 *    node is returned unchanged. Nodes whose sub-trees contain no
 *    substitutions are returned by reference (structural sharing) so
 *    resolved plans produce zero allocation for the already-resolved
 *    common case.
 * 6. Return the resolved plan plus the chosen `partitionK`, `depth`,
 *    `selfConsistencyN`, the structurally-estimated `estimatedOracleCalls`,
 *    and `memorySchema`.
 *
 * Idempotency guarantee: `resolvePlan(resolvePlan(args).plan, args)`
 * returns a plan structurally equal to the first call. The walker is
 * pure and the substitution targets (`vref('k')` / `vref('n')`) no
 * longer appear in the resolved tree, so the second call is a no-op on
 * the AST while still producing the same `partitionK`, `depth`, and `N`.
 */
export function resolvePlan(args: ResolvePlanArgs): ResolvedPlan {
  const { planningInputs } = args;
  const leafThreshold =
    args.leafThreshold ?? planningInputs.budget.leafThreshold;
  const partitionK = pickPartitionK(planningInputs);
  const depth = computeRecursionDepth({
    promptLength: planningInputs.promptLength,
    leafThreshold,
    k: partitionK,
  });
  const estimatedPerSample = estimateOracleCallsForPlan(
    rewritePlan(args.plan, { partitionK, voteWidth: 1 }),
    planningInputs.promptLength,
  );
  const selfConsistencyN = computeSelfConsistencyN({
    budget: planningInputs.budget,
    estimatedOracleCalls: estimatedPerSample,
  });
  const rewritten = rewritePlan(args.plan, {
    partitionK,
    selfConsistencyN,
  });
  const estimatedOracleCalls = estimateOracleCallsForPlan(
    rewritePlan(rewritten, { voteWidth: selfConsistencyN }),
    planningInputs.promptLength,
  );
  return {
    plan: rewritten,
    partitionK,
    depth,
    selfConsistencyN,
    estimatedOracleCalls,
    memorySchema: args.memorySchema ?? null,
  };
}

// ===========================================================================
// Internal helpers
// ===========================================================================

/**
 * Budget-derived upper bound on `k`.
 *
 * One top-level partition per oracle call is the cheapest realistic
 * plan, so `k` cannot exceed either parallelism or the total call
 * budget. We still give the quality curve the full K_SEARCH_MAX range
 * when the budget is generous.
 */
function budgetPartitionCeiling(budget: RLMBudget): number {
  return Math.max(
    K_SEARCH_MIN,
    Math.min(
      K_SEARCH_MAX,
      Math.min(budget.maxParallelism, budget.maxOracleCalls),
    ),
  );
}

/**
 * Pick `k*` respecting `preferredK` if set, otherwise the quality-curve
 * argmax. Both paths clamp to `[K_SEARCH_MIN, budgetCeiling]`.
 */
function pickPartitionK(planningInputs: PlanningInputs): number {
  if (planningInputs.taskType === 'pairwise') {
    return 2;
  }
  if (planningInputs.preferredK !== undefined) {
    const budgetCeiling = budgetPartitionCeiling(planningInputs.budget);
    const preferred = Math.trunc(planningInputs.preferredK);
    return Math.max(K_SEARCH_MIN, Math.min(budgetCeiling, preferred));
  }
  return computeOptimalPartitionSize({
    promptLength: planningInputs.promptLength,
    budget: planningInputs.budget,
    taskType: planningInputs.taskType,
  });
}

/**
 * Options for {@link rewritePlan}. `partitionK` / `selfConsistencyN`
 * substitute `vref('k')` / `vref('n')` with literals; `voteWidth` forces
 * every `vote.n` to `lit(voteWidth)` regardless of its original shape.
 * Either path can be used alone or together.
 */
interface RewriteOptions {
  readonly partitionK?: number;
  readonly selfConsistencyN?: number;
  readonly voteWidth?: number;
}

type EstimateValue =
  | { readonly kind: 'unknown' }
  | { readonly kind: 'number'; readonly value: number }
  | { readonly kind: 'string'; readonly length: number }
  | {
      readonly kind: 'list';
      readonly length: number;
      readonly item: EstimateValue;
    };

interface EstimateResult {
  readonly calls: number;
  readonly value: EstimateValue;
}

type EstimateEnv = ReadonlyMap<string, EstimateValue>;

const UNKNOWN_ESTIMATE_VALUE: EstimateValue = Object.freeze({
  kind: 'unknown',
});

function estimateOracleCallsForPlan(
  plan: CombinatorNode,
  promptLength: number,
): number {
  const env = new Map<string, EstimateValue>([
    ['input', { kind: 'string', length: Math.max(0, promptLength) }],
  ]);
  return Math.max(1, estimateNode(plan, env).calls);
}

function estimateNode(
  node: CombinatorNode,
  env: EstimateEnv,
): EstimateResult {
  switch (node.tag) {
    case 'literal':
      return { calls: 0, value: estimateLiteral(node.value) };
    case 'var':
      return { calls: 0, value: env.get(node.name) ?? UNKNOWN_ESTIMATE_VALUE };
    case 'split': {
      const input = estimateNode(node.input, env);
      const k = estimateNode(node.k, env);
      const width = estimatePositiveInteger(k.value, 1);
      const inputLength = estimateStringLength(input.value);
      const count =
        inputLength === null
          ? width
          : inputLength <= 0
            ? 0
            : Math.min(width, inputLength);
      const itemLength =
        count <= 0 || inputLength === null
          ? 0
          : Math.ceil(inputLength / count);
      return {
        calls: input.calls + k.calls,
        value: {
          kind: 'list',
          length: count,
          item:
            inputLength === null
              ? UNKNOWN_ESTIMATE_VALUE
              : { kind: 'string', length: itemLength },
        },
      };
    }
    case 'peek': {
      const input = estimateNode(node.input, env);
      const start = estimateNode(node.start, env);
      const end = estimateNode(node.end, env);
      const inputLength = estimateStringLength(input.value);
      const startValue = estimatePositiveInteger(start.value, 0);
      const endValue = estimatePositiveInteger(end.value, startValue);
      return {
        calls: input.calls + start.calls + end.calls,
        value:
          inputLength === null
            ? UNKNOWN_ESTIMATE_VALUE
            : {
                kind: 'string',
                length: Math.max(
                  0,
                  Math.min(inputLength, endValue) -
                    Math.min(inputLength, startValue),
                ),
              },
      };
    }
    case 'map': {
      const items = estimateNode(node.items, env);
      const list = asListEstimate(items.value);
      const body = estimateNode(
        node.fn.body,
        bindEstimate(env, node.fn.param, list.item),
      );
      return {
        calls: items.calls + list.length * body.calls,
        value: { kind: 'list', length: list.length, item: body.value },
      };
    }
    case 'filter': {
      const items = estimateNode(node.items, env);
      const list = asListEstimate(items.value);
      const pred = estimateNode(
        node.pred.body,
        bindEstimate(env, node.pred.param, list.item),
      );
      return {
        calls: items.calls + list.length * pred.calls,
        value: { kind: 'list', length: list.length, item: list.item },
      };
    }
    case 'reduce': {
      const items = estimateNode(node.items, env);
      const list = asListEstimate(items.value);
      const stepCalls = estimateNode(
        node.op.body,
        bindEstimate(
          bindEstimate(env, node.op.left, UNKNOWN_ESTIMATE_VALUE),
          node.op.right,
          list.item,
        ),
      ).calls;
      const stepCount =
        list.length === 0
          ? 0
          : node.init === undefined
            ? Math.max(0, list.length - 1)
            : list.length;
      const initCalls =
        node.init === undefined ? 0 : estimateNode(node.init, env).calls;
      return {
        calls: items.calls + initCalls + stepCount * stepCalls,
        value: UNKNOWN_ESTIMATE_VALUE,
      };
    }
    case 'concat': {
      const items = estimateNode(node.items, env);
      const separator =
        node.separator === undefined
          ? { calls: 0, value: { kind: 'string', length: 0 } as EstimateValue }
          : estimateNode(node.separator, env);
      return {
        calls: items.calls + separator.calls,
        value: estimateConcatValue(items.value, separator.value),
      };
    }
    case 'cross': {
      const left = estimateNode(node.left, env);
      const right = estimateNode(node.right, env);
      const leftList = asListEstimate(left.value);
      const rightList = asListEstimate(right.value);
      return {
        calls: left.calls + right.calls,
        value: {
          kind: 'list',
          length: leftList.length * rightList.length,
          item: UNKNOWN_ESTIMATE_VALUE,
        },
      };
    }
    case 'vote': {
      if (node.oracle.tag !== 'oracle') {
        return {
          calls: estimateNode(node.oracle, env).calls,
          value: { kind: 'string', length: 0 },
        };
      }
      const prompt = estimateNode(node.oracle.prompt, env);
      const width = estimatePositiveInteger(estimateNode(node.n, env).value, 1);
      return {
        calls: prompt.calls + width,
        value: { kind: 'string', length: 0 },
      };
    }
    case 'ensemble': {
      if (node.oracle.tag !== 'oracle') {
        return {
          calls: estimateNode(node.oracle, env).calls,
          value: { kind: 'string', length: 0 },
        };
      }
      const prompt = estimateNode(node.oracle.prompt, env);
      return {
        calls: prompt.calls + Math.max(1, node.models.length),
        value: { kind: 'string', length: 0 },
      };
    }
    case 'oracle': {
      const prompt = estimateNode(node.prompt, env);
      return {
        calls: prompt.calls + 1,
        value: { kind: 'string', length: 0 },
      };
    }
  }
}

function estimateLiteral(value: CombinatorValue): EstimateValue {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return { kind: 'number', value };
  }
  if (typeof value === 'string') {
    return { kind: 'string', length: value.length };
  }
  if (Array.isArray(value)) {
    return {
      kind: 'list',
      length: value.length,
      item: UNKNOWN_ESTIMATE_VALUE,
    };
  }
  return UNKNOWN_ESTIMATE_VALUE;
}

function estimatePositiveInteger(
  value: EstimateValue,
  fallback: number,
): number {
  if (value.kind !== 'number' || !Number.isFinite(value.value)) {
    return fallback;
  }
  return Math.max(0, Math.trunc(value.value));
}

function estimateStringLength(value: EstimateValue): number | null {
  return value.kind === 'string' ? value.length : null;
}

function asListEstimate(
  value: EstimateValue,
): Extract<EstimateValue, { readonly kind: 'list' }> {
  if (value.kind === 'list') return value;
  return { kind: 'list', length: 1, item: UNKNOWN_ESTIMATE_VALUE };
}

function bindEstimate(
  env: EstimateEnv,
  name: string,
  value: EstimateValue,
): EstimateEnv {
  const next = new Map(env);
  next.set(name, value);
  return next;
}

function estimateConcatValue(
  items: EstimateValue,
  separator: EstimateValue,
): EstimateValue {
  if (items.kind !== 'list') {
    return { kind: 'string', length: 0 };
  }
  if (items.item.kind !== 'string' || separator.kind !== 'string') {
    return UNKNOWN_ESTIMATE_VALUE;
  }
  const separatorCount = Math.max(0, items.length - 1);
  return {
    kind: 'string',
    length:
      items.length * items.item.length + separatorCount * separator.length,
  };
}

/**
 * Single AST walker used for every plan transformation: `vref('k')` /
 * `vref('n')` substitution and `vote.n` width normalization. Structural
 * sharing: if no descendant changed, return the input node verbatim so
 * re-resolution is a true no-op on the tree.
 */
function rewritePlan(
  node: CombinatorNode,
  opts: RewriteOptions,
): CombinatorNode {
  switch (node.tag) {
    case 'literal':
      return node;
    case 'var': {
      if (node.name === 'k' && opts.partitionK !== undefined) {
        return lit(opts.partitionK);
      }
      if (node.name === 'n' && opts.selfConsistencyN !== undefined) {
        return lit(opts.selfConsistencyN);
      }
      return node;
    }
    case 'split': {
      const input = rewritePlan(node.input, opts);
      const k = rewritePlan(node.k, opts);
      return input === node.input && k === node.k
        ? node
        : { tag: 'split', input, k };
    }
    case 'peek': {
      const input = rewritePlan(node.input, opts);
      const start = rewritePlan(node.start, opts);
      const end = rewritePlan(node.end, opts);
      return input === node.input && start === node.start && end === node.end
        ? node
        : { tag: 'peek', input, start, end };
    }
    case 'map': {
      const body = rewritePlan(node.fn.body, opts);
      const items = rewritePlan(node.items, opts);
      return body === node.fn.body && items === node.items
        ? node
        : { tag: 'map', fn: { param: node.fn.param, body }, items };
    }
    case 'filter': {
      const body = rewritePlan(node.pred.body, opts);
      const items = rewritePlan(node.items, opts);
      return body === node.pred.body && items === node.items
        ? node
        : { tag: 'filter', pred: { param: node.pred.param, body }, items };
    }
    case 'reduce': {
      const body = rewritePlan(node.op.body, opts);
      const items = rewritePlan(node.items, opts);
      const init =
        node.init === undefined ? undefined : rewritePlan(node.init, opts);
      const opChanged = body !== node.op.body;
      const itemsChanged = items !== node.items;
      const initChanged = init !== node.init;
      if (!opChanged && !itemsChanged && !initChanged) return node;
      const reducer = opChanged
        ? { left: node.op.left, right: node.op.right, body }
        : node.op;
      return init === undefined
        ? { tag: 'reduce', op: reducer, items }
        : { tag: 'reduce', op: reducer, items, init };
    }
    case 'concat': {
      const items = rewritePlan(node.items, opts);
      const separator =
        node.separator === undefined
          ? undefined
          : rewritePlan(node.separator, opts);
      const itemsChanged = items !== node.items;
      const separatorChanged = separator !== node.separator;
      if (!itemsChanged && !separatorChanged) return node;
      return separator === undefined
        ? { tag: 'concat', items }
        : { tag: 'concat', items, separator };
    }
    case 'cross': {
      const left = rewritePlan(node.left, opts);
      const right = rewritePlan(node.right, opts);
      return left === node.left && right === node.right
        ? node
        : { tag: 'cross', left, right };
    }
    case 'vote': {
      const oracleNode = rewritePlan(node.oracle, opts);
      const n =
        opts.voteWidth !== undefined
          ? lit(opts.voteWidth)
          : rewritePlan(node.n, opts);
      const oracleChanged = oracleNode !== node.oracle;
      const nChanged = n !== node.n;
      if (!oracleChanged && !nChanged) return node;
      return node.reducer === undefined
        ? { tag: 'vote', oracle: oracleNode, n }
        : { tag: 'vote', oracle: oracleNode, n, reducer: node.reducer };
    }
    case 'ensemble': {
      const oracleNode = rewritePlan(node.oracle, opts);
      if (oracleNode === node.oracle) return node;
      return node.reducer === undefined
        ? { tag: 'ensemble', oracle: oracleNode, models: node.models }
        : {
            tag: 'ensemble',
            oracle: oracleNode,
            models: node.models,
            reducer: node.reducer,
          };
    }
    case 'oracle': {
      const prompt = rewritePlan(node.prompt, opts);
      if (prompt === node.prompt) return node;
      if (node.modelHint !== undefined && node.effectHandlers !== undefined) {
        return {
          tag: 'oracle',
          prompt,
          modelHint: node.modelHint,
          effectHandlers: node.effectHandlers,
        };
      }
      if (node.modelHint !== undefined) {
        return { tag: 'oracle', prompt, modelHint: node.modelHint };
      }
      if (node.effectHandlers !== undefined) {
        return { tag: 'oracle', prompt, effectHandlers: node.effectHandlers };
      }
      return { tag: 'oracle', prompt };
    }
  }
}
