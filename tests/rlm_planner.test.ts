/**
 * Planner: optimal `k`, bounds, idempotency, property checks
 * (`docs/product/rlm-v2-architecture.md` §4.2):
 *
 * 1. Per-task-type math: the planner picks the documented-optimal `k`
 *    from the seeded quality curve at a generous budget.
 * 2. Bounds: `k ≥ 2`, `depth ≥ 1`, `selfConsistencyN ≥ 1`.
 * 3. Idempotency: `resolvePlan(resolvePlan(plan, inputs).plan, inputs)`
 *    is structurally equal to the first call.
 * 4. Property test (fast-check): for any `n ∈ [100, 100_000]`,
 *    `τ* ∈ [50, 5000]`, `budget.maxOracleCalls ∈ [10, 1000]`, the
 *    planner terminates and returns a resolvable plan.
 */

import * as fc from 'fast-check';
import { describe, expect, it } from 'vitest';

import {
  concat,
  ensemble,
  filter,
  fn,
  lit,
  map,
  oracle,
  reduce,
  split,
  vote,
  vref,
  type CombinatorNode,
} from '../src/rlm_combinators.js';
import {
  K_SEARCH_MAX,
  K_SEARCH_MIN,
  computeOptimalPartitionSize,
  computeRecursionDepth,
  computeSelfConsistencyN,
  estimateOracleCalls,
  resolvePlan,
} from '../src/rlm_planner.js';
import {
  DOCUMENTED_OPTIMAL_K,
  QUALITY_CURVES,
} from '../src/rlm_planner_quality.js';
import { DEFAULT_BUDGET, type RLMBudget } from '../src/rlm_types.js';
import type {
  PlanningInputs,
  TaskType,
} from '../src/rlm_task_router.js';
import { STATIC_PLANS } from '../src/rlm_task_router.js';
import type { MemorySchema } from '../src/rlm_memory.js';

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/**
 * A budget that is intentionally larger than K_SEARCH_MAX in every
 * dimension, so the planner's budget ceiling never truncates the
 * quality-curve argmax below the documented optimum. This is the
 * regime in which the "documented optimum" test claim holds.
 */
function generousBudget(overrides: Partial<RLMBudget> = {}): RLMBudget {
  return {
    maxOracleCalls: 1000,
    maxParallelism: 32,
    maxDepth: 8,
    leafThreshold: 1000,
    selfConsistencyN: 5,
    maxEffectTurns: 8,
    ...overrides,
  };
}

const ALL_TASK_TYPES: readonly TaskType[] = [
  'search',
  'classify',
  'aggregate',
  'pairwise',
  'summarise',
  'multi_hop',
  'unknown',
];

/**
 * A plan template that contains every substitutable surface plus a
 * handful of untouched `vref` bindings so the idempotency check covers
 * every branch of the AST walker.
 */
function sampleTemplate(): CombinatorNode {
  return concat(
    map(
      fn('chunk', oracle(vref('chunk'), 'gpt-fast')),
      split(vref('input'), vref('k')),
    ),
    lit(' | '),
  );
}

// ---------------------------------------------------------------------------
// 1. Per-task-type math
// ---------------------------------------------------------------------------

describe('computeOptimalPartitionSize — documented argmax per task type', () => {
  for (const taskType of ALL_TASK_TYPES) {
    it(`${taskType}: picks DOCUMENTED_OPTIMAL_K under a generous budget`, () => {
      const k = computeOptimalPartitionSize({
        promptLength: 10_000,
        budget: generousBudget(),
        taskType,
      });
      expect(k).toBe(DOCUMENTED_OPTIMAL_K[taskType]);
    });
  }

  it('seed curves are uni-modal with a strict argmax on the integer grid', () => {
    // Defensive: confirm the planner's linear scan is picking the sole
    // integer argmax per curve, not accidentally finding a secondary peak.
    for (const taskType of ALL_TASK_TYPES) {
      const curve = QUALITY_CURVES[taskType];
      let bestK = K_SEARCH_MIN;
      let bestQ = -Infinity;
      for (let k = K_SEARCH_MIN; k <= K_SEARCH_MAX; k += 1) {
        const q = curve(k, 10_000);
        if (q > bestQ) {
          bestQ = q;
          bestK = k;
        }
      }
      expect(bestK).toBe(DOCUMENTED_OPTIMAL_K[taskType]);
    }
  });

  it('clamps to budget.maxParallelism when parallelism is the binding constraint', () => {
    const k = computeOptimalPartitionSize({
      promptLength: 10_000,
      budget: generousBudget({ maxParallelism: 3 }),
      taskType: 'search',
    });
    expect(k).toBeLessThanOrEqual(3);
    expect(k).toBeGreaterThanOrEqual(K_SEARCH_MIN);
  });

  it('clamps to budget.maxOracleCalls when the call budget is the binding constraint', () => {
    const k = computeOptimalPartitionSize({
      promptLength: 10_000,
      budget: generousBudget({ maxOracleCalls: 4 }),
      taskType: 'search',
    });
    expect(k).toBeLessThanOrEqual(4);
    expect(k).toBeGreaterThanOrEqual(K_SEARCH_MIN);
  });

  it('returns at least K_SEARCH_MIN even when the budget forbids any fan-out', () => {
    // pairwise tasks are the canonical case: k = 2 regardless of budget.
    const k = computeOptimalPartitionSize({
      promptLength: 10_000,
      budget: generousBudget({ maxParallelism: 1, maxOracleCalls: 1 }),
      taskType: 'pairwise',
    });
    expect(k).toBe(K_SEARCH_MIN);
  });
});

// ---------------------------------------------------------------------------
// 2. Bounds
// ---------------------------------------------------------------------------

describe('computeRecursionDepth — bounds and formula', () => {
  it('returns 1 when promptLength ≤ leafThreshold', () => {
    expect(
      computeRecursionDepth({ promptLength: 500, leafThreshold: 1000, k: 4 }),
    ).toBe(1);
    expect(
      computeRecursionDepth({ promptLength: 1000, leafThreshold: 1000, k: 4 }),
    ).toBe(1);
  });

  it('computes ⌈log_k(n / τ*)⌉ when above threshold', () => {
    // n / τ* = 16, k = 2 → log₂(16) = 4.
    expect(
      computeRecursionDepth({ promptLength: 16_000, leafThreshold: 1000, k: 2 }),
    ).toBe(4);
    // n / τ* = 16, k = 4 → log₄(16) = 2.
    expect(
      computeRecursionDepth({ promptLength: 16_000, leafThreshold: 1000, k: 4 }),
    ).toBe(2);
    // n / τ* = 100, k = 4 → log₄(100) ≈ 3.32 → 4.
    expect(
      computeRecursionDepth({
        promptLength: 100_000,
        leafThreshold: 1000,
        k: 4,
      }),
    ).toBe(4);
  });

  it('floors at 1 even when ratio rounds to zero', () => {
    expect(
      computeRecursionDepth({ promptLength: 0, leafThreshold: 1000, k: 4 }),
    ).toBe(1);
  });

  it('throws on invalid k', () => {
    expect(() =>
      computeRecursionDepth({ promptLength: 1000, leafThreshold: 100, k: 1 }),
    ).toThrow(RangeError);
    expect(() =>
      computeRecursionDepth({ promptLength: 1000, leafThreshold: 100, k: 0 }),
    ).toThrow(RangeError);
    expect(() =>
      computeRecursionDepth({ promptLength: 1000, leafThreshold: 100, k: 2.5 }),
    ).toThrow(RangeError);
  });

  it('throws on invalid leafThreshold', () => {
    expect(() =>
      computeRecursionDepth({ promptLength: 1000, leafThreshold: 0, k: 4 }),
    ).toThrow(RangeError);
    expect(() =>
      computeRecursionDepth({
        promptLength: 1000,
        leafThreshold: Number.POSITIVE_INFINITY,
        k: 4,
      }),
    ).toThrow(RangeError);
  });

  it('throws on invalid promptLength', () => {
    expect(() =>
      computeRecursionDepth({
        promptLength: Number.NaN,
        leafThreshold: 100,
        k: 4,
      }),
    ).toThrow(RangeError);
    expect(() =>
      computeRecursionDepth({ promptLength: -1, leafThreshold: 100, k: 4 }),
    ).toThrow(RangeError);
  });
});

describe('computeSelfConsistencyN — bounds and budget fit', () => {
  it('returns cap when estimatedOracleCalls is zero or invalid', () => {
    expect(
      computeSelfConsistencyN({
        budget: generousBudget(),
        estimatedOracleCalls: 0,
      }),
    ).toBe(generousBudget().selfConsistencyN);
    expect(
      computeSelfConsistencyN({
        budget: generousBudget(),
        estimatedOracleCalls: Number.NaN,
      }),
    ).toBe(generousBudget().selfConsistencyN);
  });

  it('caps at budget.selfConsistencyN when the call budget is loose', () => {
    expect(
      computeSelfConsistencyN({
        budget: generousBudget({ selfConsistencyN: 7, maxOracleCalls: 10_000 }),
        estimatedOracleCalls: 4,
      }),
    ).toBe(7);
  });

  it('reduces N to fit within budget.maxOracleCalls', () => {
    // 10 calls per sample × N ≤ 25 call budget → N = 2.
    expect(
      computeSelfConsistencyN({
        budget: generousBudget({ selfConsistencyN: 5, maxOracleCalls: 25 }),
        estimatedOracleCalls: 10,
      }),
    ).toBe(2);
  });

  it('floors at 1 even when a single sample would exceed the budget', () => {
    // 100 calls > 50-call budget → still returns N = 1. Runtime
    // enforcement throws BudgetError on the first leaf hit.
    expect(
      computeSelfConsistencyN({
        budget: generousBudget({ selfConsistencyN: 5, maxOracleCalls: 50 }),
        estimatedOracleCalls: 100,
      }),
    ).toBe(1);
  });

  it('floors at 1 when budget.selfConsistencyN is set below 1', () => {
    expect(
      computeSelfConsistencyN({
        budget: generousBudget({ selfConsistencyN: 0 }),
        estimatedOracleCalls: 4,
      }),
    ).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// 3. resolvePlan — end-to-end, structure, idempotency
// ---------------------------------------------------------------------------

describe('resolvePlan — end-to-end k/d/N plus AST substitution', () => {
  it('substitutes vref("k") → partitionK and vref("n") → selfConsistencyN', () => {
    // Plan template in the same shape as `STATIC_PLANS`:
    //   vote( oracle(input), n )        where `vote.n` is the self-consistency width,
    //   split( input, k )               where `split.k` is the partition count.
    const plan = vote(oracle(vref('input')), vref('n'), 'majority');
    const resolved = resolvePlan({
      plan,
      planningInputs: {
        taskType: 'classify',
        promptLength: 500,
        budget: generousBudget(),
      },
    });
    expect(resolved.plan).toMatchObject({
      tag: 'vote',
      n: { tag: 'literal', value: resolved.selfConsistencyN },
    });
    // And the partitionK substitution wires into a `split.k` node.
    const splitPlan = split(vref('input'), vref('k'));
    const splitResolved = resolvePlan({
      plan: splitPlan,
      planningInputs: {
        taskType: 'search',
        promptLength: 123,
        budget: generousBudget(),
      },
    });
    expect(splitResolved.plan).toEqual({
      tag: 'split',
      input: { tag: 'var', name: 'input' },
      k: { tag: 'literal', value: DOCUMENTED_OPTIMAL_K.search },
    });
  });

  it('picks DOCUMENTED_OPTIMAL_K for the planning inputs task type', () => {
    const resolved = resolvePlan({
      plan: split(vref('input'), vref('k')),
      planningInputs: {
        taskType: 'classify',
        promptLength: 500,
        budget: generousBudget(),
      },
    });
    expect(resolved.partitionK).toBe(DOCUMENTED_OPTIMAL_K.classify);
  });

  it('leaves other vref names untouched', () => {
    const plan = map(
      fn('chunk', vref('chunk')),
      split(vref('input'), vref('k')),
    );
    const resolved = resolvePlan({
      plan,
      planningInputs: {
        taskType: 'search',
        promptLength: 1000,
        budget: generousBudget(),
      },
    });
    expect(resolved.plan).toEqual({
      tag: 'map',
      fn: { param: 'chunk', body: { tag: 'var', name: 'chunk' } },
      items: {
        tag: 'split',
        input: { tag: 'var', name: 'input' },
        k: { tag: 'literal', value: DOCUMENTED_OPTIMAL_K.search },
      },
    });
  });

  it('recurses into every node variant without losing fields', () => {
    const plan: CombinatorNode = {
      tag: 'reduce',
      op: {
        left: 'acc',
        right: 'x',
        body: concat(
          map(
            fn('c', oracle(vref('c'), 'gpt-fast', ['ReadContext'])),
            vref('items'),
          ),
          lit('|'),
        ),
      },
      items: filter(
        fn('c', vref('c')),
        vote(oracle(vref('input'), 'gpt-fast'), vref('n'), 'majority'),
      ),
      init: ensemble(oracle(vref('k')), ['gpt-fast', 'gpt-deep'], 'confidence'),
    };
    const resolved = resolvePlan({
      plan,
      planningInputs: {
        taskType: 'search',
        promptLength: 9000,
        budget: generousBudget(),
      },
    });
    const json = JSON.stringify(resolved.plan);
    // k/n placeholders are gone.
    expect(json).not.toContain('"name":"k"');
    expect(json).not.toContain('"name":"n"');
    // Caller's own bindings are preserved.
    expect(json).toContain('"name":"items"');
    expect(json).toContain('"name":"c"');
    expect(json).toContain('"name":"input"');
    // Model hints survive the walk.
    expect(json).toContain('"modelHint":"gpt-fast"');
    // Effect handlers survive the walk.
    expect(json).toContain('ReadContext');
  });

  it('resolvePlan is idempotent (structural equality on second application)', () => {
    const plan = sampleTemplate();
    const planningInputs: PlanningInputs = {
      taskType: 'search',
      promptLength: 2000,
      budget: generousBudget(),
    };
    const once = resolvePlan({ plan, planningInputs });
    const twice = resolvePlan({ plan: once.plan, planningInputs });
    expect(twice.plan).toEqual(once.plan);
    expect(twice.partitionK).toBe(once.partitionK);
    expect(twice.depth).toBe(once.depth);
    expect(twice.selfConsistencyN).toBe(once.selfConsistencyN);
    expect(twice.estimatedOracleCalls).toBe(once.estimatedOracleCalls);
  });

  it('preserves node identity when no substitution applies (no-op reference equality)', () => {
    const literalOnly = concat(
      map(fn('x', oracle(vref('x'))), split(lit('abcdef'), lit(3))),
      lit('|'),
    );
    const resolved = resolvePlan({
      plan: literalOnly,
      planningInputs: {
        taskType: 'search',
        promptLength: 6,
        budget: generousBudget(),
      },
    });
    expect(resolved.plan).toBe(literalOnly);
  });

  it('propagates memorySchema from args to ResolvedPlan', () => {
    const schema: MemorySchema = {
      name: 'test_memory',
      fields: [],
      maxBytesSerialized: 256,
    };
    const resolved = resolvePlan({
      plan: oracle(vref('n')),
      planningInputs: {
        taskType: 'search',
        promptLength: 50,
        budget: generousBudget(),
      },
      memorySchema: schema,
    });
    expect(resolved.memorySchema).toBe(schema);
  });

  it('defaults memorySchema to null when absent', () => {
    const resolved = resolvePlan({
      plan: oracle(vref('n')),
      planningInputs: {
        taskType: 'search',
        promptLength: 50,
        budget: generousBudget(),
      },
    });
    expect(resolved.memorySchema).toBeNull();
  });

  it('computes estimatedOracleCalls from the realized static-plan structure', () => {
    const cases: ReadonlyArray<readonly [TaskType, number]> = [
      ['search', 8],
      ['classify', 5],
      ['aggregate', 5],
      ['pairwise', 1],
      ['summarise', 7],
      ['multi_hop', 10],
    ];
    for (const [taskType, expectedCalls] of cases) {
      const staticPlan = STATIC_PLANS.get(taskType);
      expect(staticPlan).toBeDefined();
      const resolved = resolvePlan({
        plan: staticPlan!.template,
        planningInputs: {
          taskType,
          promptLength: 4000,
          budget: generousBudget(),
        },
        memorySchema: staticPlan!.memorySchema,
      });
      expect(resolved.estimatedOracleCalls).toBe(expectedCalls);
    }
  });

  it('respects preferredK override', () => {
    const resolved = resolvePlan({
      plan: split(vref('input'), vref('k')),
      planningInputs: {
        taskType: 'search',
        promptLength: 4000,
        budget: generousBudget(),
        preferredK: 3,
      },
    });
    expect(resolved.partitionK).toBe(3);
  });

  it('clamps preferredK to the search bounds and budget ceiling', () => {
    const over = resolvePlan({
      plan: split(vref('input'), vref('k')),
      planningInputs: {
        taskType: 'search',
        promptLength: 4000,
        budget: generousBudget({ maxParallelism: 4 }),
        preferredK: 99,
      },
    });
    expect(over.partitionK).toBe(4);

    const under = resolvePlan({
      plan: split(vref('input'), vref('k')),
      planningInputs: {
        taskType: 'search',
        promptLength: 4000,
        budget: generousBudget(),
        preferredK: 1,
      },
    });
    expect(under.partitionK).toBe(K_SEARCH_MIN);
  });

  it('ignores preferredK overrides for pairwise tasks and keeps k=2', () => {
    const resolved = resolvePlan({
      plan: split(vref('input'), vref('k')),
      planningInputs: {
        taskType: 'pairwise',
        promptLength: 4000,
        budget: generousBudget(),
        preferredK: 9,
      },
    });
    expect(resolved.partitionK).toBe(2);
  });

  it('overrides leafThreshold for depth computation when supplied', () => {
    const resolved = resolvePlan({
      plan: split(vref('input'), vref('k')),
      planningInputs: {
        taskType: 'search',
        promptLength: 4000,
        budget: generousBudget({ leafThreshold: 4000 }),
      },
      leafThreshold: 200,
    });
    // With τ*=200 and n=4000, log_8(20) ≈ 1.44 → depth=2.
    expect(resolved.depth).toBe(2);
  });

  it('substitutes inside reduce.init', () => {
    const plan = reduce(
      { left: 'acc', right: 'x', body: vref('acc') },
      vref('items'),
      vref('k'),
    );
    const resolved = resolvePlan({
      plan,
      planningInputs: {
        taskType: 'search',
        promptLength: 1000,
        budget: generousBudget(),
      },
    });
    expect(resolved.plan).toMatchObject({
      tag: 'reduce',
      init: { tag: 'literal', value: DOCUMENTED_OPTIMAL_K.search },
    });
  });

  it('substitutes inside concat.separator when separator is vref(k/n)', () => {
    const plan = concat(vref('items'), vref('n'));
    const resolved = resolvePlan({
      plan,
      planningInputs: {
        taskType: 'search',
        promptLength: 99,
        budget: generousBudget(),
      },
    });
    expect(resolved.plan).toEqual({
      tag: 'concat',
      items: { tag: 'var', name: 'items' },
      separator: { tag: 'literal', value: resolved.selfConsistencyN },
    });
  });
});

// ---------------------------------------------------------------------------
// 4. Property tests (fast-check)
// ---------------------------------------------------------------------------

describe('planner property tests', () => {
  it('terminates and returns a bounded resolved plan for every input', () => {
    const template = map(
      fn('chunk', oracle(vref('chunk'))),
      split(vref('input'), vref('k')),
    );
    fc.assert(
      fc.property(
        fc.integer({ min: 100, max: 100_000 }),
        fc.integer({ min: 50, max: 5000 }),
        fc.integer({ min: 10, max: 1000 }),
        fc.constantFrom(...ALL_TASK_TYPES),
        (promptLength, leafThreshold, maxOracleCalls, taskType) => {
          const budget: RLMBudget = {
            ...DEFAULT_BUDGET,
            leafThreshold,
            maxOracleCalls,
            maxParallelism: Math.min(
              DEFAULT_BUDGET.maxParallelism,
              maxOracleCalls,
            ),
          };
          const planningInputs: PlanningInputs = {
            taskType,
            promptLength,
            budget,
          };
          const resolved = resolvePlan({ plan: template, planningInputs });
          // Hard invariants (planner bounds).
          expect(resolved.partitionK).toBeGreaterThanOrEqual(K_SEARCH_MIN);
          expect(resolved.partitionK).toBeLessThanOrEqual(K_SEARCH_MAX);
          expect(resolved.depth).toBeGreaterThanOrEqual(1);
          expect(resolved.selfConsistencyN).toBeGreaterThanOrEqual(1);
          expect(resolved.selfConsistencyN).toBeLessThanOrEqual(
            budget.selfConsistencyN,
          );
          expect(resolved.estimatedOracleCalls).toBeGreaterThanOrEqual(1);
          // Structural: no vref('k') or vref('n') remains.
          const json = JSON.stringify(resolved.plan);
          expect(json).not.toContain('"name":"k"');
          expect(json).not.toContain('"name":"n"');
        },
      ),
      { numRuns: 100 },
    );
  });

  it('resolvePlan is idempotent over arbitrary inputs', () => {
    const template = sampleTemplate();
    fc.assert(
      fc.property(
        fc.constantFrom(...ALL_TASK_TYPES),
        fc.integer({ min: 100, max: 100_000 }),
        fc.integer({ min: 50, max: 5000 }),
        fc.integer({ min: 10, max: 1000 }),
        (taskType, promptLength, leafThreshold, maxOracleCalls) => {
          const budget: RLMBudget = {
            ...DEFAULT_BUDGET,
            leafThreshold,
            maxOracleCalls,
            maxParallelism: Math.min(
              DEFAULT_BUDGET.maxParallelism,
              maxOracleCalls,
            ),
          };
          const planningInputs: PlanningInputs = {
            taskType,
            promptLength,
            budget,
          };
          const once = resolvePlan({ plan: template, planningInputs });
          const twice = resolvePlan({
            plan: once.plan,
            planningInputs,
          });
          expect(twice.plan).toEqual(once.plan);
          expect(twice.partitionK).toBe(once.partitionK);
          expect(twice.depth).toBe(once.depth);
          expect(twice.selfConsistencyN).toBe(once.selfConsistencyN);
          expect(twice.estimatedOracleCalls).toBe(once.estimatedOracleCalls);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('budget constraint always dominates self-consistency fit', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 20 }),
        fc.integer({ min: 10, max: 2000 }),
        fc.integer({ min: 1, max: 500 }),
        (selfConsistencyN, maxOracleCalls, estimatedOracleCalls) => {
          const budget: RLMBudget = {
            ...DEFAULT_BUDGET,
            selfConsistencyN,
            maxOracleCalls,
          };
          const n = computeSelfConsistencyN({
            budget,
            estimatedOracleCalls,
          });
          expect(n).toBeGreaterThanOrEqual(1);
          expect(n).toBeLessThanOrEqual(Math.max(1, selfConsistencyN));
          if (estimatedOracleCalls > 0 && n > 1) {
            expect(n * estimatedOracleCalls).toBeLessThanOrEqual(maxOracleCalls);
          }
        },
      ),
      { numRuns: 200 },
    );
  });

  it('estimateOracleCalls returns at least 1 for every non-degenerate input', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 16 }),
        fc.integer({ min: 0, max: 6 }),
        (k, depth) => {
          const calls = estimateOracleCalls(k, depth);
          expect(calls).toBeGreaterThanOrEqual(1);
          expect(Number.isFinite(calls)).toBe(true);
        },
      ),
      { numRuns: 50 },
    );
  });
});
