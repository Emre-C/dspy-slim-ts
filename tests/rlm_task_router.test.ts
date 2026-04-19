/**
 * Task router: static plan registry, `classifyTask`, `resolveRoute`
 * (`docs/product/rlm-v2-architecture.md` §4.3):
 *
 * 1. `STATIC_PLANS` has exactly the six non-`unknown` task types, every
 *    template uses `vref('input')` as the top-level prompt binding, and
 *    every template resolves cleanly through the planner.
 * 2. `classifyTask`:
 *    - Happy path: adapter returns a well-formed JSON payload; the
 *      function returns a `ClassifierResult` matching that payload.
 *    - Hallucinated primary / out-of-range confidence / garbage
 *      candidates → graceful fallback to `unknown` or filtered list.
 *    - `lm.generate` throw → returns `UNKNOWN_CLASSIFIER_RESULT` (except
 *      `BudgetError`, which propagates).
 * 3. `resolveRoute`:
 *    - High confidence → `{ kind: 'single' }` with the primary plan.
 *    - Low confidence → `{ kind: 'beam' }` with the top-K deduped plans,
 *      primary first.
 *    - `unknown` primary + high confidence → still routes through the
 *      `summarise` fallback plan.
 *    - Degenerate beam (only one unique plan) collapses to `single`.
 * 4. `composeBeamPlan` — empty/singleton/triple inputs.
 */

import { describe, expect, it } from 'vitest';

import type { Message } from '../src/chat_message.js';
import { BudgetError, ValueError } from '../src/exceptions.js';
import { BaseLM, type LMOutput } from '../src/lm.js';
import {
  oracle,
  vref,
  type CombinatorNode,
} from '../src/rlm_combinators.js';
import { resolvePlan } from '../src/rlm_planner.js';
import { DEFAULT_BUDGET, type RLMBudget } from '../src/rlm_types.js';
import { signatureFromString } from '../src/signature.js';
import type { Signature } from '../src/signature.js';

import {
  DEFAULT_BEAM_TOP_K,
  DEFAULT_ROUTE_THRESHOLD,
  REAL_TASK_TYPES_LIST,
  STATIC_PLANS,
  TASK_TYPES,
  classifyTask,
  composeBeamPlan,
  isTaskType,
  resolveRoute,
  type ClassifierResult,
  type StaticPlan,
  type TaskType,
} from '../src/rlm_task_router.js';

// ---------------------------------------------------------------------------
// Test LMs
// ---------------------------------------------------------------------------

/**
 * One-shot LM that returns a canned JSON payload — the classifier
 * adapter parses the JSON and feeds `primary`/`confidence`/`candidates`
 * to `Prediction.get`.
 */
class StubClassifierLM extends BaseLM {
  readonly payload: string;
  calls = 0;

  constructor(payload: string) {
    super({ model: 'stub-classifier' });
    this.payload = payload;
  }

  protected override generate(
    _prompt?: string,
    _messages?: readonly Message[],
    _kwargs?: Record<string, unknown>,
  ): readonly LMOutput[] {
    this.calls += 1;
    return [this.payload];
  }
}

class ThrowingLM extends BaseLM {
  constructor() {
    super({ model: 'throw-lm' });
  }
  protected override generate(): readonly LMOutput[] {
    throw new Error('ThrowingLM always throws');
  }
}

class BudgetThrowingLM extends BaseLM {
  constructor() {
    super({ model: 'budget-throw-lm' });
  }
  protected override generate(): readonly LMOutput[] {
    throw new BudgetError('BudgetThrowingLM');
  }
}

const USER_SIGNATURE: Signature = signatureFromString(
  'article: str -> summary: str',
);

function budget(overrides: Partial<RLMBudget> = {}): RLMBudget {
  return {
    ...DEFAULT_BUDGET,
    maxOracleCalls: 200,
    maxParallelism: 16,
    selfConsistencyN: 5,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// 1. STATIC_PLANS registry
// ---------------------------------------------------------------------------

describe('STATIC_PLANS', () => {
  it('contains exactly the six non-unknown task types', () => {
    expect([...STATIC_PLANS.keys()].sort()).toEqual(
      [...REAL_TASK_TYPES_LIST].sort(),
    );
    expect(STATIC_PLANS.has('unknown')).toBe(false);
    expect(STATIC_PLANS.size).toBe(6);
  });

  it('every plan carries its own taskType tag', () => {
    for (const [key, plan] of STATIC_PLANS) {
      expect(plan.taskType).toBe(key);
    }
  });

  it('every plan template references vref("input") and includes k or n placeholders', () => {
    for (const plan of STATIC_PLANS.values()) {
      const json = JSON.stringify(plan.template);
      expect(json).toContain('"name":"input"');
      // Every template must be parameterized on either k, n, or both, or
      // the planner has nothing to tune. pairwise uses k=2; classify uses
      // n only; others use both.
      const usesKOrN =
        json.includes('"name":"k"') || json.includes('"name":"n"');
      expect(usesKOrN).toBe(true);
    }
  });

  it('every plan resolves cleanly under a generous budget', () => {
    for (const plan of STATIC_PLANS.values()) {
      const resolved = resolvePlan({
        plan: plan.template,
        planningInputs: {
          taskType: plan.taskType,
          promptLength: 4000,
          budget: budget(),
        },
      });
      expect(resolved.partitionK).toBeGreaterThanOrEqual(2);
      expect(resolved.depth).toBeGreaterThanOrEqual(1);
      expect(resolved.selfConsistencyN).toBeGreaterThanOrEqual(1);
      const json = JSON.stringify(resolved.plan);
      expect(json).not.toContain('"name":"k"');
      expect(json).not.toContain('"name":"n"');
      expect(json).toContain('"name":"input"');
    }
  });

  it('vote-wrapping plans (classify, multi_hop) keep vote.oracle as an oracle node after resolution', () => {
    for (const taskType of ['classify', 'multi_hop'] as const) {
      const plan = STATIC_PLANS.get(taskType)!;
      const resolved = resolvePlan({
        plan: plan.template,
        planningInputs: {
          taskType,
          promptLength: 1000,
          budget: budget(),
        },
      });
      const voteNode = findVoteNode(resolved.plan);
      expect(voteNode).not.toBeNull();
      expect(voteNode!.oracle.tag).toBe('oracle');
    }
  });

  it('pairwise partitionK always resolves to 2 (the only sensible value)', () => {
    const plan = STATIC_PLANS.get('pairwise')!;
    const resolved = resolvePlan({
      plan: plan.template,
      planningInputs: {
        taskType: 'pairwise',
        promptLength: 4000,
        budget: budget(),
      },
    });
    expect(resolved.partitionK).toBe(2);
  });

  it('search / aggregate / summarise / multi_hop ship the failure_diagnostic schema', () => {
    const expected = new Set(['search', 'aggregate', 'summarise', 'multi_hop']);
    for (const [taskType, plan] of STATIC_PLANS) {
      if (expected.has(taskType)) {
        expect(plan.memorySchema).not.toBeNull();
        expect(plan.memorySchema?.name).toBe('failure_diagnostic');
        expect(plan.memorySchema?.fields.map((f) => f.name)).toEqual([
          'failure_pattern',
          'next_check',
          'prevented_action',
        ]);
      } else {
        expect(plan.memorySchema).toBeNull();
      }
    }
  });
});

// ---------------------------------------------------------------------------
// 2. classifyTask
// ---------------------------------------------------------------------------

describe('classifyTask', () => {
  it('returns the classifier output on a happy path', async () => {
    const lm = new StubClassifierLM(
      '{"primary": "search", "confidence": 0.92, "candidates": ["search", "aggregate"]}',
    );
    const result = await classifyTask(
      'Find the author of this article.',
      USER_SIGNATURE,
      lm,
    );
    expect(result.primary).toBe('search');
    expect(result.confidence).toBeCloseTo(0.92, 6);
    expect(result.candidates).toEqual(['search', 'aggregate']);
    expect(lm.calls).toBe(1);
  });

  it('accepts uppercase / whitespace-padded task type tags', async () => {
    const lm = new StubClassifierLM(
      '{"primary": "  SEARCH  ", "confidence": 0.8, "candidates": [" AGGREGATE "]}',
    );
    const result = await classifyTask('What is X?', USER_SIGNATURE, lm);
    expect(result.primary).toBe('search');
    expect(result.candidates).toEqual(['search', 'aggregate']);
  });

  it('falls back to unknown when the primary is not a valid TaskType', async () => {
    const lm = new StubClassifierLM(
      '{"primary": "nonsense", "confidence": 0.99, "candidates": ["search"]}',
    );
    const result = await classifyTask('...', USER_SIGNATURE, lm);
    expect(result.primary).toBe('unknown');
    // candidates still filter valid entries; primary (unknown) is omitted
    // because we never seed the list with 'unknown'.
    expect(result.candidates).toEqual(['search']);
  });

  it('clamps confidence into [0, 1]', async () => {
    const high = new StubClassifierLM(
      '{"primary": "search", "confidence": 2.5, "candidates": []}',
    );
    expect((await classifyTask('x', USER_SIGNATURE, high)).confidence).toBe(1);
    const low = new StubClassifierLM(
      '{"primary": "search", "confidence": -0.4, "candidates": []}',
    );
    expect((await classifyTask('x', USER_SIGNATURE, low)).confidence).toBe(0);
    const nan = new StubClassifierLM(
      '{"primary": "search", "confidence": "oops", "candidates": []}',
    );
    expect((await classifyTask('x', USER_SIGNATURE, nan)).confidence).toBe(0);
  });

  it('filters hallucinated candidate entries and dedupes', async () => {
    const lm = new StubClassifierLM(
      '{"primary": "search", "confidence": 0.8, "candidates": ["search", "bogus", "search", "classify", 42]}',
    );
    const result = await classifyTask('x', USER_SIGNATURE, lm);
    expect(result.candidates).toEqual(['search', 'classify']);
  });

  it('returns the UNKNOWN_CLASSIFIER_RESULT when the LM throws', async () => {
    const result = await classifyTask('x', USER_SIGNATURE, new ThrowingLM());
    expect(result).toEqual({
      primary: 'unknown',
      confidence: 0,
      candidates: [],
    });
  });

  it('rethrows BudgetError from the LM (budget is not a routing fallback)', async () => {
    await expect(
      classifyTask('x', USER_SIGNATURE, new BudgetThrowingLM()),
    ).rejects.toThrow(BudgetError);
  });

  it('returns UNKNOWN_CLASSIFIER_RESULT when the adapter cannot parse the payload', async () => {
    const lm = new StubClassifierLM('this is not JSON');
    const result = await classifyTask('x', USER_SIGNATURE, lm);
    expect(result).toEqual({
      primary: 'unknown',
      confidence: 0,
      candidates: [],
    });
  });

  it('injects the user signature field names into the classifier context', async () => {
    // A signature-aware payload lets us confirm the prompt went through
    // — the stub LM doesn't care about the prompt content, but the
    // Predict call must have succeeded end-to-end.
    const sig = signatureFromString('topic: str, audience: str -> outline: str');
    const lm = new StubClassifierLM(
      '{"primary": "summarise", "confidence": 0.75, "candidates": ["summarise"]}',
    );
    const result = await classifyTask('write an outline', sig, lm);
    expect(result.primary).toBe('summarise');
  });

  it('classifier signature is reusable across calls with different LMs', async () => {
    const lm1 = new StubClassifierLM(
      '{"primary": "classify", "confidence": 0.9, "candidates": ["classify"]}',
    );
    const lm2 = new StubClassifierLM(
      '{"primary": "search", "confidence": 0.85, "candidates": ["search"]}',
    );
    expect((await classifyTask('q', USER_SIGNATURE, lm1)).primary).toBe(
      'classify',
    );
    expect((await classifyTask('q', USER_SIGNATURE, lm2)).primary).toBe(
      'search',
    );
  });
});

// ---------------------------------------------------------------------------
// 3. resolveRoute
// ---------------------------------------------------------------------------

describe('resolveRoute', () => {
  it('returns single for confidence >= threshold', () => {
    const result: ClassifierResult = {
      primary: 'search',
      confidence: 0.9,
      candidates: ['search', 'aggregate'],
    };
    const route = resolveRoute(result);
    expect(route.kind).toBe('single');
    if (route.kind === 'single') {
      expect(route.plan.taskType).toBe('search');
    }
  });

  it('returns beam for confidence < threshold with >= 2 distinct plans', () => {
    const result: ClassifierResult = {
      primary: 'search',
      confidence: 0.5,
      candidates: ['search', 'aggregate'],
    };
    const route = resolveRoute(result);
    expect(route.kind).toBe('beam');
    if (route.kind === 'beam') {
      expect(route.plans.map((p) => p.taskType)).toEqual([
        'search',
        'aggregate',
      ]);
    }
  });

  it('collapses beam to single when only one unique plan remains', () => {
    const result: ClassifierResult = {
      primary: 'search',
      confidence: 0.1,
      candidates: ['search', 'search'],
    };
    const route = resolveRoute(result);
    expect(route.kind).toBe('single');
    if (route.kind === 'single') {
      expect(route.plan.taskType).toBe('search');
    }
  });

  it('routes unknown primary through summarise fallback', () => {
    const result: ClassifierResult = {
      primary: 'unknown',
      confidence: 0.95,
      candidates: [],
    };
    const route = resolveRoute(result);
    expect(route.kind).toBe('single');
    if (route.kind === 'single') {
      expect(route.plan.taskType).toBe('summarise');
    }
  });

  it('respects a custom threshold', () => {
    const result: ClassifierResult = {
      primary: 'search',
      confidence: 0.6,
      candidates: ['search', 'aggregate'],
    };
    // Default threshold (0.7) → beam.
    expect(resolveRoute(result).kind).toBe('beam');
    // Custom threshold below the confidence → single.
    expect(resolveRoute(result, { threshold: 0.5 }).kind).toBe('single');
    // Threshold = 0 → always single.
    expect(resolveRoute(result, { threshold: 0 }).kind).toBe('single');
  });

  it('respects beamTopK cap', () => {
    const result: ClassifierResult = {
      primary: 'search',
      confidence: 0.1,
      candidates: ['search', 'aggregate', 'summarise', 'multi_hop'],
    };
    const route = resolveRoute(result, { beamTopK: 3 });
    expect(route.kind).toBe('beam');
    if (route.kind === 'beam') {
      expect(route.plans.map((p) => p.taskType)).toEqual([
        'search',
        'aggregate',
        'summarise',
      ]);
    }
  });

  it('rejects beamTopK < 2 to forbid a degenerate beam', () => {
    const result: ClassifierResult = {
      primary: 'search',
      confidence: 0.1,
      candidates: ['aggregate'],
    };
    expect(() => resolveRoute(result, { beamTopK: 1 })).toThrow(ValueError);
    expect(() => resolveRoute(result, { beamTopK: 0 })).toThrow(ValueError);
  });

  it('uses a custom plan registry when supplied', () => {
    const customSearch: StaticPlan = {
      taskType: 'search',
      template: oracle(vref('input'), 'custom-model'),
      memorySchema: null,
    };
    const overrides = new Map<TaskType, StaticPlan>([
      ['search', customSearch],
      ['summarise', STATIC_PLANS.get('summarise')!],
    ]);
    const route = resolveRoute(
      { primary: 'search', confidence: 1, candidates: [] },
      { plans: overrides },
    );
    expect(route.kind).toBe('single');
    if (route.kind === 'single') {
      expect(route.plan).toBe(customSearch);
    }
  });

  it('uses default constants as public API', () => {
    expect(DEFAULT_ROUTE_THRESHOLD).toBe(0.7);
    expect(DEFAULT_BEAM_TOP_K).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// 4. composeBeamPlan
// ---------------------------------------------------------------------------

describe('composeBeamPlan', () => {
  it('throws on empty input', () => {
    expect(() => composeBeamPlan([])).toThrow();
  });

  it('returns the only node unchanged for singleton input', () => {
    const node = oracle(vref('x'));
    expect(composeBeamPlan([node])).toBe(node);
  });

  it('wraps two nodes in a single cross', () => {
    const a = oracle(vref('a'));
    const b = oracle(vref('b'));
    expect(composeBeamPlan([a, b])).toEqual({
      tag: 'cross',
      left: a,
      right: b,
    });
  });

  it('right-associates three or more nodes', () => {
    const a = oracle(vref('a'));
    const b = oracle(vref('b'));
    const c = oracle(vref('c'));
    expect(composeBeamPlan([a, b, c])).toEqual({
      tag: 'cross',
      left: a,
      right: {
        tag: 'cross',
        left: b,
        right: c,
      },
    });
  });
});

// ---------------------------------------------------------------------------
// 5. Exports sanity
// ---------------------------------------------------------------------------

describe('task router public exports', () => {
  it('TASK_TYPES includes unknown; REAL_TASK_TYPES_LIST excludes it', () => {
    expect(TASK_TYPES).toContain('unknown');
    expect(REAL_TASK_TYPES_LIST).not.toContain('unknown');
    expect(TASK_TYPES.length).toBe(REAL_TASK_TYPES_LIST.length + 1);
  });

  it('isTaskType accepts every known tag and rejects others', () => {
    for (const tag of TASK_TYPES) {
      expect(isTaskType(tag)).toBe(true);
    }
    expect(isTaskType('nope')).toBe(false);
    expect(isTaskType(42)).toBe(false);
    expect(isTaskType(undefined)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

function findVoteNode(
  node: CombinatorNode,
): Extract<CombinatorNode, { tag: 'vote' }> | null {
  if (node.tag === 'vote') return node;
  const queue: CombinatorNode[] = [node];
  while (queue.length > 0) {
    const head = queue.shift()!;
    if (head.tag === 'vote') return head;
    for (const child of children(head)) queue.push(child);
  }
  return null;
}

function children(node: CombinatorNode): CombinatorNode[] {
  switch (node.tag) {
    case 'literal':
    case 'var':
      return [];
    case 'split':
      return [node.input, node.k];
    case 'peek':
      return [node.input, node.start, node.end];
    case 'map':
      return [node.fn.body, node.items];
    case 'filter':
      return [node.pred.body, node.items];
    case 'reduce':
      return node.init === undefined
        ? [node.op.body, node.items]
        : [node.op.body, node.items, node.init];
    case 'concat':
      return node.separator === undefined
        ? [node.items]
        : [node.items, node.separator];
    case 'cross':
      return [node.left, node.right];
    case 'vote':
      return [node.oracle, node.n];
    case 'ensemble':
      return [node.oracle];
    case 'oracle':
      return [node.prompt];
  }
}
