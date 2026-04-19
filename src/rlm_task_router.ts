/**
 * Task classification and plan routing for RLM v2.
 *
 * Exports types (`TaskType`, `PlanningInputs`, `ResolvedPlan`, `StaticPlan`,
 * routing results) plus the static plan registry (`STATIC_PLANS`),
 * `classifyTask`, and `resolveRoute`. The six default templates follow the
 * λ-RLM Table 1 recipes; the planner substitutes `k*` / `N` from quality
 * curves in `rlm_planner_quality.ts`.
 */

import { BudgetError, ConfigurationError, ValueError } from './exceptions.js';
import type { BaseLM } from './lm.js';
import { Predict } from './predict.js';
import type { Signature } from './signature.js';
import { signatureFromString } from './signature.js';

import type { CombinatorNode } from './rlm_combinators.js';
import {
  concat,
  cross,
  fn,
  lit,
  map,
  oracle,
  split,
  vote,
  vref,
} from './rlm_combinators.js';
import {
  DEFAULT_MAX_MEMORY_BYTES,
  type MemorySchema,
} from './rlm_memory.js';
import type { RLMBudget } from './rlm_types.js';

// ---------------------------------------------------------------------------
// Task type tag
// ---------------------------------------------------------------------------

/**
 * Closed set of task families the planner and router know about.
 *
 * The six "real" entries mirror Table 1 of the λ-RLM paper:
 * - `search`     — locate a fact in a large chunked context.
 * - `classify`   — assign one of a small finite set of labels.
 * - `aggregate`  — combine per-chunk findings into a holistic answer.
 * - `pairwise`   — compare exactly two items; `k` is always 2.
 * - `summarise`  — compress a long input into a shorter output.
 * - `multi_hop`  — chain multiple reasoning steps across sub-findings.
 *
 * `unknown` is the conservative fallback when the classifier can't
 * commit. `resolveRoute` maps `unknown` to the `summarise` plan (the
 * long-context-safe default); the planner's quality curve for
 * `unknown` is defensively tuned regardless.
 */
export type TaskType =
  | 'search'
  | 'classify'
  | 'aggregate'
  | 'pairwise'
  | 'summarise'
  | 'multi_hop'
  | 'unknown';

const REAL_TASK_TYPES: readonly TaskType[] = Object.freeze([
  'search',
  'classify',
  'aggregate',
  'pairwise',
  'summarise',
  'multi_hop',
]);

const ALL_TASK_TYPES: readonly TaskType[] = Object.freeze([
  ...REAL_TASK_TYPES,
  'unknown',
]);

const ALL_TASK_TYPE_SET: ReadonlySet<TaskType> = new Set(ALL_TASK_TYPES);

/** Type guard for `TaskType` including `unknown`. */
export function isTaskType(value: unknown): value is TaskType {
  return typeof value === 'string' && ALL_TASK_TYPE_SET.has(value as TaskType);
}

// ---------------------------------------------------------------------------
// Planner inputs
// ---------------------------------------------------------------------------

/**
 * Inputs the deterministic planner needs to resolve a static plan into
 * concrete `k`, `d`, `N` values and substitute them into the AST.
 *
 * - `taskType` picks the quality curve and the static plan template.
 * - `promptLength` is the raw character count of the prompt — equivalent
 *   to `n` in the λ-RLM paper. Character count (not token count) keeps
 *   the planner tokenizer-free and provider-agnostic (curves may later
 *   modulate quality by length via `nScale` in `rlm_planner_quality.ts`).
 * - `budget` is the caller's declared ceiling. The planner must return a
 *   plan whose `estimatedOracleCalls ≤ budget.maxOracleCalls`.
 * - `preferredK` overrides the quality-argmax search. When set, the
 *   planner pins that `k` and only applies the budget-based truncation.
 *   The router uses this for beam routing (each branch of the `Cross`
 *   inherits the chosen plan's preferred `k` from `STATIC_PLANS`) and
 *   power users can pin it explicitly through `RLMOptions`.
 */
export interface PlanningInputs {
  readonly taskType: TaskType;
  readonly promptLength: number;
  readonly budget: RLMBudget;
  readonly preferredK?: number;
}

// ---------------------------------------------------------------------------
// Static plan — the input to the planner
// ---------------------------------------------------------------------------

/**
 * Unresolved plan template. `STATIC_PLANS` holds one instance per
 * non-`unknown` task type. `unknown` resolves to `summarise` by
 * convention in `resolveRoute`.
 *
 * `template` contains `vref('k')` / `vref('n')` placeholders where the
 * planner's `resolvePlan` substitutes literal values. Every plan is
 * idempotent under re-resolution: running `resolvePlan` twice is
 * structurally equal to running it once.
 */
export interface StaticPlan {
  readonly taskType: TaskType;
  readonly template: CombinatorNode;
  readonly memorySchema: MemorySchema | null;
}

// ---------------------------------------------------------------------------
// Resolved plan — the output of the planner
// ---------------------------------------------------------------------------

/**
 * The planner's output. Every field is fully determined; the evaluator
 * needs no further negotiation to run it.
 */
export interface ResolvedPlan {
  readonly plan: CombinatorNode;
  readonly partitionK: number;
  readonly depth: number;
  readonly selfConsistencyN: number;
  readonly estimatedOracleCalls: number;
  readonly memorySchema: MemorySchema | null;
}

// ---------------------------------------------------------------------------
// Classifier result
// ---------------------------------------------------------------------------

/**
 * Result of the one-shot classifier `Predict('context: str -> primary:
 * str, confidence: float, candidates: list[str]')`.
 *
 * - `primary` is the top-ranked task type. `'unknown'` when the
 *   classifier produces an unparseable result or hallucinates a label
 *   outside the `TaskType` union.
 * - `confidence` is in `[0, 1]`; clamped by `classifyTask` so downstream
 *   threshold checks never see out-of-range values.
 * - `candidates` is the sorted candidate list, primary first, deduped.
 *   Any LM hallucinations outside `TaskType` are filtered out.
 */
export interface ClassifierResult {
  readonly primary: TaskType;
  readonly confidence: number;
  readonly candidates: readonly TaskType[];
}

// ---------------------------------------------------------------------------
// Route result
// ---------------------------------------------------------------------------

/**
 * Output of `resolveRoute`.
 *
 * - `{ kind: 'single'; plan }` — high-confidence classification;
 *   evaluate a single plan directly.
 * - `{ kind: 'beam'; plans }` — low-confidence classification; the RLM
 *   facade evaluates the candidate plans in order using its beam
 *   selection policy. `composeBeamPlan` remains exported for power
 *   users who want structural `cross(...)` composition themselves.
 *   The first entry in `plans` is always the classifier's primary
 *   candidate.
 */
export type RouteResult =
  | { readonly kind: 'single'; readonly plan: StaticPlan }
  | { readonly kind: 'beam'; readonly plans: readonly StaticPlan[] };

// ===========================================================================
// Static plan registry — six entries per the Table 1 recipes
// ===========================================================================
//
// Every template uses two conventions:
//
// 1. `vref('input')` is the top-level prompt string. `RLM` seeds
//    `ctx.scope` with `input → prompt` before `evaluate`.
// 2. `vref('k')` is the partition width; `vref('n')` is the
//    self-consistency sample width. The planner's `resolvePlan`
//    substitutes them.
//
// Model hints (`'gpt-fast'`, `'gpt-deep'`) are symbolic — the concrete
// `BaseLM` bound to each name is looked up from `ctx.lmRegistry` at
// evaluation time. The router does not depend on any specific provider.

/** Fast cheap-per-chunk model; paired with `'gpt-deep'` for synthesis. */
const MODEL_FAST = 'gpt-fast';

/** Deeper model for final synthesis / pairwise comparison / multi-hop. */
const MODEL_DEEP = 'gpt-deep';

const SEPARATOR_SINGLE = '\n---\n';
const SEPARATOR_DOUBLE = '\n\n';

// ---------------------------------------------------------------------------
// Default memory schemas
// ---------------------------------------------------------------------------
//
// Search / aggregate / summarise / multi_hop plans run multi-turn oracle
// trees whose chunks benefit from a shared "failure diagnostic" scratch
// pad: a previous oracle turn can emit a `WriteMemory` effect to flag a
// pattern it noticed and every later chunk sees it in the injected banner.
//
// The fields are intentionally narrow (three short strings) so the
// banner stays cheap. Plan authors that need a richer memory surface
// can supply their own `StaticPlan` via `RLMOptions.plans`.

const FAILURE_DIAGNOSTIC_SCHEMA: MemorySchema = Object.freeze({
  name: 'failure_diagnostic',
  fields: Object.freeze([
    Object.freeze({
      name: 'failure_pattern',
      type: 'string' as const,
      description:
        'Short description of a mistake or dead-end a prior oracle turn hit.',
      maxLength: 160,
    }),
    Object.freeze({
      name: 'next_check',
      type: 'string' as const,
      description:
        'Concrete verification the oracle should perform before finalising.',
      maxLength: 160,
    }),
    Object.freeze({
      name: 'prevented_action',
      type: 'string' as const,
      description:
        'Action the oracle should avoid repeating on later turns.',
      maxLength: 160,
    }),
  ]),
  maxBytesSerialized: DEFAULT_MAX_MEMORY_BYTES,
});

/**
 * Search: fan out per chunk, collect findings, return the concatenated
 * list. Quality rises monotonically with `k` up to the argmax at `k=8`.
 */
const SEARCH_TEMPLATE: CombinatorNode = concat(
  map(
    fn('chunk', oracle(vref('chunk'), MODEL_FAST)),
    split(vref('input'), vref('k')),
  ),
  lit(SEPARATOR_SINGLE),
);

/**
 * Classify: single-shot with self-consistency. Input is assumed short
 * enough that partitioning is unnecessary; all quality lift comes from
 * `N`. The planner caps `N` at `budget.selfConsistencyN`.
 */
const CLASSIFY_TEMPLATE: CombinatorNode = vote(
  oracle(vref('input'), MODEL_FAST),
  vref('n'),
  'majority',
);

/**
 * Aggregate: fan out per chunk then feed the concatenated findings to a
 * deeper synthesis oracle. The outer oracle's prompt is built by the
 * evaluator at runtime — each `oracle.prompt` is evaluated to a string
 * before the LLM call, so the synthesis prompt is the map-concat chain's
 * own output.
 */
const AGGREGATE_TEMPLATE: CombinatorNode = oracle(
  concat(
    map(
      fn('chunk', oracle(vref('chunk'), MODEL_FAST)),
      split(vref('input'), vref('k')),
    ),
    lit(SEPARATOR_DOUBLE),
  ),
  MODEL_DEEP,
);

/**
 * Pairwise: always `k = 2`. Split the input, feed the two halves
 * concatenated with a separator to a single deep oracle. No
 * self-consistency width — pairwise tasks are deterministic given the
 * input.
 */
const PAIRWISE_TEMPLATE: CombinatorNode = oracle(
  concat(split(vref('input'), vref('k')), lit(SEPARATOR_SINGLE)),
  MODEL_DEEP,
);

/**
 * Summarise: identical structure to aggregate — fan out, synthesize.
 * The distinction shows up in the quality curve: summarisation peaks at
 * `k = 6` (aggregate at `k = 4`), so the same template produces
 * different plans under the planner.
 */
const SUMMARISE_TEMPLATE: CombinatorNode = oracle(
  concat(
    map(
      fn('chunk', oracle(vref('chunk'), MODEL_FAST)),
      split(vref('input'), vref('k')),
    ),
    lit(SEPARATOR_DOUBLE),
  ),
  MODEL_DEEP,
);

/**
 * Multi-hop: fan out per chunk, synthesize, then vote over the
 * synthesis for self-consistency. Multi-hop reasoning benefits from
 * both `k` (diverse sub-findings) and `N` (consensus on the final hop);
 * the evaluator evaluates `vote.oracle.prompt` exactly once, so the
 * per-chunk oracles run once, and the synthesis oracle runs `N` times.
 */
const MULTI_HOP_TEMPLATE: CombinatorNode = vote(
  oracle(
    concat(
      map(
        fn('chunk', oracle(vref('chunk'), MODEL_FAST)),
        split(vref('input'), vref('k')),
      ),
      lit(SEPARATOR_DOUBLE),
    ),
    MODEL_DEEP,
  ),
  vref('n'),
  'majority',
);

/**
 * The six-entry static plan registry. `unknown` is intentionally
 * absent — `resolveRoute` maps it to `summarise` (see below).
 *
 * Callers may extend this registry by supplying additional
 * `StaticPlan`s to `RLMOptions.plans`. User plans override defaults on
 * duplicate `taskType`.
 */
export const STATIC_PLANS: ReadonlyMap<TaskType, StaticPlan> = Object.freeze(
  new Map<TaskType, StaticPlan>([
    [
      'search',
      {
        taskType: 'search',
        template: SEARCH_TEMPLATE,
        memorySchema: FAILURE_DIAGNOSTIC_SCHEMA,
      },
    ],
    [
      'classify',
      {
        taskType: 'classify',
        template: CLASSIFY_TEMPLATE,
        memorySchema: null,
      },
    ],
    [
      'aggregate',
      {
        taskType: 'aggregate',
        template: AGGREGATE_TEMPLATE,
        memorySchema: FAILURE_DIAGNOSTIC_SCHEMA,
      },
    ],
    [
      'pairwise',
      {
        taskType: 'pairwise',
        template: PAIRWISE_TEMPLATE,
        memorySchema: null,
      },
    ],
    [
      'summarise',
      {
        taskType: 'summarise',
        template: SUMMARISE_TEMPLATE,
        memorySchema: FAILURE_DIAGNOSTIC_SCHEMA,
      },
    ],
    [
      'multi_hop',
      {
        taskType: 'multi_hop',
        template: MULTI_HOP_TEMPLATE,
        memorySchema: FAILURE_DIAGNOSTIC_SCHEMA,
      },
    ],
  ]),
);

// ===========================================================================
// classifyTask
// ===========================================================================

/**
 * One-shot LM classifier.
 *
 * Uses the signature `context: str -> primary: str, confidence: float,
 * candidates: list[str]`. The `context` input is built from the RLM's
 * output signature (so the LM knows what kind of answer the user
 * ultimately expects) plus the prompt the user supplied.
 *
 * On validation failure — the LM hallucinates an unknown label, the
 * confidence is out of range, the candidates list is empty, or the
 * adapter cannot parse the payload — the function returns a safe default
 * (`primary: 'unknown'`, `confidence: 0`, `candidates: []`). Ordinary
 * `Predict` / LM failures are folded into that default so routing can
 * fall back to `summarise`. `BudgetError` is never swallowed: it
 * propagates so global budget exhaustion remains visible.
 *
 * Exactly one `lm.acall` per `classifyTask` call. Callers that already
 * know the task type (`RLMOptions.taskType`) skip this function
 * entirely.
 */
export async function classifyTask(
  prompt: string,
  signature: Signature,
  lm: BaseLM,
): Promise<ClassifierResult> {
  const context = buildClassifierContext(prompt, signature);
  try {
    const prediction = await new Predict(CLASSIFIER_SIGNATURE).acall({
      context,
      lm,
    });
    const primary = normalizePrimary(prediction.get('primary'));
    const confidence = normalizeConfidence(prediction.get('confidence'));
    const candidates = normalizeCandidates(
      prediction.get('candidates'),
      primary,
    );
    return { primary, confidence, candidates };
  } catch (err) {
    if (err instanceof BudgetError) throw err;
    // LM / adapter boundary: map any other failure to the documented safe default.
    return UNKNOWN_CLASSIFIER_RESULT;
  }
}

const CLASSIFIER_SIGNATURE: Signature = signatureFromString(
  'context: str -> primary: str, confidence: float, candidates: list[str]',
  'Classify the user request into one of: search, classify, aggregate, pairwise, summarise, multi_hop, or unknown. Return the top choice as `primary`, a confidence score in [0,1], and the full ranked candidate list.',
);

const UNKNOWN_CLASSIFIER_RESULT: ClassifierResult = Object.freeze({
  primary: 'unknown' as const,
  confidence: 0,
  candidates: Object.freeze([]) as readonly TaskType[],
});

function buildClassifierContext(prompt: string, signature: Signature): string {
  const inputNames = [...signature.inputFields.keys()].join(', ') || '(none)';
  const outputNames = [...signature.outputFields.keys()].join(', ') || '(none)';
  return [
    `Signature inputs: ${inputNames}`,
    `Signature outputs: ${outputNames}`,
    '',
    'Prompt:',
    prompt,
  ].join('\n');
}

function normalizePrimary(raw: unknown): TaskType {
  if (typeof raw !== 'string') return 'unknown';
  const lowered = raw.trim().toLowerCase();
  // Hallucinated aliases map to the canonical tag when unambiguous.
  // Everything unrecognised → `'unknown'`; no fuzzy matching that could
  // accidentally pick the wrong plan.
  if (ALL_TASK_TYPE_SET.has(lowered as TaskType)) {
    return lowered as TaskType;
  }
  return 'unknown';
}

function normalizeConfidence(raw: unknown): number {
  if (typeof raw !== 'number' || !Number.isFinite(raw)) return 0;
  if (raw < 0) return 0;
  if (raw > 1) return 1;
  return raw;
}

function normalizeCandidates(
  raw: unknown,
  primary: TaskType,
): readonly TaskType[] {
  const seen = new Set<TaskType>();
  const out: TaskType[] = [];
  if (primary !== 'unknown') {
    seen.add(primary);
    out.push(primary);
  }
  if (Array.isArray(raw)) {
    for (const entry of raw) {
      if (typeof entry !== 'string') continue;
      const candidate = entry.trim().toLowerCase();
      if (!ALL_TASK_TYPE_SET.has(candidate as TaskType)) continue;
      const tag = candidate as TaskType;
      if (seen.has(tag)) continue;
      seen.add(tag);
      out.push(tag);
    }
  }
  return Object.freeze(out);
}

// ===========================================================================
// resolveRoute
// ===========================================================================

export const DEFAULT_ROUTE_THRESHOLD = 0.7;
export const DEFAULT_BEAM_TOP_K = 2;

export interface ResolveRouteOptions {
  /**
   * Confidence below this threshold triggers beam routing. Defaults to
   * `DEFAULT_ROUTE_THRESHOLD = 0.7`. Pass `0` to disable beam routing
   * entirely (always take the classifier's primary).
   */
  readonly threshold?: number;
  /**
   * Max number of plans in the beam when beam routing is triggered.
   * Defaults to `DEFAULT_BEAM_TOP_K = 2`. The classifier's primary is
   * always first; additional entries come from `candidates` in order.
   */
  readonly beamTopK?: number;
  /**
   * Optional override of the static plan registry (`RLMOptions.plans`).
   * Keys missing from the override fall back to `STATIC_PLANS`.
   */
  readonly plans?: ReadonlyMap<TaskType, StaticPlan>;
}

/**
 * Pick a route: a single plan for high-confidence classifications, or
 * a beam of plans for low-confidence ones.
 *
 * Algorithm:
 *
 * 1. Look up the primary plan (with `unknown` → `summarise` fallback).
 * 2. If `confidence ≥ threshold`, return `{ kind: 'single', plan }`.
 * 3. Otherwise, take the top-`beamTopK` candidates (in classifier
 *    order, starting with the primary). Filter out any for which no
 *    plan exists. De-duplicate by plan identity. If only one plan
 *    remains, return `single`; else return `beam`.
 */
export function resolveRoute(
  classifierResult: ClassifierResult,
  options: ResolveRouteOptions = {},
): RouteResult {
  const threshold = options.threshold ?? DEFAULT_ROUTE_THRESHOLD;
  const beamTopK = options.beamTopK ?? DEFAULT_BEAM_TOP_K;
  if (!Number.isInteger(beamTopK) || beamTopK < 2) {
    throw new ValueError(
      `resolveRoute beamTopK must be an integer >= 2, got ${String(beamTopK)}.`,
    );
  }
  const registry = options.plans ?? STATIC_PLANS;
  const primaryPlan = lookupPlan(classifierResult.primary, registry);
  if (classifierResult.confidence >= threshold) {
    return { kind: 'single', plan: primaryPlan };
  }
  const tags: TaskType[] = [classifierResult.primary];
  for (const candidate of classifierResult.candidates) {
    if (tags.length >= beamTopK) break;
    if (tags.includes(candidate)) continue;
    tags.push(candidate);
  }
  const seenPlans = new Set<StaticPlan>();
  const plans: StaticPlan[] = [];
  for (const tag of tags) {
    if (plans.length >= beamTopK) break;
    const plan = lookupPlan(tag, registry);
    if (seenPlans.has(plan)) continue;
    seenPlans.add(plan);
    plans.push(plan);
  }
  if (plans.length <= 1) {
    return { kind: 'single', plan: primaryPlan };
  }
  return { kind: 'beam', plans: Object.freeze(plans) };
}

/**
 * Map a `TaskType` to the registered `StaticPlan`, falling back to
 * `summarise` for `unknown`. Throws `ConfigurationError` when the
 * registry is missing the fallback — an installer bug that the build
 * should catch.
 */
function lookupPlan(
  taskType: TaskType,
  registry: ReadonlyMap<TaskType, StaticPlan>,
): StaticPlan {
  const direct = registry.get(taskType);
  if (direct !== undefined) return direct;
  const fallback = registry.get('summarise');
  if (fallback !== undefined) return fallback;
  throw new ConfigurationError(
    `RLM task router: no plan for '${taskType}' and no summarise fallback in the registry`,
  );
}

// ===========================================================================
// Beam composition helper
// ===========================================================================

/**
 * Combine an array of resolved plan nodes into a single top-level plan.
 *
 * - One plan: return it directly.
 * - Two plans: `cross(a, b)`.
 * - Three+ plans: right-associated `cross` chain — `cross(a, cross(b, c))`.
 *
 * Exposed for callers that want to evaluate a beam as a single
 * structural AST. The `RLM` facade keeps beam selection outside the
 * AST so it can report route metadata directly.
 */
export function composeBeamPlan(
  nodes: readonly CombinatorNode[],
): CombinatorNode {
  if (nodes.length === 0) {
    throw new ValueError('composeBeamPlan: expected at least one node');
  }
  if (nodes.length === 1) return nodes[0]!;
  let acc: CombinatorNode = nodes[nodes.length - 1]!;
  for (let i = nodes.length - 2; i >= 0; i -= 1) {
    acc = cross(nodes[i]!, acc);
  }
  return acc;
}

// ===========================================================================
// Re-exports
// ===========================================================================

/** Publicly accessible list of every task type the router knows about. */
export const TASK_TYPES: readonly TaskType[] = ALL_TASK_TYPES;
/** Publicly accessible list excluding the `unknown` fallback. */
export const REAL_TASK_TYPES_LIST: readonly TaskType[] = REAL_TASK_TYPES;
