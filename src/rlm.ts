/**
 * Public RLM v2 facade.
 *
 * Thin wrapper over the typed combinator evaluator. `aforward(kwargs)`:
 *
 *   1. Resolves the primary `BaseLM` via `options.subLm ?? settings.lm`.
 *   2. Builds a single `prompt` string from the inputs against the
 *      signature.
 *   3. Classifies the task (or honors `options.taskType`) to pick a
 *      static plan; beam-routes on low confidence.
 *   4. Resolves each candidate plan with the deterministic planner
 *      (`k*`, `d`, `N`).
 *   5. Runs the combinator evaluator over the resolved plan(s).
 *   6. Wraps the result in a `Prediction` populated against the
 *      caller's output signature.
 *
 * This file is intentionally small: every runtime decision lives in a
 * dedicated module (`rlm_task_router`, `rlm_planner`, `rlm_evaluator`,
 * `rlm_effects`, `rlm_memory`). The facade's job is to wire them
 * together per the contract in
 * `docs/product/rlm-v2-architecture.md` §2.6.
 */

import { Adapter, JSONAdapter, type Message } from './adapter.js';
import type { BaseLM } from './lm.js';
import { BaseLM as BaseLMClass } from './lm.js';
import {
  BudgetError,
  ConfigurationError,
  RuntimeError,
  ValueError,
} from './exceptions.js';
import { Module } from './module.js';
import { Prediction } from './prediction.js';
import { Predict } from './predict.js';
import { settings } from './settings.js';
import { ensureSignature, type Signature } from './signature.js';
import type { InferInputs, InferOutputs, SignatureInput } from './signature_types.js';

import type { CombinatorValue } from './rlm_combinators.js';
import {
  RLM_VERIFIER_SIGNATURE,
  buildEvaluationContext,
  evaluate,
} from './rlm_evaluator.js';
import {
  EMPTY_TYPED_MEMORY_STATE,
  initialMemoryState,
  type MemorySchema,
} from './rlm_memory.js';
import {
  mergeBudget,
  type EffectHandler,
  type EvaluationContext,
  type EvaluationTrace,
  type RLMBudget,
} from './rlm_types.js';
import { resolvePlan, type ResolvePlanArgs } from './rlm_planner.js';
import {
  DEFAULT_BEAM_TOP_K,
  STATIC_PLANS,
  classifyTask,
  isTaskType,
  resolveRoute,
  type ClassifierResult,
  type RouteResult,
  type StaticPlan,
  type TaskType,
} from './rlm_task_router.js';

const EMPTY_RECORD = Object.freeze({}) as Record<string, unknown>;

/**
 * Constructor options for `RLM`.
 *
 * Every field is optional; defaults preserve the least-surprising
 * behavior. See `docs/product/rlm-v2-architecture.md` §2.6 for the full
 * intent of each knob.
 */
export interface RLMOptions {
  /**
   * Partial override of the run-wide budget. Missing fields inherit
   * `DEFAULT_BUDGET`. The evaluator enforces every field (see
   * `BudgetError` handling in `src/rlm_evaluator.ts`).
   */
  readonly budget?: Partial<RLMBudget>;
  /**
   * Skip the classifier and use this task type directly. Useful when
   * the caller already knows the task family or for deterministic
   * tests. `'unknown'` is accepted; it routes to the `summarise`
   * fallback plan.
   */
  readonly taskType?: TaskType;
  /**
   * Primary LM. Resolution order: `options.subLm ?? settings.lm`.
   * `null` is treated identically to `undefined` — the RLM looks up
   * `settings.lm`. Absent LMs throw `ConfigurationError` at call time.
   */
  readonly subLm?: BaseLM | null;
  /**
   * Named registry for `modelHint` routing inside the plan AST. E.g.
   * the default static plans emit `oracle(..., 'gpt-fast')` and
   * `oracle(..., 'gpt-deep')`; users bind those names to their own
   * `BaseLM` instances via this map. The primary `subLm` is used when
   * a `modelHint` is absent or unmatched.
   */
  readonly lmRegistry?: ReadonlyMap<string, BaseLM>;
  /**
   * Extra effect handlers, keyed by `handler.name`. Built-in handlers are
   * always present; same-name entries here override them. Duplicate names in
   * this list throw `ValueError` at construction.
   */
  readonly handlers?: readonly EffectHandler[];
  /**
   * User-supplied `StaticPlan`s. Each plan is indexed by its
   * `taskType` and overrides the built-in registry entry for the same
   * tag (if any). Plans without a matching tag add new dispatchable
   * task types beyond the six defaults.
   */
  readonly plans?: readonly StaticPlan[];
  /**
   * When `true` (default), the returned `Prediction` includes a `trace`
   * key holding the evaluator's `EvaluationTrace[]`. Set `false` to
   * minimize prediction memory for high-volume runs.
   */
  readonly trackTrace?: boolean;
  /**
   * Threshold below which the router beam-routes. Mirrors
   * `resolveRoute`'s default of 0.7. Power users can tune this per
   * RLM instance.
   */
  readonly routeThreshold?: number;
  /**
   * Max number of plans included in a beam route. Mirrors the default
   * of 2 from `resolveRoute`.
   */
  readonly routeBeamTopK?: number;
  /**
   * Injection point for tests: override the classifier. Default is
   * `classifyTask` from `./rlm_task_router.ts`. Returns a
   * `ClassifierResult` that the router dispatches on.
   */
  readonly classifier?: (
    prompt: string,
    signature: Signature,
    lm: BaseLM,
  ) => Promise<ClassifierResult>;
}

// ---------------------------------------------------------------------------
// RLM class
// ---------------------------------------------------------------------------

/**
 * RLM v2 public facade. Subclasses `Module` so it participates in
 * `Evaluate`, `Parallel`, and callback machinery just like `Predict`
 * does.
 *
 * Type parameters mirror `Predict<TSig, TInputs, TOutputs>`:
 * `TInputs` / `TOutputs` are inferred from the signature so TypeScript
 * callers get end-to-end typing of `aforward`'s kwargs and the
 * returned `Prediction`'s output fields.
 */
export class RLM<
  TSig extends SignatureInput = Signature,
  TInputs extends Record<string, unknown> = InferInputs<TSig>,
  TOutputs extends Record<string, unknown> = InferOutputs<TSig>,
> extends Module<TInputs, TOutputs> {
  readonly signature: Signature;
  readonly budget: RLMBudget;
  readonly trackTrace: boolean;
  readonly taskTypeOverride: TaskType | null;
  readonly subLm: BaseLM | null;
  readonly lmRegistry: ReadonlyMap<string, BaseLM>;
  readonly handlers: ReadonlyMap<string, EffectHandler>;
  readonly plans: ReadonlyMap<TaskType, StaticPlan>;
  readonly routeThreshold: number;
  readonly routeBeamTopK: number;
  readonly classifier: (
    prompt: string,
    signature: Signature,
    lm: BaseLM,
  ) => Promise<ClassifierResult>;

  constructor(signature: TSig, options: RLMOptions = {}) {
    super();
    this.signature = ensureSignature(signature);
    if (this.signature.outputFields.size === 0) {
      throw new ValueError(
        'RLM signature must declare at least one output field.',
      );
    }
    if (options.taskType !== undefined && !isTaskType(options.taskType)) {
      throw new ValueError(
        `RLM options.taskType must be a valid TaskType, got ${String(options.taskType)}.`,
      );
    }
    this.budget = mergeBudget(options.budget);
    this.trackTrace = options.trackTrace ?? true;
    this.taskTypeOverride = options.taskType ?? null;
    this.subLm = options.subLm ?? null;
    this.lmRegistry = options.lmRegistry ?? new Map<string, BaseLM>();
    this.handlers = normalizeHandlers(options.handlers);
    this.plans = mergePlans(options.plans);
    this.routeThreshold = normalizeRouteThreshold(options.routeThreshold);
    this.routeBeamTopK = normalizeRouteBeamTopK(options.routeBeamTopK);
    this.classifier = options.classifier ?? classifyTask;
  }

  /**
   * Synchronous invocation is unsupported — the evaluator is async
   * because oracle leaves issue async LM calls. Throwing a loud error
   * matches `LM`'s async-only posture (see `docs/product/rlm-v2-architecture.md`).
   */
  override forward(_kwargs: TInputs = EMPTY_RECORD as TInputs): never {
    throw new RuntimeError(
      'RLM is async-only. Use acall() or aforward() instead.',
    );
  }

  override async aforward(
    kwargs: TInputs = EMPTY_RECORD as TInputs,
  ): Promise<Prediction<TOutputs>> {
    const lm = this.resolvePrimaryLm();
    const inputs = this.resolvePromptInputs(kwargs as Record<string, unknown>);
    const prompt = this.buildPrompt(inputs);
    const classifierResult = await this.resolveClassifierResult(prompt, lm);
    const route = resolveRoute(classifierResult, {
      threshold: this.routeThreshold,
      beamTopK: this.routeBeamTopK,
      plans: this.plans,
    });
    // A single shared ctx carries the budget counter, trace, and scope
    // across plan execution so beam-routed runs bill against one
    // budget. Memory is plan-scoped (each plan may declare a different
    // schema) and gets attached per-plan inside `runPlan` via a
    // cloned context.
    const ctx = buildEvaluationContext({
      lm,
      signature: this.signature,
      budget: this.budget,
      lmRegistry: this.lmRegistry,
      handlers: this.handlers,
      scope: new Map<string, CombinatorValue>([['input', prompt]]),
    });
    const executed = await this.executeRoute(route, prompt, ctx);
    return this.buildPrediction(executed, classifierResult, route, ctx);
  }

  // --------------------------------------------------------------------------
  // Step 1: input validation
  // --------------------------------------------------------------------------
  private resolvePromptInputs(
    kwargs: Record<string, unknown>,
  ): Record<string, unknown> {
    const missing: string[] = [];
    const inputs: Record<string, unknown> = {};
    for (const name of this.signature.inputFields.keys()) {
      if (name in kwargs) {
        inputs[name] = kwargs[name];
        continue;
      }
      const field = this.signature.inputFields.get(name);
      if (field?.default !== undefined) {
        inputs[name] = field.default;
        continue;
      }
      missing.push(name);
    }
    if (missing.length > 0) {
      throw new ValueError(
        `RLM is missing required inputs: ${missing.join(', ')}`,
      );
    }
    return inputs;
  }

  // --------------------------------------------------------------------------
  // Step 2: LM resolution
  // --------------------------------------------------------------------------

  private resolvePrimaryLm(): BaseLM {
    const candidate = this.subLm ?? settings.lm;
    if (!(candidate instanceof BaseLMClass)) {
      throw new ConfigurationError(
        'RLM sub-LM resolution failed. Configure settings.lm or pass options.subLm.',
      );
    }
    return candidate;
  }

  // --------------------------------------------------------------------------
  // Step 3: Prompt construction
  // --------------------------------------------------------------------------

  /**
   * Assemble the top-level prompt via the active adapter, then flatten
   * the resulting message list into a deterministic single string. This
   * preserves the repository's canonical prompt formatting instead of
   * re-implementing a second prompt serializer inside RLM.
   */
  private buildPrompt(inputs: Record<string, unknown>): string {
    const adapter = this.resolveAdapter();
    const messages = adapter.format(this.signature, [], inputs);
    return flattenMessages(messages);
  }

  private resolveAdapter(): Adapter {
    const configured = settings.adapter;
    if (configured === null || configured === undefined) {
      return new JSONAdapter();
    }
    if (!(configured instanceof Adapter)) {
      throw new RuntimeError('settings.adapter must be an Adapter instance');
    }
    return configured;
  }

  // --------------------------------------------------------------------------
  // Step 4: Classifier resolution
  // --------------------------------------------------------------------------

  private async resolveClassifierResult(
    prompt: string,
    lm: BaseLM,
  ): Promise<ClassifierResult> {
    if (this.taskTypeOverride !== null) {
      return {
        primary: this.taskTypeOverride,
        confidence: 1,
        candidates: [this.taskTypeOverride],
      };
    }
    return this.classifier(prompt, this.signature, lm);
  }

  // --------------------------------------------------------------------------
  // Step 5: Plan resolution + evaluation
  // --------------------------------------------------------------------------

  /**
   * For a single-plan route, evaluate once. For a beam route, evaluate
   * every plan concurrently, discard empty-string fallbacks, and then
   * reduce the remaining candidates with a lightweight verifier oracle.
   * Beam planning reserves a small slice of `maxOracleCalls` for that
   * verifier step so each branch's self-consistency width is computed
   * against a realistic route budget.
   *
   * Why not a `cross` wrapper? `cross` is list-typed; our templates
   * return strings. Beam routing is semantically "try several and
   * pick one", which maps cleanly to `Promise.all` + first-selection
   * at the facade. The `composeBeamPlan` helper remains in the router
   * as a structural utility for advanced callers who author plans
   * that return lists and want a Cartesian product.
   */
  private async executeRoute(
    route: RouteResult,
    prompt: string,
    ctx: EvaluationContext,
  ): Promise<ExecutedRoute> {
    if (route.kind === 'single') {
      return {
        answer: await this.runPlan(route.plan, prompt.length, ctx),
        selectedTaskType: route.plan.taskType,
      };
    }
    const planningBudget = reserveBeamPlanningBudget(
      this.budget,
      route.plans.length,
    );
    const results = await Promise.all(
      route.plans.map(async (plan) => ({
        answer: await this.runPlan(plan, prompt.length, ctx, planningBudget),
        selectedTaskType: plan.taskType,
      })),
    );
    const nonEmpty = results.filter((value) => value.answer.trim() !== '');
    if (nonEmpty.length === 0) {
      return (
        results[0] ?? {
          answer: '',
          selectedTaskType: route.plans[0]?.taskType ?? 'summarise',
        }
      );
    }
    if (nonEmpty.length === 1) {
      return nonEmpty[0]!;
    }
    return this.selectBeamCandidate(prompt, nonEmpty, ctx);
  }

  private async runPlan(
    plan: StaticPlan,
    promptLength: number,
    ctx: EvaluationContext,
    planningBudget: RLMBudget = this.budget,
  ): Promise<string> {
    const args: ResolvePlanArgs = {
      plan: plan.template,
      planningInputs: {
        taskType: plan.taskType,
        promptLength,
        budget: planningBudget,
      },
      memorySchema: plan.memorySchema,
    };
    const resolved = resolvePlan(args);
    const planCtx = withPlanMemory(ctx, resolved.memorySchema);
    const value = await evaluate(resolved.plan, planCtx);
    return coerceAnswer(value);
  }

  // --------------------------------------------------------------------------
  // Step 6: Prediction assembly
  // --------------------------------------------------------------------------

  /**
   * Populate a `Prediction` against the caller's signature. The
   * evaluator returns a single string; the first output field of the
   * signature receives that string. Every other output field is set
   * to `undefined` at runtime (the generic `TOutputs` type is
   * advisory; the adapter protocol is the source of truth on shape).
   *
   * When `trackTrace` is true, the returned prediction also carries:
   *
   * - `_rlm_route.kind` — `'single'` or `'beam'`.
   * - `_rlm_route.taskTypes` — the list of dispatched task types.
   * - `_rlm_route.selectedTaskType` — the branch that produced the
   *   returned answer.
   * - `_rlm_trace` — the evaluator's `EvaluationTrace[]`.
   * - `_rlm_classifier` — the classifier result (post-override).
   *
   * These underscore-prefixed fields avoid colliding with user output
   * names while remaining discoverable through `prediction.get()`.
   */
  private buildPrediction(
    executed: ExecutedRoute,
    classifierResult: ClassifierResult,
    route: RouteResult,
    ctx: EvaluationContext,
  ): Prediction<TOutputs> {
    const payload: Record<string, unknown> = {};
    const outputNames = [...this.signature.outputFields.keys()];
    for (const name of outputNames) {
      payload[name] = undefined;
    }
    if (outputNames.length > 0 && outputNames[0] !== undefined) {
      payload[outputNames[0]] = executed.answer;
    }
    if (this.trackTrace) {
      payload._rlm_route = buildRouteMetadata(route, executed.selectedTaskType);
      payload._rlm_classifier = classifierResult;
      payload._rlm_trace = ctx.trace satisfies readonly EvaluationTrace[];
    }
    return Prediction.create<TOutputs>(payload);
  }

  private async selectBeamCandidate(
    prompt: string,
    candidates: readonly ExecutedRoute[],
    ctx: EvaluationContext,
  ): Promise<ExecutedRoute> {
    const verdicts = await Promise.all(
      candidates.map((candidate) =>
        this.verifyBeamCandidate(prompt, candidate.answer, ctx),
      ),
    );
    const selectedIndex = verdicts.findIndex((verdict) => verdict);
    return candidates[selectedIndex >= 0 ? selectedIndex : 0]!;
  }

  private async verifyBeamCandidate(
    prompt: string,
    candidate: string,
    ctx: EvaluationContext,
  ): Promise<boolean> {
    consumeOracleBudget(ctx);
      const prediction = await new Predict(RLM_VERIFIER_SIGNATURE).acall({
      prompt,
      candidate,
      lm: ctx.lm,
    });
    return prediction.get('verdict') === true;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function normalizeHandlers(
  handlers?: readonly EffectHandler[],
): ReadonlyMap<string, EffectHandler> {
  if (handlers === undefined || handlers.length === 0) {
    return new Map<string, EffectHandler>();
  }
  const out = new Map<string, EffectHandler>();
  for (const handler of handlers) {
    if (typeof handler.name !== 'string' || handler.name.trim() === '') {
      throw new ValueError(
        'RLM effect handler must have a non-empty `name` string.',
      );
    }
    if (out.has(handler.name)) {
      throw new ValueError(
        `Duplicate RLM effect handler name "${handler.name}".`,
      );
    }
    out.set(handler.name, handler);
  }
  return out;
}

/**
 * Merge a user-supplied `plans` array with the built-in static plan
 * registry. User plans take precedence on duplicate `taskType` tags.
 * Throws `ValueError` on duplicate tags within the user list itself —
 * that's almost always an authoring bug.
 */
function mergePlans(
  overrides?: readonly StaticPlan[],
): ReadonlyMap<TaskType, StaticPlan> {
  if (overrides === undefined || overrides.length === 0) {
    return STATIC_PLANS;
  }
  const out = new Map<TaskType, StaticPlan>(STATIC_PLANS);
  const seen = new Set<TaskType>();
  for (const plan of overrides) {
    if (!isTaskType(plan.taskType)) {
      throw new ValueError(
        `RLM plan has invalid taskType "${String(plan.taskType)}".`,
      );
    }
    if (seen.has(plan.taskType)) {
      throw new ValueError(
        `Duplicate RLM plan for taskType "${plan.taskType}" in options.plans.`,
      );
    }
    seen.add(plan.taskType);
    out.set(plan.taskType, plan);
  }
  return out;
}

/**
 * Coerce an arbitrary evaluator return value to the string the RLM
 * facade exposes. Strings pass through; objects/arrays are
 * `JSON.stringify`'d; primitives are `String()`'d. The evaluator only
 * legally returns strings for the default static plans, but the
 * facade still stringifies plain objects so a custom plan that returns,
 * e.g., a list is not coerced to `[object Object]`. `JSON.stringify`
 * throws on cycles — that surfaces as a hard error rather than a
 * misleading string.
 */
function coerceAnswer(value: CombinatorValue): string {
  if (typeof value === 'string') return value;
  if (value === undefined || value === null) return '';
  if (typeof value === 'object') {
    return JSON.stringify(value);
  }
  return String(value);
}

/**
 * Attach a plan's memory schema to an `EvaluationContext`.
 *
 * The shared counters (budget, trace, callsUsed, handlers) are
 * preserved so a beam-routed run bills against a single top-level
 * budget. Memory is forked per plan: the schema lands in
 * `memorySchema`, and `memoryCell` is seeded from
 * `initialMemoryState` so each plan starts with its own declared
 * initial values (or an empty state when the plan has no schema).
 *
 * The forked context shares `trace`, `callsUsed`, `handlers`, and
 * `scope` references with the parent — those are the cross-plan
 * accumulators — and replaces only the memory-related fields.
 */
function withPlanMemory(
  ctx: EvaluationContext,
  memorySchema: MemorySchema | null,
): EvaluationContext {
  if (memorySchema === null) {
    if (ctx.memorySchema === null) return ctx;
    return {
      ...ctx,
      memorySchema: null,
      memoryCell: { current: EMPTY_TYPED_MEMORY_STATE },
    };
  }
  return {
    ...ctx,
    memorySchema,
    memoryCell: { current: initialMemoryState(memorySchema) },
  };
}

function consumeOracleBudget(ctx: EvaluationContext): void {
  ctx.callsUsed.current += 1;
  if (ctx.callsUsed.current > ctx.budget.maxOracleCalls) {
    throw new BudgetError(
      `RLM oracle call budget exceeded (used=${ctx.callsUsed.current}, max=${ctx.budget.maxOracleCalls})`,
    );
  }
}

function reserveBeamPlanningBudget(
  budget: RLMBudget,
  branchCount: number,
): RLMBudget {
  if (branchCount <= 1) {
    return budget;
  }
  const remainingCalls = Math.max(1, budget.maxOracleCalls - branchCount);
  return Object.freeze({
    ...budget,
    maxOracleCalls: Math.max(1, Math.floor(remainingCalls / branchCount)),
  });
}

interface ExecutedRoute {
  readonly answer: string;
  readonly selectedTaskType: TaskType;
}

function buildRouteMetadata(
  route: RouteResult,
  selectedTaskType: TaskType,
): {
  readonly kind: RouteResult['kind'];
  readonly taskTypes: readonly TaskType[];
  readonly selectedTaskType: TaskType;
} {
  if (route.kind === 'single') {
    return {
      kind: 'single',
      taskTypes: [route.plan.taskType],
      selectedTaskType: route.plan.taskType,
    };
  }
  return {
    kind: 'beam',
    taskTypes: route.plans.map((p) => p.taskType),
    selectedTaskType,
  };
}

function normalizeRouteThreshold(value: number | undefined): number {
  if (value === undefined) {
    return 0.7;
  }
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new ValueError(
      `RLM options.routeThreshold must be a finite number in [0, 1], got ${String(value)}.`,
    );
  }
  return value;
}

function normalizeRouteBeamTopK(value: number | undefined): number {
  if (value === undefined) {
    return DEFAULT_BEAM_TOP_K;
  }
  if (!Number.isInteger(value) || value < 2) {
    throw new ValueError(
      `RLM options.routeBeamTopK must be an integer >= 2, got ${String(value)}.`,
    );
  }
  return value;
}

function flattenMessages(messages: readonly Message[]): string {
  return messages
    .map((message) => {
      const content = flattenMessageContent(message.content).trim();
      return content === ''
        ? `[${message.role.toUpperCase()}]`
        : `[${message.role.toUpperCase()}]\n${content}`;
    })
    .join('\n\n');
}

function flattenMessageContent(content: Message['content']): string {
  if (typeof content === 'string') {
    return content;
  }
  return content
    .map((part) => (part.type === 'text' ? (part.text ?? '') : ''))
    .join('\n');
}
