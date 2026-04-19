/**
 * RLM v2 combinator evaluator.
 *
 * Walks a `CombinatorNode` AST to a `CombinatorValue`, appending
 * `EvaluationTrace` after each node. `Map` / `Filter` / `Vote` / `Ensemble`
 * fan out under `budget.maxParallelism`; `Reduce` is sequential. Neural nodes
 * (`oracle`, `vote`, `ensemble`) call `Predict.acall`; each oracle slot
 * increments `ctx.callsUsed` before the LM call and throws `BudgetError` past
 * `maxOracleCalls`.
 *
 * Contract: `docs/product/rlm-v2-architecture.md` §4.1 and `spec/abstractions.md §0.5`.
 */

import type { BaseLM } from './lm.js';
import { Predict } from './predict.js';
import type { Prediction } from './prediction.js';
import {
  type Signature,
  signatureFromString,
  withInstructions,
} from './signature.js';
import { BudgetError, RuntimeError, ValueError } from './exceptions.js';
import type {
  CombinatorBinary,
  CombinatorNode,
  CombinatorValue,
  EnsembleReducer,
  VoteReducer,
} from './rlm_combinators.js';
import type {
  BuildEvaluationContextOptions,
  EffectHandler,
  EvaluationContext,
  EvaluationTrace,
} from './rlm_types.js';
import { mergeBudget } from './rlm_types.js';
import {
  EMPTY_TYPED_MEMORY_STATE,
  defaultMemoryInjector,
  type TypedMemoryState,
} from './rlm_memory.js';
import {
  EFFECT_ORACLE_SIGNATURE,
  appendEffectResult,
  builtInEffectHandlers,
  parseOracleResponse,
  type Effect,
  type EffectResult,
  type OracleResponse,
} from './rlm_effects.js';
import { typeName } from './rlm_util.js';

// ---------------------------------------------------------------------------
// Oracle signatures — module-scoped, built once.
// ---------------------------------------------------------------------------
//
// These module-scoped signatures (plus `EFFECT_ORACLE_SIGNATURE` in
// `rlm_effects.ts`) are the surfaces the combinator runtime asks of any
// wrapped `BaseLM`. Keeping them at module scope avoids the
// overhead of parsing the signature string on every oracle invocation while
// still letting the adapter pipeline do all the real JSON coercion.
//
// * `ORACLE_SIGNATURE` — plain `prompt -> answer` pipe for `vote` lanes,
//   `ensemble` (non-verifier), and `QueryOracle` delegation. Direct
//   `oracle` combinator leaves use `EFFECT_ORACLE_SIGNATURE` via the
//   effect loop (`callOracleLeafWithEffects`).
// * `CONFIDENCE_ORACLE_SIGNATURE` — used by `ensemble` with the
//   `confidence` reducer so each model can self-report a weighting signal.
// * `RLM_VERIFIER_SIGNATURE` — verifier probes for `vote` / `ensemble`
//   `verifier` reducers and for beam-route candidate selection in `RLM`.

const ORACLE_SIGNATURE: Signature = signatureFromString(
  'prompt: str -> answer: str',
);

const CONFIDENCE_ORACLE_SIGNATURE: Signature = signatureFromString(
  'prompt: str -> answer: str, confidence: float',
);

/** Shared with `RLM` beam verification — single parse site for this shape. */
export const RLM_VERIFIER_SIGNATURE: Signature = signatureFromString(
  'prompt: str, candidate: str -> verdict: bool',
);

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export type { BuildEvaluationContextOptions } from './rlm_types.js';

/**
 * Build a fresh `EvaluationContext` with all fields populated. The `trace`
 * and `callsUsed` objects are freshly created so a single builder does not
 * share mutable state across independent `aforward` invocations.
 *
 * The returned context has the four built-in effect handlers
 * (`ReadContext`, `WriteMemory`, `QueryOracle`, `Yield`) pre-registered;
 * any user-supplied handlers with matching names override the defaults.
 * User handlers with fresh names (including custom ones dispatched via
 * `Effect.kind === 'Custom'`) are merged alongside.
 */
export function buildEvaluationContext(
  options: BuildEvaluationContextOptions,
): EvaluationContext {
  const initialMemory: TypedMemoryState =
    options.memory ?? EMPTY_TYPED_MEMORY_STATE;
  return {
    budget: mergeBudget(options.budget),
    lm: options.lm,
    lmRegistry: options.lmRegistry ?? new Map<string, BaseLM>(),
    signature: options.signature,
    scope: options.scope ?? new Map<string, CombinatorValue>(),
    depth: 0,
    callsUsed: { current: 0 },
    trace: [],
    handlers: buildHandlerRegistry(options.handlers),
    memoryCell: { current: initialMemory },
    memorySchema: options.memorySchema ?? null,
  };
}

/**
 * Assemble the final handler registry: start from the built-in set,
 * then overlay any user-provided entries. Users always win on name
 * collisions — the defaults exist to give a useful runtime when no
 * handlers are supplied, not to block overrides.
 *
 * The `QueryOracle` default uses the plain oracle helper so nested
 * oracle calls never recursively enter another effect loop.
 */
function buildHandlerRegistry(
  userHandlers: ReadonlyMap<string, EffectHandler> | undefined,
): ReadonlyMap<string, EffectHandler> {
  const out = new Map<string, EffectHandler>();
  for (const handler of builtInEffectHandlers(callOracleLeafPlain)) {
    out.set(handler.name, handler);
  }
  if (userHandlers !== undefined) {
    for (const [name, handler] of userHandlers) {
      out.set(name, handler);
    }
  }
  return out;
}


/**
 * Evaluate a combinator plan under a context.
 *
 * The outer frame enforces `ctx.depth <= ctx.budget.maxDepth` and emits a
 * trace entry after `evaluateInner` returns or throws. Errors propagate:
 * `evaluate` does not swallow anything.
 *
 * Caller contract: the `trace` array and `callsUsed` counter inside `ctx`
 * are shared across the entire evaluation tree; do not mutate them from
 * outside the evaluator.
 */
export async function evaluate(
  node: CombinatorNode,
  ctx: EvaluationContext,
): Promise<CombinatorValue> {
  if (ctx.depth > ctx.budget.maxDepth) {
    throw new BudgetError(
      `RLM depth budget exceeded (depth=${ctx.depth}, max=${ctx.budget.maxDepth})`,
    );
  }
  const descended: EvaluationContext = { ...ctx, depth: ctx.depth + 1 };
  const startedAt = new Date().toISOString();
  const startNs = performance.now();
  try {
    const value = await evaluateInner(node, descended);
    pushTrace(ctx, node, startedAt, startNs, true);
    return value;
  } catch (cause) {
    // Record the failed node in the shared trace, then rethrow unchanged.
    pushTrace(ctx, node, startedAt, startNs, false, cause);
    throw cause;
  }
}

// ---------------------------------------------------------------------------
// Internal dispatch
// ---------------------------------------------------------------------------

async function evaluateInner(
  node: CombinatorNode,
  ctx: EvaluationContext,
): Promise<CombinatorValue> {
  switch (node.tag) {
    case 'literal':
      return node.value;

    case 'var': {
      const bound = ctx.scope.get(node.name);
      if (bound === undefined && !ctx.scope.has(node.name)) {
        throw new ValueError(
          `RLM: var '${node.name}' not bound in scope (known keys: ${keysOf(ctx.scope)})`,
        );
      }
      // scope.has() true implies the value was explicitly bound; the
      // `undefined` check above is sufficient for non-undefined values,
      // and `has()` handles the rare case where a caller bound `undefined`.
      return bound as CombinatorValue;
    }

    case 'split': {
      const inputValue = await evaluate(node.input, ctx);
      const kValue = await evaluate(node.k, ctx);
      const inputStr = requireString(inputValue, 'split input');
      const k = requireFiniteNumber(kValue, 'split k');
      if (k < 1) {
        throw new ValueError(`RLM: split k must be >= 1, got ${k}`);
      }
      return partitionString(inputStr, Math.floor(k));
    }

    case 'peek': {
      const inputValue = await evaluate(node.input, ctx);
      const startValue = await evaluate(node.start, ctx);
      const endValue = await evaluate(node.end, ctx);
      const inputStr = requireString(inputValue, 'peek input');
      const start = requireFiniteNumber(startValue, 'peek start');
      const end = requireFiniteNumber(endValue, 'peek end');
      const startIdx = Math.max(0, Math.floor(start));
      const endIdx = Math.min(inputStr.length, Math.floor(end));
      if (endIdx <= startIdx) {
        return '';
      }
      return inputStr.slice(startIdx, endIdx);
    }

    case 'map': {
      const itemsValue = await evaluate(node.items, ctx);
      const items = requireList(itemsValue, 'map items');
      return runBounded(items, ctx.budget.maxParallelism, async (item) =>
        evaluate(node.fn.body, withBinding(ctx, node.fn.param, item)),
      );
    }

    case 'filter': {
      const itemsValue = await evaluate(node.items, ctx);
      const items = requireList(itemsValue, 'filter items');
      const mask = await runBounded(
        items,
        ctx.budget.maxParallelism,
        async (item) => {
          const result = await evaluate(
            node.pred.body,
            withBinding(ctx, node.pred.param, item),
          );
          if (typeof result !== 'boolean') {
            throw new ValueError(
              `RLM: filter predicate must return boolean, got ${typeName(result)}`,
            );
          }
          return result;
        },
      );
      const kept: CombinatorValue[] = [];
      for (let i = 0; i < items.length; i += 1) {
        if (mask[i] === true) {
          kept.push(items[i] as CombinatorValue);
        }
      }
      return kept;
    }

    case 'reduce': {
      const itemsValue = await evaluate(node.items, ctx);
      const items = requireList(itemsValue, 'reduce items');
      return reduceSequential(items, node, ctx);
    }

    case 'concat': {
      const itemsValue = await evaluate(node.items, ctx);
      const items = requireList(itemsValue, 'concat items');
      let separator = '';
      if (node.separator !== undefined) {
        const sepValue = await evaluate(node.separator, ctx);
        separator = requireString(sepValue, 'concat separator');
      }
      const parts: string[] = [];
      for (let i = 0; i < items.length; i += 1) {
        const entry = items[i];
        if (typeof entry !== 'string') {
          throw new ValueError(
            `RLM: concat list must contain only strings at index ${i}, got ${typeName(entry)}`,
          );
        }
        parts.push(entry);
      }
      return parts.join(separator);
    }

    case 'cross': {
      const leftValue = await evaluate(node.left, ctx);
      const rightValue = await evaluate(node.right, ctx);
      const left = requireList(leftValue, 'cross left');
      const right = requireList(rightValue, 'cross right');
      const pairs: CombinatorValue[] = [];
      for (const a of left) {
        for (const b of right) {
          pairs.push([a, b]);
        }
      }
      return pairs;
    }

    case 'vote': {
      if (node.oracle.tag !== 'oracle') {
        throw new ValueError(
          `RLM: vote inner must be an oracle node, got '${node.oracle.tag}'`,
        );
      }
      const nValue = await evaluate(node.n, ctx);
      const n = requireFiniteNumber(nValue, 'vote n');
      if (n < 1) {
        throw new ValueError(`RLM: vote n must be >= 1, got ${n}`);
      }
      const rounded = Math.floor(n);
      const promptValue = await evaluate(node.oracle.prompt, ctx);
      const prompt = requireString(promptValue, 'vote oracle prompt');
      const modelHint = node.oracle.modelHint;
      const reducer: VoteReducer = node.reducer ?? 'majority';

      const indices = Array.from({ length: rounded }, (_, i) => i);
      const answers = await runBounded(
        indices,
        ctx.budget.maxParallelism,
        async () => callOracleLeafPlain(prompt, modelHint, ctx),
      );
      return reduceVote(answers, reducer, prompt, modelHint, ctx);
    }

    case 'ensemble': {
      if (node.oracle.tag !== 'oracle') {
        throw new ValueError(
          `RLM: ensemble inner must be an oracle node, got '${node.oracle.tag}'`,
        );
      }
      if (node.models.length === 0) {
        throw new ValueError('RLM: ensemble requires at least one model');
      }
      const promptValue = await evaluate(node.oracle.prompt, ctx);
      const prompt = requireString(promptValue, 'ensemble oracle prompt');
      const reducer: EnsembleReducer = node.reducer ?? 'confidence';
      const signature =
        reducer === 'confidence'
          ? CONFIDENCE_ORACLE_SIGNATURE
          : ORACLE_SIGNATURE;

      const responses = await runBounded(
        node.models,
        ctx.budget.maxParallelism,
        async (modelHint) =>
          invokePredict(signature, { prompt }, modelHint, ctx),
      );
      return reduceEnsemble(responses, reducer, prompt, ctx);
    }

    case 'oracle': {
      const promptValue = await evaluate(node.prompt, ctx);
      const prompt = requireString(promptValue, 'oracle prompt');
      return callOracleLeafWithEffects(prompt, node.modelHint, ctx);
    }

    default: {
      const _exhaustive: never = node;
      throw new RuntimeError(
        `RLM: unknown combinator node: ${JSON.stringify(_exhaustive)}`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Reduce helper — sequential accumulation
// ---------------------------------------------------------------------------

async function reduceSequential(
  items: ReadonlyArray<CombinatorValue>,
  node: Extract<CombinatorNode, { readonly tag: 'reduce' }>,
  ctx: EvaluationContext,
): Promise<CombinatorValue> {
  const op: CombinatorBinary = node.op;
  let acc: CombinatorValue;
  let cursor = 0;
  if (node.init !== undefined) {
    acc = await evaluate(node.init, ctx);
  } else if (items.length === 0) {
    throw new ValueError('RLM: reduce over empty list requires init');
  } else {
    acc = items[0] as CombinatorValue;
    cursor = 1;
  }
  for (let i = cursor; i < items.length; i += 1) {
    const element = items[i] as CombinatorValue;
    const stepCtx = withBinding(
      withBinding(ctx, op.left, acc),
      op.right,
      element,
    );
    acc = await evaluate(op.body, stepCtx);
  }
  return acc;
}

// ---------------------------------------------------------------------------
// Oracle invocation — network boundary
// ---------------------------------------------------------------------------
//
// The three helpers below are the sole point where the evaluator touches
// the LM. Every caller goes through `invokePredict`, which is the single
// place where the `maxOracleCalls` budget is checked and the
// `modelHint -> lmRegistry -> ctx.lm` fallback is applied.

/**
 * Resolve a model hint against the registry, falling back to `ctx.lm` when
 * absent or unknown (silent fallback so plans can name hints not wired in a
 * given deployment).
 */
function resolveOracleLm(
  modelHint: string | undefined,
  ctx: EvaluationContext,
): BaseLM {
  if (modelHint === undefined) return ctx.lm;
  const hit = ctx.lmRegistry.get(modelHint);
  return hit ?? ctx.lm;
}

/**
 * Shared network-boundary helper. Consumes one oracle-budget slot, resolves
 * the LM, constructs a fresh `Predict` instance for the supplied signature,
 * and awaits the async call. A fresh `Predict` per invocation is cheap
 * (sub-microsecond) and keeps the predictor stateless — we never want a
 * cross-call demo or trace accumulation to leak between oracle sites.
 *
 * When `ctx.memorySchema` is set, augments the signature instructions with
 * the memory banner each call so `WriteMemory` updates visible on the next
 * oracle turn.
 */
async function invokePredict(
  signature: Signature,
  kwargs: Record<string, unknown>,
  modelHint: string | undefined,
  ctx: EvaluationContext,
): Promise<Prediction<Record<string, unknown>>> {
  ctx.callsUsed.current += 1;
  if (ctx.callsUsed.current > ctx.budget.maxOracleCalls) {
    throw new BudgetError(
      `RLM oracle call budget exceeded (used=${ctx.callsUsed.current}, max=${ctx.budget.maxOracleCalls})`,
    );
  }
  const lm = resolveOracleLm(modelHint, ctx);
  const effectiveSignature = injectMemoryBanner(signature, ctx);
  const predictor = new Predict(effectiveSignature);
  return predictor.acall({ ...kwargs, lm });
}

/**
 * Append the rendered memory banner to the signature's instructions
 * when the plan declares a schema. Returns the input signature
 * untouched when no schema is attached — plans without memory pay no
 * per-call overhead beyond a single null check.
 *
 * The banner is placed after the existing instructions with a blank
 * line separator so the adapter's `formatTaskDescription` renders it
 * as a distinct block at the end of the system message. The LM's
 * natural reading order — objective, then context — is preserved.
 */
function injectMemoryBanner(
  signature: Signature,
  ctx: EvaluationContext,
): Signature {
  const schema = ctx.memorySchema;
  if (schema === null) return signature;
  const banner = defaultMemoryInjector.render(schema, ctx.memoryCell.current);
  if (banner === '') return signature;
  const base = signature.instructions.trim();
  const instructions = base === '' ? banner : `${base}\n\n${banner}`;
  return withInstructions(signature, instructions);
}

/**
 * Shape-checked plain-oracle call that returns the `answer` field as a
 * string. Used by:
 *
 * - `vote`'s per-lane sampling where the wrapper semantics of the
 *   effect loop would muddy the self-consistency signal.
 * - `QueryOracle` (as `callOracleFn`) since by definition that effect
 *   is a flat single-call delegation that must not recurse.
 *
 * All paths that want the full effect-loop behavior go through
 * `callOracleLeafWithEffects` below.
 */
async function callOracleLeafPlain(
  prompt: string,
  modelHint: string | undefined,
  ctx: EvaluationContext,
): Promise<string> {
  const prediction = await invokePredict(
    ORACLE_SIGNATURE,
    { prompt },
    modelHint,
    ctx,
  );
  const answer = prediction.get('answer');
  if (typeof answer !== 'string') {
    throw new ValueError(
      `RLM: oracle answer must be a string, got ${typeName(answer)}`,
    );
  }
  return answer;
}

/**
 * Drive the oracle effect loop.
 *
 * Each iteration:
 *
 * 1. Calls the LM with `EFFECT_ORACLE_SIGNATURE` and the current
 *    (possibly effect-augmented) prompt.
 * 2. Parses the prediction into an `OracleResponse`.
 *    - `kind: 'value'` — returns to the caller; nothing else traced.
 *    - `kind: 'effect'` — dispatches through the handler registry,
 *      appends the result to the prompt, and continues.
 * 3. Malformed effects or unknown handlers surface as
 *    `{ ok: false, error }` blocks appended to the prompt; the LM
 *    sees them next turn and can retry. Parse errors use the same
 *    channel.
 * 4. Handler exceptions propagate as `EffectResult` errors unless
 *    they are `BudgetError` — budget enforcement is non-recoverable
 *    by design.
 *
 * If the loop runs all `ctx.budget.maxEffectTurns` iterations without
 * a `value` response, a `BudgetError` is thrown. The partial trace
 * remains in `ctx.trace` for post-mortem inspection.
 *
 * Every effect-dispatch turn pushes a single `EvaluationTrace` entry
 * tagged `'effect'` with `extras = { turn, effectKind, handlerOk }`.
 * Value turns do not add an entry; the surrounding combinator node's
 * trace entry already reflects the successful leaf.
 */
async function callOracleLeafWithEffects(
  initialPrompt: string,
  modelHint: string | undefined,
  ctx: EvaluationContext,
): Promise<string> {
  const maxTurns = ctx.budget.maxEffectTurns;
  let prompt = initialPrompt;
  for (let turn = 0; turn < maxTurns; turn += 1) {
    const prediction = await invokePredict(
      EFFECT_ORACLE_SIGNATURE,
      { prompt },
      modelHint,
      ctx,
    );
    let response: OracleResponse;
    try {
      response = parseOracleResponse(prediction);
    } catch (err) {
      // LM output boundary: malformed oracle text becomes feedback for the next effect turn.
      const message = err instanceof Error ? err.message : String(err);
      prompt = appendRawError(prompt, message, turn);
      pushEffectTrace(ctx, turn, 'parse_error', false);
      continue;
    }
    if (response.kind === 'value') {
      return response.value;
    }
    const effect = response.effect;
    const handler = resolveEffectHandler(effect, ctx);
    const startedAt = new Date().toISOString();
    const startNs = performance.now();
    let result: EffectResult;
    if (handler === null) {
      result = {
        ok: false,
        error: `No handler registered for effect kind "${effect.kind}"${
          effect.kind === 'Custom' ? ` (name="${effect.name}")` : ''
        }`,
      };
    } else {
      try {
        result = await handler.handle(effect, ctx);
      } catch (err) {
        // Third-party handler boundary: never let handler throws abort the effect loop (budget still fatal).
        if (err instanceof BudgetError) throw err;
        const message = err instanceof Error ? err.message : String(err);
        result = {
          ok: false,
          error: `Handler "${handler.name}" threw: ${message}`,
        };
      }
    }
    pushEffectTrace(
      ctx,
      turn,
      effect.kind,
      result.ok,
      startedAt,
      startNs,
      handler?.name,
    );
    prompt = appendEffectResult(prompt, effect, result, turn);
  }
  throw new BudgetError(
    `RLM oracle exceeded maxEffectTurns=${maxTurns}`,
  );
}

/**
 * Resolve an effect to its handler. Custom effects look up by their
 * `name` first, then fall back to any user handler named `'Custom'`.
 * Non-custom effects look up strictly by `kind`.
 */
function resolveEffectHandler(
  effect: Effect,
  ctx: EvaluationContext,
): EffectHandler | null {
  if (effect.kind === 'Custom') {
    return (
      ctx.handlers.get(effect.name) ?? ctx.handlers.get('Custom') ?? null
    );
  }
  return ctx.handlers.get(effect.kind) ?? null;
}

function pushEffectTrace(
  ctx: EvaluationContext,
  turn: number,
  effectKind: string,
  ok: boolean,
  startedAt?: string,
  startNs?: number,
  handlerName?: string,
): void {
  const now = performance.now();
  const entry: EvaluationTrace = {
    step: ctx.trace.length,
    nodeTag: 'effect',
    startedAt: startedAt ?? new Date().toISOString(),
    durationMs: startNs === undefined ? 0 : now - startNs,
    ok,
    extras: Object.freeze({
      turn,
      effectKind,
      ...(handlerName !== undefined ? { handlerName } : {}),
    }),
  };
  ctx.trace.push(entry);
}

/**
 * Append a parse-error block to the next-turn prompt. Parse errors do
 * not carry an `effect` payload by definition, so `appendEffectResult`
 * is not reusable. The format mirrors the structured effect block so
 * downstream tooling can parse both with a single regex.
 */
function appendRawError(
  prompt: string,
  error: string,
  turn: number,
): string {
  const block = [
    `[[EFFECT turn=${turn} kind=parse_error]]`,
    'args: {}',
    `error: ${error}`,
    '[[/EFFECT]]',
  ].join('\n');
  return prompt.length === 0 ? block : `${prompt}\n\n${block}`;
}

// ---------------------------------------------------------------------------
// Reducers — shared between vote and ensemble
// ---------------------------------------------------------------------------

/**
 * Vote reducer.
 *
 * - `majority` / `mode`: textual equality mode. In the rare tie case we
 *   return the answer that appeared first (stable ordering is important
 *   for replay determinism).
 * - `verifier`: each candidate is cross-checked with a `verdict: bool`
 *   signature under the same `modelHint`. The first candidate that earns
 *   a `true` verdict wins; if every candidate is rejected we fall back to
 *   the first candidate (the alternative — throwing — is caller-hostile
 *   because the oracle already returned plausible answers and the
 *   verifier is a soft signal).
 */
async function reduceVote(
  answers: readonly string[],
  reducer: VoteReducer,
  prompt: string,
  modelHint: string | undefined,
  ctx: EvaluationContext,
): Promise<string> {
  if (answers.length === 0) {
    throw new ValueError('RLM: vote produced zero oracle answers');
  }
  if (reducer === 'majority' || reducer === 'mode') {
    return modeOfStrings(answers);
  }
  return verifierSelect(answers, prompt, modelHint, ctx);
}

/**
 * Ensemble reducer.
 *
 * - `majority`: plurality vote across per-model answers.
 * - `confidence`: each model also emits a `confidence: float`; we pick
 *   the answer with the strictly-highest confidence. Ties default to the
 *   first response in registration order for determinism. A non-finite or
 *   non-numeric confidence raises `ValueError` — this is louder than
 *   silently substituting zero because the contract is a typed field, not
 *   a free-form note.
 * - `verifier`: same pool-of-verifiers flow as the vote reducer, but
 *   starting from the per-model answers with no assumption about model
 *   hint (verifier probes use the default LM).
 */
async function reduceEnsemble(
  responses: ReadonlyArray<Prediction<Record<string, unknown>>>,
  reducer: EnsembleReducer,
  prompt: string,
  ctx: EvaluationContext,
): Promise<string> {
  if (responses.length === 0) {
    throw new ValueError('RLM: ensemble produced zero model responses');
  }
  const answers = responses.map((r, idx) =>
    requireStringField(r.get('answer'), `ensemble response[${idx}].answer`),
  );
  if (reducer === 'majority') {
    return modeOfStrings(answers);
  }
  if (reducer === 'confidence') {
    let bestIdx = 0;
    let bestConf = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < responses.length; i += 1) {
      const raw = responses[i]!.get('confidence');
      const conf = typeof raw === 'number' ? raw : Number.NaN;
      if (!Number.isFinite(conf)) {
        throw new ValueError(
          `RLM: ensemble response[${i}].confidence must be a finite number, got ${typeName(raw)}`,
        );
      }
      if (conf > bestConf) {
        bestConf = conf;
        bestIdx = i;
      }
    }
    return answers[bestIdx] as string;
  }
  return verifierSelect(answers, prompt, undefined, ctx);
}

/**
 * Run each candidate through the verifier signature in parallel (bounded
 * by `ctx.budget.maxParallelism`) and return the first candidate with a
 * `verdict: true`. Falls back to `candidates[0]` if none succeed, so
 * callers always receive an answer.
 */
async function verifierSelect(
  candidates: readonly string[],
  prompt: string,
  modelHint: string | undefined,
  ctx: EvaluationContext,
): Promise<string> {
  const verdicts = await runBounded(
    candidates,
    ctx.budget.maxParallelism,
    async (candidate, index) => {
      const prediction = await invokePredict(
        RLM_VERIFIER_SIGNATURE,
        { prompt, candidate },
        modelHint,
        ctx,
      );
      const verdict = prediction.get('verdict');
      if (typeof verdict !== 'boolean') {
        throw new ValueError(
          `RLM: verifier[${index}].verdict must be a boolean, got ${typeName(verdict)}`,
        );
      }
      return verdict;
    },
  );
  for (let i = 0; i < candidates.length; i += 1) {
    if (verdicts[i] === true) return candidates[i] as string;
  }
  return candidates[0] as string;
}

/**
 * Plurality vote over a non-empty list of strings. Stable under ties —
 * the first element in input order wins. The two-pass structure (tally,
 * then scan for the input-order argmax) is deliberate: a single-pass
 * incremental tally picks "first to reach max count", which is a subtly
 * different tie-break than input-order and is harder to reason about.
 * Kept separate from `reduceVote` so the rule is a small, testable
 * surface.
 */
function modeOfStrings(items: readonly string[]): string {
  if (items.length === 0) {
    throw new ValueError('RLM: mode of empty list is undefined');
  }
  const counts = new Map<string, number>();
  for (const item of items) {
    counts.set(item, (counts.get(item) ?? 0) + 1);
  }
  let bestKey = items[0] as string;
  let bestCount = counts.get(bestKey) ?? 0;
  for (const item of items) {
    const c = counts.get(item) ?? 0;
    if (c > bestCount) {
      bestCount = c;
      bestKey = item;
    }
  }
  return bestKey;
}

function requireStringField(value: unknown, what: string): string {
  if (typeof value !== 'string') {
    throw new ValueError(
      `RLM: ${what} must be a string, got ${typeName(value)}`,
    );
  }
  return value;
}

// ---------------------------------------------------------------------------
// Scope / parallelism / trace utilities
// ---------------------------------------------------------------------------

function withBinding(
  ctx: EvaluationContext,
  name: string,
  value: CombinatorValue,
): EvaluationContext {
  const nextScope = new Map(ctx.scope);
  nextScope.set(name, value);
  return { ...ctx, scope: nextScope };
}

/**
 * Bounded parallel map. Runs at most `maxParallelism` mappers concurrently
 * using a shared cursor; preserves input order in the output array.
 *
 * The worker loop is cooperatively scheduled: each iteration synchronously
 * claims a slot (single-threaded JS guarantees no races on `cursor`) and
 * then awaits the mapper.
 */
async function runBounded<T, U>(
  items: ReadonlyArray<T>,
  maxParallelism: number,
  mapper: (item: T, index: number) => Promise<U>,
): Promise<U[]> {
  if (items.length === 0) return [];
  if (!Number.isFinite(maxParallelism) || maxParallelism <= 0) {
    throw new ValueError(
      `RLM: maxParallelism must be a positive finite number, got ${String(maxParallelism)}`,
    );
  }
  const results: U[] = new Array(items.length);
  let cursor = 0;
  const lane = async (): Promise<void> => {
    while (true) {
      const slot = cursor;
      cursor += 1;
      if (slot >= items.length) return;
      const item = items[slot] as T;
      results[slot] = await mapper(item, slot);
    }
  };
  const workerCount = Math.min(Math.floor(maxParallelism), items.length);
  const workers: Promise<void>[] = [];
  for (let i = 0; i < workerCount; i += 1) {
    workers.push(lane());
  }
  await Promise.all(workers);
  return results;
}

function pushTrace(
  ctx: EvaluationContext,
  node: CombinatorNode,
  startedAt: string,
  startNs: number,
  ok: boolean,
  cause?: unknown,
): void {
  const entry: EvaluationTrace =
    ok
      ? {
          step: ctx.trace.length,
          nodeTag: node.tag,
          startedAt,
          durationMs: performance.now() - startNs,
          ok: true,
        }
      : {
          step: ctx.trace.length,
          nodeTag: node.tag,
          startedAt,
          durationMs: performance.now() - startNs,
          ok: false,
          cause,
        };
  ctx.trace.push(entry);
}

// ---------------------------------------------------------------------------
// Runtime type guards
// ---------------------------------------------------------------------------

function requireString(value: CombinatorValue, what: string): string {
  if (typeof value !== 'string') {
    throw new ValueError(`RLM: ${what} must be a string, got ${typeName(value)}`);
  }
  return value;
}

function requireFiniteNumber(value: CombinatorValue, what: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValueError(
      `RLM: ${what} must be a finite number, got ${typeName(value)}`,
    );
  }
  return value;
}

function requireList(value: CombinatorValue, what: string): ReadonlyArray<CombinatorValue> {
  if (!Array.isArray(value)) {
    throw new ValueError(`RLM: ${what} must be a list, got ${typeName(value)}`);
  }
  return value as ReadonlyArray<CombinatorValue>;
}

function keysOf(scope: ReadonlyMap<string, CombinatorValue>): string {
  const keys = [...scope.keys()];
  if (keys.length === 0) return '<empty scope>';
  return keys.map((k) => `'${k}'`).join(', ');
}

// ---------------------------------------------------------------------------
// String partitioner
// ---------------------------------------------------------------------------

/**
 * Partition `input` into `k` contiguous chunks. Chunks are as uniform as
 * possible: `k - (input.length mod k)` chunks are `⌊input.length / k⌋`
 * long and the remaining chunks are `⌈input.length / k⌉` long. Empty
 * strings yield an empty array; requests for more chunks than characters
 * yield one-character chunks followed by empty-string chunks are **not**
 * produced — the result length is `min(k, input.length)` when non-empty.
 */
function partitionString(input: string, k: number): readonly string[] {
  if (k <= 0) {
    throw new ValueError(`RLM: split k must be positive, got ${k}`);
  }
  if (input.length === 0) return [];
  const effective = Math.min(k, input.length);
  const base = Math.floor(input.length / effective);
  const remainder = input.length % effective;
  const chunks: string[] = [];
  let cursor = 0;
  for (let i = 0; i < effective; i += 1) {
    const size = base + (i < remainder ? 1 : 0);
    chunks.push(input.slice(cursor, cursor + size));
    cursor += size;
  }
  return chunks;
}

// ---------------------------------------------------------------------------
// Private helpers: exported for testing only via `__internal`
// ---------------------------------------------------------------------------

/**
 * Internal helpers exposed for direct unit-testing. Not part of the public
 * API — not re-exported from `src/index.ts`. Callers outside of the test
 * suite should use `evaluate()` and `buildEvaluationContext()` exclusively.
 */
export const __internal = Object.freeze({
  partitionString,
  runBounded,
  mergeBudget,
  modeOfStrings,
  resolveOracleLm,
});
