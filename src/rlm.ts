/**
 * Recursive Language Model (RLM) scaffold.
 */

import { coerceFieldValue } from './codec.js';
import { BudgetError, ConfigurationError, RuntimeError, ValueError } from './exceptions.js';
import { createField } from './field.js';
import { isPlainObject } from './guards.js';
import { BaseLM } from './lm.js';
import { Module } from './module.js';
import { createNodeCodeInterpreter, type SyncCodeInterpreter, type SyncCodeSession } from './node_code_interpreter.js';
import { Prediction } from './prediction.js';
import { Predict } from './predict.js';
import { settings } from './settings.js';
import { createSignature, ensureSignature, type Signature } from './signature.js';
import type { InferInputs, InferOutputs, SignatureInput } from './signature_types.js';
import { Tool, type ToolInput } from './tool.js';
import type {
  BudgetVector,
  CodeInterpreter,
  CodeInterpreterError,
  ExecuteResult,
  REPLHistory,
  REPLEntry,
  REPLVariable,
  RLMConfig,
} from './rlm_types.js';

const EMPTY_RECORD = Object.freeze({}) as Record<string, unknown>;
const DEFAULT_MAX_ITERATIONS = 20;
const DEFAULT_MAX_LLM_CALLS = 50;
const DEFAULT_MAX_BATCH_WIDTH = 8;
const DEFAULT_RESERVED_TOOL_NAMES = Object.freeze([
  'llmQuery',
  'llmQueryBatched',
  'SUBMIT',
  'print',
]);
const IDENTIFIER_PATTERN = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

interface NormalizedRLMTool {
  readonly name: string;
  readonly invoke: (...args: readonly unknown[]) => unknown;
}

export interface RLMOptions {
  readonly budget?: Partial<BudgetVector>;
  readonly trackTrace?: boolean;
  readonly tools?: readonly ToolInput[];
  readonly subLm?: BaseLM | null;
  readonly interpreter?: CodeInterpreter<Record<string, unknown>, unknown>;
  readonly reservedToolNames?: readonly string[];
}

function ensurePositiveInteger(name: string, value: unknown): number {
  if (!Number.isInteger(value) || (value as number) < 1) {
    throw new ValueError(`${name} must be a positive integer.`);
  }
  return value as number;
}

function buildBudgetVector(budget: Partial<BudgetVector> | undefined): BudgetVector {
  return Object.freeze({
    maxIterations: ensurePositiveInteger(
      'RLM budget.maxIterations',
      budget?.maxIterations ?? DEFAULT_MAX_ITERATIONS,
    ),
    maxLlmCalls: ensurePositiveInteger(
      'RLM budget.maxLlmCalls',
      budget?.maxLlmCalls ?? DEFAULT_MAX_LLM_CALLS,
    ),
    maxBatchWidth: ensurePositiveInteger(
      'RLM budget.maxBatchWidth',
      budget?.maxBatchWidth ?? DEFAULT_MAX_BATCH_WIDTH,
    ),
  });
}

function stripCodeFences(source: string): string {
  const trimmed = source.trim();
  if (!trimmed.includes('```')) {
    return trimmed;
  }

  const fenceMatch = trimmed.match(/^```(?:python|py|javascript|js|typescript|ts)?\s*\n([\s\S]*?)\n```$/i);
  if (fenceMatch) {
    return fenceMatch[1]!.trim();
  }
  return trimmed;
}

function formatVariables(variables: readonly REPLVariable[]): string {
  if (variables.length === 0) {
    return 'No variables are currently bound in the runtime session.';
  }

  return variables.map((variable) => {
    const typeName = variable.value === null
      ? 'null'
      : Array.isArray(variable.value)
        ? 'array'
        : typeof variable.value;
    return `${variable.symbol} (${typeName}, mutable=${String(variable.mutable)})`;
  }).join('\n');
}

function formatHistory(history: REPLHistory): string {
  if (history.entries.length === 0) {
    return 'No interpreter steps have executed yet.';
  }

  return history.entries.map((entry) => (
    `step=${entry.step} kind=${entry.kind} ok=${String(entry.ok)}`
  )).join('\n');
}

function createActionSignature(maxLlmCalls: number) {
  return createSignature(
    new Map([
      ['variables_info', createField({ kind: 'input', name: 'variables_info', typeTag: 'str', isTypeUndefined: false })],
      ['repl_history', createField({ kind: 'input', name: 'repl_history', typeTag: 'str', isTypeUndefined: false })],
      ['iteration', createField({ kind: 'input', name: 'iteration', typeTag: 'str', isTypeUndefined: false })],
    ]),
    new Map([
      ['reasoning', createField({ kind: 'output', name: 'reasoning', typeTag: 'str', isTypeUndefined: false })],
      ['code', createField({ kind: 'output', name: 'code', typeTag: 'str', isTypeUndefined: false })],
    ]),
    {
      instructions: [
        'You are controlling a persistent JavaScript REPL runtime.',
        'Write JavaScript code in `code` that advances the task.',
        `You may call llmQuery(prompt) or llmQueryBatched(prompts) up to ${maxLlmCalls} total sub-queries.`,
        'When final outputs are ready, call SUBMIT({...}) with all required output fields.',
      ].join('\n'),
    },
  );
}

function createExtractSignature(signature: ReturnType<typeof ensureSignature>) {
  return createSignature(
    new Map([
      ['variables_info', createField({ kind: 'input', name: 'variables_info', typeTag: 'str', isTypeUndefined: false })],
      ['repl_history', createField({ kind: 'input', name: 'repl_history', typeTag: 'str', isTypeUndefined: false })],
    ]),
    new Map(signature.outputFields),
    {
      instructions: [
        signature.instructions,
        'The runtime loop ended without SUBMIT.',
        'Extract the final structured outputs from runtime state and history.',
      ].join('\n\n'),
    },
  );
}

function normalizeTools(
  tools: readonly ToolInput[],
  reservedNames: ReadonlySet<string>,
): ReadonlyMap<string, NormalizedRLMTool> {
  const normalized = new Map<string, NormalizedRLMTool>();

  for (const candidate of tools) {
    const name = candidate.name;
    if (name.trim() === '') {
      throw new ValueError('RLM tools must have a non-empty function name.');
    }
    if (!IDENTIFIER_PATTERN.test(name)) {
      throw new ValueError(`Invalid RLM tool name "${name}".`);
    }
    if (reservedNames.has(name)) {
      throw new ValueError(`RLM tool name "${name}" is reserved by the runtime.`);
    }
    if (normalized.has(name)) {
      throw new ValueError(`Duplicate RLM tool name "${name}".`);
    }

    const invoke = (...args: readonly unknown[]): unknown => {
      const callable = candidate instanceof Tool ? candidate.func : candidate;
      const result = Reflect.apply(callable as Function, undefined, args);
      if (result instanceof Promise) {
        throw new ValueError(`RLM tool "${name}" returned a Promise; async tools are not supported in sync runtime mode.`);
      }
      return result;
    };

    normalized.set(name, Object.freeze({ name, invoke }));
  }

  return normalized;
}

function lmOutputToText(output: unknown): string {
  if (typeof output === 'string') {
    return output;
  }
  if (
    output
    && typeof output === 'object'
    && 'text' in output
    && typeof (output as { readonly text?: unknown }).text === 'string'
  ) {
    return (output as { readonly text: string }).text;
  }
  return String(output ?? '');
}

function appendHistory(
  history: REPLHistory,
  delta: readonly REPLEntry[],
  liveSymbols: readonly string[],
): REPLHistory {
  return Object.freeze({
    entries: Object.freeze([...history.entries, ...delta]),
    liveSymbols: Object.freeze([...liveSymbols]),
  });
}

/**
 * Degraded fallback when inspectGlobals fails on a faulted session.
 * Reconstructs variable symbols from history but values are lost — the extract
 * LLM sees names and types only, not contents. This is intentionally lossy:
 * a faulted session cannot be inspected, and partial symbol info is more useful
 * to the extract predictor than an empty variable list.
 */
function fallbackVariablesFromHistory(history: REPLHistory): readonly REPLVariable[] {
  return Object.freeze(history.liveSymbols.map((symbol) => Object.freeze({
    symbol,
    value: null,
    mutable: true,
  })));
}

function isSyncInterpreter(
  interpreter: CodeInterpreter<Record<string, unknown>, unknown>,
): interpreter is SyncCodeInterpreter {
  return typeof (interpreter as SyncCodeInterpreter).createSessionSync === 'function';
}

function isSyncSession(session: unknown): session is SyncCodeSession {
  return typeof (session as SyncCodeSession).executeSync === 'function'
    && typeof (session as SyncCodeSession).inspectGlobalsSync === 'function';
}

export class RLM<
  TSig extends SignatureInput = Signature,
  TInputs extends Record<string, unknown> = InferInputs<TSig>,
  TOutputs extends Record<string, unknown> = InferOutputs<TSig>,
> extends Module<TInputs, TOutputs> {
  readonly signature: ReturnType<typeof ensureSignature>;
  readonly config: RLMConfig;
  readonly generateAction: Predict;
  readonly extract: Predict;
  readonly tools: ReadonlyMap<string, NormalizedRLMTool>;
  readonly interpreter: CodeInterpreter<Record<string, unknown>, unknown>;

  subLm: BaseLM | null;

  constructor(signature: TSig, options: RLMOptions = {}) {
    super();
    this.signature = ensureSignature(signature);
    const budget = buildBudgetVector(options.budget);
    const reservedNames = new Set<string>([
      ...DEFAULT_RESERVED_TOOL_NAMES,
      ...(options.reservedToolNames ?? []),
    ]);

    this.config = Object.freeze({
      budget,
      trackTrace: options.trackTrace ?? true,
      reservedToolNames: Object.freeze([...reservedNames]),
      subLmResolution: 'instance_then_settings',
    });
    this.subLm = options.subLm ?? null;
    this.tools = normalizeTools(options.tools ?? [], reservedNames);
    this.interpreter = options.interpreter ?? createNodeCodeInterpreter();
    this.generateAction = new Predict(createActionSignature(this.config.budget.maxLlmCalls));
    this.extract = new Predict(createExtractSignature(this.signature));
  }

  override forward(kwargs: TInputs = EMPTY_RECORD as TInputs): Prediction<TOutputs> {
    if (!isPlainObject(kwargs)) {
      throw new ValueError('RLM expects a single plain-object argument.');
    }
    if (!isSyncInterpreter(this.interpreter)) {
      throw new RuntimeError('RLM.forward requires a synchronous interpreter session.');
    }

    const session = this.interpreter.createSessionSync();
    if (!isSyncSession(session)) {
      throw new RuntimeError('RLM.forward requires a synchronous interpreter session implementation.');
    }

    return this.runWithSyncSession(session, kwargs as Record<string, unknown>);
  }

  override async aforward(kwargs: TInputs = EMPTY_RECORD as TInputs): Promise<Prediction<TOutputs>> {
    if (!isPlainObject(kwargs)) {
      throw new ValueError('RLM expects a single plain-object argument.');
    }

    const session = await this.interpreter.createSession();
    return this.runWithAsyncSession(session, kwargs as Record<string, unknown>);
  }

  private createEmptyLoopState(): {
    history: REPLHistory;
    fault: CodeInterpreterError | null;
    lastVariables: readonly REPLVariable[];
  } {
    return {
      history: Object.freeze({ entries: Object.freeze([]), liveSymbols: Object.freeze([]) }),
      fault: null,
      lastVariables: Object.freeze([]),
    };
  }

  private buildStepInputs(
    variables: readonly REPLVariable[],
    history: REPLHistory,
    step: number,
  ): Record<string, unknown> {
    return {
      variables_info: formatVariables(variables),
      repl_history: formatHistory(history),
      iteration: `${step}/${this.config.budget.maxIterations}`,
    };
  }

  private processStepResult(
    result: ExecuteResult<unknown>,
    history: REPLHistory,
    variables: readonly REPLVariable[],
    fault: CodeInterpreterError | null,
  ): {
    history: REPLHistory;
    fault: CodeInterpreterError | null;
    prediction: Prediction<TOutputs> | null;
    variables: readonly REPLVariable[];
  } {
    const symbols = variables.map((variable) => variable.symbol);
    const updated = appendHistory(history, result.historyDelta, symbols);

    if (result.tag === 'fault') {
      return {
        history: updated,
        fault: result.error,
        prediction: null,
        variables,
      };
    }

    if (result.tag === 'submit') {
      return {
        history: updated,
        fault,
        prediction: this.finalizePrediction(result, updated, fault),
        variables,
      };
    }

    return { history: updated, fault, prediction: null, variables };
  }

  private completeExtractFallback(
    variables: readonly REPLVariable[],
    extracted: Prediction,
    history: REPLHistory,
    fault: CodeInterpreterError | null,
  ): Prediction<TOutputs> {
    const outputs = this.normalizeExtractedOutput(extracted);
    const historyWithExtract = appendHistory(
      history,
      Object.freeze([{ step: this.config.budget.maxIterations + 1, kind: 'extract' as const, ok: true }]),
      variables.map((variable) => variable.symbol),
    );
    return this.finalizeExtractPrediction(outputs, historyWithExtract, fault);
  }

  private buildExtractInputs(
    variables: readonly REPLVariable[],
    history: REPLHistory,
  ): Record<string, unknown> {
    return {
      variables_info: formatVariables(variables),
      repl_history: formatHistory(history),
    };
  }

  /**
   * Sync vs async entrypoints share the same conceptual loop (`buildStepInputs`,
   * `processStepResult`, extract fallback). Keep the two paths aligned so
   * `forward` and `aforward` differ only in session I/O, not control flow.
   */

  // ── Sync path ───────────────────────────────────────────────────────

  private runWithSyncSession(
    session: SyncCodeSession,
    kwargs: Record<string, unknown>,
  ): Prediction<TOutputs> {
    this.validateInputs(kwargs);
    const state = this.createEmptyLoopState();

    try {
      session.patchGlobalsSync({ bindings: this.buildExecutionBindings(kwargs) });

      for (let step = 1; step <= this.config.budget.maxIterations; step += 1) {
        const variables = session.inspectGlobalsSync();
        const action = this.generateAction.forward(this.buildStepInputs(variables, state.history, step));
        const result = session.executeSync({
          step, source: this.extractCode(action), budget: this.config.budget, allowSubmit: true,
        });
        const stepVariables = result.tag === 'fault'
          ? result.liveVariables
          : session.inspectGlobalsSync();
        const outcome = this.processStepResult(result, state.history, stepVariables, state.fault);
        state.history = outcome.history;
        state.fault = outcome.fault;
        state.lastVariables = outcome.variables;
        if (outcome.prediction) return outcome.prediction;
        if (outcome.fault) break;
      }

      let variables: readonly REPLVariable[];
      try { variables = session.inspectGlobalsSync(); }
      catch {
        variables = state.lastVariables.length > 0
          ? state.lastVariables
          : fallbackVariablesFromHistory(state.history);
      }
      const extracted = this.extract.forward(this.buildExtractInputs(variables, state.history));
      return this.completeExtractFallback(variables, extracted, state.history, state.fault);
    } finally {
      session.closeSync();
    }
  }

  // ── Async path ──────────────────────────────────────────────────────

  private async runWithAsyncSession(
    session: {
      readonly execute: SyncCodeSession['execute'];
      readonly inspectGlobals: SyncCodeSession['inspectGlobals'];
      readonly patchGlobals: SyncCodeSession['patchGlobals'];
      readonly close: SyncCodeSession['close'];
    },
    kwargs: Record<string, unknown>,
  ): Promise<Prediction<TOutputs>> {
    this.validateInputs(kwargs);
    const state = this.createEmptyLoopState();

    try {
      await session.patchGlobals({ bindings: this.buildExecutionBindings(kwargs) });

      for (let step = 1; step <= this.config.budget.maxIterations; step += 1) {
        const variables = await session.inspectGlobals();
        const action = await this.generateAction.acall(this.buildStepInputs(variables, state.history, step));
        const result = await session.execute({
          step, source: this.extractCode(action), budget: this.config.budget, allowSubmit: true,
        });
        const stepVariables = result.tag === 'fault'
          ? result.liveVariables
          : await session.inspectGlobals();
        const outcome = this.processStepResult(result, state.history, stepVariables, state.fault);
        state.history = outcome.history;
        state.fault = outcome.fault;
        state.lastVariables = outcome.variables;
        if (outcome.prediction) return outcome.prediction;
        if (outcome.fault) break;
      }

      let variables: readonly REPLVariable[];
      try { variables = await session.inspectGlobals(); }
      catch {
        variables = state.lastVariables.length > 0
          ? state.lastVariables
          : fallbackVariablesFromHistory(state.history);
      }
      const extracted = await this.extract.acall(this.buildExtractInputs(variables, state.history));
      return this.completeExtractFallback(variables, extracted, state.history, state.fault);
    } finally {
      await session.close();
    }
  }

  private validateInputs(kwargs: Record<string, unknown>): void {
    const missing = [...this.signature.inputFields.keys()].filter((name) => !(name in kwargs));
    if (missing.length > 0) {
      throw new ValueError(`Missing required RLM inputs: ${missing.join(', ')}`);
    }
  }

  private buildExecutionBindings(kwargs: Record<string, unknown>): Record<string, unknown> {
    const bindings: Record<string, unknown> = {
      ...kwargs,
      print: (...parts: readonly unknown[]) => {
        console.log(...parts);
      },
      llmQuery: (prompt: string) => this.querySubLm(prompt),
      llmQueryBatched: (prompts: readonly string[]) => this.querySubLmBatched(prompts),
    };

    for (const tool of this.tools.values()) {
      bindings[tool.name] = (...args: readonly unknown[]) => tool.invoke(...args);
    }

    return bindings;
  }

  private querySubLm(prompt: string): string {
    if (typeof prompt !== 'string' || prompt.trim() === '') {
      throw new ValueError('llmQuery(prompt) requires a non-empty prompt string.');
    }

    const lm = this.resolveSubLm();
    const outputs = lm.call(prompt);
    const first = outputs[0];
    return lmOutputToText(first);
  }

  private querySubLmBatched(prompts: readonly string[]): readonly string[] {
    if (!Array.isArray(prompts)) {
      throw new ValueError('llmQueryBatched(prompts) expects an array of strings.');
    }
    if (prompts.length > this.config.budget.maxBatchWidth) {
      throw new BudgetError(
        `llmQueryBatched batch width exceeded (${prompts.length} > ${this.config.budget.maxBatchWidth}).`,
      );
    }

    const results: string[] = [];
    for (const [index, prompt] of prompts.entries()) {
      try {
        results.push(this.querySubLm(prompt));
      } catch (error) {
        throw new RuntimeError(
          `llmQueryBatched aborted at index ${index}: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    }
    return Object.freeze(results);
  }

  private resolveSubLm(): BaseLM {
    const candidate = this.subLm ?? settings.lm;
    if (!(candidate instanceof BaseLM)) {
      throw new ConfigurationError(
        'RLM sub-LM resolution failed. Configure settings.lm or pass options.subLm.',
      );
    }
    return candidate;
  }

  private extractCode(action: Prediction): string {
    const code = action.get('code');
    if (typeof code !== 'string') {
      throw new ValueError('RLM action predictor must emit a string code field.');
    }
    return stripCodeFences(code);
  }

  private normalizeSubmittedOutput(value: unknown): Record<string, unknown> {
    // Intentionally NOT using isPlainObject() here: objects created inside the
    // Node vm context have a different Object.prototype than the host realm,
    // so prototype-based checks reject them. typeof+null+Array is cross-realm safe.
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      throw new ValueError('SUBMIT payload must be a plain object with output fields.');
    }

    const payload = value as Record<string, unknown>;
    const output: Record<string, unknown> = {};
    for (const [name, field] of this.signature.outputFields) {
      if (!(name in payload)) {
        throw new ValueError(`SUBMIT payload is missing required output field "${name}".`);
      }
      output[name] = coerceFieldValue(field.typeTag, payload[name]);
    }
    return output;
  }

  private normalizeExtractedOutput(prediction: Prediction): Record<string, unknown> {
    const output: Record<string, unknown> = {};
    for (const [name, field] of this.signature.outputFields) {
      output[name] = coerceFieldValue(field.typeTag, prediction.get(name));
    }
    return output;
  }

  private finalizePrediction(
    result: Extract<ExecuteResult<unknown>, { readonly tag: 'submit' }>,
    history: REPLHistory,
    fault: CodeInterpreterError | null,
  ): Prediction<TOutputs> {
    const outputs = this.normalizeSubmittedOutput(result.output.value);
    const payload: Record<string, unknown> = {
      ...outputs,
      final_via: result.output.via,
    };

    if (this.config.trackTrace) {
      payload.repl_history = history;
      payload.repl_error = fault;
    }

    return Prediction.create<TOutputs>(payload);
  }

  private finalizeExtractPrediction(
    outputs: Record<string, unknown>,
    history: REPLHistory,
    fault: CodeInterpreterError | null,
  ): Prediction<TOutputs> {
    const payload: Record<string, unknown> = {
      ...outputs,
      final_via: 'extract',
    };
    if (this.config.trackTrace) {
      payload.repl_history = history;
      payload.repl_error = fault;
    }
    return Prediction.create<TOutputs>(payload);
  }
}
