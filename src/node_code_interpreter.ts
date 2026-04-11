/**
 * Node-native implementation of the RLM interpreter contract.
 *
 * The runtime boundary lives in this module. RLM itself depends only on
 * CodeInterpreter / CodeSession interfaces.
 */

import * as vm from 'node:vm';
import { BudgetError, ValueError } from './exceptions.js';
import type {
  CodeInterpreter,
  CodeInterpreterError,
  CodeSession,
  ExecuteRequest,
  ExecuteResult,
  InterpreterPatch,
  REPLEntry,
  REPLVariable,
} from './rlm_types.js';

const INTERNAL_BINDINGS = new Set(['console', 'SUBMIT']);
const EMPTY_HISTORY_DELTA = Object.freeze([] as REPLEntry[]);
const EMPTY_VARIABLES = Object.freeze([] as REPLVariable[]);
const SIMPLE_VARIABLE_DECL_RE = /\b(?<kind>const|let|var)\s+(?<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*(?==|,|;)/g;
const FUNCTION_DECL_RE = /\bfunction\s+(?<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\(/g;
const CLASS_DECL_RE = /\bclass\s+(?<name>[A-Za-z_$][A-Za-z0-9_$]*)\b/g;
const MISSING_BINDING = Symbol('dspy.node_code_interpreter.missing_binding');

class SubmitSignal extends Error {
  constructor() {
    super('SUBMIT invoked');
    this.name = 'SubmitSignal';
  }
}

const MAX_CAUSE_DEPTH = 8;

function serializeCause(
  cause: unknown,
  seen: Set<object> = new Set(),
  depth = 0,
): unknown {
  if (cause === null || cause === undefined) {
    return cause;
  }

  if (typeof cause !== 'object') {
    return cause;
  }

  if (depth >= MAX_CAUSE_DEPTH) {
    return { name: 'Error', message: '[cause chain truncated]' };
  }

  if (seen.has(cause)) {
    return { name: 'Error', message: '[circular cause]' };
  }
  seen.add(cause);

  const error = cause as {
    readonly name?: unknown;
    readonly message?: unknown;
    readonly stack?: unknown;
    readonly cause?: unknown;
  };
  const serialized: Record<string, unknown> = {
    name: typeof error.name === 'string' ? error.name : 'Error',
    message: typeof error.message === 'string' ? error.message : String(cause),
  };
  if (typeof error.stack === 'string') {
    serialized.stack = error.stack;
  }
  if (error.cause !== undefined) {
    serialized.cause = serializeCause(error.cause, seen, depth + 1);
  }
  return serialized;
}

function createInterpreterError(
  kind: CodeInterpreterError['kind'],
  step: number | null,
  cause: unknown,
  fatal: boolean,
): CodeInterpreterError {
  return Object.freeze({
    kind,
    fatal,
    step,
    cause: serializeCause(cause),
  });
}

function cloneBindingValue(value: unknown): unknown {
  if (typeof value === 'function') {
    return value;
  }

  try {
    return structuredClone(value);
  } catch {
    return value;
  }
}

function createConsoleCapture(buffer: string[]): {
  readonly log: (...parts: readonly unknown[]) => void;
  readonly error: (...parts: readonly unknown[]) => void;
  readonly warn: (...parts: readonly unknown[]) => void;
} {
  const append = (...parts: readonly unknown[]): void => {
    const text = parts.map((part) => (
      typeof part === 'string' ? part : JSON.stringify(part)
    )).join(' ');
    buffer.push(text);
  };

  return Object.freeze({
    log: append,
    error: append,
    warn: append,
  });
}

function classifyExecutionError(error: unknown): CodeInterpreterError['kind'] {
  if (error instanceof BudgetError) {
    return 'budget';
  }
  return 'runtime';
}

interface RecoverableBinding {
  readonly name: string;
  readonly mutable: boolean;
}

function collectRecoverableBindings(source: string): readonly RecoverableBinding[] {
  const bindings = new Map<string, RecoverableBinding>();

  for (const match of source.matchAll(SIMPLE_VARIABLE_DECL_RE)) {
    const name = match.groups?.name;
    const kind = match.groups?.kind;
    if (!name || !kind || bindings.has(name)) {
      continue;
    }

    bindings.set(name, Object.freeze({
      name,
      mutable: kind !== 'const',
    }));
  }

  for (const match of source.matchAll(FUNCTION_DECL_RE)) {
    const name = match.groups?.name;
    if (!name || bindings.has(name)) {
      continue;
    }

    bindings.set(name, Object.freeze({ name, mutable: false }));
  }

  for (const match of source.matchAll(CLASS_DECL_RE)) {
    const name = match.groups?.name;
    if (!name || bindings.has(name)) {
      continue;
    }

    bindings.set(name, Object.freeze({ name, mutable: false }));
  }

  return Object.freeze([...bindings.values()]);
}

interface NodeCodeSessionOptions {
  readonly timeoutMs: number;
}

export interface SyncCodeSession
extends CodeSession<Record<string, unknown>, unknown> {
  readonly executeSync: <TOutput>(
    request: ExecuteRequest,
  ) => ExecuteResult<TOutput>;
  readonly inspectGlobalsSync: () => readonly REPLVariable[];
  readonly snapshotGlobalsSync: () => Record<string, unknown>;
  readonly patchGlobalsSync: (patch: InterpreterPatch<unknown>) => void;
  readonly closeSync: () => void;
}

class NodeCodeSession implements SyncCodeSession {
  readonly #context: vm.Context;
  readonly #bindings: Record<string, unknown>;
  readonly #options: NodeCodeSessionOptions;

  #state: 'open' | 'faulted' | 'closed' = 'open';
  #llmCallsUsed = 0;

  constructor(
    context: vm.Context,
    bindings: Record<string, unknown>,
    options: NodeCodeSessionOptions,
  ) {
    this.#context = context;
    this.#bindings = bindings;
    this.#options = options;
  }

  executeSync<TOutput>(
    request: ExecuteRequest,
  ): ExecuteResult<TOutput> {
    if (this.#state === 'closed') {
      return {
        tag: 'fault',
        historyDelta: EMPTY_HISTORY_DELTA,
        liveVariables: EMPTY_VARIABLES,
        error: createInterpreterError(
          'protocol',
          request.step,
          'Session is closed.',
          true,
        ),
      };
    }

    if (this.#state === 'faulted') {
      return {
        tag: 'fault',
        historyDelta: EMPTY_HISTORY_DELTA,
        liveVariables: EMPTY_VARIABLES,
        error: createInterpreterError(
          'protocol',
          request.step,
          'Session is faulted and must be closed before reuse.',
          true,
        ),
      };
    }

    if (request.step > request.budget.maxIterations) {
      return {
        tag: 'fault',
        historyDelta: EMPTY_HISTORY_DELTA,
        liveVariables: EMPTY_VARIABLES,
        error: createInterpreterError(
          'budget',
          request.step,
          `Iteration budget exceeded (${request.step} > ${request.budget.maxIterations}).`,
          false,
        ),
      };
    }

    if (typeof request.source !== 'string') {
      this.#state = 'faulted';
      return {
        tag: 'fault',
        historyDelta: EMPTY_HISTORY_DELTA,
        liveVariables: this.collectVisibleVariables(),
        error: createInterpreterError(
          'protocol',
          request.step,
          'Execute source must be a string.',
          true,
        ),
      };
    }

    const source = request.source;
    const historyDelta: REPLEntry[] = [];
    const stdout: string[] = [];
    let submitted = false;
    let submittedValue: unknown;

    const previousConsole = this.#bindings.console;
    const previousSubmit = this.#bindings.SUBMIT;
    const previousSingleQuery = this.#bindings.llmQuery;
    const previousBatchQuery = this.#bindings.llmQueryBatched;

    const restoreBindings = (): void => {
      this.#bindings.console = previousConsole;
      this.#bindings.SUBMIT = previousSubmit;
      this.#bindings.llmQuery = previousSingleQuery;
      this.#bindings.llmQueryBatched = previousBatchQuery;
    };

    this.#bindings.console = createConsoleCapture(stdout);
    this.#bindings.SUBMIT = (value: unknown): never => {
      if (!request.allowSubmit) {
        throw new ValueError('SUBMIT is disabled for this interpreter step.');
      }
      submitted = true;
      submittedValue = value;
      throw new SubmitSignal();
    };

    if (typeof previousSingleQuery === 'function') {
      this.#bindings.llmQuery = (prompt: unknown): unknown => {
        if (typeof prompt !== 'string' || prompt.trim() === '') {
          throw new ValueError('llmQuery(prompt) requires a non-empty prompt string.');
        }

        if (this.#llmCallsUsed + 1 > request.budget.maxLlmCalls) {
          throw new BudgetError(
            `LLM call budget exceeded (${this.#llmCallsUsed + 1}/${request.budget.maxLlmCalls}).`,
          );
        }

        this.#llmCallsUsed += 1;
        historyDelta.push(Object.freeze({
          step: request.step,
          kind: 'query',
          ok: true,
        }));
        return previousSingleQuery(prompt);
      };
    }

    if (typeof previousBatchQuery === 'function') {
      this.#bindings.llmQueryBatched = (prompts: unknown): unknown => {
        if (!Array.isArray(prompts) || prompts.some((entry) => typeof entry !== 'string')) {
          throw new ValueError('llmQueryBatched(prompts) requires an array of prompt strings.');
        }

        if (prompts.length > request.budget.maxBatchWidth) {
          throw new BudgetError(
            `llmQueryBatched batch width exceeded (${prompts.length} > ${request.budget.maxBatchWidth}).`,
          );
        }

        if (this.#llmCallsUsed + prompts.length > request.budget.maxLlmCalls) {
          throw new BudgetError(
            `LLM call budget exceeded (${this.#llmCallsUsed + prompts.length}/${request.budget.maxLlmCalls}).`,
          );
        }

        this.#llmCallsUsed += prompts.length;
        historyDelta.push(Object.freeze({
          step: request.step,
          kind: 'batch_query',
          ok: true,
        }));
        return previousBatchQuery(prompts);
      };
    }

    try {
      const script = new vm.Script(source, {
        filename: `rlm_step_${request.step}.js`,
      });
      const result = script.runInContext(this.#context, {
        timeout: this.#options.timeoutMs,
      });
      if (result instanceof Promise) {
        throw new ValueError(
          'Async execution result detected. Use synchronous llmQuery/tool calls in NodeCodeInterpreter sessions.',
        );
      }
    } catch (error) {
      if (error instanceof SubmitSignal) {
        historyDelta.push(Object.freeze({
          step: request.step,
          kind: 'submit',
          ok: true,
        }));
        historyDelta.push(Object.freeze({
          step: request.step,
          kind: 'exec',
          ok: true,
        }));
        return {
          tag: 'submit',
          historyDelta: Object.freeze([...historyDelta]),
          output: Object.freeze({
            value: submittedValue as TOutput,
            via: 'submit',
          }),
        };
      }

      this.#state = 'faulted';
      historyDelta.push(Object.freeze({
        step: request.step,
        kind: 'fault',
        ok: false,
      }));
      return {
        tag: 'fault',
        historyDelta: Object.freeze([...historyDelta]),
        liveVariables: this.collectVisibleVariables(source),
        error: createInterpreterError(
          classifyExecutionError(error),
          request.step,
          error,
          false,
        ),
      };
    } finally {
      restoreBindings();
    }

    const hadOutput = stdout.length > 0;
    historyDelta.push(Object.freeze({
      step: request.step,
      kind: 'exec',
      ok: true,
    }));

    if (submitted) {
      historyDelta.push(Object.freeze({
        step: request.step,
        kind: 'submit',
        ok: true,
      }));
      return {
        tag: 'submit',
        historyDelta: Object.freeze([...historyDelta]),
        output: Object.freeze({
          value: submittedValue as TOutput,
          via: 'submit',
        }),
      };
    }

    if (hadOutput) {
      // stdout is intentionally retained in runtime state for debugger-style inspection.
      this.#bindings.__last_stdout__ = Object.freeze([...stdout]);
    }

    return {
      tag: 'continue',
      historyDelta: Object.freeze([...historyDelta]),
    };
  }

  async execute<TOutput>(
    request: ExecuteRequest,
  ): Promise<ExecuteResult<TOutput>> {
    return this.executeSync(request);
  }

  inspectGlobalsSync(): readonly REPLVariable[] {
    this.ensureMutableSession('inspect globals');

    return this.collectVisibleVariables();
  }

  async inspectGlobals(): Promise<readonly REPLVariable[]> {
    return this.inspectGlobalsSync();
  }

  private collectVisibleVariables(source = ''): readonly REPLVariable[] {
    const variables: REPLVariable[] = [];
    for (const key of Object.keys(this.#bindings).sort()) {
      if (INTERNAL_BINDINGS.has(key)) {
        continue;
      }

      variables.push(Object.freeze({
        symbol: key,
        value: cloneBindingValue(this.#bindings[key]),
        mutable: typeof this.#bindings[key] !== 'function',
      }));
    }

    if (source.trim() !== '') {
      const present = new Set(variables.map((variable) => variable.symbol));
      for (const binding of collectRecoverableBindings(source)) {
        if (present.has(binding.name) || INTERNAL_BINDINGS.has(binding.name)) {
          continue;
        }

        const value = this.tryReadBinding(binding.name);
        if (value === MISSING_BINDING) {
          continue;
        }

        variables.push(Object.freeze({
          symbol: binding.name,
          value,
          mutable: binding.mutable,
        }));
      }

      variables.sort((left, right) => left.symbol.localeCompare(right.symbol));
    }

    return Object.freeze(variables);
  }

  snapshotGlobalsSync(): Record<string, unknown> {
    this.ensureMutableSession('snapshot globals');

    const snapshot: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(this.#bindings)) {
      if (INTERNAL_BINDINGS.has(key) || typeof value === 'function') {
        continue;
      }

      try {
        snapshot[key] = structuredClone(value);
      } catch {
        snapshot[key] = String(value);
      }
    }

    return Object.freeze(snapshot);
  }

  async snapshotGlobals(): Promise<Record<string, unknown>> {
    return this.snapshotGlobalsSync();
  }

  patchGlobalsSync(patch: InterpreterPatch<unknown>): void {
    this.ensureMutableSession('patch globals');
    if (!patch || typeof patch !== 'object') {
      throw new ValueError('patchGlobals expects an object patch.');
    }

    for (const [key, value] of Object.entries(patch.bindings)) {
      this.#bindings[key] = cloneBindingValue(value);
    }
  }

  async patchGlobals(
    patch: InterpreterPatch<unknown>,
  ): Promise<void> {
    this.patchGlobalsSync(patch);
  }

  closeSync(): void {
    this.#state = 'closed';
  }

  async close(): Promise<void> {
    this.closeSync();
  }

  private ensureMutableSession(action: string): void {
    if (this.#state === 'closed') {
      throw new ValueError(`Cannot ${action}: session is closed.`);
    }

    if (this.#state === 'faulted') {
      throw new ValueError(`Cannot ${action}: session is faulted.`);
    }
  }

  private tryReadBinding(name: string): unknown | typeof MISSING_BINDING {
    try {
      return cloneBindingValue(new vm.Script(name).runInContext(this.#context, {
        timeout: this.#options.timeoutMs,
      }));
    } catch {
      return MISSING_BINDING;
    }
  }
}

export interface NodeCodeInterpreterOptions {
  readonly timeoutMs?: number;
  readonly initialBindings?: Readonly<Record<string, unknown>>;
}

export interface SyncCodeInterpreter
extends CodeInterpreter<Record<string, unknown>, unknown> {
  readonly createSessionSync: () => SyncCodeSession;
}

export class NodeCodeInterpreter
implements SyncCodeInterpreter {
  readonly #options: NodeCodeInterpreterOptions;

  constructor(options: NodeCodeInterpreterOptions = {}) {
    this.#options = options;
  }

  createSessionSync(): SyncCodeSession {
    const bindings: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
    for (const [key, value] of Object.entries(this.#options.initialBindings ?? {})) {
      bindings[key] = cloneBindingValue(value);
    }

    const context = vm.createContext(bindings);
    return new NodeCodeSession(
      context,
      bindings,
      { timeoutMs: this.#options.timeoutMs ?? 5_000 },
    );
  }

  async createSession(): Promise<CodeSession<Record<string, unknown>, unknown>> {
    return this.createSessionSync();
  }
}

export function createNodeCodeInterpreter(
  options: NodeCodeInterpreterOptions = {},
): NodeCodeInterpreter {
  return new NodeCodeInterpreter(options);
}
