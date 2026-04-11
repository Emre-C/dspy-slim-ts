/**
 * Shared RLM substrate contracts.
 */

export interface BudgetVector {
  readonly maxIterations: number;
  readonly maxLlmCalls: number;
  readonly maxBatchWidth: number;
}

export interface CodeInterpreterError {
  readonly kind: 'runtime' | 'budget' | 'serialization' | 'protocol';
  readonly fatal: boolean;
  readonly step: number | null;
  readonly cause: unknown;
}

export interface REPLVariable<TValue = unknown> {
  readonly symbol: string;
  readonly value: TValue;
  readonly mutable: boolean;
}

export type REPLEntryKind =
  | 'exec'
  | 'query'
  | 'batch_query'
  | 'submit'
  | 'extract'
  | 'fault';

export interface REPLEntry {
  readonly step: number;
  readonly kind: REPLEntryKind;
  readonly ok: boolean;
}

export interface REPLHistory {
  readonly entries: readonly REPLEntry[];
  readonly liveSymbols: readonly string[];
}

export interface InterpreterPatch<TValue = unknown> {
  readonly bindings: Readonly<Record<string, TValue>>;
}

export interface ExecuteRequest {
  readonly step: number;
  readonly source: unknown;
  readonly budget: BudgetVector;
  readonly allowSubmit: boolean;
}

export interface FinalOutput<TOutput> {
  readonly value: TOutput;
  readonly via: 'submit' | 'extract';
}

export type ExecuteResult<TOutput> =
  | {
      readonly tag: 'continue';
      readonly historyDelta: readonly REPLEntry[];
    }
  | {
      readonly tag: 'submit';
      readonly historyDelta: readonly REPLEntry[];
      readonly output: FinalOutput<TOutput>;
    }
  | {
      readonly tag: 'fault';
      readonly historyDelta: readonly REPLEntry[];
      readonly liveVariables: readonly REPLVariable[];
      readonly error: CodeInterpreterError;
    };

export interface CodeSession<TSnapshot = unknown, TValue = unknown> {
  readonly execute: <TOutput>(
    request: ExecuteRequest,
  ) => Promise<ExecuteResult<TOutput>>;
  readonly inspectGlobals: () => Promise<readonly REPLVariable<TValue>[]>;
  readonly snapshotGlobals: () => Promise<TSnapshot>;
  readonly patchGlobals: (patch: InterpreterPatch<TValue>) => Promise<void>;
  readonly close: () => Promise<void>;
}

export interface CodeInterpreter<TSnapshot = unknown, TValue = unknown> {
  readonly createSession: () => Promise<CodeSession<TSnapshot, TValue>>;
}

export interface LLMQueryRequest<TInput = unknown> {
  readonly requestId: number;
  readonly payload: TInput;
}

export interface LLMQueryResult<TOutput = unknown> {
  readonly requestId: number;
  readonly ok: boolean;
  readonly output: TOutput | null;
}

export interface RLMConfig {
  readonly budget: BudgetVector;
  readonly trackTrace: boolean;
  readonly reservedToolNames: readonly string[];
  readonly subLmResolution: 'instance_then_settings';
}

export interface RLMRunResult<TOutput> {
  readonly output: FinalOutput<TOutput> | null;
  readonly history: REPLHistory;
  readonly error: CodeInterpreterError | null;
}
