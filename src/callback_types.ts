/**
 * Callback contract — leaf module (no settings / module imports) to avoid cycles.
 */

export type CallbackDispatchKind =
  | 'module'
  | 'evaluate'
  | 'lm'
  | 'adapter_format'
  | 'adapter_parse'
  | 'tool';

export interface Callback {
  onModuleStart?(callID: string, instance: unknown, inputs: Record<string, unknown>): void;
  onModuleEnd?(callID: string, outputs: unknown | null, exception: Error | null): void;
  onEvaluateStart?(callID: string, instance: unknown, inputs: Record<string, unknown>): void;
  onEvaluateEnd?(callID: string, outputs: unknown | null, exception: Error | null): void;
  onLmStart?(callID: string, instance: unknown, inputs: Record<string, unknown>): void;
  onLmEnd?(callID: string, outputs: unknown | null, exception: Error | null): void;
  onAdapterFormatStart?(callID: string, instance: unknown, inputs: Record<string, unknown>): void;
  onAdapterFormatEnd?(callID: string, outputs: unknown | null, exception: Error | null): void;
  onAdapterParseStart?(callID: string, instance: unknown, inputs: Record<string, unknown>): void;
  onAdapterParseEnd?(callID: string, outputs: unknown | null, exception: Error | null): void;
  onToolStart?(callID: string, instance: unknown, inputs: Record<string, unknown>): void;
  onToolEnd?(callID: string, outputs: unknown | null, exception: Error | null): void;
}

export class BaseCallback implements Callback {
  onModuleStart(_callID: string, _instance: unknown, _inputs: Record<string, unknown>): void {}

  onModuleEnd(_callID: string, _outputs: unknown | null, _exception: Error | null): void {}

  onEvaluateStart(_callID: string, _instance: unknown, _inputs: Record<string, unknown>): void {}

  onEvaluateEnd(_callID: string, _outputs: unknown | null, _exception: Error | null): void {}

  onLmStart(_callID: string, _instance: unknown, _inputs: Record<string, unknown>): void {}

  onLmEnd(_callID: string, _outputs: unknown | null, _exception: Error | null): void {}

  onAdapterFormatStart(_callID: string, _instance: unknown, _inputs: Record<string, unknown>): void {}

  onAdapterFormatEnd(_callID: string, _outputs: unknown | null, _exception: Error | null): void {}

  onAdapterParseStart(_callID: string, _instance: unknown, _inputs: Record<string, unknown>): void {}

  onAdapterParseEnd(_callID: string, _outputs: unknown | null, _exception: Error | null): void {}

  onToolStart(_callID: string, _instance: unknown, _inputs: Record<string, unknown>): void {}

  onToolEnd(_callID: string, _outputs: unknown | null, _exception: Error | null): void {}
}
