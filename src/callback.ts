/**
 * §9 — Callback dispatch and nested call tracking.
 */

import { AsyncLocalStorage } from 'node:async_hooks';
import { randomUUID } from 'node:crypto';
import type { Callback, CallbackDispatchKind } from './callback_types.js';
import { settings } from './settings.js';

export type { Callback } from './callback_types.js';
export { BaseCallback } from './callback_types.js';

const ACTIVE_CALL_ID = new AsyncLocalStorage<string | null>();

interface CallbackRunOptions<T> {
  readonly kind: CallbackDispatchKind;
  readonly instance: { readonly callbacks?: readonly Callback[] };
  readonly inputs: Record<string, unknown>;
  readonly execute: () => T;
}

function warnCallbackFailure(stage: 'start' | 'end', callback: Callback, error: unknown): void {
  const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  console.warn(`Callback ${stage} handler failed for ${callback.constructor?.name ?? 'callback'}: ${message}`);
}

function callbackList(instance: { readonly callbacks?: readonly Callback[] }): readonly Callback[] {
  return Object.freeze([
    ...settings.callbacks,
    ...(instance.callbacks ?? []),
  ]);
}

function dispatchStart(
  callback: Callback,
  kind: CallbackDispatchKind,
  callID: string,
  instance: unknown,
  inputs: Record<string, unknown>,
): void {
  switch (kind) {
    case 'module':
      callback.onModuleStart?.(callID, instance, inputs);
      return;
    case 'evaluate':
      callback.onEvaluateStart?.(callID, instance, inputs);
      return;
    case 'lm':
      callback.onLmStart?.(callID, instance, inputs);
      return;
    case 'adapter_format':
      callback.onAdapterFormatStart?.(callID, instance, inputs);
      return;
    case 'adapter_parse':
      callback.onAdapterParseStart?.(callID, instance, inputs);
      return;
    case 'tool':
      callback.onToolStart?.(callID, instance, inputs);
  }
}

function dispatchEnd(
  callback: Callback,
  kind: CallbackDispatchKind,
  callID: string,
  outputs: unknown | null,
  exception: Error | null,
): void {
  switch (kind) {
    case 'module':
      callback.onModuleEnd?.(callID, outputs, exception);
      return;
    case 'evaluate':
      callback.onEvaluateEnd?.(callID, outputs, exception);
      return;
    case 'lm':
      callback.onLmEnd?.(callID, outputs, exception);
      return;
    case 'adapter_format':
      callback.onAdapterFormatEnd?.(callID, outputs, exception);
      return;
    case 'adapter_parse':
      callback.onAdapterParseEnd?.(callID, outputs, exception);
      return;
    case 'tool':
      callback.onToolEnd?.(callID, outputs, exception);
  }
}

export function currentCallID(): string | null {
  return ACTIVE_CALL_ID.getStore() ?? null;
}

export function runWithCallbacks<T>(options: CallbackRunOptions<T>): T {
  const callbacks = callbackList(options.instance);
  if (callbacks.length === 0) {
    return options.execute();
  }

  const callID = randomUUID();
  for (const callback of callbacks) {
    try {
      dispatchStart(callback, options.kind, callID, options.instance, options.inputs);
    } catch (error) {
      warnCallbackFailure('start', callback, error);
    }
  }

  let result: T | null = null;
  let exception: Error | null = null;

  try {
    result = ACTIVE_CALL_ID.run(callID, options.execute);
    return result;
  } catch (error) {
    exception = error instanceof Error ? error : new Error(String(error));
    throw error;
  } finally {
    for (const callback of callbacks) {
      try {
        dispatchEnd(callback, options.kind, callID, result, exception);
      } catch (error) {
        warnCallbackFailure('end', callback, error);
      }
    }
  }
}

export async function runWithCallbacksAsync<T>(options: CallbackRunOptions<Promise<T>>): Promise<T> {
  const callbacks = callbackList(options.instance);
  if (callbacks.length === 0) {
    return options.execute();
  }

  const callID = randomUUID();
  for (const callback of callbacks) {
    try {
      dispatchStart(callback, options.kind, callID, options.instance, options.inputs);
    } catch (error) {
      warnCallbackFailure('start', callback, error);
    }
  }

  let result: T | null = null;
  let exception: Error | null = null;

  try {
    result = await ACTIVE_CALL_ID.run(callID, options.execute);
    return result;
  } catch (error) {
    exception = error instanceof Error ? error : new Error(String(error));
    throw error;
  } finally {
    for (const callback of callbacks) {
      try {
        dispatchEnd(callback, options.kind, callID, result, exception);
      } catch (error) {
        warnCallbackFailure('end', callback, error);
      }
    }
  }
}
