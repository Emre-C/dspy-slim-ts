/**
 * §11.1 — settings configure/context semantics.
 *
 * Global configuration is owned by a single thread identity, while contextual
 * overrides are async-local and never mutate the global baseline.
 */

import { AsyncLocalStorage } from 'node:async_hooks';
import type { BaseModule } from './module.js';

function freezeList<T>(items: readonly T[] | undefined): readonly T[] {
  return Object.freeze([...(items ?? [])]);
}

export interface SettingsSnapshot {
  readonly lm: unknown | null;
  readonly adapter: unknown | null;
  readonly numThreads: number;
  readonly maxErrors: number;
  readonly disableHistory: boolean;
  readonly maxHistorySize: number;
  readonly callbacks: readonly unknown[];
  readonly callerModules: readonly BaseModule[];
}

export interface SettingsOverrides {
  readonly lm?: unknown | null;
  readonly adapter?: unknown | null;
  readonly numThreads?: number;
  readonly maxErrors?: number;
  readonly disableHistory?: boolean;
  readonly maxHistorySize?: number;
  readonly callbacks?: readonly unknown[];
  readonly callerModules?: readonly BaseModule[];
}

interface SettingsContextState {
  readonly threadId: number;
  readonly overrides: SettingsOverrides;
}

const DEFAULT_SETTINGS: SettingsSnapshot = Object.freeze({
  lm: null,
  adapter: null,
  numThreads: 8,
  maxErrors: 10,
  disableHistory: false,
  maxHistorySize: 10000,
  callbacks: freezeList(undefined),
  callerModules: freezeList(undefined) as readonly BaseModule[],
});

function mergeSettings(
  base: SettingsSnapshot,
  overrides: SettingsOverrides,
): SettingsSnapshot {
  return Object.freeze({
    lm: overrides.lm === undefined ? base.lm : overrides.lm,
    adapter: overrides.adapter === undefined ? base.adapter : overrides.adapter,
    numThreads: overrides.numThreads ?? base.numThreads,
    maxErrors: overrides.maxErrors ?? base.maxErrors,
    disableHistory: overrides.disableHistory ?? base.disableHistory,
    maxHistorySize: overrides.maxHistorySize ?? base.maxHistorySize,
    callbacks:
      overrides.callbacks === undefined
        ? base.callbacks
        : freezeList(overrides.callbacks),
    callerModules:
      overrides.callerModules === undefined
        ? base.callerModules
        : freezeList(overrides.callerModules),
  });
}

export class Settings {
  readonly #storage = new AsyncLocalStorage<SettingsContextState>();

  #globalValues: SettingsSnapshot = DEFAULT_SETTINGS;
  #ownerThreadId: number | null = null;

  configure(overrides: SettingsOverrides): void {
    const threadId = this.currentThreadId();

    if (this.#ownerThreadId !== null && this.#ownerThreadId !== threadId) {
      throw new Error('RuntimeError: settings.configure() is owned by another thread');
    }

    this.#ownerThreadId = threadId;
    this.#globalValues = mergeSettings(this.#globalValues, overrides);
  }

  context<T>(overrides: SettingsOverrides, fn: () => T): T {
    const state: SettingsContextState = {
      threadId: this.currentThreadId(),
      overrides: {
        ...this.currentOverrides(),
        ...overrides,
      },
    };

    return this.#storage.run(state, fn);
  }

  withThread<T>(threadId: number, fn: () => T): T {
    const state: SettingsContextState = {
      threadId,
      overrides: this.currentOverrides(),
    };

    return this.#storage.run(state, fn);
  }

  snapshot(): SettingsSnapshot {
    const store = this.#storage.getStore();
    return store ? mergeSettings(this.#globalValues, store.overrides) : this.#globalValues;
  }

  get<K extends keyof SettingsSnapshot>(key: K): SettingsSnapshot[K] {
    return this.snapshot()[key];
  }

  get lm(): unknown | null {
    return this.snapshot().lm;
  }

  get adapter(): unknown | null {
    return this.snapshot().adapter;
  }

  get numThreads(): number {
    return this.snapshot().numThreads;
  }

  get maxErrors(): number {
    return this.snapshot().maxErrors;
  }

  get disableHistory(): boolean {
    return this.snapshot().disableHistory;
  }

  get maxHistorySize(): number {
    return this.snapshot().maxHistorySize;
  }

  get callbacks(): readonly unknown[] {
    return this.snapshot().callbacks;
  }

  get callerModules(): readonly BaseModule[] {
    return this.snapshot().callerModules;
  }

  reset(): void {
    this.#globalValues = DEFAULT_SETTINGS;
    this.#ownerThreadId = null;
  }

  private currentThreadId(): number {
    return this.#storage.getStore()?.threadId ?? 0;
  }

  private currentOverrides(): SettingsOverrides {
    return this.#storage.getStore()?.overrides ?? {};
  }
}

export const settings = new Settings();
