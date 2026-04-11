import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import type { Callback } from '../src/callback.js';
import { Module } from '../src/module.js';
import { Prediction } from '../src/prediction.js';
import { Settings } from '../src/settings.js';

interface FixtureOp {
  action: 'configure' | 'context_enter' | 'context_exit' | 'read';
  kwargs?: Record<string, unknown>;
  key?: 'lm' | 'adapter' | 'num_threads' | 'max_errors' | 'disable_history';
  expected?: unknown;
  expected_error?: string;
  thread?: number;
}

interface FixtureCase {
  id: string;
  ops: FixtureOp[];
}

class SettingsModule extends Module {
  readonly id: string;

  constructor(id: string) {
    super();
    this.id = id;
  }

  override forward(_kwargs: Record<string, unknown> = {}): Prediction {
    return Prediction.create({});
  }
}

const fixture = JSON.parse(
  readFileSync(
    new URL('../../spec/fixtures/settings_behavior.json', import.meta.url),
    'utf-8',
  ),
) as { cases: FixtureCase[] };

function runWithThread<T>(settings: Settings, thread: number | undefined, fn: () => T): T {
  return thread === undefined ? fn() : settings.withThread(thread, fn);
}

function readSetting(settings: Settings, key: NonNullable<FixtureOp['key']>): unknown {
  switch (key) {
    case 'lm':
      return settings.lm;
    case 'adapter':
      return settings.adapter;
    case 'num_threads':
      return settings.numThreads;
    case 'max_errors':
      return settings.maxErrors;
    case 'disable_history':
      return settings.disableHistory;
    default:
      throw new Error(`Unknown settings key: ${key}`);
  }
}

function executeOps(
  settings: Settings,
  ops: readonly FixtureOp[],
  startIndex = 0,
): number {
  let index = startIndex;

  while (index < ops.length) {
    const op = ops[index]!;

    if (op.action === 'context_exit') {
      return index + 1;
    }

    if (op.action === 'context_enter') {
      index = runWithThread(settings, op.thread, () => settings.context(op.kwargs ?? {}, () => (
        executeOps(settings, ops, index + 1)
      )));
      continue;
    }

    runWithThread(settings, op.thread, () => {
      if (op.action === 'configure') {
        if (op.expected_error) {
          expect(() => settings.configure(op.kwargs ?? {})).toThrow(op.expected_error);
        } else {
          settings.configure(op.kwargs ?? {});
        }
        return;
      }

      if (op.action === 'read') {
        expect(readSetting(settings, op.key!)).toEqual(op.expected);
        return;
      }

      throw new Error(`Unknown settings action: ${op.action}`);
    });

    index += 1;
  }

  return index;
}

describe('Settings behavior (spec fixtures)', () => {
  for (const c of fixture.cases) {
    it(c.id, () => {
      const settings = new Settings();
      executeOps(settings, c.ops);
    });
  }
});

describe('Settings hardening', () => {
  it('context inherits the current thread identity while overriding selected keys', () => {
    const settings = new Settings();
    const lmA = { model: 'lm-a' };
    const lmB = { model: 'lm-b' };
    const adapterA = { useNativeFunctionCalling: false };
    const adapterB = { useNativeFunctionCalling: true };

    settings.withThread(7, () => {
      settings.configure({ lm: lmA, adapter: adapterA });
      settings.context({ lm: lmB }, () => {
        expect(settings.lm).toBe(lmB);
        expect(settings.adapter).toBe(adapterA);
        settings.configure({ adapter: adapterB });
      });
      expect(settings.adapter).toBe(adapterB);
    });
  });

  it('defensively owns callback and caller module lists', () => {
    const settings = new Settings();
    const callbacks: Callback[] = [{}];
    const modules = [new SettingsModule('module-a')];

    settings.configure({ callbacks, callerModules: modules });
    callbacks.push({});
    modules.push(new SettingsModule('module-b'));

    expect(settings.callbacks).toEqual([callbacks[0]]);
    expect(settings.callerModules).toHaveLength(1);
    expect(settings.callerModules[0]).toBe(modules[0]);
    expect(() => (settings.callbacks as unknown[]).push({ name: 'x' })).toThrow();
  });
});
