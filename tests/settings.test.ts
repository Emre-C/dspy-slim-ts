import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
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

const fixture = JSON.parse(
  readFileSync(
    new URL('../../dspy-slim/spec/fixtures/settings_behavior.json', import.meta.url),
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

    settings.withThread(7, () => {
      settings.configure({ lm: 'lm-a', adapter: 'adapter-a' });
      settings.context({ lm: 'lm-b' }, () => {
        expect(settings.lm).toBe('lm-b');
        expect(settings.adapter).toBe('adapter-a');
        settings.configure({ adapter: 'adapter-b' });
      });
      expect(settings.adapter).toBe('adapter-b');
    });
  });

  it('defensively owns callback and caller module lists', () => {
    const settings = new Settings();
    const callbacks = [{ name: 'cb' }];
    const modules = [{ id: 'module-a' }] as never[];

    settings.configure({ callbacks, callerModules: modules });
    callbacks.push({ name: 'mutated' });
    modules.push({ id: 'module-b' } as never);

    expect(settings.callbacks).toEqual([{ name: 'cb' }]);
    expect(settings.callerModules).toEqual([{ id: 'module-a' }]);
    expect(() => (settings.callbacks as unknown[]).push({ name: 'x' })).toThrow();
  });
});
