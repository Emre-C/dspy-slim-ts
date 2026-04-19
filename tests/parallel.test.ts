import { afterEach, describe, expect, it, vi } from 'vitest';
import { Example, Parallel, Prediction, ValueError, settings } from '../src/index.js';

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

afterEach(() => {
  settings.reset();
  vi.restoreAllMocks();
});

describe('Parallel', () => {
  it('dispatches Example inputs through their declared input keys', () => {
    const seenInputs: Record<string, unknown>[] = [];
    const program = {
      call(kwargs: Record<string, unknown>): Prediction {
        seenInputs.push({ ...kwargs });
        return Prediction.create({ output: String(kwargs.input).toUpperCase() });
      },
    };
    const example = new Example({ input: 'alpha', label: 'ignored' }).withInputs('input');
    const parallel = new Parallel({ numThreads: 1 });

    const results = parallel.call<Prediction>([[program, example]]) as readonly (Prediction | null)[];

    expect(seenInputs).toEqual([{ input: 'alpha' }]);
    expect(results[0]?.toDict()).toEqual({ output: 'ALPHA' });
  });

  it('returns failed examples and exceptions when requested', () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const program = {
      call(kwargs: Record<string, unknown>): string {
        if (kwargs.input === 'boom') {
          throw new Error('boom');
        }
        return String(kwargs.input).toUpperCase();
      },
    };
    const pairs = [
      [program, { input: 'ok' }],
      [program, { input: 'boom' }],
      [program, { input: 'still-ok' }],
    ] as const;
    const parallel = new Parallel({
      numThreads: 1,
      maxErrors: 3,
      returnFailedExamples: true,
    });

    const [results, failedExamples, exceptions] = parallel.call<string, { input: string }>(
      pairs,
    ) as readonly [readonly (string | null)[], readonly { input: string }[], readonly Error[]];

    expect(results).toEqual(['OK', null, 'STILL-OK']);
    expect(failedExamples).toEqual([{ input: 'boom' }]);
    expect(exceptions.map((error) => error.message)).toEqual(['boom']);
    expect(errorSpy).toHaveBeenCalledTimes(1);
  });

  it('cancels execution once maxErrors is reached', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const program = {
      call(): string {
        throw new Error('failure');
      },
    };
    const parallel = new Parallel({ numThreads: 1, maxErrors: 2 });

    expect(() => parallel.call([
      [program, { input: 1 }],
      [program, { input: 2 }],
      [program, { input: 3 }],
    ])).toThrow('Execution cancelled due to errors or interruption.');
  });

  it('bounds async concurrency in aforward', async () => {
    let active = 0;
    let maxActive = 0;
    const worker = {
      async acall(kwargs: Record<string, unknown>): Promise<number> {
        const value = Number(kwargs.value);
        active += 1;
        maxActive = Math.max(maxActive, active);
        await sleep(20);
        active -= 1;
        return value * 2;
      },
    };
    const parallel = new Parallel({ numThreads: 2 });

    const results = await parallel.acall<number, { value: number }>([
      [worker, { value: 1 }],
      [worker, { value: 2 }],
      [worker, { value: 3 }],
      [worker, { value: 4 }],
    ]) as readonly (number | null)[];

    expect(results).toEqual([2, 4, 6, 8]);
    expect(maxActive).toBe(2);
  });

  it('resubmits stragglers when timeout expires near the tail', async () => {
    const calls = new Map<string, number>();
    const worker = {
      async acall(kwargs: Record<string, unknown>): Promise<string> {
        const id = String(kwargs.id);
        const count = (calls.get(id) ?? 0) + 1;
        calls.set(id, count);
        if (id === 'slow') {
          await sleep(count === 1 ? 200 : 10);
          return `slow:${count}`;
        }
        await sleep(10);
        return 'fast';
      },
    };
    const parallel = new Parallel({
      numThreads: 2,
      timeout: 40,
      stragglerLimit: 2,
    });
    const startedAt = Date.now();

    const results = await parallel.acall<string, { id: string }>([
      [worker, { id: 'slow' }],
      [worker, { id: 'fast' }],
    ]) as readonly (string | null)[];
    const duration = Date.now() - startedAt;

    expect(results).toEqual(['slow:2', 'fast']);
    expect(calls.get('slow')).toBe(2);
    expect(duration).toBeLessThan(170);
  });

  it('returns a frozen empty array for empty data', () => {
    const parallel = new Parallel({ numThreads: 1 });
    const results = parallel.execute(() => 'never', []);

    expect(results).toEqual([]);
    expect(Object.isFrozen(results)).toBe(true);
  });

  it('returns a frozen empty array for empty async data', async () => {
    const parallel = new Parallel({ numThreads: 1 });
    const results = await parallel.executeAsync(() => 'never', []);

    expect(results).toEqual([]);
    expect(Object.isFrozen(results)).toBe(true);
  });

  it('rejects invalid numThreads and maxErrors', () => {
    expect(() => new Parallel({ numThreads: 0 })).toThrow(ValueError);
    expect(() => new Parallel({ numThreads: 1.5 })).toThrow(ValueError);
    expect(() => new Parallel({ maxErrors: 0 })).toThrow(ValueError);
    expect(() => new Parallel({ maxErrors: -1 })).toThrow(ValueError);
    expect(() => new Parallel({ timeout: -1 })).toThrow(ValueError);
    expect(() => new Parallel({ stragglerLimit: 0 })).toThrow(ValueError);
  });

  it('dispatches array inputs to function targets via spread', () => {
    const fn = (...args: readonly unknown[]) => {
      const [a, b] = args as readonly [number, number];
      return a + b;
    };
    const parallel = new Parallel({ numThreads: 1 });

    const results = parallel.call<number, readonly number[]>([
      [fn, [1, 2]],
      [fn, [3, 4]],
    ]) as readonly (number | null)[];

    expect(results).toEqual([3, 7]);
  });

  it('dispatches plain function targets with object inputs', () => {
    const fn = (...args: readonly unknown[]) => {
      const [kwargs] = args as readonly [Record<string, unknown>];
      return String(kwargs.x).toUpperCase();
    };
    const parallel = new Parallel({ numThreads: 1 });

    const results = parallel.call<string, { x: string }>([
      [fn, { x: 'hello' }],
      [fn, { x: 'world' }],
    ]) as readonly (string | null)[];

    expect(results).toEqual(['HELLO', 'WORLD']);
  });

  it('supports nested Parallel as a target', () => {
    const fn = (...args: readonly unknown[]) => {
      const [kwargs] = args as readonly [Record<string, unknown>];
      return Number(kwargs.n) * 10;
    };
    const inner = new Parallel({ numThreads: 1 });
    const outer = new Parallel({ numThreads: 1 });

    const results = outer.call<unknown, readonly [typeof fn, { n: number }][]>([
      [inner, [[fn, { n: 1 }], [fn, { n: 2 }]]],
    ]) as readonly unknown[];

    expect(results).toEqual([[10, 20]]);
  });

  it('prints stack traces when provideTraceback is enabled', () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const parallel = new Parallel({ numThreads: 1, maxErrors: 5, provideTraceback: true });

    parallel.execute(() => { throw new Error('oops'); }, [1]);

    expect(errorSpy).toHaveBeenCalledTimes(1);
    const output = errorSpy.mock.calls[0]![0] as string;
    expect(output).toContain('oops');
    expect(output).toContain('Error');
  });

  it('prints short messages when provideTraceback is disabled', () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const parallel = new Parallel({ numThreads: 1, maxErrors: 5, provideTraceback: false });

    parallel.execute(() => { throw new Error('oops'); }, [1]);

    expect(errorSpy).toHaveBeenCalledTimes(1);
    const output = errorSpy.mock.calls[0]![0] as string;
    expect(output).toContain('Set provideTraceback=true');
  });

  it('does not double-count errors when a straggler resubmission also fails', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    let callCount = 0;
    const worker = {
      async acall(): Promise<string> {
        callCount += 1;
        await sleep(callCount === 1 ? 80 : 10);
        throw new Error(`fail:${callCount}`);
      },
    };
    const parallel = new Parallel({
      numThreads: 2,
      maxErrors: 5,
      timeout: 30,
      stragglerLimit: 2,
      returnFailedExamples: true,
    });

    const [results, failed, exceptions] = await parallel.acall<string, { id: string }>([
      [worker, { id: 'a' }],
    ]) as readonly [readonly (string | null)[], readonly { id: string }[], readonly Error[]];

    expect(results).toEqual([null]);
    expect(failed).toEqual([{ id: 'a' }]);
    expect(exceptions).toHaveLength(1);
  });

  it('encapsulates mutable run state from external access', () => {
    const parallel = new Parallel({ numThreads: 1 });

    expect(parallel.failedIndices).toEqual([]);
    expect(parallel.failedItems).toEqual([]);
    expect(parallel.exceptions).toEqual([]);

    expect('errorCount' in parallel).toBe(false);
    expect('cancelJobs' in parallel).toBe(false);
  });

  it('runs each async task inside its own settings thread context', async () => {
    settings.configure({ lm: null });
    const snapshots: string[] = [];
    const worker = {
      async acall(kwargs: Record<string, unknown>): Promise<string> {
        await sleep(10);
        const snap = JSON.stringify(settings.snapshot());
        snapshots.push(snap);
        return String(kwargs.n);
      },
    };
    const parallel = new Parallel({ numThreads: 3 });

    const results = await parallel.acall<string, { n: number }>([
      [worker, { n: 1 }],
      [worker, { n: 2 }],
      [worker, { n: 3 }],
    ]) as readonly (string | null)[];

    expect(results).toEqual(['1', '2', '3']);
    expect(snapshots).toHaveLength(3);
  });
});
