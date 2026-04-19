import { describe, expect, it } from 'vitest';
import type { Message } from '../src/chat_message.js';
import { BaseLM, type LMOutput } from '../src/lm.js';
import { BudgetError, RuntimeError, ValueError } from '../src/exceptions.js';
import { signatureFromString } from '../src/signature.js';
import type { Signature } from '../src/signature.js';
import {
  bop,
  concat,
  cross,
  ensemble,
  filter,
  fn,
  lit,
  map,
  oracle,
  peek,
  reduce,
  split,
  vote,
  vref,
  type CombinatorNode,
} from '../src/rlm_combinators.js';
import {
  __internal,
  buildEvaluationContext,
  evaluate,
  type BuildEvaluationContextOptions,
} from '../src/rlm_evaluator.js';
import type { EvaluationContext } from '../src/rlm_types.js';

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/**
 * `BaseLM` subclass that refuses every call. Used for tests that should
 * never reach the network (the deterministic combinator subset). Any
 * accidental invocation is a test authoring bug and must fail loudly.
 */
class RejectingLM extends BaseLM {
  constructor() {
    super({ model: 'rejecting-lm' });
  }

  protected override generate(): readonly LMOutput[] {
    throw new Error('RejectingLM should never be invoked in this test');
  }
}

/**
 * Scripted `BaseLM`: one queued JSON payload per call. Direct `oracle` leaves
 * use `EFFECT_ORACLE_SIGNATURE` — use `oracleValue()` for terminal `kind:
 * value` payloads. `vote` lanes still use the plain `answer` signature.
 */
interface ScriptedCall {
  readonly prompt: string | undefined;
  readonly messages: readonly Message[] | undefined;
  readonly kwargs: Record<string, unknown>;
}

class ScriptedLM extends BaseLM {
  readonly label: string;
  readonly outputs: LMOutput[];
  readonly calls: ScriptedCall[] = [];

  constructor(label: string, outputs: readonly LMOutput[]) {
    super({ model: label });
    this.label = label;
    this.outputs = [...outputs];
  }

  protected override generate(
    prompt?: string,
    messages?: readonly Message[],
    kwargs: Record<string, unknown> = {},
  ): readonly LMOutput[] {
    const next = this.outputs.shift();
    if (next === undefined) {
      throw new Error(
        `ScriptedLM '${this.label}' exhausted (call #${this.calls.length + 1}).`,
      );
    }
    this.calls.push({ prompt, messages, kwargs: { ...kwargs } });
    return [next];
  }
}

const DEFAULT_SIGNATURE: Signature = signatureFromString(
  'prompt: str -> answer: str',
);

/** JSON for a terminal `kind: value` oracle completion (`EFFECT_ORACLE_SIGNATURE`). */
function oracleValue(answer: string): string {
  return JSON.stringify({
    kind: 'value',
    value: answer,
    effect_name: null,
    effect_args: null,
  });
}

function makeCtx(
  overrides: Partial<BuildEvaluationContextOptions> = {},
): EvaluationContext {
  return buildEvaluationContext({
    lm: new RejectingLM(),
    signature: DEFAULT_SIGNATURE,
    ...overrides,
  });
}

// ---------------------------------------------------------------------------
// Literal / var
// ---------------------------------------------------------------------------

describe('evaluate — literal / var', () => {
  it('returns literal values unchanged', async () => {
    await expect(evaluate(lit('hello'), makeCtx())).resolves.toBe('hello');
    await expect(evaluate(lit(42), makeCtx())).resolves.toBe(42);
    await expect(evaluate(lit(true), makeCtx())).resolves.toBe(true);
  });

  it('literal lists are returned by reference', async () => {
    const payload = ['a', 'b'];
    await expect(evaluate(lit(payload), makeCtx())).resolves.toEqual(payload);
  });

  it('var resolves a bound value from scope', async () => {
    const ctx = makeCtx({ scope: new Map([['x', 'resolved']]) });
    await expect(evaluate(vref('x'), ctx)).resolves.toBe('resolved');
  });

  it('var throws ValueError when the name is unbound', async () => {
    await expect(evaluate(vref('missing'), makeCtx())).rejects.toBeInstanceOf(
      ValueError,
    );
  });
});

// ---------------------------------------------------------------------------
// Split
// ---------------------------------------------------------------------------

describe('evaluate — split', () => {
  it('partitions into k uniform chunks when length is divisible', async () => {
    const plan = split(lit('abcdef'), lit(3));
    await expect(evaluate(plan, makeCtx())).resolves.toEqual([
      'ab',
      'cd',
      'ef',
    ]);
  });

  it('distributes remainder into the leading chunks', async () => {
    const plan = split(lit('abcdefg'), lit(3));
    await expect(evaluate(plan, makeCtx())).resolves.toEqual([
      'abc',
      'de',
      'fg',
    ]);
  });

  it('returns an empty list for an empty input', async () => {
    const plan = split(lit(''), lit(5));
    await expect(evaluate(plan, makeCtx())).resolves.toEqual([]);
  });

  it('caps chunk count at input length when k > len', async () => {
    const plan = split(lit('abc'), lit(10));
    await expect(evaluate(plan, makeCtx())).resolves.toEqual(['a', 'b', 'c']);
  });

  it('rejects a non-string input', async () => {
    const plan = split(lit(42), lit(2));
    await expect(evaluate(plan, makeCtx())).rejects.toBeInstanceOf(ValueError);
  });

  it('rejects a non-numeric k', async () => {
    const plan = split(lit('abc'), lit('x'));
    await expect(evaluate(plan, makeCtx())).rejects.toBeInstanceOf(ValueError);
  });

  it('rejects k < 1', async () => {
    const plan = split(lit('abc'), lit(0));
    await expect(evaluate(plan, makeCtx())).rejects.toBeInstanceOf(ValueError);
  });
});

// ---------------------------------------------------------------------------
// Peek
// ---------------------------------------------------------------------------

describe('evaluate — peek', () => {
  it('returns the slice [start, end)', async () => {
    const plan = peek(lit('hello world'), lit(6), lit(11));
    await expect(evaluate(plan, makeCtx())).resolves.toBe('world');
  });

  it('clamps negative start to 0', async () => {
    const plan = peek(lit('abcdef'), lit(-10), lit(3));
    await expect(evaluate(plan, makeCtx())).resolves.toBe('abc');
  });

  it('clamps end past the input length to the input length', async () => {
    const plan = peek(lit('abc'), lit(0), lit(100));
    await expect(evaluate(plan, makeCtx())).resolves.toBe('abc');
  });

  it('returns empty string when end <= start', async () => {
    const plan = peek(lit('abc'), lit(2), lit(2));
    await expect(evaluate(plan, makeCtx())).resolves.toBe('');
  });

  it('rejects non-string input', async () => {
    const plan = peek(lit(42), lit(0), lit(1));
    await expect(evaluate(plan, makeCtx())).rejects.toBeInstanceOf(ValueError);
  });

  it('rejects non-numeric bounds', async () => {
    const plan = peek(lit('abc'), lit('x'), lit(1));
    await expect(evaluate(plan, makeCtx())).rejects.toBeInstanceOf(ValueError);
  });
});

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

describe('evaluate — map', () => {
  it('applies the identity body to every element', async () => {
    const plan = map(fn('x', vref('x')), lit(['a', 'b', 'c']));
    await expect(evaluate(plan, makeCtx())).resolves.toEqual(['a', 'b', 'c']);
  });

  it('returns an empty list on an empty input', async () => {
    const plan = map(fn('x', vref('x')), lit([]));
    await expect(evaluate(plan, makeCtx())).resolves.toEqual([]);
  });

  it('preserves input ordering even under parallel fan-out', async () => {
    // 20 items with parallelism 4 exercises the bounded worker pool.
    const items = Array.from({ length: 20 }, (_, i) => String(i));
    const plan = map(fn('x', vref('x')), lit(items));
    const ctx = makeCtx({ budget: { maxParallelism: 4 } });
    await expect(evaluate(plan, ctx)).resolves.toEqual(items);
  });

  it('rejects when items is not a list', async () => {
    const plan = map(fn('x', vref('x')), lit('abc'));
    await expect(evaluate(plan, makeCtx())).rejects.toBeInstanceOf(ValueError);
  });

  it('shadows outer scope for the duration of the element body', async () => {
    const ctx = makeCtx({ scope: new Map([['x', 'OUTER']]) });
    const plan = map(fn('x', vref('x')), lit(['a', 'b']));
    await expect(evaluate(plan, ctx)).resolves.toEqual(['a', 'b']);
    // outer binding is not mutated
    expect(ctx.scope.get('x')).toBe('OUTER');
  });
});

// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------

describe('evaluate — filter', () => {
  it('keeps elements for which the predicate returns true', async () => {
    const plan = filter(fn('x', lit(true)), lit(['a', 'b', 'c']));
    await expect(evaluate(plan, makeCtx())).resolves.toEqual(['a', 'b', 'c']);
  });

  it('drops elements for which the predicate returns false', async () => {
    const plan = filter(fn('x', lit(false)), lit(['a', 'b']));
    await expect(evaluate(plan, makeCtx())).resolves.toEqual([]);
  });

  it('preserves original ordering', async () => {
    // Mask-match on the first / third items via scope lookup.
    const plan = filter(
      fn('x', vref('keep')),
      lit([
        { keep: true, id: 'a' },
        { keep: false, id: 'b' },
        { keep: true, id: 'c' },
      ]),
    );
    // The predicate body reads `x.keep`; adapt by letting each element be
    // the boolean itself for this test instead of a dict lookup.
    const simplePlan = filter(fn('x', vref('x')), lit([true, false, true]));
    await expect(evaluate(simplePlan, makeCtx())).resolves.toEqual([true, true]);
    // Dict-shaped filter body is unused here — kept as a smoke artifact.
    void plan;
  });

  it('rejects when the predicate returns a non-boolean', async () => {
    const plan = filter(fn('x', lit('yes')), lit(['a']));
    await expect(evaluate(plan, makeCtx())).rejects.toBeInstanceOf(ValueError);
  });

  it('rejects when items is not a list', async () => {
    const plan = filter(fn('x', lit(true)), lit('abc'));
    await expect(evaluate(plan, makeCtx())).rejects.toBeInstanceOf(ValueError);
  });

  it('empty list round-trips to empty list', async () => {
    const plan = filter(fn('x', lit(true)), lit([]));
    await expect(evaluate(plan, makeCtx())).resolves.toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Reduce
// ---------------------------------------------------------------------------

describe('evaluate — reduce', () => {
  it('without init: folds pairwise starting from the first element', async () => {
    // body returns right, so this collapses the list to its last element
    const plan = reduce(
      bop('acc', 'elem', vref('elem')),
      lit(['a', 'b', 'c']),
    );
    await expect(evaluate(plan, makeCtx())).resolves.toBe('c');
  });

  it('with init: folds from the init value', async () => {
    // body returns left (acc), so this returns init regardless of elements
    const plan = reduce(
      bop('acc', 'elem', vref('acc')),
      lit(['a', 'b']),
      lit('INIT'),
    );
    await expect(evaluate(plan, makeCtx())).resolves.toBe('INIT');
  });

  it('errors on empty list when init is missing', async () => {
    const plan = reduce(bop('a', 'b', vref('a')), lit([]));
    await expect(evaluate(plan, makeCtx())).rejects.toBeInstanceOf(ValueError);
  });

  it('returns init when list is empty and init is provided', async () => {
    const plan = reduce(
      bop('a', 'b', vref('a')),
      lit([]),
      lit('sentinel'),
    );
    await expect(evaluate(plan, makeCtx())).resolves.toBe('sentinel');
  });

  it('is sequential: left (acc) binding sees prior step output', async () => {
    // Concat body: acc = acc + elem (string concat is left, else path).
    // Body here just returns elem; check order via a sequential-accumulating
    // body that picks the rightmost non-empty elem.
    const plan = reduce(
      bop('acc', 'elem', vref('elem')),
      lit(['1', '2', '3']),
      lit('start'),
    );
    await expect(evaluate(plan, makeCtx())).resolves.toBe('3');
  });

  it('errors when items is not a list', async () => {
    const plan = reduce(bop('a', 'b', vref('a')), lit('nope'));
    await expect(evaluate(plan, makeCtx())).rejects.toBeInstanceOf(ValueError);
  });
});

// ---------------------------------------------------------------------------
// Concat
// ---------------------------------------------------------------------------

describe('evaluate — concat', () => {
  it('joins without a separator by default', async () => {
    const plan = concat(lit(['ab', 'cd']));
    await expect(evaluate(plan, makeCtx())).resolves.toBe('abcd');
  });

  it('honors an explicit separator', async () => {
    const plan = concat(lit(['ab', 'cd']), lit(', '));
    await expect(evaluate(plan, makeCtx())).resolves.toBe('ab, cd');
  });

  it('returns empty string on empty list', async () => {
    const plan = concat(lit([]), lit('|'));
    await expect(evaluate(plan, makeCtx())).resolves.toBe('');
  });

  it('rejects non-string elements', async () => {
    const plan = concat(lit(['ab', 42]));
    await expect(evaluate(plan, makeCtx())).rejects.toBeInstanceOf(ValueError);
  });

  it('rejects non-string separator', async () => {
    const plan = concat(lit(['a', 'b']), lit(42));
    await expect(evaluate(plan, makeCtx())).rejects.toBeInstanceOf(ValueError);
  });

  it('rejects non-list items', async () => {
    const plan = concat(lit('not-a-list'));
    await expect(evaluate(plan, makeCtx())).rejects.toBeInstanceOf(ValueError);
  });
});

// ---------------------------------------------------------------------------
// Cross
// ---------------------------------------------------------------------------

describe('evaluate — cross', () => {
  it('produces the full Cartesian product in left-major order', async () => {
    const plan = cross(lit(['a', 'b']), lit(['x', 'y']));
    await expect(evaluate(plan, makeCtx())).resolves.toEqual([
      ['a', 'x'],
      ['a', 'y'],
      ['b', 'x'],
      ['b', 'y'],
    ]);
  });

  it('empty left yields empty product', async () => {
    const plan = cross(lit([]), lit(['x']));
    await expect(evaluate(plan, makeCtx())).resolves.toEqual([]);
  });

  it('empty right yields empty product', async () => {
    const plan = cross(lit(['a']), lit([]));
    await expect(evaluate(plan, makeCtx())).resolves.toEqual([]);
  });

  it('rejects non-list operands', async () => {
    await expect(
      evaluate(cross(lit('a'), lit(['x'])), makeCtx()),
    ).rejects.toBeInstanceOf(ValueError);
    await expect(
      evaluate(cross(lit(['a']), lit('x')), makeCtx()),
    ).rejects.toBeInstanceOf(ValueError);
  });
});

// Oracle: scripted LM, prompt forwarding, registry fallback, budget.

describe('evaluate — oracle', () => {
  it('returns the scripted answer on the happy path', async () => {
    const lm = new ScriptedLM('primary', [oracleValue('Paris')]);
    const ctx = makeCtx({ lm });
    await expect(
      evaluate(oracle(lit('what is the capital of France?')), ctx),
    ).resolves.toBe('Paris');
    expect(lm.calls).toHaveLength(1);
    expect(ctx.callsUsed.current).toBe(1);
  });

  it('forwards the prompt string into the adapter messages', async () => {
    const lm = new ScriptedLM('primary', [oracleValue('42')]);
    const ctx = makeCtx({ lm });
    await evaluate(oracle(lit('the ultimate question')), ctx);
    const text = JSON.stringify(lm.calls[0]?.messages ?? []);
    expect(text).toContain('the ultimate question');
  });

  it('routes via modelHint when the registry has a match', async () => {
    const fallback = new ScriptedLM('default', [oracleValue('wrong')]);
    const fast = new ScriptedLM('fast', [oracleValue('fast-answer')]);
    const ctx = makeCtx({
      lm: fallback,
      lmRegistry: new Map([['fast', fast]]),
    });
    await expect(
      evaluate(oracle(lit('hi'), 'fast'), ctx),
    ).resolves.toBe('fast-answer');
    expect(fast.calls).toHaveLength(1);
    expect(fallback.calls).toHaveLength(0);
  });

  it('falls back to ctx.lm when the modelHint is unknown', async () => {
    const fallback = new ScriptedLM('default', [oracleValue('fallback')]);
    const ctx = makeCtx({
      lm: fallback,
      lmRegistry: new Map(), // empty registry
    });
    await expect(
      evaluate(oracle(lit('hi'), 'does-not-exist'), ctx),
    ).resolves.toBe('fallback');
    expect(fallback.calls).toHaveLength(1);
  });

  it('throws BudgetError when callsUsed would exceed maxOracleCalls', async () => {
    const lm = new ScriptedLM('primary', [oracleValue('never')]);
    const ctx = makeCtx({ lm, budget: { maxOracleCalls: 0 } });
    await expect(
      evaluate(oracle(lit('hi')), ctx),
    ).rejects.toBeInstanceOf(BudgetError);
    // Budget enforcement is pre-call: the ScriptedLM must not be touched
    // once the ceiling is reached.
    expect(lm.calls).toHaveLength(0);
    // Counter still records the attempted slot, which is what the budget
    // check compared against. Callers that want "calls actually served"
    // should subtract trace failures; the counter is the monotonic lease
    // register.
    expect(ctx.callsUsed.current).toBe(1);
  });
});

// Vote: N lanes, majority / verifier, parallelism cap, budget.

describe('evaluate — vote', () => {
  it('majority reducer picks the most common answer', async () => {
    const lm = new ScriptedLM('primary', [
      '{"answer": "A"}',
      '{"answer": "B"}',
      '{"answer": "A"}',
      '{"answer": "A"}',
      '{"answer": "B"}',
    ]);
    const ctx = makeCtx({ lm, budget: { maxParallelism: 2 } });
    await expect(
      evaluate(vote(oracle(lit('q')), lit(5)), ctx),
    ).resolves.toBe('A');
    expect(lm.calls).toHaveLength(5);
    expect(ctx.callsUsed.current).toBe(5);
  });

  it('every lane consumes one budget slot', async () => {
    const lm = new ScriptedLM('primary', [
      '{"answer": "x"}',
      '{"answer": "x"}',
      '{"answer": "x"}',
    ]);
    const ctx = makeCtx({ lm, budget: { maxOracleCalls: 3 } });
    await evaluate(vote(oracle(lit('q')), lit(3)), ctx);
    expect(ctx.callsUsed.current).toBe(3);
  });

  it('stops on BudgetError when n exceeds maxOracleCalls', async () => {
    const lm = new ScriptedLM('primary', [
      '{"answer": "x"}',
      '{"answer": "x"}',
      '{"answer": "x"}',
    ]);
    const ctx = makeCtx({ lm, budget: { maxOracleCalls: 2 } });
    await expect(
      evaluate(vote(oracle(lit('q')), lit(5)), ctx),
    ).rejects.toBeInstanceOf(BudgetError);
  });

  it('honors maxParallelism: never more than cap lanes in flight at once', async () => {
    // Use 20 calls with a 3-cap and verify via a custom shared counter.
    const scripted = Array.from({ length: 20 }, () => '{"answer": "ok"}');
    let inFlight = 0;
    let maxInFlight = 0;

    class CountingLM extends BaseLM {
      constructor() {
        super({ model: 'counting' });
      }
      protected override async agenerate(
        _prompt?: string,
        _messages?: readonly Message[],
        _kwargs?: Record<string, unknown>,
      ): Promise<readonly LMOutput[]> {
        inFlight += 1;
        maxInFlight = Math.max(maxInFlight, inFlight);
        await new Promise((r) => setTimeout(r, 1));
        inFlight -= 1;
        return [scripted.shift() ?? '{"answer": "extra"}'];
      }
    }

    const lm = new CountingLM();
    const ctx = makeCtx({ lm, budget: { maxParallelism: 3 } });
    await evaluate(vote(oracle(lit('q')), lit(20)), ctx);
    expect(maxInFlight).toBeLessThanOrEqual(3);
    expect(maxInFlight).toBeGreaterThan(1);
  });

  it('majority tie-breaks on first-seen order for determinism', async () => {
    // Tie between "A" and "B" at 2-2; "A" appears first, so it wins.
    const lm = new ScriptedLM('primary', [
      '{"answer": "A"}',
      '{"answer": "B"}',
      '{"answer": "A"}',
      '{"answer": "B"}',
    ]);
    const ctx = makeCtx({ lm });
    await expect(
      evaluate(vote(oracle(lit('q')), lit(4)), ctx),
    ).resolves.toBe('A');
  });

  it('verifier reducer selects the first positively-verdicted candidate', async () => {
    // Three oracle answers, then three verifier verdicts: [false, true, false].
    // "B" wins because it earned the first `true` verdict.
    const lm = new ScriptedLM('primary', [
      '{"answer": "A"}',
      '{"answer": "B"}',
      '{"answer": "C"}',
      '{"verdict": false}',
      '{"verdict": true}',
      '{"verdict": false}',
    ]);
    const ctx = makeCtx({ lm, budget: { maxOracleCalls: 10 } });
    await expect(
      evaluate(vote(oracle(lit('q')), lit(3), 'verifier'), ctx),
    ).resolves.toBe('B');
    // 3 oracle + 3 verifier = 6 slots consumed.
    expect(ctx.callsUsed.current).toBe(6);
  });

  it('rejects n < 1 before touching the network', async () => {
    const lm = new ScriptedLM('primary', []);
    const ctx = makeCtx({ lm });
    await expect(
      evaluate(vote(oracle(lit('q')), lit(0)), ctx),
    ).rejects.toBeInstanceOf(ValueError);
    expect(lm.calls).toHaveLength(0);
  });

  it('rejects a non-oracle inner node', async () => {
    const ctx = makeCtx();
    await expect(
      evaluate(vote(lit('not-an-oracle') as CombinatorNode, lit(3)), ctx),
    ).rejects.toBeInstanceOf(ValueError);
  });
});

// Ensemble: per-model hints, confidence / majority / verifier reducers.

describe('evaluate — ensemble', () => {
  it('confidence reducer picks the highest-confidence response (default)', async () => {
    const fast = new ScriptedLM('fast', [
      '{"answer": "fast-pick", "confidence": 0.3}',
    ]);
    const precise = new ScriptedLM('precise', [
      '{"answer": "precise-pick", "confidence": 0.9}',
    ]);
    const registry = new Map([
      ['fast', fast as BaseLM],
      ['precise', precise as BaseLM],
    ]);
    const ctx = makeCtx({
      lm: new RejectingLM(),
      lmRegistry: registry,
    });
    await expect(
      evaluate(
        ensemble(oracle(lit('hi')), ['fast', 'precise']),
        ctx,
      ),
    ).resolves.toBe('precise-pick');
    expect(fast.calls).toHaveLength(1);
    expect(precise.calls).toHaveLength(1);
    expect(ctx.callsUsed.current).toBe(2);
  });

  it('majority reducer votes across model responses', async () => {
    const a = new ScriptedLM('a', ['{"answer": "yes"}']);
    const b = new ScriptedLM('b', ['{"answer": "no"}']);
    const c = new ScriptedLM('c', ['{"answer": "yes"}']);
    const registry = new Map<string, BaseLM>([
      ['a', a],
      ['b', b],
      ['c', c],
    ]);
    const ctx = makeCtx({
      lm: new RejectingLM(),
      lmRegistry: registry,
    });
    await expect(
      evaluate(
        ensemble(oracle(lit('hi')), ['a', 'b', 'c'], 'majority'),
        ctx,
      ),
    ).resolves.toBe('yes');
  });

  it('verifier reducer chains a second LM round to pick a candidate', async () => {
    const a = new ScriptedLM('a', ['{"answer": "alpha"}']);
    const b = new ScriptedLM('b', ['{"answer": "beta"}']);
    // Verifier probes go through the default LM (no modelHint). The first
    // candidate ("alpha") fails, the second ("beta") passes — beta wins.
    const fallback = new ScriptedLM('default', [
      '{"verdict": false}',
      '{"verdict": true}',
    ]);
    const ctx = makeCtx({
      lm: fallback,
      lmRegistry: new Map<string, BaseLM>([
        ['a', a],
        ['b', b],
      ]),
      budget: { maxOracleCalls: 10 },
    });
    await expect(
      evaluate(
        ensemble(oracle(lit('q')), ['a', 'b'], 'verifier'),
        ctx,
      ),
    ).resolves.toBe('beta');
    // 2 oracle (ensemble) + 2 verifier probes = 4 slots consumed.
    expect(ctx.callsUsed.current).toBe(4);
  });

  it('rejects an empty model list before any call', async () => {
    const lm = new ScriptedLM('primary', []);
    const ctx = makeCtx({ lm });
    await expect(
      evaluate(ensemble(oracle(lit('q')), []), ctx),
    ).rejects.toBeInstanceOf(ValueError);
    expect(lm.calls).toHaveLength(0);
  });

  it('rejects a non-oracle inner node', async () => {
    const ctx = makeCtx();
    await expect(
      evaluate(
        ensemble(lit('not-an-oracle') as CombinatorNode, ['fast']),
        ctx,
      ),
    ).rejects.toBeInstanceOf(ValueError);
  });

  it('rejects non-finite confidence as an error', async () => {
    // A non-numeric confidence string is caught by the adapter coercion
    // layer (AdapterParseError extends RuntimeError) before reaching the
    // evaluator's own ValueError path. Either way the call must reject.
    const a = new ScriptedLM('a', ['{"answer": "ok", "confidence": "not-a-number"}']);
    const ctx = makeCtx({
      lm: new RejectingLM(),
      lmRegistry: new Map<string, BaseLM>([['a', a]]),
    });
    await expect(
      evaluate(ensemble(oracle(lit('q')), ['a']), ctx),
    ).rejects.toBeInstanceOf(RuntimeError);
  });
});

// ---------------------------------------------------------------------------
// Budget exhaustion with trace preservation
// ---------------------------------------------------------------------------

describe('evaluate — budget exhaustion with trace', () => {
  it('preserves partial trace when oracle budget runs out mid-vote', async () => {
    // 5 vote lanes, budget of 3. The first 3 calls succeed, the 4th throws.
    // We verify partial trace entries exist for the work that completed.
    const lm = new ScriptedLM('primary', [
      '{"answer": "a"}',
      '{"answer": "b"}',
      '{"answer": "a"}',
      '{"answer": "extra"}',
      '{"answer": "extra"}',
    ]);
    const ctx = makeCtx({ lm, budget: { maxOracleCalls: 3, maxParallelism: 1 } });
    await expect(
      evaluate(vote(oracle(lit('q')), lit(5)), ctx),
    ).rejects.toBeInstanceOf(BudgetError);
    // Trace should contain entries for the work that happened before the
    // error. With parallelism=1, evaluation is sequential and predictable.
    expect(ctx.trace.length).toBeGreaterThan(0);
    // The budget counter should reflect attempted slots.
    expect(ctx.callsUsed.current).toBe(4);
    expect(lm.calls).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// Map parallelism against LM
// ---------------------------------------------------------------------------

describe('evaluate — map parallelism', () => {
  it('map(oracle, items) respects maxParallelism cap against a real LM', async () => {
    const items = Array.from({ length: 10 }, (_, i) => `item-${i}`);
    let inFlight = 0;
    let maxInFlight = 0;
    const scripted = Array.from({ length: 10 }, () => oracleValue('ok'));

    class TrackingLM extends BaseLM {
      constructor() {
        super({ model: 'tracking' });
      }
      protected override async agenerate(): Promise<readonly LMOutput[]> {
        inFlight += 1;
        maxInFlight = Math.max(maxInFlight, inFlight);
        await new Promise((r) => setTimeout(r, 1));
        inFlight -= 1;
        return [scripted.shift() ?? oracleValue('extra')];
      }
    }

    const lm = new TrackingLM();
    const plan = map(
      fn('item', oracle(vref('item'))),
      lit(items),
    );
    const ctx = makeCtx({ lm, budget: { maxParallelism: 2 } });
    const result = await evaluate(plan, ctx);
    expect(result).toHaveLength(10);
    expect(maxInFlight).toBeLessThanOrEqual(2);
    expect(maxInFlight).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Internal reducer — spot-check the tie-break rule directly
// ---------------------------------------------------------------------------

describe('evaluator internals — reducers', () => {
  it('modeOfStrings: plurality wins', () => {
    expect(__internal.modeOfStrings(['a', 'b', 'a', 'c', 'a'])).toBe('a');
  });

  it('modeOfStrings: ties break on first-seen order', () => {
    expect(__internal.modeOfStrings(['a', 'b', 'a', 'b'])).toBe('a');
    expect(__internal.modeOfStrings(['b', 'a', 'a', 'b'])).toBe('b');
  });

  it('modeOfStrings: throws on empty input', () => {
    expect(() => __internal.modeOfStrings([])).toThrow(ValueError);
  });

  it('resolveOracleLm: returns ctx.lm for missing hint', () => {
    const lm = new RejectingLM();
    const ctx = makeCtx({ lm });
    expect(__internal.resolveOracleLm(undefined, ctx)).toBe(lm);
  });

  it('resolveOracleLm: returns registry hit when present', () => {
    const fallback = new RejectingLM();
    const fast = new RejectingLM();
    const ctx = makeCtx({
      lm: fallback,
      lmRegistry: new Map<string, BaseLM>([['fast', fast]]),
    });
    expect(__internal.resolveOracleLm('fast', ctx)).toBe(fast);
  });

  it('resolveOracleLm: falls back to ctx.lm on unknown hint', () => {
    const fallback = new RejectingLM();
    const ctx = makeCtx({ lm: fallback });
    expect(__internal.resolveOracleLm('missing', ctx)).toBe(fallback);
  });
});

// ---------------------------------------------------------------------------
// Depth budget
// ---------------------------------------------------------------------------

describe('evaluate — depth budget', () => {
  it('throws BudgetError when context depth exceeds budget (pre-seeded)', async () => {
    const ctx = makeCtx({ budget: { maxDepth: 3 } });
    const lifted: EvaluationContext = { ...ctx, depth: 99 };
    await expect(evaluate(lit('x'), lifted)).rejects.toBeInstanceOf(
      BudgetError,
    );
  });

  it('increments depth on every recursive evaluate call', async () => {
    // maxDepth = 2 means depth 0, 1, 2 are allowed; depth 3 is rejected.
    // split(lit('ab'), lit(2)) evaluates: lit('ab') at depth 1, lit(2)
    // at depth 1, then split itself at depth 0. All at depth ≤ 2 → fine.
    const ctx = makeCtx({ budget: { maxDepth: 2 } });
    await expect(evaluate(split(lit('ab'), lit(2)), ctx)).resolves.toEqual([
      'a',
      'b',
    ]);
  });

  it('rejects naturally deep plans that exceed maxDepth', async () => {
    // maxDepth = 0 means only the outermost evaluate is allowed (depth 0);
    // its children evaluate at depth 1, which exceeds the budget.
    // split needs to evaluate lit('ab') at depth 1 → BudgetError.
    const ctx = makeCtx({ budget: { maxDepth: 0 } });
    await expect(
      evaluate(split(lit('ab'), lit(2)), ctx),
    ).rejects.toBeInstanceOf(BudgetError);
  });
});

// ---------------------------------------------------------------------------
// End-to-end integration of deterministic combinators
// ---------------------------------------------------------------------------

describe('evaluate — split → map(identity) → concat round-trip', () => {
  it('recovers the original string through the pipeline', async () => {
    const plan: CombinatorNode = concat(
      map(fn('chunk', vref('chunk')), split(lit('hello world'), lit(3))),
      lit(''),
    );
    await expect(evaluate(plan, makeCtx())).resolves.toBe('hello world');
  });

  it('survives non-trivial sizes under a tight parallelism cap', async () => {
    const original = 'the quick brown fox jumps over the lazy dog';
    const plan: CombinatorNode = concat(
      map(fn('chunk', vref('chunk')), split(lit(original), lit(7))),
      lit(''),
    );
    const ctx = makeCtx({ budget: { maxParallelism: 2 } });
    await expect(evaluate(plan, ctx)).resolves.toBe(original);
  });
});

// ---------------------------------------------------------------------------
// Trace emission
// ---------------------------------------------------------------------------

describe('evaluate — trace emission', () => {
  it('pushes one entry per evaluate() invocation in completion order', async () => {
    const ctx = makeCtx();
    await evaluate(split(lit('abc'), lit(2)), ctx);
    // split evaluates input (lit) then k (lit) then itself
    expect(ctx.trace.map((e) => e.nodeTag)).toEqual([
      'literal',
      'literal',
      'split',
    ]);
    ctx.trace.forEach((entry, index) => {
      expect(entry.step).toBe(index);
      expect(entry.ok).toBe(true);
      expect(entry.durationMs).toBeGreaterThanOrEqual(0);
    });
  });

  it('records failures with the cause attached', async () => {
    const ctx = makeCtx();
    await expect(evaluate(vref('missing'), ctx)).rejects.toBeInstanceOf(
      ValueError,
    );
    expect(ctx.trace).toHaveLength(1);
    const entry = ctx.trace[0];
    expect(entry?.ok).toBe(false);
    expect(entry?.cause).toBeInstanceOf(ValueError);
  });

  it('survives mid-walk failures: partial trace is preserved on throw', async () => {
    const ctx = makeCtx();
    // split fails after its two literal children succeed because k is
    // non-numeric.
    await expect(
      evaluate(split(lit('abc'), lit('nope')), ctx),
    ).rejects.toBeInstanceOf(ValueError);
    expect(ctx.trace.map((e) => e.nodeTag)).toEqual([
      'literal',
      'literal',
      'split',
    ]);
    expect(ctx.trace[0]?.ok).toBe(true);
    expect(ctx.trace[1]?.ok).toBe(true);
    expect(ctx.trace[2]?.ok).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Internal helpers (exported via `__internal` for coverage)
// ---------------------------------------------------------------------------

describe('evaluator internals', () => {
  it('partitionString matches the documented chunking', () => {
    expect(__internal.partitionString('abcdef', 3)).toEqual(['ab', 'cd', 'ef']);
    expect(__internal.partitionString('abcdefg', 3)).toEqual([
      'abc',
      'de',
      'fg',
    ]);
    expect(__internal.partitionString('', 5)).toEqual([]);
  });

  it('runBounded preserves ordering with single lane', async () => {
    const seen: number[] = [];
    await __internal.runBounded([1, 2, 3, 4], 1, async (x) => {
      seen.push(x);
      return x * 2;
    });
    expect(seen).toEqual([1, 2, 3, 4]);
  });

  it('runBounded enforces the parallelism cap', async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const items = Array.from({ length: 16 }, (_, i) => i);
    await __internal.runBounded(items, 4, async (x) => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      // Yield the event loop so siblings can ramp up and hit the cap.
      await new Promise<void>((r) => setTimeout(r, 0));
      inFlight -= 1;
      return x;
    });
    expect(maxInFlight).toBeLessThanOrEqual(4);
    expect(maxInFlight).toBeGreaterThan(1); // confirms parallelism is real
  });

  it('mergeBudget returns DEFAULT_BUDGET when no override is given', () => {
    const merged = __internal.mergeBudget(undefined);
    expect(merged.maxOracleCalls).toBe(200);
    expect(merged.maxParallelism).toBe(16);
  });

  it('mergeBudget applies partial overrides', () => {
    const merged = __internal.mergeBudget({ maxParallelism: 2 });
    expect(merged.maxParallelism).toBe(2);
    expect(merged.maxOracleCalls).toBe(200);
  });
});
