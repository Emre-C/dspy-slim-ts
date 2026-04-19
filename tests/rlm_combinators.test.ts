import { describe, expect, it } from 'vitest';
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

// Every constructor test asserts the full shape of the returned node. The
// contract for plan authors is that constructors produce plain data and
// omit optional fields so `JSON.stringify(plan)` is a stable canonical form
// under `exactOptionalPropertyTypes`.

describe('combinator constructors — leaf nodes', () => {
  it('lit builds a literal node carrying the value verbatim', () => {
    expect(lit('hello')).toEqual({ tag: 'literal', value: 'hello' });
    expect(lit(42)).toEqual({ tag: 'literal', value: 42 });
    expect(lit(['a', 'b'])).toEqual({ tag: 'literal', value: ['a', 'b'] });
  });

  it('vref builds a var node with the named binding', () => {
    expect(vref('x')).toEqual({ tag: 'var', name: 'x' });
  });
});

describe('combinator constructors — deterministic combinators', () => {
  it('split carries input and k children as AST nodes', () => {
    expect(split(lit('abcd'), lit(2))).toEqual({
      tag: 'split',
      input: { tag: 'literal', value: 'abcd' },
      k: { tag: 'literal', value: 2 },
    });
  });

  it('peek carries input, start, and end children', () => {
    expect(peek(lit('abcd'), lit(0), lit(2))).toEqual({
      tag: 'peek',
      input: { tag: 'literal', value: 'abcd' },
      start: { tag: 'literal', value: 0 },
      end: { tag: 'literal', value: 2 },
    });
  });

  it('map embeds a CombinatorFn with param and body', () => {
    const body = vref('x');
    const node = map(fn('x', body), lit(['a', 'b']));
    expect(node).toEqual({
      tag: 'map',
      fn: { param: 'x', body: { tag: 'var', name: 'x' } },
      items: { tag: 'literal', value: ['a', 'b'] },
    });
  });

  it('filter embeds a CombinatorFn as pred', () => {
    const node = filter(fn('x', lit(true)), lit(['a']));
    expect(node).toEqual({
      tag: 'filter',
      pred: { param: 'x', body: { tag: 'literal', value: true } },
      items: { tag: 'literal', value: ['a'] },
    });
  });

  it('reduce omits init when not provided', () => {
    const node = reduce(bop('acc', 'x', vref('acc')), lit([1, 2, 3]));
    expect(node).toEqual({
      tag: 'reduce',
      op: {
        left: 'acc',
        right: 'x',
        body: { tag: 'var', name: 'acc' },
      },
      items: { tag: 'literal', value: [1, 2, 3] },
    });
    expect(Object.prototype.hasOwnProperty.call(node, 'init')).toBe(false);
  });

  it('reduce includes init when provided', () => {
    const node = reduce(
      bop('acc', 'x', vref('acc')),
      lit([1, 2, 3]),
      lit(0),
    );
    expect(node).toEqual({
      tag: 'reduce',
      op: {
        left: 'acc',
        right: 'x',
        body: { tag: 'var', name: 'acc' },
      },
      items: { tag: 'literal', value: [1, 2, 3] },
      init: { tag: 'literal', value: 0 },
    });
  });

  it('concat omits separator when not provided', () => {
    const node = concat(lit(['a', 'b']));
    expect(node).toEqual({
      tag: 'concat',
      items: { tag: 'literal', value: ['a', 'b'] },
    });
    expect(Object.prototype.hasOwnProperty.call(node, 'separator')).toBe(false);
  });

  it('concat includes separator when provided', () => {
    const node = concat(lit(['a', 'b']), lit(','));
    expect(node).toEqual({
      tag: 'concat',
      items: { tag: 'literal', value: ['a', 'b'] },
      separator: { tag: 'literal', value: ',' },
    });
  });

  it('cross builds a pair-producing node from two child plans', () => {
    expect(cross(lit(['a']), lit(['x']))).toEqual({
      tag: 'cross',
      left: { tag: 'literal', value: ['a'] },
      right: { tag: 'literal', value: ['x'] },
    });
  });
});

describe('combinator constructors — neural combinators', () => {
  it('vote omits reducer when unspecified', () => {
    const node = vote(oracle(lit('p')), lit(3));
    expect(node).toEqual({
      tag: 'vote',
      oracle: { tag: 'oracle', prompt: { tag: 'literal', value: 'p' } },
      n: { tag: 'literal', value: 3 },
    });
    expect(Object.prototype.hasOwnProperty.call(node, 'reducer')).toBe(false);
  });

  it('vote accepts all three reducer strategies', () => {
    const strategies = ['majority', 'mode', 'verifier'] as const;
    for (const reducer of strategies) {
      const node = vote(oracle(lit('p')), lit(3), reducer);
      expect(node).toMatchObject({ tag: 'vote', reducer });
    }
  });

  it('ensemble omits reducer when unspecified', () => {
    const node = ensemble(oracle(lit('p')), ['fast', 'deep']);
    expect(node).toEqual({
      tag: 'ensemble',
      oracle: { tag: 'oracle', prompt: { tag: 'literal', value: 'p' } },
      models: ['fast', 'deep'],
    });
    expect(Object.prototype.hasOwnProperty.call(node, 'reducer')).toBe(false);
  });

  it('ensemble accepts all three reducer strategies', () => {
    const strategies = ['majority', 'confidence', 'verifier'] as const;
    for (const reducer of strategies) {
      const node = ensemble(oracle(lit('p')), ['a', 'b'], reducer);
      expect(node).toMatchObject({ tag: 'ensemble', reducer });
    }
  });

  it('oracle omits both optional fields when unspecified', () => {
    const node = oracle(lit('p'));
    expect(node).toEqual({
      tag: 'oracle',
      prompt: { tag: 'literal', value: 'p' },
    });
    expect(Object.prototype.hasOwnProperty.call(node, 'modelHint')).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(node, 'effectHandlers')).toBe(
      false,
    );
  });

  it('oracle includes only modelHint when only modelHint is provided', () => {
    const node = oracle(lit('p'), 'fast');
    expect(node).toEqual({
      tag: 'oracle',
      prompt: { tag: 'literal', value: 'p' },
      modelHint: 'fast',
    });
    expect(Object.prototype.hasOwnProperty.call(node, 'effectHandlers')).toBe(
      false,
    );
  });

  it('oracle includes only effectHandlers when only effectHandlers is provided', () => {
    const node = oracle(lit('p'), undefined, ['ReadContext', 'WriteMemory']);
    expect(node).toEqual({
      tag: 'oracle',
      prompt: { tag: 'literal', value: 'p' },
      effectHandlers: ['ReadContext', 'WriteMemory'],
    });
    expect(Object.prototype.hasOwnProperty.call(node, 'modelHint')).toBe(false);
  });

  it('oracle includes both when both are provided', () => {
    const node = oracle(lit('p'), 'deep', ['ReadContext']);
    expect(node).toEqual({
      tag: 'oracle',
      prompt: { tag: 'literal', value: 'p' },
      modelHint: 'deep',
      effectHandlers: ['ReadContext'],
    });
  });
});

describe('combinator constructors — function bodies', () => {
  it('fn pairs a param name with a body AST', () => {
    expect(fn('x', vref('x'))).toEqual({
      param: 'x',
      body: { tag: 'var', name: 'x' },
    });
  });

  it('bop pairs left/right param names with a body AST', () => {
    expect(bop('acc', 'elem', vref('acc'))).toEqual({
      left: 'acc',
      right: 'elem',
      body: { tag: 'var', name: 'acc' },
    });
  });
});

describe('combinator AST — serializability invariant', () => {
  it('plans round-trip through JSON unchanged', () => {
    const plan: CombinatorNode = concat(
      map(fn('chunk', vref('chunk')), split(lit('hello world'), lit(3))),
      lit(''),
    );
    const roundTripped = JSON.parse(JSON.stringify(plan)) as unknown;
    expect(roundTripped).toEqual(plan);
  });

  it('omitted optional fields survive the round-trip', () => {
    const plan: CombinatorNode = reduce(
      bop('a', 'b', vref('a')),
      lit([1, 2]),
    );
    const text = JSON.stringify(plan);
    expect(text).not.toContain('"init"');
    const roundTripped = JSON.parse(text) as CombinatorNode;
    expect(roundTripped).toEqual(plan);
    expect(Object.prototype.hasOwnProperty.call(roundTripped, 'init')).toBe(
      false,
    );
  });
});
