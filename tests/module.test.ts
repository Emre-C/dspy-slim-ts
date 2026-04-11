import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { Prediction } from '../src/prediction.js';
import {
  Module,
  markPredictor,
} from '../src/module.js';
import { signatureFromString } from '../src/signature.js';

interface FixtureRecord {
  readonly [key: string]: FixtureValue;
}

type FixtureValue = FixtureNode | readonly FixtureValue[] | FixtureRecord;

interface FixtureNode {
  type: 'Module' | 'Predict' | 'ChainOfThought' | 'ReAct';
  signature?: string;
  _compiled?: boolean;
  id?: string;
  attrs?: Record<string, FixtureValue>;
}

interface FixtureCase {
  id: string;
  structure: FixtureNode;
  expected_named_parameters?: [string, string][];
  expected_named_predictors?: [string, string][];
  expected_named_parameters_count?: number;
}

class ModuleFixture extends Module {
  override forward(_kwargs: Record<string, unknown> = {}): Prediction {
    return Prediction.create({});
  }
}

type ModuleFixtureWithInner = ModuleFixture & { inner?: Predict };
type ModuleFixtureWithPredictors = ModuleFixture & {
  predictor?: Predict;
  nested?: ModuleFixtureWithInner;
  predict?: Predict;
};

class Predict extends Module {
  signature;
  lm: unknown | null = null;
  demos: unknown[] = [];
  traces: unknown[] = [];

  constructor(signature = 'input -> output') {
    super();
    this.signature = signatureFromString(signature);
    markPredictor(this);
  }

  reset(): void {
    this.lm = null;
    this.demos = [];
    this.traces = [];
  }

  override forward(_kwargs: Record<string, unknown> = {}): Prediction {
    return Prediction.create({});
  }
}

class ChainOfThought extends Module {
  predict = new Predict();

  override forward(kwargs: Record<string, unknown>): Prediction {
    return this.predict.forward(kwargs);
  }
}

class ReAct extends Module {
  react = new Predict();
  extract = new ChainOfThought();

  override forward(): Prediction {
    return Prediction.create({});
  }
}

function isFixtureNode(value: FixtureValue): value is FixtureNode {
  return typeof value === 'object' && value !== null && 'type' in value;
}

function buildValue(
  value: FixtureValue,
  shared: Map<string, unknown>,
): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => buildValue(item, shared));
  }

  if (isFixtureNode(value)) {
    return buildNode(value, shared);
  }

  const result: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) {
    result[key] = buildValue(item, shared);
  }
  return result;
}

function buildNode(node: FixtureNode, shared: Map<string, unknown>): Module {
  if (node.id && shared.has(node.id)) {
    return shared.get(node.id) as Module;
  }

  let instance: Module;
  switch (node.type) {
    case 'Module':
      instance = new ModuleFixture();
      break;
    case 'Predict':
      instance = new Predict(node.signature);
      break;
    case 'ChainOfThought':
      instance = new ChainOfThought();
      break;
    case 'ReAct':
      instance = new ReAct();
      break;
    default:
      throw new Error(`Unknown fixture node type: ${node.type}`);
  }

  if (node.id) {
    shared.set(node.id, instance);
  }

  instance._compiled = node._compiled ?? false;

  if (node.attrs) {
    for (const [key, value] of Object.entries(node.attrs)) {
      Reflect.set(instance, key, buildValue(value, shared));
    }
  }

  return instance;
}

const fixture = JSON.parse(
  readFileSync(
    new URL('../../spec/fixtures/module_tree_walk.json', import.meta.url),
    'utf-8',
  ),
) as { cases: FixtureCase[] };

describe('Module tree walk (spec fixtures)', () => {
  for (const c of fixture.cases) {
    it(c.id, () => {
      const root = buildNode(c.structure, new Map());

      if (c.expected_named_parameters) {
        expect(
          root.namedParameters().map(([name, value]) => [name, value.constructor.name]),
        ).toEqual(c.expected_named_parameters);
      }

      if (c.expected_named_predictors) {
        expect(
          root.namedPredictors().map(([name, value]) => [name, value.constructor.name]),
        ).toEqual(c.expected_named_predictors);
      }

      if (c.expected_named_parameters_count !== undefined) {
        expect(root.namedParameters()).toHaveLength(c.expected_named_parameters_count);
      }
    });
  }
});

describe('Module hardening', () => {
  it('treats predictor identity as terminal during parameter traversal', () => {
    const predict = new Predict('question -> answer');
    const wrapper = new ModuleFixture() as ModuleFixtureWithPredictors;
    const nested = new ModuleFixture() as ModuleFixtureWithInner;

    nested.inner = predict;
    wrapper.predictor = predict;
    wrapper.nested = nested;

    expect(wrapper.namedParameters().map(([name]) => name)).toEqual(['predictor']);
  });

  it('resetCopy deep-clones the module graph and resets parameterized leaves', () => {
    const root = new ModuleFixture() as ModuleFixtureWithPredictors;
    const predict = new Predict('question -> answer');
    predict.lm = { id: 'lm-a' };
    predict.demos = [{ question: 'Why?' }];
    predict.traces = [{ answer: 'Because.' }];
    root.predict = predict;

    const copy = root.resetCopy() as ModuleFixture & { predict: Predict };

    expect(copy).not.toBe(root);
    expect(copy.predict).not.toBe(predict);
    expect(copy.predict.lm).toBeNull();
    expect(copy.predict.demos).toEqual([]);
    expect(copy.predict.traces).toEqual([]);
    expect(predict.lm).toEqual({ id: 'lm-a' });
  });
});
