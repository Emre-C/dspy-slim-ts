/**
 * §3 — Module system.
 *
 * Traversal is intentionally runtime-branded rather than inheritance-only so a
 * future Predict can be both a Module and a Parameter without losing identity.
 */

import { isObjectLike, isPlainObject } from './guards.js';
import type { HistoryEntry } from './history_entry.js';
import type { Prediction } from './prediction.js';
import { type Parameter, isParameter, markParameter } from './parameter.js';
import type { Callback } from './callback.js';
import { runWithCallbacks, runWithCallbacksAsync } from './callback.js';
import { settings } from './settings.js';
import type { LMLike } from './types.js';

const PREDICTOR_BRAND = Symbol('dspy.predictor');

function appendUnique<T extends object>(items: readonly T[], item: T): readonly T[] {
  return items.includes(item) ? items : Object.freeze([...items, item]);
}

function cloneValue<T>(value: T, seen: Map<object, unknown>): T {
  if (!isObjectLike(value)) {
    return value;
  }

  if (seen.has(value)) {
    return seen.get(value) as T;
  }

  if (value instanceof Date) {
    return new Date(value.getTime()) as T;
  }

  if (Array.isArray(value)) {
    const clone: unknown[] = [];
    seen.set(value, clone);
    for (const item of value) {
      clone.push(cloneValue(item, seen));
    }
    return clone as T;
  }

  if (value instanceof Map) {
    const clone = new Map<unknown, unknown>();
    seen.set(value, clone);
    for (const [key, item] of value) {
      clone.set(cloneValue(key, seen), cloneValue(item, seen));
    }
    return clone as T;
  }

  if (value instanceof Set) {
    const clone = new Set<unknown>();
    seen.set(value, clone);
    for (const item of value) {
      clone.add(cloneValue(item, seen));
    }
    return clone as T;
  }

  // Descriptor-copying opaque class instances breaks private slots and host
  // resources. Only recurse into module nodes and plain data containers.
  if (!(value instanceof BaseModule) && !isPlainObject(value)) {
    return value;
  }

  const clone = Object.create(Object.getPrototypeOf(value)) as Record<PropertyKey, unknown>;
  seen.set(value, clone);

  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor) {
      continue;
    }

    if ('value' in descriptor) {
      descriptor.value = cloneValue(descriptor.value, seen);
    }

    Object.defineProperty(clone, key, descriptor);
  }

  return clone as T;
}

function maybeResetParameter(parameter: Parameter): void {
  const candidate = parameter as Parameter & { reset?: () => void };
  if (typeof candidate.reset === 'function') {
    candidate.reset();
  }
}

export interface PredictorLike extends Parameter {
  readonly [PREDICTOR_BRAND]: true;
  lm: LMLike | null;
}

export function markPredictor<T extends BaseModule & object>(value: T): T & PredictorLike {
  const parameter = markParameter(value);

  if (!isPredictorLike(parameter)) {
    Object.defineProperty(parameter, PREDICTOR_BRAND, {
      value: true,
      enumerable: false,
      configurable: false,
      writable: false,
    });
  }

  return parameter as T & PredictorLike;
}

export function isPredictorLike(value: unknown): value is BaseModule & PredictorLike {
  return value instanceof BaseModule
    && isParameter(value)
    && (value as Partial<PredictorLike>)[PREDICTOR_BRAND] === true;
}

export abstract class BaseModule {
  namedParameters(): Array<[string, Parameter]> {
    const visited = new Set<object>();
    const namedParameters: Array<[string, Parameter]> = [];

    const addParameter = (name: string, value: unknown): void => {
      if (isParameter(value)) {
        if (!visited.has(value)) {
          visited.add(value);
          namedParameters.push([name, value]);
        }
        return;
      }

      if (value instanceof BaseModule) {
        if (!(value instanceof Module) || !value._compiled) {
          for (const [subName, parameter] of value.namedParameters()) {
            addParameter(`${name}.${subName}`, parameter);
          }
        }
      }
    };

    if (isParameter(this)) {
      addParameter('self', this);
    }

    for (const [name, value] of Object.entries(this as Record<string, unknown>)) {
      if (isParameter(value)) {
        addParameter(name, value);
      } else if (value instanceof BaseModule) {
        if (!(value instanceof Module) || !value._compiled) {
          for (const [subName, parameter] of value.namedParameters()) {
            addParameter(`${name}.${subName}`, parameter);
          }
        }
      } else if (Array.isArray(value)) {
        for (const [index, item] of value.entries()) {
          addParameter(`${name}[${index}]`, item);
        }
      } else if (isPlainObject(value)) {
        for (const [key, item] of Object.entries(value)) {
          addParameter(`${name}['${key}']`, item);
        }
      }
    }

    return namedParameters;
  }

  parameters(): Parameter[] {
    return this.namedParameters().map(([, parameter]) => parameter);
  }

  deepcopy(): this {
    return cloneValue(this, new Map()) as this;
  }

  resetCopy(): this {
    const copy = this.deepcopy();
    for (const parameter of copy.parameters()) {
      maybeResetParameter(parameter);
    }
    return copy;
  }
}

/**
 * Generic base for every DSPy module. `TInputs` names the kwargs shape
 * accepted by `forward`/`call`; `TOutputs` names the output record shape
 * carried by the returned `Prediction`. Both default to
 * `Record<string, unknown>` so existing subclasses that did not declare
 * generics keep their current (permissive) runtime surface.
 *
 * Typed subclasses (`Predict<'q -> a'>`, `ChainOfThought<'q -> a'>`, etc.)
 * substitute narrower types here so that `.forward({...})` gets
 * excess-property checks and the returned prediction exposes `TOutputs`
 * through `Prediction.getTyped(...)`.
 */
export abstract class Module<
  TInputs extends Record<string, unknown> = Record<string, unknown>,
  TOutputs extends Record<string, unknown> = Record<string, unknown>,
> extends BaseModule {
  _compiled = false;
  callbacks: Callback[] = [];
  history: HistoryEntry[] = [];

  abstract forward(kwargs: TInputs): Prediction<TOutputs>;

  async aforward(kwargs: TInputs): Promise<Prediction<TOutputs>> {
    return this.forward(kwargs);
  }

  call(kwargs: TInputs): Prediction<TOutputs> {
    return runWithCallbacks({
      kind: 'module',
      instance: this,
      inputs: kwargs,
      execute: () => {
        const callerModules = appendUnique(settings.callerModules, this);
        return settings.context({ callerModules }, () => this.forward(kwargs));
      },
    }) as Prediction<TOutputs>;
  }

  async acall(kwargs: TInputs): Promise<Prediction<TOutputs>> {
    return (await runWithCallbacksAsync({
      kind: 'module',
      instance: this,
      inputs: kwargs,
      execute: async () => {
        const callerModules = appendUnique(settings.callerModules, this);
        return settings.context({ callerModules }, () => this.aforward(kwargs));
      },
    })) as Prediction<TOutputs>;
  }

  namedPredictors(): Array<[string, BaseModule & PredictorLike]> {
    return this.namedParameters().flatMap(([name, parameter]) => (
      isPredictorLike(parameter) ? [[name, parameter]] : []
    ));
  }

  predictors(): Array<BaseModule & PredictorLike> {
    return this.namedPredictors().map(([, predictor]) => predictor);
  }

  setLm(lm: LMLike): void {
    for (const predictor of this.predictors()) {
      predictor.lm = lm;
    }
  }

  getLm(): LMLike | null {
    for (const predictor of this.predictors()) {
      if (predictor.lm !== null) {
        return predictor.lm;
      }
    }

    return settings.lm;
  }
}
