/**
 * §3 — Module system.
 *
 * Traversal is intentionally runtime-branded rather than inheritance-only so a
 * future Predict can be both a Module and a Parameter without losing identity.
 */

import { isObjectLike, isPlainObject } from './guards.js';
import type { Prediction } from './prediction.js';
import { type Parameter, isParameter, markParameter } from './parameter.js';
import { settings } from './settings.js';

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
  lm: unknown | null;
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

export abstract class Module extends BaseModule {
  _compiled = false;
  callbacks: unknown[] = [];
  history: unknown[] = [];

  abstract forward(kwargs: Record<string, unknown>): Prediction;

  async aforward(kwargs: Record<string, unknown>): Promise<Prediction> {
    return this.forward(kwargs);
  }

  call(kwargs: Record<string, unknown>): Prediction {
    const callerModules = appendUnique(settings.callerModules, this);
    return settings.context({ callerModules }, () => this.forward(kwargs));
  }

  async acall(kwargs: Record<string, unknown>): Promise<Prediction> {
    const callerModules = appendUnique(settings.callerModules, this);
    return settings.context({ callerModules }, () => this.aforward(kwargs));
  }

  namedPredictors(): Array<[string, BaseModule & PredictorLike]> {
    return this.namedParameters().flatMap(([name, parameter]) => (
      isPredictorLike(parameter) ? [[name, parameter]] : []
    ));
  }

  predictors(): Array<BaseModule & PredictorLike> {
    return this.namedPredictors().map(([, predictor]) => predictor);
  }

  setLm(lm: unknown): void {
    for (const predictor of this.predictors()) {
      predictor.lm = lm;
    }
  }

  getLm(): unknown {
    for (const predictor of this.predictors()) {
      if (predictor.lm !== null) {
        return predictor.lm;
      }
    }

    return settings.lm;
  }
}

export { PREDICTOR_BRAND };
