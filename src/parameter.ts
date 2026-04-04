const PARAMETER_BRAND = Symbol('dspy.parameter');

function isObjectLike(value: unknown): value is object {
  return typeof value === 'object' && value !== null;
}

export interface Parameter {
  readonly [PARAMETER_BRAND]: true;
}

export function markParameter<T extends object>(value: T): T & Parameter {
  if (!isObjectLike(value)) {
    throw new Error('Only objects can be marked as Parameters');
  }

  if (!isParameter(value)) {
    Object.defineProperty(value, PARAMETER_BRAND, {
      value: true,
      enumerable: false,
      configurable: false,
      writable: false,
    });
  }

  return value as T & Parameter;
}

export function isParameter(value: unknown): value is Parameter {
  return isObjectLike(value) && (value as Partial<Parameter>)[PARAMETER_BRAND] === true;
}

export { PARAMETER_BRAND };
