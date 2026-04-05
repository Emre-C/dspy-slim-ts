/**
 * Shared runtime type guards.
 *
 * Every module that needs isPlainObject or isObjectLike imports from here.
 * Zero dependencies — safe to import from any layer.
 */

export function isObjectLike(value: unknown): value is object {
  return typeof value === 'object' && value !== null;
}

export function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!isObjectLike(value)) {
    return false;
  }

  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}
