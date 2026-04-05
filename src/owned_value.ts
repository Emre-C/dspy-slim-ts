import { isObjectLike, isPlainObject } from './guards.js'

type DictLike = Record<string, unknown>

function hasToDict(value: unknown): value is { toDict(): DictLike } {
  return isObjectLike(value) && 'toDict' in value && typeof value.toDict === 'function'
}

function snapshotPlainObject(value: DictLike): DictLike {
  const snapshot: DictLike = {}

  for (const [key, item] of Object.entries(value)) {
    snapshot[key] = snapshotOwnedValue(item)
  }

  return Object.freeze(snapshot)
}

function serializePlainObject(value: DictLike): DictLike {
  const serialized: DictLike = {}

  for (const [key, item] of Object.entries(value)) {
    serialized[key] = serializeOwnedValue(item)
  }

  return serialized
}

export function snapshotOwnedValue(value: unknown): unknown {
  if (!isObjectLike(value)) {
    return value
  }

  if (value instanceof Date) {
    return new Date(value.getTime())
  }

  if (Array.isArray(value)) {
    return Object.freeze(value.map((item) => snapshotOwnedValue(item)))
  }

  if (hasToDict(value)) {
    return snapshotPlainObject(value.toDict())
  }

  if (isPlainObject(value)) {
    return snapshotPlainObject(value)
  }

  return value
}

export function snapshotRecord(record: DictLike | undefined): DictLike {
  if (record === undefined) {
    return {}
  }

  const snapshot: DictLike = {}

  for (const [key, value] of Object.entries(record)) {
    snapshot[key] = snapshotOwnedValue(value)
  }

  return snapshot
}

export function serializeOwnedValue(value: unknown): unknown {
  if (!isObjectLike(value)) {
    return value
  }

  if (value instanceof Date) {
    return new Date(value.getTime())
  }

  if (Array.isArray(value)) {
    return value.map((item) => serializeOwnedValue(item))
  }

  if (hasToDict(value)) {
    return serializePlainObject(value.toDict())
  }

  if (isPlainObject(value)) {
    return serializePlainObject(value)
  }

  return value
}

export function ownedValueEquals(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) {
    return true
  }

  if (a instanceof Date && b instanceof Date) {
    return a.getTime() === b.getTime()
  }

  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) {
      return false
    }

    for (let i = 0; i < a.length; i++) {
      if (!ownedValueEquals(a[i], b[i])) {
        return false
      }
    }

    return true
  }

  if (isPlainObject(a) && isPlainObject(b)) {
    const aKeys = Object.keys(a)
    const bKeys = Object.keys(b)

    if (aKeys.length !== bKeys.length) {
      return false
    }

    for (const key of aKeys) {
      if (!(key in b)) {
        return false
      }

      if (!ownedValueEquals(a[key], b[key])) {
        return false
      }
    }

    return true
  }

  return false
}
