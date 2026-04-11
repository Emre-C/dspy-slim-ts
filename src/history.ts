/**
 * §7.3 — History: immutable conversation history payload.
 */

import { ValueError } from './exceptions.js';
import { isPlainObject } from './guards.js';
import { serializeOwnedValue, snapshotRecord } from './owned_value.js';

export interface HistoryMessage {
  readonly [key: string]: unknown;
}

export function isHistoryLike(value: unknown): value is { readonly messages: readonly Record<string, unknown>[] } {
  if (!isPlainObject(value) || !('messages' in value)) {
    return false;
  }

  const messages = value.messages;
  return Array.isArray(messages) && messages.every(isPlainObject);
}

export class History {
  readonly messages: readonly HistoryMessage[];

  constructor(messages: readonly HistoryMessage[]) {
    if (!Array.isArray(messages)) {
      throw new ValueError('History messages must be an array');
    }

    this.messages = Object.freeze(messages.map((message, index) => {
      if (!isPlainObject(message)) {
        throw new ValueError(`History message at index ${index} must be a plain object`);
      }

      return Object.freeze(snapshotRecord(message));
    }));
  }

  toDict(): { messages: readonly HistoryMessage[] } {
    return {
      messages: this.messages.map((message) => serializeOwnedValue(message) as HistoryMessage),
    };
  }

  toJSON(): { messages: readonly HistoryMessage[] } {
    return this.toDict();
  }
}
