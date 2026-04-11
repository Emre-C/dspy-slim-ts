/**
 * Split a string at a delimiter, respecting nested brackets and quotes.
 *
 * Handles `"`, `'`, and backtick quotes with backslash escaping.
 * Respects `()`, `[]`, `{}` nesting — splits only at depth zero.
 *
 * When `validate` is true, throws on mismatched brackets, unterminated
 * quotes, and unbalanced delimiters.
 */

import { ValueError } from './exceptions.js';

export function splitTopLevel(
  input: string,
  delimiter = ',',
  validate = false,
): string[] {
  const parts: string[] = [];
  const stack: string[] = [];
  let activeQuote: string | null = null;
  let escaping = false;
  let start = 0;

  for (let index = 0; index < input.length; index += 1) {
    const char = input[index]!;

    if (activeQuote !== null) {
      if (escaping) {
        escaping = false;
        continue;
      }

      if (char === '\\') {
        escaping = true;
        continue;
      }

      if (char === activeQuote) {
        activeQuote = null;
      }

      continue;
    }

    if (char === '"' || char === "'" || char === '`') {
      activeQuote = char;
      continue;
    }

    if (char === '(' || char === '[' || char === '{') {
      stack.push(char);
      continue;
    }

    if (char === ')' || char === ']' || char === '}') {
      if (validate) {
        const open = stack.pop();
        const matches =
          (char === ')' && open === '(')
          || (char === ']' && open === '[')
          || (char === '}' && open === '{');

        if (!matches) {
          throw new ValueError(`Mismatched delimiter in "${input}"`);
        }
      } else {
        stack.pop();
      }

      continue;
    }

    if (stack.length === 0 && input.startsWith(delimiter, index)) {
      parts.push(input.slice(start, index));
      start = index + delimiter.length;
      index += delimiter.length - 1;
    }
  }

  if (validate) {
    if (activeQuote !== null) {
      throw new ValueError(`Unterminated string literal in "${input}"`);
    }

    if (stack.length > 0) {
      throw new ValueError(`Unbalanced delimiters in "${input}"`);
    }
  }

  parts.push(input.slice(start));
  return parts;
}
