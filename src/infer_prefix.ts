/**
 * §1.5 — Pure function: field name → human-readable prefix.
 *
 * Algorithm:
 *   1. Insert `_` before `[A-Z][a-z]+` preceded by another char
 *   2. Insert `_` between `[a-z0-9]` and `[A-Z]`
 *   3. Insert `_` between `[a-zA-Z]` and `\d`, and between `\d` and `[a-zA-Z]`
 *   4. Split on `_`
 *   5. Map: all-uppercase words preserved; others capitalized
 *   6. Join with `" "`
 */
export function inferPrefix(name: string): string {
  let s = name;

  s = s.replace(/(.(?=[A-Z][a-z]))/g, '$1_');
  s = s.replace(/([a-z0-9])([A-Z])/g, '$1_$2');
  s = s.replace(/([a-zA-Z])(\d)/g, '$1_$2');
  s = s.replace(/(\d)([a-zA-Z])/g, '$1_$2');

  const parts = s.split('_').filter((p) => p.length > 0);

  const mapped = parts.map((word) => {
    if (word === word.toUpperCase() && /[a-zA-Z]/.test(word)) {
      return word;
    }
    return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
  });

  return mapped.join(' ');
}
