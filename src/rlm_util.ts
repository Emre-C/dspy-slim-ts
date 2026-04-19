/**
 * Small shared helpers for RLM v2 runtime modules (no dependency on
 * `rlm_types` / `rlm_memory` to avoid import cycles).
 */

export function typeName(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}
