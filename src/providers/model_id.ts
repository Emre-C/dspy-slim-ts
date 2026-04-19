/**
 * Model-identifier parsing primitives.
 *
 * A model string uses the convention `<provider>/<model>`, e.g.
 * `openai/gpt-4.1-mini` or `openrouter/minimax/minimax-m2.7`.  When no
 * provider prefix is present we default to `openai` so that bare OpenAI
 * model names (e.g. `gpt-4.1-mini`) continue to work.
 *
 * These helpers are intentionally dependency-free so they can be shared
 * between the transport (`lm.ts`) and the provider profiles without
 * introducing module cycles.
 */

const DEFAULT_PROVIDER = 'openai';

/**
 * Providers whose leading path segment is a transport prefix rather than
 * part of the upstream model identifier. For these, `providerModelName`
 * drops the prefix before forwarding the request, so
 * `openai/gpt-4.1-mini` → `gpt-4.1-mini` and
 * `openrouter/minimax/minimax-m2.7` → `minimax/minimax-m2.7` (the upstream
 * routing key that OpenRouter itself expects). Providers not listed here
 * are assumed to require the full model string as-is.
 */
const STRIPPABLE_PROVIDERS: ReadonlySet<string> = new Set(['openai', 'openrouter']);

export function providerNameFromModel(model: string): string {
  if (model.includes('/')) {
    return model.split('/', 1)[0]!.toLowerCase();
  }

  return DEFAULT_PROVIDER;
}

export function providerModelName(model: string): string {
  const provider = providerNameFromModel(model);
  if (STRIPPABLE_PROVIDERS.has(provider) && model.includes('/')) {
    return model.split('/').slice(1).join('/');
  }

  return model;
}
