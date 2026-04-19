/**
 * OpenRouter / Minimax provider profile.
 *
 * Minimax models hosted on OpenRouter have two well-known quirks:
 *   1. They silently truncate short completions unless `max_tokens` (or
 *      `max_output_tokens`) is at least `MINIMUM_OUTPUT_TOKENS`.
 *   2. They occasionally emit empty structured outputs when reasoning is
 *      unconstrained; forcing `reasoning.effort = 'minimal'` on retry
 *      recovers the generation.
 *
 * This profile encapsulates both so that core transport and adapter code
 * remains provider-agnostic.
 */

import { isPlainObject } from '../guards.js';
import { snapshotRecord } from '../owned_value.js';
import { providerModelName, providerNameFromModel } from './model_id.js';
import type { ProviderProfile } from './profile.js';

const PROFILE_ID = 'openrouter-minimax';

/**
 * Empirical floor for `max_tokens` on OpenRouter-Minimax. Below this threshold
 * the upstream service frequently returns truncated or empty completions
 * without surfacing an error, which the adapter then fails to parse. 4096 is
 * the smallest value that consistently produced complete structured outputs
 * during the release-gate evaluations tracked in `benchmarks/`. Revisit this
 * number if those benchmarks show the upstream behavior has changed.
 */
const MINIMUM_OUTPUT_TOKENS = 4096;

function isOpenRouterMinimaxModel(model: string): boolean {
  return providerNameFromModel(model) === 'openrouter'
    && providerModelName(model).toLowerCase().startsWith('minimax/');
}

function applyReasoningDefaults(
  request: Record<string, unknown>,
): Record<string, unknown> {
  if (request.reasoning !== undefined) {
    return request;
  }

  return {
    ...request,
    reasoning: {
      exclude: true,
    },
  };
}

function applyOutputFloor(
  request: Record<string, unknown>,
): Record<string, unknown> {
  const normalized = { ...request };

  if (typeof normalized.max_tokens === 'number' && Number.isFinite(normalized.max_tokens)) {
    normalized.max_tokens = Math.max(normalized.max_tokens, MINIMUM_OUTPUT_TOKENS);
    return normalized;
  }

  if (typeof normalized.max_output_tokens === 'number' && Number.isFinite(normalized.max_output_tokens)) {
    normalized.max_output_tokens = Math.max(normalized.max_output_tokens, MINIMUM_OUTPUT_TOKENS);
    return normalized;
  }

  normalized.max_tokens = MINIMUM_OUTPUT_TOKENS;
  return normalized;
}

// `AdapterParseError` is duck-typed on `error.name` rather than imported to
// avoid a module cycle through `adapter.ts -> providers/* -> adapter.ts`.
// Renaming the error class requires updating this check.
function isAdapterParseError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AdapterParseError';
}

function hasExplicitReasoningOverride(
  lmKwargs: Record<string, unknown>,
): boolean {
  if (lmKwargs.reasoning !== undefined) {
    return true;
  }

  const extraBody = isPlainObject(lmKwargs.extra_body)
    ? lmKwargs.extra_body
    : isPlainObject(lmKwargs.extraBody)
      ? lmKwargs.extraBody
      : null;

  return isPlainObject(extraBody) && extraBody.reasoning !== undefined;
}

function withMinimalReasoning(
  lmKwargs: Record<string, unknown>,
): Record<string, unknown> {
  const nextKwargs = snapshotRecord(lmKwargs);
  const extraBody = isPlainObject(nextKwargs.extra_body)
    ? snapshotRecord(nextKwargs.extra_body)
    : isPlainObject(nextKwargs.extraBody)
      ? snapshotRecord(nextKwargs.extraBody)
      : {};

  delete nextKwargs.extraBody;
  nextKwargs.extra_body = snapshotRecord({
    ...extraBody,
    reasoning: {
      exclude: true,
      effort: 'minimal',
    },
  });

  return nextKwargs;
}

export const openRouterMinimaxProfile: ProviderProfile = {
  id: PROFILE_ID,

  matches: isOpenRouterMinimaxModel,

  mapRequest: (req) => applyOutputFloor(applyReasoningDefaults(req)),

  adapterRetry: (_lm, lmKwargs, error) => {
    if (!isAdapterParseError(error)) {
      return null;
    }

    if (hasExplicitReasoningOverride(lmKwargs)) {
      return null;
    }

    return withMinimalReasoning(lmKwargs);
  },
};
