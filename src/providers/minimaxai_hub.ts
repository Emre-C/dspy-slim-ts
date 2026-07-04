/**
 * MiniMax models referenced by Hugging Face / Together-style ids (`MiniMaxAI/...`).
 *
 * The leading segment is `minimaxai` (see `providerNameFromModel`). Same empirical
 * output-token floor as OpenRouter-hosted Minimax — low caps often yield empty
 * `message.content` without a transport error. OpenRouter-specific `reasoning`
 * request knobs are not applied here because other gateways use different shapes
 * (see Together docs); response-side handling lives in `lm.ts` (`reasoning` /
 * `reasoning_content` fallbacks).
 */

import { providerNameFromModel } from './model_id.js';
import type { ProviderProfile } from './profile.js';

const PROFILE_ID = 'minimax-ai-hub';

/** Kept aligned with `openrouter_minimax.ts` — shared release-gate history. */
const MINIMUM_OUTPUT_TOKENS = 4096;

function isMiniMaxAiHubModel(model: string): boolean {
  return providerNameFromModel(model) === 'minimaxai';
}

function applyOutputFloor(request: Record<string, unknown>): Record<string, unknown> {
  const normalized = { ...request };

  if (typeof normalized.max_tokens === 'number' && Number.isFinite(normalized.max_tokens)) {
    normalized.max_tokens = Math.max(normalized.max_tokens, MINIMUM_OUTPUT_TOKENS);
    return normalized;
  }

  if (typeof normalized.max_output_tokens === 'number' && Number.isFinite(normalized.max_output_tokens)) {
    normalized.max_output_tokens = Math.max(normalized.max_output_tokens, MINIMUM_OUTPUT_TOKENS);
    return normalized;
  }

  if (
    typeof normalized.max_completion_tokens === 'number'
    && Number.isFinite(normalized.max_completion_tokens)
  ) {
    normalized.max_completion_tokens = Math.max(normalized.max_completion_tokens, MINIMUM_OUTPUT_TOKENS);
    return normalized;
  }

  normalized.max_tokens = MINIMUM_OUTPUT_TOKENS;
  return normalized;
}

export const minimaxAiHubProfile: ProviderProfile = {
  id: PROFILE_ID,

  matches: isMiniMaxAiHubModel,

  mapRequest: (req) => applyOutputFloor(req),
};
