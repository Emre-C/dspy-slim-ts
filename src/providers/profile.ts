/**
 * Provider profile contract and registry.
 *
 * Core LM/adapter modules resolve provider-specific behavior through this
 * registry rather than hard-coding vendor names.  See the decision record at
 * docs/product/provider-profile-boundary.md.
 *
 * Resolution is first-match-wins over the registration order.  Register more
 * specific profiles before more general ones.
 */

import { ValueError } from '../exceptions.js';
import type { LMLike } from '../types.js';

export interface ProviderProfile {
  readonly id: string;
  readonly matches: (model: string) => boolean;
  readonly mapRequest?: (req: Record<string, unknown>) => Record<string, unknown>;
  /** First argument is the LM instance; `LMLike` is the structural minimum (leaf `types.ts`, no `lm.ts`). */
  readonly adapterRetry?: (
    lm: LMLike,
    lmKwargs: Record<string, unknown>,
    error: unknown,
  ) => Record<string, unknown> | null;
}

const profiles: ProviderProfile[] = [];

/**
 * Register a profile.  Throws if a profile with the same `id` is already
 * registered — double-registration is almost always a bug (re-imported
 * module, copy-pasted setup code, etc.) and failing loudly is better than
 * silently stacking dead profiles behind the first match.
 */
export function registerProfile(profile: ProviderProfile): void {
  if (profiles.some((candidate) => candidate.id === profile.id)) {
    throw new ValueError(`Provider profile '${profile.id}' is already registered.`);
  }

  profiles.push(profile);
}

export function resolveProfile(model: string): ProviderProfile | null {
  for (const profile of profiles) {
    if (profile.matches(model)) {
      return profile;
    }
  }

  return null;
}

/**
 * Remove all registered profiles.  Intended for test isolation; not part of
 * the library's public surface (not re-exported from `src/index.ts`).
 */
export function clearProfiles(): void {
  profiles.splice(0, profiles.length);
}
