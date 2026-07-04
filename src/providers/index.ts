/**
 * Provider profiles — entry point.
 *
 * Importing this module registers all built-in profiles.  Core modules should
 * import `resolveProfile` from here rather than from individual profile files
 * so that all registrations happen exactly once.
 */

import { minimaxAiHubProfile } from './minimaxai_hub.js';
import { openRouterMinimaxProfile } from './openrouter_minimax.js';
import { registerProfile } from './profile.js';

export { registerProfile, resolveProfile } from './profile.js';

registerProfile(minimaxAiHubProfile);
registerProfile(openRouterMinimaxProfile);
