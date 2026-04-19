/**
 * Call-site module surface used by settings and LM history fan-out.
 * Decouples `settings.ts` from `module.ts` to break import cycles.
 */

import type { HistoryEntry } from './history_entry.js';

export interface CallerModuleLike {
  history: HistoryEntry[];
}
