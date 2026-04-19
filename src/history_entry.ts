/**
 * LM history row shape — leaf module (no settings / module / lm imports).
 */

import type { Message } from './chat_message.js';
import type { LMOutput } from './lm_output.js';
import type { ModelType } from './types.js';

export interface HistoryEntry {
  readonly prompt: string | null;
  readonly messages: readonly Message[] | null;
  readonly kwargs: Record<string, unknown>;
  readonly response: unknown;
  readonly outputs: readonly LMOutput[];
  readonly usage: Record<string, number>;
  readonly cost: number | null;
  readonly timestamp: string;
  readonly uuid: string;
  readonly model: string;
  readonly responseModel: string;
  readonly modelType: ModelType;
}
