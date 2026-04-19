/**
 * Chat message shapes shared by adapter and LM history — leaf module (Role-only dep).
 */

import type { Role } from './types.js';

export interface ContentPart {
  readonly type: 'text' | 'image_url' | 'file';
  readonly text?: string;
  readonly image_url?: { readonly url: string };
  readonly file?: {
    readonly file_data?: string;
    readonly filename?: string;
    readonly file_id?: string;
  };
}

export interface Message {
  readonly role: Role;
  readonly content: string | readonly ContentPart[];
}
