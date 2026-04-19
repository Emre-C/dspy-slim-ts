/**
 * LM output wire / envelope types — leaf module for history and transport.
 */

export interface ToolCallWire {
  readonly id?: string | undefined;
  readonly type?: 'function' | undefined;
  readonly function: {
    readonly name: string;
    readonly arguments: string;
  };
}

export interface LMOutputEnvelope {
  readonly text: string;
  readonly logprobs?: unknown;
  readonly citations?: readonly unknown[] | undefined;
  readonly toolCalls?: readonly ToolCallWire[] | undefined;
}

export type LMOutput = string | LMOutputEnvelope;
