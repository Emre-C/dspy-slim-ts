/**
 * §8 — ReAct agent loop.
 */

import { ChatAdapter } from './adapter.js';
import { ChainOfThought } from './chain_of_thought.js';
import { ContextWindowExceededError, ValueError } from './exceptions.js';
import { createField, type Field } from './field.js';
import { isPlainObject } from './guards.js';
import { Module } from './module.js';
import { snapshotRecord } from './owned_value.js';
import { Predict } from './predict.js';
import { Prediction } from './prediction.js';
import { settings } from './settings.js';
import {
  createSignature,
  ensureSignature,
  type Signature,
} from './signature.js';
import type { InferInputs, InferOutputs, SignatureInput } from './signature_types.js';
import { Tool } from './tool.js';

const EMPTY_RECORD = Object.freeze({}) as Record<string, unknown>;
const MAX_TRUNCATION_ATTEMPTS = 3;
const TRAJECTORY_KEYS_PER_STEP = 4;
const RESERVED_REACT_INPUT_KEYS = new Set(['trajectory', 'max_iters']);

/**
 * Kwargs accepted by `ReAct.forward` / `.aforward`: the inferred inputs plus
 * the optional `max_iters` control override extracted in
 * `normalizeInvocation`.
 */
export type ReActKwargs<TInputs extends Record<string, unknown>> =
  TInputs & { readonly max_iters?: number };

type ToolInput = Tool | ((...args: unknown[]) => unknown);

interface ReActStep {
  readonly nextThought: string;
  readonly nextToolName: string;
  readonly nextToolArgs: Record<string, unknown>;
}

interface TruncatedCallResult {
  readonly prediction: Prediction;
  readonly trajectory: Record<string, unknown>;
}

function cloneField(field: Field, kind = field.kind): Field {
  return createField({
    kind,
    name: field.name,
    ...(field.isTypeUndefined ? {} : { typeTag: field.typeTag }),
    typeArgs: field.typeArgs,
    description: field.description,
    prefix: field.prefix,
    constraints: field.constraints,
    default: field.default,
    isTypeUndefined: field.isTypeUndefined,
  });
}

function ensureReactCompatibleSignature(signature: Signature): Signature {
  const reservedInputs = [...signature.inputFields.keys()].filter((name) => RESERVED_REACT_INPUT_KEYS.has(name));
  if (reservedInputs.length > 0) {
    throw new ValueError(`ReAct input field names are reserved for control flow: ${reservedInputs.join(', ')}.`);
  }

  return signature;
}

function normalizeTools(tools: readonly ToolInput[]): Map<string, Tool> {
  const normalized = new Map<string, Tool>();

  for (const candidate of tools) {
    const tool = candidate instanceof Tool ? candidate : new Tool(candidate);
    if (tool.name === 'finish') {
      throw new ValueError('ReAct tool name "finish" is reserved.');
    }

    if (normalized.has(tool.name)) {
      throw new ValueError(`Duplicate ReAct tool name: ${tool.name}`);
    }

    normalized.set(tool.name, tool);
  }

  normalized.set('finish', new Tool(() => 'Completed.', {
    name: 'finish',
    desc: 'Marks the task as complete. Signals that all information needed for the final outputs is now available.',
    args: {},
  }));

  return normalized;
}

function buildInstructions(signature: Signature, tools: ReadonlyMap<string, Tool>): string {
  const inputs = [...signature.inputFields.keys()].map((name) => `\`${name}\``).join(', ');
  const outputs = [...signature.outputFields.keys()].map((name) => `\`${name}\``).join(', ');

  const instructions: string[] = [];
  if (signature.instructions.trim() !== '') {
    instructions.push(`${signature.instructions}\n`);
  }

  instructions.push(
    `You are an Agent. In each episode, you will be given the fields ${inputs} as input. And you can see your past trajectory so far.`,
    `Your goal is to use one or more of the supplied tools to collect any necessary information for producing ${outputs}.\n`,
    'To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.',
    'After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n',
    'When writing next_thought, you may reason about the current situation and plan for future steps.',
    'When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n',
  );

  let index = 1;
  for (const tool of tools.values()) {
    instructions.push(`(${index}) ${tool.toString()}`);
    index += 1;
  }

  instructions.push('When providing `next_tool_args`, the value inside the field must be in JSON format.');
  return instructions.join('\n');
}

function createReactSignature(signature: Signature, tools: ReadonlyMap<string, Tool>): Signature {
  return createSignature(
    new Map([
      ...signature.inputFields,
      ['trajectory', createField({ kind: 'input', name: 'trajectory', typeTag: 'str', isTypeUndefined: false })],
    ]),
    new Map([
      ['next_thought', createField({ kind: 'output', name: 'next_thought', typeTag: 'str', isTypeUndefined: false })],
      ['next_tool_name', createField({ kind: 'output', name: 'next_tool_name', typeTag: 'str', isTypeUndefined: false })],
      ['next_tool_args', createField({ kind: 'output', name: 'next_tool_args', typeTag: 'dict', isTypeUndefined: false })],
    ]),
    {
      name: signature.name,
      instructions: buildInstructions(signature, tools),
    },
  );
}

function createExtractSignature(signature: Signature): Signature {
  return createSignature(
    new Map([
      ...signature.inputFields,
      ['trajectory', createField({ kind: 'input', name: 'trajectory', typeTag: 'str', isTypeUndefined: false })],
    ]),
    new Map([...signature.outputFields].map(([name, field]) => [name, cloneField(field, 'output')])),
    {
      name: signature.name,
      instructions: signature.instructions,
    },
  );
}

function normalizeStep(prediction: Prediction, tools: ReadonlyMap<string, Tool>): ReActStep {
  const nextThought = prediction.get('next_thought');
  const nextToolName = prediction.get('next_tool_name');
  const nextToolArgs = prediction.get('next_tool_args');

  if (typeof nextThought !== 'string') {
    throw new ValueError('ReAct predictor must return a string next_thought.');
  }

  if (typeof nextToolName !== 'string' || !tools.has(nextToolName)) {
    throw new ValueError(`ReAct predictor selected an unknown tool: ${String(nextToolName)}`);
  }

  if (!isPlainObject(nextToolArgs)) {
    throw new ValueError('ReAct predictor must return next_tool_args as a plain object.');
  }

  return {
    nextThought,
    nextToolName,
    nextToolArgs: snapshotRecord(nextToolArgs),
  };
}

function formatExecutionError(toolName: string, error: unknown): string {
  if (error instanceof Error) {
    return `Execution error in ${toolName}: ${error.name}: ${error.message}`;
  }

  return `Execution error in ${toolName}: ${String(error)}`;
}

export class ReAct<
  TSig extends SignatureInput = Signature,
  TInputs extends Record<string, unknown> = InferInputs<TSig>,
  TOutputs extends Record<string, unknown> = InferOutputs<TSig>,
> extends Module<ReActKwargs<TInputs>, TOutputs> {
  readonly signature: Signature;
  readonly maxIters: number;
  readonly tools: ReadonlyMap<string, Tool>;
  readonly react: Predict;
  readonly extract: ChainOfThought;

  constructor(signature: TSig, tools: readonly ToolInput[], maxIters = 20) {
    super();
    this.signature = ensureReactCompatibleSignature(ensureSignature(signature));
    this.maxIters = maxIters;
    this.tools = normalizeTools(tools);
    this.react = new Predict(createReactSignature(this.signature, this.tools));
    this.extract = new ChainOfThought(createExtractSignature(this.signature));
  }

  truncateTrajectory(trajectory: Record<string, unknown>): Record<string, unknown> {
    const entries = Object.entries(trajectory);
    if (entries.length <= TRAJECTORY_KEYS_PER_STEP) {
      throw new ValueError(
        'The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be truncated because it only has one tool call.',
      );
    }

    return Object.fromEntries(entries.slice(TRAJECTORY_KEYS_PER_STEP));
  }

  override forward(kwargs: ReActKwargs<TInputs> = EMPTY_RECORD as ReActKwargs<TInputs>): Prediction<TOutputs> {
    const { inputArgs, maxIters } = this.normalizeInvocation(kwargs);
    let trajectory: Record<string, unknown> = {};

    for (let index = 0; index < maxIters; index += 1) {
      let step: ReActStep;

      try {
        const result = this.callWithPotentialTrajectoryTruncation(this.react, trajectory, inputArgs);
        trajectory = result.trajectory;
        step = normalizeStep(result.prediction, this.tools);
      } catch (error) {
        if (error instanceof ValueError) {
          break;
        }

        throw error;
      }

      trajectory = {
        ...trajectory,
        [`thought_${index}`]: step.nextThought,
        [`tool_name_${index}`]: step.nextToolName,
        [`tool_args_${index}`]: step.nextToolArgs,
        [`observation_${index}`]: this.executeTool(step.nextToolName, step.nextToolArgs),
      };

      if (step.nextToolName === 'finish') {
        break;
      }
    }

    const extractResult = this.callWithPotentialTrajectoryTruncation(this.extract, trajectory, inputArgs);
    return Prediction.create<TOutputs>({
      trajectory: extractResult.trajectory,
      ...extractResult.prediction.toDict(),
    });
  }

  override async aforward(kwargs: ReActKwargs<TInputs> = EMPTY_RECORD as ReActKwargs<TInputs>): Promise<Prediction<TOutputs>> {
    const { inputArgs, maxIters } = this.normalizeInvocation(kwargs);
    let trajectory: Record<string, unknown> = {};

    for (let index = 0; index < maxIters; index += 1) {
      let step: ReActStep;

      try {
        const result = await this.acallWithPotentialTrajectoryTruncation(this.react, trajectory, inputArgs);
        trajectory = result.trajectory;
        step = normalizeStep(result.prediction, this.tools);
      } catch (error) {
        if (error instanceof ValueError) {
          break;
        }

        throw error;
      }

      trajectory = {
        ...trajectory,
        [`thought_${index}`]: step.nextThought,
        [`tool_name_${index}`]: step.nextToolName,
        [`tool_args_${index}`]: step.nextToolArgs,
        [`observation_${index}`]: await this.aexecuteTool(step.nextToolName, step.nextToolArgs),
      };

      if (step.nextToolName === 'finish') {
        break;
      }
    }

    const extractResult = await this.acallWithPotentialTrajectoryTruncation(this.extract, trajectory, inputArgs);
    return Prediction.create<TOutputs>({
      trajectory: extractResult.trajectory,
      ...extractResult.prediction.toDict(),
    });
  }

  private normalizeInvocation(kwargs: Record<string, unknown>): {
    readonly inputArgs: Record<string, unknown>;
    readonly maxIters: number;
  } {
    if (!isPlainObject(kwargs)) {
      throw new ValueError('ReAct expects a single plain-object argument.');
    }

    const raw = snapshotRecord(kwargs);
    const rawMaxIters = raw.max_iters;
    const maxIters = rawMaxIters === undefined ? this.maxIters : Number(rawMaxIters);
    if (!Number.isInteger(maxIters) || maxIters < 0) {
      throw new ValueError('ReAct max_iters override must be a non-negative integer.');
    }

    const { max_iters: _discard, ...inputArgs } = raw;
    return {
      inputArgs,
      maxIters,
    };
  }

  private formatTrajectory(trajectory: Record<string, unknown>): string {
    const adapter = settings.adapter instanceof ChatAdapter ? settings.adapter : new ChatAdapter();
    const trajectorySignature = createSignature(
      new Map(Object.keys(trajectory).map((name) => [name, createField({ kind: 'input', name })])),
      new Map([['x', createField({ kind: 'output', name: 'x' })]]),
    );

    return adapter.formatUserMessageContent(trajectorySignature, trajectory);
  }

  private callWithPotentialTrajectoryTruncation(
    module: Pick<Module, 'call'>,
    trajectory: Record<string, unknown>,
    inputArgs: Record<string, unknown>,
  ): TruncatedCallResult {
    let currentTrajectory = trajectory;

    for (let attempt = 0; attempt < MAX_TRUNCATION_ATTEMPTS; attempt += 1) {
      try {
        return {
          prediction: module.call({
            ...inputArgs,
            trajectory: this.formatTrajectory(currentTrajectory),
          }),
          trajectory: currentTrajectory,
        };
      } catch (error) {
        if (!(error instanceof ContextWindowExceededError)) {
          throw error;
        }

        currentTrajectory = this.truncateTrajectory(currentTrajectory);
      }
    }

    throw new ValueError('The context window was exceeded even after 3 attempts to truncate the trajectory.');
  }

  private async acallWithPotentialTrajectoryTruncation(
    module: Pick<Module, 'acall'>,
    trajectory: Record<string, unknown>,
    inputArgs: Record<string, unknown>,
  ): Promise<TruncatedCallResult> {
    let currentTrajectory = trajectory;

    for (let attempt = 0; attempt < MAX_TRUNCATION_ATTEMPTS; attempt += 1) {
      try {
        return {
          prediction: await module.acall({
            ...inputArgs,
            trajectory: this.formatTrajectory(currentTrajectory),
          }),
          trajectory: currentTrajectory,
        };
      } catch (error) {
        if (!(error instanceof ContextWindowExceededError)) {
          throw error;
        }

        currentTrajectory = this.truncateTrajectory(currentTrajectory);
      }
    }

    throw new ValueError('The context window was exceeded even after 3 attempts to truncate the trajectory.');
  }

  private executeTool(toolName: string, toolArgs: Record<string, unknown>): unknown {
    try {
      return this.tools.get(toolName)!.call(toolArgs);
    } catch (error) {
      return formatExecutionError(toolName, error);
    }
  }

  private async aexecuteTool(toolName: string, toolArgs: Record<string, unknown>): Promise<unknown> {
    try {
      return await this.tools.get(toolName)!.acall(toolArgs);
    } catch (error) {
      return formatExecutionError(toolName, error);
    }
  }
}
