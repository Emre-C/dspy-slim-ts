import { afterEach, describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import {
  BaseLM,
  ContextWindowExceededError,
  ReAct,
  settings,
  Tool,
  ValueError,
  type LMOutput,
  type Message,
} from '../src/index.js';

interface ToolSequenceStep {
  tool_name?: string;
  tool_args?: Record<string, unknown>;
  observation?: string;
  tool_raises?: string;
  observation_prefix?: string;
  error?: 'ValueError';
}

interface ReactFixtureCase {
  id: string;
  max_iters?: number;
  tool_sequence?: ToolSequenceStep[];
  trajectory_before?: Record<string, string>;
  expected_iterations?: number;
  expected_terminal_state?: 'done';
  expected_trajectory_keys?: string[];
  expected_trajectory_after_truncation_keys?: string[];
  truncation_expected_error?: string;
  tools?: string[];
  expected_tool_names?: string[];
  original_signature?: string;
  expected_react_inputs?: string[];
  expected_react_outputs?: string[];
}

interface QueueEntry {
  readonly type: 'output' | 'error';
  readonly value: string | Error;
}

class QueueLM extends BaseLM {
  readonly queue: QueueEntry[];
  calls = 0;

  constructor(queue: readonly QueueEntry[]) {
    super({ model: 'queue-lm' });
    this.queue = [...queue];
  }

  protected override generate(
    _prompt?: string,
    _messages?: readonly Message[],
    _kwargs?: Record<string, unknown>,
  ): readonly LMOutput[] {
    const entry = this.queue[this.calls];
    this.calls += 1;

    if (!entry) {
      throw new Error('QueueLM ran out of scripted outputs');
    }

    if (entry.type === 'error') {
      throw entry.value;
    }

    return [entry.value];
  }
}

const fixture = JSON.parse(
  readFileSync(
    new URL('../../dspy-slim/spec/fixtures/react_state_machine.json', import.meta.url),
    'utf-8',
  ),
) as { cases: ReactFixtureCase[] };

afterEach(() => {
  settings.reset();
});

function buildQueue(toolSequence: readonly ToolSequenceStep[]): readonly QueueEntry[] {
  const reactSteps = toolSequence.map((step, index) => {
    if (step.error === 'ValueError') {
      return {
        type: 'error' as const,
        value: new ValueError('invalid tool selection'),
      };
    }

    return {
      type: 'output' as const,
      value: JSON.stringify({
        next_thought: `thought ${index}`,
        next_tool_name: step.tool_name,
        next_tool_args: step.tool_args ?? {},
      }),
    };
  });

  return [
    ...reactSteps,
    {
      type: 'output' as const,
      value: JSON.stringify({
        reasoning: 'Summarized the collected observations.',
        answer: 'final answer',
      }),
    },
  ];
}

function createTools(toolSequence: readonly ToolSequenceStep[]) {
  const sequenceByName = new Map<string, ToolSequenceStep[]>();
  for (const step of toolSequence) {
    if (!step.tool_name || step.tool_name === 'finish') {
      continue;
    }

    const existing = sequenceByName.get(step.tool_name) ?? [];
    existing.push(step);
    sequenceByName.set(step.tool_name, existing);
  }

  return [...sequenceByName.entries()].map(([toolName, steps]) => {
    let callIndex = 0;
    const argNames = new Set<string>();
    for (const step of steps) {
      for (const name of Object.keys(step.tool_args ?? {})) {
        argNames.add(name);
      }
    }

    return new Tool(({ ...args }: Record<string, unknown>) => {
      const step = steps[callIndex];
      callIndex += 1;

      expect(args).toEqual(step?.tool_args ?? {});

      if (step?.tool_raises) {
        const error = new Error(`${step.tool_raises}: boom`);
        error.name = step.tool_raises;
        throw error;
      }

      return step?.observation ?? `${toolName} observation`;
    }, {
      name: toolName,
      args: Object.fromEntries([...argNames].map((name) => [name, {}])),
    });
  });
}

describe('ReAct (spec fixtures)', () => {
  for (const c of fixture.cases) {
    it(c.id, async () => {
      switch (c.id) {
        case 'immediate_finish':
        case 'tool_then_finish':
        case 'max_iters_exhausted':
        case 'value_error_breaks_loop':
        case 'tool_error_captured_as_observation':
        case 'trajectory_keys_structure': {
          const toolSequence = c.tool_sequence ?? [];
          const react = new ReAct('question -> answer', createTools(toolSequence), c.max_iters ?? 20);
          const lm = new QueueLM(buildQueue(toolSequence));
          settings.configure({ lm });

          const prediction = react.forward({ question: 'Where is the answer?' });
          const result = prediction.toDict() as {
            reasoning: string;
            answer: string;
            trajectory: Record<string, unknown>;
          };

          if (c.expected_iterations !== undefined) {
            expect(lm.calls).toBe(c.expected_iterations + 1);
          }

          if (c.expected_terminal_state === 'done') {
            expect(result.answer).toBe('final answer');
            expect(result.reasoning).toBe('Summarized the collected observations.');
          }

          if (c.expected_trajectory_keys) {
            expect(Object.keys(result.trajectory)).toEqual(c.expected_trajectory_keys);
          }

          if (c.id === 'tool_error_captured_as_observation') {
            const observation = result.trajectory.observation_0;
            expect(typeof observation).toBe('string');
            expect(observation as string).toContain((toolSequence[0]?.observation_prefix) ?? 'Execution error');
          }

          return;
        }

        case 'truncation_drops_oldest_4_keys': {
          const react = new ReAct('question -> answer', []);
          const truncated = react.truncateTrajectory(c.trajectory_before ?? {});
          expect(Object.keys(truncated)).toEqual(c.expected_trajectory_after_truncation_keys);
          return;
        }

        case 'truncation_of_minimal_trajectory_errors': {
          const react = new ReAct('question -> answer', []);
          expect(() => react.truncateTrajectory(c.trajectory_before ?? {})).toThrow(
            c.truncation_expected_error,
          );
          return;
        }

        case 'finish_tool_always_present': {
          const react = new ReAct(
            'question -> answer',
            (c.tools ?? []).map((toolName) => new Tool(() => `${toolName} result`, { name: toolName })),
          );
          expect([...react.tools.keys()]).toEqual(c.expected_tool_names);
          return;
        }

        case 'react_signature_structure': {
          const react = new ReAct(c.original_signature ?? 'question -> answer', [function search() {
            return 'result';
          }]);
          expect([...react.react.signature.inputFields.keys()]).toEqual(c.expected_react_inputs);
          expect([...react.react.signature.outputFields.keys()]).toEqual(c.expected_react_outputs);
          return;
        }

        case 'extract_signature_structure': {
          const react = new ReAct(c.original_signature ?? 'question -> answer', []);
          expect([...react.extract.predict.signature.inputFields.keys()]).toEqual(['question', 'trajectory']);
          expect([...react.extract.predict.signature.outputFields.keys()]).toEqual(['reasoning', 'answer']);
          return;
        }

        default:
          throw new Error(`Unhandled ReAct fixture case: ${c.id}`);
      }
    });
  }
});

describe('ReAct hardening', () => {
  it('preserves module traversal to both inner predictors', () => {
    const react = new ReAct('question -> answer', [function search() {
      return 'result';
    }]);

    expect(react.namedPredictors().map(([name]) => name)).toEqual(['react', 'extract.predict']);
  });

  it('retries after context window overflow and carries the truncated trajectory forward', () => {
    const react = new ReAct('question -> answer', [function search() {
      return 'result';
    }]);

    const queue = new QueueLM([
      {
        type: 'output',
        value: JSON.stringify({
          next_thought: 'thought 0',
          next_tool_name: 'search',
          next_tool_args: {},
        }),
      },
      {
        type: 'output',
        value: JSON.stringify({
          next_thought: 'thought 1',
          next_tool_name: 'search',
          next_tool_args: {},
        }),
      },
      {
        type: 'error',
        value: new ContextWindowExceededError(),
      },
      {
        type: 'output',
        value: JSON.stringify({
          next_thought: 'thought 2',
          next_tool_name: 'finish',
          next_tool_args: {},
        }),
      },
      {
        type: 'output',
        value: JSON.stringify({
          reasoning: 'wrapped up',
          answer: 'done',
        }),
      },
    ]);

    settings.configure({ lm: queue });

    const result = react.forward({ question: 'Why?' }).toDict() as { trajectory: Record<string, unknown> };

    expect(result.trajectory.thought_0).toBeUndefined();
    expect(result.trajectory.thought_1).toBe('thought 1');
    expect(result.trajectory.tool_name_2).toBe('finish');
  });
});
