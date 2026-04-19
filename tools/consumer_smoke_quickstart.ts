/**
 * Consumer smoke version of `examples/rlm_quickstart.ts`.
 *
 * Copied to a temp directory by `tools/consumer_smoke.sh` and compiled
 * + run against the published tarball. Imports from `dspy-slim-ts`,
 * not from `../src/...`, because the whole point is to validate the
 * published public surface.
 */

import { env } from 'node:process';
import type { Message } from 'dspy-slim-ts';
import {
  BaseLM,
  LM,
  RLM,
  settings,
  type LMOutput,
} from 'dspy-slim-ts';

const SEARCH_CANNED: readonly string[] = [
  'finding-chunk-1: rose vase observed',
  'finding-chunk-2: owl figurine noted',
  'finding-chunk-3: blue room mentioned',
  'finding-chunk-4: shelf referenced',
  'finding-chunk-5: no further clues',
  'finding-chunk-6: mundane objects only',
  'finding-chunk-7: final chunk seen',
  'finding-chunk-8: no secret item found',
];

const AGGREGATE_CANNED: readonly string[] = [
  'partial: item a priced 10, qty 3',
  'partial: item b priced 5, qty 9',
  'partial: region east noted',
  'partial: region west noted',
  'SYNTHESIS: item a: 10x3=30 (east); item b: 5x9=45 (west); total 75',
];

class QuickstartScriptedLM extends BaseLM {
  private readonly queue: string[];

  constructor(answers: readonly string[]) {
    super({ model: 'quickstart-scripted' });
    this.queue = [...answers];
  }

  protected override generate(
    _prompt?: string,
    messages?: readonly Message[],
    _kwargs?: Record<string, unknown>,
  ): readonly LMOutput[] {
    const next = this.queue.shift();
    if (next === undefined) {
      throw new Error('QuickstartScriptedLM exhausted.');
    }
    const fullContent = (messages ?? [])
      .map((m) =>
        typeof m.content === 'string'
          ? m.content
          : m.content
              .map((p) => (p.type === 'text' ? (p.text ?? '') : ''))
              .join('\n'),
      )
      .join('\n');
    if (fullContent.includes("- `kind` (")) {
      return [
        JSON.stringify({
          kind: 'value',
          value: next,
          effect_name: null,
          effect_args: null,
        }),
      ];
    }
    return [JSON.stringify({ answer: next })];
  }
}

function buildLm(canned: readonly string[]): BaseLM {
  const key = env.OPENROUTER_API_KEY;
  if (typeof key === 'string' && key.length > 0) {
    return new LM({
      model: 'openrouter/openai/gpt-4o-mini',
      apiKey: key,
      apiBase: 'https://openrouter.ai/api/v1',
    });
  }
  return new QuickstartScriptedLM(canned);
}

async function runSearch(): Promise<void> {
  settings.configure({ lm: buildLm(SEARCH_CANNED) });
  const rlm = new RLM('question: str, context: str -> answer: str', {
    taskType: 'search',
  });
  const prediction = await rlm.aforward({
    question: 'Where is the secret item?',
    context:
      'The shelf in the blue room holds a vase of roses and a small owl figurine. '
      + 'Nothing else is visible; the other rooms were not searched.',
  });
  console.log('=== Task 1 — search ===');
  console.log('answer:', String(prediction.getOr('answer', '')).slice(0, 160));
  const trace = prediction.getOr('_rlm_trace', []) as readonly {
    readonly nodeTag: string;
  }[];
  console.log('oracle calls:', trace.filter((t) => t.nodeTag === 'oracle').length);
}

async function runAggregate(): Promise<void> {
  settings.configure({ lm: buildLm(AGGREGATE_CANNED) });
  const rlm = new RLM('data: str -> summary: str', {
    taskType: 'aggregate',
  });
  const prediction = await rlm.aforward({
    data:
      'item a priced 10 quantity 3 region east; '
      + 'item b priced 5 quantity 9 region west',
  });
  console.log('\n=== Task 2 — aggregate ===');
  console.log('summary:', String(prediction.getOr('summary', '')).slice(0, 160));
}

async function main(): Promise<void> {
  await runSearch();
  await runAggregate();
  console.log('\nConsumer smoke (quickstart): OK');
}

main().catch((err) => {
  console.error('smoke failed:', err);
  process.exitCode = 1;
});
