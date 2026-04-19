/**
 * RLM v2 quickstart: `search` and `aggregate` with OpenRouter if
 * `OPENROUTER_API_KEY` is set, else a scripted LM.
 *
 *   OPENROUTER_API_KEY=... npx tsx examples/rlm_quickstart.ts
 *   npx tsx examples/rlm_quickstart.ts
 */

import { env } from 'node:process';
import type { Message } from '../src/chat_message.js';
import { BaseLM, type LMOutput } from '../src/lm.js';
import { LM } from '../src/lm.js';
import { RLM } from '../src/rlm.js';
import { settings } from '../src/settings.js';

// ---------------------------------------------------------------------------
// Scripted fallback LM
// ---------------------------------------------------------------------------
//
// The oracle signature carries a `kind` literal whose value is
// `"value" | "effect"`; the fallback LM always returns `kind: "value"`
// (no tool calls) because the quickstart's two plans do not request
// effects. The classifier is routed around by passing `taskType` into
// the RLM options below.

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
      throw new Error(
        'QuickstartScriptedLM exhausted. Extend the canned queues or use OPENROUTER_API_KEY for a live LM.',
      );
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

// ---------------------------------------------------------------------------
// LM selection
// ---------------------------------------------------------------------------

function buildSearchLm(): BaseLM {
  const key = env.OPENROUTER_API_KEY;
  if (typeof key === 'string' && key.length > 0) {
    return new LM({
      model: 'openrouter/openai/gpt-4o-mini',
      apiKey: key,
      apiBase: 'https://openrouter.ai/api/v1',
    });
  }
  return new QuickstartScriptedLM(SEARCH_CANNED);
}

function buildAggregateLm(): BaseLM {
  const key = env.OPENROUTER_API_KEY;
  if (typeof key === 'string' && key.length > 0) {
    return new LM({
      model: 'openrouter/openai/gpt-4o-mini',
      apiKey: key,
      apiBase: 'https://openrouter.ai/api/v1',
    });
  }
  return new QuickstartScriptedLM(AGGREGATE_CANNED);
}

// ---------------------------------------------------------------------------
// Task 1: Search over chunked context
// ---------------------------------------------------------------------------
//
// The `search` static plan fans out per chunk (map + split), runs a
// fast-model oracle on each partition, and concatenates the findings.
// `taskType: 'search'` bypasses the classifier entirely, so the run
// emits only the 8 oracle calls — perfectly observable in `_rlm_trace`.

async function runSearch(): Promise<void> {
  const lm = buildSearchLm();
  settings.configure({ lm });

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
  console.log('answer:', String(prediction.getOr('answer', '')).slice(0, 240));
  const route = prediction.getOr('_rlm_route', null) as
    | { readonly kind: string; readonly taskTypes: readonly string[] }
    | null;
  console.log('route:', route);
  const trace = prediction.getOr('_rlm_trace', []) as readonly {
    readonly nodeTag: string;
  }[];
  console.log('trace tags:', Array.from(new Set(trace.map((t) => t.nodeTag))));
  console.log('oracle calls:', trace.filter((t) => t.nodeTag === 'oracle').length);
}

// ---------------------------------------------------------------------------
// Task 2: Aggregate with memory
// ---------------------------------------------------------------------------
//
// The `aggregate` static plan declares a `failure_diagnostic`
// `MemorySchema` — a tiny typed scratchpad the oracle may optionally
// populate via `WriteMemory` effects. This quickstart's scripted LM
// never emits a `WriteMemory` effect (the canned answers all carry
// `kind: "value"`), so the run produces identical output to the
// "no memory" path. Swap the scripted LM for a live model and it may
// begin writing diagnostics — the evaluator transparently reinjects
// them into the next oracle system message.

async function runAggregate(): Promise<void> {
  const lm = buildAggregateLm();
  settings.configure({ lm });

  const rlm = new RLM('data: str -> summary: str', {
    taskType: 'aggregate',
  });

  const prediction = await rlm.aforward({
    data:
      'item a priced 10 quantity 3 region east; '
      + 'item b priced 5 quantity 9 region west',
  });

  console.log('\n=== Task 2 — aggregate ===');
  console.log('summary:', String(prediction.getOr('summary', '')).slice(0, 240));
  const route = prediction.getOr('_rlm_route', null) as
    | { readonly kind: string; readonly taskTypes: readonly string[] }
    | null;
  console.log('route:', route);
  const trace = prediction.getOr('_rlm_trace', []) as readonly {
    readonly nodeTag: string;
  }[];
  console.log('trace tags:', Array.from(new Set(trace.map((t) => t.nodeTag))));
  console.log('oracle calls:', trace.filter((t) => t.nodeTag === 'oracle').length);
}

async function main(): Promise<void> {
  await runSearch();
  await runAggregate();
  console.log(
    '\nDone. Read src/rlm.ts and docs/product/rlm-v2-architecture.md next.',
  );
}

main().catch((err) => {
  console.error('rlm_quickstart failed:', err);
  throw err;
});
