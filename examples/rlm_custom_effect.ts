/**
 * Custom `Search` and `Custom` handlers plus a user `StaticPlan` merged via
 * `RLMOptions.handlers` / `RLMOptions.plans`. OpenRouter when
 * `OPENROUTER_API_KEY` is set, else scripted LM.
 */

import { env } from 'node:process';
import type { Message } from '../src/chat_message.js';
import { oracle, vref } from '../src/rlm_combinators.js';
import { BaseLM, LM, type LMOutput } from '../src/lm.js';
import { RLM } from '../src/rlm.js';
import type { StaticPlan } from '../src/rlm_task_router.js';
import type { JsonObject } from '../src/json_value.js';
import type { EffectHandler } from '../src/rlm_types.js';
import { settings } from '../src/settings.js';

// ---------------------------------------------------------------------------
// Custom static plan — single-turn oracle over the user prompt
// ---------------------------------------------------------------------------
//
// Single `oracle(vref('input'))` template; `RLM.aforward` seeds the
// `'input'` scope binding with the flattened user prompt before
// handing the plan to the evaluator. The effect loop runs inside that
// oracle leaf, bounded by `budget.maxEffectTurns`. Any tool call the
// LM issues is resolved by one of the user-registered handlers below.

const ENRICHED_PAIRWISE_PLAN: StaticPlan = Object.freeze({
  taskType: 'pairwise' as const,
  template: oracle(vref('input'), undefined),
  memorySchema: null,
});

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------
//
// `searchHandler` resolves queries against a toy corpus; a production
// consumer would swap in an HTTP call or a vector-DB lookup here.
// `wordCounterHandler` is a `Custom`-kind handler — the open-world
// escape hatch. Both return structured `EffectResult` values so the
// evaluator can reinject the outcome into the next oracle turn.

const TOY_CORPUS: Readonly<Record<string, string>> = Object.freeze({
  zinc: 'Zinc is a chemical element with atomic number 30.',
  copper: 'Copper is a chemical element with atomic number 29.',
  nickel: 'Nickel is a chemical element with atomic number 28.',
});

const searchHandler: EffectHandler = {
  name: 'Search',
  async handle(effect) {
    if (effect.kind !== 'Search') {
      return { ok: false, error: 'Search handler invoked on wrong effect kind' };
    }
    const key = effect.query.trim().toLowerCase();
    const hit = TOY_CORPUS[key];
    if (hit === undefined) {
      return { ok: false, error: `No corpus entry for "${effect.query}"` };
    }
    return { ok: true, value: hit };
  },
};

const wordCounterHandler: EffectHandler = {
  name: 'WordCounter',
  async handle(effect) {
    if (effect.kind !== 'Custom') {
      return {
        ok: false,
        error: 'WordCounter handler invoked on wrong effect kind',
      };
    }
    const text = effect.args.text;
    if (typeof text !== 'string') {
      return { ok: false, error: 'WordCounter requires args.text: string' };
    }
    const count = text.trim() === '' ? 0 : text.trim().split(/\s+/).length;
    return { ok: true, value: String(count) };
  },
};

// ---------------------------------------------------------------------------
// Scripted fallback LM
// ---------------------------------------------------------------------------
//
// Turn 1 emits a `Search` effect; turn 2 emits a `Custom` (WordCounter)
// effect; turn 3 returns a value. The scripted LM cycles through one
// queued response per oracle call. A live OpenRouter model is
// substituted when `OPENROUTER_API_KEY` is present.

const SCRIPTED_TURNS: readonly object[] = [
  {
    kind: 'effect',
    value: null,
    effect_name: 'Search',
    effect_args: { query: 'zinc' },
  },
  {
    kind: 'effect',
    value: null,
    effect_name: 'Custom',
    effect_args: {
      name: 'WordCounter',
      args: { text: 'Zinc is a chemical element with atomic number 30.' },
    },
  },
  {
    kind: 'value',
    value:
      'zinc: atomic number 30, corpus-described in 9 words. Pairwise winner: zinc.',
    effect_name: null,
    effect_args: null,
  },
];

class CustomEffectScriptedLM extends BaseLM {
  private readonly queue: object[];

  constructor(answers: readonly object[]) {
    super({ model: 'custom-effect-scripted' });
    this.queue = [...answers];
  }

  protected override generate(
    _prompt?: string,
    _messages?: readonly Message[],
    _kwargs?: Record<string, unknown>,
  ): readonly LMOutput[] {
    const next = this.queue.shift();
    if (next === undefined) {
      throw new Error(
        'CustomEffectScriptedLM exhausted. Extend SCRIPTED_TURNS or use OPENROUTER_API_KEY.',
      );
    }
    return [JSON.stringify(next)];
  }
}

// ---------------------------------------------------------------------------
// LM selection
// ---------------------------------------------------------------------------

function buildLm(): BaseLM {
  const key = env.OPENROUTER_API_KEY;
  if (typeof key === 'string' && key.length > 0) {
    return new LM({
      model: 'openrouter/openai/gpt-4o-mini',
      apiKey: key,
      apiBase: 'https://openrouter.ai/api/v1',
    });
  }
  return new CustomEffectScriptedLM(SCRIPTED_TURNS);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const lm = buildLm();
  settings.configure({ lm });

  const rlm = new RLM(
    'option_a: str, option_b: str -> analysis: str',
    {
      taskType: 'pairwise',
      handlers: [searchHandler, wordCounterHandler],
      plans: [ENRICHED_PAIRWISE_PLAN],
      budget: { maxEffectTurns: 4 },
    },
  );

  const prediction = await rlm.aforward({
    option_a: 'zinc',
    option_b: 'copper',
  });

  console.log('analysis:');
  console.log(String(prediction.getOr('analysis', '')));
  console.log();

  const trace = prediction.getOr('_rlm_trace', []) as readonly {
    readonly nodeTag: string;
    readonly extras?: JsonObject;
  }[];
  const effectEntries = trace.filter((t) => t.nodeTag === 'effect');
  const effectKinds = effectEntries.map((t) => t.extras?.effectKind ?? 'unknown');
  console.log('effects observed:', effectKinds);
  console.log(
    'oracle calls:',
    trace.filter((t) => t.nodeTag === 'oracle').length,
  );
}

main().catch((err) => {
  console.error('rlm_custom_effect failed:', err);
  throw err;
});
