/**
 * Consumer smoke version of `examples/rlm_custom_effect.ts`.
 *
 * Copied to a temp directory by `tools/consumer_smoke.sh` and compiled
 * + run against the published tarball. Imports only from
 * `dspy-slim-ts`.
 */

import { env } from 'node:process';
import type { Message, StaticPlan, EffectHandler, LMOutput } from 'dspy-slim-ts';
import {
  BaseLM,
  LM,
  RLM,
  oracle,
  settings,
  vref,
} from 'dspy-slim-ts';

const ENRICHED_PAIRWISE_PLAN: StaticPlan = Object.freeze({
  taskType: 'pairwise' as const,
  template: oracle(vref('input'), undefined),
  memorySchema: null,
});

const TOY_CORPUS: Readonly<Record<string, string>> = Object.freeze({
  zinc: 'Zinc is a chemical element with atomic number 30.',
  copper: 'Copper is a chemical element with atomic number 29.',
});

const searchHandler: EffectHandler = {
  name: 'Search',
  async handle(effect) {
    if (effect.kind !== 'Search') {
      return { ok: false, error: 'wrong effect kind for Search' };
    }
    const hit = TOY_CORPUS[effect.query.trim().toLowerCase()];
    if (hit === undefined) {
      return { ok: false, error: `no corpus entry for "${effect.query}"` };
    }
    return { ok: true, value: hit };
  },
};

const wordCounterHandler: EffectHandler = {
  name: 'WordCounter',
  async handle(effect) {
    if (effect.kind !== 'Custom') {
      return { ok: false, error: 'wrong effect kind for Custom' };
    }
    const text = effect.args.text;
    if (typeof text !== 'string') {
      return { ok: false, error: 'WordCounter requires args.text: string' };
    }
    return {
      ok: true,
      value: String(text.trim() === '' ? 0 : text.trim().split(/\s+/).length),
    };
  },
};

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
    value: 'zinc wins. 9 words described it.',
    effect_name: null,
    effect_args: null,
  },
];

class ScriptedLM extends BaseLM {
  private readonly queue: object[];

  constructor(turns: readonly object[]) {
    super({ model: 'smoke-scripted' });
    this.queue = [...turns];
  }

  protected override generate(
    _prompt?: string,
    _messages?: readonly Message[],
  ): readonly LMOutput[] {
    const next = this.queue.shift();
    if (next === undefined) {
      throw new Error('ScriptedLM queue exhausted');
    }
    return [JSON.stringify(next)];
  }
}

function buildLm(): BaseLM {
  const key = env.OPENROUTER_API_KEY;
  if (typeof key === 'string' && key.length > 0) {
    return new LM({
      model: 'openrouter/openai/gpt-4o-mini',
      apiKey: key,
      apiBase: 'https://openrouter.ai/api/v1',
    });
  }
  return new ScriptedLM(SCRIPTED_TURNS);
}

async function main(): Promise<void> {
  settings.configure({ lm: buildLm() });
  const rlm = new RLM('option_a: str, option_b: str -> analysis: str', {
    taskType: 'pairwise',
    handlers: [searchHandler, wordCounterHandler],
    plans: [ENRICHED_PAIRWISE_PLAN],
    budget: { maxEffectTurns: 4 },
  });
  const prediction = await rlm.aforward({
    option_a: 'zinc',
    option_b: 'copper',
  });
  console.log('analysis:', String(prediction.getOr('analysis', '')));
  const trace = prediction.getOr('_rlm_trace', []) as readonly {
    readonly nodeTag: string;
    readonly extras?: Readonly<Record<string, unknown>>;
  }[];
  console.log(
    'effects observed:',
    trace
      .filter((t) => t.nodeTag === 'effect')
      .map((t) => t.extras?.effectKind ?? 'unknown'),
  );
  console.log('Consumer smoke (custom effect): OK');
}

main().catch((err) => {
  console.error('smoke failed:', err);
  process.exitCode = 1;
});
