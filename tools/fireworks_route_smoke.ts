/**
 * Smoke: (1) Fireworks OpenAI API directly, (2) same model family via Hugging Face router.
 * Loads repo-root `.env` without overriding existing `process.env`.
 *
 * Usage: npx tsx tools/fireworks_route_smoke.ts
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');

const FIREWORKS_OPENAI_BASE = 'https://api.fireworks.ai/inference/v1';
const DEFAULT_FIREWORKS_MODEL = 'accounts/fireworks/routers/kimi-k2p5-turbo';
const HF_ROUTER_BASE = 'https://router.huggingface.co/v1';
const HF_FIREWORKS_MODEL = 'MiniMaxAI/MiniMax-M2.7:fireworks-ai';

const PROMPT = 'What is the capital of France?';

function loadRootEnvFile(): void {
  const path = resolve(REPO_ROOT, '.env');
  let raw: string;
  try {
    raw = readFileSync(path, 'utf-8');
  } catch {
    return;
  }
  for (const line of raw.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.length === 0 || trimmed.startsWith('#')) {
      continue;
    }
    const eq = trimmed.indexOf('=');
    if (eq <= 0) {
      continue;
    }
    const key = trimmed.slice(0, eq).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
      continue;
    }
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (process.env[key] === undefined) {
      process.env[key] = value;
    }
  }
}

async function postChatCompletions(
  baseUrl: string,
  apiKey: string,
  model: string,
): Promise<unknown> {
  const url = `${baseUrl.replace(/\/$/, '')}/chat/completions`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'User-Agent': 'dspy-slim-ts/fireworks_route_smoke',
    },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: PROMPT }],
      max_tokens: 256,
    }),
  });
  const text = await res.text();
  let parsed: unknown = text;
  try {
    parsed = text.trim() === '' ? {} : JSON.parse(text);
  } catch {
    /* raw text */
  }
  if (!res.ok) {
    throw new Error(
      `HTTP ${String(res.status)} ${res.statusText}\n${typeof parsed === 'string' ? parsed : JSON.stringify(parsed)}`,
    );
  }
  return parsed;
}

function printMessage(label: string, body: unknown): void {
  console.log(`\n=== ${label} ===`);
  if (isRecord(body) && Array.isArray(body.choices) && body.choices.length > 0) {
    const first = body.choices[0];
    if (isRecord(first) && isRecord(first.message)) {
      console.log(JSON.stringify(first.message, null, 2));
      return;
    }
  }
  console.log(JSON.stringify(body, null, 2));
}

function isRecord(x: unknown): x is Record<string, unknown> {
  return x !== null && typeof x === 'object' && !Array.isArray(x);
}

async function main(): Promise<void> {
  loadRootEnvFile();

  const fwKey = process.env.FIREWORKS_API_KEY;
  const hfToken = process.env.HF_TOKEN;
  const fwModel = process.env.FIREWORKS_SMOKE_MODEL ?? DEFAULT_FIREWORKS_MODEL;
  const fwBase = process.env.FIREWORKS_API_BASE ?? FIREWORKS_OPENAI_BASE;

  if (!fwKey) {
    console.error('Missing FIREWORKS_API_KEY (set in env or repo-root .env).');
    process.exit(1);
  }

  console.log('Test 1: Fireworks direct (OpenAI-compatible POST /chat/completions)');
  console.log(`  baseURL: ${fwBase}`);
  console.log(`  model: ${fwModel}`);
  const direct = await postChatCompletions(fwBase, fwKey, fwModel);
  printMessage('Fireworks direct — choices[0].message', direct);

  if (!hfToken) {
    console.error('\nSkipping test 2: missing HF_TOKEN for Hugging Face router.');
    process.exit(0);
  }

  console.log('\nTest 2: Hugging Face router → Fireworks backend (MiniMax id with :fireworks-ai)');
  console.log(`  baseURL: ${HF_ROUTER_BASE}`);
  console.log(`  model: ${HF_FIREWORKS_MODEL}`);
  const viaHf = await postChatCompletions(HF_ROUTER_BASE, hfToken, HF_FIREWORKS_MODEL);
  printMessage('HF router — choices[0].message', viaHf);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
