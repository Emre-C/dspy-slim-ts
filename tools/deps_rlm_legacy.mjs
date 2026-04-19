#!/usr/bin/env node
// Guard against legacy RLM-v1 REPL surface slipping back into production code.
//
// Usage:
//   node tools/deps_rlm_legacy.mjs            # informational report, exits 0
//   node tools/deps_rlm_legacy.mjs --strict   # fails with exit 1 if any legacy symbol is found
//
// The --strict form is attached to the release gate starting in Phase 9 of
// the RLM v2 rollout (see docs/RLM_V2_IMPLEMENTATION_PLAN.md).

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { argv, exit, cwd } from 'node:process';
import { join, relative } from 'node:path';

const LEGACY_SYMBOLS = [
  'NodeCodeInterpreter',
  'createNodeCodeInterpreter',
  'SyncCodeInterpreter',
  'SyncCodeSession',
  'NodeCodeInterpreterOptions',
  'REPLHistory',
  'REPLEntry',
  'REPLEntryKind',
  'REPLVariable',
  'CodeSession',
  'CodeInterpreter',
  'ExecuteRequest',
  'ExecuteResult',
  'FinalOutput',
  'InterpreterPatch',
  'CodeInterpreterError',
  'BudgetVector',
  'LLMQueryRequest',
  'LLMQueryResult',
  'RLMConfig',
  'node:vm',
];

const LEGACY_RE = new RegExp(
  LEGACY_SYMBOLS.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'),
);

const SEARCH_ROOT = 'src';

function* walkTypeScript(dir) {
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      yield* walkTypeScript(full);
    } else if (entry.isFile() && /\.(ts|mts|cts|tsx)$/.test(entry.name)) {
      yield full;
    }
  }
}

const strict = argv.includes('--strict');
const repoRoot = cwd();
const matches = [];

for (const file of walkTypeScript(SEARCH_ROOT)) {
  let content;
  try {
    content = readFileSync(file, 'utf8');
  } catch (err) {
    console.error(`[deps:rlm-legacy] failed to read ${file}: ${err.message}`);
    exit(2);
  }
  const lines = content.split('\n');
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (LEGACY_RE.test(line)) {
      matches.push({ file: relative(repoRoot, file), line: i + 1, text: line.trimEnd() });
    }
  }
}

if (matches.length === 0) {
  console.log(`[deps:rlm-legacy] ${SEARCH_ROOT}/ is clean of legacy RLM-v1 symbols.`);
  exit(0);
}

console.log(
  `[deps:rlm-legacy] ${matches.length} legacy symbol reference(s) in ${SEARCH_ROOT}/:`,
);
for (const m of matches) {
  console.log(`  ${m.file}:${m.line}: ${m.text}`);
}

if (strict) {
  console.error(
    '[deps:rlm-legacy] FAIL: legacy symbols present under --strict mode.',
  );
  exit(1);
}

console.log(
  '[deps:rlm-legacy] Informational only. Run with --strict (Phase 9+) to fail CI on residue.',
);
exit(0);
