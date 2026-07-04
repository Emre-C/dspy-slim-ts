import fs from 'fs';
const content = fs.readFileSync('tools/longcot/runs/longcot_compare_predict_logic_easy_2026-04-20T02-43-48-887Z.jsonl', 'utf-8');
const lines = content.trim().split('\n');
const row = JSON.parse(lines[0]);
const text = row.response_text;

function repairJson(candidate: string): string {
  const withQuotedStrings = candidate.replace(
    /'([^'\\]*(?:\\.[^'\\]*)*)'/g,
    (_match, content: string) => JSON.stringify(content.replace(/\\'/g, "'")),
  );

  return withQuotedStrings.replace(
    /([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)/g,
    '$1"$2"$3',
  );
}

console.log("Text length: ", text.length);
const start = Date.now();
repairJson(text);
console.log(`Regex finished in ${Date.now() - start}ms`);
