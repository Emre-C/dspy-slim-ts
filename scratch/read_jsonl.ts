import fs from 'fs';
const content = fs.readFileSync('tools/longcot/runs/longcot_compare_predict_logic_easy_2026-04-20T02-43-48-887Z.jsonl', 'utf-8');
const lines = content.trim().split('\n');
const row = JSON.parse(lines[0]);
console.log("Q:", row.question.id);
console.log("Trace length:", row.trace_turns);
console.log("Full row keys:", Object.keys(row));
console.log("Response len:", row.response_text.length);
