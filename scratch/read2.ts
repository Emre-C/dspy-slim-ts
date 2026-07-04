import fs from 'fs';
const pred = fs.readFileSync('tools/longcot/runs/longcot_predict_logic_easy_2026-04-20T11-06-01-792Z.jsonl', 'utf-8');
const row = JSON.parse(pred.trim().split('\n')[0]);
console.log("Q:", row.question.question_id);
console.log("Model from payload?", row.payload ? row.payload.model : "no payload property");
