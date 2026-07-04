import fs from 'fs';
const pred = fs.readFileSync('tools/longcot/runs/longcot_compare_predict_logic_easy_2026-04-20T11-01-41-034Z.jsonl', 'utf-8');
const rlm = fs.readFileSync('tools/longcot/runs/longcot_compare_rlm_logic_easy_2026-04-20T11-01-41-034Z.jsonl', 'utf-8');
console.log("PREDICT:");
console.dir(JSON.parse(pred.split('\n')[0]), {depth: null});
console.log("RLM:");
console.dir(JSON.parse(rlm.split('\n')[0]), {depth: null});
