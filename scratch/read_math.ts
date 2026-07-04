import fs from 'fs';
const f = fs.readFileSync('tools/longcot/runs/longcot_rlm_math_easy_2026-04-20T11-22-27-445Z.jsonl', 'utf-8');
const row = JSON.parse(f.trim().split('\n')[0]);
console.log("Error:", row.error);
console.log("Latency:", row.latency_ms);
console.log("Trace length:", row._rlm_trace?.length || row.diagnostics?._rlm_trace?.length || "no trace");
