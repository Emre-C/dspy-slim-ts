import fs from 'fs';
const content = fs.readFileSync('tools/longcot/runs/longcot_rlm_logic_easy_2026-04-20T03-26-13-663Z.jsonl', 'utf-8');
const lines = content.trim().split('\n');
for (const line of lines) {
  if (!line) continue;
  const row = JSON.parse(line);
  console.log("Q:", row.question.id);
  const trace = row._rlm_trace || row.diagnostics?._rlm_trace;
  if (!trace) {
    if (row.payload && row.payload._rlm_trace) {
      console.log("Trace length:", row.payload._rlm_trace.length);
      console.log(row.payload._rlm_trace.map((t:any) => t.nodeTag + " " + t.extras?.effectKind).join(", "));
    } else {
      console.log("No trace found. Keys:", Object.keys(row), row.payload ? Object.keys(row.payload) : "no payload");
    }
  } else {
    console.log("Trace turns:", trace.length);
  }
}
