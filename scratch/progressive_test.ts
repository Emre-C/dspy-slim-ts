import { LM } from '../src/lm.js';
import { RLM } from '../src/rlm.js';

async function progressiveTest() {
  console.log("==========================================");
  console.log("PROGRESSIVE TEST: 1. Short LM Call Validation");
  console.log("==========================================\n");

  const lm = new LM({
    model: 'accounts/fireworks/routers/kimi-k2p5-turbo',
    apiKey: process.env.FIREWORKS_API_KEY,
    apiBase: process.env.FIREWORKS_API_BASE ?? 'https://api.fireworks.ai/inference/v1',
    kwargs: { max_tokens: 8192, stream: true }
  });

  const prompt = `Initial state: [[1, 2], [3]]
Goal state: [[1, 2, 3], []]
Number of blocks: 3
Number of stacks: 2

Find a sequence of moves that will transform the initial state into the goal state.
Format your solution as:
solution = [move0, move1, ..., movek].`;

  const rlm = new RLM('prompt: str -> answer: str', {
     subLm: lm,
     maxOracleCalls: 10,
     maxEffectTurns: 5
  });

  const t0 = Date.now();
  console.log("[LM] Sending prompt to RLM... (This should take < 1 minute, no infinite hanging!)");
  try {
    const result = await rlm.aforward({ prompt });
    console.log(`\n[LM] Finished short validation in ${Date.now() - t0}ms`);
    
    console.log("\n--- TRACE DUMP ---");
    const trace = result.trace;
    if (trace && trace.length > 0) {
       for (const entry of trace) {
           console.log(`[Trace] ${entry.name} -> duration: ${entry.duration_ms}ms`);
           if (entry.extras && entry.extras.effectKind) {
               console.log(`   -> Effect: ${entry.extras.effectKind}, Handler OK: ${entry.extras.handlerOk}`);
           }
       }
    } else {
       console.log("No detailed trace captured or single value emitted.");
    }
    
    console.log("\n--- FINAL ANSWER ---");
    console.log(result.get('answer'));

  } catch (e: any) {
    console.error("\n[ERROR] RLM Validation Failed:", e.message);
  }
  
  console.log("\n==========================================");
  console.log("PROGRESSIVE TEST: 2. LongCoT Mini Validation");
  console.log("==========================================");
  console.log("To validate thoroughly, run the comprehensive LongCoT mini-suite:");
  console.log("    ./tools/test_rlm_mini.sh");
}

progressiveTest().catch(console.error);
