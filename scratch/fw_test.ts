import { LM } from '../src/lm.js';
async function run() {
  const lm = new LM({
    model: 'fireworks/accounts/fireworks/routers/kimi-k2p5-turbo',
    apiKey: process.env.FIREWORKS_API_KEY,
    kwargs: { stream: true, max_tokens: 128 },
  });
  console.log("Sending query...");
  const t0 = Date.now();
  const outputs = await lm.acall("Solve 2+2 step by step", []);
  console.log(`Finished in ${Date.now() - t0}ms, output:`, outputs[0]);
}
run().catch(console.error);
