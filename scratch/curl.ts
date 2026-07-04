import { LM } from './src/lm.js';
import { settings } from './src/settings.js';

async function main() {
  const lm = new LM({
    model: 'fireworks/accounts/fireworks/models/qwen2p5-72b-instruct', // Or kimi-k2p5-turbo? Wait! The env LONGCOT_LM_BACKEND was exported. Let me just use the actual configured model.
  });
}
