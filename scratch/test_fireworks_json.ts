import { LM } from '../src/lm.js';

async function main() {
  const lm = new LM('fireworks/accounts/fireworks/models/kimi-k2p5-turbo');
  const response = await lm.acall(
    undefined,
    [{ role: 'user', content: 'Give me a JSON object with a field "name" equal to "Emre" and "age" equal to 30. Do not write anything else.' }],
    { response_format: { type: 'json_object' } }
  );
  console.log('Response:', response[0].text);
}
main().catch(console.error);
