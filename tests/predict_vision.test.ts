import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  ChatAdapter,
  ConfigurationError,
  Image,
  Predict,
  ReplayLM,
  settings,
} from '../src/index.js';

const PNG_BYTES = new Uint8Array([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
  0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
]);

/** ReplayLM with vision capability stubbed in for tests. */
class VisionReplayLM extends ReplayLM {
  override get supportsVision(): boolean {
    return true;
  }
}

describe('Predict + Image inputs', () => {
  // Use ChatAdapter so the replayed `[[ ## answer ## ]]\n…` output
  // shape parses cleanly. Restore the previous adapter after each test.
  let previousAdapter: ReturnType<typeof settings.snapshot>['adapter'];
  beforeEach(() => {
    previousAdapter = settings.adapter;
    settings.configure({ adapter: new ChatAdapter() });
  });
  afterEach(() => {
    settings.configure({ adapter: previousAdapter });
  });

  it('throws ConfigurationError when the LM does not advertise vision', async () => {
    const lm = new ReplayLM([
      '[[ ## answer ## ]]\nirrelevant\n[[ ## completed ## ]]',
    ]);
    const signature: string = 'page_image: Image -> answer: str';
    const predict = new Predict(signature);
    const image = Image.fromBuffer(PNG_BYTES, 'image/png');

    await expect(
      predict.aforward({ page_image: image, lm }),
    ).rejects.toThrow(ConfigurationError);
  });

  it('rejects non-Image values for Image input fields', async () => {
    const lm = new VisionReplayLM([
      '[[ ## answer ## ]]\nirrelevant\n[[ ## completed ## ]]',
    ]);
    const signature: string = 'page_image: Image -> answer: str';
    const predict = new Predict(signature);

    await expect(
      predict.aforward({ page_image: 'data:image/png;base64,aGVsbG8=', lm }),
    ).rejects.toThrow('Image input field');
  });

  it('rejects Image values passed through non-Image fields', async () => {
    const lm = new VisionReplayLM([
      '[[ ## answer ## ]]\nirrelevant\n[[ ## completed ## ]]',
    ]);
    const signature: string = 'page_image: str -> answer: str';
    const predict = new Predict(signature);
    const image = Image.fromBuffer(PNG_BYTES, 'image/png');

    await expect(
      predict.aforward({ page_image: image, lm }),
    ).rejects.toThrow('Image values require Image-typed signature fields');
  });

  it('rejects Image output fields because phase 1 supports inputs only', () => {
    expect(() => new Predict('question: str -> preview: Image')).toThrow(
      'Image output fields',
    );
  });

  it('returns a parsed prediction when LM is vision-capable (replayed)', async () => {
    const lm = new VisionReplayLM([
      '[[ ## answer ## ]]\nfound a wall\n[[ ## completed ## ]]',
    ]);
    const predict = new Predict('page_image: Image, n: int -> answer: str');
    const image = Image.fromBuffer(PNG_BYTES, 'image/png');

    const prediction = await predict.aforward({
      page_image: image,
      n: 5,
      lm,
    });

    expect(prediction.getOr('answer', '')).toBe('found a wall');
  });

  it('does not invoke the supportsVision check when no Image fields are declared', async () => {
    // Plain non-vision ReplayLM (supportsVision === false) must still
    // serve text-only signatures.
    const lm = new ReplayLM([
      '[[ ## answer ## ]]\nplain text\n[[ ## completed ## ]]',
    ]);
    const predict = new Predict('q: str -> answer: str');

    const prediction = await predict.aforward({ q: 'hi', lm });
    expect(prediction.getOr('answer', '')).toBe('plain text');
  });
});
