import { describe, expect, it } from 'vitest';
import {
  ChatAdapter,
  Image,
  signatureFromString,
  type ContentPart,
  type Message,
} from '../src/index.js';

const PNG_BYTES = new Uint8Array([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
  0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
]);

const JPEG_BYTES = new Uint8Array([
  0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46,
  0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
]);

function userMessage(messages: readonly Message[]): Message {
  const last = messages[messages.length - 1];
  if (last === undefined) {
    throw new Error('expected at least one message');
  }
  return last;
}

describe('ChatAdapter — image content parts', () => {
  it('emits ContentPart[] when an Image input is present, in spec order', () => {
    const sig = signatureFromString(
      'page_image: Image, n: int -> answer: str',
    );
    const adapter = new ChatAdapter();
    const image = Image.fromBuffer(PNG_BYTES, 'image/png');

    const messages = adapter.format(sig, [], { page_image: image, n: 5 });
    const userMsg = userMessage(messages);

    expect(Array.isArray(userMsg.content)).toBe(true);
    const parts = userMsg.content as readonly ContentPart[];

    // We expect: text(non-image marker for n=5), text('[[ ## page_image ## ]]'),
    // image_url(data:image/png;base64,…), text(output requirements).
    expect(parts).toHaveLength(4);

    expect(parts[0]?.type).toBe('text');
    expect(parts[0]?.text).toContain('[[ ## n ## ]]');
    expect(parts[0]?.text).toContain('5');

    expect(parts[1]?.type).toBe('text');
    expect(parts[1]?.text?.trim()).toBe('[[ ## page_image ## ]]');

    expect(parts[2]?.type).toBe('image_url');
    expect(parts[2]?.image_url?.url.startsWith('data:image/png;base64,')).toBe(true);

    expect(parts[3]?.type).toBe('text');
    expect(parts[3]?.text).toContain('Respond with the corresponding output fields');
    expect(parts[3]?.text).toContain('[[ ## answer ## ]]');
    expect(parts[3]?.text).toContain('[[ ## completed ## ]]');
  });

  it('returns content as a string when no Image input is present', () => {
    const sig = signatureFromString('a: str -> b: int');
    const adapter = new ChatAdapter();

    const messages = adapter.format(sig, [], { a: 'hello' });
    const userMsg = userMessage(messages);

    expect(typeof userMsg.content).toBe('string');
    expect(userMsg.content).toContain('[[ ## a ## ]]');
    expect(userMsg.content).toContain('hello');
  });

  it('interleaves multiple Image fields with their markers in declaration order', () => {
    const sig = signatureFromString(
      'left: Image, right: Image, n: int -> answer: str',
    );
    const adapter = new ChatAdapter();
    const left = Image.fromBuffer(PNG_BYTES, 'image/png');
    const right = Image.fromBuffer(JPEG_BYTES, 'image/jpeg');

    const messages = adapter.format(sig, [], {
      left,
      right,
      n: 7,
    });
    const userMsg = userMessage(messages);
    const parts = userMsg.content as readonly ContentPart[];

    // Pattern: [text(<n=7>), text("[[ ## left ## ]]"), image_url(left),
    //          text("[[ ## right ## ]]"), image_url(right),
    //          text(output requirements)]
    expect(parts).toHaveLength(6);
    expect(parts[0]?.type).toBe('text');
    expect(parts[0]?.text).toContain('[[ ## n ## ]]');
    expect(parts[0]?.text).toContain('7');
    expect(parts[1]?.type).toBe('text');
    expect(parts[1]?.text?.trim()).toBe('[[ ## left ## ]]');
    expect(parts[2]?.type).toBe('image_url');
    expect(parts[2]?.image_url?.url.startsWith('data:image/png;base64,')).toBe(true);
    expect(parts[3]?.type).toBe('text');
    expect(parts[3]?.text?.trim()).toBe('[[ ## right ## ]]');
    expect(parts[4]?.type).toBe('image_url');
    expect(parts[4]?.image_url?.url.startsWith('data:image/jpeg;base64,')).toBe(true);
    expect(parts[5]?.type).toBe('text');
    expect(parts[5]?.text).toContain('Respond with the corresponding output fields');
  });

  it('elides image content in demos and substitutes a placeholder string', () => {
    const sig = signatureFromString(
      'page_image: Image, n: int -> answer: str',
    );
    const adapter = new ChatAdapter();
    const image = Image.fromBuffer(PNG_BYTES, 'image/png');

    const messages = adapter.format(
      sig,
      [{ page_image: image, n: 1, answer: 'demo answer' }],
      { page_image: image, n: 5 },
    );

    // First user message is the demo; it must remain a string (no
    // ContentPart array) and contain a placeholder, never the data URI.
    const demoUser = messages.find((m) => m.role === 'user');
    expect(demoUser).toBeDefined();
    expect(typeof demoUser!.content).toBe('string');
    expect(demoUser!.content as string).toContain('<image elided>');
    expect(demoUser!.content as string).not.toContain('data:image/png;base64,');
  });

  it('produces frozen content parts to prevent mutation', () => {
    const sig = signatureFromString('page_image: Image -> answer: str');
    const adapter = new ChatAdapter();
    const image = Image.fromBuffer(PNG_BYTES, 'image/png');
    const messages = adapter.format(sig, [], { page_image: image });
    const userMsg = userMessage(messages);
    const parts = userMsg.content as readonly ContentPart[];
    expect(Object.isFrozen(parts)).toBe(true);
    for (const part of parts) {
      expect(Object.isFrozen(part)).toBe(true);
    }
  });
});
