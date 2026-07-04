import { describe, expect, it, vi } from 'vitest';
import {
  ConfigurationError,
  Image,
  isImage,
} from '../src/index.js';

const TINY_PNG_BYTES = new Uint8Array([
  // 1×1 transparent PNG, ~67 bytes (truncated for unit-test purposes).
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
  0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
  0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4,
  0x89,
]);

describe('Image', () => {
  it('round-trips fromBuffer → toDataUri', () => {
    const image = Image.fromBuffer(TINY_PNG_BYTES, 'image/png');
    const dataUri = image.toDataUri();
    expect(dataUri.startsWith('data:image/png;base64,')).toBe(true);

    const reparsed = Image.fromDataUri(dataUri);
    expect(reparsed.mimeType).toBe('image/png');
    expect(reparsed.bytes).toBeInstanceOf(Uint8Array);
    expect(Array.from(reparsed.bytes!)).toEqual(Array.from(TINY_PNG_BYTES));
  });

  it('parses fromDataUri with mime + bytes', () => {
    const original = Image.fromBuffer(TINY_PNG_BYTES, 'image/jpeg');
    const dataUri = original.toDataUri();
    const reparsed = Image.fromDataUri(dataUri);
    expect(reparsed.mimeType).toBe('image/jpeg');
    expect(reparsed.bytes).toBeInstanceOf(Uint8Array);
    expect(reparsed.bytes!.length).toBe(TINY_PNG_BYTES.length);
  });

  it('rejects unsupported mime types from fromBuffer', () => {
    expect(() =>
      // @ts-expect-error — testing invalid mime
      Image.fromBuffer(TINY_PNG_BYTES, 'image/bmp'),
    ).toThrow(ConfigurationError);
  });

  it('rejects empty buffers', () => {
    expect(() => Image.fromBuffer(new Uint8Array(), 'image/png')).toThrow(ConfigurationError);
  });

  it('rejects malformed data URIs', () => {
    expect(() => Image.fromDataUri('not-a-data-uri')).toThrow(ConfigurationError);
    expect(() => Image.fromDataUri('data:image/bmp;base64,xxx')).toThrow(ConfigurationError);
    expect(() => Image.fromDataUri('data:image/png;base64,')).toThrow(ConfigurationError);
    expect(() => Image.fromDataUri('data:image/png;base64,!!!!')).toThrow(ConfigurationError);
  });

  it('rejects fromBase64 when payload is a data URI', () => {
    expect(() =>
      Image.fromBase64('data:image/png;base64,abc', 'image/png'),
    ).toThrow(ConfigurationError);
  });

  it('rejects fromBase64 when payload is empty', () => {
    expect(() => Image.fromBase64('   ', 'image/png')).toThrow(ConfigurationError);
  });

  it('rejects malformed base64 payloads', () => {
    expect(() => Image.fromBase64('!!!!', 'image/png')).toThrow(ConfigurationError);
    expect(() => Image.fromBase64('abcd=ef', 'image/png')).toThrow(ConfigurationError);
  });

  it('fromUrl preserves https url unchanged in toDataUri', () => {
    const image = Image.fromUrl('https://example.com/foo.png', 'image/png');
    expect(image.toDataUri()).toBe('https://example.com/foo.png');
    expect(image.url).toBe('https://example.com/foo.png');
    expect(image.bytes).toBeUndefined();
  });

  it('rejects non-HTTPS URLs', () => {
    expect(() => Image.fromUrl('http://example.com/foo.png', 'image/png')).toThrow(ConfigurationError);
    expect(() => Image.fromUrl('not-a-url', 'image/png')).toThrow(ConfigurationError);
  });

  it('fromUrl with a data URI delegates to fromDataUri', () => {
    const original = Image.fromBuffer(TINY_PNG_BYTES, 'image/webp');
    const fromUrl = Image.fromUrl(original.toDataUri());
    expect(fromUrl.mimeType).toBe('image/webp');
    expect(fromUrl.bytes!.length).toBe(TINY_PNG_BYTES.length);
  });

  it('isImage discriminates correctly', () => {
    const image = Image.fromBuffer(TINY_PNG_BYTES, 'image/png');
    expect(isImage(image)).toBe(true);
    expect(isImage('foo')).toBe(false);
    expect(isImage(null)).toBe(false);
    expect(isImage({ _isDspyImage: true })).toBe(false);
    expect(isImage({ mimeType: 'image/png', bytes: TINY_PNG_BYTES })).toBe(false);
  });

  it('soft-warns when payload is larger than 20 MB', () => {
    const big = new Uint8Array(21 * 1024 * 1024);
    big[0] = 0x89; // ensure non-empty payload
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      Image.fromBuffer(big, 'image/png');
      expect(spy).toHaveBeenCalledTimes(1);
      expect(String(spy.mock.calls[0]?.[0])).toContain('21.0 MB');
    } finally {
      spy.mockRestore();
    }
  });

  it('Image instances are frozen', () => {
    const image = Image.fromBuffer(TINY_PNG_BYTES, 'image/png');
    expect(Object.isFrozen(image)).toBe(true);
  });

  it('defensively copies bytes on input and output', () => {
    const mutable = new Uint8Array([1, 2, 3]);
    const image = Image.fromBuffer(mutable, 'image/png');
    mutable[0] = 9;

    expect(Array.from(image.bytes!)).toEqual([1, 2, 3]);

    const exposed = image.bytes!;
    exposed[1] = 9;
    expect(Array.from(image.bytes!)).toEqual([1, 2, 3]);
  });
});
