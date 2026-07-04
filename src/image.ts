/**
 * §2 — Multimodal `Image` value type.
 *
 * Represents an image that can be inlined into a chat message as an
 * OpenAI-compatible `image_url` content part. The adapter detects
 * `Image` instances via `isImage(value)` and emits the wire-shape
 * `{ type: 'image_url', image_url: { url } }` rather than stringifying
 * the value into a text block.
 *
 * Construction surfaces:
 *   - `Image.fromBuffer(bytes, mimeType)`  — raw bytes + explicit mime
 *   - `Image.fromBase64(base64, mimeType)` — bare base64 (no data: prefix)
 *   - `Image.fromDataUri('data:<mime>;base64,…')`
 *   - `Image.fromUrl('https://…')`         — public URL (mime optional)
 *
 * `toDataUri()` always produces `data:<mime>;base64,…` for buffer-/base64-
 * backed instances, and returns the original URL for url-backed instances.
 *
 * Validation rules:
 *   - mimeType must be one of {png, jpeg, webp, gif} (rejected otherwise).
 *   - Empty payloads are rejected.
 *   - A buffer/base64 payload over the soft warning threshold emits a
 *     single `console.warn`; OpenRouter accepts large payloads but
 *     downstream serializers may not.
 */

import { ConfigurationError } from './exceptions.js';

export type SupportedImageMime =
  | 'image/png'
  | 'image/jpeg'
  | 'image/webp'
  | 'image/gif';

const SUPPORTED_MIMES: ReadonlySet<SupportedImageMime> = new Set([
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/gif',
]);

const SOFT_WARN_BYTES = 20 * 1024 * 1024; // 20 MB

const DATA_URI_RE =
  /^data:(image\/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/=\s]+)$/;
const BASE64_RE = /^[A-Za-z0-9+/]*={0,2}$/;

function assertSupportedMime(value: string): asserts value is SupportedImageMime {
  if (!SUPPORTED_MIMES.has(value as SupportedImageMime)) {
    throw new ConfigurationError(
      `Unsupported image mime type: "${value}". Supported: ${[...SUPPORTED_MIMES].join(', ')}.`,
    );
  }
}

function asUint8Array(input: Uint8Array | ArrayBuffer | ArrayLike<number>): Uint8Array {
  if (input instanceof Uint8Array) {
    return input;
  }
  if (input instanceof ArrayBuffer) {
    return new Uint8Array(input);
  }
  return Uint8Array.from(input as ArrayLike<number>);
}

function bytesToBase64(bytes: Uint8Array): string {
  // `Buffer` is available on Node.js (the only target today). We fall
  // back to a manual encoder for unusual runtimes.
  const maybeBuffer = (globalThis as { Buffer?: { from(b: Uint8Array): { toString(enc: string): string } } }).Buffer;
  if (maybeBuffer !== undefined) {
    return maybeBuffer.from(bytes).toString('base64');
  }

  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]!);
  }
  // eslint-disable-next-line @typescript-eslint/no-deprecated
  return (globalThis as unknown as { btoa(input: string): string }).btoa(binary);
}

function base64ToBytes(base64: string): Uint8Array {
  const cleaned = base64.replace(/\s+/g, '');
  if (
    cleaned === ''
    || cleaned.length % 4 === 1
    || !BASE64_RE.test(cleaned)
    || (cleaned.includes('=') && !/^[A-Za-z0-9+/]+={1,2}$/.test(cleaned))
  ) {
    throw new ConfigurationError('Image base64 payload is malformed.');
  }
  const maybeBuffer = (globalThis as { Buffer?: { from(s: string, enc: string): Uint8Array } }).Buffer;
  if (maybeBuffer !== undefined) {
    return new Uint8Array(maybeBuffer.from(cleaned, 'base64'));
  }
  // eslint-disable-next-line @typescript-eslint/no-deprecated
  const binary = (globalThis as unknown as { atob(input: string): string }).atob(cleaned);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    out[i] = binary.charCodeAt(i);
  }
  return out;
}

function softWarnIfTooLarge(byteLength: number): void {
  if (byteLength > SOFT_WARN_BYTES) {
    console.warn(
      `Image payload is ${(byteLength / (1024 * 1024)).toFixed(1)} MB; downstream serializers may reject payloads larger than ~32 MB.`,
    );
  }
}

function assertNonEmptyBytes(bytes: Uint8Array, source: string): void {
  if (bytes.length === 0) {
    throw new ConfigurationError(`${source} payload is empty.`);
  }
}

interface ImageInternal {
  readonly mimeType: SupportedImageMime;
  readonly url?: string;
  readonly bytes?: Uint8Array;
}

export class Image {
  readonly mimeType: SupportedImageMime;
  readonly url?: string;
  /** Tag used by `isImage` and adapters; never collide with user objects. */
  readonly _isDspyImage = true as const;

  readonly #bytes?: Uint8Array;

  private constructor(args: ImageInternal) {
    this.mimeType = args.mimeType;
    if (args.url !== undefined) {
      this.url = args.url;
    }
    if (args.bytes !== undefined) {
      this.#bytes = new Uint8Array(args.bytes);
    }
    Object.freeze(this);
  }

  get bytes(): Uint8Array | undefined {
    return this.#bytes === undefined ? undefined : new Uint8Array(this.#bytes);
  }

  static fromBuffer(
    bytes: Uint8Array | ArrayBuffer | ArrayLike<number>,
    mimeType: SupportedImageMime,
  ): Image {
    assertSupportedMime(mimeType);
    const view = asUint8Array(bytes);
    if (view.length === 0) {
      throw new ConfigurationError('Image.fromBuffer requires a non-empty payload.');
    }
    softWarnIfTooLarge(view.length);
    return new Image({ mimeType, bytes: view });
  }

  static fromBase64(base64: string, mimeType: SupportedImageMime): Image {
    assertSupportedMime(mimeType);
    if (typeof base64 !== 'string' || base64.trim() === '') {
      throw new ConfigurationError('Image.fromBase64 requires a non-empty base64 string.');
    }
    if (base64.startsWith('data:')) {
      throw new ConfigurationError(
        'Image.fromBase64 expects bare base64 (no "data:" prefix). Use Image.fromDataUri instead.',
      );
    }
    const bytes = base64ToBytes(base64);
    assertNonEmptyBytes(bytes, 'Image.fromBase64');
    softWarnIfTooLarge(bytes.length);
    return new Image({ mimeType, bytes });
  }

  static fromDataUri(dataUri: string): Image {
    if (typeof dataUri !== 'string' || !dataUri.startsWith('data:')) {
      throw new ConfigurationError(
        `Image.fromDataUri requires a data URI; got ${typeof dataUri === 'string' ? `"${dataUri.slice(0, 32)}…"` : typeof dataUri}.`,
      );
    }
    const match = DATA_URI_RE.exec(dataUri);
    if (match === null) {
      throw new ConfigurationError(
        'Image.fromDataUri expects shape "data:image/<png|jpeg|webp|gif>;base64,<payload>".',
      );
    }
    const mimeType = match[1] as SupportedImageMime;
    const bytes = base64ToBytes(match[2]!);
    assertNonEmptyBytes(bytes, 'Image.fromDataUri');
    softWarnIfTooLarge(bytes.length);
    return new Image({ mimeType, bytes });
  }

  static fromUrl(url: string, mimeType: SupportedImageMime = 'image/png'): Image {
    const trimmed = typeof url === 'string' ? url.trim() : '';
    if (trimmed === '') {
      throw new ConfigurationError('Image.fromUrl requires a non-empty URL.');
    }
    if (trimmed.startsWith('data:')) {
      return Image.fromDataUri(trimmed);
    }
    assertSupportedMime(mimeType);
    let parsed: URL;
    try {
      parsed = new URL(trimmed);
    } catch {
      throw new ConfigurationError('Image.fromUrl requires a valid HTTPS URL.');
    }
    if (parsed.protocol !== 'https:') {
      throw new ConfigurationError('Image.fromUrl only accepts HTTPS URLs or data URIs.');
    }
    return new Image({ mimeType, url: trimmed });
  }

  /**
   * Wire-format URL accepted by OpenAI/OpenRouter `image_url.url`.
   * - For URL-backed images, returns the original URL unchanged.
   * - For buffer/base64-backed images, returns `data:<mime>;base64,…`.
   */
  toDataUri(): string {
    if (this.url !== undefined) {
      return this.url;
    }
    if (this.#bytes === undefined) {
      // Defensive: constructor invariants forbid this branch.
      throw new ConfigurationError('Image instance has neither url nor bytes.');
    }
    return `data:${this.mimeType};base64,${bytesToBase64(this.#bytes)}`;
  }
}

export function isImage(value: unknown): value is Image {
  return (
    typeof value === 'object'
    && value !== null
    && (value as { _isDspyImage?: unknown })._isDspyImage === true
    && value instanceof Image
  );
}
