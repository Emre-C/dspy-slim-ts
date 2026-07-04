import { describe, expect, it } from 'vitest';
import { parseSignature, signatureFromString } from '../src/index.js';

describe('Signature parser — Image alias', () => {
  it('accepts an Image-typed input field', () => {
    const parsed = parseSignature('page_image: Image, n: int -> a: str');
    expect(parsed.inputs).toHaveLength(2);
    expect(parsed.inputs[0]).toMatchObject({
      name: 'page_image',
      typeTag: 'image',
      isTypeUndefined: false,
    });
    expect(parsed.inputs[1]).toMatchObject({
      name: 'n',
      typeTag: 'int',
    });
    expect(parsed.outputs[0]).toMatchObject({
      name: 'a',
      typeTag: 'str',
    });
  });

  it('still parses non-image signatures verbatim', () => {
    const parsed = parseSignature('a: str -> b: int');
    expect(parsed.inputs[0]).toMatchObject({ name: 'a', typeTag: 'str' });
    expect(parsed.outputs[0]).toMatchObject({ name: 'b', typeTag: 'int' });
  });

  it('produces a Signature with image typeTag for Image fields', () => {
    const sig = signatureFromString('page_image: Image -> answer: str');
    const field = sig.inputFields.get('page_image')!;
    expect(field.typeTag).toBe('image');
    expect(field.isTypeUndefined).toBe(false);
  });

  it('does not alias other capitalized type names', () => {
    // `Str`, `Int`, etc. should still fall through to `custom` so we
    // do not silently mask user typos.
    const parsed = parseSignature('a: Str -> b: Int');
    expect(parsed.inputs[0]).toMatchObject({ name: 'a', typeTag: 'custom' });
    expect(parsed.outputs[0]).toMatchObject({ name: 'b', typeTag: 'custom' });
  });
});
