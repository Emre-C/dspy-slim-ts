import { describe, expect, it } from 'vitest';
import { ConfigurationError, RLM } from '../src/index.js';

describe('RLM + Image fields', () => {
  it('rejects native Image input fields in phase 1', () => {
    expect(() => new RLM('page_image: Image, prompt: str -> answer: str')).toThrow(
      ConfigurationError,
    );
  });
});
