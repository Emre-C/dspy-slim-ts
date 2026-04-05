/**
 * Shared runtime exceptions.
 */

export class ValueError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ValueError';
  }
}

export class ContextWindowExceededError extends Error {
  readonly model: string | null;

  constructor(options: { readonly model?: string | null; readonly message?: string } = {}) {
    const model = options.model ?? null;
    const message = options.message ?? 'Context window exceeded';
    super(model === null ? message : `[${model}] ${message}`);
    this.name = 'ContextWindowExceededError';
    this.model = model;
  }
}
