/**
 * Shared runtime exceptions.
 */

export class DSPyError extends Error {
  constructor(name: string, message: string) {
    super(message);
    this.name = name;
  }
}

export class ValueError extends DSPyError {
  constructor(message: string) {
    super('ValueError', message);
  }
}

export class KeyError extends DSPyError {
  constructor(message: string) {
    super('KeyError', message);
  }
}

export class RuntimeError extends DSPyError {
  constructor(message: string) {
    super('RuntimeError', message);
  }
}

export class BudgetError extends ValueError {
  constructor(message: string) {
    super(message);
    this.name = 'BudgetError';
  }
}

export class ConfigurationError extends ValueError {
  constructor(message: string) {
    super(message);
    this.name = 'ConfigurationError';
  }
}

export class InvariantError extends RuntimeError {
  constructor(message: string) {
    super(message);
    this.name = 'InvariantError';
  }
}

export class ContextWindowExceededError extends RuntimeError {
  readonly model: string | null;

  constructor(options: { readonly model?: string | null; readonly message?: string } = {}) {
    const model = options.model ?? null;
    const message = options.message ?? 'Context window exceeded';
    super(model === null ? message : `[${model}] ${message}`);
    this.name = 'ContextWindowExceededError';
    this.model = model;
  }
}
