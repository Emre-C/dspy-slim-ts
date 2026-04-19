export type { FieldKind, TypeTag, Role, ModelType, AdapterKind, LMLike, AdapterLike } from './types.js';
export { FIELD_KINDS, TYPE_TAGS, isFieldKind, isTypeTag } from './types.js';

export { isObjectLike, isPlainObject } from './guards.js';

export type { JSONSchemaType } from './codec.js';
export {
  coerceBoolean,
  coerceNumber,
  coerceJsonContainer,
  coerceFieldValue,
  typeTagToSchemaType,
  schemaTypeToTypeTag,
} from './codec.js';

export type { Parameter } from './parameter.js';
export { isParameter, markParameter } from './parameter.js';

export { inferPrefix } from './infer_prefix.js';

export type { Callback, CallbackDispatchKind } from './callback_types.js';
export { BaseCallback, currentCallID } from './callback.js';

export type { FieldInit } from './field.js';
export { Field, createField } from './field.js';

export type { ParsedField, ParseResult, SignatureOptions } from './signature.js';
export {
  Signature,
  createSignature,
  ensureSignature,
  appendField,
  prependField,
  deleteField,
  withInstructions,
  withUpdatedField,
  signatureEquals,
  signatureString,
  signatureFields,
  parseSignature,
  signatureFromString,
} from './signature.js';

export type {
  ParseSignature,
  IsStringLiteral,
  InferInputs,
  InferOutputs,
  SignatureInput,
  Flatten,
} from './signature_types.js';

export { Example } from './example.js';

export type { HistoryMessage } from './history.js';
export { History, isHistoryLike } from './history.js';

export { Completions, Prediction } from './prediction.js';

export type { JSONSchema, ToolInput, ToolOptions, ToolCall } from './tool.js';
export { Tool, ToolCalls } from './tool.js';

export type {
  ContentPart,
  Message,
  Demo,
  AdapterOptions,
} from './adapter.js';
export {
  Adapter,
  AdapterParseError,
  ChatAdapter,
  JSONAdapter,
} from './adapter.js';

export type {
  BaseLMOptions,
  LMOptions,
  LMResponse,
  ChatCompletionResponse,
  ResponsesResponse,
  LMOutput,
  LMOutputEnvelope,
  HistoryEntry,
  ToolCallWire,
} from './lm.js';
export { BaseLM, LM, getGlobalHistory, resetGlobalHistory } from './lm.js';

export type { ProviderProfile } from './providers/profile.js';
export { registerProfile, resolveProfile } from './providers/index.js';

export type { GoldenTranscriptEntry } from './replay_lm.js';
export { ReplayLM } from './replay_lm.js';

export type { PredictorLike } from './module.js';
export {
  BaseModule,
  Module,
  isPredictorLike,
  markPredictor,
} from './module.js';

export type {
  PredictTrace,
  PredictPreprocessResult,
  PredictForwardOverrides,
  PredictKwargs,
} from './predict.js';
export { Predict } from './predict.js';

export type {
  ParallelOptions,
  ParallelCallable,
  ParallelInput,
  ParallelTarget,
  ParallelExecPair,
  ParallelResults,
  ParallelFailureBundle,
  ParallelForwardResult,
} from './parallel.js';
export { Parallel } from './parallel.js';

export type {
  EvaluationScore,
  EvaluationRow,
  EvaluationMetric,
  EvaluateProgram,
  EvaluableProgram,
  EvaluateOptions,
  EvaluateCallOptions,
} from './evaluate.js';
export { Evaluate, EvaluationResult } from './evaluate.js';

export { ChainOfThought } from './chain_of_thought.js';
export type { ReActKwargs } from './react.js';
export { ReAct } from './react.js';
export type {
  BudgetVector,
  CodeInterpreterError,
  REPLVariable,
  REPLEntryKind,
  REPLEntry,
  REPLHistory,
  InterpreterPatch,
  ExecuteRequest,
  FinalOutput,
  ExecuteResult,
  CodeSession,
  CodeInterpreter,
  LLMQueryRequest,
  LLMQueryResult,
  RLMConfig,
  RLMRunResult,
} from './rlm_types.js';
export { NodeCodeInterpreter, createNodeCodeInterpreter } from './node_code_interpreter.js';
export type { NodeCodeInterpreterOptions, SyncCodeInterpreter, SyncCodeSession } from './node_code_interpreter.js';
export type { RLMOptions } from './rlm.js';
export { RLM } from './rlm.js';
export type {
  PredictorTarget,
  MetricRecord,
  PredictorTrace,
  ReflectiveDatum,
  InstructionCell,
  ProgramProjection,
  CandidateVector,
  InstructionProposal,
  OptimizationArtifact,
  GEPAEngineRequest,
  GEPAEngineResult,
  GEPAEngine,
  GEPAAdapter,
  GEPAConfig,
  GEPACompileResult,
} from './gepa_types.js';
export {
  normalizeMetricRecord,
  projectPredictorTargets,
  capturePredictorTraces,
  materializeReflectiveDataset,
  ensureNonEmptyTargets,
} from './gepa_trace.js';
export {
  GEPA,
  createGatedGEPAEngine,
  createStaticGEPAEngine,
  createModuleGEPAAdapter,
  getOptimizationArtifact,
} from './gepa.js';

export {
  DSPyError,
  ValueError,
  BudgetError,
  KeyError,
  RuntimeError,
  ConfigurationError,
  InvariantError,
  ContextWindowExceededError,
} from './exceptions.js';

export type { SettingsOverrides, SettingsSnapshot } from './settings.js';
export { Settings, settings } from './settings.js';
