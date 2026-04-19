export type { FieldKind, TypeTag, Role, ModelType, AdapterKind, LMLike, AdapterLike } from './types.js';
export { FIELD_KINDS, TYPE_TAGS, isFieldKind, isTypeTag } from './types.js';

export { isObjectLike, isPlainObject } from './guards.js';

export type { JsonObject, JsonValue } from './json_value.js';

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

// RLM v2 public surface. Legacy v1 interpreter exports are blocked by
// `tools/deps_rlm_legacy.mjs --strict`.

export type {
  RLMBudget,
  EvaluationContext,
  EvaluationTrace,
  EffectHandler,
  Effect,
  EffectResult,
  OracleResponse,
  QueryOracleCallFn,
  BuildEvaluationContextOptions,
} from './rlm_types.js';
export { DEFAULT_BUDGET, mergeBudget } from './rlm_types.js';

export type {
  CombinatorNode,
  CombinatorFn,
  CombinatorBinary,
  CombinatorValue,
  CombinatorList,
  VoteReducer,
  EnsembleReducer,
} from './rlm_combinators.js';
export {
  split,
  peek,
  map,
  filter,
  reduce,
  concat,
  cross,
  vote,
  ensemble,
  oracle,
  lit,
  vref,
  fn,
  bop,
} from './rlm_combinators.js';

export {
  evaluate,
  buildEvaluationContext,
  RLM_VERIFIER_SIGNATURE,
} from './rlm_evaluator.js';

export type {
  TaskType,
  PlanningInputs,
  ResolvedPlan,
  StaticPlan,
  ClassifierResult,
  RouteResult,
  ResolveRouteOptions,
} from './rlm_task_router.js';
export {
  STATIC_PLANS,
  TASK_TYPES,
  REAL_TASK_TYPES_LIST,
  DEFAULT_ROUTE_THRESHOLD,
  DEFAULT_BEAM_TOP_K,
  classifyTask,
  resolveRoute,
  composeBeamPlan,
  isTaskType,
} from './rlm_task_router.js';

export type { ResolvePlanArgs } from './rlm_planner.js';
export { resolvePlan } from './rlm_planner.js';

// Effects protocol, oracle wire signature, parser/serializer, built-in
// handlers, and `builtInEffectHandlers` for custom evaluator wiring.
export {
  EFFECT_KINDS,
  EFFECT_ORACLE_SIGNATURE,
  appendEffectResult,
  builtInEffectHandlers,
  isEffect,
  isEffectResult,
  parseOracleResponse,
  queryOracleHandler,
  readContextHandler,
  writeMemoryHandler,
  yieldHandler,
} from './rlm_effects.js';

export type {
  MemorySchema,
  MemoryFieldSchema,
  MemoryInjector,
  MemoryWrite,
  MemoryScalar,
  MemoryScalarType,
  TypedMemoryState,
} from './rlm_memory.js';
export {
  DEFAULT_MAX_MEMORY_BYTES,
  applyMemoryWrite,
  defaultMemoryInjector,
  initialMemoryState,
} from './rlm_memory.js';

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
