/**
 * Shared GEPA substrate contracts.
 */

import type { REPLHistory } from './rlm_types.js';

export interface PredictorTarget {
  readonly targetId: number;
  readonly predictorIndex: number;
  readonly path: readonly number[];
}

export interface MetricRecord {
  readonly score: number | null;
  readonly subscores: readonly number[];
  readonly feedback: unknown | null;
  readonly failed: boolean;
}

export interface PredictorTrace<TInput = unknown, TOutput = unknown> {
  readonly exampleId: number;
  readonly target: PredictorTarget;
  readonly input: TInput;
  readonly output: TOutput | null;
  readonly metric: MetricRecord;
  readonly history: REPLHistory | null;
}

export interface ReflectiveDatum<TInput = unknown, TOutput = unknown> {
  readonly datumId: number;
  readonly target: PredictorTarget;
  readonly trace: PredictorTrace<TInput, TOutput>;
}

export interface InstructionCell<TInstruction = unknown> {
  readonly target: PredictorTarget;
  readonly instruction: TInstruction;
}

export interface ProgramProjection<TInstruction = unknown> {
  readonly targets: readonly PredictorTarget[];
  readonly instructions: readonly InstructionCell<TInstruction>[];
}

export interface CandidateVector {
  readonly candidateId: number;
  readonly objective: readonly number[];
  readonly feasible: boolean;
}

export interface InstructionProposal<TInstruction = unknown> {
  readonly candidateId: number;
  readonly target: PredictorTarget;
  readonly instruction: TInstruction;
}

export interface OptimizationArtifact<TInstruction = unknown> {
  readonly tracked: boolean;
  readonly selectedCandidateId: number | null;
  readonly frontier: readonly CandidateVector[];
  readonly instructionMap: readonly InstructionProposal<TInstruction>[];
}

export interface GEPAEngineRequest<TInstruction = unknown> {
  readonly projection: ProgramProjection<TInstruction>;
  readonly trainset: readonly ReflectiveDatum[];
  readonly valset: readonly ReflectiveDatum[];
  readonly tracked: boolean;
}

export interface GEPAEngineResult<TInstruction = unknown> {
  readonly artifact: OptimizationArtifact<TInstruction>;
}

export interface GEPAEngine<TInstruction = unknown> {
  readonly capability: 'available' | 'gated';
  readonly optimize: (
    request: GEPAEngineRequest<TInstruction>,
  ) => Promise<GEPAEngineResult<TInstruction>>;
}

export interface GEPAAdapter<TProgram, TInstruction = unknown> {
  readonly project: (program: TProgram) => ProgramProjection<TInstruction>;
  readonly buildTrainset: (
    traces: readonly PredictorTrace[],
  ) => readonly ReflectiveDatum[];
  readonly buildValset: (
    traces: readonly PredictorTrace[],
  ) => readonly ReflectiveDatum[];
  readonly attach: (
    program: TProgram,
    artifact: OptimizationArtifact<TInstruction>,
  ) => TProgram;
}

export interface GEPAConfig<TProgram, TInstruction = unknown> {
  readonly tracked: boolean;
  readonly adapter: GEPAAdapter<TProgram, TInstruction>;
  readonly engine: GEPAEngine<TInstruction>;
}

export interface GEPACompileResult<TProgram, TInstruction = unknown> {
  readonly program: TProgram;
  readonly artifact: OptimizationArtifact<TInstruction>;
}
