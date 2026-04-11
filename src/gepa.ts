/**
 * GEPA facade and adapter boundary.
 */

import { RuntimeError, ValueError } from './exceptions.js';
import type { Module } from './module.js';
import { Signature, withInstructions } from './signature.js';
import {
  ensureNonEmptyTargets,
  materializeReflectiveDataset,
  projectPredictorTargets,
} from './gepa_trace.js';
import type {
  CandidateVector,
  GEPAAdapter,
  GEPACompileResult,
  GEPAConfig,
  GEPAEngine,
  InstructionCell,
  OptimizationArtifact,
  PredictorTrace,
  ProgramProjection,
} from './gepa_types.js';

const OPTIMIZATION_ARTIFACT_SYMBOL = Symbol.for('dspy.gepa.optimization_artifact');

function cloneCandidateVector(vector: CandidateVector): CandidateVector {
  return Object.freeze({
    candidateId: vector.candidateId,
    objective: Object.freeze([...vector.objective]),
    feasible: vector.feasible,
  });
}

function normalizeArtifact<TInstruction>(
  artifact: OptimizationArtifact<TInstruction>,
  tracked: boolean,
): OptimizationArtifact<TInstruction> {
  const selectedCandidateId = artifact.selectedCandidateId;
  if (
    selectedCandidateId !== null
    && !Number.isInteger(selectedCandidateId)
  ) {
    throw new ValueError('Optimization artifact selectedCandidateId must be an integer or null.');
  }

  return Object.freeze({
    tracked,
    selectedCandidateId,
    frontier: Object.freeze(artifact.frontier.map((candidate) => cloneCandidateVector(candidate))),
    instructionMap: Object.freeze(artifact.instructionMap.map((proposal) => Object.freeze({
      candidateId: proposal.candidateId,
      target: Object.freeze({
        targetId: proposal.target.targetId,
        predictorIndex: proposal.target.predictorIndex,
        path: Object.freeze([...proposal.target.path]),
      }),
      instruction: proposal.instruction,
    }))),
  });
}

interface PredictLike {
  signature: Signature;
}

function asPredictLike(predictor: unknown): PredictLike | null {
  const candidate = predictor as Partial<PredictLike>;
  if (candidate.signature instanceof Signature) {
    return candidate as PredictLike;
  }
  return null;
}

function readPredictorInstruction(predictor: unknown): string | null {
  return asPredictLike(predictor)?.signature.instructions ?? null;
}

function applyPredictorInstruction(
  predictor: unknown,
  instruction: string,
): void {
  const predict = asPredictLike(predictor);
  if (!predict) return;
  predict.signature = withInstructions(predict.signature, instruction);
}

function attachOptimizationArtifact<TProgram>(
  program: TProgram,
  artifact: OptimizationArtifact,
): void {
  Object.defineProperty(program as object, OPTIMIZATION_ARTIFACT_SYMBOL, {
    value: artifact,
    enumerable: false,
    configurable: true,
    writable: true,
  });
}

export function getOptimizationArtifact<TInstruction = unknown>(
  program: unknown,
): OptimizationArtifact<TInstruction> | null {
  if (!program || typeof program !== 'object') {
    return null;
  }

  const carrier = program as {
    readonly [OPTIMIZATION_ARTIFACT_SYMBOL]?: OptimizationArtifact<TInstruction>;
  };
  return carrier[OPTIMIZATION_ARTIFACT_SYMBOL] ?? null;
}

export function createGatedGEPAEngine<TInstruction = unknown>(
  reason = 'No approved GEPA engine backend is configured.',
): GEPAEngine<TInstruction> {
  return Object.freeze({
    capability: 'gated',
    optimize: async () => {
      throw new RuntimeError(reason);
    },
  });
}

export function createStaticGEPAEngine<TInstruction = unknown>(
  artifact: OptimizationArtifact<TInstruction>,
): GEPAEngine<TInstruction> {
  const frozenArtifact = normalizeArtifact(artifact, artifact.tracked);
  return Object.freeze({
    capability: 'available',
    optimize: async () => ({ artifact: frozenArtifact }),
  });
}

export function createModuleGEPAAdapter<
  TProgram extends Module,
  TInstruction = string,
>(): GEPAAdapter<TProgram, TInstruction> {
  return Object.freeze({
    project: (program: TProgram): ProgramProjection<TInstruction> => {
      const targets = projectPredictorTargets(program);
      ensureNonEmptyTargets(targets);
      const namedPredictors = program.namedPredictors();
      const instructions: InstructionCell<TInstruction>[] = [];

      for (const [index, [, predictor]] of namedPredictors.entries()) {
        const instruction = readPredictorInstruction(predictor);
        if (instruction === null) {
          continue;
        }

        instructions.push(Object.freeze({
          target: targets[index]!,
          instruction: instruction as unknown as TInstruction,
        }));
      }

      return Object.freeze({
        targets,
        instructions: Object.freeze(instructions),
      });
    },
    buildTrainset: (traces: readonly PredictorTrace[]) => materializeReflectiveDataset(traces),
    buildValset: (traces: readonly PredictorTrace[]) => materializeReflectiveDataset(traces),
    attach: (
      program: TProgram,
      artifact: OptimizationArtifact<TInstruction>,
    ): TProgram => {
      const clone = program.deepcopy() as TProgram;
      const namedPredictors = clone.namedPredictors();

      for (const proposal of artifact.instructionMap) {
        if (
          artifact.selectedCandidateId !== null
          && proposal.candidateId !== artifact.selectedCandidateId
        ) {
          continue;
        }

        const predictorEntry = namedPredictors[proposal.target.predictorIndex];
        if (!predictorEntry) {
          continue;
        }

        applyPredictorInstruction(
          predictorEntry[1],
          String(proposal.instruction),
        );
      }

      attachOptimizationArtifact(clone, normalizeArtifact(artifact, artifact.tracked));
      return clone;
    },
  });
}

export class GEPA<TProgram, TInstruction = unknown> {
  readonly tracked: boolean;
  readonly adapter: GEPAAdapter<TProgram, TInstruction>;
  readonly engine: GEPAEngine<TInstruction>;

  constructor(config: GEPAConfig<TProgram, TInstruction>) {
    this.tracked = config.tracked;
    this.adapter = config.adapter;
    this.engine = config.engine;
  }

  get capability(): GEPAEngine<TInstruction>['capability'] {
    return this.engine.capability;
  }

  async compile(
    student: TProgram,
    trainset: readonly PredictorTrace[] = Object.freeze([]),
    valset: readonly PredictorTrace[] = Object.freeze([]),
  ): Promise<GEPACompileResult<TProgram, TInstruction>> {
    if (this.engine.capability === 'gated') {
      throw new RuntimeError(
        'GEPA compile is gated because no approved external engine is configured.',
      );
    }

    const projection = this.adapter.project(student);
    ensureNonEmptyTargets(projection.targets);

    const trainsetData = this.adapter.buildTrainset(trainset);
    const valsetInput = valset.length > 0 ? valset : trainset;
    const valsetData = this.adapter.buildValset(valsetInput);

    const { artifact } = await this.engine.optimize({
      projection,
      trainset: trainsetData,
      valset: valsetData,
      tracked: this.tracked,
    });

    const normalizedArtifact = normalizeArtifact(artifact, this.tracked);
    const program = this.adapter.attach(student, normalizedArtifact);

    return Object.freeze({
      program,
      artifact: normalizedArtifact,
    });
  }
}
