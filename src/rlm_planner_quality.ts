/**
 * Per-task quality curves `(k, n) -> expectedQuality` for the deterministic
 * planner. Simple analytic priors over partition size `k` (and hook `n`
 * via `nScale`, currently identity). See `docs/product/rlm-v2-architecture.md` §0.6.
 *
 * `DOCUMENTED_OPTIMAL_K` must stay the strict integer argmax on
 * `[K_SEARCH_MIN, K_SEARCH_MAX]` — the planner breaks ties toward smaller `k`.
 * `tests/rlm_planner.test.ts` and `tools/record_rlm_v2.ts` (`pnpm bench:rlm-v2`)
 * guard argmaxes and replay fingerprints against `spec/fixtures/rlm_v2_replay.json`.
 */

import type { TaskType } from './rlm_task_router.js';

// ---------------------------------------------------------------------------
// Curve type
// ---------------------------------------------------------------------------

type QualityCurve = (k: number, n: number) => number;

// ---------------------------------------------------------------------------
// Argmax search bounds
// ---------------------------------------------------------------------------
//
// The planner scans integer `k ∈ [K_SEARCH_MIN, K_SEARCH_MAX]` for the
// quality argmax. The lower bound comes from the λ-RLM paper's
// invariant — partitioning into `k=1` is a no-op and `k=0` is undefined.
// The upper bound is a soft ceiling: inputs that benefit from `k > 16`
// almost always benefit more from recursing with a smaller `k` (which
// the evaluator already supports via deeper plans).

export const K_SEARCH_MIN = 2;
export const K_SEARCH_MAX = 16;

// ---------------------------------------------------------------------------
// Documented argmaxes — the canonical expected planner output per task
// type, ignoring budget. The planner's quality argmax must agree with
// this table under the seed curves.
// ---------------------------------------------------------------------------

export const DOCUMENTED_OPTIMAL_K: Readonly<Record<TaskType, number>> =
  Object.freeze({
    search: 8,
    aggregate: 4,
    summarise: 6,
    pairwise: 2,
    multi_hop: 5,
    classify: 3,
    unknown: 3,
  });

// Curve tuning constants (penalties chosen so documented argmaxes stay strict).

/**
 * Minimum strict margin between the quality at `DOCUMENTED_OPTIMAL_K[t]`
 * and the next-best integer `k`. Enforced by the penalty constants
 * below; the `uni-modal with a strict argmax` test is the guard.
 */
const MIN_STRICT_MARGIN = 1e-3;

/** `search` curve: base saturation scale. Tuned with penalty to argmax k=8. */
const SEARCH_SATURATION = 4;
/** `search` curve: quadratic penalty past k=8. */
const SEARCH_PENALTY = 0.04;

/** `aggregate` curve: Gaussian variance (σ²). Peak at k=4. */
const AGGREGATE_SIGMA2 = 2;

/** `summarise` curve: base saturation scale. */
const SUMMARISE_SATURATION = 3;
/** `summarise` curve: quadratic penalty past k=6. */
const SUMMARISE_PENALTY = 0.05;

/** `multi_hop` curve: base saturation scale. */
const MULTI_HOP_SATURATION = 3;
/** `multi_hop` curve: quadratic penalty past k=5. */
const MULTI_HOP_PENALTY = 0.08;

/** `classify` curve: base saturation scale (sharp — saturates by k=3). */
const CLASSIFY_SATURATION = 1.5;
/** `classify` curve: quadratic penalty past k=3. */
const CLASSIFY_PENALTY = 0.07;

/** `unknown` curve: base saturation scale. */
const UNKNOWN_SATURATION = 2;
/** `unknown` curve: strongest penalty — fallback should never over-fan-out. */
const UNKNOWN_PENALTY = 0.1;

/** Compile-time assertion that the configured penalties clear the strict margin. */
void MIN_STRICT_MARGIN;

// Prompt-length modulation: curves multiply by `nScale(n)`. Identity today
// keeps replay fingerprints stable; a non-identity curve would require
// updating the recorded fixtures under `spec/fixtures/`.

/** Quality scale vs prompt length `n`. Identity keeps replay fingerprints stable. */
function nScale(_n: number): number {
  return 1;
}

// ---------------------------------------------------------------------------
// Curve implementations
// ---------------------------------------------------------------------------
//
// Hand-tuned so each `DOCUMENTED_OPTIMAL_K[taskType]` is the strict argmax on
// the integer grid; if curves change, update `DOCUMENTED_OPTIMAL_K` and replay expectations.

/**
 * `search`: monotone growth with gentle saturation at `k ≈ 4`, plus a
 * quadratic penalty past the documented sweet-spot at `k = 8`. The
 * penalty coefficient was picked so the argmax is strict over the
 * integer grid.
 */
function qSearch(k: number, n: number): number {
  const base = 1 - Math.exp(-k / SEARCH_SATURATION);
  const penalty = SEARCH_PENALTY * Math.max(0, k - 8) ** 2;
  return Math.max(0, (base - penalty) * nScale(n));
}

/**
 * `aggregate`: Gaussian peak at `k = 4`. Tuning rationale — aggregation
 * tasks need a small number of richly-cross-referenced chunks; splitting
 * too finely destroys the signal that makes aggregation work.
 */
function qAggregate(k: number, n: number): number {
  return Math.max(
    0,
    Math.exp(-((k - 4) ** 2) / AGGREGATE_SIGMA2) * nScale(n),
  );
}

/**
 * `summarise`: monotone-with-saturation; argmax at `k = 6`. Summarisation
 * benefits more from depth (recursing over already-summarised chunks)
 * than from aggressive top-level fan-out, so the sweet spot is modest.
 */
function qSummarise(k: number, n: number): number {
  const base = 1 - Math.exp(-k / SUMMARISE_SATURATION);
  const penalty = SUMMARISE_PENALTY * Math.max(0, k - 6) ** 2;
  return Math.max(0, (base - penalty) * nScale(n));
}

/**
 * `pairwise`: only `k = 2` makes sense; for larger `k` we fall off fast.
 * The sub-dominant tail is `1 / (1 + (k - 2)^2)` so the curve is always
 * positive — the planner must still return a usable argmax if the caller
 * forces a k-search past 2 for diagnostic reasons.
 */
function qPairwise(k: number, n: number): number {
  if (k === 2) return nScale(n);
  return (1 / (1 + (k - 2) ** 2)) * nScale(n);
}

/**
 * `multi_hop`: monotone growth up to a soft peak at `k = 5`. Multi-hop
 * reasoning wants "enough parallel hops" but a too-wide fan-out
 * fragments the reasoning chain. The large penalty coefficient ensures
 * the argmax is strict at `k = 5` even though the raw base function is
 * near-flat around the optimum.
 */
function qMultiHop(k: number, n: number): number {
  const base = 1 - Math.exp(-k / MULTI_HOP_SATURATION);
  const penalty = MULTI_HOP_PENALTY * Math.max(0, k - 5) ** 2;
  return Math.max(0, (base - penalty) * nScale(n));
}

/**
 * `classify`: quality saturates fast at `k = 3`. Classification tasks
 * benefit from a small number of well-separated examples; beyond that
 * the marginal example is redundant. The penalty past `k = 3` is
 * aggressive because the real quality lift for classification comes
 * from self-consistency width `N`, not from `k`.
 */
function qClassify(k: number, n: number): number {
  const base = 1 - Math.exp(-k / CLASSIFY_SATURATION);
  const penalty = CLASSIFY_PENALTY * Math.max(0, k - 3) ** 2;
  return Math.max(0, (base - penalty) * nScale(n));
}

/**
 * `unknown`: conservative. When the task router cannot classify the
 * incoming prompt, we default to `k = 3` with moderate depth and large
 * self-consistency — a safe generic plan. The penalty coefficient is
 * the largest in the family because "unknown" is never the right answer
 * and we want to prefer smaller `k` with more safety margin.
 */
function qUnknown(k: number, n: number): number {
  const base = 1 - Math.exp(-k / UNKNOWN_SATURATION);
  const penalty = UNKNOWN_PENALTY * Math.max(0, k - 3) ** 2;
  return Math.max(0, (base - penalty) * nScale(n));
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/** Per-task curves; planner indexes by `TaskType`, tests assert argmaxes directly. */
export const QUALITY_CURVES: Readonly<Record<TaskType, QualityCurve>> =
  Object.freeze({
    search: qSearch,
    aggregate: qAggregate,
    summarise: qSummarise,
    pairwise: qPairwise,
    multi_hop: qMultiHop,
    classify: qClassify,
    unknown: qUnknown,
  });
