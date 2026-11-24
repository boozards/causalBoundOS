"""Core Engine A: Exact Coupled Slice-Threshold Optimizer.

Solves the 16-slice coupled policy optimization problem to guaranteed global
optimality under:
  1. Per-slice calibration gate:        ECE_m <= max_ece_m
  2. Per-slice false-ban ceiling:       FPR_m(t_m) <= max_fpr_m
  3. Pairwise TPR fairness delta:       |TPR_i(t_i) - TPR_j(t_j)| <= delta
  4. Global non-separable FP budget:    sum_m FP_m(t_m) <= max_global_fp

Algorithmic Reduction:
  |TPR_i - TPR_j| <= delta for all pairs (i, j)  <==>  max_m TPR_m - min_m TPR_m <= delta.
  Enumerating each distinct candidate TPR value L as the window lower bound
  restricts every slice m to candidates with TPR_m in [L, L + delta]. For each
  unique feasible window, the remaining problem is a Multiple-Choice Knapsack
  Problem (MCKP) over integer global FP budget B, solved exactly via DP with
  exact 3-tier Pareto-dominance pruning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from causalbound.schemas import (
    PolicyOptimizationRequest,
    PolicyOptimizationResult,
    RiskSample,
    SliceInput,
    SliceMetricOutput,
)

ECE_NUM_BINS: int = 10
SMALL_N_ECE_CUTOFF: int = 5


@dataclass(frozen=True, slots=True)
class SliceCandidate:
    """Precomputed exact statistics for one (slice, threshold) pair."""

    threshold_dec: Decimal
    threshold_float: float
    fp: int
    fn: int
    tp: int
    tn: int
    tpr: Fraction
    fpr: Fraction
    loss_cents: int


def compute_slice_ece(samples: list[RiskSample]) -> float:
    """Compute 10-bin Expected Calibration Error with mandatory N < 5 fallback.

    Bins: [0.0, 0.1), [0.1, 0.2), ..., [0.8, 0.9), [0.9, 1.0] (right-inclusive).
    """
    n_total = len(samples)
    if n_total < SMALL_N_ECE_CUTOFF:
        return 0.0

    bin_counts = [0] * ECE_NUM_BINS
    bin_score_sums = [0.0] * ECE_NUM_BINS
    bin_label_sums = [0] * ECE_NUM_BINS

    for sample in samples:
        score = sample.score
        bin_idx = min(int(math.floor(score * ECE_NUM_BINS)), ECE_NUM_BINS - 1)
        if bin_idx < 0:
            bin_idx = 0
        bin_counts[bin_idx] += 1
        bin_score_sums[bin_idx] += score
        bin_label_sums[bin_idx] += sample.label

    ece = 0.0
    for count, score_sum, label_sum in zip(bin_counts, bin_score_sums, bin_label_sums):
        if count == 0:
            continue
        avg_conf = score_sum / count
        avg_acc = label_sum / count
        ece += (count / n_total) * abs(avg_acc - avg_conf)

    if not math.isfinite(ece):
        raise ValueError(f"Non-finite ECE computed: {ece}")
    return ece


def _build_slice_candidates(
    slice_data: SliceInput,
    sorted_thresholds: list[Decimal],
) -> list[SliceCandidate]:
    """Evaluate all candidate thresholds for a single slice using exact arithmetic."""
    samples = slice_data.samples
    pos_total = sum(1 for s in samples if s.label == 1)
    neg_total = len(samples) - pos_total
    max_fpr_frac = Fraction(str(slice_data.max_fpr))

    candidates: list[SliceCandidate] = []
    for t_dec in sorted_thresholds:
        t_float = float(t_dec)
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for s in samples:
            predicted_pos = s.score >= t_float
            if predicted_pos:
                if s.label == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if s.label == 1:
                    fn += 1
                else:
                    tn += 1

        # Mandatory zero-division guards
        tpr = Fraction(1, 1) if pos_total == 0 else Fraction(tp, pos_total)
        fpr = Fraction(0, 1) if neg_total == 0 else Fraction(fp, neg_total)

        if fpr > max_fpr_frac:
            continue

        loss_cents = slice_data.fp_cost_cents * fp + slice_data.fn_cost_cents * fn
        candidates.append(
            SliceCandidate(
                threshold_dec=t_dec,
                threshold_float=t_float,
                fp=fp,
                fn=fn,
                tp=tp,
                tn=tn,
                tpr=tpr,
                fpr=fpr,
                loss_cents=loss_cents,
            )
        )
    return candidates


def _dedup_same_fp(candidates: list[SliceCandidate]) -> list[SliceCandidate]:
    """Within one slice, keep only the strictly best candidate per exact FP count.

    For equal FP, lower (loss_cents, threshold_dec) strictly improves or ties all
    3 tie-breaking tiers simultaneously without consuming additional FP budget.
    """
    best_by_fp: dict[int, SliceCandidate] = {}
    for cand in candidates:
        existing = best_by_fp.get(cand.fp)
        if existing is None or (cand.loss_cents, cand.threshold_dec) < (
            existing.loss_cents,
            existing.threshold_dec,
        ):
            best_by_fp[cand.fp] = cand
    return [best_by_fp[fp] for fp in sorted(best_by_fp)]


# DP value type: (total_loss_cents, sum_threshold_dec, threshold_tuple, candidate_tuple)
_DPValue = tuple[int, Decimal, tuple[Decimal, ...], tuple[SliceCandidate, ...]]


def _is_better(lhs: _DPValue, rhs: _DPValue) -> bool:
    """Exact 3-tier deterministic tie-breaking comparison."""
    return (lhs[0], lhs[1], lhs[2]) < (rhs[0], rhs[1], rhs[2])


def _prune_pareto(dp: dict[int, _DPValue]) -> dict[int, _DPValue]:
    """Discard states with higher FP and worse-or-equal 3-tier objective key.

    A state (fp_b, key_b) is strictly dominated by (fp_a, key_a) iff:
      fp_a < fp_b  AND  key_a <= key_b
    Because key_a uses the full 3-tier tuple (loss, sum_t, lex_tuple), any
    higher-FP state with better Tier 2 or Tier 3 key is preserved.
    """
    pruned: dict[int, _DPValue] = {}
    best_key: tuple[int, Decimal, tuple[Decimal, ...]] | None = None

    for fp in sorted(dp):
        val = dp[fp]
        key = (val[0], val[1], val[2])
        if best_key is None or key < best_key:
            pruned[fp] = val
            best_key = key
    return pruned


def _solve_mckp_window(
    window_candidates: list[list[SliceCandidate]],
    max_global_fp: int,
) -> _DPValue | None:
    """Exact Multiple-Choice Knapsack DP across sorted slices for a fixed TPR window."""
    num_slices = len(window_candidates)
    deduped = [_dedup_same_fp(cands) for cands in window_candidates]

    # Precompute suffix minimum achievable FP for early budget-feasibility pruning
    suffix_min_fp = [0] * (num_slices + 1)
    for i in range(num_slices - 1, -1, -1):
        suffix_min_fp[i] = suffix_min_fp[i + 1] + deduped[i][0].fp

    if suffix_min_fp[0] > max_global_fp:
        return None

    dp: dict[int, _DPValue] = {0: (0, Decimal("0"), (), ())}

    for idx, cands in enumerate(deduped):
        rem_min_fp = suffix_min_fp[idx + 1]
        fp_limit = max_global_fp - rem_min_fp
        next_dp: dict[int, _DPValue] = {}

        for prev_fp, (prev_loss, prev_sum_t, prev_t_tuple, prev_cands) in dp.items():
            for cand in cands:
                new_fp = prev_fp + cand.fp
                if new_fp > fp_limit:
                    break  # cands are sorted ascending by fp
                new_val: _DPValue = (
                    prev_loss + cand.loss_cents,
                    prev_sum_t + cand.threshold_dec,
                    prev_t_tuple + (cand.threshold_dec,),
                    prev_cands + (cand,),
                )
                existing = next_dp.get(new_fp)
                if existing is None or _is_better(new_val, existing):
                    next_dp[new_fp] = new_val

        if not next_dp:
            return None
        dp = _prune_pareto(next_dp)

    best_overall: _DPValue | None = None
    for val in dp.values():
        if best_overall is None or _is_better(val, best_overall):
            best_overall = val
    return best_overall


def _build_failed_result(
    request: PolicyOptimizationRequest,
    slice_ece: dict[str, float],
    violated_constraints: list[str],
) -> PolicyOptimizationResult:
    """Construct strict FAILED_AUDIT payload with null selections and sorted diagnostics."""
    return PolicyOptimizationResult(
        policy_id=request.policy_id,
        evaluated_at_utc=request.evaluated_at_utc,
        audit_status="FAILED_AUDIT",
        selected_thresholds=None,
        total_loss_cents=None,
        total_fp=None,
        total_fn=None,
        global_tpr_spread=None,
        slice_ece=slice_ece,
        slice_metrics=None,
        violated_constraints=sorted(set(violated_constraints)),
    )


def optimize_policy(request: PolicyOptimizationRequest) -> PolicyOptimizationResult:
    """Solve coupled multi-slice threshold optimization to exact global optimality."""
    # Step 1: Deterministic slice & threshold ordering (never assume pre-sorted input)
    ordered_slices = sorted(request.slices, key=lambda s: s.slice_id)
    sorted_thresholds = sorted({Decimal(str(t)) for t in request.candidate_thresholds})

    # Step 2: Evaluate per-slice calibration (ECE) across all slices
    slice_ece: dict[str, float] = {}
    ece_violations: list[str] = []
    for sl in ordered_slices:
        ece_val = compute_slice_ece(sl.samples)
        slice_ece[sl.slice_id] = ece_val
        if ece_val > sl.max_ece:
            ece_violations.append(f"SLICE_ECE_EXCEEDED:{sl.slice_id}")

    if ece_violations:
        return _build_failed_result(request, slice_ece, ece_violations)

    # Step 3: Build per-slice FPR-feasible candidates
    candidates_per_slice: list[list[SliceCandidate]] = []
    fpr_violations: list[str] = []
    for sl in ordered_slices:
        cands = _build_slice_candidates(sl, sorted_thresholds)
        if not cands:
            fpr_violations.append(f"SLICE_FPR_INFEASIBLE:{sl.slice_id}")
        candidates_per_slice.append(cands)

    if fpr_violations:
        return _build_failed_result(request, slice_ece, fpr_violations)

    # Step 4: Enumerate exact [L, L + delta] TPR fairness windows
    delta_frac = Fraction(str(request.pairwise_fairness_delta))
    distinct_tprs = sorted({c.tpr for cands in candidates_per_slice for c in cands})

    seen_window_signatures: set[tuple[tuple[Decimal, ...], ...]] = set()
    any_fairness_window_feasible = False
    best_solution: _DPValue | None = None

    for lower_tpr in distinct_tprs:
        upper_tpr = lower_tpr + delta_frac
        window_cands: list[list[SliceCandidate]] = []
        window_feasible = True

        for cands in candidates_per_slice:
            in_window = [c for c in cands if lower_tpr <= c.tpr <= upper_tpr]
            if not in_window:
                window_feasible = False
                break
            window_cands.append(in_window)

        if not window_feasible:
            continue

        any_fairness_window_feasible = True

        # Deduplicate identical candidate windows across adjacent L values
        sig = tuple(tuple(c.threshold_dec for c in wc) for wc in window_cands)
        if sig in seen_window_signatures:
            continue
        seen_window_signatures.add(sig)

        candidate_sol = _solve_mckp_window(window_cands, request.max_global_fp)
        if candidate_sol is not None:
            if best_solution is None or _is_better(candidate_sol, best_solution):
                best_solution = candidate_sol

    if not any_fairness_window_feasible:
        return _build_failed_result(request, slice_ece, ["PAIRWISE_FAIRNESS_INFEASIBLE"])

    if best_solution is None:
        return _build_failed_result(request, slice_ece, ["GLOBAL_FP_BUDGET_EXCEEDED"])

    # Step 5: Assemble strict, verified PASSED result
    total_loss_cents, _, _, chosen_cands = best_solution
    selected_thresholds: dict[str, float] = {}
    slice_metrics: dict[str, SliceMetricOutput] = {}
    total_fp = 0
    total_fn = 0
    chosen_tprs: list[Fraction] = []

    for sl, cand in zip(ordered_slices, chosen_cands):
        selected_thresholds[sl.slice_id] = cand.threshold_float
        total_fp += cand.fp
        total_fn += cand.fn
        chosen_tprs.append(cand.tpr)
        slice_metrics[sl.slice_id] = SliceMetricOutput(
            threshold=cand.threshold_float,
            fp=cand.fp,
            fn=cand.fn,
            tp=cand.tp,
            tn=cand.tn,
            tpr=float(cand.tpr),
            fpr=float(cand.fpr),
            loss_cents=cand.loss_cents,
        )

    tpr_spread = float(max(chosen_tprs) - min(chosen_tprs))
    return PolicyOptimizationResult(
        policy_id=request.policy_id,
        evaluated_at_utc=request.evaluated_at_utc,
        audit_status="PASSED",
        selected_thresholds=selected_thresholds,
        total_loss_cents=total_loss_cents,
        total_fp=total_fp,
        total_fn=total_fn,
        global_tpr_spread=tpr_spread,
        slice_ece=slice_ece,
        slice_metrics=slice_metrics,
        violated_constraints=[],
    )
