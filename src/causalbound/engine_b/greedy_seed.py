"""Fast greedy upper-bound seeders for Engine B Branch-and-Bound pruning.

Provides:
  1. solve_greedy_1d_baseline: Naive independent 1D greedy search (used to prove
     why coupled cross-terms beta_ij > 0 defeat 1D baselines in benchmarks & UI).
  2. find_greedy_upper_bound: Multi-strategy seeder combining 1D exact sweeps,
     steepest risk-drop-per-marginal-cost ascent, and coordinate backward shaving
     to supply an aggressive initial upper bound to Priority-Queue B&B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from causalbound.schemas import MerchantFeatures
from causalbound.scm import (
    SCMModelSpec,
    apply_steps_batch,
    evaluate_cost_batch,
    evaluate_cost_scalar,
    is_better_recourse,
)


class RiskClassifier(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SeedSolution:
    steps: tuple[int, int, int, int]
    cost: float
    risk_score: float
    evaluations: int


def _predict_probs(clf: RiskClassifier, X: np.ndarray) -> np.ndarray:
    return np.asarray(clf.predict_proba(X)[:, 1], dtype=np.float64)


def solve_greedy_1d_baseline(
    base_features: MerchantFeatures,
    threshold: float,
    max_steps: tuple[int, int, int, int],
    clf: RiskClassifier,
    scm_spec: SCMModelSpec,
) -> SeedSolution | None:
    """Evaluate pure single-dimension interventions and return the cheapest valid 1D flip."""
    best_steps: tuple[int, int, int, int] | None = None
    best_cost = float("inf")
    best_prob = float("inf")
    evals = 0

    for axis in range(4):
        hi = max_steps[axis]
        if hi <= 0:
            continue
        grid = np.zeros((hi + 1, 4), dtype=np.int64)
        grid[:, axis] = np.arange(0, hi + 1, dtype=np.int64)
        X = apply_steps_batch(base_features, grid, scm_spec)
        probs = _predict_probs(clf, X)
        evals += len(probs)

        valid_indices = np.flatnonzero(probs < threshold)
        if valid_indices.size == 0:
            continue
        costs = evaluate_cost_batch(grid[valid_indices], scm_spec)
        for rel_idx in range(valid_indices.size):
            idx = int(valid_indices[rel_idx])
            cand_steps = (int(grid[idx, 0]), int(grid[idx, 1]), int(grid[idx, 2]), int(grid[idx, 3]))
            cand_cost = float(costs[rel_idx])
            if is_better_recourse(
                cand_cost, cand_steps, best_cost, best_steps, scm_spec.tie_break_cost_eps
            ):
                best_cost = cand_cost
                best_steps = cand_steps
                best_prob = float(probs[idx])

    if best_steps is None:
        return None
    return SeedSolution(steps=best_steps, cost=best_cost, risk_score=best_prob, evaluations=evals)


def _shave_coordinates(
    base_features: MerchantFeatures,
    threshold: float,
    start_steps: tuple[int, int, int, int],
    clf: RiskClassifier,
    scm_spec: SCMModelSpec,
) -> tuple[tuple[int, int, int, int], float, float, int]:
    """Coordinate-wise binary search reduction preserving f(x') < threshold."""
    cur = list(start_steps)
    evals = 0

    for _ in range(2):
        improved = False
        for axis in range(4):
            if cur[axis] == 0:
                continue
            lo, hi = 0, cur[axis]
            best_k = cur[axis]
            while lo <= hi:
                mid = (lo + hi) // 2
                probe = list(cur)
                probe[axis] = mid
                X = apply_steps_batch(base_features, np.array([probe], dtype=np.int64), scm_spec)
                prob = float(_predict_probs(clf, X)[0])
                evals += 1
                if prob < threshold:
                    best_k = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            if best_k < cur[axis]:
                cur[axis] = best_k
                improved = True
        if not improved:
            break

    final_steps = (cur[0], cur[1], cur[2], cur[3])
    X_final = apply_steps_batch(base_features, np.array([cur], dtype=np.int64), scm_spec)
    final_prob = float(_predict_probs(clf, X_final)[0])
    evals += 1
    final_cost = evaluate_cost_scalar(final_steps, scm_spec)
    return final_steps, final_cost, final_prob, evals


def _steepest_ascent_seed(
    base_features: MerchantFeatures,
    threshold: float,
    max_steps: tuple[int, int, int, int],
    clf: RiskClassifier,
    scm_spec: SCMModelSpec,
) -> SeedSolution | None:
    """Greedy risk-drop-per-marginal-cost ascent with multi-stride probes + backward shave."""
    cur = [0, 0, 0, 0]
    X0 = apply_steps_batch(base_features, np.array([cur], dtype=np.int64), scm_spec)
    cur_prob = float(_predict_probs(clf, X0)[0])
    cur_cost = 0.0
    evals = 1

    if cur_prob < threshold:
        return SeedSolution(steps=(0, 0, 0, 0), cost=0.0, risk_score=cur_prob, evaluations=evals)

    strides = (1, 4, 16, 48)
    for _ in range(80):
        proposals: list[list[int]] = []
        for axis in range(4):
            rem = max_steps[axis] - cur[axis]
            if rem <= 0:
                continue
            used_deltas: set[int] = set()
            for stride in strides:
                delta = min(stride, rem)
                if delta > 0 and delta not in used_deltas:
                    used_deltas.add(delta)
                    prop = list(cur)
                    prop[axis] += delta
                    proposals.append(prop)

        if not proposals:
            break

        prop_arr = np.array(proposals, dtype=np.int64)
        probs = _predict_probs(clf, apply_steps_batch(base_features, prop_arr, scm_spec))
        costs = evaluate_cost_batch(prop_arr, scm_spec)
        evals += len(proposals)

        # If any probe already flips below threshold, pick the cheapest flip immediately
        flipped_idx = np.flatnonzero(probs < threshold)
        if flipped_idx.size > 0:
            best_rel = int(flipped_idx[int(np.argmin(costs[flipped_idx]))])
            chosen = proposals[best_rel]
            shaved_steps, shaved_cost, shaved_prob, shave_evals = _shave_coordinates(
                base_features,
                threshold,
                (chosen[0], chosen[1], chosen[2], chosen[3]),
                clf,
                scm_spec,
            )
            return SeedSolution(
                steps=shaved_steps,
                cost=shaved_cost,
                risk_score=shaved_prob,
                evaluations=evals + shave_evals,
            )

        # Otherwise maximize risk drop per unit marginal cost
        best_score = -1.0
        best_idx = -1
        for i in range(len(proposals)):
            drop = cur_prob - float(probs[i])
            dc = float(costs[i]) - cur_cost
            if dc <= 0.0:
                continue
            ratio = drop / dc
            if ratio > best_score:
                best_score = ratio
                best_idx = i

        if best_idx < 0 or best_score <= 0.0:
            # No positive drop at current strides; jump directly toward max_steps on largest drop
            fallback_props: list[list[int]] = []
            for axis in range(4):
                if cur[axis] < max_steps[axis]:
                    prop = list(cur)
                    prop[axis] = max_steps[axis]
                    fallback_props.append(prop)
            if not fallback_props:
                break
            fb_arr = np.array(fallback_props, dtype=np.int64)
            fb_probs = _predict_probs(clf, apply_steps_batch(base_features, fb_arr, scm_spec))
            evals += len(fallback_props)
            min_i = int(np.argmin(fb_probs))
            if float(fb_probs[min_i]) >= cur_prob:
                break
            cur = fallback_props[min_i]
            cur_prob = float(fb_probs[min_i])
            cur_cost = evaluate_cost_scalar((cur[0], cur[1], cur[2], cur[3]), scm_spec)
            if cur_prob < threshold:
                shaved_steps, shaved_cost, shaved_prob, shave_evals = _shave_coordinates(
                    base_features,
                    threshold,
                    (cur[0], cur[1], cur[2], cur[3]),
                    clf,
                    scm_spec,
                )
                return SeedSolution(
                    steps=shaved_steps,
                    cost=shaved_cost,
                    risk_score=shaved_prob,
                    evaluations=evals + shave_evals,
                )
            continue

        cur = proposals[best_idx]
        cur_prob = float(probs[best_idx])
        cur_cost = float(costs[best_idx])

    return None


def find_greedy_upper_bound(
    base_features: MerchantFeatures,
    threshold: float,
    max_steps: tuple[int, int, int, int],
    clf: RiskClassifier,
    scm_spec: SCMModelSpec,
) -> SeedSolution | None:
    """Combine 1D exact sweeps, steepest ascent, and shaved max-steps for a tight UB."""
    candidates: list[SeedSolution] = []
    total_evals = 0

    one_d = solve_greedy_1d_baseline(base_features, threshold, max_steps, clf, scm_spec)
    if one_d is not None:
        candidates.append(one_d)
        total_evals += one_d.evaluations

    ascent = _steepest_ascent_seed(base_features, threshold, max_steps, clf, scm_spec)
    if ascent is not None:
        candidates.append(ascent)
        total_evals += ascent.evaluations

    # Check if max_steps itself flips; if so, shave it as a guaranteed feasible fallback seed
    X_max = apply_steps_batch(base_features, np.array([list(max_steps)], dtype=np.int64), scm_spec)
    max_prob = float(_predict_probs(clf, X_max)[0])
    total_evals += 1
    if max_prob < threshold:
        shaved_steps, shaved_cost, shaved_prob, shave_evals = _shave_coordinates(
            base_features, threshold, max_steps, clf, scm_spec
        )
        total_evals += shave_evals
        candidates.append(
            SeedSolution(
                steps=shaved_steps,
                cost=shaved_cost,
                risk_score=shaved_prob,
                evaluations=shave_evals,
            )
        )

    if not candidates:
        return None

    best = candidates[0]
    for cand in candidates[1:]:
        if is_better_recourse(
            cand.cost, cand.steps, best.cost, best.steps, scm_spec.tie_break_cost_eps
        ):
            best = cand
    return SeedSolution(
        steps=best.steps,
        cost=best.cost,
        risk_score=best.risk_score,
        evaluations=total_evals,
    )
