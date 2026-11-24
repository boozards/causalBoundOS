"""Core Engine B: Exact Priority-Queue Branch-and-Bound Causal Recourse Solver.

Finds the globally optimal, causally valid discrete intervention Delta_x flipping
f(x') < threshold across the non-linear SCM DAG and ~2.20e7 candidate space.

Key Optimizations (100% Exact — Zero Optimality Loss):
  1. Greedy Upper-Bound Seeding: Rapidly tightens best_cost before B&B starts.
  2. Quadratic Domain Tightening: Strictly positive (w_i, alpha_i, beta_ij) yields
     an exact closed-form integer upper bound k_u^hi per unassigned variable.
  3. Monotonic Optimistic Feasibility Pruning: For monotonic classifiers aligned to
     the SCM DAG, evaluating (assigned, k_u^hi for unassigned) gives the lowest
     achievable risk score in the entire subtree. If >= threshold, prune immediately.
  4. Minimum Remaining Values (MRV): Branches on the unassigned variable with the
     smallest tightened feasible domain |0..k_u^hi| first.
  5. @lru_cache Memoization: Caches classifier evaluations on immutable 4-tuples.
  6. Vectorized NumPy Leaf Evaluation: When 1 variable remains, evaluates all
     feasible steps [0..k_v^hi] in a single batched predict_proba + cost call.
  7. Strict 3-Tier Deterministic Tie-Breaking Ladder.
"""

from __future__ import annotations

import heapq
import math
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from causalbound.engine_b.greedy_seed import RiskClassifier, find_greedy_upper_bound
from causalbound.schemas import DagResiduals, MerchantRecourseCase, RecourseResult
from causalbound.scm import (
    SCMModelSpec,
    apply_steps_batch,
    apply_steps_scalar,
    compute_dag_residuals,
    compute_instance_max_steps,
    evaluate_cost_batch,
    evaluate_cost_scalar,
    features_to_array,
    incremental_cost_for_var,
    is_better_recourse,
    max_cost_feasible_step,
    steps_to_action_dicts,
)

_PartialAssignment = tuple[int | None, int | None, int | None, int | None]


class _FastHistGradientBoostingWrapper:
    """Zero-overhead predict_proba fast-path for fitted HistGradientBoostingClassifier."""

    __slots__ = ("_predictors", "_predict_iters", "_dtype", "_base_val")

    def __init__(self, clf: Any) -> None:
        self._predictors = clf._predictors
        self._predict_iters = clf._predict_iterations
        self._dtype = clf._baseline_prediction.dtype
        self._base_val = float(clf._baseline_prediction.item(0))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not X.flags["C_CONTIGUOUS"] or X.dtype != np.float64:
            X = np.ascontiguousarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        raw = np.empty((n_samples, 1), dtype=self._dtype, order="F")
        raw.fill(self._base_val)
        self._predict_iters(X, self._predictors, raw, False, 1)
        p1 = 1.0 / (1.0 + np.exp(-raw[:, 0]))
        return np.column_stack([1.0 - p1, p1])


def _wrap_fast_classifier(clf: RiskClassifier) -> RiskClassifier:
    if (
        hasattr(clf, "_predict_iterations")
        and hasattr(clf, "_predictors")
        and hasattr(clf, "_baseline_prediction")
    ):
        return _FastHistGradientBoostingWrapper(clf)
    return clf


def _build_result_from_steps(
    case: MerchantRecourseCase,
    status: str,
    init_prob: float,
    steps: tuple[int, int, int, int],
    cf_prob: float,
    total_cost: float,
    states_visited: int,
    scm_spec: SCMModelSpec,
) -> RecourseResult:
    cf_features, _ = apply_steps_scalar(case.features, steps, scm_spec)
    residuals = compute_dag_residuals(cf_features)
    if (
        residuals["effective_risk_ratio"] > scm_spec.dag_residual_tol
        or residuals["liquidity_coverage_index"] > scm_spec.dag_residual_tol
    ):
        raise RuntimeError(f"SCM DAG residual exceeded tolerance: {residuals}")

    actions_dict, steps_dict = steps_to_action_dicts(steps, scm_spec)
    return RecourseResult(
        merchant_id=case.merchant_id,
        slice_id=case.slice_id,
        audit_status=status,  # type: ignore[arg-type]
        threshold=case.threshold,
        initial_risk_score=init_prob,
        counterfactual_risk_score=cf_prob,
        actions=actions_dict,
        step_counts=steps_dict,
        counterfactual_features=cf_features,
        total_cost=total_cost,
        dag_residuals=DagResiduals(**residuals),
        states_visited=states_visited,
        violated_constraints=[],
    )


def _build_infeasible_result(
    case: MerchantRecourseCase,
    init_prob: float,
    states_visited: int,
) -> RecourseResult:
    return RecourseResult(
        merchant_id=case.merchant_id,
        slice_id=case.slice_id,
        audit_status="FAILED_AUDIT",
        threshold=case.threshold,
        initial_risk_score=init_prob,
        counterfactual_risk_score=None,
        actions=None,
        step_counts=None,
        counterfactual_features=None,
        total_cost=None,
        dag_residuals=None,
        states_visited=states_visited,
        violated_constraints=["NO_FEASIBLE_FLIP"],
    )


def solve_recourse(
    case: MerchantRecourseCase | dict[str, Any],
    clf: RiskClassifier | None = None,
    scm_spec: SCMModelSpec | None = None,
    *,
    assume_monotonic: bool = True,
) -> RecourseResult:
    """Solve non-linear causal recourse to exact global optimality via Priority-Queue B&B."""
    if isinstance(case, dict):
        case = MerchantRecourseCase.model_validate(case)

    if clf is None or scm_spec is None:
        repo_root = Path(__file__).resolve().parents[3]
        if clf is None:
            clf = joblib.load(repo_root / "data" / "model.joblib")
        if scm_spec is None:
            scm_spec = SCMModelSpec.from_file(repo_root / "data" / "scm_config.json")

    clf = _wrap_fast_classifier(clf)

    base = case.features
    threshold = case.threshold
    tie_eps = scm_spec.tie_break_cost_eps

    # Step 0: Evaluate initial merchant score
    x_init = features_to_array(base)
    init_prob = float(clf.predict_proba(x_init)[0, 1])
    if not math.isfinite(init_prob):
        raise ValueError(f"Non-finite initial risk score: {init_prob}")

    if init_prob < threshold:
        return _build_result_from_steps(
            case=case,
            status="ALREADY_APPROVED",
            init_prob=init_prob,
            steps=(0, 0, 0, 0),
            cf_prob=init_prob,
            total_cost=0.0,
            states_visited=1,
            scm_spec=scm_spec,
        )

    max_steps = compute_instance_max_steps(base, scm_spec)

    @lru_cache(maxsize=100_000)
    def eval_prob_cached(steps: tuple[int, int, int, int]) -> float:
        _, x_row = apply_steps_scalar(base, steps, scm_spec)
        return float(clf.predict_proba(x_row)[0, 1])

    # Step 1: O(1) Global infeasibility check for monotonic classifiers
    if assume_monotonic:
        opt_max_prob = eval_prob_cached(max_steps)
        if opt_max_prob >= threshold:
            return _build_infeasible_result(case, init_prob, states_visited=1)

    # Step 2: Greedy upper-bound seeding
    seed = find_greedy_upper_bound(base, threshold, max_steps, clf, scm_spec)
    best_steps: tuple[int, int, int, int] | None = seed.steps if seed is not None else None
    best_cost: float = seed.cost if seed is not None else float("inf")
    best_prob: float = seed.risk_score if seed is not None else float("inf")
    # Track minimum seen cost separately so tie_eps pruning never loses a tied competitor
    min_feasible_cost: float = best_cost

    # Step 3: Priority-Queue Branch-and-Bound
    # Heap item: (lower_bound, num_assigned_neg, tie_counter, assigned_tuple)
    # Using exact partial cost as admissible lower bound (all w, alpha, beta > 0)
    pq: list[tuple[float, int, int, _PartialAssignment]] = []
    counter = 0
    initial_assigned: _PartialAssignment = (None, None, None, None)
    heapq.heappush(pq, (0.0, 0, counter, initial_assigned))

    states_visited = 0

    while pq:
        lb, _, _, assigned = heapq.heappop(pq)

        # Admissible lower-bound pruning (respecting Tier-1 tie tolerance)
        if best_steps is not None and lb > min_feasible_cost + tie_eps:
            break  # Priority queue yields states in non-decreasing lb order

        states_visited += 1
        unassigned = [i for i in range(4) if assigned[i] is None]
        cost_budget = min_feasible_cost + tie_eps if best_steps is not None else float("inf")

        # Compute exact cost-tightened upper bounds for every unassigned variable
        upper_bounds: dict[int, int] = {}
        empty_domain = False
        for var_idx in unassigned:
            k_hi = max_cost_feasible_step(
                var_idx=var_idx,
                assigned=assigned,
                current_cost=lb,
                cost_budget=cost_budget,
                domain_max_k=max_steps[var_idx],
                scm_spec=scm_spec,
            )
            if k_hi < 0:
                empty_domain = True
                break
            upper_bounds[var_idx] = k_hi

        if empty_domain:
            continue

        # Optimistic feasibility pruning for monotonic classifiers
        if assume_monotonic:
            optimistic_steps = (
                assigned[0] if assigned[0] is not None else upper_bounds[0],
                assigned[1] if assigned[1] is not None else upper_bounds[1],
                assigned[2] if assigned[2] is not None else upper_bounds[2],
                assigned[3] if assigned[3] is not None else upper_bounds[3],
            )
            if eval_prob_cached(optimistic_steps) >= threshold:
                continue

        # Base / Leaf Case: Exactly 1 unassigned variable -> vectorized NumPy batch sweep
        if len(unassigned) == 1:
            leaf_var = unassigned[0]
            k_hi = upper_bounds[leaf_var]
            n_leaf = k_hi + 1

            step_matrix = np.empty((n_leaf, 4), dtype=np.int64)
            for idx in range(4):
                if idx == leaf_var:
                    step_matrix[:, idx] = np.arange(0, n_leaf, dtype=np.int64)
                else:
                    step_matrix[:, idx] = int(assigned[idx])  # type: ignore[arg-type]

            X_leaf = apply_steps_batch(base, step_matrix, scm_spec)
            probs = np.asarray(clf.predict_proba(X_leaf)[:, 1], dtype=np.float64)
            valid_indices = np.flatnonzero(probs < threshold)
            if valid_indices.size == 0:
                continue

            valid_steps = step_matrix[valid_indices]
            costs = evaluate_cost_batch(valid_steps, scm_spec)

            # Because cost is strictly increasing in k_leaf, the earliest valid index
            # minimizes both Tier 1 cost and Tier 2 sum(k) for monotonic classifiers.
            # We still inspect all valid indices within tie_eps for full generality.
            for rel_i in range(valid_indices.size):
                cand_cost = float(costs[rel_i])
                if best_steps is not None and cand_cost > min_feasible_cost + tie_eps:
                    break
                row = valid_steps[rel_i]
                cand_steps = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
                if is_better_recourse(cand_cost, cand_steps, best_cost, best_steps, tie_eps):
                    best_cost = cand_cost
                    best_steps = cand_steps
                    best_prob = float(probs[int(valid_indices[rel_i])])
                    if cand_cost < min_feasible_cost:
                        min_feasible_cost = cand_cost
            continue

        # MRV Heuristic: choose unassigned variable with smallest tightened domain |0..k_hi|
        # Deterministic tie-break by ascending var_idx (already sorted by action name)
        target_var = min(unassigned, key=lambda v: (upper_bounds[v] + 1, v))
        target_hi = upper_bounds[target_var]

        assigned_list = list(assigned)
        next_depth_neg = -(4 - len(unassigned) + 1)

        for k_val in range(0, target_hi + 1):
            delta_c = incremental_cost_for_var(target_var, k_val, assigned, scm_spec)
            new_lb = lb + delta_c
            if best_steps is not None and new_lb > min_feasible_cost + tie_eps:
                break  # Incremental cost is strictly increasing in k_val

            assigned_list[target_var] = k_val
            child_assigned: _PartialAssignment = (
                assigned_list[0],
                assigned_list[1],
                assigned_list[2],
                assigned_list[3],
            )
            counter += 1
            heapq.heappush(pq, (new_lb, next_depth_neg, counter, child_assigned))

    if best_steps is None:
        return _build_infeasible_result(case, init_prob, states_visited=states_visited)

    return _build_result_from_steps(
        case=case,
        status="PASSED",
        init_prob=init_prob,
        steps=best_steps,
        cf_prob=best_prob,
        total_cost=best_cost,
        states_visited=states_visited,
        scm_spec=scm_spec,
    )


def solve_recourse_brute_force(
    case: MerchantRecourseCase,
    clf: RiskClassifier,
    scm_spec: SCMModelSpec,
    max_candidates_guard: int = 50_000,
) -> RecourseResult:
    """Reference brute-force oracle strictly for small-scale (< 5e4) parity verification."""
    clf = _wrap_fast_classifier(clf)
    base = case.features
    threshold = case.threshold
    tie_eps = scm_spec.tie_break_cost_eps

    init_prob = float(clf.predict_proba(features_to_array(base))[0, 1])
    if init_prob < threshold:
        return _build_result_from_steps(
            case=case,
            status="ALREADY_APPROVED",
            init_prob=init_prob,
            steps=(0, 0, 0, 0),
            cf_prob=init_prob,
            total_cost=0.0,
            states_visited=1,
            scm_spec=scm_spec,
        )

    max_steps = compute_instance_max_steps(base, scm_spec)
    total_space = math.prod(m + 1 for m in max_steps)
    if total_space > max_candidates_guard:
        raise ValueError(
            f"Brute-force oracle refused space of {total_space:,} > guard {max_candidates_guard:,}"
        )

    all_steps = np.array(
        list(
            product(
                range(max_steps[0] + 1),
                range(max_steps[1] + 1),
                range(max_steps[2] + 1),
                range(max_steps[3] + 1),
            )
        ),
        dtype=np.int64,
    )
    X_all = apply_steps_batch(base, all_steps, scm_spec)
    probs = np.asarray(clf.predict_proba(X_all)[:, 1], dtype=np.float64)
    valid_idx = np.flatnonzero(probs < threshold)
    if valid_idx.size == 0:
        return _build_infeasible_result(case, init_prob, states_visited=int(total_space))

    valid_steps = all_steps[valid_idx]
    costs = evaluate_cost_batch(valid_steps, scm_spec)

    best_steps: tuple[int, int, int, int] | None = None
    best_cost = float("inf")
    best_prob = float("inf")

    for i in range(valid_idx.size):
        cand_cost = float(costs[i])
        row = valid_steps[i]
        cand_steps = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        if is_better_recourse(cand_cost, cand_steps, best_cost, best_steps, tie_eps):
            best_cost = cand_cost
            best_steps = cand_steps
            best_prob = float(probs[int(valid_idx[i])])

    assert best_steps is not None
    return _build_result_from_steps(
        case=case,
        status="PASSED",
        init_prob=init_prob,
        steps=best_steps,
        cf_prob=best_prob,
        total_cost=best_cost,
        states_visited=int(total_space),
        scm_spec=scm_spec,
    )


class RecourseSolver:
    """Object-oriented wrapper preserving backward compatibility with eval harnesses."""

    def __init__(
        self,
        model: Any | None = None,
        scm_config: dict[str, Any] | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        if model is None:
            model_path = repo_root / "data" / "model.joblib"
            self.model = joblib.load(model_path)
        else:
            self.model = model

        if scm_config is None:
            scm_path = repo_root / "data" / "scm_config.json"
            self.scm_spec = SCMModelSpec.from_file(scm_path)
        else:
            self.scm_spec = SCMModelSpec.from_dict(scm_config)

    def solve(self, case: MerchantRecourseCase | dict[str, Any]) -> RecourseResult:
        if isinstance(case, dict):
            case = MerchantRecourseCase.model_validate(case)
        return solve_recourse(case, self.model, self.scm_spec)


def compute_step_cost(
    k0: int,
    k1: int,
    k2: int,
    k3: int,
    scm_spec: SCMModelSpec | None = None,
) -> float:
    """Evaluate coupled recourse cost for 4 step integers."""
    if scm_spec is None:
        repo_root = Path(__file__).resolve().parents[3]
        scm_spec = SCMModelSpec.from_file(repo_root / "data" / "scm_config.json")
    return evaluate_cost_scalar((int(k0), int(k1), int(k2), int(k3)), scm_spec)
