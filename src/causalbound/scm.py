"""Non-linear Structural Causal Model (SCM) DAG, coupled cost engine, and tie-breakers.

Implements:
  - Scalar and vectorized NumPy forward propagation for the 7-feature merchant DAG
  - DAG residual validation (<= 1e-4 cross-arch tolerance)
  - Coupled step-normalized quadratic cost function with positive cross-terms (beta_ij > 0)
  - Exact quadratic integer upper-bound solver for Branch-and-Bound domain tightening
  - Deterministic 3-tier tie-breaking comparator
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from causalbound.schemas import DagResiduals, MerchantFeatures, ensure_finite

ACTION_NAMES_SORTED: tuple[str, str, str, str] = (
    "dispute_reduction_bps",
    "kyc_tier_delta",
    "reserve_deposit_usd_delta",
    "settlement_hold_days_delta",
)

FEATURE_ORDER: tuple[str, ...] = (
    "account_tenure_months",
    "kyc_tier",
    "reserve_deposit_usd",
    "dispute_rate_bps",
    "settlement_hold_days",
    "effective_risk_ratio",
    "liquidity_coverage_index",
)

DAG_RESIDUAL_TOLERANCE: float = 1e-4


@dataclass(frozen=True, slots=True)
class ActionSpec:
    name: str
    target_feature: str
    direction: int  # +1 or -1
    min_delta: float
    max_delta: float
    step: float
    max_steps: int
    linear_weight: float
    quadratic_weight: float


@dataclass(frozen=True, slots=True)
class SCMModelSpec:
    """Immutable parsed representation of data/scm_config.json."""

    actions: tuple[ActionSpec, ...]  # Strictly ordered by ACTION_NAMES_SORTED
    cross_matrix: tuple[tuple[float, ...], ...]  # Symmetric 4x4 matrix of beta_ij (0 on diag)
    dag_residual_tol: float
    optimality_tol: float
    tie_break_cost_eps: float
    candidates_per_instance: int

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> SCMModelSpec:
        raw_actions = cfg["action_specs"]
        actions: list[ActionSpec] = []
        for name in ACTION_NAMES_SORTED:
            a = raw_actions[name]
            step = float(a["step"])
            max_delta = float(a["max_delta"])
            min_delta = float(a["min_delta"])
            max_steps = int(round((max_delta - min_delta) / step))
            actions.append(
                ActionSpec(
                    name=name,
                    target_feature=str(a["target_feature"]),
                    direction=int(a["direction"]),
                    min_delta=min_delta,
                    max_delta=max_delta,
                    step=step,
                    max_steps=max_steps,
                    linear_weight=float(a["linear_weight"]),
                    quadratic_weight=float(a["quadratic_weight"]),
                )
            )

        n = len(actions)
        matrix = [[0.0] * n for _ in range(n)]
        raw_cross = cfg["cross_weights"]
        for i in range(n):
            for j in range(i + 1, n):
                pair_key = f"{ACTION_NAMES_SORTED[i]}__{ACTION_NAMES_SORTED[j]}"
                rev_key = f"{ACTION_NAMES_SORTED[j]}__{ACTION_NAMES_SORTED[i]}"
                if pair_key in raw_cross:
                    beta = float(raw_cross[pair_key])
                elif rev_key in raw_cross:
                    beta = float(raw_cross[rev_key])
                else:
                    raise KeyError(f"Missing cross-weight for pair {pair_key}")
                if beta <= 0.0 or not math.isfinite(beta):
                    raise ValueError(f"Cross-weight {pair_key} must be finite and > 0, got {beta}")
                matrix[i][j] = beta
                matrix[j][i] = beta

        tols = cfg["tolerances"]
        return cls(
            actions=tuple(actions),
            cross_matrix=tuple(tuple(row) for row in matrix),
            dag_residual_tol=float(tols["dag_residual_abs"]),
            optimality_tol=float(tols["optimality_one_sided"]),
            tie_break_cost_eps=float(tols["tie_break_cost_eps"]),
            candidates_per_instance=int(cfg["candidates_per_instance"]),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> SCMModelSpec:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def compute_derived_scalar(
    account_tenure_months: float,
    kyc_tier: float,
    reserve_deposit_usd: float,
    dispute_rate_bps: float,
    settlement_hold_days: float,
) -> tuple[float, float]:
    """Evaluate non-linear SCM structural equations for a single merchant."""
    if dispute_rate_bps < 0.0 or settlement_hold_days < 0.0 or reserve_deposit_usd < 0.0:
        raise ValueError("Root features must be non-negative before SCM propagation")

    effective_risk_ratio = math.exp(0.0025 * dispute_rate_bps) / (
        1.0
        + 0.35 * kyc_tier
        + 0.15 * math.log1p(account_tenure_months)
        + 0.08 * math.sqrt(settlement_hold_days)
    )
    liquidity_coverage_index = (
        math.log1p(reserve_deposit_usd)
        + 0.6 * math.sqrt(settlement_hold_days + 1.0)
        + 0.4 * kyc_tier
        - 0.12 * math.sqrt(dispute_rate_bps)
    )
    if not (math.isfinite(effective_risk_ratio) and math.isfinite(liquidity_coverage_index)):
        raise ValueError("Non-finite derived SCM node produced")
    return effective_risk_ratio, liquidity_coverage_index


# Backward compatibility alias
compute_derived_scalars = compute_derived_scalar


def compute_derived_vectorized(
    account_tenure_months: np.ndarray | float,
    kyc_tier: np.ndarray | float,
    reserve_deposit_usd: np.ndarray | float,
    dispute_rate_bps: np.ndarray | float,
    settlement_hold_days: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized NumPy SCM forward propagation for leaf-block batch evaluation."""
    eff_risk = np.exp(0.0025 * dispute_rate_bps) / (
        1.0
        + 0.35 * kyc_tier
        + 0.15 * np.log1p(account_tenure_months)
        + 0.08 * np.sqrt(settlement_hold_days)
    )
    liq_idx = (
        np.log1p(reserve_deposit_usd)
        + 0.6 * np.sqrt(settlement_hold_days + 1.0)
        + 0.4 * kyc_tier
        - 0.12 * np.sqrt(dispute_rate_bps)
    )
    return eff_risk, liq_idx


def compute_instance_max_steps(
    features: MerchantFeatures,
    scm_spec: SCMModelSpec,
) -> tuple[int, int, int, int]:
    """Compute valid integer upper step bounds k_i^max respecting non-negative root domains.

    In particular, dispute_reduction_bps cannot reduce dispute_rate_bps below 0.
    """
    max_steps: list[int] = []
    for action in scm_spec.actions:
        if action.direction == -1:
            current_val = float(getattr(features, action.target_feature))
            domain_cap = int(math.floor((current_val + 1e-12) / action.step))
            max_steps.append(max(0, min(action.max_steps, domain_cap)))
        else:
            max_steps.append(action.max_steps)
    return (max_steps[0], max_steps[1], max_steps[2], max_steps[3])


def features_to_array(features: MerchantFeatures) -> np.ndarray:
    """Convert MerchantFeatures into a (1, 7) float64 feature matrix in locked FEATURE_ORDER."""
    return np.array(
        [
            [
                float(features.account_tenure_months),
                float(features.kyc_tier),
                float(features.reserve_deposit_usd),
                float(features.dispute_rate_bps),
                float(features.settlement_hold_days),
                float(features.effective_risk_ratio),
                float(features.liquidity_coverage_index),
            ]
        ],
        dtype=np.float64,
    )


def apply_steps_scalar(
    base: MerchantFeatures,
    steps: tuple[int, int, int, int],
    scm_spec: SCMModelSpec,
) -> tuple[MerchantFeatures, np.ndarray]:
    """Apply discrete step counts (k_dispute, k_kyc, k_reserve, k_hold) and propagate DAG."""
    k_disp, k_kyc, k_res, k_hold = steps
    a_disp, a_kyc, a_res, a_hold = scm_spec.actions

    tenure = float(base.account_tenure_months)  # Immutable root
    kyc = int(base.kyc_tier + k_kyc * int(round(a_kyc.step)))
    reserve = float(base.reserve_deposit_usd + k_res * a_res.step)
    dispute = float(base.dispute_rate_bps - k_disp * a_disp.step)
    hold = int(base.settlement_hold_days + k_hold * int(round(a_hold.step)))

    eff_risk, liq_idx = compute_derived_scalar(tenure, float(kyc), reserve, dispute, float(hold))
    cf = MerchantFeatures(
        account_tenure_months=tenure,
        kyc_tier=kyc,
        reserve_deposit_usd=reserve,
        dispute_rate_bps=dispute,
        settlement_hold_days=hold,
        effective_risk_ratio=eff_risk,
        liquidity_coverage_index=liq_idx,
    )
    x_row = np.array(
        [[tenure, float(kyc), reserve, dispute, float(hold), eff_risk, liq_idx]],
        dtype=np.float64,
    )
    return cf, x_row


def apply_steps_batch(
    base: MerchantFeatures,
    step_matrix: np.ndarray,
    scm_spec: SCMModelSpec,
) -> np.ndarray:
    """Vectorized SCM propagation for a batch of step vectors shaped (M, 4)."""
    n_rows = step_matrix.shape[0]
    a_disp, a_kyc, a_res, a_hold = scm_spec.actions

    tenure = np.full(n_rows, float(base.account_tenure_months), dtype=np.float64)
    kyc = float(base.kyc_tier) + step_matrix[:, 1] * a_kyc.step
    reserve = float(base.reserve_deposit_usd) + step_matrix[:, 2] * a_res.step
    dispute = float(base.dispute_rate_bps) - step_matrix[:, 0] * a_disp.step
    hold = float(base.settlement_hold_days) + step_matrix[:, 3] * a_hold.step

    eff_risk, liq_idx = compute_derived_vectorized(tenure, kyc, reserve, dispute, hold)
    return np.column_stack([tenure, kyc, reserve, dispute, hold, eff_risk, liq_idx])


def compute_dag_residuals(cf: MerchantFeatures) -> dict[str, float]:
    """Verify non-linear SCM structural equations against reported derived features."""
    expected_eff, expected_liq = compute_derived_scalar(
        cf.account_tenure_months,
        float(cf.kyc_tier),
        cf.reserve_deposit_usd,
        cf.dispute_rate_bps,
        float(cf.settlement_hold_days),
    )
    return {
        "effective_risk_ratio": abs(cf.effective_risk_ratio - expected_eff),
        "liquidity_coverage_index": abs(cf.liquidity_coverage_index - expected_liq),
    }


def verify_dag_residuals(
    features: MerchantFeatures | dict[str, Any],
    tol: float = DAG_RESIDUAL_TOLERANCE,
) -> tuple[bool, DagResiduals]:
    """Verify reported derived features match DAG predictions within tolerance."""
    if isinstance(features, MerchantFeatures):
        feat_dict = features.model_dump()
    else:
        feat_dict = features

    tenure = ensure_finite(feat_dict["account_tenure_months"], "account_tenure_months")
    kyc = ensure_finite(feat_dict["kyc_tier"], "kyc_tier")
    reserve = ensure_finite(feat_dict["reserve_deposit_usd"], "reserve_deposit_usd")
    dispute = ensure_finite(feat_dict["dispute_rate_bps"], "dispute_rate_bps")
    hold = ensure_finite(feat_dict["settlement_hold_days"], "settlement_hold_days")

    actual_eff = ensure_finite(feat_dict["effective_risk_ratio"], "effective_risk_ratio")
    actual_liq = ensure_finite(feat_dict["liquidity_coverage_index"], "liquidity_coverage_index")

    expected_eff, expected_liq = compute_derived_scalar(tenure, kyc, reserve, dispute, hold)

    res_eff = abs(actual_eff - expected_eff)
    res_liq = abs(actual_liq - expected_liq)

    is_valid = (res_eff <= tol) and (res_liq <= tol)
    residuals = DagResiduals(
        effective_risk_ratio=round(res_eff, 8),
        liquidity_coverage_index=round(res_liq, 8),
    )
    return is_valid, residuals


def propagate_scm(
    initial_features: MerchantFeatures | dict[str, Any],
    kyc_tier_delta: int = 0,
    reserve_deposit_usd_delta: float = 0.0,
    dispute_reduction_bps: float = 0.0,
    settlement_hold_days_delta: int = 0,
) -> MerchantFeatures:
    """Apply intervention to actionable roots and propagate forward through the DAG."""
    if isinstance(initial_features, MerchantFeatures):
        f = initial_features.model_dump()
    else:
        f = initial_features

    tenure = float(f["account_tenure_months"])
    new_kyc = int(f["kyc_tier"]) + int(kyc_tier_delta)
    new_reserve = float(f["reserve_deposit_usd"]) + float(reserve_deposit_usd_delta)
    new_dispute = max(0.0, float(f["dispute_rate_bps"]) - float(dispute_reduction_bps))
    new_hold = int(f["settlement_hold_days"]) + int(settlement_hold_days_delta)

    new_eff, new_liq = compute_derived_scalar(
        tenure, float(new_kyc), new_reserve, new_dispute, float(new_hold)
    )

    return MerchantFeatures(
        account_tenure_months=tenure,
        kyc_tier=new_kyc,
        reserve_deposit_usd=new_reserve,
        dispute_rate_bps=new_dispute,
        settlement_hold_days=new_hold,
        effective_risk_ratio=new_eff,
        liquidity_coverage_index=new_liq,
    )


def evaluate_cost_scalar(steps: tuple[int, int, int, int], scm_spec: SCMModelSpec) -> float:
    """Compute exact step-normalized coupled quadratic cost for 4 action dimensions."""
    total = 0.0
    actions = scm_spec.actions
    cross = scm_spec.cross_matrix
    for i in range(4):
        ki = steps[i]
        if ki:
            total += actions[i].linear_weight * ki + actions[i].quadratic_weight * (ki * ki)
    for i in range(4):
        ki = steps[i]
        if not ki:
            continue
        row = cross[i]
        for j in range(i + 1, 4):
            kj = steps[j]
            if kj:
                total += row[j] * (ki * kj)
    return total


def evaluate_cost_batch(step_matrix: np.ndarray, scm_spec: SCMModelSpec) -> np.ndarray:
    """Vectorized coupled quadratic cost for step_matrix shaped (M, 4)."""
    k = step_matrix.astype(np.float64, copy=False)
    w = np.array([a.linear_weight for a in scm_spec.actions], dtype=np.float64)
    alpha = np.array([a.quadratic_weight for a in scm_spec.actions], dtype=np.float64)
    costs = k @ w + (k * k) @ alpha

    cross = scm_spec.cross_matrix
    for i in range(4):
        for j in range(i + 1, 4):
            beta = cross[i][j]
            if beta != 0.0:
                costs = costs + beta * k[:, i] * k[:, j]
    return costs


def incremental_cost_for_var(
    var_idx: int,
    k_val: int,
    assigned: tuple[int | None, int | None, int | None, int | None],
    scm_spec: SCMModelSpec,
) -> float:
    """Exact marginal cost added when fixing var_idx = k_val given already-assigned vars."""
    if k_val == 0:
        return 0.0
    action = scm_spec.actions[var_idx]
    cross_row = scm_spec.cross_matrix[var_idx]
    linear_eff = action.linear_weight
    for u in range(4):
        ku = assigned[u]
        if ku is not None and ku != 0:
            linear_eff += cross_row[u] * ku
    return linear_eff * k_val + action.quadratic_weight * (k_val * k_val)


def max_cost_feasible_step(
    var_idx: int,
    assigned: tuple[int | None, int | None, int | None, int | None],
    current_cost: float,
    cost_budget: float,
    domain_max_k: int,
    scm_spec: SCMModelSpec,
) -> int:
    """Tightest integer upper bound k_hi in [0, domain_max_k] such that delta_cost <= budget.

    Because w_v > 0, alpha_v > 0, beta_{uv} > 0, marginal cost is strictly increasing in k:
      alpha_v * k^2 + b_eff * k <= remaining_budget
    """
    if not math.isfinite(cost_budget):
        return domain_max_k
    remaining = cost_budget - current_cost
    if remaining < 0.0:
        return -1

    action = scm_spec.actions[var_idx]
    cross_row = scm_spec.cross_matrix[var_idx]
    b_eff = action.linear_weight
    for u in range(4):
        ku = assigned[u]
        if ku is not None and ku != 0:
            b_eff += cross_row[u] * ku

    a = action.quadratic_weight
    # Solve a*k^2 + b_eff*k - remaining <= 0 for k >= 0
    discriminant = b_eff * b_eff + 4.0 * a * remaining
    k_real = (-b_eff + math.sqrt(max(0.0, discriminant))) / (2.0 * a)
    k_hi = min(domain_max_k, int(math.floor(k_real + 1e-12)))

    # Guard against floating-point overshoot by at most 1 integer step
    while k_hi >= 0 and (b_eff * k_hi + a * k_hi * k_hi) > remaining + 1e-12:
        k_hi -= 1
    return k_hi


def is_better_recourse(
    cand_cost: float,
    cand_steps: tuple[int, int, int, int],
    best_cost: float,
    best_steps: tuple[int, int, int, int] | None,
    tie_eps: float,
) -> bool:
    """Locked 3-tier deterministic tie-breaking comparator for Engine B.

    Tier 1: Minimum coupled total_cost (tied if |c1 - c2| <= tie_eps)
    Tier 2: Minimum integer step sum sum(k_i)
    Tier 3: Lexicographically smallest step tuple sorted by ascending action name
    """
    if best_steps is None:
        return True
    diff = cand_cost - best_cost
    if diff < -tie_eps:
        return True
    if diff > tie_eps:
        return False
    cand_sum = sum(cand_steps)
    best_sum = sum(best_steps)
    if cand_sum != best_sum:
        return cand_sum < best_sum
    return cand_steps < best_steps


def steps_to_action_dicts(
    steps: tuple[int, int, int, int],
    scm_spec: SCMModelSpec,
) -> tuple[dict[str, int | float], dict[str, int]]:
    """Build strict sorted actions and step_counts dicts matching ENGINE_B_ACTION_KEYS."""
    actions_dict: dict[str, int | float] = {}
    steps_dict: dict[str, int] = {}
    for idx, action in enumerate(scm_spec.actions):
        k = int(steps[idx])
        steps_dict[action.name] = k
        raw_delta = k * action.step
        # Preserve integer representation for exact-integer step specs
        actions_dict[action.name] = int(round(raw_delta)) if action.step.is_integer() else raw_delta
    return actions_dict, steps_dict
