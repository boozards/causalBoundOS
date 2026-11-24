#!/usr/bin/env python3
"""Single-source-of-truth deterministic asset generator for CausalBound.

Generates and synchronizes in lockstep (seed=42):
  1. data/scm_config.json        — Locked SCM DAG, bounds, steps, coupled costs, tolerances
  2. data/model.joblib           — Monotonic HistGradientBoostingClassifier (sklearn==1.6.1)
  3. data/slices_dataset.json    — 16-slice policy dataset verified for Audit A (Anti-Slack)
  4. data/merchant_cases.json    — 30 blocked merchants spanning ~2.20e7 candidates each

Also enforces:
  - 1-ULP Mid-Boundary Rule (timestamps at .500 UTC; scores >= 0.005 away from 0.05 grid)
  - Exact serialization version metadata capture
  - Lockstep Anti-Slack verification (removing max_fp, max_fpr, or delta changes outcome)
"""

from __future__ import annotations

import json
import math
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

# Allow direct script execution without editable install
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from causalbound.engine_a.slice_optimizer import optimize_policy  # noqa: E402
from causalbound.schemas import PolicyOptimizationRequest  # noqa: E402

SEED = 42
REGIONS = ("NA", "EU", "APAC", "LATAM")
TIERS = ("micro", "smb", "mid_market", "enterprise")
FEATURE_ORDER = (
    "account_tenure_months",
    "kyc_tier",
    "reserve_deposit_usd",
    "dispute_rate_bps",
    "settlement_hold_days",
    "effective_risk_ratio",
    "liquidity_coverage_index",
)
MONOTONIC_CST = (-1, -1, -1, 1, -1, 1, -1)
CANDIDATE_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95


def format_utc_mid_second(base_dt: datetime, offset_seconds: int) -> str:
    """Format strict UTC ISO-8601 timestamp centered at .500ms (1-ULP safe)."""
    dt = (base_dt + timedelta(seconds=offset_seconds)).astimezone(timezone.utc)
    ms = round(dt.microsecond / 1000.0)
    if ms == 1000:
        dt = dt.replace(microsecond=0) + timedelta(seconds=1)
        ms = 0
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"


def offset_score_from_grid(score: float, min_margin: float = 0.006) -> float:
    """Clamp score into [0.006, 0.994] and push >= min_margin away from 0.05 multiples."""
    clamped = min(max(float(score), min_margin), 1.0 - min_margin)
    nearest = round(clamped / 0.05) * 0.05
    diff = clamped - nearest
    if abs(diff) < min_margin:
        direction = 1.0 if diff >= 0.0 else -1.0
        clamped = nearest + direction * min_margin
        clamped = min(max(clamped, min_margin), 1.0 - min_margin)
    return round(clamped, 6)


def compute_derived_nodes(
    account_tenure_months: np.ndarray,
    kyc_tier: np.ndarray,
    reserve_deposit_usd: np.ndarray,
    dispute_rate_bps: np.ndarray,
    settlement_hold_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-linear SCM DAG forward equations (vectorized)."""
    effective_risk_ratio = np.exp(0.0025 * dispute_rate_bps) / (
        1.0
        + 0.35 * kyc_tier
        + 0.15 * np.log1p(account_tenure_months)
        + 0.08 * np.sqrt(settlement_hold_days)
    )
    liquidity_coverage_index = (
        np.log1p(reserve_deposit_usd)
        + 0.6 * np.sqrt(settlement_hold_days + 1.0)
        + 0.4 * kyc_tier
        - 0.12 * np.sqrt(dispute_rate_bps)
    )
    return effective_risk_ratio, liquidity_coverage_index


def build_scm_config() -> dict:
    """Build locked SCM DAG configuration and coupled cost matrix."""
    action_specs = {
        "dispute_reduction_bps": {
            "target_feature": "dispute_rate_bps",
            "direction": -1,
            "min_delta": 0,
            "max_delta": 500,
            "step": 5,
            "num_steps": 101,
            "linear_weight": 1.25,
            "quadratic_weight": 0.04,
        },
        "kyc_tier_delta": {
            "target_feature": "kyc_tier",
            "direction": 1,
            "min_delta": 0,
            "max_delta": 2,
            "step": 1,
            "num_steps": 3,
            "linear_weight": 4.50,
            "quadratic_weight": 1.20,
        },
        "reserve_deposit_usd_delta": {
            "target_feature": "reserve_deposit_usd",
            "direction": 1,
            "min_delta": 0,
            "max_delta": 100000,
            "step": 250,
            "num_steps": 401,
            "linear_weight": 1.00,
            "quadratic_weight": 0.02,
        },
        "settlement_hold_days_delta": {
            "target_feature": "settlement_hold_days",
            "direction": 1,
            "min_delta": 0,
            "max_delta": 180,
            "step": 1,
            "num_steps": 181,
            "linear_weight": 1.15,
            "quadratic_weight": 0.03,
        },
    }

    # Strictly positive cross-term interaction weights beta_ij (defeats 1D greedy search)
    cross_weights = {
        "dispute_reduction_bps__kyc_tier_delta": 0.45,
        "dispute_reduction_bps__reserve_deposit_usd_delta": 0.015,
        "dispute_reduction_bps__settlement_hold_days_delta": 0.025,
        "kyc_tier_delta__reserve_deposit_usd_delta": 0.18,
        "kyc_tier_delta__settlement_hold_days_delta": 0.30,
        "reserve_deposit_usd_delta__settlement_hold_days_delta": 0.012,
    }

    candidates_per_instance = math.prod(spec["num_steps"] for spec in action_specs.values())
    assert 10**7 <= candidates_per_instance <= 10**9, (
        f"Candidate space {candidates_per_instance} outside [1e7, 1e9]"
    )

    min_single_step_cost = min(
        spec["linear_weight"] + spec["quadratic_weight"] for spec in action_specs.values()
    )
    tol_opt = 0.05
    assert tol_opt < min_single_step_cost, "tol_opt must be strictly less than min step cost"

    return {
        "schema_version": "1.0.0",
        "seed": SEED,
        "runtime_provenance": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
        },
        "feature_order": list(FEATURE_ORDER),
        "immutable_features": ["account_tenure_months"],
        "derived_features": ["effective_risk_ratio", "liquidity_coverage_index"],
        "monotonic_cst": list(MONOTONIC_CST),
        "action_specs": action_specs,
        "cross_weights": cross_weights,
        "candidates_per_instance": candidates_per_instance,
        "tolerances": {
            "dag_residual_abs": 1e-4,
            "optimality_one_sided": tol_opt,
            "tie_break_cost_eps": 1e-9,
        },
    }


def train_monotonic_classifier() -> HistGradientBoostingClassifier:
    """Train deterministic monotonic risk classifier aligned to the SCM DAG."""
    rng = np.random.default_rng(SEED)
    n_train = 14_000

    tenure = rng.uniform(1.0, 96.0, size=n_train)
    kyc = rng.integers(0, 4, size=n_train).astype(float)
    reserve = rng.uniform(0.0, 120_000.0, size=n_train)
    dispute = rng.uniform(0.0, 900.0, size=n_train)
    hold_days = rng.integers(0, 181, size=n_train).astype(float)

    eff_risk, liq_idx = compute_derived_nodes(tenure, kyc, reserve, dispute, hold_days)

    # Structural ground-truth log-odds strictly consistent with MONOTONIC_CST
    logit = (
        -1.35
        - 0.018 * tenure
        - 0.65 * kyc
        - 0.000022 * reserve
        + 0.0055 * dispute
        - 0.014 * hold_days
        + 1.35 * eff_risk
        - 0.32 * liq_idx
        + rng.normal(0.0, 0.35, size=n_train)
    )
    probs = 1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0)))
    labels = (rng.uniform(0.0, 1.0, size=n_train) < probs).astype(int)

    X = np.column_stack([tenure, kyc, reserve, dispute, hold_days, eff_risk, liq_idx])
    clf = HistGradientBoostingClassifier(
        monotonic_cst=list(MONOTONIC_CST),
        max_iter=120,
        max_depth=5,
        learning_rate=0.06,
        min_samples_leaf=30,
        random_state=SEED,
    )
    clf.fit(X, labels)
    return clf


def build_slices_dataset() -> dict:
    """Generate deterministic 16-slice dataset and prove Audit A (Anti-Slack) bindings."""
    rng = np.random.default_rng(SEED + 100)
    base_dt = datetime(2026, 11, 1, 12, 0, 0, 500_000, tzinfo=timezone.utc)

    region_fp_mult = {"NA": 1.3, "EU": 1.2, "APAC": 1.0, "LATAM": 0.9}
    tier_fn_base_cents = {
        "micro": 18_000,
        "smb": 45_000,
        "mid_market": 120_000,
        "enterprise": 320_000,
    }
    tier_fp_base_cents = {
        "micro": 12_000,
        "smb": 32_000,
        "mid_market": 85_000,
        "enterprise": 260_000,
    }
    region_base_rate = {"NA": 0.22, "EU": 0.19, "APAC": 0.26, "LATAM": 0.31}
    tier_separation = {"micro": 1.45, "smb": 1.65, "mid_market": 1.80, "enterprise": 2.05}

    slices_payload: list[dict] = []
    ts_counter = 0

    for r_idx, region in enumerate(REGIONS):
        for t_idx, tier in enumerate(TIERS):
            slice_id = f"{region}_{tier}"
            n_samples = 140 + (r_idx * 11 + t_idx * 17) % 55
            base_rate = region_base_rate[region]
            sep = tier_separation[tier]

            # Draw true calibrated probabilities from a bimodal mixture, then sample labels
            # directly from p_true so ECE is naturally well-calibrated (< 0.08).
            n_high = int(round(n_samples * base_rate))
            n_low = n_samples - n_high
            high_scores = rng.beta(a=2.2 + 0.6 * sep, b=1.4, size=n_high)
            low_scores = rng.beta(a=1.5, b=2.5 + 0.7 * sep, size=n_low)
            raw_scores = np.concatenate([high_scores, low_scores])
            rng.shuffle(raw_scores)

            samples: list[dict] = []
            for idx, s_raw in enumerate(raw_scores):
                score = offset_score_from_grid(float(s_raw))
                label = int(rng.uniform(0.0, 1.0) < score)
                ts_counter += 17
                samples.append(
                    {
                        "sample_id": f"{slice_id}_{idx:04d}",
                        "score": score,
                        "label": label,
                        "timestamp_utc": format_utc_mid_second(base_dt, ts_counter),
                    }
                )

            fp_cost_cents = int(round(tier_fp_base_cents[tier] * region_fp_mult[region]))
            fn_cost_cents = int(round(tier_fn_base_cents[tier] * (1.0 + 0.1 * r_idx)))
            # Make FN substantially more expensive on a few slices so unconstrained optimum
            # pushes thresholds low (violating max_fpr & max_global_fp) while high-FP slices
            # push thresholds high (violating pairwise delta).
            if (r_idx + t_idx) % 3 == 0:
                fn_cost_cents = int(fn_cost_cents * 3.2)
            elif (r_idx + t_idx) % 3 == 1:
                fp_cost_cents = int(fp_cost_cents * 2.4)

            slices_payload.append(
                {
                    "slice_id": slice_id,
                    "fp_cost_cents": fp_cost_cents,
                    "fn_cost_cents": fn_cost_cents,
                    "max_ece": 0.12,
                    "max_fpr": 0.08,
                    "samples": samples,
                }
            )

    # Calibrate global FP budget and fairness delta to be strictly active (Audit A: Anti-Slack)
    evaluated_at = format_utc_mid_second(base_dt, 86_400)
    base_req = {
        "policy_id": "policy_2026_11_nominal_v1",
        "evaluated_at_utc": evaluated_at,
        "candidate_thresholds": CANDIDATE_THRESHOLDS,
        "pairwise_fairness_delta": 0.18,
        "max_global_fp": 85,
        "slices": slices_payload,
    }

    # Verify all constraints are simultaneously feasible AND strictly binding
    constrained = optimize_policy(PolicyOptimizationRequest.model_validate(base_req))
    assert constrained.audit_status == "PASSED", (
        f"Nominal dataset must pass constrained audit, got: {constrained.violated_constraints}"
    )

    # Audit A check 1: Removing max_global_fp must change optimal loss/thresholds
    relaxed_fp_req = {**base_req, "max_global_fp": 100_000}
    relaxed_fp = optimize_policy(PolicyOptimizationRequest.model_validate(relaxed_fp_req))
    assert relaxed_fp.total_fp is not None and relaxed_fp.total_fp > base_req["max_global_fp"], (
        "Anti-Slack violation: max_global_fp is slack!"
    )

    # Audit A check 2: Removing pairwise_fairness_delta must change optimal solution
    relaxed_delta_req = {**base_req, "pairwise_fairness_delta": 1.0}
    relaxed_delta = optimize_policy(PolicyOptimizationRequest.model_validate(relaxed_delta_req))
    assert (
        relaxed_delta.global_tpr_spread is not None
        and relaxed_delta.global_tpr_spread > base_req["pairwise_fairness_delta"]
    ), "Anti-Slack violation: pairwise_fairness_delta is slack!"

    # Audit A check 3: Removing per-slice max_fpr must change optimal solution
    relaxed_fpr_slices = [{**sl, "max_fpr": 1.0} for sl in slices_payload]
    relaxed_fpr_req = {**base_req, "slices": relaxed_fpr_slices}
    relaxed_fpr = optimize_policy(PolicyOptimizationRequest.model_validate(relaxed_fpr_req))
    assert relaxed_fpr.selected_thresholds != constrained.selected_thresholds, (
        "Anti-Slack violation: per-slice max_fpr is slack!"
    )

    return {
        "policy_request": base_req,
        "oracle_summary": {
            "constrained_loss_cents": constrained.total_loss_cents,
            "constrained_total_fp": constrained.total_fp,
            "constrained_tpr_spread": constrained.global_tpr_spread,
            "relaxed_fp_total_fp": relaxed_fp.total_fp,
            "relaxed_delta_tpr_spread": relaxed_delta.global_tpr_spread,
            "relaxed_fpr_loss_cents": relaxed_fpr.total_loss_cents,
        },
    }


def build_merchant_cases(
    clf: HistGradientBoostingClassifier,
    selected_thresholds: dict[str, float],
) -> list[dict]:
    """Generate 30 blocked merchants that are each provably flippable within action bounds."""
    rng = np.random.default_rng(SEED + 200)
    base_dt = datetime(2026, 11, 2, 8, 0, 0, 500_000, tzinfo=timezone.utc)
    ordered_slice_ids = sorted(selected_thresholds.keys())

    cases: list[dict] = []
    attempts = 0

    while len(cases) < 30:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("Failed to sample 30 feasible blocked merchant cases")

        slice_id = ordered_slice_ids[len(cases) % len(ordered_slice_ids)]
        threshold = selected_thresholds[slice_id]

        tenure = round(float(rng.uniform(2.0, 36.0)), 2)
        kyc = int(rng.integers(0, 2))  # Allows full +0/+1/+2 delta up to tier 3
        reserve = float(int(rng.integers(0, 60)) * 250)  # $0 .. $14,750
        dispute = float(int(rng.integers(105, 165)) * 5)  # 525 .. 820 bps (>= 500 bps!)
        hold_days = int(rng.integers(0, 20))

        eff_risk, liq_idx = compute_derived_nodes(
            np.array([tenure]),
            np.array([float(kyc)]),
            np.array([reserve]),
            np.array([dispute]),
            np.array([float(hold_days)]),
        )
        x_init = np.array(
            [[tenure, float(kyc), reserve, dispute, float(hold_days), eff_risk[0], liq_idx[0]]]
        )
        init_prob = float(clf.predict_proba(x_init)[0, 1])

        # Require merchant to be clearly blocked (away from boundary by >= 0.02)
        if init_prob < threshold + 0.02:
            continue

        # Verify optimistic max intervention strictly flips below threshold
        max_eff, max_liq = compute_derived_nodes(
            np.array([tenure]),
            np.array([float(kyc + 2)]),
            np.array([reserve + 100_000.0]),
            np.array([dispute - 500.0]),
            np.array([float(hold_days + 180)]),
        )
        x_max = np.array(
            [
                [
                    tenure,
                    float(kyc + 2),
                    reserve + 100_000.0,
                    dispute - 500.0,
                    float(hold_days + 180),
                    max_eff[0],
                    max_liq[0],
                ]
            ]
        )
        max_prob = float(clf.predict_proba(x_max)[0, 1])
        if max_prob >= threshold - 0.02:
            continue

        cases.append(
            {
                "merchant_id": f"MRC-{len(cases) + 1:04d}",
                "slice_id": slice_id,
                "threshold": threshold,
                "timestamp_utc": format_utc_mid_second(base_dt, len(cases) * 43),
                "features": {
                    "account_tenure_months": tenure,
                    "kyc_tier": kyc,
                    "reserve_deposit_usd": reserve,
                    "dispute_rate_bps": dispute,
                    "settlement_hold_days": hold_days,
                    "effective_risk_ratio": round(float(eff_risk[0]), 8),
                    "liquidity_coverage_index": round(float(liq_idx[0]), 8),
                },
            }
        )

    return cases


def main() -> None:
    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    scm_config = build_scm_config()
    clf = train_monotonic_classifier()
    slices_bundle = build_slices_dataset()

    nominal_res = optimize_policy(
        PolicyOptimizationRequest.model_validate(slices_bundle["policy_request"])
    )
    assert nominal_res.selected_thresholds is not None
    merchant_cases = build_merchant_cases(clf, nominal_res.selected_thresholds)

    (data_dir / "scm_config.json").write_text(
        json.dumps(scm_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    joblib.dump(clf, data_dir / "model.joblib", compress=3)
    (data_dir / "slices_dataset.json").write_text(
        json.dumps(slices_bundle["policy_request"], indent=2) + "\n", encoding="utf-8"
    )
    (data_dir / "merchant_cases.json").write_text(
        json.dumps(merchant_cases, indent=2) + "\n", encoding="utf-8"
    )

    print("ASSETS GENERATED & ANTI-SLACK VERIFIED")
    print(f"  Candidates per instance : {scm_config['candidates_per_instance']:,}")
    print(
        f"  Engine A constrained    : loss={nominal_res.total_loss_cents}c, "
        f"fp={nominal_res.total_fp}, spread={nominal_res.global_tpr_spread:.4f}"
    )
    print(f"  Anti-Slack witnesses    : {slices_bundle['oracle_summary']}")
    print(f"  Blocked merchant cases  : {len(merchant_cases)}")


if __name__ == "__main__":
    main()
