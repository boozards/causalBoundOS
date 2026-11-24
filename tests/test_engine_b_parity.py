"""Days 4-7 Exit Gate Tests for Core Engine B (SCM + B&B Recourse Solver).

Verifies:
  1. 100% exact parity with brute-force oracle across small-scale coupled grids
  2. Non-separability witness: coupled optimal B&B beats naive 1D greedy baseline
  3. Exact 3-tier tie-breaking (Audit B: Anti-Collapse with >= 2 equal-cost candidates)
  4. Edge states: ALREADY_APPROVED, FAILED_AUDIT (strict null mapping), immutable tenure
  5. Strict key-set invariance, finiteness, and SCM DAG residuals <= 1e-4
  6. Determinism across 5 consecutive runs on shuffled input dictionaries
  7. Full-scale (~2.20e7 candidates/instance) latency (< 10s) and pruning ratio (<= 1e-4)
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pytest

from causalbound.engine_b.bnb_recourse import solve_recourse, solve_recourse_brute_force
from causalbound.engine_b.greedy_seed import solve_greedy_1d_baseline
from causalbound.schemas import (
    ENGINE_B_OUTPUT_KEYS,
    MerchantFeatures,
    MerchantRecourseCase,
    assert_strict_keys,
)
from causalbound.scm import (
    SCMModelSpec,
    compute_derived_scalar,
    evaluate_cost_scalar,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


class MonotonicThresholdMockClassifier:
    """Deterministic monotonic classifier for controlled small-grid & tie-break tests."""

    def __init__(self, weights: tuple[float, ...], bias: float) -> None:
        self._w = np.asarray(weights, dtype=np.float64)
        self._b = float(bias)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self._w + self._b
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return np.column_stack([1.0 - p1, p1])


def _make_small_scm_spec(
    *,
    cross_scale: float = 1.0,
    equal_weights: bool = False,
) -> SCMModelSpec:
    """Create a 3 x 9 x 9 x 9 = 2,187-candidate SCM spec for exact brute-force parity."""
    if equal_weights:
        action_specs = {
            "dispute_reduction_bps": {
                "target_feature": "dispute_rate_bps",
                "direction": -1,
                "min_delta": 0,
                "max_delta": 40,
                "step": 5,
                "num_steps": 9,
                "linear_weight": 2.0,
                "quadratic_weight": 0.5,
            },
            "kyc_tier_delta": {
                "target_feature": "kyc_tier",
                "direction": 1,
                "min_delta": 0,
                "max_delta": 2,
                "step": 1,
                "num_steps": 3,
                "linear_weight": 2.0,
                "quadratic_weight": 0.5,
            },
            "reserve_deposit_usd_delta": {
                "target_feature": "reserve_deposit_usd",
                "direction": 1,
                "min_delta": 0,
                "max_delta": 2000,
                "step": 250,
                "num_steps": 9,
                "linear_weight": 2.0,
                "quadratic_weight": 0.5,
            },
            "settlement_hold_days_delta": {
                "target_feature": "settlement_hold_days",
                "direction": 1,
                "min_delta": 0,
                "max_delta": 8,
                "step": 1,
                "num_steps": 9,
                "linear_weight": 2.0,
                "quadratic_weight": 0.5,
            },
        }
    else:
        action_specs = {
            "dispute_reduction_bps": {
                "target_feature": "dispute_rate_bps",
                "direction": -1,
                "min_delta": 0,
                "max_delta": 40,
                "step": 5,
                "num_steps": 9,
                "linear_weight": 1.25,
                "quadratic_weight": 0.08,
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
                "max_delta": 2000,
                "step": 250,
                "num_steps": 9,
                "linear_weight": 1.00,
                "quadratic_weight": 0.05,
            },
            "settlement_hold_days_delta": {
                "target_feature": "settlement_hold_days",
                "direction": 1,
                "min_delta": 0,
                "max_delta": 8,
                "step": 1,
                "num_steps": 9,
                "linear_weight": 1.15,
                "quadratic_weight": 0.06,
            },
        }

    cfg = {
        "action_specs": action_specs,
        "cross_weights": {
            "dispute_reduction_bps__kyc_tier_delta": 0.45 * cross_scale,
            "dispute_reduction_bps__reserve_deposit_usd_delta": 0.12 * cross_scale,
            "dispute_reduction_bps__settlement_hold_days_delta": 0.15 * cross_scale,
            "kyc_tier_delta__reserve_deposit_usd_delta": 0.25 * cross_scale,
            "kyc_tier_delta__settlement_hold_days_delta": 0.30 * cross_scale,
            "reserve_deposit_usd_delta__settlement_hold_days_delta": 0.10 * cross_scale,
        },
        "candidates_per_instance": 3 * 9 * 9 * 9,
        "tolerances": {
            "dag_residual_abs": 1e-4,
            "optimality_one_sided": 0.05,
            "tie_break_cost_eps": 1e-9,
        },
    }
    return SCMModelSpec.from_dict(cfg)


def _make_case(
    merchant_id: str,
    threshold: float,
    tenure: float = 12.0,
    kyc: int = 0,
    reserve: float = 1000.0,
    dispute: float = 400.0,
    hold: int = 2,
) -> MerchantRecourseCase:
    eff, liq = compute_derived_scalar(tenure, float(kyc), reserve, dispute, float(hold))
    return MerchantRecourseCase(
        merchant_id=merchant_id,
        slice_id="NA_smb",
        threshold=threshold,
        timestamp_utc="2026-11-02T08:00:00.500Z",
        features=MerchantFeatures(
            account_tenure_months=tenure,
            kyc_tier=kyc,
            reserve_deposit_usd=reserve,
            dispute_rate_bps=dispute,
            settlement_hold_days=hold,
            effective_risk_ratio=eff,
            liquidity_coverage_index=liq,
        ),
    )


def test_exact_parity_against_brute_force_oracle() -> None:
    """Engine B must match brute-force oracle 100% across 8 diverse small-grid instances."""
    scm_spec = _make_small_scm_spec()
    # Strictly monotonic weights aligned with MONOTONIC_CST = (-1, -1, -1, +1, -1, +1, -1)
    clf = MonotonicThresholdMockClassifier(
        weights=(-0.01, -0.55, -0.0005, 0.012, -0.18, 0.90, -0.25),
        bias=-3.1,
    )

    test_configs = [
        ("BF-1", 0.50, 6.0, 0, 500.0, 420.0, 1),
        ("BF-2", 0.45, 18.0, 1, 1500.0, 460.0, 0),
        ("BF-3", 0.35, 4.0, 0, 0.0, 380.0, 3),
        ("BF-4", 0.60, 24.0, 0, 250.0, 520.0, 0),
        ("BF-5", 0.55, 10.0, 1, 750.0, 510.0, 2),
        ("BF-6", 0.25, 36.0, 0, 2000.0, 360.0, 1),
        ("BF-7", 0.70, 3.0, 0, 0.0, 580.0, 0),
        ("BF-8", 0.40, 14.0, 1, 500.0, 410.0, 2),
    ]

    for mid, thresh, tenure, kyc, res, disp, hold in test_configs:
        case = _make_case(mid, thresh, tenure, kyc, res, disp, hold)
        oracle = solve_recourse_brute_force(case, clf, scm_spec)
        bnb = solve_recourse(case, clf, scm_spec)

        assert bnb.audit_status == oracle.audit_status, f"{mid}: status mismatch"
        assert bnb.step_counts == oracle.step_counts, f"{mid}: step_counts mismatch"
        assert bnb.actions == oracle.actions, f"{mid}: actions mismatch"

        if oracle.audit_status == "PASSED":
            assert bnb.total_cost is not None and oracle.total_cost is not None
            assert math.isfinite(bnb.total_cost)
            assert bnb.total_cost == pytest.approx(oracle.total_cost, abs=1e-9)
            assert bnb.counterfactual_features is not None
            assert bnb.counterfactual_features.account_tenure_months == tenure
            assert bnb.dag_residuals is not None
            assert bnb.dag_residuals.effective_risk_ratio <= 1e-4
            assert bnb.dag_residuals.liquidity_coverage_index <= 1e-4

        strict_payload = bnb.to_strict_dict()
        assert_strict_keys(strict_payload, ENGINE_B_OUTPUT_KEYS)


def test_coupled_bnb_beats_naive_1d_greedy() -> None:
    """Verify exact B&B discovers a strictly cheaper multi-axis recourse than 1D greedy."""
    scm_spec = _make_small_scm_spec(cross_scale=0.1)
    clf = MonotonicThresholdMockClassifier(
        weights=(-0.01, -0.15, -0.00025, 0.008, -0.10, 0.80, -0.45),
        bias=-1.55,
    )
    case = _make_case("COUPLED-1", threshold=0.40, tenure=5.0, kyc=0, reserve=0.0, dispute=420.0, hold=0)

    bnb = solve_recourse(case, clf, scm_spec)
    oracle = solve_recourse_brute_force(case, clf, scm_spec)
    greedy_1d = solve_greedy_1d_baseline(
        case.features, case.threshold, (8, 2, 8, 8), clf, scm_spec
    )

    assert bnb.audit_status == "PASSED"
    assert bnb.step_counts == oracle.step_counts
    assert bnb.total_cost is not None
    # Either 1D greedy fails completely or is strictly more expensive than exact coupled B&B
    if greedy_1d is not None:
        assert bnb.total_cost < greedy_1d.cost - 0.05


def test_anti_collapse_tier3_lexicographical_tie_breaker() -> None:
    """Audit B: >= 2 valid equal-cost, equal-step-sum candidates force Tier-3 lex resolution."""
    scm_spec = _make_small_scm_spec(equal_weights=True, cross_scale=0.5)

    class EqualOneStepFlipClassifier:
        """Flips below threshold iff at least 1 step is taken along ANY single axis."""

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            # Base merchant has kyc=0, reserve=1000, dispute=400, hold=2
            changed = (
                (X[:, 1] > 0.5)
                | (X[:, 2] > 1000.5)
                | (X[:, 3] < 399.5)
                | (X[:, 4] > 2.5)
            )
            p1 = np.where(changed, 0.20, 0.80)
            return np.column_stack([1.0 - p1, p1])

    clf = EqualOneStepFlipClassifier()
    case = _make_case("TIE-LEX", threshold=0.50)

    # All 4 single-step vectors have identical cost (2.5) and identical step sum (1):
    # (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1).
    # Ascending lexicographical tuple order selects (0,0,0,1) -> settlement_hold_days_delta=1!
    c0 = evaluate_cost_scalar((1, 0, 0, 0), scm_spec)
    c3 = evaluate_cost_scalar((0, 0, 0, 1), scm_spec)
    assert c0 == pytest.approx(c3, abs=1e-12)

    bnb = solve_recourse(case, clf, scm_spec)
    oracle = solve_recourse_brute_force(case, clf, scm_spec)
    assert bnb.step_counts == oracle.step_counts
    assert bnb.step_counts == {
        "dispute_reduction_bps": 0,
        "kyc_tier_delta": 0,
        "reserve_deposit_usd_delta": 0,
        "settlement_hold_days_delta": 1,
    }


def test_edge_states_already_approved_and_infeasible() -> None:
    """Verify ALREADY_APPROVED zero-cost return and FAILED_AUDIT null-field contract."""
    scm_spec = _make_small_scm_spec()
    clf = MonotonicThresholdMockClassifier(
        weights=(-0.01, -0.55, -0.0005, 0.012, -0.18, 0.90, -0.25),
        bias=-3.1,
    )

    approved_case = _make_case("OK-1", threshold=0.99)
    res_ok = solve_recourse(approved_case, clf, scm_spec)
    assert res_ok.audit_status == "ALREADY_APPROVED"
    assert res_ok.total_cost == 0.0
    assert res_ok.step_counts == {
        "dispute_reduction_bps": 0,
        "kyc_tier_delta": 0,
        "reserve_deposit_usd_delta": 0,
        "settlement_hold_days_delta": 0,
    }

    impossible_case = _make_case("FAIL-1", threshold=0.0001, dispute=400.0)
    res_fail = solve_recourse(impossible_case, clf, scm_spec)
    assert res_fail.audit_status == "FAILED_AUDIT"
    assert res_fail.actions is None
    assert res_fail.step_counts is None
    assert res_fail.counterfactual_features is None
    assert res_fail.total_cost is None
    assert res_fail.dag_residuals is None
    assert res_fail.violated_constraints == ["NO_FEASIBLE_FLIP"]
    assert_strict_keys(res_fail.to_strict_dict(), ENGINE_B_OUTPUT_KEYS)


def test_full_scale_determinism_speed_and_pruning_ratio() -> None:
    """Days 4-7 Exit Gate on generated production assets (2.20e7 space/instance)."""
    scm_path = DATA_DIR / "scm_config.json"
    model_path = DATA_DIR / "model.joblib"
    cases_path = DATA_DIR / "merchant_cases.json"
    if not (scm_path.exists() and model_path.exists() and cases_path.exists()):
        pytest.skip("Run `python scripts/build_assets.py` first to generate full-scale assets")

    scm_spec = SCMModelSpec.from_file(scm_path)
    clf = joblib.load(model_path)
    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    assert len(raw_cases) == 30

    # 1. Full 30-merchant timing & pruning verification (< 10s total, ratio <= 1e-4)
    start = time.perf_counter()
    results = []
    for raw in raw_cases:
        case = MerchantRecourseCase.model_validate(raw)
        res = solve_recourse(case, clf, scm_spec)
        assert res.audit_status == "PASSED", f"{case.merchant_id} failed: {res.violated_constraints}"
        assert res.counterfactual_risk_score is not None
        assert res.counterfactual_risk_score < case.threshold
        assert res.states_visited / scm_spec.candidates_per_instance <= 1e-4, (
            f"{case.merchant_id} visited {res.states_visited} states "
            f"(ratio={res.states_visited / scm_spec.candidates_per_instance:.2e})"
        )
        assert_strict_keys(res.to_strict_dict(), ENGINE_B_OUTPUT_KEYS)
        results.append(res.to_strict_dict())
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"Full-scale 30-merchant solve took {elapsed:.2f}s (limit: 10.0s)"

    # 2. Determinism across 5 consecutive runs with shuffled input key order
    rng = random.Random(42)
    sample_raw = raw_cases[0]
    baseline = results[0]
    for _ in range(5):
        shuffled_features = list(sample_raw["features"].items())
        rng.shuffle(shuffled_features)
        permuted = copy.deepcopy(sample_raw)
        permuted["features"] = dict(shuffled_features)
        rerun = solve_recourse(
            MerchantRecourseCase.model_validate(permuted), clf, scm_spec
        ).to_strict_dict()
        assert rerun == baseline
