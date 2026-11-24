"""Tests for Core Engine A: Exact Coupled Slice-Threshold Optimizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalbound.engine_a.slice_optimizer import optimize_policy
from causalbound.schemas import (
    ENGINE_A_OUTPUT_KEYS,
    PolicyOptimizationRequest,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SLICES_DATASET_PATH = DATA_DIR / "slices_dataset.json"


@pytest.fixture
def base_request() -> PolicyOptimizationRequest:
    raw = json.loads(SLICES_DATASET_PATH.read_text(encoding="utf-8"))
    return PolicyOptimizationRequest.model_validate(raw)


def test_engine_a_feasible_nominal_solve(base_request: PolicyOptimizationRequest) -> None:
    res = optimize_policy(base_request)
    assert res.audit_status == "PASSED"
    assert res.total_loss_cents is not None
    assert res.total_loss_cents > 0
    assert res.total_fp is not None
    assert res.total_fp <= base_request.max_global_fp
    assert res.global_tpr_spread is not None
    assert res.global_tpr_spread <= base_request.pairwise_fairness_delta + 1e-9
    assert res.selected_thresholds is not None
    assert len(res.selected_thresholds) == len(base_request.slices)
    assert res.violated_constraints == []

    d = res.to_strict_dict()
    assert set(d.keys()) == ENGINE_A_OUTPUT_KEYS


def test_engine_a_infeasible_fairness_delta(base_request: PolicyOptimizationRequest) -> None:
    # Set near-zero fairness delta
    strict_req = base_request.model_copy(update={"pairwise_fairness_delta": 0.001})
    res = optimize_policy(strict_req)

    assert res.audit_status == "FAILED_AUDIT"
    assert res.selected_thresholds is None
    assert res.total_loss_cents is None
    assert res.total_fp is None
    assert res.total_fn is None
    assert res.global_tpr_spread is None
    assert res.slice_metrics is None
    assert "PAIRWISE_FAIRNESS_INFEASIBLE" in res.violated_constraints

    d = res.to_strict_dict()
    assert set(d.keys()) == ENGINE_A_OUTPUT_KEYS


def test_engine_a_infeasible_fpr_ceiling(base_request: PolicyOptimizationRequest) -> None:
    # Restrict candidate thresholds to low values where FPR exceeds ceiling
    modified_slices = [s.model_copy(update={"max_fpr": 0.0001}) for s in base_request.slices]
    fpr_req = base_request.model_copy(
        update={
            "candidate_thresholds": [0.10, 0.20],
            "slices": modified_slices,
        }
    )
    res = optimize_policy(fpr_req)

    assert res.audit_status == "FAILED_AUDIT"
    assert res.selected_thresholds is None
    assert any("SLICE_FPR_INFEASIBLE" in vc for vc in res.violated_constraints)

    d = res.to_strict_dict()
    assert set(d.keys()) == ENGINE_A_OUTPUT_KEYS


def test_engine_a_ece_calibration_gate(base_request: PolicyOptimizationRequest) -> None:
    # Set impossible ECE gate on the first slice
    modified_slices = list(base_request.slices)
    modified_slices[0] = modified_slices[0].model_copy(update={"max_ece": 0.00001})
    ece_req = base_request.model_copy(update={"slices": modified_slices})
    res = optimize_policy(ece_req)

    assert res.audit_status == "FAILED_AUDIT"
    assert res.selected_thresholds is None
    assert any("SLICE_ECE_EXCEEDED" in vc for vc in res.violated_constraints)

    d = res.to_strict_dict()
    assert set(d.keys()) == ENGINE_A_OUTPUT_KEYS
