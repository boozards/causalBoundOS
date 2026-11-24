"""Strict Pydantic v2 contracts, exact key-set validators, and numeric guards."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ENGINE_A_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        "policy_id",
        "evaluated_at_utc",
        "audit_status",
        "selected_thresholds",
        "total_loss_cents",
        "total_fp",
        "total_fn",
        "global_tpr_spread",
        "slice_ece",
        "slice_metrics",
        "violated_constraints",
    }
)

ENGINE_A_SLICE_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "threshold",
        "fp",
        "fn",
        "tp",
        "tn",
        "tpr",
        "fpr",
        "loss_cents",
    }
)

ENGINE_B_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        "merchant_id",
        "slice_id",
        "audit_status",
        "threshold",
        "initial_risk_score",
        "counterfactual_risk_score",
        "actions",
        "step_counts",
        "counterfactual_features",
        "total_cost",
        "dag_residuals",
        "states_visited",
        "violated_constraints",
    }
)

ENGINE_B_ACTION_KEYS: frozenset[str] = frozenset(
    {
        "dispute_reduction_bps",
        "kyc_tier_delta",
        "reserve_deposit_usd_delta",
        "settlement_hold_days_delta",
    }
)

ENGINE_B_FEATURE_KEYS: frozenset[str] = frozenset(
    {
        "account_tenure_months",
        "kyc_tier",
        "reserve_deposit_usd",
        "dispute_rate_bps",
        "settlement_hold_days",
        "effective_risk_ratio",
        "liquidity_coverage_index",
    }
)

ENGINE_B_DAG_RESIDUAL_KEYS: frozenset[str] = frozenset(
    {
        "effective_risk_ratio",
        "liquidity_coverage_index",
    }
)


def ensure_finite(value: float, field_name: str) -> float:
    """Reject NaN, +/-Inf, or boolean-disguised values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a real number, got {type(value).__name__}")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{field_name} must be finite, got {value}")
    return val


def assert_strict_keys(data: dict, expected_keys: frozenset[str], context: str = "root") -> None:
    """Enforce exact key-set invariance (no missing keys, no extra leaked keys)."""
    actual_keys = set(data.keys())
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        raise ValueError(
            f"Strict key-set violation in [{context}]: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


class StrictModel(BaseModel):
    """Base schema forbidding extra keys and validating assignments."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Core Engine A Schemas
# ---------------------------------------------------------------------------


class RiskSample(StrictModel):
    sample_id: str = Field(min_length=1)
    score: float
    label: Literal[0, 1]
    timestamp_utc: str = Field(min_length=20)

    @field_validator("score")
    @classmethod
    def _validate_score(cls, v: float) -> float:
        v = ensure_finite(v, "score")
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"score must be in [0.0, 1.0], got {v}")
        return v


class SliceInput(StrictModel):
    slice_id: str = Field(min_length=1)
    fp_cost_cents: int = Field(ge=0)
    fn_cost_cents: int = Field(ge=0)
    max_ece: float
    max_fpr: float
    samples: list[RiskSample]

    @field_validator("max_ece", "max_fpr")
    @classmethod
    def _validate_bounds(cls, v: float, info) -> float:
        v = ensure_finite(v, info.field_name)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{info.field_name} must be in [0.0, 1.0], got {v}")
        return v


class PolicyOptimizationRequest(StrictModel):
    policy_id: str = Field(min_length=1)
    evaluated_at_utc: str = Field(min_length=20)
    candidate_thresholds: list[float] = Field(min_length=1)
    pairwise_fairness_delta: float
    max_global_fp: int = Field(ge=0)
    slices: list[SliceInput] = Field(min_length=1)

    @field_validator("pairwise_fairness_delta")
    @classmethod
    def _validate_delta(cls, v: float) -> float:
        v = ensure_finite(v, "pairwise_fairness_delta")
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"pairwise_fairness_delta must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("candidate_thresholds")
    @classmethod
    def _validate_thresholds(cls, vals: list[float]) -> list[float]:
        cleaned: list[float] = []
        for t in vals:
            t_val = ensure_finite(t, "candidate_threshold")
            if not (0.0 <= t_val <= 1.0):
                raise ValueError(f"candidate_threshold must be in [0.0, 1.0], got {t_val}")
            cleaned.append(t_val)
        return cleaned

    @model_validator(mode="after")
    def _unique_slice_ids(self) -> PolicyOptimizationRequest:
        ids = [s.slice_id for s in self.slices]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate slice_id entries are forbidden")
        return self


class SliceMetricOutput(StrictModel):
    threshold: float
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    tp: int = Field(ge=0)
    tn: int = Field(ge=0)
    tpr: float
    fpr: float
    loss_cents: int = Field(ge=0)

    @field_validator("threshold", "tpr", "fpr")
    @classmethod
    def _validate_floats(cls, v: float, info) -> float:
        return ensure_finite(v, info.field_name)


class PolicyOptimizationResult(StrictModel):
    policy_id: str
    evaluated_at_utc: str
    audit_status: Literal["PASSED", "FAILED_AUDIT"]
    selected_thresholds: dict[str, float] | None
    total_loss_cents: int | None
    total_fp: int | None
    total_fn: int | None
    global_tpr_spread: float | None
    slice_ece: dict[str, float]
    slice_metrics: dict[str, SliceMetricOutput] | None
    violated_constraints: list[str]

    @field_validator("slice_ece")
    @classmethod
    def _validate_ece_map(cls, v: dict[str, float]) -> dict[str, float]:
        return {k: ensure_finite(val, f"slice_ece[{k}]") for k, val in v.items()}

    @field_validator("global_tpr_spread")
    @classmethod
    def _validate_spread(cls, v: float | None) -> float | None:
        return None if v is None else ensure_finite(v, "global_tpr_spread")

    @model_validator(mode="after")
    def _validate_status_invariants(self) -> PolicyOptimizationResult:
        if self.audit_status == "FAILED_AUDIT":
            nullable_fields = (
                self.selected_thresholds,
                self.total_loss_cents,
                self.total_fp,
                self.total_fn,
                self.global_tpr_spread,
                self.slice_metrics,
            )
            if any(f is not None for f in nullable_fields):
                raise ValueError("FAILED_AUDIT requires all selection fields to be null")
            if not self.violated_constraints:
                raise ValueError("FAILED_AUDIT requires non-empty violated_constraints")
        else:
            required_fields = (
                self.selected_thresholds,
                self.total_loss_cents,
                self.total_fp,
                self.total_fn,
                self.global_tpr_spread,
                self.slice_metrics,
            )
            if any(f is None for f in required_fields):
                raise ValueError("PASSED status requires all selection fields to be populated")
            if self.violated_constraints:
                raise ValueError("PASSED status requires empty violated_constraints")
        return self

    def to_strict_dict(self) -> dict:
        payload = self.model_dump(mode="python")
        assert_strict_keys(payload, ENGINE_A_OUTPUT_KEYS, context="engine_a_result")
        if payload["slice_metrics"] is not None:
            for sid, sm in payload["slice_metrics"].items():
                assert_strict_keys(sm, ENGINE_A_SLICE_METRIC_KEYS, context=f"slice_metrics[{sid}]")
        return payload


# ---------------------------------------------------------------------------
# Core Engine B Schemas
# ---------------------------------------------------------------------------


class MerchantFeatures(StrictModel):
    account_tenure_months: float = Field(ge=0.0)
    kyc_tier: int = Field(ge=0)
    reserve_deposit_usd: float = Field(ge=0.0)
    dispute_rate_bps: float = Field(ge=0.0)
    settlement_hold_days: int = Field(ge=0)
    effective_risk_ratio: float
    liquidity_coverage_index: float

    @field_validator(
        "account_tenure_months",
        "reserve_deposit_usd",
        "dispute_rate_bps",
        "effective_risk_ratio",
        "liquidity_coverage_index",
    )
    @classmethod
    def _validate_finite_features(cls, v: float, info) -> float:
        return ensure_finite(v, info.field_name)


class MerchantRecourseCase(StrictModel):
    merchant_id: str = Field(min_length=1)
    slice_id: str = Field(min_length=1)
    threshold: float
    timestamp_utc: str = Field(min_length=20)
    features: MerchantFeatures

    @field_validator("threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        v = ensure_finite(v, "threshold")
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {v}")
        return v


class RecourseActions(StrictModel):
    dispute_reduction_bps: float = Field(ge=0.0)
    kyc_tier_delta: int = Field(ge=0)
    reserve_deposit_usd_delta: float = Field(ge=0.0)
    settlement_hold_days_delta: int = Field(ge=0)


class RecourseStepCounts(StrictModel):
    dispute_reduction_bps: int = Field(ge=0)
    kyc_tier_delta: int = Field(ge=0)
    reserve_deposit_usd_delta: int = Field(ge=0)
    settlement_hold_days_delta: int = Field(ge=0)


class DagResiduals(StrictModel):
    effective_risk_ratio: float = Field(ge=0.0)
    liquidity_coverage_index: float = Field(ge=0.0)

    @field_validator("effective_risk_ratio", "liquidity_coverage_index")
    @classmethod
    def _validate_finite(cls, v: float, info) -> float:
        return ensure_finite(v, info.field_name)


class RecourseResult(StrictModel):
    merchant_id: str
    slice_id: str
    audit_status: Literal["PASSED", "ALREADY_APPROVED", "FAILED_AUDIT"]
    threshold: float
    initial_risk_score: float
    counterfactual_risk_score: float | None
    actions: dict[str, int | float] | None
    step_counts: dict[str, int] | None
    counterfactual_features: MerchantFeatures | None
    total_cost: float | None
    dag_residuals: DagResiduals | None
    states_visited: int = Field(ge=0)
    violated_constraints: list[str]

    @field_validator("threshold", "initial_risk_score")
    @classmethod
    def _validate_required_floats(cls, v: float, info) -> float:
        return ensure_finite(v, info.field_name)

    @field_validator("counterfactual_risk_score", "total_cost")
    @classmethod
    def _validate_optional_floats(cls, v: float | None, info) -> float | None:
        return None if v is None else ensure_finite(v, info.field_name)

    @model_validator(mode="after")
    def _validate_recourse_status(self) -> RecourseResult:
        if self.audit_status == "FAILED_AUDIT":
            nullable = (
                self.counterfactual_risk_score,
                self.actions,
                self.step_counts,
                self.counterfactual_features,
                self.total_cost,
                self.dag_residuals,
            )
            if any(field is not None for field in nullable):
                raise ValueError("FAILED_AUDIT requires all recourse selection fields to be null")
            if not self.violated_constraints:
                raise ValueError("FAILED_AUDIT requires non-empty violated_constraints")
        else:
            required = (
                self.counterfactual_risk_score,
                self.actions,
                self.step_counts,
                self.counterfactual_features,
                self.total_cost,
                self.dag_residuals,
            )
            if any(field is None for field in required):
                raise ValueError(f"{self.audit_status} requires all recourse fields to be populated")
            if self.violated_constraints:
                raise ValueError(f"{self.audit_status} requires empty violated_constraints")
            assert self.actions is not None and self.step_counts is not None
            assert_strict_keys(self.actions, ENGINE_B_ACTION_KEYS, context="recourse.actions")
            assert_strict_keys(
                self.step_counts, ENGINE_B_ACTION_KEYS, context="recourse.step_counts"
            )
        return self

    def to_strict_dict(self) -> dict:
        payload = self.model_dump(mode="python")
        assert_strict_keys(payload, ENGINE_B_OUTPUT_KEYS, context="engine_b_result")
        if payload["actions"] is not None:
            assert_strict_keys(payload["actions"], ENGINE_B_ACTION_KEYS, context="actions")
        if payload["step_counts"] is not None:
            assert_strict_keys(payload["step_counts"], ENGINE_B_ACTION_KEYS, context="step_counts")
        if payload["counterfactual_features"] is not None:
            assert_strict_keys(
                payload["counterfactual_features"],
                ENGINE_B_FEATURE_KEYS,
                context="counterfactual_features",
            )
        if payload["dag_residuals"] is not None:
            assert_strict_keys(
                payload["dag_residuals"],
                ENGINE_B_DAG_RESIDUAL_KEYS,
                context="dag_residuals",
            )
        return payload
