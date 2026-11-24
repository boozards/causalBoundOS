"""Core Engine A: Coupled 16-slice merchant risk threshold optimizer."""

from causalbound.engine_a.slice_optimizer import (
    compute_slice_ece,
    optimize_policy,
)

__all__ = ["compute_slice_ece", "optimize_policy"]
