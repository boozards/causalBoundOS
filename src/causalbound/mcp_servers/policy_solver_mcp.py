"""Typed MCP Server exposing Core Engine A (Coupled Slice-Threshold Optimizer).

Supports both:
  1. Direct in-process typed tool invocation by ADK agents & FastAPI (zero IPC overhead)
  2. Standalone stdio MCP server mode (`python -m causalbound.mcp_servers.policy_solver_mcp`)
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from causalbound.engine_a.slice_optimizer import optimize_policy
from causalbound.schemas import PolicyOptimizationRequest
from causalbound.telemetry import trace_solver_span, utc_now_iso

DEFAULT_DATA_DIR = Path(os.environ.get("CAUSALBOUND_DATA_DIR", "data"))


def _load_default_dataset(data_dir: Path | None = None) -> dict[str, Any]:
    root = data_dir or DEFAULT_DATA_DIR
    return json.loads((root / "slices_dataset.json").read_text(encoding="utf-8"))


def solve_merchant_policy_tool(
    max_global_fp: int,
    pairwise_fairness_delta: float,
    max_fpr_override: float | None = None,
    max_ece_override: float | None = None,
    region_abuse_multipliers: dict[str, float] | None = None,
    policy_id: str | None = None,
    slices_payload: dict[str, Any] | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Strict MCP tool: compile overrides onto slice dataset and run exact Engine A."""
    raw = copy.deepcopy(slices_payload) if slices_payload is not None else _load_default_dataset(data_dir)
    raw["policy_id"] = policy_id or f"policy_live_{int(max_global_fp)}_{round(pairwise_fairness_delta, 3)}"
    raw["evaluated_at_utc"] = utc_now_iso()
    raw["max_global_fp"] = int(max_global_fp)
    raw["pairwise_fairness_delta"] = float(pairwise_fairness_delta)

    mults = region_abuse_multipliers or {}
    for sl in raw["slices"]:
        if max_fpr_override is not None:
            sl["max_fpr"] = float(max_fpr_override)
        if max_ece_override is not None:
            sl["max_ece"] = float(max_ece_override)
        region = sl["slice_id"].split("_", 1)[0]
        if region in mults:
            scale = max(0.1, float(mults[region]))
            sl["fn_cost_cents"] = int(round(sl["fn_cost_cents"] * scale))

    req = PolicyOptimizationRequest.model_validate(raw)
    with trace_solver_span(
        "engine_a",
        "optimize_policy",
        {
            "policy.max_global_fp": req.max_global_fp,
            "policy.pairwise_fairness_delta": req.pairwise_fairness_delta,
            "policy.num_slices": len(req.slices),
        },
    ) as span:
        result = optimize_policy(req).to_strict_dict()
        span.set_attribute("policy.audit_status", result["audit_status"])
        if result["total_loss_cents"] is not None:
            span.set_attribute("policy.total_loss_cents", result["total_loss_cents"])
            span.set_attribute("policy.total_fp", result["total_fp"])
    return result


def diagnose_infeasibility_tool(
    max_global_fp: int,
    pairwise_fairness_delta: float,
    max_fpr_override: float | None = None,
    region_abuse_multipliers: dict[str, float] | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute minimal single-constraint relaxations when Engine A returns FAILED_AUDIT."""
    base = solve_merchant_policy_tool(
        max_global_fp=max_global_fp,
        pairwise_fairness_delta=pairwise_fairness_delta,
        max_fpr_override=max_fpr_override,
        region_abuse_multipliers=region_abuse_multipliers,
        data_dir=data_dir,
    )
    if base["audit_status"] == "PASSED":
        return {"status": "FEASIBLE", "base_result": base, "relaxation_options": []}

    options: list[dict[str, Any]] = []

    # Option 1: Binary search minimum feasible global FP budget
    if "GLOBAL_FP_BUDGET_EXCEEDED" in base["violated_constraints"]:
        lo, hi = max_global_fp + 1, 500
        best_fp_opt = None
        while lo <= hi:
            mid = (lo + hi) // 2
            probe = solve_merchant_policy_tool(
                max_global_fp=mid,
                pairwise_fairness_delta=pairwise_fairness_delta,
                max_fpr_override=max_fpr_override,
                region_abuse_multipliers=region_abuse_multipliers,
                data_dir=data_dir,
            )
            if probe["audit_status"] == "PASSED":
                best_fp_opt = (mid, probe)
                hi = mid - 1
            else:
                lo = mid + 1
        if best_fp_opt is not None:
            min_fp, res = best_fp_opt
            options.append(
                {
                    "lever": "max_global_fp",
                    "requested": max_global_fp,
                    "minimal_feasible": min_fp,
                    "resulting_loss_cents": res["total_loss_cents"],
                    "resulting_tpr_spread": res["global_tpr_spread"],
                }
            )

    # Option 2: Sweep minimal feasible pairwise fairness delta in 0.02 increments
    for step in range(1, 51):
        cand_delta = round(min(1.0, pairwise_fairness_delta + step * 0.02), 2)
        probe = solve_merchant_policy_tool(
            max_global_fp=max_global_fp,
            pairwise_fairness_delta=cand_delta,
            max_fpr_override=max_fpr_override,
            region_abuse_multipliers=region_abuse_multipliers,
            data_dir=data_dir,
        )
        if probe["audit_status"] == "PASSED":
            options.append(
                {
                    "lever": "pairwise_fairness_delta",
                    "requested": pairwise_fairness_delta,
                    "minimal_feasible": cand_delta,
                    "resulting_loss_cents": probe["total_loss_cents"],
                    "resulting_total_fp": probe["total_fp"],
                }
            )
            break

    return {
        "status": "INFEASIBLE",
        "violated_constraints": base["violated_constraints"],
        "relaxation_options": options,
    }


def build_fastmcp_server():
    """Create runnable FastMCP server when `mcp` package is installed."""
    from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]

    mcp = FastMCP("causalbound-policy-solver")
    mcp.tool()(solve_merchant_policy_tool)
    mcp.tool()(diagnose_infeasibility_tool)
    return mcp


if __name__ == "__main__":
    build_fastmcp_server().run()
