"""Typed MCP Server exposing Core Engine B (Non-Linear Causal Recourse Solver)."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from causalbound.engine_b.bnb_recourse import solve_recourse
from causalbound.engine_b.greedy_seed import solve_greedy_1d_baseline
from causalbound.schemas import MerchantRecourseCase
from causalbound.scm import SCMModelSpec, compute_instance_max_steps, steps_to_action_dicts
from causalbound.telemetry import trace_solver_span

DEFAULT_DATA_DIR = Path(os.environ.get("CAUSALBOUND_DATA_DIR", "data"))


@lru_cache(maxsize=4)
def _load_engine_b_assets(data_dir_str: str) -> tuple[SCMModelSpec, Any, dict[str, dict[str, Any]]]:
    root = Path(data_dir_str)
    scm_spec = SCMModelSpec.from_file(root / "scm_config.json")
    clf = joblib.load(root / "model.joblib")
    cases_list = json.loads((root / "merchant_cases.json").read_text(encoding="utf-8"))
    cases_by_id = {c["merchant_id"]: c for c in cases_list}
    return scm_spec, clf, cases_by_id


def solve_merchant_recourse_tool(
    merchant_id: str | None = None,
    threshold_override: float | None = None,
    case_payload: dict[str, Any] | None = None,
    include_greedy_comparison: bool = True,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Strict MCP tool: compute exact optimal causal recourse + DAG residuals + 1D greedy comparison."""
    root = (data_dir or DEFAULT_DATA_DIR).resolve()
    scm_spec, clf, cases_by_id = _load_engine_b_assets(str(root))

    if case_payload is not None:
        raw_case = dict(case_payload)
    else:
        if merchant_id is None or merchant_id not in cases_by_id:
            raise ValueError(f"Unknown merchant_id: {merchant_id}")
        raw_case = dict(cases_by_id[merchant_id])

    if threshold_override is not None:
        raw_case["threshold"] = float(threshold_override)

    case = MerchantRecourseCase.model_validate(raw_case)

    with trace_solver_span(
        "engine_b",
        "solve_recourse",
        {
            "recourse.merchant_id": case.merchant_id,
            "recourse.slice_id": case.slice_id,
            "recourse.threshold": case.threshold,
            "recourse.action_space_size": scm_spec.candidates_per_instance,
        },
    ) as span:
        result = solve_recourse(case, clf, scm_spec).to_strict_dict()
        span.set_attribute("recourse.audit_status", result["audit_status"])
        span.set_attribute("recourse.states_visited", result["states_visited"])
        span.set_attribute(
            "recourse.pruning_ratio",
            round(result["states_visited"] / scm_spec.candidates_per_instance, 8),
        )
        if result["total_cost"] is not None:
            span.set_attribute("recourse.total_cost", result["total_cost"])

    response: dict[str, Any] = {"recourse": result}

    if include_greedy_comparison:
        max_steps = compute_instance_max_steps(case.features, scm_spec)
        greedy = solve_greedy_1d_baseline(case.features, case.threshold, max_steps, clf, scm_spec)
        if greedy is None:
            response["greedy_1d_baseline"] = {
                "feasible": False,
                "total_cost": None,
                "actions": None,
                "cost_savings_vs_greedy": None,
            }
        else:
            g_actions, _ = steps_to_action_dicts(greedy.steps, scm_spec)
            savings = (
                round(greedy.cost - result["total_cost"], 6)
                if result["total_cost"] is not None
                else None
            )
            response["greedy_1d_baseline"] = {
                "feasible": True,
                "total_cost": round(greedy.cost, 6),
                "actions": g_actions,
                "cost_savings_vs_greedy": savings,
            }

    return response


def build_fastmcp_server():
    from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]

    mcp = FastMCP("causalbound-recourse-solver")
    mcp.tool()(solve_merchant_recourse_tool)
    return mcp


if __name__ == "__main__":
    build_fastmcp_server().run()
