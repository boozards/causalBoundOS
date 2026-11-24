"""FastAPI backend + SSE telemetry stream for the CausalBound Risk War Room UI."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from causalbound.agents.workflow import CausalBoundWorkflow
from causalbound.mcp_servers.policy_solver_mcp import (
    diagnose_infeasibility_tool,
    solve_merchant_policy_tool,
)
from causalbound.telemetry import get_waterfall_exporter, init_telemetry

DATA_DIR = Path(os.environ.get("CAUSALBOUND_DATA_DIR", "data"))
BENCHMARK_REPORT_JSON = Path(os.environ.get("CAUSALBOUND_BENCHMARK_JSON", "benchmarks/latest_report.json"))

init_telemetry()
app = FastAPI(title="CausalBound Risk Policy & Recourse API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKFLOW = CausalBoundWorkflow(data_dir=DATA_DIR)


class PolicySimulatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_global_fp: int = Field(ge=0, le=2000)
    pairwise_fairness_delta: float = Field(ge=0.0, le=1.0)
    max_fpr_override: float | None = Field(default=None, ge=0.0, le=1.0)
    region_abuse_multipliers: dict[str, float] = Field(default_factory=dict)


class PolicyPromptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=4, max_length=2000)


class HITLDecisionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approve: bool
    reviewer: str = Field(min_length=1, max_length=120)


class RecourseRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    merchant_id: str = Field(min_length=1)
    threshold_override: float | None = Field(default=None, ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "causalbound"}


@app.post("/api/policy/simulate")
def simulate_policy(body: PolicySimulatePayload) -> dict[str, Any]:
    res = solve_merchant_policy_tool(
        max_global_fp=body.max_global_fp,
        pairwise_fairness_delta=body.pairwise_fairness_delta,
        max_fpr_override=body.max_fpr_override,
        region_abuse_multipliers=body.region_abuse_multipliers,
        data_dir=DATA_DIR,
    )
    diagnosis = None
    if res["audit_status"] == "FAILED_AUDIT":
        diagnosis = diagnose_infeasibility_tool(
            max_global_fp=body.max_global_fp,
            pairwise_fairness_delta=body.pairwise_fairness_delta,
            max_fpr_override=body.max_fpr_override,
            region_abuse_multipliers=body.region_abuse_multipliers,
            data_dir=DATA_DIR,
        )
    return {"policy_result": res, "infeasibility_diagnosis": diagnosis}


@app.post("/api/policy/propose")
def propose_policy(body: PolicyPromptPayload) -> dict[str, Any]:
    return WORKFLOW.propose_policy_from_prompt(body.prompt)


@app.post("/api/policy/proposals/{proposal_id}/hitl")
def hitl_decision(proposal_id: str, body: HITLDecisionPayload) -> dict[str, Any]:
    try:
        return WORKFLOW.resolve_hitl(proposal_id, approve=body.approve, reviewer=body.reviewer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/policy/ledger")
def get_policy_ledger() -> dict[str, Any]:
    return {"entries": WORKFLOW.list_ledger()}


@app.get("/api/merchants")
def list_merchants() -> dict[str, Any]:
    cases_path = DATA_DIR / "merchant_cases.json"
    if not cases_path.is_file():
        raise HTTPException(status_code=503, detail="merchant_cases.json not generated yet")
    return {"merchants": json.loads(cases_path.read_text(encoding="utf-8"))}


@app.post("/api/recourse/solve")
def solve_merchant_appeal(body: RecourseRequestPayload) -> dict[str, Any]:
    try:
        return WORKFLOW.resolve_merchant_appeal(
            merchant_id=body.merchant_id,
            threshold_override=body.threshold_override,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/telemetry/waterfall")
def get_telemetry_waterfall(limit: int = 100) -> dict[str, Any]:
    exporter = get_waterfall_exporter()
    return {
        "summary": exporter.summary(),
        "spans": exporter.recent_spans(limit=limit),
    }


@app.get("/api/telemetry/stream")
async def stream_telemetry() -> StreamingResponse:
    """Server-Sent Events stream pushing live OTel span snapshots to the War Room UI."""
    exporter = get_waterfall_exporter()

    async def event_generator():
        last_count = -1
        for _ in range(120):
            spans = exporter.recent_spans(limit=60)
            if len(spans) != last_count:
                last_count = len(spans)
                payload = json.dumps({"summary": exporter.summary(), "spans": spans})
                yield f"data: {payload}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/benchmark/latest")
def get_latest_benchmark() -> dict[str, Any]:
    if not BENCHMARK_REPORT_JSON.is_file():
        return {"available": False, "report": None}
    return {
        "available": True,
        "report": json.loads(BENCHMARK_REPORT_JSON.read_text(encoding="utf-8")),
    }
