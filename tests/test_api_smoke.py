"""Smoke tests for the CausalBound Risk Policy & Recourse FastAPI application."""

from __future__ import annotations

import asyncio

from causalbound.api import (
    HITLDecisionPayload,
    PolicyPromptPayload,
    PolicySimulatePayload,
    RecourseRequestPayload,
    get_latest_benchmark,
    get_policy_ledger,
    get_telemetry_waterfall,
    health,
    hitl_decision,
    list_merchants,
    propose_policy,
    simulate_policy,
    solve_merchant_appeal,
    stream_telemetry,
)


def test_health():
    res = health()
    assert res == {"status": "ok", "service": "causalbound"}


def test_simulate_feasible():
    res = simulate_policy(
        PolicySimulatePayload(
            max_global_fp=120,
            pairwise_fairness_delta=0.20,
            max_fpr_override=None,
            region_abuse_multipliers={"LATAM": 1.5},
        )
    )
    assert res["policy_result"]["audit_status"] == "PASSED"
    assert res["policy_result"]["total_loss_cents"] > 0
    assert res["infeasibility_diagnosis"] is None


def test_simulate_infeasible():
    res = simulate_policy(
        PolicySimulatePayload(
            max_global_fp=1,
            pairwise_fairness_delta=0.01,
        )
    )
    assert res["policy_result"]["audit_status"] == "FAILED_AUDIT"
    assert res["infeasibility_diagnosis"] is not None
    assert "relaxation_options" in res["infeasibility_diagnosis"]
    assert len(res["infeasibility_diagnosis"]["relaxation_options"]) > 0


def test_propose_and_hitl_workflow():
    prompt = "Cap global false bans at 95 with fairness delta 0.18 during LATAM wave"
    prop = propose_policy(PolicyPromptPayload(prompt=prompt))
    assert prop["status"] == "AWAITING_HITL_APPROVAL"
    pid = prop["proposal_id"]

    # Approve proposal
    approved = hitl_decision(
        pid,
        HITLDecisionPayload(
            approve=True,
            reviewer="risk-director@stripe.com",
        ),
    )
    assert approved["status"] == "DEPLOYED"
    assert approved["approved_by"] == "risk-director@stripe.com"

    # Verify ledger
    ledger = get_policy_ledger()
    assert any(entry["proposal_id"] == pid for entry in ledger["entries"])


def test_merchants_and_recourse():
    merchants_resp = list_merchants()
    merchants = merchants_resp["merchants"]
    assert len(merchants) == 30
    first_id = merchants[0]["merchant_id"]

    appeal_resp = solve_merchant_appeal(RecourseRequestPayload(merchant_id=first_id))
    assert "recovery_card" in appeal_resp
    card = appeal_resp["recovery_card"]
    assert card["verified_guarantee"] is True
    assert "action_steps" in card
    assert "headline" in card


def test_telemetry_waterfall():
    wat = get_telemetry_waterfall()
    assert "summary" in wat
    assert "spans" in wat
    assert wat["summary"]["total_spans"] > 0


def test_stream_telemetry():
    async def _run():
        resp = await stream_telemetry()
        chunks = []
        async for chunk in resp.body_iterator:
            chunks.append(chunk)
            if len(chunks) >= 1:
                break
        return chunks

    chunks = asyncio.run(_run())
    assert len(chunks) == 1
    assert chunks[0].startswith("data: ")


def test_latest_benchmark():
    bench = get_latest_benchmark()
    assert bench["available"] is True
    assert bench["report"] is not None
    assert "v1_pure_llm" in bench["report"]
    assert "v2_causalbound" in bench["report"]
