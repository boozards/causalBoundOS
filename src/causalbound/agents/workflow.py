"""ADK 2.0 Graph Workflow with HITL Policy Deployment Gate.

Orchestrates:
  1. PolicyCompilerAgent (Flash) -> Strict Spec Contract
  2. policy_solver_mcp (Engine A) -> Exact optimal 16-slice thresholds
  3. Conditional escalation to PolicyCompilerAgent (Pro) ONLY on FAILED_AUDIT
  4. HITL Pause state ("AWAITING_HITL_APPROVAL") before writing to Policy Ledger
  5. AppealsRecourseAgent + recourse_mcp (Engine B) for blocked merchants
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalbound.agents.appeals_agent import AppealsRecourseAgent
from causalbound.agents.policy_compiler import PolicyCompilerAgent
from causalbound.mcp_servers.policy_solver_mcp import (
    diagnose_infeasibility_tool,
    solve_merchant_policy_tool,
)
from causalbound.mcp_servers.recourse_mcp import solve_merchant_recourse_tool
from causalbound.telemetry import init_telemetry, utc_now_iso


@dataclass
class PendingPolicyProposal:
    proposal_id: str
    created_at_utc: str
    operator_prompt: str
    compiled_spec: dict[str, Any]
    solver_result: dict[str, Any]
    status: str = "AWAITING_HITL_APPROVAL"  # AWAITING_HITL_APPROVAL | DEPLOYED | REJECTED | INFEASIBLE
    escalation_analysis: dict[str, Any] | None = None
    approved_by: str | None = None
    deployed_at_utc: str | None = None


class CausalBoundWorkflow:
    """End-to-end neuro-symbolic workflow with explicit HITL deployment pause."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir
        self.compiler = PolicyCompilerAgent()
        self.appeals = AppealsRecourseAgent()
        self._proposals: dict[str, PendingPolicyProposal] = {}
        self._deployed_ledger: list[dict[str, Any]] = []

    def propose_policy_from_prompt(self, operator_prompt: str) -> dict[str, Any]:
        tracer = init_telemetry()
        with tracer.start_as_current_span("workflow.propose_policy") as span:
            span.set_attribute("causalbound.span_category", "workflow")
            compiled = self.compiler.compile_intent(operator_prompt)
            spec_dict = compiled.model_dump()

            solver_res = solve_merchant_policy_tool(
                max_global_fp=compiled.max_global_fp,
                pairwise_fairness_delta=compiled.pairwise_fairness_delta,
                max_fpr_override=compiled.max_fpr_override,
                region_abuse_multipliers=compiled.region_abuse_multipliers,
                data_dir=self.data_dir,
            )

            proposal_id = f"PROP-{uuid.uuid4().hex[:8].upper()}"
            escalation: dict[str, Any] | None = None
            status = "AWAITING_HITL_APPROVAL"

            if solver_res["audit_status"] == "FAILED_AUDIT":
                status = "INFEASIBLE"
                diagnosis = diagnose_infeasibility_tool(
                    max_global_fp=compiled.max_global_fp,
                    pairwise_fairness_delta=compiled.pairwise_fairness_delta,
                    max_fpr_override=compiled.max_fpr_override,
                    region_abuse_multipliers=compiled.region_abuse_multipliers,
                    data_dir=self.data_dir,
                )
                escalation = self.compiler.negotiate_infeasibility(operator_prompt, diagnosis)

            proposal = PendingPolicyProposal(
                proposal_id=proposal_id,
                created_at_utc=utc_now_iso(),
                operator_prompt=operator_prompt,
                compiled_spec=spec_dict,
                solver_result=solver_res,
                status=status,
                escalation_analysis=escalation,
            )
            self._proposals[proposal_id] = proposal
            span.set_attribute("workflow.proposal_status", status)
            return self._serialize_proposal(proposal)

    def resolve_hitl(self, proposal_id: str, approve: bool, reviewer: str) -> dict[str, Any]:
        """HITL deployment gate: deploys verified thresholds to immutable ledger on approval."""
        if proposal_id not in self._proposals:
            raise KeyError(f"Unknown proposal_id: {proposal_id}")
        proposal = self._proposals[proposal_id]
        if proposal.status != "AWAITING_HITL_APPROVAL":
            raise ValueError(f"Proposal {proposal_id} is not awaiting HITL (status={proposal.status})")

        tracer = init_telemetry()
        with tracer.start_as_current_span("workflow.hitl_gate") as span:
            span.set_attribute("causalbound.span_category", "hitl")
            span.set_attribute("hitl.approved", approve)
            span.set_attribute("hitl.reviewer", reviewer)
            proposal.approved_by = reviewer
            if approve:
                proposal.status = "DEPLOYED"
                proposal.deployed_at_utc = utc_now_iso()
                self._deployed_ledger.append(
                    {
                        "version": len(self._deployed_ledger) + 1,
                        "proposal_id": proposal.proposal_id,
                        "deployed_at_utc": proposal.deployed_at_utc,
                        "approved_by": reviewer,
                        "total_loss_cents": proposal.solver_result["total_loss_cents"],
                        "total_fp": proposal.solver_result["total_fp"],
                        "global_tpr_spread": proposal.solver_result["global_tpr_spread"],
                        "selected_thresholds": proposal.solver_result["selected_thresholds"],
                    }
                )
            else:
                proposal.status = "REJECTED"
        return self._serialize_proposal(proposal)

    def resolve_merchant_appeal(
        self,
        merchant_id: str,
        threshold_override: float | None = None,
    ) -> dict[str, Any]:
        tracer = init_telemetry()
        with tracer.start_as_current_span("workflow.merchant_appeal") as span:
            span.set_attribute("causalbound.span_category", "workflow")
            mcp_out = solve_merchant_recourse_tool(
                merchant_id=merchant_id,
                threshold_override=threshold_override,
                include_greedy_comparison=True,
                data_dir=self.data_dir,
            )
            card = self.appeals.explain_recourse(mcp_out)
            return {
                "recourse": mcp_out["recourse"],
                "greedy_1d_baseline": mcp_out["greedy_1d_baseline"],
                "recovery_card": card,
            }

    def list_ledger(self) -> list[dict[str, Any]]:
        return list(self._deployed_ledger)

    @staticmethod
    def _serialize_proposal(p: PendingPolicyProposal) -> dict[str, Any]:
        return {
            "proposal_id": p.proposal_id,
            "created_at_utc": p.created_at_utc,
            "operator_prompt": p.operator_prompt,
            "compiled_spec": p.compiled_spec,
            "solver_result": p.solver_result,
            "status": p.status,
            "escalation_analysis": p.escalation_analysis,
            "approved_by": p.approved_by,
            "deployed_at_utc": p.deployed_at_utc,
        }
