# ADR 0003: Neuro-Symbolic ADK Routing, Conditional Pro Escalation & HITL Ledger Gate

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Risk operations require natural-language usability ("cap global false bans at 92 during a 1.6x LATAM fraud wave") alongside strict mathematical guarantees (`0.00%` constraint violations, zero hallucinated numbers, auditable deployments). Routing every request to a large reasoning model increases latency and per-update cost without solving coupled combinatorial constraints accurately.

## Decision

1. **Strict Neuro-Symbolic Boundary via Typed MCP Servers:** LLMs are strictly prohibited from choosing thresholds or inventing recourse numbers. `PolicyCompilerAgent` outputs only a Pydantic-validated `CompiledPolicyIntent` (`extra="forbid"`), and `AppealsRecourseAgent` narrates only the verified output of `recourse_mcp.py`.
2. **Tiered Model Routing (Externalized Env Vars):**
   - `CAUSALBOUND_FAST_MODEL` (default `gemini-3.8-flash`) handles routine NL-to-schema compilation and plain-language merchant recovery card drafting.
   - `CAUSALBOUND_PRO_MODEL` (default `gemini-3.1-pro`) is invoked **only** when Engine A returns `FAILED_AUDIT` and `diagnose_infeasibility_tool` supplies verified single-lever relaxation trade-offs requiring executive negotiation.
3. **Mandatory HITL Pause Before Deployment:** Every feasible policy proposal enters `AWAITING_HITL_APPROVAL`. Thresholds are written to the versioned policy ledger only after an explicit reviewer approval call (`POST /api/policy/proposals/{id}/hitl`), recorded with OpenTelemetry `workflow.hitl_gate` spans.
4. **Offline Deterministic Parity:** Both agents provide deterministic offline parsers/formatters when `GOOGLE_API_KEY` is unset so CI pipelines, Harbor containers, and local demos run end-to-end without network dependencies.

## Consequences

- **Positive:** Keeps routine policy updates sub-second and low-cost, guarantees zero constraint violations regardless of LLM stochasticity, and isolates OpenTelemetry `gen_ai.*` metrics cleanly from solver spans.
- **Trade-off:** Operators expressing completely novel constraint classes outside `CompiledPolicyIntent` require an explicit schema and Engine A contract extension rather than ad-hoc prompt injection—an intentional safety barrier.
