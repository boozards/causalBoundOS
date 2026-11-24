# ADR 003: Model Externalization & Strict Typed MCP Contracts

## Context
The agent control plane translates natural language risk targets into exact JSON optimization contracts and explains recourse actions to merchants. Hardcoding model aliases leads to brittle deployments.

## Decision
1. **Model Externalization:**
   Model strings are injected via environment variables:
   - `CAUSALBOUND_FAST_MODEL` (default: `gemini-3.8-flash`)
   - `CAUSALBOUND_PRO_MODEL` (default: `gemini-3.1-pro`)
2. **Deterministic Solver Isolation:**
   Agents never execute optimization math internally. All math is delegated via typed Model Context Protocol (MCP) tool servers:
   - `policy_solver_mcp.py` for Engine A
   - `recourse_mcp.py` for Engine B
3. **Escalation Pattern:**
   Fast model handles routine schema compilation; if multi-constraint infeasibility occurs, the system escalates to Pro for policy tradeoff negotiations.

## Consequences
- Clean separation between semantic reasoning and deterministic optimization.
- Verifiable audit logs with zero hallucinated numerical outputs.
