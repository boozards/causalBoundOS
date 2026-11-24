# CausalBound Discovery Brief & SLA Contract

## Client Profile: Horizon Global Payments (Tier-1 Merchant Acquirer)
- **Scale:** 16 merchant risk slices spanning 4 global regions (NA, EU, APAC, LATAM) and 4 merchant operational tiers (micro, smb, mid_market, enterprise).
- **Transaction Flow:** $42B annual processed volume across 450,000 active merchant accounts.
- **Pain Points:**
  1. *Uncoordinated Policy Drift:* Independent regional teams set local false-ban thresholds, repeatedly blowing global fraud loss budgets and causing discriminatory merchant ban rates across geographical corridors.
  2. *Hallucinated Recourse Advice:* Under merchant appeals, LLM-generated operational guidance recommended impossible steps (e.g., retroactively altering account tenure) or expensive non-optimal actions that failed to unblock merchants.
  3. *Audit & Regulatory Scrutiny:* Regulators require mathematical proof of pairwise fairness bounds ($\le \delta$) across risk slices and deterministic reproducibility of policy changes.

---

## Service Level Agreements (SLAs) & Numerical Gates

| SLA / Metric | Target | Failure Action |
| :--- | :--- | :--- |
| **Engine A Policy Solver Latency** | $< 5.0\text{s}$ for 16 coupled slices | Revert to fallback nominal policy, alert risk ops |
| **Engine B Causal Recourse Latency** | $< 1.5\text{s}$ per merchant ($2.2 \times 10^7$ space) | Queue async recalculation |
| **Deterministic Tie-Breaking** | $100\%$ byte-for-byte identical across runs | Automatic CI block |
| **Fairness Violation Rate** | $0.0\%$ (Strict $\le \delta$ guarantee) | Block policy deployment (`FAILED_AUDIT`) |
| **Global FP Budget Adherence** | $\sum \text{FP}_m \le B_{\text{global\_fp}}$ ($100\%$ enforcement) | Trigger risk operator escalation |
| **SCM DAG Residual Bound** | $\le 10^{-4}$ max absolute residual | Reject recourse proposal |
| **Recourse Optimality Gap** | $0.00\%$ ($C \le C^* + \text{tol}_{\text{opt}}$) | Reject recourse recommendation |
