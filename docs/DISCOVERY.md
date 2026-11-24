# Client Discovery Brief & Production SLAs (`DISCOVERY.md`)

**Engagement Context:** Mid-to-Enterprise Payment Facilitator & Merchant Acquiring Risk Desk  
**Scope:** Automated multi-region fraud threshold recalibration (16 operational slices) and verifiable account-recovery recourse for blocked merchants.

---

## 1. Current Operational Pain Points

1. **Silent Global False-Ban Overruns:** Regional risk managers tune thresholds independently by geography (`NA`, `EU`, `APAC`, `LATAM`) and merchant tier (`micro`, `smb`, `mid_market`, `enterprise`). During localized fraud waves, independent 1D tuning blows past the platform's global manual-review and false-ban capacity (`max_global_fp`), triggering merchant churn and support SLA breaches.
2. **Cross-Region Fairness Drift:** Lowering thresholds aggressively in `LATAM` or `APAC` while keeping `NA` enterprise thresholds relaxed creates wide True Positive Rate (TPR) disparities across merchant cohorts, failing internal fair-lending and payment-network compliance audits (`|TPR_i - TPR_j| <= delta`).
3. **Ungrounded LLM Policy & Appeal Drafts:** Early internal prototypes asked frontier LLMs to propose threshold matrices and draft merchant appeal instructions directly. In testing, pure-LLM workflows violated coupled global constraints on > 35% of tight-budget scenarios and frequently recommended physically impossible or non-flipping recovery steps (e.g., asking merchants to increase `account_tenure_months` or post reserves that fail to move the downstream risk score below the slice threshold).
4. **Unreproducible Model & Calibration Drift:** Policy backtests lacked strict `ECE` gates before threshold selection and suffered subtle score shifts when pickled `scikit-learn` models were deployed across mismatched Python/library minor versions.

---

## 2. Target System Requirements & SLAs

| Capability | Production SLA / Hard Requirement | Verification Mechanism |
| :--- | :--- | :--- |
| **Policy Constraint Adherence** | `0.00%` violation rate across `max_ece`, `max_fpr`, pairwise `delta`, and `max_global_fp` | Exact Engine A DP + Stage 3 Verifier |
| **Financial Precision** | Zero floating-point drift on dollar loss accounting | Exact integer cents (`*_cents: int`) |
| **Infeasibility Transparency** | Never silently relax a hard constraint; halt with `FAILED_AUDIT`, `null` selections, and exact minimal relaxation options | Audit D + `diagnose_infeasibility_tool` |
| **Recourse Actionability** | 100% immutable `account_tenure_months`, valid monotonic steps, SCM DAG residual `<= 1e-4`, verified score flip `f(x') < t_m` | Engine B Exact B&B + Stage 3–5 Verifier |
| **Compute & Latency Ceiling** | Policy solve `< 150ms`; 30-merchant recourse batch `< 10s` under `1 vCPU / 2048 MB RAM` | Non-root Harbor container gate |
| **Human-in-the-Loop Governance** | No natural-language policy proposal enters production without explicit reviewer approval and versioned ledger entry | ADK Workflow `AWAITING_HITL_APPROVAL` gate |
| **Observability** | Per-request OpenTelemetry breakdown of LLM TTFT, tokens/sec, USD cost, and solver latency | `telemetry.py` + War Room OTel Waterfall |

---

## 3. Out-of-Scope Boundaries (v1)

- Live payment-gateway webhook ingestion and real-time card-network chargeback settlement feeds (simulated deterministically via `scripts/build_assets.py` for reproducible benchmarking).
- Multi-currency FX conversion (all costs normalized to USD integer cents in `SPEC_CONTRACT.md`).
- Automatic classifier retraining in the online serving loop (model binary `model.joblib` is version-locked and regenerated exclusively via lockstep pipeline).
