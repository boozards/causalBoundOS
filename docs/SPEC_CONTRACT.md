# CausalBound Locked Specification Contract (`SPEC_CONTRACT.md`)

> **Status:** LOCKED — Every boundary, schema, fallback, tie-breaker, and mathematical equation below is binding across Engines A & B, MCP servers, agents, and the Harbor verifier suite.

---

## 1. Environment & Resource Profile

| Parameter | Locked Value |
| :--- | :--- |
| Python Version | `3.13.x` (`>=3.13,<3.14`) |
| Pinned ML/Math Stack | `scikit-learn==1.6.1`, `numpy==2.2.3`, `pandas==2.2.3`, `scipy==1.15.2`, `joblib==1.4.2` |
| Container Profile | Non-root `UID 1000`, `1 vCPU`, `2048 MB RAM`, no internet in eval |
| Timeout Budgets | Verifier: `300.0s`; Reference Oracle: `< 30.0s` (`< 10%`) |
| Random Seed | `42` across all deterministic generators and model training |
| Timestamps | Strict UTC ISO-8601 (`%Y-%m-%dT%H:%M:%S.%fZ`, 3-digit rounded ms), synthetic values centered at `.500` |

---

## 2. Core Engine A: Coupled 16-Slice Threshold Optimizer

### 2.1 Inputs & Defaults
- **Slices (`slices`):** List of slice payloads (`4 regions × 4 tiers = 16 slices` in production). Unsorted input lists MUST be sorted ascending by `slice_id` before processing.
  - Regions: `NA`, `EU`, `APAC`, `LATAM`
  - Tiers: `micro`, `smb`, `mid_market`, `enterprise`
  - `slice_id` format: `"{region}_{tier}"` (e.g., `"APAC_enterprise"`)
- **Per-Slice Required Fields (No Implicit Defaults):**
  - `slice_id: str`
  - `fp_cost_cents: int` ($\ge 0$, exact integer cents per false positive / false merchant ban)
  - `fn_cost_cents: int` ($\ge 0$, exact integer cents per false negative / missed fraud)
  - `max_ece: float` (finite, $\in [0.0, 1.0]$)
  - `max_fpr: float` (finite, $\in [0.0, 1.0]$)
  - `samples: list[{sample_id: str, score: float, label: Literal[0, 1], timestamp_utc: str}]`
- **Global Required Fields:**
  - `policy_id: str`
  - `evaluated_at_utc: str`
  - `candidate_thresholds: list[float]` — default grid `[0.05, 0.10, ..., 0.95]` (19 values); solver deduplicates and sorts ascending using exact `Decimal(str(t))`.
  - `pairwise_fairness_delta: float` ($\delta \in [0.0, 1.0]$)
  - `max_global_fp: int` ($B_{\text{global\_fp}} \ge 0$)

### 2.2 Decision Rule, Metrics & Mandatory Edge-Case Fallbacks
- **Classification Rule:** Sample $k$ with risk score $s_k \in [0, 1]$ is **blocked (predicted positive)** iff $s_k \ge t_m$.
- **Score Mid-Boundary Invariant (Synthetic Assets):** All generated scores satisfy $\min_{t \in \mathcal{T} \cup \{0, 1\}} |s_k - t| \ge 0.005$.
- **Confusion Counts per Slice $m$ at Threshold $t_m$:**
  - $\text{TP}_m = \sum [s_k \ge t_m \land y_k = 1]$, $\quad \text{FP}_m = \sum [s_k \ge t_m \land y_k = 0]$
  - $\text{FN}_m = \sum [s_k < t_m \land y_k = 1]$, $\quad \text{TN}_m = \sum [s_k < t_m \land y_k = 0]$
  - $\text{pos\_total}_m = \text{TP}_m + \text{FN}_m$, $\quad \text{neg\_total}_m = \text{FP}_m + \text{TN}_m$, $\quad N_m = \text{pos\_total}_m + \text{neg\_total}_m$
- **Zero-Division & Small-$N$ Fallbacks (Exact `Fraction` for TPR/FPR):**
  - If $\text{pos\_total}_m == 0 \implies \text{TPR}_m = 1.0$ (`Fraction(1, 1)`). Otherwise $\text{TPR}_m = \text{TP}_m / \text{pos\_total}_m$.
  - If $\text{neg\_total}_m == 0 \implies \text{FPR}_m = 0.0$ (`Fraction(0, 1)`). Otherwise $\text{FPR}_m = \text{FP}_m / \text{neg\_total}_m$.
  - If $N_m < 5 \implies \text{ECE}_m = 0.0$. Otherwise compute 10 equal-width bins $[0.0, 0.1), \dots, [0.8, 0.9), [0.9, 1.0]$ (last bin right-inclusive):
    $$\text{ECE}_m = \sum_{b=0}^{9} \frac{n_b}{N_m} \left| \bar{y}_b - \bar{s}_b \right|$$
  - Empty slices ($N_m == 0$) inherit all three fallbacks (`TPR = 1.0`, `FPR = 0.0`, `ECE = 0.0`, `FP = 0`, `FN = 0`, `loss_cents = 0`) and participate in pairwise fairness.

### 2.3 Objective & Coupled Hard Constraints
$$\min_{\mathbf{t}} \sum_{m=1}^{M} \left( c_{\text{FP}, m} \cdot \text{FP}_m(t_m) + c_{\text{FN}, m} \cdot \text{FN}_m(t_m) \right) \quad \text{(exact integer cents)}$$

Subject to:
1. **Per-Slice Calibration Gate:** $\text{ECE}_m \le \text{max\_ece}_m$ for every slice $m$. Any single violation triggers global `FAILED_AUDIT`.
2. **Per-Slice False-Ban Ceiling:** $\text{FPR}_m(t_m) \le \text{max\_fpr}_m$ (exact `Fraction` comparison).
3. **Pairwise Fairness Delta:** $|\text{TPR}_i(t_i) - \text{TPR}_j(t_j)| \le \delta$ for all pairs $(i, j)$ (exact `Fraction` comparison, equivalent to $\max_m \text{TPR}_m - \min_m \text{TPR}_m \le \delta$).
4. **Global False-Ban Budget:** $\sum_{m=1}^{M} \text{FP}_m(t_m) \le B_{\text{global\_fp}}$.

### 2.4 Engine A 3-Tier Deterministic Tie-Breaking Ladder
1. **Tier 1:** Minimum exact integer-cent `total_loss_cents`.
2. **Tier 2:** Minimum exact `Decimal` threshold sum $\sum_{m=1}^{M} t_m$.
3. **Tier 3:** Lexicographically smallest tuple $(t_1, t_2, \dots, t_M)$ ordered by ascending `slice_id`.

### 2.5 Engine A Exact Output Schema (Strict Key-Set Invariance)
Top-level keys MUST be **exactly**:
```json
{
  "policy_id": "str",
  "evaluated_at_utc": "str",
  "audit_status": "PASSED | FAILED_AUDIT",
  "selected_thresholds": {"<slice_id>": 0.55} | null,
  "total_loss_cents": 1234500 | null,
  "total_fp": 42 | null,
  "total_fn": 19 | null,
  "global_tpr_spread": 0.08 | null,
  "slice_ece": {"<slice_id>": 0.021},
  "slice_metrics": {
    "<slice_id>": {
      "threshold": 0.55,
      "fp": 3,
      "fn": 1,
      "tp": 18,
      "tn": 110,
      "tpr": 0.9473684210526315,
      "fpr": 0.02654867256637168,
      "loss_cents": 85000
    }
  } | null,
  "violated_constraints": []
}
```

When `audit_status == "FAILED_AUDIT"`: `selected_thresholds`, `total_loss_cents`, `total_fp`, `total_fn`, `global_tpr_spread`, and `slice_metrics` MUST all be strictly `null`; `slice_ece` and `violated_constraints` (sorted ascending, non-empty) MUST be populated.

Canonical `violated_constraints` codes:
- `"SLICE_ECE_EXCEEDED:<slice_id>"`
- `"SLICE_FPR_INFEASIBLE:<slice_id>"`
- `"PAIRWISE_FAIRNESS_INFEASIBLE"`
- `"GLOBAL_FP_BUDGET_EXCEEDED"`

---

## 3. Core Engine B: Non-Linear Causal Recourse Solver

### 3.1 Classifier & SCM DAG Specification (7 Features: 5 Roots + 2 Derived Nodes)
Feature vector order supplied to pinned `HistGradientBoostingClassifier` (`scikit-learn==1.6.1`):
1. `account_tenure_months` — immutable root (`monotonic_cst = -1`)
2. `kyc_tier` — actionable root, integer (`monotonic_cst = -1`)
3. `reserve_deposit_usd` — actionable root, USD (`monotonic_cst = -1`)
4. `dispute_rate_bps` — actionable root, basis points lowered by `dispute_reduction_bps` (`monotonic_cst = +1`)
5. `settlement_hold_days` — actionable root, integer days (`monotonic_cst = -1`)
6. `effective_risk_ratio` — derived non-linear node (`monotonic_cst = +1`)
7. `liquidity_coverage_index` — derived non-linear node (`monotonic_cst = -1`)

#### Structural Equations (Non-Linear Forward Propagation):
$$\text{effective\_risk\_ratio} = \frac{\exp(0.0025 \cdot \text{dispute\_rate\_bps})}{1 + 0.35 \cdot \text{kyc\_tier} + 0.15 \cdot \ln(1 + \text{account\_tenure\_months}) + 0.08 \cdot \sqrt{\text{settlement\_hold\_days}}}$$

$$\text{liquidity\_coverage\_index} = \ln(1 + \text{reserve\_deposit\_usd}) + 0.6 \cdot \sqrt{\text{settlement\_hold\_days} + 1} + 0.4 \cdot \text{kyc\_tier} - 0.12 \cdot \sqrt{\text{dispute\_rate\_bps}}$$

### 3.2 Calibrated Discrete Action Space ($2.20 \times 10^7$ per Instance)

| Action Variable ($i$) | Effect on Root Feature | Delta Bounds | Step Size | Discrete Steps $k_i$ | Cardinality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `kyc_tier_delta` | $\text{kyc\_tier}' = \text{kyc\_tier} + \Delta$ | $[0, 2]$ | $1$ | $0..2$ | $3$ |
| `reserve_deposit_usd_delta` | $\text{reserve\_deposit\_usd}' = \text{reserve\_deposit\_usd} + \Delta$ | $[0, 100\,000]$ | $250$ | $0..400$ | $401$ |
| `dispute_reduction_bps` | $\text{dispute\_rate\_bps}' = \text{dispute\_rate\_bps} - \Delta$ | $[0, 500]$ | $5$ | $0..100$ | $101$ |
| `settlement_hold_days_delta` | $\text{settlement\_hold\_days}' = \text{settlement\_hold\_days} + \Delta$ | $[0, 180]$ | $1$ | $0..180$ | $181$ |

$$\text{Candidates per Instance} = 3 \times 401 \times 101 \times 181 = 21{,}992{,}043 \approx 2.20 \times 10^7$$

### 3.3 Step-Normalized Coupled Cost Objective & Tolerances
Let $k_i = \Delta_i / \text{step}_i \in \mathbb{Z}_{\ge 0}$ for the 4 sorted action dimensions:
$$\text{Cost}(\mathbf{k}) = \sum_{i=1}^{4} w_i k_i + \sum_{i=1}^{4} \alpha_i k_i^2 + \sum_{1 \le i < j \le 4} \beta_{ij} k_i k_j, \quad w_i > 0, \alpha_i > 0, \beta_{ij} > 0$$

All single-step costs satisfy $w_i + \alpha_i \ge 1.0$.

- **SCM DAG Residual Tolerance:** $\text{tol}_{\text{residual}} = 10^{-4}$.
- **One-Sided Optimality Tolerance:** $C_{\text{agent}} \le C^* + 0.05$ ($0.05 < 1.0 = \min_i \text{step cost}$).

#### Status Modes:
- `"ALREADY_APPROVED"`: $f(\mathbf{x}) < t_m$ already $\implies \mathbf{k} = \mathbf{0}$, `total_cost = 0.0`.
- `"PASSED"`: globally optimal flip found ($f(\mathbf{x}') < t_m$).
- `"FAILED_AUDIT"`: no feasible intervention flips $f(\mathbf{x}') < t_m \implies$ `actions = null`, `counterfactual_features = null`, `total_cost = null`, `violated_constraints = ["NO_FEASIBLE_FLIP"]`.

### 3.4 Engine B 3-Tier Deterministic Tie-Breaking Ladder
1. **Tier 1:** Minimum coupled `total_cost` (ties triggered within $|C_a - C_b| \le 10^{-9}$).
2. **Tier 2:** Minimum integer step sum $\sum_{i=1}^{4} k_i$.
3. **Tier 3:** Lexicographically smallest step tuple $(k_{\text{dispute}}, k_{\text{kyc}}, k_{\text{reserve}}, k_{\text{hold}})$ ordered by ascending action name.

### 3.5 Engine B Exact Output Schema (Strict Key-Set Invariance)
Top-level keys MUST be **exactly**:
```json
{
  "merchant_id": "str",
  "slice_id": "str",
  "audit_status": "PASSED | ALREADY_APPROVED | FAILED_AUDIT",
  "threshold": 0.55,
  "initial_risk_score": 0.68,
  "counterfactual_risk_score": 0.54 | null,
  "actions": {
    "dispute_reduction_bps": 25.0,
    "kyc_tier_delta": 1,
    "reserve_deposit_usd_delta": 1000.0,
    "settlement_hold_days_delta": 3
  } | null,
  "step_counts": {
    "dispute_reduction_bps": 5,
    "kyc_tier_delta": 1,
    "reserve_deposit_usd_delta": 4,
    "settlement_hold_days_delta": 3
  } | null,
  "counterfactual_features": {
    "account_tenure_months": 12.0,
    "kyc_tier": 2,
    "reserve_deposit_usd": 6000.0,
    "dispute_rate_bps": 50.0,
    "settlement_hold_days": 10,
    "effective_risk_ratio": 0.42,
    "liquidity_coverage_index": 7.8
  } | null,
  "total_cost": 34.52 | null,
  "dag_residuals": {
    "effective_risk_ratio": 0.0,
    "liquidity_coverage_index": 0.0
  } | null,
  "states_visited": 1420,
  "violated_constraints": []
}
```

---

## 4. Control Plane Model Externalization
Model identifiers are injected via environment variables rather than hardcoded:
- `CAUSALBOUND_FAST_MODEL` (default: `gemini-3.8-flash`)
- `CAUSALBOUND_PRO_MODEL` (default: `gemini-3.1-pro`)
