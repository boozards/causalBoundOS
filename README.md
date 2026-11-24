# CausalBound

**Neuro-Symbolic Merchant Risk Policy & Causal Recourse Autopilot**

CausalBound separates natural-language policy reasoning from exact combinatorial optimization. Instead of letting an LLM guess fraud thresholds or hallucinate account-recovery advice, an agentic control plane compiles plain-English risk directives into strict specification contracts, while two deterministic solvers compute globally optimal 16-slice thresholds and causally valid non-linear recourse vectors.

---

## Architecture

```text
[Risk Operator Prompt]                             [Flagged Merchant Appeal]
         │                                                     │
         ▼                                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│             Google ADK 2.0 Control Plane (FastAPI / Cloud Run)             │
│                                                                            │
│  [Policy Compiler Agent]                        [Appeals & Recourse Agent] │
│   • CAUSALBOUND_FAST_MODEL (Spec compile)        • CAUSALBOUND_FAST_MODEL  │
│   • Escalates to CAUSALBOUND_PRO_MODEL ONLY      • Fetches exact Δx vector │
│     on multi-constraint infeasibility              from Engine B via MCP   │
│   • HITL Pause before ledger deployment          • Drafts verified steps   │
└───────────────────┬───────────────────────────────────────┬────────────────┘
                    │ (Strict Typed MCP Tool Calls)         │
                    ▼                                       ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────┐
│  MCP Server: policy_solver_mcp.py    │  │  MCP Server: recourse_mcp.py     │
│  • Core Engine A (Exact MCKP DP)     │  │  • Core Engine B (Exact PQ B&B)  │
│  • 16 slices, exact integer cents,   │  │  • 2.20e7 space/merchant,        │
│    pairwise TPR delta, global FP cap │  │    non-linear SCM DAG <= 1e-4    │
└───────────────────┬──────────────────┘  └─────────────────┬────────────────┘
                    │                                       │
         ┌──────────┴───────────────────┬───────────────────┴──────────┐
         ▼                              ▼                              ▼
[Postgres Policy Ledger]    [OpenTelemetry gen_ai.*]      [Next.js 15 Risk War Room]
- Versioned policy runs     • TTFT, tokens/sec,           • Warm editorial light UI
- Immutable audit trail       LLM-ms vs Solver-ms,        • Live global-FP sliders
                              cost-per-policy-update      • Interactive SCM DAG view
```

---

## Core Mathematical Engines

### Core Engine A: Coupled 16-Slice Threshold Optimizer (`engine_a/slice_optimizer.py`)

Minimizes exact integer-cent global financial loss across 16 merchant slices ($4\text{ regions} \times 4\text{ tiers}$):

$$\min_{\mathbf{t}} \sum_{m=1}^{16} \left( c_{\text{FP}, m} \cdot \text{FP}_m(t_m) + c_{\text{FN}, m} \cdot \text{FN}_m(t_m) \right)$$

Subject simultaneously to:
- **Per-Slice Calibration Gate:** $\text{ECE}_m \le \text{max\_ece}_m$ (10 equal-width bins; $N < 5 \implies \text{ECE} = 0.0$)
- **Per-Slice False-Ban Ceiling:** $\text{FPR}_m(t_m) \le \text{max\_fpr}_m$ (`neg_total == 0` $\implies \text{FPR} = 0.0$)
- **Pairwise Fairness Delta:** $\lvert \text{TPR}_i(t_i) - \text{TPR}_j(t_j) \rvert \le \delta$ (`pos_total == 0` $\implies \text{TPR} = 1.0$)
- **Non-Separable Global False-Ban Budget:** $\sum_{m=1}^{16} \text{FP}_m(t_m) \le B_{\text{global\_fp}}$

**Exact solver reduction:** enumerates distinct candidate TPR lower bounds $L$ over $[L, L + \delta]$ windows and solves each feasible window via Multiple-Choice Knapsack DP with exact `Fraction` rates, integer cents, 3-tier Pareto-dominance pruning, and deterministic tie-breaking.

### Core Engine B: Non-Linear Causal Recourse Solver (`engine_b/bnb_recourse.py`)

For any blocked merchant ($f(\mathbf{x}) \ge t_m$), computes the globally optimal discrete intervention $\Delta \mathbf{x}$ that flips a pinned monotonic `HistGradientBoostingClassifier` (`scikit-learn==1.6.1`) across a 7-feature non-linear SCM DAG (`exp`, `log1p`, `sqrt`):

- **Calibrated Action Space:** $3 \times 401 \times 101 \times 181 = 21{,}992{,}043 \approx 2.20 \times 10^7$ candidates per merchant (`account_tenure_months` strictly immutable)
- **Coupled Objective (Defeats 1D Greedy):** $\text{Cost}(\mathbf{k}) = \sum w_i k_i + \sum \alpha_i k_i^2 + \sum_{i < j} \beta_{ij} k_i k_j$ with $\beta_{ij} > 0$
- **Exact Search:** Priority-Queue Branch-and-Bound (`heapq`) with multi-strategy greedy upper-bound seeding, closed-form quadratic domain tightening, monotonic optimistic feasibility pruning, MRV branching, `@lru_cache` memoization, and vectorized NumPy leaf sweeps
- **Verified Tolerances:** SCM DAG residual $\le 10^{-4}$; one-sided optimality bound $C \le C^* + 0.05$ ($< \min_i \text{step cost} = 1.0$)

---

## Benchmark: V1 (Pure-LLM / 1D Baseline) vs. V2 (CausalBound)

Evaluated across **25 coupled 16-slice policy recalibration scenarios** and **30 blocked merchant non-linear SCM recourse cases** (`2.20e7` discrete candidates/merchant). Run `python benchmarks/run_v1_vs_v2.py` to regenerate:

| Metric | V1: Pure-LLM / 1D Baseline | V2: CausalBound (Neuro-Symbolic) |
| :--- | :---: | :---: |
| **Policy Hard Constraint Violation Rate** | `100.00%` | **`0.00%`** |
| **Recourse 1D Infeasibility / Failure Rate** | `0.00%` | **`0.00%`** |
| **Combined Constraint Violation Rate** | `45.45%` | **`0.00%`** |
| **Mean Optimality Gap (Feasible Cases)** | `153.52%` | **`0.00%`** |
| **Exact Dollar Loss Saved (Feasible Delta)** | `$0.00` | **`$0.00`** |
| **Engine B States Visited / `2.20e7` Space** | `N/A (1D heuristic)` | **`6.88e-06`** |
| **P95 Policy Update Latency (ms)** | `685.0 ms` | **`29.2 ms`** |
| **Avg Cost per Policy Update ($)** | `$0.000530` | **`$0.000051`** |

---

## Harbor Verification Suite (`eval_task/`)

CausalBound packages its entire evaluation harness as a runnable non-root Harbor benchmark (`1 vCPU`, `2048 MB RAM`, `UID 1000`, `allow_internet = false`, `300s` verifier timeout, `< 30s` Oracle budget):

- **Five-Stage Verification Ladder:** Existence $\to$ Strict Key-Set & Finiteness $\to$ Hard Invariants & SCM Residuals $\to$ One-Sided Optimality $\to$ Exact 3-Tier Match
- **Audits A–D:** Anti-Slack (removing `max_global_fp`, `delta`, or `max_fpr` breaks optimality), Anti-Collapse ($\ge 2$ equal-cost candidates force Tier-2/Tier-3 tie-break), Anti-Memorization (runtime-generated slices), and Failure-State Bi-Directional coverage (`FAILED_AUDIT` + strict `null` selections)
- **Anti-Tamper Lifecycle:** In-memory Load-Purge-Verify ground-truth scrubbing, raw binary `"rb"` forbidden-token scan, and 3-tier regime verification (Nop `0.0`, Oracle `1.0`, $\pm 1$ Perturbation `0.0`)

---

## Quickstart

### 1. Local Python Setup & Lockstep Asset Build

```bash
uv venv --python 3.13 && source .venv/bin/activate
uv pip install -e ".[dev,control-plane]"

# Generate deterministic assets (seed=42), sync eval_task/, and verify Anti-Slack bindings
python scripts/sync_eval_task.py
```

### 2. Run Parity Tests, Harbor Regimes & Benchmark

```bash
# Engine B brute-force parity, tie-breaking, determinism & full-scale speed (< 10s)
pytest -v tests/test_engine_b_parity.py

# 3-Tier Harbor regimes: Nop (fail) / Oracle (pass < 30s) / +1 Perturbation (fail)
python eval_task/verify_regimes.py

# V1 vs. V2 benchmark report -> benchmarks/latest_report.json & BENCHMARK_TABLE.md
python benchmarks/run_v1_vs_v2.py
```

### 3. Verify Inside Fresh Non-Root Container

```bash
docker build -f eval_task/environment/Dockerfile -t causalbound-eval eval_task
docker run --rm --cpus=1 --memory=2048m --network=none \
  -v "$(pwd)/eval_task/solution:/solution:ro" \
  -v "$(pwd)/eval_task/tests:/tests:ro" \
  causalbound-eval bash -c "bash /solution/solve.sh && bash /tests/test.sh"
```

### 4. Launch Full Stack (FastAPI + Postgres + Next.js 15 Risk War Room)

```bash
docker compose up --build
# Risk War Room UI : http://localhost:3000
# FastAPI OpenAPI  : http://localhost:8000/docs
```

---

## Repository Structure

```text
causalbound/
├── AGENTS.md                        # Master operational & verification contract
├── pyproject.toml                   # Exact minor-version pinned dependencies (Python 3.13)
├── docs/
│   ├── SPEC_CONTRACT.md             # Locked schemas, fallbacks, tie-breakers, SCM equations
│   ├── DISCOVERY.md                 # Client discovery brief, failure modes & SLAs
│   └── adr/                         # Architecture Decision Records (0001–0003)
├── src/causalbound/
│   ├── schemas.py                   # Strict Pydantic v2 contracts & key-set guards
│   ├── scm.py                       # Non-linear SCM DAG & coupled quadratic cost math
│   ├── engine_a/slice_optimizer.py  # Exact 16-slice coupled MCKP DP optimizer
│   ├── engine_b/                    # Greedy UB seeder + Priority-Queue B&B solver
│   ├── mcp_servers/                 # Strict typed MCP servers for Engine A & Engine B
│   ├── agents/                      # ADK 2.0 Policy Compiler, Appeals Agent, HITL Workflow
│   ├── telemetry.py                 # OpenTelemetry gen_ai.* & solver waterfall collector
│   └── api.py                       # FastAPI control plane + SSE telemetry stream
├── scripts/
│   ├── build_assets.py              # Single-source-of-truth generator (seed=42)
│   ├── sync_eval_task.py            # Lockstep asset & expected.json synchronizer
│   └── deploy_cloud_run.sh          # Google Cloud Run deployment script
├── eval_task/                       # Runnable non-root Harbor benchmark task
├── benchmarks/run_v1_vs_v2.py       # Pure-LLM vs. CausalBound benchmark suite
├── frontend/                        # Next.js 15 editorial light-mode Risk War Room UI
├── docker-compose.yml               # Local API + Postgres + Frontend stack
└── .github/workflows/ci.yml         # CI + non-root Harbor container verification
```
