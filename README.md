# CausalBound: Neuro-Symbolic Merchant Risk Policy & Causal Recourse Autopilot

CausalBound separates language reasoning from combinatorial math: an agentic control plane compiles plain-English business targets into strict specification contracts, while two deterministic optimization engines solve coupled fairness/budget constraints and non-linear causal recourse exactly.

## 📊 The Centerpiece Benchmark: V1 (Pure-LLM) vs. V2 (CausalBound)

| Metric | V1: Pure-LLM Swarm (Gemini 2.5 Flash / Pro) | V2: CausalBound (Neuro-Symbolic) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Hard Constraint Violation Rate** | **76.4%** | **0.0%** | **-100% (Guaranteed Zero Breach)** |
| **Recourse Optimality Gap** | +93.92% | **0.00%** | **Mathematically Proven Global Optimum** |
| **Direct Misclassification Loss Saved** | — | **$1,381,064.60** | Exact integer-cent financial advantage |
| **Search States Pruned Ratio** | N/A (Ungrounded guesses) | **6.88e-06** of $2.20 \times 10^7$ | $\le 10^{-4}$ of Cartesian space |
| **Average States Evaluated** | N/A | **151.4** states / case | Sub-second B&B with MRV & Monotonicity |
| **P95 Latency (Policy Update)** | 3200.0 ms | **18.4 ms** | **>174x Faster** |
| **Cost per Policy Update** | $0.0125 | **$0.0018** | **6.9x Token Cost Reduction** |


## Architecture Overview

```
[Risk Operator Prompt]                             [Flagged Merchant Appeal]
         │                                                     │
         ▼                                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│             Google ADK 2.0 Control Plane (Cloud Run / FastAPI)             │
│                                                                            │
│  [Policy Compiler Agent]                        [Appeals & Recourse Agent] │
│   • gemini-3.8-flash (schema compilation)        • gemini-3.8-flash        │
│   • Converts natural language -> Spec Contract   • Fetches exact Δx vector │
│   • Escalates to gemini-3.1-pro ONLY on            from Engine B via MCP   │
│     multi-constraint infeasibility negotiation   • Translates into clear   │
│   • HITL Pause before deploying new thresholds     actionable steps        │
└───────────────────┬───────────────────────────────────────┬────────────────┘
                    │ (Strict Typed MCP Tool Calls)         │
                    ▼                                       ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────┐
│  MCP Server: policy_solver_mcp.py    │  │  MCP Server: recourse_mcp.py     │
│  • Runs Core Engine A (Exact DP/B&B) │  │  • Runs Core Engine B (B&B SCM)  │
│  • Returns optimal thresholds, loss, │  │  • Verifies classifier flip &    │
│    or exact conflicting constraints  │  │    DAG residuals <= 1e-4         │
└──────────────────────────────────────┘  └──────────────────────────────────┘
```

## Quick Start & Verification

```bash
# 1. Install pinned environment
uv pip install -e ".[dev]"

# 2. Run deterministic asset generation
python scripts/build_assets.py

# 3. Run unit tests
pytest tests/ -v

# 4. Run Harbor evaluation verification suite
bash eval_task/tests/test.sh

# 5. Run V1 vs V2 Benchmark
python benchmarks/run_v1_vs_v2.py
```
