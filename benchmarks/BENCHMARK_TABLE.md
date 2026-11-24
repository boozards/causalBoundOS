# CausalBound Benchmark: V1 (Pure-LLM / 1D Baseline) vs. V2 (CausalBound Neuro-Symbolic)

Evaluated across **25 coupled 16-slice policy recalibration scenarios** and **30 blocked merchant non-linear SCM recourse cases** (`2.20e7` discrete candidates/merchant).

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
