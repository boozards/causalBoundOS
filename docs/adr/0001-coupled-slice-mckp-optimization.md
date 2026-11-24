# ADR 0001: Exact Coupled 16-Slice Optimization via `[L, L + delta]` Windowing & MCKP DP

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Core Engine A must select thresholds $\mathbf{t} = (t_1, \dots, t_{16})$ minimizing exact financial loss subject to per-slice `max_ece` and `max_fpr`, global false-ban budget $\sum_m \text{FP}_m(t_m) \le B_{\text{global\_fp}}$, and pairwise fairness $|\text{TPR}_i(t_i) - \text{TPR}_j(t_j)| \le \delta$ for all pairs $(i, j)$.

Naive approaches fail:
- **Independent 1D per-slice selection** ignores both $B_{\text{global\_fp}}$ and pairwise $\delta$, violating hard constraints whenever either coupling is active.
- **Unpruned Cartesian search** over 16 slices $\times$ 19 thresholds requires $19^{16} \approx 2.88 \times 10^{20}$ states.
- **General-purpose MILP solvers** introduce floating-point solver tolerances that complicate exact 3-tier deterministic tie-breaking over integer cents and exact rational `TPR`/`FPR` fractions.

## Decision

1. **Exact Fairness Window Reduction:** Observe that $|\text{TPR}_i - \text{TPR}_j| \le \delta \; \forall (i, j) \iff \max_m \text{TPR}_m - \min_m \text{TPR}_m \le \delta$. Every optimal feasible assignment has its minimum slice TPR equal to some distinct candidate TPR value $L$. Enumerating all distinct candidate `Fraction` TPR values $L$ (deduplicated by identical per-slice candidate window signatures) guarantees 100% exact coverage of all fairness-feasible combinations.
2. **Exact Multiple-Choice Knapsack DP:** Inside each window $[L, L + \delta]$, slices are coupled solely by integer $\text{FP}_m \ge 0$. Deduplicating per-slice candidates by exact `FP` count (keeping minimum `(loss_cents, threshold_dec)`) and sweeping across sorted `slice_id`s with suffix-minimum-FP pruning and 3-tier Pareto-dominance pruning solves the joint problem in milliseconds.
3. **Exact Arithmetic:** All financial losses use Python arbitrary-precision `int` cents; all `TPR`, `FPR`, and `delta` comparisons use `fractions.Fraction`; threshold sums use `decimal.Decimal`.

## Consequences

- **Positive:** 100% exact global optimality and exact 3-tier determinism with zero external MILP binary dependencies; solves 16 slices in `< 50ms` on 1 vCPU.
- **Trade-off:** Runtime scales linearly with `max_global_fp` and the number of distinct candidate TPR windows. If future deployments exceed thousands of slices or FP budgets $> 10^5$, a branch-and-price or exact integer-programming backend would be warranted.
