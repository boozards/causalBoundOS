# ADR 002: Pairwise Fairness Window Reduction & MCKP Dynamic Programming for Engine A

## Context
Engine A optimizes thresholds across 16 slices subject to per-slice FPR, pairwise fairness $\forall (i, j): |\text{TPR}_i - \text{TPR}_j| \le \delta$, and global FP budget $\sum_{m=1}^{16} \text{FP}_m \le B_{\text{global\_fp}}$. Naive Cartesian product of 19 thresholds across 16 slices yields $19^{16} \approx 2.9 \times 10^{20}$ states.

## Decision
1. **Window Reduction:**
   The pairwise constraint is mathematically identical to $\max_m \text{TPR}_m - \min_m \text{TPR}_m \le \delta$.
   Iterating over all candidate distinct TPR values $L$ as the lower bound restricts each slice $m$ to candidates with $\text{TPR}_m(t_m) \in [L, L + \delta]$.
2. **Multiple-Choice Knapsack DP:**
   For each fixed valid window $[L, L + \delta]$, each slice $m$ selects exactly one threshold from its valid candidate subset. This is an instance of the Multiple-Choice Knapsack Problem (MCKP) where knapsack capacity is $B_{\text{global\_fp}}$ and items contribute (FP, loss_cents).
3. **Pareto Dominance Pruning:**
   Within each slice, equal FP candidates are filtered to the best 3-tier candidate. Across slices, DP states with higher FP and worse-or-equal 3-tier objective key are pruned.

## Consequences
- Guaranteed exact global optimum in $< 50\text{ms}$.
- Fully satisfies 3-tier tie-breaking deterministically.
