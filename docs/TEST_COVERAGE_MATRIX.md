# BoundaryOS — Bi-Directional Test Coverage Matrix

Every rule in the Spec Contract has both a **Passing Witness** and a **Rejection / Anti-Slack Witness** audited in automated tests.

| # | Stated Spec Rule | Passing Test Case | Rejection / Anti-Slack Witness | Binding? | Anti-Collapse Verified? |
|---|---|---|---|---|---|
| 1 | Per-slice FPR ceiling ($\text{FPR}_s \le \text{max\_fpr}_s$) | `test_layer3_hard_constraints_hold_on_16_slice_prod` | `test_audit1_antislack_all_three_constraints_genuinely_binding` (`enforce_slice_fpr=False` strictly lowers loss) | Yes | Yes |
| 2 | Global false-ban budget ($\sum \text{FP}_s \le B$) | `test_layer3_hard_constraints_hold_on_16_slice_prod` | `test_audit1_antislack_all_three_constraints_genuinely_binding` (`enforce_global_fp_budget=False` strictly lowers loss) | Yes | Yes |
| 3 | Cross-slice fairness delta ($\max - \min \le \Delta$) | `test_layer3_hard_constraints_hold_on_16_slice_prod` | `test_audit1_antislack_all_three_constraints_genuinely_binding` (`enforce_fairness_delta=False` strictly lowers loss) | Yes | Yes |
| 4 | Joint budget + fairness infeasibility diagnostics | `test_joint_budget_and_fairness_infeasibility_diagnostics` | Asserts `status="INFEASIBLE"`, `null` selection fields, and `min_achievable_global_fp=4 > 3` | Yes | Yes |
| 5 | 3-Tier Deterministic Tie-Breaking (Loss $\to$ FP $\to$ Lex) | `test_audit2_anticollapse_three_tier_tie_breaking_ladder` | Equal-loss distinct-FP pair forces Tier 2; equal-loss equal-FP pair forces Tier 3 | Yes | Yes (both candidates survive upstream filters) |
| 6 | Edge Cases (`N_pos=0`, `N_neg=0`, `N_neg < min_samples`) | `test_layer3_edge_cases_zero_pos_zero_neg_and_low_sample_fairness` | Low-N slice excluded from fairness delta while high-N slice enforces `delta=0.0` | Yes | Yes |
| 7 | Exact Output Key Set & Finite Number Guards | `test_layer2_exact_schema_and_finiteness_on_optimal` | `test_layer2_rejects_nan_inf_and_extra_keys` rejects `NaN` and unknown keys | Yes | Yes |
| 8 | Anti-Hardcode Global Optimality | `test_layer5_dynamic_unseen_configurations_match_bruteforce_oracle` | 5 dynamically generated unseen 4-slice configs matched against embedded brute-force oracle | Yes | Yes |
