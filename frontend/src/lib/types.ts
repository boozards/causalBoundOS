export interface SliceMetric {
  threshold: number;
  fp: number;
  fn: number;
  tp: number;
  tn: number;
  tpr: number;
  fpr: number;
  loss_cents: number;
}

export interface PolicyResult {
  policy_id: string;
  evaluated_at_utc: string;
  audit_status: "PASSED" | "FAILED_AUDIT";
  selected_thresholds: Record<string, number> | null;
  total_loss_cents: number | null;
  total_fp: number | null;
  total_fn: number | null;
  global_tpr_spread: number | null;
  slice_ece: Record<string, number>;
  slice_metrics: Record<string, SliceMetric> | null;
  violated_constraints: string[];
}

export interface RelaxationOption {
  lever: string;
  requested: number;
  minimal_feasible: number;
  resulting_loss_cents: number;
  resulting_tpr_spread?: number;
  resulting_total_fp?: number;
}

export interface SimulateResponse {
  policy_result: PolicyResult;
  infeasibility_diagnosis: {
    status: string;
    violated_constraints: string[];
    relaxation_options: RelaxationOption[];
  } | null;
}

export interface PolicyProposal {
  proposal_id: string;
  created_at_utc: string;
  operator_prompt: string;
  compiled_spec: {
    max_global_fp: number;
    pairwise_fairness_delta: number;
    max_fpr_override: number | null;
    region_abuse_multipliers: Record<string, number>;
    rationale: string;
  };
  solver_result: PolicyResult;
  status: "AWAITING_HITL_APPROVAL" | "DEPLOYED" | "REJECTED" | "INFEASIBLE";
  escalation_analysis: {
    executive_summary: string;
    recommended_lever: string | null;
    tradeoff_analysis: string[];
  } | null;
  approved_by: string | null;
  deployed_at_utc: string | null;
}

export interface MerchantFeatures {
  account_tenure_months: number;
  kyc_tier: number;
  reserve_deposit_usd: number;
  dispute_rate_bps: number;
  settlement_hold_days: number;
  effective_risk_ratio: number;
  liquidity_coverage_index: number;
}

export interface MerchantCase {
  merchant_id: string;
  slice_id: string;
  threshold: number;
  timestamp_utc: string;
  features: MerchantFeatures;
}

export interface RecourseSolveResponse {
  recourse: {
    merchant_id: string;
    slice_id: string;
    audit_status: "PASSED" | "ALREADY_APPROVED" | "FAILED_AUDIT";
    threshold: number;
    initial_risk_score: number;
    counterfactual_risk_score: number | null;
    actions: Record<string, number> | null;
    step_counts: Record<string, number> | null;
    counterfactual_features: MerchantFeatures | null;
    total_cost: number | null;
    dag_residuals: {
      effective_risk_ratio: number;
      liquidity_coverage_index: number;
    } | null;
    states_visited: number;
    violated_constraints: string[];
  };
  greedy_1d_baseline: {
    feasible: boolean;
    total_cost: number | null;
    actions: Record<string, number> | null;
    cost_savings_vs_greedy: number | null;
  };
  recovery_card: {
    merchant_id: string;
    headline: string;
    action_steps: string[];
    causal_summary: string;
    verified_guarantee: boolean;
  };
}

export interface TelemetrySpan {
  trace_id: string;
  span_id: string;
  parent_span_id: string | null;
  name: string;
  category: "llm" | "solver" | "mcp" | "hitl" | "workflow";
  start_ms: number;
  duration_ms: number;
  status: string;
  attributes: Record<string, unknown>;
}

export interface TelemetrySnapshot {
  summary: {
    total_spans: number;
    llm_avg_ms: number;
    solver_avg_ms: number;
    avg_ttft_ms: number;
    avg_tokens_per_sec: number;
    cumulative_llm_cost_usd: number;
  };
  spans: TelemetrySpan[];
}

export interface BenchmarkReport {
  mode: string;
  policy_scenarios: number;
  recourse_cases: number;
  v1_pure_llm: {
    policy_violation_rate_pct: number;
    recourse_infeasible_rate_pct: number;
    combined_violation_rate_pct: number;
    mean_optimality_gap_pct: number;
    p95_latency_ms: number;
    avg_ttft_ms: number;
    avg_tokens_per_sec: number;
    avg_cost_per_policy_update_usd: number;
  };
  v2_causalbound: {
    policy_violation_rate_pct: number;
    recourse_infeasible_rate_pct: number;
    combined_violation_rate_pct: number;
    mean_optimality_gap_pct: number;
    exact_dollar_loss_saved_usd: string;
    avg_states_pruned_ratio: string;
    p95_latency_ms: number;
    p95_recourse_solver_ms: number;
    avg_ttft_ms: number;
    avg_tokens_per_sec: number;
    avg_cost_per_policy_update_usd: number;
  };
}
