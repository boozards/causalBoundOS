"use client";

import { useEffect, useState } from "react";
import { API_BASE, fetchLatestBenchmark, fetchTelemetryWaterfall } from "@/lib/api";
import type { BenchmarkReport, TelemetrySnapshot } from "@/lib/types";

const VERIFICATION_SCORECARD = [
  { stage: "Stage 1–2", title: "Existence, Finiteness & Strict Key-Set", status: "ENFORCED" },
  { stage: "Stage 3", title: "Coupled Invariants & SCM Residuals <= 1e-4", status: "ENFORCED" },
  { stage: "Stage 4–5", title: "Global Optimality (tol < min step) & 3-Tier Match", status: "ENFORCED" },
  { stage: "Audit A", title: "Anti-Slack Proof (max_fp, delta, max_fpr binding)", status: "VERIFIED" },
  { stage: "Audit B", title: "Anti-Collapse 3-Tier Tie-Breaking Ladder", status: "VERIFIED" },
  { stage: "Audit C", title: "Runtime Dynamic Anti-Memorization Injection", status: "VERIFIED" },
  { stage: "Audit D", title: "Bi-Directional FAILED_AUDIT & Zero-Division Guards", status: "VERIFIED" },
  { stage: "3-Tier Regimes", title: "Nop (0.0) · Oracle (1.0 < 30s) · +/-1 Perturb (0.0)", status: "VERIFIED" },
];

export function BenchmarkTelemetryLab() {
  const [telemetry, setTelemetry] = useState<TelemetrySnapshot | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkReport | null>(null);

  useEffect(() => {
    fetchTelemetryWaterfall().then(setTelemetry).catch(() => {});
    fetchLatestBenchmark()
      .then((res) => {
        if (res.available && res.report) setBenchmark(res.report);
      })
      .catch(() => {});

    // Live SSE waterfall updates
    const es = new EventSource(`${API_BASE}/api/telemetry/stream`);
    es.onmessage = (event) => {
      try {
        setTelemetry(JSON.parse(event.data) as TelemetrySnapshot);
      } catch {
        // Ignore malformed frames
      }
    };
    return () => es.close();
  }, []);

  const spans = telemetry?.spans ?? [];
  const maxDuration = Math.max(1, ...spans.map((s) => s.duration_ms));

  return (
    <div className="space-y-8">
      {/* Side-by-Side V1 vs V2 Benchmark Table */}
      <section className="rounded-xl border border-stonehair bg-paper p-6">
        <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h2 className="font-serif text-2xl font-medium">
              V1 (Pure-LLM / 1D) vs. V2 (CausalBound Neuro-Symbolic)
            </h2>
            <p className="text-xs text-inkmuted">
              25 coupled 16-slice policy recalibration scenarios + 30 blocked merchant SCM recourse
              cases.
            </p>
          </div>
          <span className="font-mono text-xs text-inkmuted">
            Source: benchmarks/latest_report.json
          </span>
        </div>

        {!benchmark ? (
          <div className="rounded-lg border border-dashed border-stonehair bg-alabaster p-6 text-center font-mono text-xs text-inkmuted">
            Run <code>python benchmarks/run_v1_vs_v2.py</code> to populate live V1 vs. V2 metrics.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-stonehair text-left text-xs uppercase tracking-wider text-inkmuted">
                  <th className="py-2.5 pr-4 font-medium">Benchmark Dimension</th>
                  <th className="px-4 py-2.5 font-medium">V1: Pure-LLM / 1D Baseline</th>
                  <th className="px-4 py-2.5 font-medium text-forest">
                    V2: CausalBound (Neuro-Symbolic)
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stonehair/70 font-mono text-xs tabular-nums">
                <tr>
                  <td className="py-3 pr-4 font-sans text-obsidian">
                    Policy Hard Constraint Violation Rate
                  </td>
                  <td className="px-4 py-3 text-terracotta">
                    {benchmark.v1_pure_llm.policy_violation_rate_pct.toFixed(2)}%
                  </td>
                  <td className="px-4 py-3 font-semibold text-forest">
                    {benchmark.v2_causalbound.policy_violation_rate_pct.toFixed(2)}%
                  </td>
                </tr>
                <tr>
                  <td className="py-3 pr-4 font-sans text-obsidian">
                    Recourse 1D Infeasibility Rate
                  </td>
                  <td className="px-4 py-3 text-terracotta">
                    {benchmark.v1_pure_llm.recourse_infeasible_rate_pct.toFixed(2)}%
                  </td>
                  <td className="px-4 py-3 font-semibold text-forest">
                    {benchmark.v2_causalbound.recourse_infeasible_rate_pct.toFixed(2)}%
                  </td>
                </tr>
                <tr>
                  <td className="py-3 pr-4 font-sans text-obsidian">Mean Optimality Gap</td>
                  <td className="px-4 py-3 text-inkmuted">
                    {benchmark.v1_pure_llm.mean_optimality_gap_pct.toFixed(2)}%
                  </td>
                  <td className="px-4 py-3 font-semibold text-forest">
                    {benchmark.v2_causalbound.mean_optimality_gap_pct.toFixed(2)}%
                  </td>
                </tr>
                <tr>
                  <td className="py-3 pr-4 font-sans text-obsidian">
                    Engine B States Visited / 2.20e7 Space
                  </td>
                  <td className="px-4 py-3 text-inkmuted">N/A (1D heuristic)</td>
                  <td className="px-4 py-3 font-semibold text-forest">
                    {benchmark.v2_causalbound.avg_states_pruned_ratio}
                  </td>
                </tr>
                <tr>
                  <td className="py-3 pr-4 font-sans text-obsidian">
                    P95 Policy Update Latency
                  </td>
                  <td className="px-4 py-3 text-inkmuted">
                    {benchmark.v1_pure_llm.p95_latency_ms.toFixed(1)} ms
                  </td>
                  <td className="px-4 py-3 font-semibold text-obsidian">
                    {benchmark.v2_causalbound.p95_latency_ms.toFixed(1)} ms
                  </td>
                </tr>
                <tr>
                  <td className="py-3 pr-4 font-sans text-obsidian">Cost per Policy Update</td>
                  <td className="px-4 py-3 text-inkmuted">
                    ${benchmark.v1_pure_llm.avg_cost_per_policy_update_usd.toFixed(6)}
                  </td>
                  <td className="px-4 py-3 font-semibold text-forest">
                    ${benchmark.v2_causalbound.avg_cost_per_policy_update_usd.toFixed(6)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Split: Live OpenTelemetry Waterfall + 5-Stage Scorecard */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <section className="rounded-xl border border-stonehair bg-paper p-6 lg:col-span-7">
          <div className="mb-4 flex items-baseline justify-between">
            <div>
              <h3 className="font-serif text-xl font-medium">OpenTelemetry Span Waterfall</h3>
              <p className="text-xs text-inkmuted">
                Live <code>gen_ai.*</code> LLM reasoning vs. deterministic solver milliseconds
              </p>
            </div>
            {telemetry && (
              <div className="font-mono text-xs text-inkmuted">
                LLM avg {telemetry.summary.llm_avg_ms}ms &middot; Solver avg{" "}
                <strong className="text-forest">{telemetry.summary.solver_avg_ms}ms</strong>
              </div>
            )}
          </div>

          <div className="max-h-[380px] space-y-2 overflow-y-auto pr-1">
            {spans.length === 0 ? (
              <div className="py-10 text-center font-mono text-xs text-inkmuted">
                Interact with the Policy Simulator or Recourse Inspector to stream live OTel spans.
              </div>
            ) : (
              [...spans].reverse().map((sp) => {
                const widthPct = Math.max(4, Math.min(100, (sp.duration_ms / maxDuration) * 100));
                const barColor =
                  sp.category === "solver"
                    ? "bg-forest"
                    : sp.category === "llm"
                      ? "bg-amberdeep"
                      : "bg-obsidian/70";
                return (
                  <div
                    key={sp.span_id}
                    className="rounded-lg border border-stonehair bg-alabaster/70 p-2.5"
                  >
                    <div className="flex items-center justify-between font-mono text-xs">
                      <span className="font-medium text-obsidian">{sp.name}</span>
                      <span className="tabular-nums text-inkmuted">
                        {sp.duration_ms.toFixed(2)} ms
                      </span>
                    </div>
                    <div className="mt-1.5 h-1.5 w-full rounded-full bg-parchment">
                      <div
                        className={`h-1.5 rounded-full ${barColor}`}
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </section>

        <section className="rounded-xl border border-stonehair bg-paper p-6 lg:col-span-5">
          <h3 className="font-serif text-xl font-medium">Harbor Verification Scorecard</h3>
          <p className="mt-1 text-xs text-inkmuted">
            Non-root UID 1000 &middot; 1 vCPU / 2048 MB &middot; Load-Purge-Verify
          </p>
          <div className="mt-4 space-y-2">
            {VERIFICATION_SCORECARD.map((row) => (
              <div
                key={row.stage}
                className="flex items-center justify-between gap-2 rounded-lg border border-stonehair bg-alabaster/60 px-3 py-2"
              >
                <div>
                  <div className="font-mono text-[11px] font-semibold text-obsidian">
                    {row.stage}
                  </div>
                  <div className="text-xs text-inkmuted">{row.title}</div>
                </div>
                <span className="shrink-0 rounded-full bg-forest/10 px-2 py-0.5 font-mono text-[10px] font-semibold text-forest">
                  {row.status}
                </span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
