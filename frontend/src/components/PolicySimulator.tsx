"use client";

import { useEffect, useRef, useState } from "react";
import { proposePolicy, simulatePolicy, submitHitlDecision } from "@/lib/api";
import type { PolicyProposal, SimulateResponse } from "@/lib/types";

const REGIONS = ["NA", "EU", "APAC", "LATAM"] as const;
const TIERS = ["micro", "smb", "mid_market", "enterprise"] as const;

function formatUSD(cents: number | null): string {
  if (cents === null) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(cents / 100);
}

export function PolicySimulator() {
  const [maxGlobalFp, setMaxGlobalFp] = useState<number>(95);
  const [fairnessDelta, setFairnessDelta] = useState<number>(0.18);
  const [regionMults, setRegionMults] = useState<Record<string, number>>({
    NA: 1.0,
    EU: 1.0,
    APAC: 1.0,
    LATAM: 1.0,
  });

  const [simData, setSimData] = useState<SimulateResponse | null>(null);
  const [baselineLossCents, setBaselineLossCents] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // ADK Policy Compiler + HITL Gate state
  const [operatorPrompt, setOperatorPrompt] = useState<string>(
    "Cap global false bans at 92 with pairwise fairness delta 0.18 during a 1.6x LATAM fraud wave."
  );
  const [proposal, setProposal] = useState<PolicyProposal | null>(null);
  const [proposing, setProposing] = useState<boolean>(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await simulatePolicy({
          max_global_fp: maxGlobalFp,
          pairwise_fairness_delta: fairnessDelta,
          region_abuse_multipliers: regionMults,
        });
        setSimData(res);
        if (baselineLossCents === null && res.policy_result.total_loss_cents !== null) {
          setBaselineLossCents(res.policy_result.total_loss_cents);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to run Engine A");
      } finally {
        setLoading(false);
      }
    }, 140);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [maxGlobalFp, fairnessDelta, regionMults, baselineLossCents]);

  const handleCompilePrompt = async () => {
    setProposing(true);
    setError(null);
    try {
      const prop = await proposePolicy(operatorPrompt);
      setProposal(prop);
      setMaxGlobalFp(prop.compiled_spec.max_global_fp);
      setFairnessDelta(prop.compiled_spec.pairwise_fairness_delta);
      if (Object.keys(prop.compiled_spec.region_abuse_multipliers).length > 0) {
        setRegionMults((prev) => ({ ...prev, ...prop.compiled_spec.region_abuse_multipliers }));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Policy compilation failed");
    } finally {
      setProposing(false);
    }
  };

  const handleHitl = async (approve: boolean) => {
    if (!proposal) return;
    const updated = await submitHitlDecision(proposal.proposal_id, approve, "risk.lead@desk");
    setProposal(updated);
  };

  const pol = simData?.policy_result;
  const lossDeltaCents =
    pol?.total_loss_cents != null && baselineLossCents != null
      ? pol.total_loss_cents - baselineLossCents
      : null;

  return (
    <div className="space-y-8">
      {/* Top Bar: ADK Policy Compiler Directive */}
      <section className="rounded-xl border border-stonehair bg-paper p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="flex-1 space-y-2">
            <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-inkmuted">
              <span>ADK 2.0 Policy Compiler</span>
              <span className="h-1 w-1 rounded-full bg-stonehair" />
              <span>Natural Language &rarr; Strict Engine A Contract</span>
            </div>
            <input
              type="text"
              value={operatorPrompt}
              onChange={(e) => setOperatorPrompt(e.target.value)}
              className="w-full rounded-lg border border-stonehair bg-alabaster px-4 py-2.5 text-sm text-obsidian focus:border-obsidian focus:outline-none"
            />
          </div>
          <button
            onClick={handleCompilePrompt}
            disabled={proposing}
            className="rounded-lg bg-obsidian px-5 py-2.5 text-sm font-medium text-alabaster transition hover:opacity-90 disabled:opacity-50"
          >
            {proposing ? "Compiling & Solving..." : "Compile & Propose Policy"}
          </button>
        </div>

        {proposal && (
          <div className="mt-4 flex flex-col gap-3 rounded-lg border border-stonehair bg-parchment/70 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1 text-sm">
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-semibold text-obsidian">
                  {proposal.proposal_id}
                </span>
                <span
                  className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                    proposal.status === "DEPLOYED"
                      ? "bg-forest/10 text-forest"
                      : proposal.status === "INFEASIBLE" || proposal.status === "REJECTED"
                        ? "bg-terracotta/10 text-terracotta"
                        : "bg-amberdeep/15 text-amberdeep"
                  }`}
                >
                  {proposal.status.replaceAll("_", " ")}
                </span>
              </div>
              <p className="text-xs text-inkmuted">
                Compiled spec: max_global_fp={proposal.compiled_spec.max_global_fp}, delta=
                {proposal.compiled_spec.pairwise_fairness_delta.toFixed(2)}
                {proposal.escalation_analysis
                  ? ` — Escalated to Pro: ${proposal.escalation_analysis.executive_summary}`
                  : ""}
              </p>
            </div>
            {proposal.status === "AWAITING_HITL_APPROVAL" && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleHitl(true)}
                  className="rounded-md bg-forest px-3.5 py-1.5 text-xs font-medium text-white hover:opacity-90"
                >
                  Approve &amp; Deploy to Ledger
                </button>
                <button
                  onClick={() => handleHitl(false)}
                  className="rounded-md border border-stonehair bg-paper px-3.5 py-1.5 text-xs font-medium text-obsidian hover:bg-parchment"
                >
                  Reject
                </button>
              </div>
            )}
          </div>
        )}
      </section>

      {/* Main Split: Sliders + 16-Slice Matrix */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Left Rail: Coupled Policy Constraints */}
        <aside className="space-y-6 rounded-xl border border-stonehair bg-paper p-6 lg:col-span-4">
          <div>
            <h2 className="font-serif text-xl font-medium text-obsidian">Coupled Constraints</h2>
            <p className="mt-1 text-xs text-inkmuted">
              Every slider fires exact Engine A MCKP DP over all 16 merchant slices simultaneously.
            </p>
          </div>

          <div className="space-y-5">
            <div>
              <div className="mb-1.5 flex items-center justify-between text-sm">
                <span className="font-medium">Global False-Ban Budget (max_fp)</span>
                <span className="font-mono font-semibold tabular-nums">{maxGlobalFp}</span>
              </div>
              <input
                type="range"
                min={40}
                max={180}
                step={1}
                value={maxGlobalFp}
                onChange={(e) => setMaxGlobalFp(Number(e.target.value))}
                className="editorial-slider w-full"
              />
              <div className="mt-1 flex justify-between font-mono text-[11px] text-inkmuted">
                <span>40 (Tight)</span>
                <span>95 (Nominal)</span>
                <span>180 (Relaxed)</span>
              </div>
            </div>

            <div>
              <div className="mb-1.5 flex items-center justify-between text-sm">
                <span className="font-medium">Pairwise Fairness (&delta; TPR spread)</span>
                <span className="font-mono font-semibold tabular-nums">
                  {fairnessDelta.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={0.04}
                max={0.5}
                step={0.01}
                value={fairnessDelta}
                onChange={(e) => setFairnessDelta(Number(e.target.value))}
                className="editorial-slider w-full"
              />
              <div className="mt-1 flex justify-between font-mono text-[11px] text-inkmuted">
                <span>0.04 (Strict)</span>
                <span>0.18 (Nominal)</span>
                <span>0.50 (Loose)</span>
              </div>
            </div>

            <div className="border-t border-stonehair pt-4">
              <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-inkmuted">
                Regional Abuse Wave Multipliers (FN Loss)
              </div>
              <div className="space-y-3">
                {REGIONS.map((region) => (
                  <div key={region}>
                    <div className="mb-1 flex items-center justify-between text-xs">
                      <span className="font-medium">{region}</span>
                      <span className="font-mono tabular-nums">
                        {regionMults[region].toFixed(2)}&times;
                      </span>
                    </div>
                    <input
                      type="range"
                      min={0.5}
                      max={2.5}
                      step={0.05}
                      value={regionMults[region]}
                      onChange={(e) =>
                        setRegionMults((prev) => ({ ...prev, [region]: Number(e.target.value) }))
                      }
                      className="editorial-slider w-full"
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>

        {/* Right Panel: KPI Summary + 4x4 Threshold Matrix */}
        <div className="space-y-6 lg:col-span-8">
          {error && (
            <div className="rounded-lg border border-terracotta/30 bg-terracotta/5 p-4 text-sm text-terracotta">
              {error}
            </div>
          )}

          {/* KPI Ribbon */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="rounded-xl border border-stonehair bg-paper p-4">
              <div className="text-[11px] uppercase tracking-wider text-inkmuted">Audit Status</div>
              <div
                className={`mt-1.5 font-mono text-base font-semibold ${
                  pol?.audit_status === "PASSED" ? "text-forest" : "text-terracotta"
                }`}
              >
                {loading ? "SOLVING..." : pol?.audit_status ?? "—"}
              </div>
            </div>
            <div className="rounded-xl border border-stonehair bg-paper p-4">
              <div className="text-[11px] uppercase tracking-wider text-inkmuted">
                Global Exact Loss
              </div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-obsidian">
                {formatUSD(pol?.total_loss_cents ?? null)}
              </div>
              {lossDeltaCents !== null && lossDeltaCents !== 0 && (
                <div
                  className={`font-mono text-[11px] tabular-nums ${
                    lossDeltaCents > 0 ? "text-terracotta" : "text-forest"
                  }`}
                >
                  {lossDeltaCents > 0 ? "+" : ""}
                  {formatUSD(lossDeltaCents)} vs nominal
                </div>
              )}
            </div>
            <div className="rounded-xl border border-stonehair bg-paper p-4">
              <div className="text-[11px] uppercase tracking-wider text-inkmuted">
                Global False Bans
              </div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-obsidian">
                {pol?.total_fp ?? "—"} / {maxGlobalFp}
              </div>
            </div>
            <div className="rounded-xl border border-stonehair bg-paper p-4">
              <div className="text-[11px] uppercase tracking-wider text-inkmuted">
                Pairwise TPR Spread
              </div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-obsidian">
                {pol?.global_tpr_spread != null ? pol.global_tpr_spread.toFixed(3) : "—"}
              </div>
            </div>
          </div>

          {/* Infeasibility Warning & Minimal Mathematical Relaxations */}
          {pol?.audit_status === "FAILED_AUDIT" && (
            <div className="rounded-xl border border-terracotta/40 bg-terracotta/5 p-5">
              <div className="flex items-center justify-between">
                <h3 className="font-serif text-lg font-medium text-terracotta">
                  Infeasible Policy Surface (FAILED_AUDIT)
                </h3>
                <span className="font-mono text-xs text-terracotta">
                  {pol.violated_constraints.join(" · ")}
                </span>
              </div>
              <p className="mt-1 text-xs text-inkmuted">
                All threshold selections are locked strictly to <code>null</code>. Verified minimal
                single-lever relaxations computed by <code>policy_solver_mcp</code>:
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {simData?.infeasibility_diagnosis?.relaxation_options.map((opt) => (
                  <button
                    key={opt.lever}
                    onClick={() => {
                      if (opt.lever === "max_global_fp") setMaxGlobalFp(opt.minimal_feasible);
                      if (opt.lever === "pairwise_fairness_delta")
                        setFairnessDelta(opt.minimal_feasible);
                    }}
                    className="rounded-lg border border-terracotta/30 bg-paper px-3 py-2 text-left text-xs transition hover:border-obsidian"
                  >
                    <div className="font-mono font-semibold text-obsidian">
                      Apply {opt.lever} = {opt.minimal_feasible}
                    </div>
                    <div className="text-inkmuted">
                      Exact loss: {formatUSD(opt.resulting_loss_cents)}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* 16-Slice (4x4) Exact Threshold Matrix */}
          <div className="rounded-xl border border-stonehair bg-paper p-6">
            <div className="mb-4 flex items-baseline justify-between">
              <h3 className="font-serif text-xl font-medium">16-Slice Optimal Threshold Matrix</h3>
              <span className="font-mono text-xs text-inkmuted">
                4 Regions &times; 4 Merchant Tiers
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-stonehair text-[11px] uppercase tracking-wider text-inkmuted">
                    <th className="py-2 pr-3 font-medium">Region</th>
                    {TIERS.map((tier) => (
                      <th key={tier} className="px-2 py-2 font-medium">
                        {tier.replace("_", " ")}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-stonehair/70">
                  {REGIONS.map((region) => (
                    <tr key={region}>
                      <td className="py-3 pr-3 font-mono text-xs font-semibold text-obsidian">
                        {region}
                      </td>
                      {TIERS.map((tier) => {
                        const sid = `${region}_${tier}`;
                        const sm = pol?.slice_metrics?.[sid];
                        const ece = pol?.slice_ece?.[sid];
                        return (
                          <td key={sid} className="p-2 align-top">
                            <div className="rounded-lg border border-stonehair bg-alabaster/70 p-2.5">
                              <div className="flex items-baseline justify-between">
                                <span className="font-mono text-sm font-semibold tabular-nums text-obsidian">
                                  {sm ? `t=${sm.threshold.toFixed(2)}` : "null"}
                                </span>
                                <span className="font-mono text-[10px] text-inkmuted">
                                  ECE {ece != null ? ece.toFixed(3) : "—"}
                                </span>
                              </div>
                              {sm && (
                                <div className="mt-1.5 grid grid-cols-2 gap-x-2 gap-y-0.5 font-mono text-[10px] tabular-nums text-inkmuted">
                                  <span>FP {sm.fp}</span>
                                  <span>FN {sm.fn}</span>
                                  <span>TPR {(sm.tpr * 100).toFixed(0)}%</span>
                                  <span>FPR {(sm.fpr * 100).toFixed(1)}%</span>
                                </div>
                              )}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
