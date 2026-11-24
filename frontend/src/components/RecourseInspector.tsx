"use client";

import { useEffect, useState } from "react";
import { fetchMerchants, solveMerchantRecourse } from "@/lib/api";
import type { MerchantCase, RecourseSolveResponse } from "@/lib/types";

export function RecourseInspector() {
  const [merchants, setMerchants] = useState<MerchantCase[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [result, setResult] = useState<RecourseSolveResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    fetchMerchants()
      .then((res) => {
        setMerchants(res.merchants);
        if (res.merchants.length > 0) {
          setSelectedId(res.merchants[0].merchant_id);
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    setLoading(true);
    solveMerchantRecourse(selectedId)
      .then(setResult)
      .finally(() => setLoading(false));
  }, [selectedId]);

  const currentMerchant = merchants.find((m) => m.merchant_id === selectedId);
  const rec = result?.recourse;
  const cf = rec?.counterfactual_features;
  const greedy = result?.greedy_1d_baseline;
  const card = result?.recovery_card;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
      {/* Left Rail: Blocked Merchant Directory */}
      <aside className="rounded-xl border border-stonehair bg-paper p-4 lg:col-span-3">
        <div className="mb-3 px-2">
          <h2 className="font-serif text-lg font-medium">Blocked Accounts</h2>
          <p className="text-xs text-inkmuted">30 accounts &middot; 2.20&times;10&sup7; space each</p>
        </div>
        <div className="max-h-[600px] space-y-1.5 overflow-y-auto pr-1">
          {merchants.map((m) => {
            const active = m.merchant_id === selectedId;
            return (
              <button
                key={m.merchant_id}
                onClick={() => setSelectedId(m.merchant_id)}
                className={`w-full rounded-lg border px-3 py-2.5 text-left transition ${
                  active
                    ? "border-obsidian bg-parchment"
                    : "border-transparent bg-alabaster/60 hover:border-stonehair"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-semibold">{m.merchant_id}</span>
                  <span className="font-mono text-[11px] text-inkmuted">
                    t={m.threshold.toFixed(2)}
                  </span>
                </div>
                <div className="mt-0.5 text-[11px] text-inkmuted">
                  {m.slice_id} &middot; Dispute {m.features.dispute_rate_bps} bps
                </div>
              </button>
            );
          })}
        </div>
      </aside>

      {/* Main Inspector: SCM DAG + Coupled vs 1D Comparison + Appeals Recovery Card */}
      <div className="space-y-6 lg:col-span-9">
        {/* Interactive Editorial SCM DAG Diagram */}
        <section className="rounded-xl border border-stonehair bg-paper p-6">
          <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
            <div>
              <h2 className="font-serif text-2xl font-medium">
                Non-Linear Structural Causal Model (SCM) DAG
              </h2>
              <p className="text-xs text-inkmuted">
                Exact Priority-Queue B&amp;B over coupled quadratic cross-terms (&beta;<sub>ij</sub>{" "}
                &gt; 0) with DAG residual tolerance &le; 10<sup>&minus;4</sup>.
              </p>
            </div>
            {rec && (
              <div className="font-mono text-xs tabular-nums text-inkmuted">
                Visited <strong className="text-obsidian">{rec.states_visited.toLocaleString()}</strong>{" "}
                / 21,992,043 states
              </div>
            )}
          </div>

          {loading || !currentMerchant || !rec || !cf ? (
            <div className="py-16 text-center font-mono text-xs text-inkmuted">
              Running exact Branch-and-Bound solver...
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              {/* Column 1: Root Nodes */}
              <div className="space-y-2.5">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-inkmuted">
                  1. Root Variables
                </div>
                <div className="rounded-lg border border-dashed border-stonehair bg-parchment/60 p-3">
                  <div className="flex justify-between text-xs">
                    <span className="font-medium">account_tenure_months</span>
                    <span className="rounded bg-stonehair/70 px-1.5 py-0.5 font-mono text-[10px]">
                      IMMUTABLE
                    </span>
                  </div>
                  <div className="mt-1 font-mono text-xs tabular-nums text-inkmuted">
                    {currentMerchant.features.account_tenure_months} mo &rarr;{" "}
                    {cf.account_tenure_months} mo
                  </div>
                </div>

                {[
                  {
                    label: "kyc_tier",
                    before: currentMerchant.features.kyc_tier,
                    after: cf.kyc_tier,
                    delta: rec.actions?.kyc_tier_delta ?? 0,
                    prefix: "+",
                    unit: "",
                  },
                  {
                    label: "reserve_deposit_usd",
                    before: currentMerchant.features.reserve_deposit_usd,
                    after: cf.reserve_deposit_usd,
                    delta: rec.actions?.reserve_deposit_usd_delta ?? 0,
                    prefix: "+$",
                    unit: "",
                  },
                  {
                    label: "dispute_rate_bps",
                    before: currentMerchant.features.dispute_rate_bps,
                    after: cf.dispute_rate_bps,
                    delta: rec.actions?.dispute_reduction_bps ?? 0,
                    prefix: "-",
                    unit: " bps",
                  },
                  {
                    label: "settlement_hold_days",
                    before: currentMerchant.features.settlement_hold_days,
                    after: cf.settlement_hold_days,
                    delta: rec.actions?.settlement_hold_days_delta ?? 0,
                    prefix: "+",
                    unit: "d",
                  },
                ].map((node) => {
                  const changed = node.delta > 0;
                  return (
                    <div
                      key={node.label}
                      className={`rounded-lg border p-3 ${
                        changed
                          ? "border-forest/40 bg-forest/5"
                          : "border-stonehair bg-alabaster/50"
                      }`}
                    >
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-mono font-medium">{node.label}</span>
                        {changed && (
                          <span className="font-mono text-[11px] font-semibold text-forest">
                            {node.prefix}
                            {node.delta.toLocaleString()}
                            {node.unit}
                          </span>
                        )}
                      </div>
                      <div className="mt-1 font-mono text-[11px] tabular-nums text-inkmuted">
                        {node.before.toLocaleString()} &rarr;{" "}
                        <strong className="text-obsidian">{node.after.toLocaleString()}</strong>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Column 2: Derived Non-Linear Nodes */}
              <div className="flex flex-col justify-center space-y-4">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-inkmuted">
                  2. Non-Linear Derived Nodes
                </div>
                <div className="rounded-lg border border-stonehair bg-alabaster p-4">
                  <div className="font-mono text-xs font-semibold">effective_risk_ratio</div>
                  <div className="mt-1 font-mono text-[11px] text-inkmuted">
                    exp(0.0025&middot;dispute) / (1 + 0.35&middot;kyc + 0.15&middot;ln(1+t) +
                    0.08&middot;&radic;hold)
                  </div>
                  <div className="mt-2 font-mono text-sm tabular-nums">
                    {currentMerchant.features.effective_risk_ratio.toFixed(3)} &rarr;{" "}
                    <strong className="text-forest">{cf.effective_risk_ratio.toFixed(3)}</strong>
                  </div>
                  <div className="mt-1 font-mono text-[10px] text-inkmuted">
                    Residual: {rec.dag_residuals?.effective_risk_ratio.toExponential(2)} (&le; 1e-4)
                  </div>
                </div>

                <div className="rounded-lg border border-stonehair bg-alabaster p-4">
                  <div className="font-mono text-xs font-semibold">liquidity_coverage_index</div>
                  <div className="mt-1 font-mono text-[11px] text-inkmuted">
                    ln(1+reserve) + 0.6&middot;&radic;(hold+1) + 0.4&middot;kyc &minus;
                    0.12&middot;&radic;dispute
                  </div>
                  <div className="mt-2 font-mono text-sm tabular-nums">
                    {currentMerchant.features.liquidity_coverage_index.toFixed(3)} &rarr;{" "}
                    <strong className="text-forest">
                      {cf.liquidity_coverage_index.toFixed(3)}
                    </strong>
                  </div>
                  <div className="mt-1 font-mono text-[10px] text-inkmuted">
                    Residual: {rec.dag_residuals?.liquidity_coverage_index.toExponential(2)} (&le;
                    1e-4)
                  </div>
                </div>
              </div>

              {/* Column 3: Classifier Flip & Cost Verdict */}
              <div className="flex flex-col justify-center space-y-4">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-inkmuted">
                  3. Verified Classifier Flip
                </div>
                <div className="rounded-lg border border-forest/40 bg-forest/5 p-4">
                  <div className="text-xs text-inkmuted">Monotonic HGB Risk Score</div>
                  <div className="mt-1 font-mono text-2xl font-semibold tabular-nums text-obsidian">
                    {rec.initial_risk_score.toFixed(3)} &rarr;{" "}
                    <span className="text-forest">
                      {rec.counterfactual_risk_score?.toFixed(3)}
                    </span>
                  </div>
                  <div className="mt-1 font-mono text-xs text-forest">
                    Cleared threshold t = {rec.threshold.toFixed(2)}
                  </div>
                </div>

                <div className="rounded-lg border border-stonehair bg-parchment/70 p-4 text-xs">
                  <div className="font-semibold text-obsidian">Exact B&amp;B vs. 1D Greedy</div>
                  <div className="mt-2 flex justify-between font-mono tabular-nums">
                    <span>CausalBound Exact Cost</span>
                    <strong className="text-forest">{rec.total_cost?.toFixed(2)}</strong>
                  </div>
                  <div className="mt-1 flex justify-between font-mono tabular-nums text-inkmuted">
                    <span>Naive 1D Greedy Cost</span>
                    <span>
                      {greedy?.feasible && greedy.total_cost != null
                        ? greedy.total_cost.toFixed(2)
                        : "INFEASIBLE"}
                    </span>
                  </div>
                  {greedy?.cost_savings_vs_greedy != null && greedy.cost_savings_vs_greedy > 0 && (
                    <div className="mt-2 border-t border-stonehair pt-2 font-mono text-[11px] font-semibold text-forest">
                      Saved {greedy.cost_savings_vs_greedy.toFixed(2)} coupled cost units
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Appeals & Recourse Agent Plain-Language Recovery Card */}
        {card && (
          <section className="rounded-xl border border-stonehair bg-paper p-6">
            <div className="flex items-center justify-between">
              <div className="text-xs font-medium uppercase tracking-wider text-inkmuted">
                Appeals &amp; Recourse Agent &middot; Merchant-Facing Card
              </div>
              <span className="rounded-full bg-forest/10 px-2.5 py-0.5 font-mono text-[11px] font-medium text-forest">
                {card.verified_guarantee ? "MATHEMATICALLY VERIFIED" : "ESCALATED"}
              </span>
            </div>
            <h3 className="mt-2 font-serif text-xl font-medium text-obsidian">{card.headline}</h3>
            <p className="mt-2 text-sm leading-relaxed text-inkmuted">{card.causal_summary}</p>
            {card.action_steps.length > 0 && (
              <ol className="mt-4 space-y-2">
                {card.action_steps.map((step, i) => (
                  <li
                    key={step}
                    className="flex items-start gap-3 rounded-lg border border-stonehair bg-alabaster px-3.5 py-2.5 text-sm"
                  >
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-obsidian font-mono text-[11px] text-alabaster">
                      {i + 1}
                    </span>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
