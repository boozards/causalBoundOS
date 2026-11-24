"use client";

import { useState } from "react";
import { BenchmarkTelemetryLab } from "@/components/BenchmarkTelemetryLab";
import { PolicySimulator } from "@/components/PolicySimulator";
import { RecourseInspector } from "@/components/RecourseInspector";

type TabKey = "policy" | "recourse" | "telemetry";

const TABS: { key: TabKey; label: string; subtitle: string }[] = [
  { key: "policy", label: "Live Policy Simulator", subtitle: "Engine A · 16-Slice Coupled DP" },
  { key: "recourse", label: "Causal Recourse Inspector", subtitle: "Engine B · B&B SCM DAG" },
  { key: "telemetry", label: "Benchmark & OTel Lab", subtitle: "V1 vs. V2 · Live Waterfall" },
];

export default function RiskWarRoomPage() {
  const [activeTab, setActiveTab] = useState<TabKey>("policy");

  return (
    <div className="min-h-screen bg-alabaster">
      {/* Editorial Top Header */}
      <header className="border-b border-stonehair bg-paper">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-6 py-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="font-serif text-2xl font-semibold tracking-tight text-obsidian">
                CausalBound
              </span>
              <span className="rounded-full border border-stonehair bg-parchment px-2.5 py-0.5 font-mono text-[11px] text-inkmuted">
                RISK WAR ROOM
              </span>
            </div>
            <p className="mt-1 text-xs text-inkmuted">
              Neuro-Symbolic Merchant Risk Policy &amp; Exact Causal Recourse Autopilot
            </p>
          </div>

          <nav className="flex flex-wrap gap-2">
            {TABS.map((tab) => {
              const active = activeTab === tab.key;
              return (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key)}
                  className={`rounded-lg border px-3.5 py-2 text-left transition ${
                    active
                      ? "border-obsidian bg-obsidian text-alabaster"
                      : "border-stonehair bg-alabaster text-obsidian hover:bg-parchment"
                  }`}
                >
                  <div className="text-xs font-medium">{tab.label}</div>
                  <div
                    className={`font-mono text-[10px] ${
                      active ? "text-alabaster/75" : "text-inkmuted"
                    }`}
                  >
                    {tab.subtitle}
                  </div>
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">
        {activeTab === "policy" && <PolicySimulator />}
        {activeTab === "recourse" && <RecourseInspector />}
        {activeTab === "telemetry" && <BenchmarkTelemetryLab />}
      </main>
    </div>
  );
}
