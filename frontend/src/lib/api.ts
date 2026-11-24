import type {
  BenchmarkReport,
  MerchantCase,
  PolicyProposal,
  RecourseSolveResponse,
  SimulateResponse,
  TelemetrySnapshot,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export function simulatePolicy(payload: {
  max_global_fp: number;
  pairwise_fairness_delta: number;
  max_fpr_override?: number | null;
  region_abuse_multipliers: Record<string, number>;
}): Promise<SimulateResponse> {
  return fetchJSON("/api/policy/simulate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function proposePolicy(prompt: string): Promise<PolicyProposal> {
  return fetchJSON("/api/policy/propose", {
    method: "POST",
    body: JSON.stringify({ prompt }),
  });
}

export function submitHitlDecision(
  proposalId: string,
  approve: boolean,
  reviewer: string
): Promise<PolicyProposal> {
  return fetchJSON(`/api/policy/proposals/${proposalId}/hitl`, {
    method: "POST",
    body: JSON.stringify({ approve, reviewer }),
  });
}

export function fetchMerchants(): Promise<{ merchants: MerchantCase[] }> {
  return fetchJSON("/api/merchants");
}

export function solveMerchantRecourse(
  merchant_id: string,
  threshold_override?: number | null
): Promise<RecourseSolveResponse> {
  return fetchJSON("/api/recourse/solve", {
    method: "POST",
    body: JSON.stringify({ merchant_id, threshold_override: threshold_override ?? null }),
  });
}

export function fetchTelemetryWaterfall(): Promise<TelemetrySnapshot> {
  return fetchJSON("/api/telemetry/waterfall?limit=80");
}

export function fetchLatestBenchmark(): Promise<{
  available: boolean;
  report: BenchmarkReport | null;
}> {
  return fetchJSON("/api/benchmark/latest");
}
