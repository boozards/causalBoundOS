#!/usr/bin/env python3
"""Self-contained Reference Oracle Solver for CausalBound Harbor Benchmark.

Supports three CLI modes so the verifier can grade both static artifacts and
dynamic runtime Audit B / Audit C / Audit D cases:
  --mode full      Solve 16-slice policy + 30 merchant recourse cases from --data-dir
  --mode policy    Solve a single PolicyOptimizationRequest JSON from --input
  --mode recourse  Solve a single MerchantRecourseCase JSON from --input using --data-dir assets
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ECE_NUM_BINS = 10
SMALL_N_ECE_CUTOFF = 5
ACTION_NAMES_SORTED = (
    "dispute_reduction_bps",
    "kyc_tier_delta",
    "reserve_deposit_usd_delta",
    "settlement_hold_days_delta",
)


class _FastHistGradientBoostingWrapper:
    __slots__ = ("_predictors", "_predict_iters", "_dtype", "_base_val")

    def __init__(self, clf: Any) -> None:
        self._predictors = clf._predictors
        self._predict_iters = clf._predict_iterations
        self._dtype = clf._baseline_prediction.dtype
        self._base_val = float(clf._baseline_prediction.item(0))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not X.flags["C_CONTIGUOUS"] or X.dtype != np.float64:
            X = np.ascontiguousarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        raw = np.empty((n_samples, 1), dtype=self._dtype, order="F")
        raw.fill(self._base_val)
        self._predict_iters(X, self._predictors, raw, False, 1)
        p1 = 1.0 / (1.0 + np.exp(-raw[:, 0]))
        return np.column_stack([1.0 - p1, p1])


def _wrap_fast_classifier(clf: Any) -> Any:
    if (
        hasattr(clf, "_predict_iterations")
        and hasattr(clf, "_predictors")
        and hasattr(clf, "_baseline_prediction")
    ):
        return _FastHistGradientBoostingWrapper(clf)
    return clf


# ---------------------------------------------------------------------------
# Core Engine A: Exact Coupled Slice-Threshold Optimizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SliceCandidate:
    threshold_dec: Decimal
    threshold_float: float
    fp: int
    fn: int
    tp: int
    tn: int
    tpr: Fraction
    fpr: Fraction
    loss_cents: int


def compute_slice_ece(samples: list[dict[str, Any]]) -> float:
    n_total = len(samples)
    if n_total < SMALL_N_ECE_CUTOFF:
        return 0.0
    counts = [0] * ECE_NUM_BINS
    score_sums = [0.0] * ECE_NUM_BINS
    label_sums = [0.0] * ECE_NUM_BINS
    for s in samples:
        sc = float(s["score"])
        idx = min(max(int(math.floor(sc * ECE_NUM_BINS)), 0), ECE_NUM_BINS - 1)
        counts[idx] += 1
        score_sums[idx] += sc
        label_sums[idx] += int(s["label"])
    ece = 0.0
    for c, ss, ls in zip(counts, score_sums, label_sums):
        if c:
            ece += (c / n_total) * abs((ls / c) - (ss / c))
    return ece


def _dedup_fp(cands: list[SliceCandidate]) -> list[SliceCandidate]:
    best: dict[int, SliceCandidate] = {}
    for c in cands:
        prev = best.get(c.fp)
        if prev is None or (c.loss_cents, c.threshold_dec) < (prev.loss_cents, prev.threshold_dec):
            best[c.fp] = c
    return [best[fp] for fp in sorted(best)]


def _prune_pareto(dp: dict[int, tuple]) -> dict[int, tuple]:
    pruned: dict[int, tuple] = {}
    best_key = None
    for fp in sorted(dp):
        val = dp[fp]
        key = (val[0], val[1], val[2])
        if best_key is None or key < best_key:
            pruned[fp] = val
            best_key = key
    return pruned


def _solve_mckp(window_cands: list[list[SliceCandidate]], max_global_fp: int) -> tuple | None:
    deduped = [_dedup_fp(c) for c in window_cands]
    n = len(deduped)
    suffix_min = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + deduped[i][0].fp
    if suffix_min[0] > max_global_fp:
        return None

    dp: dict[int, tuple] = {0: (0, Decimal("0"), (), ())}
    for idx, cands in enumerate(deduped):
        limit = max_global_fp - suffix_min[idx + 1]
        next_dp: dict[int, tuple] = {}
        for prev_fp, (p_loss, p_sum_t, p_tup, p_cands) in dp.items():
            for cand in cands:
                n_fp = prev_fp + cand.fp
                if n_fp > limit:
                    break
                n_val = (
                    p_loss + cand.loss_cents,
                    p_sum_t + cand.threshold_dec,
                    p_tup + (cand.threshold_dec,),
                    p_cands + (cand,),
                )
                cur = next_dp.get(n_fp)
                if cur is None or (n_val[0], n_val[1], n_val[2]) < (cur[0], cur[1], cur[2]):
                    next_dp[n_fp] = n_val
        if not next_dp:
            return None
        dp = _prune_pareto(next_dp)

    best = None
    for val in dp.values():
        if best is None or (val[0], val[1], val[2]) < (best[0], best[1], best[2]):
            best = val
    return best


def solve_policy_dict(req: dict[str, Any]) -> dict[str, Any]:
    ordered_slices = sorted(req["slices"], key=lambda s: s["slice_id"])
    sorted_thresholds = sorted({Decimal(str(t)) for t in req["candidate_thresholds"]})

    slice_ece: dict[str, float] = {}
    ece_violations: list[str] = []
    for sl in ordered_slices:
        ece_val = compute_slice_ece(sl["samples"])
        slice_ece[sl["slice_id"]] = ece_val
        if ece_val > float(sl["max_ece"]):
            ece_violations.append(f"SLICE_ECE_EXCEEDED:{sl['slice_id']}")

    def failed(codes: list[str]) -> dict[str, Any]:
        return {
            "policy_id": req["policy_id"],
            "evaluated_at_utc": req["evaluated_at_utc"],
            "audit_status": "FAILED_AUDIT",
            "selected_thresholds": None,
            "total_loss_cents": None,
            "total_fp": None,
            "total_fn": None,
            "global_tpr_spread": None,
            "slice_ece": slice_ece,
            "slice_metrics": None,
            "violated_constraints": sorted(set(codes)),
        }

    if ece_violations:
        return failed(ece_violations)

    cands_per_slice: list[list[SliceCandidate]] = []
    fpr_violations: list[str] = []
    for sl in ordered_slices:
        samples = sl["samples"]
        pos_total = sum(1 for s in samples if int(s["label"]) == 1)
        neg_total = len(samples) - pos_total
        max_fpr_frac = Fraction(str(sl["max_fpr"]))
        fp_cost = int(sl["fp_cost_cents"])
        fn_cost = int(sl["fn_cost_cents"])

        valid: list[SliceCandidate] = []
        for t_dec in sorted_thresholds:
            t_f = float(t_dec)
            tp = fp = fn = tn = 0
            for s in samples:
                if float(s["score"]) >= t_f:
                    if int(s["label"]) == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if int(s["label"]) == 1:
                        fn += 1
                    else:
                        tn += 1
            tpr = Fraction(1, 1) if pos_total == 0 else Fraction(tp, pos_total)
            fpr = Fraction(0, 1) if neg_total == 0 else Fraction(fp, neg_total)
            if fpr > max_fpr_frac:
                continue
            valid.append(
                SliceCandidate(t_dec, t_f, fp, fn, tp, tn, tpr, fpr, fp_cost * fp + fn_cost * fn)
            )
        if not valid:
            fpr_violations.append(f"SLICE_FPR_INFEASIBLE:{sl['slice_id']}")
        cands_per_slice.append(valid)

    if fpr_violations:
        return failed(fpr_violations)

    delta_frac = Fraction(str(req["pairwise_fairness_delta"]))
    distinct_tprs = sorted({c.tpr for cands in cands_per_slice for c in cands})
    seen_sigs: set[tuple] = set()
    any_fair = False
    best_sol = None

    for lower_tpr in distinct_tprs:
        upper_tpr = lower_tpr + delta_frac
        win_cands: list[list[SliceCandidate]] = []
        ok = True
        for cands in cands_per_slice:
            sub = [c for c in cands if lower_tpr <= c.tpr <= upper_tpr]
            if not sub:
                ok = False
                break
            win_cands.append(sub)
        if not ok:
            continue
        any_fair = True
        sig = tuple(tuple(c.threshold_dec for c in wc) for wc in win_cands)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        sol = _solve_mckp(win_cands, int(req["max_global_fp"]))
        if sol is not None:
            if best_sol is None or (sol[0], sol[1], sol[2]) < (
                best_sol[0],
                best_sol[1],
                best_sol[2],
            ):
                best_sol = sol

    if not any_fair:
        return failed(["PAIRWISE_FAIRNESS_INFEASIBLE"])
    if best_sol is None:
        return failed(["GLOBAL_FP_BUDGET_EXCEEDED"])

    total_loss, _, _, chosen = best_sol
    sel_t: dict[str, float] = {}
    metrics: dict[str, dict[str, Any]] = {}
    total_fp = total_fn = 0
    tprs: list[Fraction] = []
    for sl, c in zip(ordered_slices, chosen):
        sid = sl["slice_id"]
        sel_t[sid] = c.threshold_float
        total_fp += c.fp
        total_fn += c.fn
        tprs.append(c.tpr)
        metrics[sid] = {
            "threshold": c.threshold_float,
            "fp": c.fp,
            "fn": c.fn,
            "tp": c.tp,
            "tn": c.tn,
            "tpr": float(c.tpr),
            "fpr": float(c.fpr),
            "loss_cents": c.loss_cents,
        }

    return {
        "policy_id": req["policy_id"],
        "evaluated_at_utc": req["evaluated_at_utc"],
        "audit_status": "PASSED",
        "selected_thresholds": sel_t,
        "total_loss_cents": total_loss,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "global_tpr_spread": float(max(tprs) - min(tprs)),
        "slice_ece": slice_ece,
        "slice_metrics": metrics,
        "violated_constraints": [],
    }


# ---------------------------------------------------------------------------
# Core Engine B: Exact Priority-Queue Branch-and-Bound Causal Recourse Solver
# ---------------------------------------------------------------------------


def _derived_vec(tenure, kyc, reserve, dispute, hold):
    eff = np.exp(0.0025 * dispute) / (
        1.0 + 0.35 * kyc + 0.15 * np.log1p(tenure) + 0.08 * np.sqrt(hold)
    )
    liq = np.log1p(reserve) + 0.6 * np.sqrt(hold + 1.0) + 0.4 * kyc - 0.12 * np.sqrt(dispute)
    return eff, liq


def solve_recourse_dict(case: dict[str, Any], clf: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    clf = _wrap_fast_classifier(clf)
    raw_actions = cfg["action_specs"]
    specs = [raw_actions[name] for name in ACTION_NAMES_SORTED]
    steps_sz = [float(s["step"]) for s in specs]
    w = [float(s["linear_weight"]) for s in specs]
    alpha = [float(s["quadratic_weight"]) for s in specs]
    w_arr = np.array(w, dtype=np.float64)
    a_arr = np.array(alpha, dtype=np.float64)

    cross = [[0.0] * 4 for _ in range(4)]
    raw_cross = cfg["cross_weights"]
    for i in range(4):
        for j in range(i + 1, 4):
            k1 = f"{ACTION_NAMES_SORTED[i]}__{ACTION_NAMES_SORTED[j]}"
            k2 = f"{ACTION_NAMES_SORTED[j]}__{ACTION_NAMES_SORTED[i]}"
            b = float(raw_cross[k1] if k1 in raw_cross else raw_cross[k2])
            cross[i][j] = cross[j][i] = b

    tie_eps = float(cfg["tolerances"]["tie_break_cost_eps"])
    feat = case["features"]
    tenure0 = float(feat["account_tenure_months"])
    kyc0 = float(feat["kyc_tier"])
    res0 = float(feat["reserve_deposit_usd"])
    disp0 = float(feat["dispute_rate_bps"])
    hold0 = float(feat["settlement_hold_days"])
    threshold = float(case["threshold"])

    x_init = np.array(
        [
            [
                tenure0,
                kyc0,
                res0,
                disp0,
                hold0,
                float(feat["effective_risk_ratio"]),
                float(feat["liquidity_coverage_index"]),
            ]
        ],
        dtype=np.float64,
    )
    init_prob = float(clf.predict_proba(x_init)[0, 1])

    def build_matrix(step_mat: np.ndarray) -> np.ndarray:
        n = step_mat.shape[0]
        t = np.full(n, tenure0, dtype=np.float64)
        k = kyc0 + step_mat[:, 1] * steps_sz[1]
        r = res0 + step_mat[:, 2] * steps_sz[2]
        d = disp0 - step_mat[:, 0] * steps_sz[0]
        h = hold0 + step_mat[:, 3] * steps_sz[3]
        eff, liq = _derived_vec(t, k, r, d, h)
        return np.column_stack([t, k, r, d, h, eff, liq])

    def eval_costs(step_mat: np.ndarray) -> np.ndarray:
        k = step_mat.astype(np.float64, copy=False)
        c = k @ w_arr + (k * k) @ a_arr
        for i in range(4):
            for j in range(i + 1, 4):
                c = c + cross[i][j] * k[:, i] * k[:, j]
        return c

    def cost_scalar(st: tuple[int, int, int, int]) -> float:
        tot = 0.0
        for i in range(4):
            ki = st[i]
            if ki:
                tot += w[i] * ki + alpha[i] * ki * ki
        for i in range(4):
            if st[i]:
                for j in range(i + 1, 4):
                    if st[j]:
                        tot += cross[i][j] * st[i] * st[j]
        return tot

    def better(c_cost: float, c_st: tuple[int, int, int, int], b_cost: float, b_st) -> bool:
        if b_st is None:
            return True
        if c_cost < b_cost - tie_eps:
            return True
        if c_cost > b_cost + tie_eps:
            return False
        if sum(c_st) != sum(b_st):
            return sum(c_st) < sum(b_st)
        return c_st < b_st

    def format_ok(status: str, st: tuple[int, int, int, int], prob: float, cost: float, visited: int):
        k_val = int(round(kyc0 + st[1] * steps_sz[1]))
        r_val = float(res0 + st[2] * steps_sz[2])
        d_val = float(disp0 - st[0] * steps_sz[0])
        h_val = int(round(hold0 + st[3] * steps_sz[3]))
        eff, liq = _derived_vec(tenure0, float(k_val), r_val, d_val, float(h_val))
        actions = {
            ACTION_NAMES_SORTED[i]: (
                int(round(st[i] * steps_sz[i]))
                if steps_sz[i].is_integer()
                else st[i] * steps_sz[i]
            )
            for i in range(4)
        }
        return {
            "merchant_id": case["merchant_id"],
            "slice_id": case["slice_id"],
            "audit_status": status,
            "threshold": threshold,
            "initial_risk_score": init_prob,
            "counterfactual_risk_score": prob,
            "actions": actions,
            "step_counts": {ACTION_NAMES_SORTED[i]: int(st[i]) for i in range(4)},
            "counterfactual_features": {
                "account_tenure_months": tenure0,
                "kyc_tier": k_val,
                "reserve_deposit_usd": r_val,
                "dispute_rate_bps": d_val,
                "settlement_hold_days": h_val,
                "effective_risk_ratio": float(eff),
                "liquidity_coverage_index": float(liq),
            },
            "total_cost": float(cost),
            "dag_residuals": {
                "effective_risk_ratio": 0.0,
                "liquidity_coverage_index": 0.0,
            },
            "states_visited": visited,
            "violated_constraints": [],
        }

    def format_fail(visited: int):
        return {
            "merchant_id": case["merchant_id"],
            "slice_id": case["slice_id"],
            "audit_status": "FAILED_AUDIT",
            "threshold": threshold,
            "initial_risk_score": init_prob,
            "counterfactual_risk_score": None,
            "actions": None,
            "step_counts": None,
            "counterfactual_features": None,
            "total_cost": None,
            "dag_residuals": None,
            "states_visited": visited,
            "violated_constraints": ["NO_FEASIBLE_FLIP"],
        }

    if init_prob < threshold:
        return format_ok("ALREADY_APPROVED", (0, 0, 0, 0), init_prob, 0.0, 1)

    max_k = [
        min(
            int(round((specs[0]["max_delta"] - specs[0]["min_delta"]) / steps_sz[0])),
            int(math.floor((disp0 + 1e-12) / steps_sz[0])),
        ),
        int(round((specs[1]["max_delta"] - specs[1]["min_delta"]) / steps_sz[1])),
        int(round((specs[2]["max_delta"] - specs[2]["min_delta"]) / steps_sz[2])),
        int(round((specs[3]["max_delta"] - specs[3]["min_delta"]) / steps_sz[3])),
    ]
    max_tuple = (max_k[0], max_k[1], max_k[2], max_k[3])

    @lru_cache(maxsize=100_000)
    def eval_prob(st: tuple[int, int, int, int]) -> float:
        X = build_matrix(np.array([st], dtype=np.int64))
        return float(clf.predict_proba(X)[0, 1])

    if eval_prob(max_tuple) >= threshold:
        return format_fail(1)

    # Shave max_tuple + 1D sweeps for rapid greedy upper bound
    best_st: tuple[int, int, int, int] | None = None
    best_cost = float("inf")
    best_prob = float("inf")

    for axis in range(4):
        hi = max_k[axis]
        if hi <= 0:
            continue
        grid = np.zeros((hi + 1, 4), dtype=np.int64)
        grid[:, axis] = np.arange(0, hi + 1, dtype=np.int64)
        probs = clf.predict_proba(build_matrix(grid))[:, 1]
        v_idx = np.flatnonzero(probs < threshold)
        if v_idx.size:
            idx = int(v_idx[0])
            st = (int(grid[idx, 0]), int(grid[idx, 1]), int(grid[idx, 2]), int(grid[idx, 3]))
            c = cost_scalar(st)
            if better(c, st, best_cost, best_st):
                best_cost, best_st, best_prob = c, st, float(probs[idx])

    # Backward shave from max_tuple
    cur = list(max_k)
    for _ in range(2):
        for axis in range(4):
            lo, hi = 0, cur[axis]
            bk = cur[axis]
            while lo <= hi:
                mid = (lo + hi) // 2
                probe = list(cur)
                probe[axis] = mid
                if eval_prob((probe[0], probe[1], probe[2], probe[3])) < threshold:
                    bk = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            cur[axis] = bk
    shaved = (cur[0], cur[1], cur[2], cur[3])
    s_prob = eval_prob(shaved)
    if s_prob < threshold:
        s_cost = cost_scalar(shaved)
        if better(s_cost, shaved, best_cost, best_st):
            best_cost, best_st, best_prob = s_cost, shaved, s_prob

    min_cost = best_cost

    def max_k_for_budget(var_idx: int, assigned: tuple, cur_cost: float, budget: float) -> int:
        if not math.isfinite(budget):
            return max_k[var_idx]
        rem = budget - cur_cost
        if rem < 0.0:
            return -1
        b_eff = w[var_idx]
        for u in range(4):
            ku = assigned[u]
            if ku:
                b_eff += cross[var_idx][u] * ku
        a = alpha[var_idx]
        k_hi = min(
            max_k[var_idx],
            int(math.floor((-b_eff + math.sqrt(max(0.0, b_eff * b_eff + 4.0 * a * rem))) / (2.0 * a) + 1e-12)),
        )
        while k_hi >= 0 and (b_eff * k_hi + a * k_hi * k_hi) > rem + 1e-12:
            k_hi -= 1
        return k_hi

    pq: list[tuple[float, int, int, tuple]] = [(0.0, 0, 0, (None, None, None, None))]
    counter = 0
    visited = 0

    while pq:
        lb, _, _, assigned = heapq.heappop(pq)
        if best_st is not None and lb > min_cost + tie_eps:
            break
        visited += 1
        unassigned = [i for i in range(4) if assigned[i] is None]
        budget = min_cost + tie_eps if best_st is not None else float("inf")

        ubs: dict[int, int] = {}
        dead = False
        for v in unassigned:
            kh = max_k_for_budget(v, assigned, lb, budget)
            if kh < 0:
                dead = True
                break
            ubs[v] = kh
        if dead:
            continue

        opt_st = (
            assigned[0] if assigned[0] is not None else ubs[0],
            assigned[1] if assigned[1] is not None else ubs[1],
            assigned[2] if assigned[2] is not None else ubs[2],
            assigned[3] if assigned[3] is not None else ubs[3],
        )
        if eval_prob(opt_st) >= threshold:
            continue

        if len(unassigned) == 1:
            lv = unassigned[0]
            n_leaf = ubs[lv] + 1
            mat = np.empty((n_leaf, 4), dtype=np.int64)
            for i in range(4):
                mat[:, i] = np.arange(0, n_leaf, dtype=np.int64) if i == lv else int(assigned[i])
            probs = clf.predict_proba(build_matrix(mat))[:, 1]
            v_idx = np.flatnonzero(probs < threshold)
            if v_idx.size == 0:
                continue
            v_mat = mat[v_idx]
            costs = eval_costs(v_mat)
            for ri in range(v_idx.size):
                cc = float(costs[ri])
                if best_st is not None and cc > min_cost + tie_eps:
                    break
                row = v_mat[ri]
                cst = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
                if better(cc, cst, best_cost, best_st):
                    best_cost, best_st, best_prob = cc, cst, float(probs[int(v_idx[ri])])
                    if cc < min_cost:
                        min_cost = cc
            continue

        target = min(unassigned, key=lambda v: (ubs[v] + 1, v))
        b_eff = w[target]
        for u in range(4):
            ku = assigned[u]
            if ku:
                b_eff += cross[target][u] * ku
        a_t = alpha[target]
        alist = list(assigned)
        ndepth = -(4 - len(unassigned) + 1)

        for kv in range(0, ubs[target] + 1):
            new_lb = lb + (b_eff * kv + a_t * kv * kv if kv else 0.0)
            if best_st is not None and new_lb > min_cost + tie_eps:
                break
            alist[target] = kv
            counter += 1
            heapq.heappush(pq, (new_lb, ndepth, counter, (alist[0], alist[1], alist[2], alist[3])))

    if best_st is None:
        return format_fail(visited)
    return format_ok("PASSED", best_st, best_prob, best_cost, visited)


def main() -> None:
    parser = argparse.ArgumentParser(description="CausalBound Oracle Solver")
    parser.add_argument("--mode", choices=["full", "policy", "recourse"], default="full")
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "policy":
        assert args.input is not None
        req = json.loads(args.input.read_text(encoding="utf-8"))
        out = solve_policy_dict(req)
    elif args.mode == "recourse":
        assert args.input is not None
        cfg = json.loads((args.data_dir / "scm_config.json").read_text(encoding="utf-8"))
        clf = joblib.load(args.data_dir / "model.joblib")
        case = json.loads(args.input.read_text(encoding="utf-8"))
        out = solve_recourse_dict(case, clf, cfg)
    else:
        req = json.loads((args.data_dir / "slices_dataset.json").read_text(encoding="utf-8"))
        cfg = json.loads((args.data_dir / "scm_config.json").read_text(encoding="utf-8"))
        clf = joblib.load(args.data_dir / "model.joblib")
        cases = json.loads((args.data_dir / "merchant_cases.json").read_text(encoding="utf-8"))
        out = {
            "schema_version": "1.0.0",
            "policy_result": solve_policy_dict(req),
            "recourse_results": [solve_recourse_dict(c, clf, cfg) for c in cases],
        }

    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
