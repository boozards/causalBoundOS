"""CausalBound Harbor Verification Harness (Five-Stage Ladder + Audits A-D).

Architecture:
  - Load-Purge-Verify session fixture: loads pristine fixtures + expected.json into memory
    and purges expected.json from disk before executing any untrusted candidate subprocess.
  - Raw binary ("rb") anti-tamper scanner: zero drop-byte vulnerability.
  - Stage 1 (Existence) -> Stage 2 (Strict Key-Set Schema & Finiteness) ->
    Stage 3 (Hard Invariants & SCM Residuals <= 1e-4) ->
    Stage 4 (Optimality & One-Sided Tolerance < min_step_cost) ->
    Stage 5 (Exact 3-Tier Deterministic Match).
  - Audit A (Anti-Slack), Audit B (Anti-Collapse), Audit C (Dynamic Anti-Memorization),
    Audit D (Bi-Directional Failure States + Zero-Division & Small-N Fallbacks).
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
TESTS_DIR = Path(__file__).resolve().parent

output_file_env = os.environ.get("OUTPUT_FILE")
if output_file_env:
    OUTPUT_FILE = Path(output_file_env)
elif (APP_DIR / "output" / "solution.json").is_file():
    OUTPUT_FILE = APP_DIR / "output" / "solution.json"
elif (TESTS_DIR.parent / "solution" / "output" / "solution.json").is_file():
    OUTPUT_FILE = TESTS_DIR.parent / "solution" / "output" / "solution.json"
else:
    OUTPUT_FILE = APP_DIR / "output" / "solution.json"

candidate_solver_env = os.environ.get("CANDIDATE_SOLVER")
if candidate_solver_env:
    CANDIDATE_SOLVER = Path(candidate_solver_env)
elif (APP_DIR / "solver.py").is_file():
    CANDIDATE_SOLVER = APP_DIR / "solver.py"
elif (TESTS_DIR.parent / "solution" / "solve.py").is_file():
    CANDIDATE_SOLVER = TESTS_DIR.parent / "solution" / "solve.py"
else:
    CANDIDATE_SOLVER = APP_DIR / "solver.py"

CASES_DIR = Path(os.environ.get("TESTS_CASES_DIR", str(TESTS_DIR / "cases")))
PURGE_EXPECTED = os.environ.get("CAUSALBOUND_PURGE_EXPECTED", "0") == "1"

ROOT_KEYS = frozenset({"schema_version", "policy_result", "recourse_results"})
ENGINE_A_KEYS = frozenset(
    {
        "policy_id",
        "evaluated_at_utc",
        "audit_status",
        "selected_thresholds",
        "total_loss_cents",
        "total_fp",
        "total_fn",
        "global_tpr_spread",
        "slice_ece",
        "slice_metrics",
        "violated_constraints",
    }
)
SLICE_METRIC_KEYS = frozenset({"threshold", "fp", "fn", "tp", "tn", "tpr", "fpr", "loss_cents"})
ENGINE_B_KEYS = frozenset(
    {
        "merchant_id",
        "slice_id",
        "audit_status",
        "threshold",
        "initial_risk_score",
        "counterfactual_risk_score",
        "actions",
        "step_counts",
        "counterfactual_features",
        "total_cost",
        "dag_residuals",
        "states_visited",
        "violated_constraints",
    }
)
ACTION_KEYS = frozenset(
    {
        "dispute_reduction_bps",
        "kyc_tier_delta",
        "reserve_deposit_usd_delta",
        "settlement_hold_days_delta",
    }
)
FEATURE_KEYS = frozenset(
    {
        "account_tenure_months",
        "kyc_tier",
        "reserve_deposit_usd",
        "dispute_rate_bps",
        "settlement_hold_days",
        "effective_risk_ratio",
        "liquidity_coverage_index",
    }
)
RESIDUAL_KEYS = frozenset({"effective_risk_ratio", "liquidity_coverage_index"})
FORBIDDEN_TOKENS = (
    "expected.json",
    "tests/cases",
    "test_outputs.py",
    "pytest.approx",
)

# In-memory immutable store populated by load_and_purge_ground_truth
STATE: dict[str, Any] = {}


def _assert_finite(value: Any, name: str) -> float:
    assert value is not None, f"{name} is None"
    assert not isinstance(value, bool) and isinstance(value, (int, float)), (
        f"{name} must be numeric, got {type(value).__name__}"
    )
    fval = float(value)
    assert math.isfinite(fval), f"{name} is non-finite: {value}"
    return fval


def _assert_strict_keys(obj: dict[str, Any], expected: frozenset[str], ctx: str) -> None:
    assert isinstance(obj, dict), f"{ctx} must be dict, got {type(obj).__name__}"
    actual = set(obj.keys())
    missing = expected - actual
    extra = actual - expected
    assert not missing and not extra, f"[{ctx}] key mismatch: missing={sorted(missing)}, extra={sorted(extra)}"


def _compute_derived(tenure: float, kyc: float, reserve: float, dispute: float, hold: float):
    eff = math.exp(0.0025 * dispute) / (
        1.0 + 0.35 * kyc + 0.15 * math.log1p(tenure) + 0.08 * math.sqrt(hold)
    )
    liq = math.log1p(reserve) + 0.6 * math.sqrt(hold + 1.0) + 0.4 * kyc - 0.12 * math.sqrt(dispute)
    return eff, liq


def _compute_coupled_cost(step_counts: dict[str, int], cfg: dict[str, Any]) -> float:
    names = sorted(ACTION_KEYS)
    specs = cfg["action_specs"]
    cross = cfg["cross_weights"]
    total = 0.0
    for name in names:
        k = int(step_counts[name])
        total += float(specs[name]["linear_weight"]) * k + float(specs[name]["quadratic_weight"]) * (k * k)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            k1 = f"{names[i]}__{names[j]}"
            k2 = f"{names[j]}__{names[i]}"
            beta = float(cross[k1] if k1 in cross else cross[k2])
            total += beta * int(step_counts[names[i]]) * int(step_counts[names[j]])
    return total


def _run_candidate_cli(mode: str, payload: dict[str, Any], timeout_sec: int = 30) -> dict[str, Any]:
    """Defensive subprocess execution of untrusted /app/solver.py on dynamic test cases."""
    assert CANDIDATE_SOLVER.is_file(), f"Missing candidate CLI solver at {CANDIDATE_SOLVER}"
    with tempfile.TemporaryDirectory(prefix="cb_probe_") as tmpdir:
        tmp = Path(tmpdir)
        in_path = tmp / "input.json"
        out_path = tmp / "output.json"
        in_path.write_text(json.dumps(payload), encoding="utf-8")

        cmd = [
            sys.executable,
            str(CANDIDATE_SOLVER),
            "--mode",
            mode,
            "--data-dir",
            str(CASES_DIR),
            "--input",
            str(in_path),
            "--output",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        assert proc.returncode == 0, (
            f"Candidate solver exited {proc.returncode} in mode={mode}.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        assert out_path.is_file() and out_path.stat().st_size > 0, (
            f"Candidate solver returned 0 but failed to write {out_path}"
        )
        return json.loads(out_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Embedded Dynamic Oracle (for Audit C runtime anti-memorization checks)
# ---------------------------------------------------------------------------


def _oracle_solve_policy_exact(req: dict[str, Any]) -> dict[str, Any]:
    """Embedded exact Engine A oracle consuming pristine or dynamically generated inputs."""
    ordered = sorted(req["slices"], key=lambda s: s["slice_id"])
    thresholds = sorted({Decimal(str(t)) for t in req["candidate_thresholds"]})
    slice_ece: dict[str, float] = {}
    ece_bad: list[str] = []
    for sl in ordered:
        samples = sl["samples"]
        n = len(samples)
        if n < 5:
            ece = 0.0
        else:
            cnt = [0] * 10
            sc = [0.0] * 10
            lb = [0.0] * 10
            for s in samples:
                idx = min(max(int(math.floor(float(s["score"]) * 10)), 0), 9)
                cnt[idx] += 1
                sc[idx] += float(s["score"])
                lb[idx] += int(s["label"])
            ece = sum((c / n) * abs((lbl / c) - (s / c)) for c, s, lbl in zip(cnt, sc, lb) if c)
        slice_ece[sl["slice_id"]] = ece
        if ece > float(sl["max_ece"]):
            ece_bad.append(f"SLICE_ECE_EXCEEDED:{sl['slice_id']}")

    def fail(codes: list[str]):
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

    if ece_bad:
        return fail(ece_bad)

    per_slice = []
    fpr_bad = []
    for sl in ordered:
        samples = sl["samples"]
        pos = sum(1 for s in samples if int(s["label"]) == 1)
        neg = len(samples) - pos
        max_fpr = Fraction(str(sl["max_fpr"]))
        cands = []
        for t_dec in thresholds:
            tf = float(t_dec)
            tp = sum(1 for s in samples if float(s["score"]) >= tf and int(s["label"]) == 1)
            fp = sum(1 for s in samples if float(s["score"]) >= tf and int(s["label"]) == 0)
            fn = pos - tp
            tn = neg - fp
            tpr = Fraction(1, 1) if pos == 0 else Fraction(tp, pos)
            fpr = Fraction(0, 1) if neg == 0 else Fraction(fp, neg)
            if fpr <= max_fpr:
                loss = int(sl["fp_cost_cents"]) * fp + int(sl["fn_cost_cents"]) * fn
                cands.append((t_dec, tf, fp, fn, tp, tn, tpr, fpr, loss))
        if not cands:
            fpr_bad.append(f"SLICE_FPR_INFEASIBLE:{sl['slice_id']}")
        per_slice.append(cands)

    if fpr_bad:
        return fail(fpr_bad)

    delta = Fraction(str(req["pairwise_fairness_delta"]))
    max_fp = int(req["max_global_fp"])
    tprs = sorted({c[6] for cs in per_slice for c in cs})
    best_overall = None
    any_fair = False

    for lo in tprs:
        hi = lo + delta
        win = []
        ok = True
        for cs in per_slice:
            sub = [c for c in cs if lo <= c[6] <= hi]
            if not sub:
                ok = False
                break
            win.append(sub)
        if not ok:
            continue
        any_fair = True
        dp: dict[int, tuple] = {0: (0, Decimal("0"), (), ())}
        for cs in win:
            ndp: dict[int, tuple] = {}
            for pfp, (ploss, psum, ptup, pc) in dp.items():
                for c in cs:
                    nfp = pfp + c[2]
                    if nfp > max_fp:
                        continue
                    nv = (ploss + c[8], psum + c[0], ptup + (c[0],), pc + (c,))
                    cur = ndp.get(nfp)
                    if cur is None or (nv[0], nv[1], nv[2]) < (cur[0], cur[1], cur[2]):
                        ndp[nfp] = nv
            dp = ndp
            if not dp:
                break
        for val in dp.values():
            if best_overall is None or (val[0], val[1], val[2]) < (
                best_overall[0],
                best_overall[1],
                best_overall[2],
            ):
                best_overall = val

    if not any_fair:
        return fail(["PAIRWISE_FAIRNESS_INFEASIBLE"])
    if best_overall is None:
        return fail(["GLOBAL_FP_BUDGET_EXCEEDED"])

    tot_loss, _, _, chosen = best_overall
    sel = {sl["slice_id"]: c[1] for sl, c in zip(ordered, chosen)}
    mets = {
        sl["slice_id"]: {
            "threshold": c[1],
            "fp": c[2],
            "fn": c[3],
            "tp": c[4],
            "tn": c[5],
            "tpr": float(c[6]),
            "fpr": float(c[7]),
            "loss_cents": c[8],
        }
        for sl, c in zip(ordered, chosen)
    }
    chosen_tprs = [c[6] for c in chosen]
    return {
        "policy_id": req["policy_id"],
        "evaluated_at_utc": req["evaluated_at_utc"],
        "audit_status": "PASSED",
        "selected_thresholds": sel,
        "total_loss_cents": tot_loss,
        "total_fp": sum(c[2] for c in chosen),
        "total_fn": sum(c[3] for c in chosen),
        "global_tpr_spread": float(max(chosen_tprs) - min(chosen_tprs)),
        "slice_ece": slice_ece,
        "slice_metrics": mets,
        "violated_constraints": [],
    }


@pytest.fixture(scope="session", autouse=True)
def load_and_purge_ground_truth() -> None:
    """Anti-Cheat Load-Purge-Verify: cache ground truth in RAM and purge expected.json."""
    STATE["scm_config"] = json.loads((CASES_DIR / "scm_config.json").read_text(encoding="utf-8"))
    STATE["slices_dataset"] = json.loads(
        (CASES_DIR / "slices_dataset.json").read_text(encoding="utf-8")
    )
    STATE["merchant_cases"] = json.loads(
        (CASES_DIR / "merchant_cases.json").read_text(encoding="utf-8")
    )
    STATE["clf"] = joblib.load(CASES_DIR / "model.joblib")

    expected_path = CASES_DIR / "expected.json"
    if expected_path.is_file():
        STATE["expected"] = json.loads(expected_path.read_text(encoding="utf-8"))
        if PURGE_EXPECTED:
            try:
                expected_path.unlink()
            except OSError:
                pass
    else:
        raise RuntimeError(f"Missing immutable expected.json at {expected_path}")


# ---------------------------------------------------------------------------
# Anti-Tamper Security Gate (Raw Binary "rb" Scan — Zero Drop-Byte Bypass)
# ---------------------------------------------------------------------------


def test_00_binary_safe_anti_tamper_scan() -> None:
    """Scan candidate files in raw bytes mode to prevent non-UTF-8 drop-byte bypasses."""
    if not CANDIDATE_SOLVER.is_file():
        pytest.fail(f"Missing candidate solver at {CANDIDATE_SOLVER}")
    raw = CANDIDATE_SOLVER.read_bytes()
    for token in FORBIDDEN_TOKENS:
        assert token.encode("utf-8") not in raw, (
            f"Anti-tamper violation: forbidden token '{token}' found in {CANDIDATE_SOLVER}"
        )


# ---------------------------------------------------------------------------
# Stage 1: Artifact Existence
# ---------------------------------------------------------------------------


def test_01_stage1_artifact_existence() -> None:
    assert OUTPUT_FILE.is_file(), f"Missing output artifact at {OUTPUT_FILE}"
    assert OUTPUT_FILE.stat().st_size > 0, f"Output artifact {OUTPUT_FILE} is empty"
    STATE["candidate"] = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Stage 2: Strict Schema & Finiteness Validation
# ---------------------------------------------------------------------------


def test_02_stage2_strict_schema_and_finiteness() -> None:
    cand = STATE.get("candidate") or json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    _assert_strict_keys(cand, ROOT_KEYS, "root")

    pol = cand["policy_result"]
    _assert_strict_keys(pol, ENGINE_A_KEYS, "policy_result")
    assert pol["audit_status"] == "PASSED"
    assert isinstance(pol["total_loss_cents"], int) and not isinstance(pol["total_loss_cents"], bool)
    assert isinstance(pol["total_fp"], int) and isinstance(pol["total_fn"], int)
    _assert_finite(pol["global_tpr_spread"], "global_tpr_spread")
    for sid, ece_val in pol["slice_ece"].items():
        _assert_finite(ece_val, f"slice_ece[{sid}]")
    assert pol["slice_metrics"] is not None
    for sid, sm in pol["slice_metrics"].items():
        _assert_strict_keys(sm, SLICE_METRIC_KEYS, f"slice_metrics[{sid}]")
        _assert_finite(sm["threshold"], f"{sid}.threshold")
        _assert_finite(sm["tpr"], f"{sid}.tpr")
        _assert_finite(sm["fpr"], f"{sid}.fpr")
        assert isinstance(sm["loss_cents"], int)

    recs = cand["recourse_results"]
    assert isinstance(recs, list) and len(recs) == len(STATE["merchant_cases"])
    for r in recs:
        _assert_strict_keys(r, ENGINE_B_KEYS, f"recourse[{r.get('merchant_id')}]")
        _assert_finite(r["threshold"], "threshold")
        _assert_finite(r["initial_risk_score"], "initial_risk_score")
        _assert_finite(r["counterfactual_risk_score"], "counterfactual_risk_score")
        _assert_finite(r["total_cost"], "total_cost")
        _assert_strict_keys(r["actions"], ACTION_KEYS, "actions")
        _assert_strict_keys(r["step_counts"], ACTION_KEYS, "step_counts")
        _assert_strict_keys(r["counterfactual_features"], FEATURE_KEYS, "counterfactual_features")
        _assert_strict_keys(r["dag_residuals"], RESIDUAL_KEYS, "dag_residuals")


# ---------------------------------------------------------------------------
# Stage 3: Hard Invariants (Coupled Policy + SCM DAG + Immutability + Flips)
# ---------------------------------------------------------------------------


def test_03_stage3_engine_a_hard_invariants() -> None:
    cand = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))["policy_result"]
    req = STATE["slices_dataset"]
    allowed_thresholds = {round(float(t), 6) for t in req["candidate_thresholds"]}

    total_fp = 0
    total_fn = 0
    total_loss = 0
    tprs: list[Fraction] = []

    for sl in req["slices"]:
        sid = sl["slice_id"]
        t = cand["selected_thresholds"][sid]
        assert round(float(t), 6) in allowed_thresholds, f"{sid}: invalid threshold {t}"

        samples = sl["samples"]
        tp = sum(1 for s in samples if s["score"] >= t and s["label"] == 1)
        fp = sum(1 for s in samples if s["score"] >= t and s["label"] == 0)
        fn = sum(1 for s in samples if s["score"] < t and s["label"] == 1)
        tn = sum(1 for s in samples if s["score"] < t and s["label"] == 0)
        pos = tp + fn
        neg = fp + tn
        tpr = Fraction(1, 1) if pos == 0 else Fraction(tp, pos)
        fpr = Fraction(0, 1) if neg == 0 else Fraction(fp, neg)

        assert fpr <= Fraction(str(sl["max_fpr"])), f"{sid}: FPR {float(fpr)} > {sl['max_fpr']}"
        assert cand["slice_ece"][sid] <= sl["max_ece"], f"{sid}: ECE exceeds max_ece"

        loss = sl["fp_cost_cents"] * fp + sl["fn_cost_cents"] * fn
        sm = cand["slice_metrics"][sid]
        assert sm["fp"] == fp and sm["fn"] == fn and sm["tp"] == tp and sm["tn"] == tn
        assert sm["loss_cents"] == loss

        total_fp += fp
        total_fn += fn
        total_loss += loss
        tprs.append(tpr)

    assert total_fp <= req["max_global_fp"], f"Global FP {total_fp} > {req['max_global_fp']}"
    assert (max(tprs) - min(tprs)) <= Fraction(str(req["pairwise_fairness_delta"])), (
        "Pairwise TPR fairness delta violated"
    )
    assert cand["total_fp"] == total_fp
    assert cand["total_fn"] == total_fn
    assert cand["total_loss_cents"] == total_loss


def test_04_stage3_engine_b_hard_invariants_and_scm_residuals() -> None:
    cand_recs = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))["recourse_results"]
    cfg = STATE["scm_config"]
    clf = STATE["clf"]
    tol_res = float(cfg["tolerances"]["dag_residual_abs"])
    case_by_id = {c["merchant_id"]: c for c in STATE["merchant_cases"]}

    assert len(cand_recs) == len(case_by_id)
    seen_ids = set()

    for r in cand_recs:
        mid = r["merchant_id"]
        assert mid in case_by_id and mid not in seen_ids
        seen_ids.add(mid)
        base_case = case_by_id[mid]
        base = base_case["features"]
        cf = r["counterfactual_features"]

        # 1. Immutable root strictly unchanged
        assert cf["account_tenure_months"] == pytest.approx(base["account_tenure_months"], abs=1e-9)

        # 2. Discrete grid adherence, directionality, and action bounds
        for aname, aspec in cfg["action_specs"].items():
            k = r["step_counts"][aname]
            delta = r["actions"][aname]
            assert isinstance(k, int) and not isinstance(k, bool) and k >= 0
            assert 0 <= delta <= aspec["max_delta"]
            assert delta == pytest.approx(k * aspec["step"], abs=1e-9)

            target_feat = aspec["target_feature"]
            expected_feat_val = base[target_feat] + aspec["direction"] * delta
            assert cf[target_feat] == pytest.approx(expected_feat_val, abs=1e-9)
            assert cf[target_feat] >= 0.0

        # 3. Non-linear SCM DAG residual <= 1e-4
        exp_eff, exp_liq = _compute_derived(
            cf["account_tenure_months"],
            float(cf["kyc_tier"]),
            cf["reserve_deposit_usd"],
            cf["dispute_rate_bps"],
            float(cf["settlement_hold_days"]),
        )
        assert abs(cf["effective_risk_ratio"] - exp_eff) <= tol_res
        assert abs(cf["liquidity_coverage_index"] - exp_liq) <= tol_res

        # 4. Classifier flip verified independently on pristine model.joblib
        x_cf = np.array(
            [
                [
                    cf["account_tenure_months"],
                    float(cf["kyc_tier"]),
                    cf["reserve_deposit_usd"],
                    cf["dispute_rate_bps"],
                    float(cf["settlement_hold_days"]),
                    exp_eff,
                    exp_liq,
                ]
            ],
            dtype=np.float64,
        )
        prob_cf = float(clf.predict_proba(x_cf)[0, 1])
        assert prob_cf < base_case["threshold"], (
            f"{mid}: counterfactual prob {prob_cf:.6f} failed to flip below {base_case['threshold']}"
        )
        assert r["counterfactual_risk_score"] == pytest.approx(prob_cf, abs=1e-4)

        # 5. Reported cost matches coupled quadratic formula
        recomputed_cost = _compute_coupled_cost(r["step_counts"], cfg)
        assert r["total_cost"] == pytest.approx(recomputed_cost, abs=1e-6)


# ---------------------------------------------------------------------------
# Stage 4 & Stage 5: Global Optimality & Exact 3-Tier Match
# ---------------------------------------------------------------------------


def test_05_stage4_and_stage5_optimality_and_exact_tie_break() -> None:
    cand = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    exp = STATE["expected"]
    cfg = STATE["scm_config"]
    tol_opt = float(cfg["tolerances"]["optimality_one_sided"])

    # Engine A: Exact integer-cent optimality + exact 3-tier threshold match
    c_pol = cand["policy_result"]
    e_pol = exp["policy_result"]
    assert c_pol["total_loss_cents"] <= e_pol["total_loss_cents"], (
        f"Suboptimal Engine A loss: {c_pol['total_loss_cents']} > {e_pol['total_loss_cents']}"
    )
    assert c_pol["selected_thresholds"] == e_pol["selected_thresholds"], (
        "Engine A thresholds diverge from exact 3-tier deterministic tie-breaker"
    )

    # Engine B: One-sided optimality bound (< min step cost) + exact 3-tier step match
    exp_by_id = {r["merchant_id"]: r for r in exp["recourse_results"]}
    for c_rec in cand["recourse_results"]:
        mid = c_rec["merchant_id"]
        e_rec = exp_by_id[mid]
        assert c_rec["total_cost"] <= e_rec["total_cost"] + tol_opt, (
            f"{mid}: suboptimal recourse cost {c_rec['total_cost']} > "
            f"{e_rec['total_cost']} + {tol_opt}"
        )
        assert c_rec["step_counts"] == e_rec["step_counts"], (
            f"{mid}: step_counts {c_rec['step_counts']} != exact tie-break {e_rec['step_counts']}"
        )


# ---------------------------------------------------------------------------
# Audit A: Anti-Slack Proof
# ---------------------------------------------------------------------------


def test_06_audit_a_anti_slack_constraints_are_binding() -> None:
    """Confirm unconstrained optimum violates max_global_fp, pairwise delta, and max_fpr."""
    base_req = copy.deepcopy(STATE["slices_dataset"])
    constrained = STATE["expected"]["policy_result"]

    no_fp = _oracle_solve_policy_exact({**base_req, "max_global_fp": 100_000})
    assert no_fp["total_fp"] > base_req["max_global_fp"], "Audit A failed: max_global_fp is slack"

    no_delta = _oracle_solve_policy_exact({**base_req, "pairwise_fairness_delta": 1.0})
    assert no_delta["global_tpr_spread"] > base_req["pairwise_fairness_delta"], (
        "Audit A failed: pairwise_fairness_delta is slack"
    )

    relaxed_slices = [{**sl, "max_fpr": 1.0} for sl in base_req["slices"]]
    no_fpr = _oracle_solve_policy_exact({**base_req, "slices": relaxed_slices})
    assert no_fpr["selected_thresholds"] != constrained["selected_thresholds"], (
        "Audit A failed: per-slice max_fpr is slack"
    )


# ---------------------------------------------------------------------------
# Audit B: Anti-Collapse 3-Tier Tie-Breaking Test
# ---------------------------------------------------------------------------


def test_07_audit_b_anti_collapse_tie_breaking_ladder() -> None:
    """Ensure >= 2 equal-loss candidates survive all filters and exercise Tier 2 & Tier 3."""
    # Construct 2 identical slices where thresholds 0.30 and 0.60 yield identical TP/FP/FN/TN/loss
    # and another pair where (0.30, 0.70) vs (0.70, 0.30) tie on BOTH loss AND sum(t_m) = 1.00!
    tie_req = {
        "policy_id": "audit_b_tie_break",
        "evaluated_at_utc": "2026-11-03T10:00:00.500Z",
        "candidate_thresholds": [0.30, 0.70],
        "pairwise_fairness_delta": 0.50,
        "max_global_fp": 1,  # Forces exactly one slice at 0.30 (1 FP) and one at 0.70 (0 FP)!
        "slices": [
            {
                "slice_id": "SLICE_B",  # Intentionally unsorted input order
                "fp_cost_cents": 1000,
                "fn_cost_cents": 1000,
                "max_ece": 1.0,
                "max_fpr": 1.0,
                "samples": [
                    {"sample_id": "b1", "score": 0.85, "label": 1, "timestamp_utc": "2026-11-03T10:00:01.500Z"},
                    {"sample_id": "b2", "score": 0.50, "label": 1, "timestamp_utc": "2026-11-03T10:00:02.500Z"},
                    {"sample_id": "b3", "score": 0.50, "label": 0, "timestamp_utc": "2026-11-03T10:00:03.500Z"},
                    {"sample_id": "b4", "score": 0.15, "label": 0, "timestamp_utc": "2026-11-03T10:00:04.500Z"},
                ],
            },
            {
                "slice_id": "SLICE_A",
                "fp_cost_cents": 1000,
                "fn_cost_cents": 1000,
                "max_ece": 1.0,
                "max_fpr": 1.0,
                "samples": [
                    {"sample_id": "a1", "score": 0.85, "label": 1, "timestamp_utc": "2026-11-03T10:00:05.500Z"},
                    {"sample_id": "a2", "score": 0.50, "label": 1, "timestamp_utc": "2026-11-03T10:00:06.500Z"},
                    {"sample_id": "a3", "score": 0.50, "label": 0, "timestamp_utc": "2026-11-03T10:00:07.500Z"},
                    {"sample_id": "a4", "score": 0.15, "label": 0, "timestamp_utc": "2026-11-03T10:00:08.500Z"},
                ],
            },
        ],
    }
    # Both (SLICE_A=0.30, SLICE_B=0.70) and (SLICE_A=0.70, SLICE_B=0.30) have:
    #   loss = 2000c, FP = 1 <= 1, TPRs = (1.0, 0.5) spread = 0.5 <= 0.5, sum(t) = 1.00.
    # Tier 3 lexicographical order on sorted slice_ids (SLICE_A, SLICE_B) MUST pick (0.30, 0.70).
    out = _run_candidate_cli("policy", tie_req)
    _assert_strict_keys(out, ENGINE_A_KEYS, "audit_b")
    assert out["audit_status"] == "PASSED"
    assert out["selected_thresholds"] == {"SLICE_A": 0.30, "SLICE_B": 0.70}


# ---------------------------------------------------------------------------
# Audit C: Anti-Memorization (Dynamic Runtime Case Injection)
# ---------------------------------------------------------------------------


def test_08_audit_c_dynamic_runtime_anti_memorization() -> None:
    """Dynamically construct an unseen 4-slice policy at runtime and verify exact match."""
    rng = random.Random(int.from_bytes(os.urandom(4), "big"))
    dyn_slices = []
    for idx in range(4):
        samples = []
        for j in range(24):
            raw = rng.uniform(0.08, 0.92)
            # Push away from 0.05 grid boundaries (1-ULP rule)
            nearest = round(raw / 0.05) * 0.05
            if abs(raw - nearest) < 0.008:
                raw = min(0.94, max(0.06, nearest + 0.012))
            label = 1 if rng.random() < raw else 0
            samples.append(
                {
                    "sample_id": f"dyn_{idx}_{j}",
                    "score": round(raw, 6),
                    "label": label,
                    "timestamp_utc": f"2026-11-04T12:00:{j:02d}.500Z",
                }
            )
        dyn_slices.append(
            {
                "slice_id": f"DYN_SLICE_{idx}",
                "fp_cost_cents": rng.randint(10_000, 90_000),
                "fn_cost_cents": rng.randint(20_000, 120_000),
                "max_ece": 0.50,
                "max_fpr": 0.45,
                "samples": samples,
            }
        )

    dyn_req = {
        "policy_id": f"runtime_dyn_{rng.randint(1000, 9999)}",
        "evaluated_at_utc": "2026-11-04T12:30:00.500Z",
        "candidate_thresholds": [0.20, 0.35, 0.50, 0.65, 0.80],
        "pairwise_fairness_delta": 0.35,
        "max_global_fp": 18,
        "slices": dyn_slices,
    }
    expected_dyn = _oracle_solve_policy_exact(dyn_req)
    actual_dyn = _run_candidate_cli("policy", dyn_req)

    _assert_strict_keys(actual_dyn, ENGINE_A_KEYS, "audit_c_dynamic")
    assert actual_dyn["audit_status"] == expected_dyn["audit_status"]
    assert actual_dyn["selected_thresholds"] == expected_dyn["selected_thresholds"]
    assert actual_dyn["total_loss_cents"] == expected_dyn["total_loss_cents"]


# ---------------------------------------------------------------------------
# Audit D: Failure-State Bi-Directional Coverage & Zero-Division Fallbacks
# ---------------------------------------------------------------------------


def test_09_audit_d_failure_states_and_zero_division_guards() -> None:
    """Verify every distinct FAILED_AUDIT mode + pos_total==0 / neg_total==0 / N<5 guards."""
    # 1. ECE Exceeded -> FAILED_AUDIT + strict null selections + populated slice_ece
    ece_fail_req = {
        "policy_id": "fail_ece",
        "evaluated_at_utc": "2026-11-05T09:00:00.500Z",
        "candidate_thresholds": [0.50],
        "pairwise_fairness_delta": 1.0,
        "max_global_fp": 100,
        "slices": [
            {
                "slice_id": "BAD_CAL",
                "fp_cost_cents": 100,
                "fn_cost_cents": 100,
                "max_ece": 0.05,
                "max_fpr": 1.0,
                "samples": [
                    {"sample_id": f"e{i}", "score": 0.92, "label": 0, "timestamp_utc": f"2026-11-05T09:00:0{i}.500Z"}
                    for i in range(6)
                ],
            }
        ],
    }
    res_ece = _run_candidate_cli("policy", ece_fail_req)
    _assert_strict_keys(res_ece, ENGINE_A_KEYS, "fail_ece")
    assert res_ece["audit_status"] == "FAILED_AUDIT"
    assert res_ece["selected_thresholds"] is None
    assert res_ece["total_loss_cents"] is None
    assert res_ece["slice_metrics"] is None
    assert res_ece["slice_ece"]["BAD_CAL"] > 0.80
    assert "SLICE_ECE_EXCEEDED:BAD_CAL" in res_ece["violated_constraints"]

    # 2. Pairwise Fairness Infeasible -> FAILED_AUDIT
    fair_fail_req = {
        "policy_id": "fail_fairness",
        "evaluated_at_utc": "2026-11-05T09:05:00.500Z",
        "candidate_thresholds": [0.50],
        "pairwise_fairness_delta": 0.05,
        "max_global_fp": 100,
        "slices": [
            {
                "slice_id": "S_HIGH_TPR",
                "fp_cost_cents": 100,
                "fn_cost_cents": 100,
                "max_ece": 1.0,
                "max_fpr": 1.0,
                "samples": [
                    {"sample_id": "h1", "score": 0.85, "label": 1, "timestamp_utc": "2026-11-05T09:05:01.500Z"},
                ],
            },
            {
                "slice_id": "S_LOW_TPR",
                "fp_cost_cents": 100,
                "fn_cost_cents": 100,
                "max_ece": 1.0,
                "max_fpr": 1.0,
                "samples": [
                    {"sample_id": "l1", "score": 0.15, "label": 1, "timestamp_utc": "2026-11-05T09:05:02.500Z"},
                ],
            },
        ],
    }
    res_fair = _run_candidate_cli("policy", fair_fail_req)
    assert res_fair["audit_status"] == "FAILED_AUDIT"
    assert res_fair["selected_thresholds"] is None
    assert res_fair["violated_constraints"] == ["PAIRWISE_FAIRNESS_INFEASIBLE"]

    # 3. Global FP Budget Exceeded -> FAILED_AUDIT
    fp_fail_req = {
        "policy_id": "fail_global_fp",
        "evaluated_at_utc": "2026-11-05T09:10:00.500Z",
        "candidate_thresholds": [0.10],
        "pairwise_fairness_delta": 1.0,
        "max_global_fp": 0,
        "slices": [
            {
                "slice_id": "S_FP",
                "fp_cost_cents": 100,
                "fn_cost_cents": 100,
                "max_ece": 1.0,
                "max_fpr": 1.0,
                "samples": [
                    {"sample_id": "f1", "score": 0.85, "label": 0, "timestamp_utc": "2026-11-05T09:10:01.500Z"},
                ],
            }
        ],
    }
    res_fp = _run_candidate_cli("policy", fp_fail_req)
    assert res_fp["audit_status"] == "FAILED_AUDIT"
    assert res_fp["selected_thresholds"] is None
    assert res_fp["violated_constraints"] == ["GLOBAL_FP_BUDGET_EXCEEDED"]

    # 4. Zero-division & N < 5 guards: pos_total==0 -> TPR=1.0, neg_total==0 -> FPR=0.0, N<5 -> ECE=0.0
    zero_div_req = {
        "policy_id": "zero_div_guards",
        "evaluated_at_utc": "2026-11-05T09:15:00.500Z",
        "candidate_thresholds": [0.50],
        "pairwise_fairness_delta": 0.0,
        "max_global_fp": 5,
        "slices": [
            {
                "slice_id": "ALL_NEG_SMALL_N",
                "fp_cost_cents": 100,
                "fn_cost_cents": 100,
                "max_ece": 0.0,  # Only passes if N < 5 forces ECE = 0.0
                "max_fpr": 1.0,
                "samples": [
                    {"sample_id": "n1", "score": 0.90, "label": 0, "timestamp_utc": "2026-11-05T09:15:01.500Z"},
                    {"sample_id": "n2", "score": 0.10, "label": 0, "timestamp_utc": "2026-11-05T09:15:02.500Z"},
                ],
            },
            {
                "slice_id": "ALL_POS_SMALL_N",
                "fp_cost_cents": 100,
                "fn_cost_cents": 100,
                "max_ece": 0.0,
                "max_fpr": 0.0,  # Only passes if neg_total == 0 forces FPR = 0.0
                "samples": [
                    {"sample_id": "p1", "score": 0.90, "label": 1, "timestamp_utc": "2026-11-05T09:15:03.500Z"},
                ],
            },
        ],
    }
    res_zd = _run_candidate_cli("policy", zero_div_req)
    assert res_zd["audit_status"] == "PASSED"
    assert res_zd["slice_ece"] == {"ALL_NEG_SMALL_N": 0.0, "ALL_POS_SMALL_N": 0.0}
    assert res_zd["slice_metrics"]["ALL_NEG_SMALL_N"]["tpr"] == pytest.approx(1.0, abs=1e-9)
    assert res_zd["slice_metrics"]["ALL_POS_SMALL_N"]["fpr"] == pytest.approx(0.0, abs=1e-9)

    # 5. Engine B Infeasible Recourse -> FAILED_AUDIT + strict null fields
    impossible_merchant = copy.deepcopy(STATE["merchant_cases"][0])
    impossible_merchant["merchant_id"] = "MRC-IMPOSSIBLE"
    impossible_merchant["threshold"] = 0.00001
    res_rec_fail = _run_candidate_cli("recourse", impossible_merchant)
    _assert_strict_keys(res_rec_fail, ENGINE_B_KEYS, "fail_recourse")
    assert res_rec_fail["audit_status"] == "FAILED_AUDIT"
    assert res_rec_fail["actions"] is None
    assert res_rec_fail["total_cost"] is None
    assert res_rec_fail["violated_constraints"] == ["NO_FEASIBLE_FLIP"]
