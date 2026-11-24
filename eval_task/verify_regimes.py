#!/usr/bin/env python3
"""3-Tier Verification Regime Harness for CausalBound Harbor Task.

Proves deterministically:
  1. NOP regime (empty output)                  -> FAILS decisively (reward = 0)
  2. ORACLE regime (solution/solve.sh)          -> PASSES 100% (reward = 1) in < 30s
  3. +/-1 PERTURBATION regime (single-step shift) -> FAILS decisively (reward = 0)
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent


def _run_verifier(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(EVAL_ROOT / "tests" / "test.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cb_regimes_") as tmpdir:
        tmp = Path(tmpdir)
        app_dir = tmp / "app"
        data_dir = app_dir / "data"
        out_dir = app_dir / "output"
        cases_dir = tmp / "cases"

        shutil.copytree(EVAL_ROOT / "environment" / "data", data_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def reset_cases() -> None:
            if cases_dir.exists():
                shutil.rmtree(cases_dir)
            shutil.copytree(EVAL_ROOT / "tests" / "cases", cases_dir)

        venv_bin = str(EVAL_ROOT.parent / ".venv" / "bin")
        path_env = f"{venv_bin}:{os.environ.get('PATH', '')}" if Path(venv_bin).is_dir() else os.environ.get("PATH", "")

        base_env = {
            **os.environ,
            "PATH": path_env,
            "APP_DIR": str(app_dir),
            "DATA_DIR": str(data_dir),
            "OUTPUT_DIR": str(out_dir),
            "OUTPUT_FILE": str(out_dir / "solution.json"),
            "CANDIDATE_SOLVER": str(app_dir / "solver.py"),
            "TESTS_CASES_DIR": str(cases_dir),
            "CAUSALBOUND_PURGE_EXPECTED": "1",
        }

        # --- Regime 1: NOP (empty artifact, no solver) ---
        reset_cases()
        (out_dir / "solution.json").write_text("{}", encoding="utf-8")
        nop_res = _run_verifier(base_env)
        assert nop_res.returncode != 0, "NOP regime unexpectedly passed!"
        print(f"[PASS] Regime 1 (NOP)          : rejected decisively (exit={nop_res.returncode})")

        # --- Regime 2: ORACLE (reference solve.sh) ---
        reset_cases()
        t0 = time.perf_counter()
        solve_res = subprocess.run(
            ["bash", str(EVAL_ROOT / "solution" / "solve.sh")],
            env=base_env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        oracle_elapsed = time.perf_counter() - t0
        assert solve_res.returncode == 0 and "SOLVER COMPLETED" in solve_res.stdout, (
            f"Oracle solve.sh failed:\n{solve_res.stdout}\n{solve_res.stderr}"
        )
        assert oracle_elapsed < 30.0, f"Oracle exceeded 10% budget: {oracle_elapsed:.2f}s >= 30s"

        oracle_ver = _run_verifier(base_env)
        assert oracle_ver.returncode == 0 and "VERIFIER PASSED" in oracle_ver.stdout, (
            f"Oracle verifier failed:\n{oracle_ver.stdout}\n{oracle_ver.stderr}"
        )
        print(
            f"[PASS] Regime 2 (ORACLE)       : passed 100% in {oracle_elapsed:.2f}s "
            f"(budget < 30.0s)"
        )

        # --- Regime 3: +1 Discrete Step Perturbation on Engine B ---
        reset_cases()
        sol_path = out_dir / "solution.json"
        perturbed = copy.deepcopy(json.loads(sol_path.read_text(encoding="utf-8")))
        first_rec = perturbed["recourse_results"][0]
        first_rec["step_counts"]["settlement_hold_days_delta"] += 1
        first_rec["actions"]["settlement_hold_days_delta"] += 1
        first_rec["counterfactual_features"]["settlement_hold_days"] += 1
        sol_path.write_text(json.dumps(perturbed, indent=2), encoding="utf-8")

        pert_ver = _run_verifier(base_env)
        assert pert_ver.returncode != 0, "+1 Perturbation regime unexpectedly passed!"
        print(
            f"[PASS] Regime 3 (+/-1 PERTURB) : rejected decisively (exit={pert_ver.returncode})"
        )

    print("ALL 3 VERIFICATION REGIMES PASSED")


if __name__ == "__main__":
    main()
