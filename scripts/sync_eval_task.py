#!/usr/bin/env python3
"""Lockstep asset synchronizer for eval_task/environment/data and eval_task/tests/cases.

Runs build_assets.py first, copies pristine inputs to both isolated locations,
and computes eval_task/tests/cases/expected.json directly from the reference oracle.
Never hand-edit generated fixtures in eval_task/.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_assets import main as build_assets_main  # noqa: E402
from causalbound.engine_a.slice_optimizer import optimize_policy  # noqa: E402
from causalbound.engine_b.bnb_recourse import solve_recourse  # noqa: E402
from causalbound.schemas import MerchantRecourseCase, PolicyOptimizationRequest  # noqa: E402
from causalbound.scm import SCMModelSpec  # noqa: E402

ASSET_FILENAMES = (
    "scm_config.json",
    "model.joblib",
    "slices_dataset.json",
    "merchant_cases.json",
)


def main() -> None:
    # 1. Regenerate canonical assets in data/
    build_assets_main()

    data_dir = REPO_ROOT / "data"
    env_data_dir = REPO_ROOT / "eval_task" / "environment" / "data"
    cases_dir = REPO_ROOT / "eval_task" / "tests" / "cases"

    env_data_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    # 2. Mirror pristine inputs to both runtime and isolated verifier directories
    for fname in ASSET_FILENAMES:
        src = data_dir / fname
        shutil.copy2(src, env_data_dir / fname)
        shutil.copy2(src, cases_dir / fname)

    # 3. Compute exact ground-truth expected.json via reference oracles
    start = time.perf_counter()
    policy_req = PolicyOptimizationRequest.model_validate(
        json.loads((cases_dir / "slices_dataset.json").read_text(encoding="utf-8"))
    )
    policy_res = optimize_policy(policy_req).to_strict_dict()

    scm_spec = SCMModelSpec.from_file(cases_dir / "scm_config.json")
    clf = joblib.load(cases_dir / "model.joblib")
    raw_cases = json.loads((cases_dir / "merchant_cases.json").read_text(encoding="utf-8"))

    recourse_results = [
        solve_recourse(MerchantRecourseCase.model_validate(raw), clf, scm_spec).to_strict_dict()
        for raw in raw_cases
    ]
    elapsed = time.perf_counter() - start

    expected_payload = {
        "schema_version": "1.0.0",
        "policy_result": policy_res,
        "recourse_results": recourse_results,
    }
    (cases_dir / "expected.json").write_text(
        json.dumps(expected_payload, indent=2) + "\n", encoding="utf-8"
    )

    print("EVAL_TASK ASSETS SYNCHRONIZED IN LOCKSTEP")
    print(f"  Oracle compute time : {elapsed:.3f}s (budget < 30.0s)")
    print(f"  Policy status       : {policy_res['audit_status']} (loss={policy_res['total_loss_cents']}c)")
    print(f"  Recourse count      : {len(recourse_results)}")


if __name__ == "__main__":
    main()
