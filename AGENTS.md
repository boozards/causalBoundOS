# CausalBound Master Agent Contract (`AGENTS.md`)

> Single source of operational rules governing all AI coding assistants, solvers, verifiers, and CI pipelines in this repository.

---

## 1. Prime Directives

1. **Verify via Execution, Never via Assumptions:**
   - Never report a task complete without executing the test suite or verification script and inspecting both exit code and `stdout`/`stderr`.
   - Never claim an algorithm is optimal or bug-free without validating against edge cases, stress workloads, and brute-force small-scale oracles.
2. **Zero Regressions & Minimal Diffs:**
   - Keep changes tightly scoped to the active milestone. Do not refactor unrelated modules or reformat untouched files.
   - Maintain strict backward compatibility with locked schemas, signatures, and CLIs defined in `docs/SPEC_CONTRACT.md`.
3. **Hermetic Environments & Path Portability:**
   - Never hardcode absolute host paths (`/Users/...`, `C:\...`). Resolve paths dynamically relative to `SCRIPT_DIR` or environment-injected base directories.
   - Clean up scratch files, caches, and temporary test artifacts via cleanup traps before exiting.

---

## 2. Non-Root Container & Shell Hygiene

1. **Unprivileged Execution & Resource Caps:**
   - Target sandbox profile: non-root user (`UID 1000` / `nobody`), read-only `/` and `/logs`, **1 vCPU**, **2048 MB RAM**.
   - Verifier timeout: `300s`. Reference Oracle solvers MUST complete in `< 30s` (`< 10%` of verifier budget).
2. **Multi-Tier Log Directory Fallback:**
   - Never assume `/logs/verifier` is writable. Resolve log paths dynamically:
     ```bash
     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
     if [ -w /logs ] 2>/dev/null || ([ ! -e /logs ] && mkdir -p /logs/verifier 2>/dev/null); then
         LOG_DIR="/logs/verifier"
     elif [ -d /tmp ] && mkdir -p /tmp/verifier_logs 2>/dev/null; then
         LOG_DIR="/tmp/verifier_logs"
     else
         LOG_DIR="${SCRIPT_DIR}/.logs_verifier"
         mkdir -p "${LOG_DIR}" 2>/dev/null || true
     fi
     ```
3. **Guarded Bash Redirections & Dynamic CLI Probing:**
   - Never execute bare redirections (`echo 1 > /path/file 2>/dev/null`) without guarding `[ -d /path ]` first.
   - Never pass optional pytest plugins (`--ctrf`, `--html`) unconditionally; probe `pytest --help` first to prevent exit code `4`.
   - Emit explicit uppercase completion sentinels (`SOLVER COMPLETED`, `VERIFIER PASSED`, `VERIFIER FAILED`) and clean scratch files via `trap cleanup EXIT INT TERM`.

---

## 3. Numeric Safety, Boundaries & Strict Schema Guards

1. **Mandatory Finiteness & Ban on Raw Float `==`:**
   - Gate every float check with `math.isfinite()` before comparison.
   - Use justified tolerances only: `1e-4` for cross-arch float summation / non-linear SCM DAG residuals; `1e-9` for exact numeric derivations; `tol_opt = 0.05` ($< \min_i \text{step cost} = 1.0$) for Engine B one-sided optimality bounds.
   - Store and compute financial loss in exact integer cents (`int`) or `decimal.Decimal` — never raw IEEE 754 floats.
   - Use exact `fractions.Fraction` for rational rate comparisons (`TPR`, `FPR`, pairwise `delta`).
2. **The 1-ULP Mid-Boundary Rule:**
   - Never place synthetic timestamps, scores, or threshold boundaries on exact integer/grid edges (`.000` or exact `0.05` multiples).
   - Center all synthetic timestamps at `.500` UTC ISO-8601 (`%Y-%m-%dT%H:%M:%S.%fZ` with rounded milliseconds and explicit `timezone.utc`).
   - Offset all synthetic risk scores by $\ge 0.005$ from candidate threshold grid values and ECE bin edges.
3. **Strict Key-Set Invariance:**
   - Validate exact key sets (`set(actual.keys()) == set(expected.keys())`) on all top-level and nested JSON payloads. Both missing keys and extra leaked keys fail verification.
   - On `FAILED_AUDIT`, all selected threshold/action fields MUST be strictly `null` (`None`), while diagnostic fields (`slice_ece`, `violated_constraints`) remain populated.

---

## 4. Algorithmic & Search Standards

1. **No Unbounded Cartesian Searches:**
   - Engine A couples 16 slices via pairwise `delta` and global `max_fp`; solve exactly via `[L, L + delta]` TPR windowing + Multiple-Choice Knapsack DP with exact Pareto-dominance pruning.
   - Engine B spans $\approx 2.20 \times 10^7$ discrete candidates per instance with coupled quadratic cross-terms ($\beta_{ij} > 0$); solve via Priority-Queue Branch-and-Bound (`heapq`) with greedy upper-bound seeding, MRV variable ordering, monotonic-classifier optimistic feasibility pruning, `@lru_cache` memoization, and vectorized NumPy leaf evaluation.
2. **Explicit 3-Tier Determinism:**
   - Always sort unsorted input collections explicitly (`slice_id` / feature name ascending) before search.
   - Enforce the locked 3-tier tie-breaking ladder (Primary Objective $\to$ Parameter Sum $\to$ Lexicographical Tuple).

---

## 5. Verification Integrity, Anti-Cheat & Version Locks

1. **Five-Stage Verification Ladder & Audits A–D:**
   - Structure verifiers in ascending dependency order: Existence $\to$ Schema $\to$ Invariants $\to$ Optimality $\to$ Exact Match.
   - Enforce **Audit A (Anti-Slack)**, **Audit B (Anti-Collapse)**, **Audit C (Anti-Memorization via runtime generation)**, **Audit D (Failure-State Bi-Directional)**, and 3-tier regimes (**Nop**, **Oracle**, **$\pm 1$ Perturbation**).
   - Pair every filter/branch with both Positive and Negative dataset witnesses to survive automated mutation probes.
2. **Ground-Truth Isolation & Binary-Safe Scanning:**
   - Load ground-truth fixtures into memory and immediately purge them from disk before executing untrusted candidate code.
   - Exclude `solution/`, `tests/`, and `*.expected.json` from container build contexts via `.dockerignore`.
   - Scan candidate files in raw binary mode (`open(path, "rb")`) — never text mode with `errors="ignore"`.
   - Wrap all candidate subprocess calls with explicit `timeout`, `capture_output=True`, `returncode == 0`, and artifact existence checks.
3. **Lockstep Asset Regeneration & Serialization Pinning:**
   - Pin exact Python minor (`3.13.x`) and package versions (`scikit-learn==1.6.1`, `numpy==2.2.3`, `pandas==2.2.3`, `scipy==1.15.2`, `joblib==1.4.2`).
   - Regenerate `scm_config.json`, `model.joblib`, datasets, and expected outputs in lockstep exclusively through `scripts/build_assets.py` (`seed=42`). Never hand-edit generated JSON fixtures.

---

## 6. Git & PR Protocol

- Run `git status` and `git diff` before every commit; use atomic conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`) explaining root cause and rationale.
- Check `gh pr list` before creating a PR; never force-push over shared branches.
