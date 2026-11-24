#!/usr/bin/env bash
# eval_task/solution/solve.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-}"
if [ -z "${APP_DIR}" ]; then
    if [ -d /app ] && [ -w /app ]; then
        APP_DIR="/app"
    else
        APP_DIR="${SCRIPT_DIR}"
    fi
fi

if [ "${APP_DIR}" = "/app" ]; then
    DATA_DIR="${DATA_DIR:-/app/data}"
    OUTPUT_DIR="${OUTPUT_DIR:-/app/output}"
else
    DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/../environment/data}"
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
fi

mkdir -p "${OUTPUT_DIR}" 2>/dev/null || true

cleanup() {
    rm -rf /tmp/causalbound_solver_scratch_* 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Install standalone solver entrypoint into /app/solver.py for runtime Audit B/C/D probes
if [ -w "${APP_DIR}" ]; then
    cp "${SCRIPT_DIR}/solve.py" "${APP_DIR}/solver.py"
    chmod +x "${APP_DIR}/solver.py"
    SOLVER_BIN="${APP_DIR}/solver.py"
else
    SOLVER_BIN="${SCRIPT_DIR}/solve.py"
fi

PYTHON_BIN="python3"
if [ -f "${SCRIPT_DIR}/../../.venv/bin/python" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/../../.venv/bin/python"
fi

"${PYTHON_BIN}" "${SOLVER_BIN}" \
    --mode full \
    --data-dir "${DATA_DIR}" \
    --output "${OUTPUT_DIR}/solution.json"

echo "SOLVER COMPLETED"
exit 0
