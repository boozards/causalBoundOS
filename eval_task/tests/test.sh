#!/usr/bin/env bash
# eval_task/tests/test.sh — Non-root Harbor verifier entrypoint
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Tier 1: Multi-tier writable log directory fallback (non-root UID 1000 safe)
if [ -w /logs ] 2>/dev/null || ([ ! -e /logs ] && mkdir -p /logs/verifier 2>/dev/null); then
    LOG_DIR="/logs/verifier"
elif [ -d /tmp ] && mkdir -p /tmp/verifier_logs 2>/dev/null; then
    LOG_DIR="/tmp/verifier_logs"
else
    LOG_DIR="${SCRIPT_DIR}/.logs_verifier"
    mkdir -p "${LOG_DIR}" 2>/dev/null || true
fi

echo "Harness log directory: ${LOG_DIR}"

# Tier 2: Dynamic pytest plugin probing (avoids exit code 4 when plugins absent)
EXTRA_PYTEST_ARGS=()

PYTEST_BIN="pytest"
if [ -f "${SCRIPT_DIR}/../../.venv/bin/pytest" ]; then
    PYTEST_BIN="${SCRIPT_DIR}/../../.venv/bin/pytest"
fi

if "${PYTEST_BIN}" --help 2>&1 | grep -q -- "--ctrf"; then
    EXTRA_PYTEST_ARGS+=("--ctrf" "${LOG_DIR}/ctrf-report.json")
fi

if "${PYTEST_BIN}" --help 2>&1 | grep -q -- "--html"; then
    EXTRA_PYTEST_ARGS+=("--html=${LOG_DIR}/report.html" "--self-contained-html")
fi

# Tier 3: Execute verification suite
if [ ${#EXTRA_PYTEST_ARGS[@]} -gt 0 ]; then
    "${PYTEST_BIN}" -rA "${EXTRA_PYTEST_ARGS[@]}" "${SCRIPT_DIR}/test_outputs.py"
else
    "${PYTEST_BIN}" -rA "${SCRIPT_DIR}/test_outputs.py"
fi
PYTEST_EXIT=$?

# Tier 4: Guarded reward file output + explicit completion sentinel
if [ "${PYTEST_EXIT}" -eq 0 ]; then
    if [ -d "${LOG_DIR}" ]; then
        echo 1 > "${LOG_DIR}/reward.txt"
    fi
    if [ -d /logs/verifier ] && [ "${LOG_DIR}" != "/logs/verifier" ]; then
        echo 1 > /logs/verifier/reward.txt 2>/dev/null || true
    fi
    echo "VERIFIER PASSED"
    exit 0
else
    if [ -d "${LOG_DIR}" ]; then
        echo 0 > "${LOG_DIR}/reward.txt"
    fi
    if [ -d /logs/verifier ] && [ "${LOG_DIR}" != "/logs/verifier" ]; then
        echo 0 > /logs/verifier/reward.txt 2>/dev/null || true
    fi
    echo "VERIFIER FAILED (Exit code: ${PYTEST_EXIT})"
    exit "${PYTEST_EXIT}"
fi
