#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMAT_RUN_DIR="${FORMAT_RUN_DIR:-${ROOT}/logs/e2e_runs/20261014_002307_claude_opus_5_5_format26}"
TUNING_NAME="${TUNING_NAME:-strategy_claude_opus_5_5_3iter_w2_r16_backendfix_retry}"
FAILED_SOURCE_SUMMARY="${FAILED_SOURCE_SUMMARY:-${FORMAT_RUN_DIR}/${TUNING_NAME}_summary.json}"
ACTION="${1:-start}"

if [[ ! -f "${FAILED_SOURCE_SUMMARY}" ]]; then
    echo "ERROR: missing source summary: ${FAILED_SOURCE_SUMMARY}" >&2
    exit 1
fi

export FAILED_ONLY=1 FAILED_SOURCE_SUMMARY TUNING_NAME
exec bash "${ROOT}/scripts/run_claude_opus_5_5_strategy_26.sh" \
    "${ACTION}" "${FORMAT_RUN_DIR}"
