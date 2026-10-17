#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/run_claude_opus_5_format_design_26.sh"
MODEL="claude-opus-5-5"
export MODEL

ACTION="${1:-start}"
if [[ "${ACTION}" == "smoke" ]]; then
    shift
    DATASETS_CSV="DD"
    MAX_ATTEMPTS="1"
    export DATASETS_CSV MAX_ATTEMPTS
    if [[ -z "${1:-}" ]]; then
        RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/../logs/e2e_runs}"
        RUN_DIR="${RUN_ROOT}/$(date +%Y%m%d_%H%M%S)_claude_opus_5_5_smoke_DD"
        export RUN_DIR
        set -- "${RUN_DIR}"
    fi
    exec bash "${BASE_SCRIPT}" start "$@"
fi

if [[ "${ACTION}" == "resume" ]]; then
    shift
    if [[ -z "${1:-}" ]]; then
        echo "Usage: $0 resume RUN_DIR" >&2
        exit 2
    fi
    unset DATASETS_CSV
    MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
    export MAX_ATTEMPTS
    # The base launcher detects the existing manifest, clears the prior stop
    # marker, and starts the batch CLI with --resume. Accepted datasets are
    # reused; failed and pending datasets are processed.
    exec bash "${BASE_SCRIPT}" start "$@"
fi

if [[ "${ACTION}" == "start" && -z "${DATASETS_CSV:-}" ]]; then
    # Full campaign: use the base launcher's canonical 26-dataset list.
    exec bash "${BASE_SCRIPT}" "$@"
fi

exec bash "${BASE_SCRIPT}" "$@"
