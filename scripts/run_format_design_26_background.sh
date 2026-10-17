
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/mnt/lustre-client/huoda/venv/llm-sog"
RUN_ROOT="${RUN_ROOT:-${ROOT}/logs/e2e_runs}"
MODEL="${MODEL:-deepseek-v4-pro}"
MODEL_CONFIG="${MODEL_CONFIG:-${ROOT}/llm_models.yaml}"
FEATURE_DIR="${FEATURE_DIR:-/mnt/lustre-client/huoda/selected_mtx}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [existing-or-new-run-dir]" >&2
    exit 2
fi

if [[ $# -eq 1 ]]; then
    RUN_DIR="$(realpath -m "$1")"
else
    STAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="${RUN_ROOT}/${STAMP}_${MODEL}_format26"
fi

mkdir -p "${RUN_DIR}/connectivity"

# Required project environment. Keep activation in this shell so the detached
# Python process inherits the exact interpreter and dependencies.
source "${VENV}/bin/activate"

# Paid connectivity probing is opt-in. Local prompt/schema tests should be used
# before a batch so model tokens are reserved for actual format design.
if [[ "${RUN_CONNECTIVITY:-0}" == "1" ]] && \
   ! compgen -G "${RUN_DIR}/connectivity/*_model_connectivity/summary.json" >/dev/null; then
    PYTHONPATH="${ROOT}/python" python3 "${ROOT}/python/test/test_run_llm.py" \
        --model "${MODEL}" \
        --config "${MODEL_CONFIG}" \
        --log-dir "${RUN_DIR}/connectivity" \
        > "${RUN_DIR}/connectivity_test.log" 2>&1
fi

RESUME_ARGS=()
if [[ -f "${RUN_DIR}/run_manifest.json" ]]; then
    RESUME_ARGS+=(--resume)
fi

BATCH_LOG="${RUN_DIR}/format_design_batch.log"
nohup env PYTHONPATH="${ROOT}/python" \
    python3 "${ROOT}/python/test/run_format_design_batch.py" \
    --feature-dir "${FEATURE_DIR}" \
    --run-root "${RUN_ROOT}" \
    --run-dir "${RUN_DIR}" \
    --model "${MODEL}" \
    --model-config "${MODEL_CONFIG}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    "${RESUME_ARGS[@]}" \
    > "${BATCH_LOG}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${RUN_DIR}/format_design_batch.pid"

echo "Started format-design batch."
echo "  PID:       ${PID}"
echo "  Run dir:   ${RUN_DIR}"
echo "  Log:       ${BATCH_LOG}"
echo "  Monitor:   tail -f ${BATCH_LOG}"
echo "  Progress:  find ${RUN_DIR}/formats -maxdepth 1 -name '*.format.json' | wc -l"
