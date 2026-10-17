#!/usr/bin/env bash
# set -x  # uncomment to debug unexpected exits
set -eu

# Log unexpected exits (signals, etc.)
_exit_trap() {
    local rc=$?
    if [ $rc -ne 0 ] && [ $rc -ne 143 ]; then  # 143 = SIGTERM from timeout/system
        echo "=== SCRIPT EXITED WITH CODE $rc at $(date) ===" >&2
    fi
}
trap _exit_trap EXIT

source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${DATASET_DIR:-/mnt/lustre-client/zhangkaige/selected_mtx}"

EPOCH=$(date +%Y%m%d_%H%M%S)
BATCH_DIR="${REPO_ROOT}/logs/logs_all_datasets_${EPOCH}"
FAILED_FILE="${BATCH_DIR}/end2end_all_failed.txt"

mkdir -p "${BATCH_DIR}"

cd "${REPO_ROOT}"

if [ ! -f "${REPO_ROOT}/llm_models.yaml" ]; then
    echo "ERROR: llm_models.yaml not found at ${REPO_ROOT}/llm_models.yaml"
    exit 1
fi

echo "Dataset dir:     ${DATASET_DIR}"
echo "Batch log dir:   ${BATCH_DIR}"
echo "Failed log:      ${FAILED_FILE}"
echo "PID:             $$"
echo ""

MTX_FILES=("${DATASET_DIR}"/*.mtx)
TOTAL=${#MTX_FILES[@]}

if [ "${TOTAL}" -eq 0 ]; then
    echo "ERROR: no .mtx files found in ${DATASET_DIR}"
    exit 1
fi

echo "Found ${TOTAL} matrices"
echo ""

rm -f "${FAILED_FILE}"

INDEX=0
for MTX_PATH in "${MTX_FILES[@]}"; do
    INDEX=$((INDEX + 1))
    MTX_NAME="$(basename "${MTX_PATH}")"
    RUN_LABEL="e2e_${MTX_NAME%.mtx}"

    echo "=== [${INDEX}/${TOTAL}] ${MTX_NAME} (label=${RUN_LABEL}) ==="

    if ! python "${REPO_ROOT}/python/test/end2end_pipeline.py" \
        -d "${DATASET_DIR}" \
        -m "${MTX_NAME}" \
        -l "${RUN_LABEL}" \
        --log-root "${BATCH_DIR}"; then
        echo "  -> FAILED"
        echo "${MTX_NAME}" >> "${FAILED_FILE}"
    else
        echo "  -> OK"
    fi

    echo ""
done

echo "===== DONE ====="
echo "Total: ${TOTAL}"

if [ -f "${FAILED_FILE}" ]; then
    FAILED_COUNT=$(wc -l < "${FAILED_FILE}" || echo 0)
    echo "Failed: ${FAILED_COUNT}"
    echo "Failed list:"
    cat "${FAILED_FILE}"
else
    echo "Failed: 0"
fi
