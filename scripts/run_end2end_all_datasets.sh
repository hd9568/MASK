#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_ROOT="${REPO_ROOT}/logs"
DATASET_DIR="${DATASET_DIR:-/mnt/lustre-client/zhangkaige/selected_mtx}"
FAILED_FILE="${LOG_ROOT}/end2end_all_failed.txt"

cd "${REPO_ROOT}"

if [ ! -f "${REPO_ROOT}/llm_models.yaml" ]; then
    echo "ERROR: llm_models.yaml not found at ${REPO_ROOT}/llm_models.yaml"
    exit 1
fi

echo "Dataset dir:    ${DATASET_DIR}"
echo "Log root:       ${LOG_ROOT}"
echo "Failed log:     ${FAILED_FILE}"
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
        --log-root "${LOG_ROOT}"; then
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
