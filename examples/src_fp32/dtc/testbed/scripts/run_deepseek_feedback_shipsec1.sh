#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
RUN_SCRIPT="${ROOT}/examples/src_fp32/dtc/testbed/scripts/run_agent_feedback_multidataset.sh"
PY_DIR="${ROOT}/python"

DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-${1:-}}"
if [[ -z "${DEEPSEEK_API_KEY}" ]]; then
  echo "Usage: DEEPSEEK_API_KEY=<key> $0"
  echo "   or: $0 <api_key>"
  exit 1
fi

ITERATIONS="${ITERATIONS:-5}"
FEEDBACK_DATASET="${FEEDBACK_DATASET:-/mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx}"
MTX_CACHE_DIR="${SPARSENE_MTX_CACHE_DIR:-/mnt/lustre-client/huoda/mtx_cache}"
DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}"
STRATEGY_MAX_TOKENS="${STRATEGY_MAX_TOKENS:-4096}"
MODEL_TIMEOUT_SECONDS="${MODEL_TIMEOUT_SECONDS:-10800}"

echo "[1/3] Testing model connectivity"
(
  cd "${PY_DIR}"
  python test/check_strategy_model_connectivity.py \
    --deepseek-api-key "${DEEPSEEK_API_KEY}"
)

echo "[2/3] Running deepseek-v4-flash feedback loop (${ITERATIONS} iterations)"
SPARSENE_MTX_CACHE_DIR="${MTX_CACHE_DIR}" \
  timeout --foreground "${MODEL_TIMEOUT_SECONDS}" \
  bash "${RUN_SCRIPT}" \
  --iterations "${ITERATIONS}" \
  --strategy-provider openai_http \
  --strategy-api-key "${DEEPSEEK_API_KEY}" \
  --strategy-api-base-url "${DEEPSEEK_BASE_URL}" \
  --strategy-model deepseek-v4-flash \
  --strategy-max-tokens "${STRATEGY_MAX_TOKENS}" \
  --feedback-dataset "${FEEDBACK_DATASET}" \
  --run-label deepseek-v4-flash

echo "[3/3] Running deepseek-v4-pro feedback loop (${ITERATIONS} iterations)"
SPARSENE_MTX_CACHE_DIR="${MTX_CACHE_DIR}" \
  timeout --foreground "${MODEL_TIMEOUT_SECONDS}" \
  bash "${RUN_SCRIPT}" \
  --iterations "${ITERATIONS}" \
  --strategy-provider openai_http \
  --strategy-api-key "${DEEPSEEK_API_KEY}" \
  --strategy-api-base-url "${DEEPSEEK_BASE_URL}" \
  --strategy-model deepseek-v4-pro \
  --strategy-max-tokens "${STRATEGY_MAX_TOKENS}" \
  --feedback-dataset "${FEEDBACK_DATASET}" \
  --run-label deepseek-v4-pro
