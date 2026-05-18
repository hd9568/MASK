#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
PY_DIR="${ROOT}/python"
VENV_PATH="${HOME}/llm-sog-venv"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PY_DIR}:${PYTHONPATH:-}"
export SPARSENE_MTX_CACHE_DIR="${SPARSENE_MTX_CACHE_DIR:-/mnt/lustre-client/huoda/mtx_cache}"

cd "${PY_DIR}"
python test/run_strategy_feedback_multidataset.py "$@"
