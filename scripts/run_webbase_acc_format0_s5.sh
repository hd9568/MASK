#!/usr/bin/env bash
# Rerun only webbase-1M's historical ACC-BIT-TCF format0; reuse old format1/2 offline.
set -euo pipefail

ROOT=/home/huoda/sparsene
RUN_DIR="${RUN_DIR:-${ROOT}/logs/e2e_runs/20261108_webbase_acc_format0_s5}"
FEATURE_DIR="${FEATURE_DIR:-/mnt/lustre-client/huoda/selected_mtx}"
VENV="${VENV:-/mnt/lustre-client/huoda/venv/llm-sog}"
NCU_BIN="${NCU_BIN:-/home/huoda/local/cuda/nsight-compute-2025.1.0/ncu}"
export SPARSEKG_TMP_DIR="${SPARSEKG_TMP_DIR:-/mnt/lustre-client/huoda/sparsekg_tmp}"
export SPARSEKG_GPU_LOCK_FILE="${SPARSEKG_GPU_LOCK_FILE:-/tmp/sparsekg_sm80_gpu_benchmark.lock}"

usage() {
  echo "Usage: bash $0 {check|run|resume}"
  echo "Foreground output is streamed and saved in ${RUN_DIR}/driver.log"
}

action="${1:-}"
[[ "${action}" == check || "${action}" == run || "${action}" == resume ]] || { usage; exit 2; }
[[ -f "${VENV}/bin/activate" ]] || { echo "Missing project venv" >&2; exit 1; }
[[ -f "${FEATURE_DIR}/webbase-1M.features.v3.json" ]] || { echo "Missing webbase-1M features" >&2; exit 1; }
[[ -f "${NCU_BIN}" ]] || { echo "Missing NCU executable" >&2; exit 1; }
[[ -n "${OPENLUX_API_KEY:-}" ]] || { echo "OPENLUX_API_KEY is not set" >&2; exit 1; }
if [[ "${SPARSEKG_TMP_DIR}" == /mnt/lustre-client/huoda/sparsekg_tmp ]]; then
  mountpoint -q /mnt/lustre-client || { echo "Lustre is not mounted" >&2; exit 1; }
fi
mkdir -p "${SPARSEKG_TMP_DIR}/toolchain_tmp"
export TMPDIR="${SPARSEKG_TMP_DIR}/toolchain_tmp" TMP="${SPARSEKG_TMP_DIR}/toolchain_tmp" TEMP="${SPARSEKG_TMP_DIR}/toolchain_tmp"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/python:${ROOT}/python/test${PYTHONPATH:+:${PYTHONPATH}}"

python -c 'from sparsene.utils.sparsene_reference import load_sparsene_reference; r=load_sparsene_reference("webbase-1M"); assert r.variant == "ACC-BIT-TCF"; print(f"Reference: {r.variant}, {r.milliseconds:.6f} ms; generated candidates will be measured separately.")'
if [[ "${action}" == check ]]; then
  echo "Preflight OK: webbase-1M, format0=ACC-BIT-TCF, strategy generated iterations=0..5, N=512 FP32, warmup=3 repeat=32."
  exit 0
fi
if [[ "${action}" == run && -e "${RUN_DIR}/run_manifest.json" ]]; then
  echo "Run already exists; use resume or choose a new RUN_DIR" >&2; exit 1
fi
if [[ "${action}" == resume && ! -f "${RUN_DIR}/run_manifest.json" ]]; then
  echo "No run to resume: ${RUN_DIR}" >&2; exit 1
fi
mkdir -p "${RUN_DIR}"
command=(
  python -u "${ROOT}/python/test/run_joint_format_strategy_feedback.py"
  --run-dir "${RUN_DIR}"
  --feature-dir "${FEATURE_DIR}"
  --datasets webbase-1M
  --format-iterations 1
  --initial-format sparsene_historical_best
  --strategy-agent-iterations 5
  --incumbent-dir "${ROOT}/logs/e2e_runs/incumbents/a100_a800_n512_fp32"
  --strategy-provider openai_http
  --strategy-model deepseek-v4-pro
  --strategy-max-tokens 16384
  --strategy-timeout-sec 600
  --model-config "${ROOT}/llm_models.yaml"
  --n 512 --warmup 3 --repeat 32 --measurement-runs 3
  --ncu-bin "${NCU_BIN}"
)
[[ "${action}" == resume ]] && command+=(--resume)
"${command[@]}" 2>&1 | tee -a "${RUN_DIR}/driver.log"
