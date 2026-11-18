#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATFORM="a800"
MODEL="deepseek-v4-pro"
RUNS_ROOT="${ROOT}/logs/e2e_runs"
RUN_DIR="$(realpath -m "${RUN_DIR:-${RUNS_ROOT}/a800_deepseek_v4_pro_adaptive26}")"
case "${RUN_DIR}/" in "${RUNS_ROOT}/"*) ;; *) echo "RUN_DIR must be inside ${RUNS_ROOT}" >&2; exit 2 ;; esac
FEATURE_DIR="${FEATURE_DIR:-/mnt/lustre-client/huoda/selected_mtx}"
WORKERS="${WORKERS:-8}"
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid WORKERS' >&2; exit 2; }
export PATH="/usr/local/cuda/bin:${PATH}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" PYTHONDONTWRITEBYTECODE=1
if [[ "${PLATFORM}" == h100 ]]; then
 PYTHON=/usr/bin/python3
 export PYTHONPATH="${ROOT}/python:${ROOT}/python/test:/mnt/lustre-client/huoda/venv/llm-sog/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
 export SPARSEKG_CUDA_ARCH=90 CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++}"
 export SPARSEKG_CUTLASS_ROOT="${SPARSEKG_CUTLASS_ROOT:-/home/huoda/sparsekg-h100-baseline/cutlass-afa1772203677c5118fcd82537a9c8fefbcc7008}"
 NCU_BIN="${NCU_BIN:-/usr/local/cuda/bin/ncu}"
else
 PYTHON=/mnt/lustre-client/huoda/venv/llm-sog/bin/python
 export PYTHONPATH="${ROOT}/python:${ROOT}/python/test${PYTHONPATH:+:${PYTHONPATH}}"
 NCU_BIN="${NCU_BIN:-/home/huoda/local/cuda/nsight-compute-2025.1.0/ncu}"
fi
export SPARSEKG_GPU_LOCK_FILE="${SPARSEKG_GPU_LOCK_FILE:-/tmp/sparsekg_sm80_gpu_benchmark.lock}"
SCRATCH_ID="$(printf '%s' "${RUN_DIR}" | sha256sum | cut -c1-24)"
export SPARSEKG_TMP_DIR="/mnt/lustre-client/huoda/sparsekg_tmp/${PLATFORM}_adaptive_campaigns/${SCRATCH_ID}"
export TMPDIR="${SPARSEKG_TMP_DIR}/tmp" TMP="${SPARSEKG_TMP_DIR}/tmp" TEMP="${SPARSEKG_TMP_DIR}/tmp"
export CUDA_CACHE_PATH="${SPARSEKG_TMP_DIR}/cuda_cache" XDG_CACHE_HOME="${SPARSEKG_TMP_DIR}/xdg_cache"
export SPARSENE_MTX_CACHE_DIR="${SPARSEKG_TMP_DIR}/mtx_cache" SPARSENE_DISABLE_MTX_CACHE=1
ACTION="${1:-}"
case "${ACTION}" in status) [[ ! -f "${RUN_DIR}/campaign.pid" ]] || { P=$(<"${RUN_DIR}/campaign.pid"); ps -p "$P" -o pid,etime,state,args || true; }; [[ ! -f "${RUN_DIR}/campaign.exit" ]] || cat "${RUN_DIR}/campaign.exit"; echo "Log: ${RUN_DIR}/campaign.log"; exit 0;; tail) exec tail -f "${RUN_DIR}/campaign.log";; check|run|resume|_worker) ;; *) echo "Usage: bash $0 {check|run|resume|status|tail}" >&2; exit 2;; esac
mountpoint -q /mnt/lustre-client || { echo 'Lustre is not mounted' >&2; exit 1; }
[[ "${ACTION}" == check ]] || { [[ -n "${OPENLUX_API_KEY:-}" ]] || { echo 'Export OPENLUX_API_KEY first' >&2; exit 1; }; }
mkdir -p "${RUN_DIR}" "${TMPDIR}" "${CUDA_CACHE_PATH}" "${XDG_CACHE_HOME}"
MODEL_CONFIG="${RUN_DIR}/model_config.json"
"${PYTHON}" - "${MODEL_CONFIG}" "${MODEL}" <<'PYCONFIG'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); model=sys.argv[2]
defaults={'stream':True,'timeout_sec':900,'max_tokens':32768}
if model=='deepseek-v4-pro': defaults.update(thinking_mode='enabled',provider_sort='success_rate')
else: defaults.update(reasoning_effort='high')
cfg={'models':{model:{'provider':'openai_compatible','model':model,'api_base_url':'https://api.openlux.ai/v1','api_key_env':'OPENLUX_API_KEY','defaults':defaults}}}
text=json.dumps(cfg,indent=2)+'\n'
if p.exists() and p.read_text()!=text: raise SystemExit('Existing campaign model config differs; choose another RUN_DIR')
if not p.exists(): p.write_text(text)
PYCONFIG
ARGS=(--repo "${ROOT}" --run-dir "${RUN_DIR}" --feature-dir "${FEATURE_DIR}" --seed-run-dir "${RUN_DIR}" --incumbent-dir "${RUN_DIR}/isolated-incumbents" --model-config "${MODEL_CONFIG}" --ncu-bin "${NCU_BIN}" --parallel "${WORKERS}" --model "${MODEL}" --platform "${PLATFORM}")
DRIVER="${ROOT}/python/test/run_adaptive_all26.py"
if [[ "${ACTION}" == check ]]; then OPENLUX_API_KEY="${OPENLUX_API_KEY:-offline-placeholder}" exec "${PYTHON}" "${DRIVER}" check "${ARGS[@]}"; fi
if [[ "${ACTION}" == _worker ]]; then
 set +e
 "${PYTHON}" -u "${DRIVER}" run "${ARGS[@]}"
 rc=$?; printf '%s\n' "$rc" > "${RUN_DIR}/campaign.exit"; exit "$rc"
fi
exec 9>"${RUN_DIR}/campaign.lock"; flock -n 9 || { echo 'Campaign is already running' >&2; exit 1; }
if [[ -f "${RUN_DIR}/campaign.pid" ]] && kill -0 "$(<"${RUN_DIR}/campaign.pid")" 2>/dev/null; then echo 'Campaign PID is active' >&2; exit 1; fi
if [[ "${ACTION}" == run && -d "${RUN_DIR}/workers" ]]; then echo 'Existing run: use resume or a new RUN_DIR' >&2; exit 1; fi
if [[ "${ACTION}" == resume && ! -d "${RUN_DIR}/workers" ]]; then echo 'No run to resume' >&2; exit 1; fi
"${PYTHON}" "${DRIVER}" check "${ARGS[@]}"
export RUN_DIR WORKERS FEATURE_DIR NCU_BIN
nohup setsid bash "$(realpath "${BASH_SOURCE[0]}")" _worker >> "${RUN_DIR}/campaign.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "${RUN_DIR}/campaign.pid"
echo "Started PID $!; directory: ${RUN_DIR}; log: ${RUN_DIR}/campaign.log"
