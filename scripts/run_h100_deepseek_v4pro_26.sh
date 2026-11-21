#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="$(realpath -m "${ROOT}/logs/e2e_runs")"
RUN_DIR="$(realpath -m "${RUN_DIR:-${RUNS_ROOT}/h100_deepseek_v4pro_historical_format3_strategy6}")"
case "${RUN_DIR}/" in "${RUNS_ROOT}/"*) ;; *) echo "RUN_DIR must be inside ${RUNS_ROOT}" >&2; exit 2 ;; esac
[[ "${RUN_DIR}" != "${RUNS_ROOT}" ]] || { echo 'RUN_DIR must identify one campaign' >&2; exit 2; }
FEATURE_DIR="${FEATURE_DIR:-/mnt/lustre-client/huoda/selected_mtx}"
CUTLASS_ROOT="${CUTLASS_ROOT:-/home/huoda/sparsekg-h100-baseline/cutlass-afa1772203677c5118fcd82537a9c8fefbcc7008}"
SITE="${SITE:-/mnt/lustre-client/huoda/venv/llm-sog/lib/python3.12/site-packages}"
WORKERS="${WORKERS:-8}"
export PATH="/usr/local/cuda/bin:${PATH}" CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++}"
export PYTHONPATH="${ROOT}/python:${ROOT}/python/test:${SITE}${PYTHONPATH:+:${PYTHONPATH}}"
export SPARSEKG_CUDA_ARCH=90 SPARSEKG_CUTLASS_ROOT="${CUTLASS_ROOT}"
export SPARSEKG_NO_AMBIGUOUS_RETRY=1
# This shared lock coordinates with other project GPU jobs; it is not a result artifact.
export SPARSEKG_GPU_LOCK_FILE="${SPARSEKG_GPU_LOCK_FILE:-/tmp/sparsekg_sm80_gpu_benchmark.lock}"
# Separate H100 campaigns from A800 and from each other on the shared Lustre.
# Hash the absolute run directory so the same campaign has stable rebuild paths.
LUSTRE_MOUNT=/mnt/lustre-client
LUSTRE_TMP_ROOT="${LUSTRE_MOUNT}/huoda/sparsekg_tmp"
RUN_STORAGE_ID="$(printf '%s' "${RUN_DIR}" | sha256sum | cut -c1-24)"
export SPARSEKG_TMP_DIR="${LUSTRE_TMP_ROOT}/h100_campaigns/run-${RUN_STORAGE_ID}"
export TMPDIR="${SPARSEKG_TMP_DIR}/toolchain_tmp" TMP="${SPARSEKG_TMP_DIR}/toolchain_tmp" TEMP="${SPARSEKG_TMP_DIR}/toolchain_tmp"
export CUDA_CACHE_PATH="${SPARSEKG_TMP_DIR}/cuda_cache" XDG_CACHE_HOME="${SPARSEKG_TMP_DIR}/xdg_cache"
export SPARSENE_MTX_CACHE_DIR="${SPARSEKG_TMP_DIR}/mtx_cache" SPARSENE_DISABLE_MTX_CACHE=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || { echo 'WORKERS must be positive' >&2; exit 2; }
[[ -f "${CUTLASS_ROOT}/include/cute/tensor.hpp" && -d "${SITE}" ]] || { echo 'Missing CUTLASS or Python packages' >&2; exit 1; }
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
export RUN_DIR
active_driver() {
  /usr/bin/python3 - "${RUN_DIR}" <<'PYPROC'
import os, sys
from pathlib import Path
run = sys.argv[1]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        if proc.stat().st_uid != os.getuid(): continue
        args = proc.joinpath('cmdline').read_bytes().split(b'\0')
        args = [os.fsdecode(x) for x in args if x]
        if not any(x.endswith('/python/test/run_h100_ds_v4pro_all26.py') for x in args): continue
        if '--run-dir' not in args or args[args.index('--run-dir')+1] != run: continue
        stat = proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if stat[0] == 'Z': continue
        print(f"PID {proc.name} state={stat[0]} process_group={stat[2]}")
    except (OSError, ValueError, IndexError): continue
PYPROC
}
action="${1:-}"
case "$action" in
 status)
   active="$(active_driver)"
   if [[ -n "$active" ]]; then printf 'Active campaign: %s\n%s\n' "${RUN_DIR}" "$active"; else echo "Not running: ${RUN_DIR}"; fi
   [[ ! -f "${RUN_DIR}/campaign.exit" ]] || echo "Last recorded exit code: $(< "${RUN_DIR}/campaign.exit")"
   echo "Log: ${RUN_DIR}/campaign.log"; exit 0 ;;
 tail)
   file="${RUN_DIR}/campaign.log"
   if [[ -n "${2:-}" ]]; then
     [[ "${2}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid dataset' >&2; exit 2; }
     file="${RUN_DIR}/workers/${2}/driver.log"
   fi
   [[ -f "$file" ]] || { echo "No log: $file" >&2; exit 1; }
   exec tail -f "$file" ;;
 check) export OPENLUX_API_KEY="${OPENLUX_API_KEY:-offline-placeholder}" ;;
 run|resume|_worker)
   [[ -n "${OPENLUX_API_KEY:-}" && "${OPENLUX_API_KEY}" != offline-placeholder ]] || { echo 'Export OPENLUX_API_KEY first' >&2; exit 1; } ;;
 summarize) ;;
 *) echo "Usage: bash $0 {check|run|resume|status|tail [dataset]|summarize}"; exit 2 ;;
esac
if [[ "$action" == check || "$action" == run || "$action" == resume || "$action" == _worker || "$action" == summarize ]]; then
  mountpoint -q "${LUSTRE_MOUNT}" || { echo "Lustre not mounted at ${LUSTRE_MOUNT}; refusing local fallback" >&2; exit 1; }
  [[ -d "${LUSTRE_TMP_ROOT}" && -w "${LUSTRE_TMP_ROOT}" ]] || { echo "Lustre scratch root is missing or not writable: ${LUSTRE_TMP_ROOT}" >&2; exit 1; }
fi
if [[ "$action" == _worker ]]; then
  [[ "${2:-}" == "${RUN_DIR}" ]] || { echo 'Worker directory mismatch' >&2; exit 2; }
  set +e
  /usr/bin/python3 -u "${ROOT}/python/test/run_h100_ds_v4pro_all26.py" run --repo "${ROOT}" --run-dir "${RUN_DIR}" --feature-dir "${FEATURE_DIR}" --seed-run-dir "${RUN_DIR}" --incumbent-dir "${RUN_DIR}/isolated-incumbents" --model-config "${ROOT}/llm_models.yaml" --ncu-bin /usr/local/cuda/bin/ncu --parallel "${WORKERS}"
  rc=$?
  printf '%s\n' "$rc" > "${RUN_DIR}/campaign.exit"
  echo "[$(date -Is)] Campaign exited: $rc"
  exit "$rc"
fi
if [[ "$action" == run || "$action" == resume ]]; then
  mkdir -p "${RUN_DIR}"
  exec 9>"${RUN_DIR}/campaign.lock"
  flock -n 9 || { echo 'A campaign for this directory is already running' >&2; exit 1; }
  active="$(active_driver)"
  [[ -z "$active" ]] || { echo "Campaign already active or stopped: $active" >&2; exit 1; }
  if [[ "$action" == run && -d "${RUN_DIR}/workers" ]]; then
    echo "Existing campaign: ${RUN_DIR}; inspect it and use resume or choose another RUN_DIR" >&2; exit 1
  fi
  if [[ "$action" == resume && ! -d "${RUN_DIR}/workers" ]]; then
    echo "No campaign to resume: ${RUN_DIR}" >&2; exit 1
  fi
  if [[ "$action" == resume ]]; then
    echo 'WARNING: inspect prior request audits and billing first; resume may resend failed requests.' >&2
  fi
  mkdir -p "${RUN_DIR}/isolated-incumbents" "${TMPDIR}" "${CUDA_CACHE_PATH}" "${XDG_CACHE_HOME}"
  if [[ -e "${RUN_DIR}/scratch_path.txt" && "$(< "${RUN_DIR}/scratch_path.txt")" != "${SPARSEKG_TMP_DIR}" ]]; then
    echo "Existing scratch_path.txt does not match this campaign's Lustre path" >&2; exit 1
  fi
  printf '%s\n' "${SPARSEKG_TMP_DIR}" > "${RUN_DIR}/scratch_path.txt"
  echo "Lustre build and cache directory: ${SPARSEKG_TMP_DIR}"
  /usr/bin/python3 -u "${ROOT}/python/test/run_h100_ds_v4pro_all26.py" check --repo "${ROOT}" --run-dir "${RUN_DIR}" --feature-dir "${FEATURE_DIR}" --seed-run-dir "${RUN_DIR}" --incumbent-dir "${RUN_DIR}/isolated-incumbents" --model-config "${ROOT}/llm_models.yaml" --ncu-bin /usr/local/cuda/bin/ncu --parallel "${WORKERS}"
  nohup setsid bash "${SCRIPT_PATH}" _worker "${RUN_DIR}" >> "${RUN_DIR}/campaign.log" 2>&1 < /dev/null &
  worker_pid=$!
  echo "$worker_pid" > "${RUN_DIR}/campaign.pid"
  echo "Started background campaign: PID ${worker_pid}"
  echo "Directory: ${RUN_DIR}"
  echo "Status: bash ${SCRIPT_PATH} status"
  echo "Log: bash ${SCRIPT_PATH} tail"
  exit 0
fi
mkdir -p "${RUN_DIR}/isolated-incumbents" "${TMPDIR}" "${CUDA_CACHE_PATH}" "${XDG_CACHE_HOME}"
echo "Campaign directory: ${RUN_DIR}"
if [[ "$action" == check ]]; then echo "Lustre build and cache directory: ${SPARSEKG_TMP_DIR}"; fi
exec /usr/bin/python3 -u "${ROOT}/python/test/run_h100_ds_v4pro_all26.py" "$action" --repo "${ROOT}" --run-dir "${RUN_DIR}" --feature-dir "${FEATURE_DIR}" --seed-run-dir "${RUN_DIR}" --incumbent-dir "${RUN_DIR}/isolated-incumbents" --model-config "${ROOT}/llm_models.yaml" --ncu-bin /usr/local/cuda/bin/ncu --parallel "${WORKERS}"
