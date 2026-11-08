#!/usr/bin/env bash
# DeepSeek V4 Pro all-26 joint rerun: per-dataset historical Sparsene format0, two Agent formats,
# and heuristic + five Agent strategies for each executable format.
set -euo pipefail

ROOT=/home/huoda/sparsene
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
RUN_DIR="${RUN_DIR:-${ROOT}/logs/e2e_runs/20261108_ds_v4pro_joint_all26_f3_s5_sparsene_reference_v1}"
FEATURE_DIR="${FEATURE_DIR:-/mnt/lustre-client/huoda/selected_mtx}"
SEED_RUN_DIR="${SEED_RUN_DIR:-${ROOT}/logs/e2e_runs/20260722_232100_mask_deepseek_v4_pro_format26_v7_empirical_feedback}"
INCUMBENT_DIR="${INCUMBENT_DIR:-${ROOT}/logs/e2e_runs/incumbents/a100_a800_n512_fp32}"
VENV="${VENV:-/mnt/lustre-client/huoda/venv/llm-sog}"
NCU_BIN="${NCU_BIN:-/home/huoda/local/cuda/nsight-compute-2025.1.0/ncu}"
MAX_PARALLEL_DATASETS="${MAX_PARALLEL_DATASETS:-8}"
PID_FILE="${RUN_DIR}/campaign.pid"
LOG_FILE="${RUN_DIR}/campaign.log"
export SPARSEKG_TMP_DIR="${SPARSEKG_TMP_DIR:-/mnt/lustre-client/huoda/sparsekg_tmp}"
export SPARSEKG_GPU_LOCK_FILE="${SPARSEKG_GPU_LOCK_FILE:-/tmp/sparsekg_sm80_gpu_benchmark.lock}"

usage() {
  echo "Usage: bash $0 {check|probe|start|resume|status|tail [dataset]|stop|summarize}"
  echo "Run: ${RUN_DIR}; CPU dataset workers=${MAX_PARALLEL_DATASETS}; GPU work is lock-serialized."
}

owned_process() {
  local pid="$1" pgid state command_line
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
  state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ')"
  [[ "${state}" != Z* && "${pgid}" == "${pid}" && -r "/proc/${pid}/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  [[ "${command_line}" == *"${SCRIPT_PATH}"* && "${command_line}" == *"_worker"* && "${command_line}" == *"${RUN_DIR}"* ]]
}

setup() {
  [[ "${MAX_PARALLEL_DATASETS}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_PARALLEL_DATASETS must be positive" >&2; return 2; }
  [[ "${SPARSEKG_TMP_DIR}" == /* ]] || { echo "SPARSEKG_TMP_DIR must be absolute" >&2; return 2; }
  if [[ "${SPARSEKG_TMP_DIR}" == /mnt/lustre-client/huoda/sparsekg_tmp ]]; then
    mountpoint -q /mnt/lustre-client || { echo "Lustre is not mounted; refusing local-disk fallback" >&2; return 1; }
  fi
  mkdir -p "${SPARSEKG_TMP_DIR}/toolchain_tmp"
  [[ -w "${SPARSEKG_TMP_DIR}/toolchain_tmp" ]] || { echo "Temporary storage is not writable" >&2; return 1; }
  export TMPDIR="${SPARSEKG_TMP_DIR}/toolchain_tmp"
  export TMP="${TMPDIR}" TEMP="${TMPDIR}"
  [[ -f "${VENV}/bin/activate" ]] || { echo "Missing venv: ${VENV}" >&2; return 1; }
  [[ -n "${OPENLUX_API_KEY:-}" ]] || { echo "OPENLUX_API_KEY is not set" >&2; return 1; }
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  export PYTHONPATH="${ROOT}/python:${ROOT}/python/test${PYTHONPATH:+:${PYTHONPATH}}"
}

driver() {
  python -u "${ROOT}/python/test/run_ds_v4pro_joint_all26_campaign.py" "$1" \
    --repo "${ROOT}" --run-dir "${RUN_DIR}" --feature-dir "${FEATURE_DIR}" \
    --seed-run-dir "${SEED_RUN_DIR}" --incumbent-dir "${INCUMBENT_DIR}" \
    --model-config "${ROOT}/llm_models.yaml" --ncu-bin "${NCU_BIN}" \
    --parallel "${MAX_PARALLEL_DATASETS}"
}

action="${1:-}"
case "${action}" in
  check) setup; driver check ;;
  probe) setup; driver check; driver probe ;;
  _worker)
    [[ "${2:-}" == "${RUN_DIR}" ]] || { echo "Worker directory mismatch" >&2; exit 2; }
    setup
    echo "[$(date -Is)] START workers=${MAX_PARALLEL_DATASETS} GPU_LOCK=${SPARSEKG_GPU_LOCK_FILE}"
    driver run
    echo "[$(date -Is)] CAMPAIGN_FINISHED"
    ;;
  start|resume)
    setup
    if [[ "${action}" == start && -e "${LOG_FILE}" ]]; then
      echo "Run already exists; use resume or set a new RUN_DIR" >&2; exit 1
    fi
    if [[ "${action}" == resume && ! -d "${RUN_DIR}" ]]; then
      echo "Run directory missing: ${RUN_DIR}" >&2; exit 1
    fi
    driver check
    mkdir -p "${RUN_DIR}"
    if [[ -f "${PID_FILE}" ]] && owned_process "$(< "${PID_FILE}")"; then
      echo "Already running: PID $(< "${PID_FILE}")" >&2; exit 1
    fi
    driver probe
    setsid bash "${SCRIPT_PATH}" _worker "${RUN_DIR}" >> "${LOG_FILE}" 2>&1 &
    worker_pid=$!
    echo "${worker_pid}" > "${PID_FILE}"
    echo "Started PID/PGID ${worker_pid}; log: ${LOG_FILE}"
    echo "Follow: bash ${SCRIPT_PATH} tail [dataset]; stop: bash ${SCRIPT_PATH} stop"
    ;;
  status)
    if [[ -f "${PID_FILE}" ]] && owned_process "$(< "${PID_FILE}")"; then
      echo "Running: PID $(< "${PID_FILE}")"
    else
      echo "Not running; log: ${LOG_FILE}"
    fi
    if [[ -f "${RUN_DIR}/campaign_summary.csv" ]]; then
      echo "Current summary: ${RUN_DIR}/campaign_summary.csv"
    fi
    ;;
  tail)
    tail_file="${LOG_FILE}"
    if [[ -n "${2:-}" ]]; then
      [[ "${2}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid dataset name" >&2; exit 2; }
      tail_file="${RUN_DIR}/workers/${2}/driver.log"
    fi
    [[ -f "${tail_file}" ]] || { echo "No log yet: ${tail_file}" >&2; exit 1; }
    tail -f "${tail_file}"
    ;;
  stop)
    [[ -f "${PID_FILE}" ]] || { echo "No PID file" >&2; exit 1; }
    worker_pid="$(< "${PID_FILE}")"
    owned_process "${worker_pid}" || { echo "PID is not this campaign; refusing to signal" >&2; exit 1; }
    kill -TERM -- "-${worker_pid}"
    echo "Sent TERM to campaign process group ${worker_pid}"
    ;;
  summarize)
    [[ -d "${RUN_DIR}" ]] || { echo "Run directory missing: ${RUN_DIR}" >&2; exit 1; }
    source "${VENV}/bin/activate"
    export PYTHONPATH="${ROOT}/python:${ROOT}/python/test${PYTHONPATH:+:${PYTHONPATH}}"
    driver summarize
    ;;
  *) usage; exit 2 ;;
esac
