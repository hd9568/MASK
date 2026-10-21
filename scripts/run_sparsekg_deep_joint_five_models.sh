#!/usr/bin/env bash
# Six matrices x five models x five formats x (heuristic + ten strategies).
# Model campaigns overlap; correctness/timing/NCU remain serialized on one GPU.
set -euo pipefail

ROOT=/home/huoda/sparsene
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${ROOT}/logs/e2e_runs/20261020_sparsekg_deep_joint6_f5_s10_five_models}"
FEATURE_DIR="${FEATURE_DIR:-/mnt/lustre-client/huoda/selected_mtx}"
SEED_RUN_DIR="${SEED_RUN_DIR:-${ROOT}/logs/e2e_runs/20260722_232100_mask_deepseek_v4_pro_format26_v7_empirical_feedback}"
INCUMBENT_DIR="${INCUMBENT_DIR:-${ROOT}/logs/e2e_runs/incumbents/a100_a800_n512_fp32}"
VENV=/mnt/lustre-client/huoda/venv/llm-sog
PID_FILE="${CAMPAIGN_ROOT}/campaign.pid"
LOG_FILE="${CAMPAIGN_ROOT}/campaign.log"
FORMAT_ITERATIONS=5
STRATEGY_AGENT_ITERATIONS=10
MAX_PARALLEL_MODELS="${MAX_PARALLEL_MODELS:-3}"
export SPARSEKG_TMP_DIR="${SPARSEKG_TMP_DIR:-/mnt/lustre-client/huoda/sparsekg_tmp}"
export SPARSEKG_GPU_LOCK_FILE="${SPARSEKG_GPU_LOCK_FILE:-/tmp/sparsekg_sm80_gpu_benchmark.lock}"
DATASETS=(ddi cop20k_A mip1 reddit consph cant)
MODELS=(deepseek-v4-pro gpt-6.1-sol claude-opus-5-5 deepseek-flash gpt-6-astra)

usage() {
  echo "Usage: bash $0 {check|start|status|tail|stop} [all|model-name]"
  echo "Models: ${MODELS[*]}"
  echo "start all runs up to MAX_PARALLEL_MODELS=3 models concurrently; GPU calls are cross-process serialized."
  echo "Per dataset/model: 5 accepted format rounds x (heuristic + 10 strategies), N=512 FP32, warmup=3, repeat=32."
}

valid_scope() {
  local scope="$1" item
  [[ "${scope}" == all ]] && return 0
  for item in "${MODELS[@]}"; do [[ "${item}" == "${scope}" ]] && return 0; done
  return 1
}

owned_process() {
  local pid="$1" pgid process_state command_line
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
  process_state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ')"
  [[ "${process_state}" != Z* ]] || return 1
  [[ "${pgid}" == "${pid}" && -r "/proc/${pid}/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  [[ "${command_line}" == *"${SCRIPT_PATH}"* && "${command_line}" == *"_worker"* && "${command_line}" == *"${CAMPAIGN_ROOT}"* ]]
}

check() {
  local scope="$1" model key_env expected_endpoint
  valid_scope "${scope}" || { echo "Unknown model: ${scope}" >&2; return 2; }
  [[ "${MAX_PARALLEL_MODELS}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_PARALLEL_MODELS must be positive" >&2; return 2; }
  [[ "${SPARSEKG_TMP_DIR}" == /* ]] || { echo "SPARSEKG_TMP_DIR must be absolute" >&2; return 2; }
  if [[ "${SPARSEKG_TMP_DIR}" == /mnt/lustre-client/huoda/sparsekg_tmp ]]; then
    mountpoint -q /mnt/lustre-client || { echo "Lustre is not mounted; refusing system-disk fallback" >&2; return 1; }
  fi
  mkdir -p "${SPARSEKG_TMP_DIR}/toolchain_tmp"
  [[ -w "${SPARSEKG_TMP_DIR}/toolchain_tmp" ]] || { echo "Temporary storage is not writable: ${SPARSEKG_TMP_DIR}" >&2; return 1; }
  export TMPDIR="${SPARSEKG_TMP_DIR}/toolchain_tmp"
  export TMP="${TMPDIR}" TEMP="${TMPDIR}"
  [[ -f "${VENV}/bin/activate" ]] || { echo "Missing venv: ${VENV}" >&2; return 1; }
  [[ -d "${SEED_RUN_DIR}" ]] || { echo "Missing seed run: ${SEED_RUN_DIR}" >&2; return 1; }
  [[ -f "${INCUMBENT_DIR}/summary.csv" ]] || { echo "Missing incumbent registry" >&2; return 1; }
  [[ -f "${ROOT}/llm_models.yaml" ]] || { echo "Missing model config" >&2; return 1; }
  [[ -f /home/huoda/local/cuda/nsight-compute-2025.1.0/ncu ]] || { echo "Missing NCU" >&2; return 1; }
  for model in "${DATASETS[@]}"; do
    [[ -f "${FEATURE_DIR}/${model}.features.v3.json" ]] || { echo "Missing features: ${model}" >&2; return 1; }
  done
  local selected=("${MODELS[@]}")
  [[ "${scope}" == all ]] || selected=("${scope}")
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
  for model in "${selected[@]}"; do
    case "${model}" in
      deepseek-*) key_env=DEEPSEEK_API_KEY; expected_endpoint=https://api.deepseek.com/v1 ;;
      *) key_env=OPENLUX_API_KEY; expected_endpoint=https://api.openlux.ai/v1 ;;
    esac
    [[ -n "${!key_env:-}" ]] || { echo "${key_env} is not set (${model})" >&2; return 1; }
    python - "${model}" "${expected_endpoint}" "${ROOT}/llm_models.yaml" <<'PY'
import sys
from sparsene.agent.config import resolve_model_config
name, expected, config = sys.argv[1:]
model = resolve_model_config(model_name=name, yaml_path=config)
if model.provider != "openai_compatible" or model.api_base_url != expected or not model.api_key:
    raise SystemExit(f"Invalid local endpoint/key configuration for {name}")
print(f"Local model preflight OK: {name} via {expected}; key present=True")
PY
  done
  echo "Plan: ${#selected[@]} model(s), up to ${MAX_PARALLEL_MODELS} concurrent; ${#DATASETS[@]} matrices/model, $((FORMAT_ITERATIONS * (STRATEGY_AGENT_ITERATIONS + 1))) candidates/matrix."
  echo "GPU benchmark lock: ${SPARSEKG_GPU_LOCK_FILE}"
  echo "Rebuildable binaries and compiler scratch: ${SPARSEKG_TMP_DIR}"
  echo "Campaign root: ${CAMPAIGN_ROOT}"
}

run_one_model() {
  local model="$1" slug format_tokens reasoning_args=() run_dir model_log
  slug="${model//-/_}"
  run_dir="${CAMPAIGN_ROOT}/${slug}"
  mkdir -p "${run_dir}"
  model_log="${run_dir}/model_driver.log"
  case "${model}" in
    gpt-6.1-sol) format_tokens=8192 ;;
    gpt-6-astra)
      format_tokens=32768
      reasoning_args=(--format-reasoning-effort low)
      ;;
    *) format_tokens=32768 ;;
  esac
  echo "[$(date -Is)] START model=${model} run=${run_dir} log=${model_log}"
  if python -u "${ROOT}/python/test/run_joint_format_strategy_feedback.py" \
    --run-dir "${run_dir}" --feature-dir "${FEATURE_DIR}" \
    --datasets "${DATASETS[@]}" \
    --format-iterations "${FORMAT_ITERATIONS}" \
    --strategy-agent-iterations "${STRATEGY_AGENT_ITERATIONS}" \
    --initial-feedback-run-dir "${SEED_RUN_DIR}" \
    --incumbent-dir "${INCUMBENT_DIR}" \
    --format-model "${model}" --format-max-tokens "${format_tokens}" \
    --format-timeout-sec 900 --format-max-attempts 3 --format-retry-delay-sec 5 \
    "${reasoning_args[@]}" \
    --strategy-provider openai_http --strategy-model "${model}" \
    --strategy-max-tokens 16384 --strategy-timeout-sec 600 \
    --model-config "${ROOT}/llm_models.yaml" \
    --n 512 --warmup 3 --repeat 32 --measurement-runs 3 \
    --abort-on-format-failure --resume 2>&1 \
    | tee -a "${model_log}" \
    | sed -u "s/^/[${model}] /"; then
    echo "[$(date -Is)] DONE model=${model} run=${run_dir}"
  else
    echo "[$(date -Is)] FAILED model=${model} run=${run_dir}; inspect ${model_log}" >&2
    return 1
  fi
}

worker() {
  local scope="$1" model active=0 failures=0
  check "${scope}"
  cd "${ROOT}"
  local selected=("${MODELS[@]}")
  [[ "${scope}" == all ]] || selected=("${scope}")
  for model in "${selected[@]}"; do
    run_one_model "${model}" &
    ((active += 1))
    if (( active >= MAX_PARALLEL_MODELS )); then
      if ! wait -n; then ((failures += 1)); fi
      active=$((active - 1))
    fi
  done
  while (( active > 0 )); do
    if ! wait -n; then ((failures += 1)); fi
    active=$((active - 1))
  done
  echo "[$(date -Is)] CAMPAIGN_FINISHED model_failures=${failures}"
  (( failures == 0 ))
}

action="${1:-}"
scope="${2:-all}"
case "${action}" in
  check) check "${scope}" ;;
  _worker)
    [[ "${3:-}" == "${CAMPAIGN_ROOT}" ]] || { echo "Worker directory mismatch" >&2; exit 2; }
    worker "${scope}" ;;
  start)
    check "${scope}"
    mkdir -p "${CAMPAIGN_ROOT}"
    if [[ -f "${PID_FILE}" ]] && owned_process "$(< "${PID_FILE}")"; then
      echo "Already running: PID $(< "${PID_FILE}")"; exit 1
    fi
    setsid bash "${SCRIPT_PATH}" _worker "${scope}" "${CAMPAIGN_ROOT}" >> "${LOG_FILE}" 2>&1 &
    worker_pid=$!
    echo "${worker_pid}" > "${PID_FILE}"
    echo "Started PID/PGID ${worker_pid}; live log: ${LOG_FILE}"
    ;;
  status)
    if [[ -f "${PID_FILE}" ]] && owned_process "$(< "${PID_FILE}")"; then
      echo "Running: PID $(< "${PID_FILE}")"; exit 0
    fi
    echo "Not running; log: ${LOG_FILE}"
    ;;
  tail)
    [[ -f "${LOG_FILE}" ]] || { echo "No log yet: ${LOG_FILE}" >&2; exit 1; }
    tail -f "${LOG_FILE}" ;;
  stop)
    [[ -f "${PID_FILE}" ]] || { echo "No PID file" >&2; exit 1; }
    worker_pid="$(< "${PID_FILE}")"
    owned_process "${worker_pid}" || { echo "PID is not this campaign; refusing to signal" >&2; exit 1; }
    kill -TERM -- "-${worker_pid}"
    echo "Sent TERM to campaign process group ${worker_pid}"
    ;;
  *) usage; exit 2 ;;
esac
