#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
TESTBED_DIR="${ROOT}/examples/src_fp32/dtc/testbed"
SCRIPT_DIR="${TESTBED_DIR}/scripts"
LOG_DIR="${TESTBED_DIR}/build_perf_logs"

MODE="${MODE:-generated}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"
FORCE_RUN="${FORCE_RUN:-1}"

mkdir -p "${LOG_DIR}"

run_one() {
  local mode="$1"
  local log_file="$2"
  shift 2

  local cmd=()
  case "${mode}" in
    generated)
      cmd=(env FORCE_RUN="${FORCE_RUN}" PATCHED_ONLY=0 CUDA_ARCH="${CUDA_ARCH}" \
        "${SCRIPT_DIR}/validate_generated_kernel.sh")
      ;;
    patched)
      cmd=(env FORCE_RUN="${FORCE_RUN}" PATCHED_ONLY=1 CUDA_ARCH="${CUDA_ARCH}" \
        "${SCRIPT_DIR}/validate_generated_kernel.sh")
      ;;
    reference)
      cmd=(env FORCE_RUN="${FORCE_RUN}" CUDA_ARCH="${CUDA_ARCH}" \
        "${SCRIPT_DIR}/validate_reference_kernel.sh")
      ;;
    *)
      echo "Unsupported MODE=${mode}, expected generated|patched|reference|all" >&2
      exit 1
      ;;
  esac

  echo "== Running ${mode} =="
  "${cmd[@]}" "$@" | tee "${log_file}"
}

summarize_one() {
  local mode="$1"
  local log_file="$2"

  local mykernel_time
  local cusparse_time
  mykernel_time="$(grep -Eo 'mykernel_time: [0-9.]+ ms' "${log_file}" | tail -n 1 | awk '{print $2}')"
  cusparse_time="$(grep -Eo 'cusparse_time: [0-9.]+ ms' "${log_file}" | tail -n 1 | awk '{print $2}')"

  if [[ -z "${mykernel_time}" || -z "${cusparse_time}" ]]; then
    echo "Failed to parse performance numbers from ${log_file}" >&2
    return 1
  fi

  python - "${mode}" "${mykernel_time}" "${cusparse_time}" <<'PY'
import sys
mode, mykernel, cusparse = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
speedup = cusparse / mykernel if mykernel else float("inf")
print(f"{mode}: mykernel_time={mykernel:.6f} ms, cusparse_time={cusparse:.6f} ms, speedup_vs_cusparse={speedup:.4f}x")
PY
}

main() {
  local args=("$@")
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"

  case "${MODE}" in
    generated|patched|reference)
      local log_file="${LOG_DIR}/${MODE}_${timestamp}.log"
      run_one "${MODE}" "${log_file}" "${args[@]}"
      echo
      summarize_one "${MODE}" "${log_file}"
      echo "log_file=${log_file}"
      ;;
    all)
      local generated_log="${LOG_DIR}/generated_${timestamp}.log"
      local reference_log="${LOG_DIR}/reference_${timestamp}.log"
      run_one generated "${generated_log}" "${args[@]}"
      echo
      run_one reference "${reference_log}" "${args[@]}"
      echo
      summarize_one generated "${generated_log}"
      summarize_one reference "${reference_log}"
      echo "generated_log=${generated_log}"
      echo "reference_log=${reference_log}"
      ;;
    *)
      echo "Unsupported MODE=${MODE}, expected generated|patched|reference|all" >&2
      exit 1
      ;;
  esac
}

main "$@"
