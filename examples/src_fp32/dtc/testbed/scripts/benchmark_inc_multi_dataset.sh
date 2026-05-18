#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
TESTBED_DIR="${ROOT}/examples/src_fp32/dtc/testbed"
PY_DIR="${ROOT}/python"
BUILD_DIR="${TESTBED_DIR}/build_multi_dataset"
RESULT_ROOT="${RESULT_ROOT:-${TESTBED_DIR}/build_perf_logs}"
VENV_PATH="${HOME}/llm-sog-venv"
HOST_CXX="${NVCC_CCBIN:-/opt/rh/gcc-toolset-13/root/usr/bin/g++}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"

MODE="${MODE:-generated}"
DATASET_DIR="${DATASET_DIR:-/mnt/lustre-client/zhangkaige/selected_mtx}"
N_COLS="${N_COLS:-512}"
WARMUP="${WARMUP:-20}"
REPEAT="${REPEAT:-100}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"
MAX_DATASETS="${MAX_DATASETS:-0}"
FORCE_RUN="${FORCE_RUN:-1}"
DATASETS="${DATASETS:-}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PY_DIR}:${PYTHONPATH:-}"
export PATH="$(dirname "${HOST_CXX}"):${PATH}"
export SPARSENE_DISABLE_MTX_CACHE="${SPARSENE_DISABLE_MTX_CACHE:-1}"
export SPARSENE_MTX_CACHE_DIR="${SPARSENE_MTX_CACHE_DIR:-/mnt/lustre-client/huoda/mtx_cache}"

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULT_ROOT}/multi_dataset_${MODE}_${timestamp}"
mkdir -p "${BUILD_DIR}" "${RUN_DIR}"

mode_binary_name() {
  case "$1" in
    generated) echo "demo-dtc-generated-multi-dataset-tf32" ;;
    patched) echo "demo-dtc-patched-multi-dataset-tf32" ;;
    reference) echo "demo-dtc-reference-multi-dataset-tf32" ;;
    *)
      echo "Unsupported MODE=$1, expected generated|patched|reference" >&2
      exit 1
      ;;
  esac
}

build_binary() {
  local mode="$1"
  local binary_path="${BUILD_DIR}/$(mode_binary_name "${mode}")"

  if [[ "${mode}" == "generated" ]]; then
    if [[ "${SKIP_GENERATE}" != "1" ]]; then
      echo "[build] Generating ${PY_DIR}/me_tcf_kernel3.inc" >&2
      (
        cd "${PY_DIR}"
        python test/test_generate.py >/dev/null
      )
    else
      echo "[build] Skipping regeneration and using existing ${PY_DIR}/me_tcf_kernel3.inc" >&2
    fi
  fi

  echo "[build] Building ${mode} benchmark binary" >&2
  case "${mode}" in
    generated|patched)
      nvcc \
        -O3 \
        -std=c++17 \
        --expt-relaxed-constexpr \
        -lineinfo \
        -ftemplate-backtrace-limit=0 \
        -arch="${CUDA_ARCH}" \
        -ccbin "${HOST_CXX}" \
        -Xcompiler -fopenmp \
        -Df32 \
        -I"${TESTBED_DIR}/cutlass/include" \
        -I"${TESTBED_DIR}/cutlass/examples/common" \
        -I"${TESTBED_DIR}/cutlass/tools/util/include" \
        -I"${TESTBED_DIR}/src" \
        -I"${TESTBED_DIR}" \
        $([[ "${mode}" == "patched" ]] && printf '%s' "-DUSE_PATCHED_INC") \
        "${TESTBED_DIR}/host_program_generated.cu" \
        "${TESTBED_DIR}/main.cu" \
        "${TESTBED_DIR}/src/mmio.cu" \
        "${TESTBED_DIR}/src/mmio_highlevel.cu" \
        "${TESTBED_DIR}/src/utils.cu" \
        -lcusparse \
        -lcublas \
        -o "${binary_path}"
      ;;
    reference)
      nvcc \
        -O3 \
        -std=c++17 \
        --expt-relaxed-constexpr \
        -lineinfo \
        -ftemplate-backtrace-limit=0 \
        -arch="${CUDA_ARCH}" \
        -ccbin "${HOST_CXX}" \
        -Xcompiler -fopenmp \
        -Df32 \
        -I"${TESTBED_DIR}/cutlass/include" \
        -I"${TESTBED_DIR}/cutlass/examples/common" \
        -I"${TESTBED_DIR}/cutlass/tools/util/include" \
        -I"${TESTBED_DIR}/src" \
        -I"${TESTBED_DIR}" \
        "${TESTBED_DIR}/host_program.cu" \
        "${TESTBED_DIR}/main.cu" \
        "${TESTBED_DIR}/src/mmio.cu" \
        "${TESTBED_DIR}/src/mmio_highlevel.cu" \
        "${TESTBED_DIR}/src/utils.cu" \
        -lcusparse \
        -lcublas \
        -o "${binary_path}"
      ;;
  esac

  echo "${binary_path}"
}

collect_datasets() {
  if [[ -n "${DATASETS}" ]]; then
    python - "${DATASET_DIR}" "${DATASETS}" <<'PY'
import os
import sys
dataset_dir = sys.argv[1]
datasets = [x.strip() for x in sys.argv[2].split(",") if x.strip()]
for item in datasets:
    path = item if os.path.isabs(item) else os.path.join(dataset_dir, item)
    print(path)
PY
    return
  fi

  python - "${DATASET_DIR}" "${MAX_DATASETS}" <<'PY'
import glob
import os
import sys
dataset_dir = sys.argv[1]
max_datasets = int(sys.argv[2])
files = sorted(glob.glob(os.path.join(dataset_dir, "*.mtx")))
if max_datasets > 0:
    files = files[:max_datasets]
for path in files:
    print(path)
PY
}

run_dataset() {
  local binary_path="$1"
  local dataset_path="$2"
  local dataset_name
  dataset_name="$(basename "${dataset_path}")"
  local log_path="${RUN_DIR}/${dataset_name%.mtx}.log"
  local rc=0

  echo "[run] ${dataset_name}" >&2
  timeout "${TIMEOUT_SEC}" "${binary_path}" \
    -filename "${dataset_path}" \
    -mtx_flag 1 \
    -N "${N_COLS}" \
    -warmup "${WARMUP}" \
    -repeat "${REPEAT}" \
    >"${log_path}" 2>&1 || {
    rc=$?
  }

  python - "${MODE}" "${dataset_path}" "${log_path}" "${rc}" <<'PY'
import os
import re
import sys

mode, dataset_path, log_path, rc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
text = open(log_path, "r", encoding="utf-8", errors="replace").read()
dataset = os.path.basename(dataset_path)

dim_match = re.search(r"FLOAT: Sparse A\[(\d+),(\d+)\] x B\[(\d+),(\d+)\] = C\[(\d+),(\d+)\]", text)
my_match = re.search(r"mykernel_time:\s*([0-9.]+)\s*ms", text)
cu_match = re.search(r"cusparse_time:\s*([0-9.]+)\s*ms", text)
error_count = len(re.findall(r"Error\(", text))
cuda_304 = "error: OS call failed or operation not supported on this OS (304)" in text
glibc_assert = "assertion failed" in text or "Fatal glibc error" in text

status = "ok"
reason = ""
if rc == 124:
    status = "timeout"
    reason = "timeout"
elif rc != 0:
    status = "failed"
    reason = f"exit_{rc}"
elif cuda_304:
    status = "failed"
    reason = "cuda_304"
elif glibc_assert:
    status = "failed"
    reason = "glibc_assert"
elif my_match is None or cu_match is None:
    status = "failed"
    reason = "missing_timing"
elif error_count > 0:
    status = "failed"
    reason = f"correctness_errors_{error_count}"

M = K = N = ""
if dim_match:
    M, K, _, N, _, _ = dim_match.groups()

mykernel = my_match.group(1) if my_match else ""
cusparse = cu_match.group(1) if cu_match else ""
speedup = ""
if my_match and cu_match:
    my_val = float(my_match.group(1))
    cu_val = float(cu_match.group(1))
    if my_val != 0.0:
        speedup = f"{cu_val / my_val:.4f}"

print("\t".join([
    dataset,
    dataset_path,
    mode,
    M,
    K,
    N,
    mykernel,
    cusparse,
    speedup,
    status,
    reason,
    str(error_count),
    log_path,
]))
PY
}

write_reports() {
  local tsv_path="$1"
  local csv_path="${RUN_DIR}/summary.csv"
  local md_path="${RUN_DIR}/summary.md"

  python - "${tsv_path}" "${csv_path}" "${md_path}" <<'PY'
import csv
import sys

tsv_path, csv_path, md_path = sys.argv[1], sys.argv[2], sys.argv[3]
headers = [
    "dataset", "dataset_path", "mode", "M", "K", "N",
    "mykernel_ms", "cusparse_ms", "speedup_vs_cusparse",
    "status", "reason", "error_count", "log_path",
]

rows = []
with open(tsv_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != len(headers):
            continue
        rows.append(parts)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

table_headers = ["dataset", "M", "K", "N", "mykernel_ms", "cusparse_ms", "speedup_vs_cusparse", "status", "reason"]
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Multi-dataset INC Benchmark\n\n")
    f.write("| " + " | ".join(table_headers) + " |\n")
    f.write("|" + "|".join(["---"] * len(table_headers)) + "|\n")
    for row in rows:
        reduced = [row[0], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]]
        f.write("| " + " | ".join(x if x else "-" for x in reduced) + " |\n")

print(csv_path)
print(md_path)
PY
}

main() {
  local binary_path
  binary_path="$(build_binary "${MODE}")"

  local tsv_path="${RUN_DIR}/summary.tsv"
  : > "${tsv_path}"

  mapfile -t dataset_paths < <(collect_datasets)
  if [[ "${#dataset_paths[@]}" -eq 0 ]]; then
    echo "No datasets found under ${DATASET_DIR}" >&2
    exit 1
  fi

  echo "[info] mode=${MODE} datasets=${#dataset_paths[@]} N=${N_COLS} warmup=${WARMUP} repeat=${REPEAT}"
  for dataset_path in "${dataset_paths[@]}"; do
    run_dataset "${binary_path}" "${dataset_path}" >> "${tsv_path}"
  done

  mapfile -t report_paths < <(write_reports "${tsv_path}")
  echo "[done] csv=${report_paths[0]}"
  echo "[done] markdown=${report_paths[1]}"
}

main "$@"
