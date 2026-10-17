#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
TESTBED_DIR="${ROOT}/examples/src_fp32/dtc/testbed"
BUILD_DIR="${TESTBED_DIR}/build_reference"
VENV_PATH="/mnt/lustre-client/huoda/venv/llm-sog"
HOST_CXX="${NVCC_CCBIN:-/opt/rh/gcc-toolset-13/root/usr/bin/g++}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"
FORCE_RUN="${FORCE_RUN:-0}"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

export PATH="$(dirname "${HOST_CXX}"):${PATH}"

mkdir -p "${BUILD_DIR}"

echo "[1/2] Building reference-kernel validator"
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
  -o "${BUILD_DIR}/demo-dtc-reference-tf32"

echo "[2/2] Running validator"
GPU_AVAILABLE=0
if [[ "${FORCE_RUN}" == "1" ]]; then
  GPU_AVAILABLE=1
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  GPU_AVAILABLE=1
elif [[ -e /dev/nvidiactl || -e /dev/nvidia0 ]]; then
  GPU_AVAILABLE=1
fi

if [[ "${GPU_AVAILABLE}" == "1" ]]; then
  "${BUILD_DIR}/demo-dtc-reference-tf32" "$@"
else
  echo "GPU runtime was not detected in the current command environment."
  echo "This often happens inside IDE/sandbox terminals even when the login shell has GPUs."
  echo "Build succeeded, but runtime validation is skipped."
  echo "If your shell can access GPUs, run one of these commands manually:"
  echo "  CUDA_ARCH=${CUDA_ARCH} ${BUILD_DIR}/demo-dtc-reference-tf32 ${*:-<args>}"
  echo "  FORCE_RUN=1 CUDA_ARCH=${CUDA_ARCH} ${BASH_SOURCE[0]} ${*:-<args>}"
fi
