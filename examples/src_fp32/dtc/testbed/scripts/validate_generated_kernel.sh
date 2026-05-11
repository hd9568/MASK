#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
TESTBED_DIR="${ROOT}/examples/src_fp32/dtc/testbed"
PY_DIR="${ROOT}/python"
BUILD_DIR="${TESTBED_DIR}/build_generated"
VENV_PATH="${HOME}/llm-sog-venv"
HOST_CXX="${NVCC_CCBIN:-/opt/rh/gcc-toolset-13/root/usr/bin/g++}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"
FORCE_RUN="${FORCE_RUN:-0}"
PATCHED_ONLY="${PATCHED_ONLY:-1}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PY_DIR}:${PYTHONPATH:-}"
export PATH="$(dirname "${HOST_CXX}"):${PATH}"
export SPARSENE_DISABLE_MTX_CACHE="${SPARSENE_DISABLE_MTX_CACHE:-1}"

if [[ "${PATCHED_ONLY}" != "1" && "${SKIP_GENERATE}" != "1" ]]; then
  echo "[1/3] Generating ${PY_DIR}/me_tcf_kernel3.inc"
  (
    cd "${PY_DIR}"
    python test/test_generate.py
  )
elif [[ "${PATCHED_ONLY}" != "1" ]]; then
  echo "[1/3] Skipping regeneration and using existing ${PY_DIR}/me_tcf_kernel3.inc"
fi

mkdir -p "${BUILD_DIR}"

if [[ "${PATCHED_ONLY}" == "1" ]]; then
  echo "[1/2] Building generated-kernel validator from patched inc"
else
  echo "[2/3] Building generated-kernel validator"
fi
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
  $([[ "${PATCHED_ONLY}" == "1" ]] && printf '%s' "-DUSE_PATCHED_INC") \
  "${TESTBED_DIR}/host_program_generated.cu" \
  "${TESTBED_DIR}/main.cu" \
  "${TESTBED_DIR}/src/mmio.cu" \
  "${TESTBED_DIR}/src/mmio_highlevel.cu" \
  "${TESTBED_DIR}/src/utils.cu" \
  -lcusparse \
  -lcublas \
  -o "${BUILD_DIR}/demo-dtc-generated-tf32"

if [[ "${PATCHED_ONLY}" == "1" ]]; then
  echo "[2/2] Running validator"
else
  echo "[3/3] Running validator"
fi
GPU_AVAILABLE=0
if [[ "${FORCE_RUN}" == "1" ]]; then
  GPU_AVAILABLE=1
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  GPU_AVAILABLE=1
elif [[ -e /dev/nvidiactl || -e /dev/nvidia0 ]]; then
  GPU_AVAILABLE=1
fi

if [[ "${GPU_AVAILABLE}" == "1" ]]; then
  "${BUILD_DIR}/demo-dtc-generated-tf32" "$@"
else
  echo "GPU runtime was not detected in the current command environment."
  echo "This often happens inside IDE/sandbox terminals even when the login shell has GPUs."
  echo "Build succeeded, but runtime validation is skipped."
  echo "If your shell can access GPUs, run one of these commands manually:"
  echo "  CUDA_ARCH=${CUDA_ARCH} ${BUILD_DIR}/demo-dtc-generated-tf32 ${*:-<args>}"
  echo "  FORCE_RUN=1 CUDA_ARCH=${CUDA_ARCH} ${BASH_SOURCE[0]} ${*:-<args>}"
fi
