#!/usr/bin/env bash
# Build the GPU-accelerated matrix feature extractor (optimized v2)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_CXX="${NVCC_CCBIN:-/opt/rh/gcc-toolset-13/root/usr/bin/g++}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"

nvcc -O3 -std=c++17 -arch="${CUDA_ARCH}" -ccbin "${HOST_CXX}" \
  -I"${ROOT}" \
  -I"${ROOT}/src" \
  -o "${ROOT}/featurize_mtx" \
  "${ROOT}/featurize_mtx.cu" \
  "${ROOT}/src/mmio.cu" \
  "${ROOT}/src/mmio_highlevel.cu"

echo "Built: ${ROOT}/featurize_mtx"
