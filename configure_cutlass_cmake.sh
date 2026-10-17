#!/usr/bin/env bash
# 在本地 examples/cutlass 目录配置 CUTLASS（解决 CMake CUDA_ARCHITECTURES 为空、CUDA compiler unknown 等问题）
#
# 用法（在仓库 examples 目录下）:
#   bash configure_cutlass_cmake.sh
#
# 可选环境变量:
#   CUTLASS_SRC   默认: 本脚本所在目录下的 cutlass
#   BUILD         默认: $CUTLASS_SRC/build
#   CMAKE_CUDA_ARCHITECTURES / CUTLASS_NVCC_ARCHS  默认: 80
#   CMAKE_CUDA_COMPILER        默认: $HOME/local/cuda/bin/nvcc
#   CMAKE_CUDA_HOST_COMPILER   默认: gcc-toolset-13 的 g++（避免系统仅装 gcc 无 cc1plus）
#   CUTLASS_ENABLE_HEADERS_ONLY  默认 ON（仅头文件+工具，configure 很快；若需完整库设为 OFF）
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUTLASS_SRC="${CUTLASS_SRC:-$SCRIPT_DIR/cutlass}"
BUILD_DIR="${BUILD:-$CUTLASS_SRC/build}"
ARCH="${CMAKE_CUDA_ARCHITECTURES:-${CUTLASS_NVCC_ARCHS:-80}}"
NVCC="${CMAKE_CUDA_COMPILER:-$HOME/local/cuda/bin/nvcc}"
HOSTCXX="${CMAKE_CUDA_HOST_COMPILER:-/opt/rh/gcc-toolset-13/root/usr/bin/g++}"
HEADERS_ONLY="${CUTLASS_ENABLE_HEADERS_ONLY:-ON}"

if [[ ! -f "$CUTLASS_SRC/CMakeLists.txt" ]]; then
  echo "ERROR: 未找到 CUTLASS: $CUTLASS_SRC/CMakeLists.txt" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring CUTLASS in $BUILD_DIR ..."
cmake "$CUTLASS_SRC" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCUTLASS_NVCC_ARCHS="$ARCH" \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_HOST_COMPILER="$HOSTCXX" \
  -DCUTLASS_ENABLE_HEADERS_ONLY="$HEADERS_ONLY" \
  "$@"

echo "OK. Build with: cmake --build $BUILD_DIR -j"
