#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Backend selector (extensible): dtc | acc | bitbsr | sr_bcrs | torch
export SPARSENE_SPMM_BACKEND="${SPARSENE_SPMM_BACKEND:-dtc}"

# DTC selector: base | strict_lb | multi_binding
export SPARSENE_DTC_VARIANT="${SPARSENE_DTC_VARIANT:-base}"

# src_fp32 DTC source root (must point to your implementation, not baseline)
export SPARSENE_DTC_SOURCE_ROOT="${SPARSENE_DTC_SOURCE_ROOT:-/workspace/sparsene/examples/src_fp32/dtc/testbed}"

# Optional: python extension module path/name under src_fp32 root.
# Example:
#   SPARSENE_DTC_MODULE_PATH=/workspace/sparsene/examples/src_fp32/dtc/testbed/build/lib.linux-x86_64-3.10
#   SPARSENE_DTC_MODULE_NAME=DTCSpMM
export SPARSENE_DTC_MODULE_PATH="${SPARSENE_DTC_MODULE_PATH:-}"
export SPARSENE_DTC_MODULE_NAME="${SPARSENE_DTC_MODULE_NAME:-DTCSpMM}"

# Keep strict by default: reject non-src_fp32 DTC modules unless explicitly overridden.
export SPARSENE_DTC_ALLOW_NON_SRC_FP32="${SPARSENE_DTC_ALLOW_NON_SRC_FP32:-0}"

# DTC execution options
export SPARSENE_DTC_BALANCE="${SPARSENE_DTC_BALANCE:-0}"
export SPARSENE_DTC_EXEPLAN="${SPARSENE_DTC_EXEPLAN:-float2_nonsplit}"
export SPARSENE_DTC_BLK_H="${SPARSENE_DTC_BLK_H:-16}"
export SPARSENE_DTC_BLK_W="${SPARSENE_DTC_BLK_W:-8}"

python3 "$ROOT_DIR/gcn/eva_gcn_sparsene.py" \
  --dataset-dir /workspace/baselines/dtc_datasets \
  --datasets DD,OVCAR-8H,Yeast,YeastH,ddi,protein,reddit,web-BerkStan \
  --hidden-list 64,128,256 \
  --layer-list 3,6 \
  --epochs 300 \
  --warmup-epochs 10 \
  --feature-dim 512 \
  --num-classes 16 \
  --device cuda:0 \
  --backend external \
  --external-module external_backend_spmm \
  --external-function dtc_spmm \
  --output-csv "$ROOT_DIR/result/gcn_e2e_dtc.csv"
