#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="$ROOT_DIR/result"
mkdir -p "$RESULT_DIR"

DATASET_LIST_FILE="${DATASET_LIST_FILE:-/workspace/scripts/sparsene_test_mtx_list.txt}"
DATASET_DIR="${DATASET_DIR:-/workspace/scripts/selected_npz}"

# Benchmark defaults (supports comma-separated hidden sizes, e.g. 128,256,512)
HIDDEN_LIST="${HIDDEN_LIST:-128}"
LAYER_LIST="${LAYER_LIST:-3}"
EPOCHS="${EPOCHS:-100}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-30}"
REPEAT_RUNS="${REPEAT_RUNS:-2}"
SEGMENT_TIMING="${SEGMENT_TIMING:-1}"
EXCLUDE_PREPROCESS="${EXCLUDE_PREPROCESS:-1}"
BACKEND_WARMUP_ITERS="${BACKEND_WARMUP_ITERS:-2}"
FEATURE_DIM="${FEATURE_DIM:-128}"
NUM_CLASSES="${NUM_CLASSES:-16}"
DEVICE="${DEVICE:-cuda:0}"

# Normalize hidden list (remove spaces) and build output suffix.
HIDDEN_LIST="${HIDDEN_LIST// /}"
if [[ -z "$HIDDEN_LIST" ]]; then
  echo "[ERROR] HIDDEN_LIST is empty"
  exit 1
fi
if [[ ! "$HIDDEN_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "[ERROR] HIDDEN_LIST must be comma-separated integers, got: $HIDDEN_LIST"
  exit 1
fi
HIDDEN_TAG="${HIDDEN_LIST//,/-}"
RESULT_TAG="${RESULT_TAG:-n${HIDDEN_TAG}}"
IFS=',' read -r -a HIDDEN_ARR <<< "$HIDDEN_LIST"

if [[ ! -f "$DATASET_LIST_FILE" ]]; then
  echo "[ERROR] dataset list not found: $DATASET_LIST_FILE"
  exit 1
fi

mapfile -t DATASETS_ARR < <(sed -E 's#.*/##; s#\.mtx$##' "$DATASET_LIST_FILE")
if [[ "${#DATASETS_ARR[@]}" -eq 0 ]]; then
  echo "[ERROR] no datasets found in: $DATASET_LIST_FILE"
  exit 1
fi
DATASETS_CSV="$(printf '%s\n' "${DATASETS_ARR[@]}" | paste -sd, -)"

echo "[INFO] dataset count: ${#DATASETS_ARR[@]}"
echo "[INFO] dataset dir: $DATASET_DIR"
echo "[INFO] hidden-list: $HIDDEN_LIST"
echo "[INFO] hidden count: ${#HIDDEN_ARR[@]}"
echo "[INFO] result tag: $RESULT_TAG"

run_all_dataset_once() {
  local name="$1"
  local ext_func="$2"
  local out_csv="$3"
  shift 3
  local -a env_kv=("$@")

  echo "[RUN] $name -> $out_csv"
  env "${env_kv[@]}" \
    python3 "$ROOT_DIR/gcn/eva_gcn_sparsene.py" \
      --dataset-dir "$DATASET_DIR" \
      --datasets "$DATASETS_CSV" \
      --hidden-list "$HIDDEN_LIST" \
      --layer-list "$LAYER_LIST" \
      --epochs "$EPOCHS" \
      --warmup-epochs "$WARMUP_EPOCHS" \
      --repeat-runs "$REPEAT_RUNS" \
      --segment-timing "$SEGMENT_TIMING" \
      --exclude-preprocess "$EXCLUDE_PREPROCESS" \
      --backend-warmup-iters "$BACKEND_WARMUP_ITERS" \
      --feature-dim "$FEATURE_DIM" \
      --num-classes "$NUM_CLASSES" \
      --device "$DEVICE" \
      --backend external \
      --external-module external_backend_spmm \
      --external-function "$ext_func" \
      --output-csv "$out_csv"
}

run_per_dataset_and_merge() {
  local name="$1"
  local ext_func="$2"
  local out_csv="$3"
  local tmp_prefix="$4"
  shift 4
  local -a env_kv=("$@")

  rm -f "${tmp_prefix}"_*.csv "$out_csv"

  for ds in "${DATASETS_ARR[@]}"; do
    local ds_csv="${tmp_prefix}_${ds}.csv"
    echo "[RUN] $name dataset=$ds"
    env "${env_kv[@]}" \
      python3 "$ROOT_DIR/gcn/eva_gcn_sparsene.py" \
        --dataset-dir "$DATASET_DIR" \
        --datasets "$ds" \
        --hidden-list "$HIDDEN_LIST" \
        --layer-list "$LAYER_LIST" \
        --epochs "$EPOCHS" \
        --warmup-epochs "$WARMUP_EPOCHS" \
        --repeat-runs "$REPEAT_RUNS" \
        --segment-timing "$SEGMENT_TIMING" \
        --exclude-preprocess "$EXCLUDE_PREPROCESS" \
        --backend-warmup-iters "$BACKEND_WARMUP_ITERS" \
        --feature-dim "$FEATURE_DIM" \
        --num-classes "$NUM_CLASSES" \
        --device "$DEVICE" \
        --backend external \
        --external-module external_backend_spmm \
        --external-function "$ext_func" \
        --output-csv "$ds_csv"
  done

  local first=1
  for ds in "${DATASETS_ARR[@]}"; do
    local ds_csv="${tmp_prefix}_${ds}.csv"
    if [[ ! -f "$ds_csv" ]]; then
      echo "[ERROR] missing per-dataset csv: $ds_csv"
      exit 1
    fi
    if [[ "$first" -eq 1 ]]; then
      cat "$ds_csv" > "$out_csv"
      first=0
    else
      tail -n +2 "$ds_csv" >> "$out_csv"
    fi
  done

  echo "[DONE] $name merged -> $out_csv"
}

run_per_dataset_per_hidden_and_merge() {
  local name="$1"
  local ext_func="$2"
  local out_csv="$3"
  local tmp_prefix="$4"
  shift 4
  local -a env_kv=("$@")

  rm -f "${tmp_prefix}"_*.csv "$out_csv"

  for h in "${HIDDEN_ARR[@]}"; do
    for ds in "${DATASETS_ARR[@]}"; do
      local ds_csv="${tmp_prefix}_${ds}_h${h}.csv"
      echo "[RUN] $name dataset=$ds hidden=$h"
      env "${env_kv[@]}" \
        python3 "$ROOT_DIR/gcn/eva_gcn_sparsene.py" \
          --dataset-dir "$DATASET_DIR" \
          --datasets "$ds" \
          --hidden-list "$h" \
          --layer-list "$LAYER_LIST" \
          --epochs "$EPOCHS" \
          --warmup-epochs "$WARMUP_EPOCHS" \
          --repeat-runs "$REPEAT_RUNS" \
          --segment-timing "$SEGMENT_TIMING" \
          --exclude-preprocess "$EXCLUDE_PREPROCESS" \
          --backend-warmup-iters "$BACKEND_WARMUP_ITERS" \
          --feature-dim "$FEATURE_DIM" \
          --num-classes "$NUM_CLASSES" \
          --device "$DEVICE" \
          --backend external \
          --external-module external_backend_spmm \
          --external-function "$ext_func" \
          --output-csv "$ds_csv"
    done
  done

  local first=1
  for h in "${HIDDEN_ARR[@]}"; do
    for ds in "${DATASETS_ARR[@]}"; do
      local ds_csv="${tmp_prefix}_${ds}_h${h}.csv"
      if [[ ! -f "$ds_csv" ]]; then
        echo "[ERROR] missing per-dataset-hidden csv: $ds_csv"
        exit 1
      fi
      if [[ "$first" -eq 1 ]]; then
        cat "$ds_csv" > "$out_csv"
        first=0
      else
        tail -n +2 "$ds_csv" >> "$out_csv"
      fi
    done
  done

  echo "[DONE] $name merged -> $out_csv"
}

run_flashsparse_per_dataset_per_hidden_and_merge() {
  local name="$1"
  local ext_func="$2"
  local out_csv="$3"
  local tmp_prefix="$4"
  shift 4
  local -a env_kv=("$@")

  local skip_list="${FLASHSPARSE_SKIP_DATASETS:-}"
  local continue_on_error="${FLASHSPARSE_CONTINUE_ON_ERROR:-1}"
  local failed_log="$RESULT_DIR/failed_flashsparse_${RESULT_TAG}.txt"

  skip_list="${skip_list// /}"
  rm -f "${tmp_prefix}"_*.csv "$out_csv" "$failed_log"

  for h in "${HIDDEN_ARR[@]}"; do
    for ds in "${DATASETS_ARR[@]}"; do
      if [[ -n "$skip_list" && ",${skip_list}," == *",${ds},"* ]]; then
        echo "[SKIP] $name dataset=$ds hidden=$h (FLASHSPARSE_SKIP_DATASETS)"
        continue
      fi

      local ds_csv="${tmp_prefix}_${ds}_h${h}.csv"
      echo "[RUN] $name dataset=$ds hidden=$h"
      if env "${env_kv[@]}" \
        python3 "$ROOT_DIR/gcn/eva_gcn_sparsene.py" \
          --dataset-dir "$DATASET_DIR" \
          --datasets "$ds" \
          --hidden-list "$h" \
          --layer-list "$LAYER_LIST" \
          --epochs "$EPOCHS" \
          --warmup-epochs "$WARMUP_EPOCHS" \
          --repeat-runs "$REPEAT_RUNS" \
          --segment-timing "$SEGMENT_TIMING" \
          --exclude-preprocess "$EXCLUDE_PREPROCESS" \
          --backend-warmup-iters "$BACKEND_WARMUP_ITERS" \
          --feature-dim "$FEATURE_DIM" \
          --num-classes "$NUM_CLASSES" \
          --device "$DEVICE" \
          --backend external \
          --external-module external_backend_spmm \
          --external-function "$ext_func" \
          --output-csv "$ds_csv"; then
        :
      else
        echo "[WARN] $name failed at dataset=$ds hidden=$h"
        echo "${ds},${h}" >> "$failed_log"
        if [[ "$continue_on_error" != "1" ]]; then
          echo "[ERROR] FLASHSPARSE_CONTINUE_ON_ERROR=0, aborting"
          exit 1
        fi
      fi
    done
  done

  local first=1
  local merged_rows=0
  for h in "${HIDDEN_ARR[@]}"; do
    for ds in "${DATASETS_ARR[@]}"; do
      local ds_csv="${tmp_prefix}_${ds}_h${h}.csv"
      if [[ ! -f "$ds_csv" ]]; then
        continue
      fi
      if [[ "$first" -eq 1 ]]; then
        cat "$ds_csv" > "$out_csv"
        first=0
      else
        tail -n +2 "$ds_csv" >> "$out_csv"
      fi
      merged_rows=$((merged_rows + 1))
    done
  done

  if [[ "$merged_rows" -eq 0 ]]; then
    echo "[ERROR] $name produced no csv rows to merge"
    exit 1
  fi

  echo "[DONE] $name merged -> $out_csv (segments=$merged_rows)"
  if [[ -f "$failed_log" ]]; then
    echo "[WARN] failed flashsparse cases logged at: $failed_log"
  fi
}

DTC_SOURCE_ROOT="${SPARSENE_DTC_SOURCE_ROOT:-/workspace/sparsene/examples/src_fp32/dtc/testbed}"
DTC_MODULE_NAME="${SPARSENE_DTC_MODULE_NAME:-DTCSpMM}"
SRBCRS_SOURCE_ROOT="${SPARSENE_SRBCRS_SOURCE_ROOT:-/workspace/sparsene/examples/src_fp32/sr_bcrs/testbed}"
SRBCRS_MODULE_NAME="${SPARSENE_SRBCRS_MODULE_NAME:-SRBCRSSpMM}"
FLASHSPARSE_SOURCE_ROOT="${SPARSENE_FLASHSPARSE_SOURCE_ROOT:-/workspace/baselines/FlashSparse/FlashSparse}"

# 1) DTC base
run_all_dataset_once \
  "dtc_base" \
  "dtc_spmm" \
  "$RESULT_DIR/gcn_e2e_dtc_base_26_${RESULT_TAG}.csv" \
  "SPARSENE_SPMM_BACKEND=dtc" \
  "SPARSENE_DTC_VARIANT=base" \
  "SPARSENE_DTC_SOURCE_ROOT=$DTC_SOURCE_ROOT" \
  "SPARSENE_DTC_MODULE_NAME=$DTC_MODULE_NAME" \
  "SPARSENE_DTC_ALLOW_NON_SRC_FP32=${SPARSENE_DTC_ALLOW_NON_SRC_FP32:-0}" \
  "SPARSENE_ALLOW_CSR_FALLBACK=${SPARSENE_ALLOW_CSR_FALLBACK:-0}" \
  "SPARSENE_DTC_TILE_B=${SPARSENE_DTC_TILE_B:-auto}"

# 2) DTC multi-binding
run_all_dataset_once \
  "dtc_multi_binding" \
  "dtc_spmm" \
  "$RESULT_DIR/gcn_e2e_dtc_multibind_26_${RESULT_TAG}.csv" \
  "SPARSENE_SPMM_BACKEND=dtc" \
  "SPARSENE_DTC_VARIANT=multi_binding" \
  "SPARSENE_DTC_SOURCE_ROOT=$DTC_SOURCE_ROOT" \
  "SPARSENE_DTC_MODULE_NAME=$DTC_MODULE_NAME" \
  "SPARSENE_DTC_ALLOW_NON_SRC_FP32=${SPARSENE_DTC_ALLOW_NON_SRC_FP32:-0}" \
  "SPARSENE_ALLOW_CSR_FALLBACK=${SPARSENE_ALLOW_CSR_FALLBACK:-0}" \
  "SPARSENE_DTC_TILE_B=${SPARSENE_DTC_TILE_B:-auto}"

# 3) DTC strict-lb (run one dataset+hidden per process for stability)
run_per_dataset_per_hidden_and_merge \
  "dtc_strict_lb" \
  "dtc_spmm" \
  "$RESULT_DIR/gcn_e2e_dtc_strict_lb_26_${RESULT_TAG}.csv" \
  "$RESULT_DIR/_tmp_dtc_strict_lb_${RESULT_TAG}" \
  "SPARSENE_SPMM_BACKEND=dtc" \
  "SPARSENE_DTC_VARIANT=strict_lb" \
  "SPARSENE_DTC_SOURCE_ROOT=$DTC_SOURCE_ROOT" \
  "SPARSENE_DTC_MODULE_NAME=$DTC_MODULE_NAME" \
  "SPARSENE_DTC_ALLOW_NON_SRC_FP32=${SPARSENE_DTC_ALLOW_NON_SRC_FP32:-0}" \
  "SPARSENE_EXTERNAL_RESET_PER_CASE=${SPARSENE_EXTERNAL_RESET_PER_CASE:-1}" \
  "SPARSENE_ALLOW_CSR_FALLBACK=${SPARSENE_ALLOW_CSR_FALLBACK:-0}" \
  "SPARSENE_DTC_TILE_B=${SPARSENE_DTC_TILE_B:-auto}"

# 4) SR-BCRS base (run one dataset+hidden per process to avoid OOM)
run_per_dataset_per_hidden_and_merge \
  "srbcrs_base" \
  "srbcrs_spmm" \
  "$RESULT_DIR/gcn_e2e_srbcrs_base_26_${RESULT_TAG}.csv" \
  "$RESULT_DIR/_tmp_srbcrs_base_${RESULT_TAG}" \
  "SPARSENE_SPMM_BACKEND=sr_bcrs" \
  "SPARSENE_SRBCRS_VARIANT=base" \
  "SPARSENE_SRBCRS_SOURCE_ROOT=$SRBCRS_SOURCE_ROOT" \
  "SPARSENE_SRBCRS_MODULE_NAME=$SRBCRS_MODULE_NAME" \
  "SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32=${SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32:-0}" \
  "SPARSENE_EXTERNAL_RESET_PER_CASE=${SPARSENE_EXTERNAL_RESET_PER_CASE:-1}" \
  "SPARSENE_ALLOW_CSR_FALLBACK=${SPARSENE_ALLOW_CSR_FALLBACK:-0}" \
  "SPARSENE_SRBCRS_TILE_B=${SPARSENE_SRBCRS_TILE_B:-auto}"

# 5) SR-BCRS 16x8 (run one dataset+hidden per process to avoid OOM)
run_per_dataset_per_hidden_and_merge \
  "srbcrs_16x8" \
  "srbcrs_spmm" \
  "$RESULT_DIR/gcn_e2e_srbcrs_16x8_26_${RESULT_TAG}.csv" \
  "$RESULT_DIR/_tmp_srbcrs_16x8_${RESULT_TAG}" \
  "SPARSENE_SPMM_BACKEND=sr_bcrs" \
  "SPARSENE_SRBCRS_VARIANT=16x8" \
  "SPARSENE_SRBCRS_SOURCE_ROOT=$SRBCRS_SOURCE_ROOT" \
  "SPARSENE_SRBCRS_MODULE_NAME=$SRBCRS_MODULE_NAME" \
  "SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32=${SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32:-0}" \
  "SPARSENE_EXTERNAL_RESET_PER_CASE=${SPARSENE_EXTERNAL_RESET_PER_CASE:-1}" \
  "SPARSENE_ALLOW_CSR_FALLBACK=${SPARSENE_ALLOW_CSR_FALLBACK:-0}" \
  "SPARSENE_SRBCRS_TILE_B=${SPARSENE_SRBCRS_TILE_B:-auto}"

# 6) FlashSparse (run one dataset+hidden per process; continue on per-case failures)
run_flashsparse_per_dataset_per_hidden_and_merge \
  "flashsparse" \
  "flashsparse_spmm" \
  "$RESULT_DIR/gcn_e2e_flashsparse_26_${RESULT_TAG}.csv" \
  "$RESULT_DIR/_tmp_flashsparse_${RESULT_TAG}" \
  "SPARSENE_SPMM_BACKEND=flashsparse" \
  "SPARSENE_FLASHSPARSE_VARIANT=${SPARSENE_FLASHSPARSE_VARIANT:-tf32_balance}" \
  "SPARSENE_FLASHSPARSE_SOURCE_ROOT=$FLASHSPARSE_SOURCE_ROOT" \
  "SPARSENE_FLASHSPARSE_ALLOW_NON_DEFAULT_ROOT=${SPARSENE_FLASHSPARSE_ALLOW_NON_DEFAULT_ROOT:-1}" \
  "SPARSENE_ALLOW_CSR_FALLBACK=${SPARSENE_ALLOW_CSR_FALLBACK:-0}" \
  "SPARSENE_EXTERNAL_RESET_PER_CASE=${SPARSENE_EXTERNAL_RESET_PER_CASE:-1}"

echo ""
echo "[DONE] all benchmarks completed."
echo "[DONE] result files:"
echo "  - $RESULT_DIR/gcn_e2e_dtc_base_26_${RESULT_TAG}.csv"
echo "  - $RESULT_DIR/gcn_e2e_dtc_multibind_26_${RESULT_TAG}.csv"
echo "  - $RESULT_DIR/gcn_e2e_dtc_strict_lb_26_${RESULT_TAG}.csv"
echo "  - $RESULT_DIR/gcn_e2e_srbcrs_base_26_${RESULT_TAG}.csv"
echo "  - $RESULT_DIR/gcn_e2e_srbcrs_16x8_26_${RESULT_TAG}.csv"
echo "  - $RESULT_DIR/gcn_e2e_flashsparse_26_${RESULT_TAG}.csv"
