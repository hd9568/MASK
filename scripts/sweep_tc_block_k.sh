#!/usr/bin/env bash
# ==============================================================================
# TC_BLOCK_K sweep: find the best K (8/16/32) per dataset across all 26 .mtx files.
#
# Usage:
#   bash scripts/sweep_tc_block_k.sh
#
# Env overrides:
#   DATASET_DIR        path to .mtx files (default /mnt/lustre-client/zhangkaige/selected_mtx)
#   RESULT_ROOT        output root (default REPO/.tc_block_k_sweep_<timestamp>)
#   SKIP_GENERATE      skip .inc regeneration (default 0)
#   GPU_IDLE_THRESHOLD max GPU util % before launching (default 25)
#   GPU_IDLE_TIMEOUT   max seconds to wait for an idle GPU (default 600)
#   RUN_TIMEOUT_SEC    max seconds for one benchmark process (default 600)
#   N_COLS             B matrix columns (default 512)
#   WARMUP             warmup runs (default 5)
#   REPEAT             benchmark runs (default 10)
# ==============================================================================
set -eu

# ── exit trap for debugging silent failures ──
_on_exit() {
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "SCRIPT EXITED WITH CODE ${rc} at line ${BASH_LINENO[0]:-?}" >&2
    fi
}
trap _on_exit EXIT

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_DIR="${ROOT}/python"
TESTBED_DIR="${ROOT}/examples/src_fp32/dtc/testbed"
BUILD_DIR="${TESTBED_DIR}/build_tc_block_k_sweep"
DATASET_DIR="${DATASET_DIR:-/mnt/lustre-client/zhangkaige/selected_mtx}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/.tc_block_k_sweep_${TIMESTAMP}}"
INC_PATH="${PY_DIR}/test/results/dtc.inc"

# ── tunables ──────────────────────────────────────────────────────────────
K_VALUES=(8 16 32)
GPU_IDLE_THRESHOLD="${GPU_IDLE_THRESHOLD:-25}"
GPU_IDLE_TIMEOUT="${GPU_IDLE_TIMEOUT:-600}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-600}"
N_COLS="${N_COLS:-512}"
WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-10}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
VENV_PATH="/mnt/lustre-client/huoda/venv/llm-sog"

# ── helpers ───────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
log2() { echo "[$(date '+%H:%M:%S')]   $*"; }

gpu_util_pct() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null \
        | sed 's/ %//' | head -1 | tr -d ' '
}

wait_gpu_idle() {
    local start_seconds=$SECONDS
    while true; do
        local util
        util="$(gpu_util_pct)"
        if [[ -z "$util" ]]; then
            if (( SECONDS - start_seconds >= GPU_IDLE_TIMEOUT )); then
                log "FATAL: nvidia-smi unavailable for ${GPU_IDLE_TIMEOUT}s"
                return 1
            fi
            log "WARNING: nvidia-smi failed, waiting 10s..."
            sleep 10
            continue
        fi
        if [[ "$util" -lt "$GPU_IDLE_THRESHOLD" ]]; then
            log "GPU util=${util}% < ${GPU_IDLE_THRESHOLD}%, proceeding."
            return 0
        fi
        if (( SECONDS - start_seconds >= GPU_IDLE_TIMEOUT )); then
            log "FATAL: GPU did not become idle within ${GPU_IDLE_TIMEOUT}s"
            return 1
        fi
        log "GPU util=${util}% >= ${GPU_IDLE_THRESHOLD}%, waiting 5s..."
        sleep 5
    done
}

is_positive_number() {
    awk -v value="$1" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value + 0 > 0) }'
}

# ── setup ─────────────────────────────────────────────────────────────────
mkdir -p "${BUILD_DIR}" "${RESULT_ROOT}"
export PYTHONPATH="${PY_DIR}:${PYTHONPATH:-}"
export PATH="$(dirname "${NVCC_CCBIN:-/opt/rh/gcc-toolset-13/root/usr/bin/g++}"):${PATH}"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    source "${VENV_PATH}/bin/activate"
fi

log "============================================================"
log "TC_BLOCK_K sweep started"
log "  K values:       ${K_VALUES[*]}"
log "  datasets from:  ${DATASET_DIR}"
log "  N (columns):    ${N_COLS}"
log "  warmup/repeat:  ${WARMUP}/${REPEAT}"
log "  GPU threshold:  ${GPU_IDLE_THRESHOLD}%"
log "  result root:    ${RESULT_ROOT}"
log "  build dir:      ${BUILD_DIR}"
log "============================================================"

# ── 1. Generate / reuse .inc ──────────────────────────────────────────────
if [[ "${SKIP_GENERATE}" != "1" ]]; then
    log "Generating ${INC_PATH} ..."
    ( cd "${PY_DIR}" && python test/generate_sparse_formats.py \
        --results-dir test/results --formats ME_TCF >/dev/null ) || {
        log "FATAL: .inc generation failed"
        exit 1
    }
    log "Generation OK"
else
    log "SKIP_GENERATE=1, reusing ${INC_PATH}"
fi

if [[ ! -s "${INC_PATH}" ]]; then
    log "FATAL: generated include is missing or empty: ${INC_PATH}"
    exit 1
fi
INC_SHA256="$(sha256sum "${INC_PATH}" | awk '{print $1}')"
GENERATED_TILE_B="$(grep -m1 -oP '#define\s+SPARSENE_GENERATED_TILE_B\s+\K[0-9]+' "${INC_PATH}" || true)"
GENERATED_TILE_B="${GENERATED_TILE_B:-unknown}"
log "Generated include: ${INC_PATH} (sha256=${INC_SHA256})"

# ── 2. Compile binaries ───────────────────────────────────────────────────
BINARIES=()
for k in "${K_VALUES[@]}"; do
    BIN="${BUILD_DIR}/generated_tbk${k}"
    BUILD_LOG="${RESULT_ROOT}/build_tbk${k}.log"
    TMP_BIN="${BIN}.tmp.$$"
    log "Compiling K=${k} ..."
    if nvcc -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo \
        -ftemplate-backtrace-limit=0 \
        -arch="${CUDA_ARCH:-sm_80}" \
        -ccbin "${NVCC_CCBIN:-/opt/rh/gcc-toolset-13/root/usr/bin/g++}" \
        -Xcompiler -fopenmp \
        -Df32 \
        "-DTC_BLOCK_K=${k}" \
        "-DTC_BLOCK_M=16" \
        "-DROW_WINDOW_M=16" \
        "-DSPARSENE_GENERATED_INC_PATH=\"${INC_PATH}\"" \
        "-I${TESTBED_DIR}/cutlass/include" \
        "-I${TESTBED_DIR}/cutlass/examples/common" \
        "-I${TESTBED_DIR}/cutlass/tools/util/include" \
        "-I${TESTBED_DIR}/src" \
        "-I${TESTBED_DIR}" \
        "${TESTBED_DIR}/host_program_generated.cu" \
        "${TESTBED_DIR}/main.cu" \
        "${TESTBED_DIR}/src/mmio.cu" \
        "${TESTBED_DIR}/src/mmio_highlevel.cu" \
        "${TESTBED_DIR}/src/utils.cu" \
        -lcusparse -lcublas \
        -o "${TMP_BIN}" > "${BUILD_LOG}" 2>&1; then
        compile_rc=0
    else
        compile_rc=$?
    fi
    if [[ "${compile_rc}" -ne 0 ]]; then
        rm -f "${TMP_BIN}"
        log "FATAL: compile failed for K=${k}"
        tail -20 "${BUILD_LOG}" >&2
        exit 1
    fi
    mv -f "${TMP_BIN}" "${BIN}"
    log "Compile K=${k} OK  →  ${BIN}"
    BINARIES+=("${BIN}")
done

PROVENANCE_FILE="${RESULT_ROOT}/provenance.txt"
{
    echo "generated_inc=${INC_PATH}"
    echo "generated_inc_sha256=${INC_SHA256}"
    echo "host_source=${TESTBED_DIR}/host_program_generated.cu"
    echo "main_source=${TESTBED_DIR}/main.cu"
    echo "cuda_arch=${CUDA_ARCH:-sm_80}"
    echo "tc_block_m=16"
    echo "row_window_m=16"
    echo "tile_b=${GENERATED_TILE_B}"
    for idx in "${!BINARIES[@]}"; do
        echo "binary_${K_VALUES[$idx]}=${BINARIES[$idx]}"
        echo "binary_${K_VALUES[$idx]}_sha256=$(sha256sum "${BINARIES[$idx]}" | awk '{print $1}')"
    done
} > "${PROVENANCE_FILE}"

# ── 3. Collect datasets ───────────────────────────────────────────────────
mapfile -t DATASET_PATHS < <(ls -1 "${DATASET_DIR}"/*.mtx 2>/dev/null | sort)
if [[ "${#DATASET_PATHS[@]}" -eq 0 ]]; then
    log "FATAL: no .mtx files in ${DATASET_DIR}"
    exit 1
fi
log "Found ${#DATASET_PATHS[@]} datasets"

# ── 4. CSV header ─────────────────────────────────────────────────────────
CSV="${RESULT_ROOT}/sweep_summary.csv"
{
    echo "dataset,M,K,N,tc_block_k,mykernel_ms,cusparse_ms,speedup_vs_cusparse,status,error_count,nonfinite_count,exit_code,inc_sha256,binary_sha256"
} > "${CSV}"

BEST_CSV="${RESULT_ROOT}/best_per_dataset.csv"
{
    echo "dataset,M,K,N,best_tc_block_k,best_mykernel_ms,cusparse_ms,best_speedup"
} > "${BEST_CSV}"

# ── 5. Main sweep ─────────────────────────────────────────────────────────
total_combos=$((${#DATASET_PATHS[@]} * ${#K_VALUES[@]}))
combo_idx=0

for mtx_path in "${DATASET_PATHS[@]}"; do
    ds="$(basename "${mtx_path}" .mtx)"
    ds_log_dir="${RESULT_ROOT}/logs/${ds}"
    mkdir -p "${ds_log_dir}"

    # Per-dataset best tracker
    best_k=0
    best_time=999999.0
    best_cu=0.0

    log "──────────────────────────────────────────────────"
    log "Dataset [${ds}]  (${mtx_path})"

    for k in "${K_VALUES[@]}"; do
        combo_idx=$((combo_idx + 1))
        bin_idx=$((k == 8 ? 0 : k == 16 ? 1 : 2))
        BIN="${BINARIES[$bin_idx]}"
        LOG_FILE="${ds_log_dir}/tbk${k}.log"
        BIN_SHA256="$(sha256sum "${BIN}" | awk '{print $1}')"

        log2 "  [${combo_idx}/${total_combos}] K=${k}"

        # Wait for GPU to be idle
        wait_gpu_idle

        # Warmup first (to trigger any checkpoint writes), then benchmark
        log2 "    running benchmark (warmup=${WARMUP}, repeat=${REPEAT})..."

        run_rc=0
        if timeout "${RUN_TIMEOUT_SEC}" env SPARSENE_DISABLE_MTX_CACHE=1 \
                "${BIN}" \
                -filename "${mtx_path}" \
                -mtx_flag 1 \
                -N "${N_COLS}" \
                -warmup "${WARMUP}" \
                -repeat "${REPEAT}" \
                > "${LOG_FILE}" 2>&1; then
            run_rc=0
        else
            run_rc=$?
        fi

        # ── parse output ──
        my=""
        cu=""
        err_count=""
        nonfinite_count=""
        M=0
        K_dim=0
        N_dim=0
        status="unknown"

        # Matrix dims
        dim_match="$(grep -m1 'Sparse A\[' "${LOG_FILE}" || true)"
        if [[ "$dim_match" =~ A\[([0-9]+),([0-9]+)\].*B\[([0-9]+),([0-9]+)\].*C\[([0-9]+),([0-9]+)\] ]]; then
            M="${BASH_REMATCH[1]}"
            K_dim="${BASH_REMATCH[2]}"
            N_dim="${BASH_REMATCH[6]}"
        fi

        # Timings
        if my_line="$(grep -oP 'mykernel_time:\s*\K[0-9.]+' "${LOG_FILE}" | head -1 || true)"; then
            my="$(echo "$my_line" | head -1 || true)"
        else
            my=""
        fi
        if cu_line="$(grep -oP 'cusparse_time:\s*\K[0-9.]+' "${LOG_FILE}" | head -1 || true)"; then
            cu="$(echo "$cu_line" | head -1 || true)"
        else
            cu=""
        fi

        # Full correctness summary emitted by main.cu.
        verification_line="$(grep -m1 '^VERIFICATION ' "${LOG_FILE}" || true)"
        if [[ -n "$verification_line" ]]; then
            err_count="$(grep -oP 'errors=\K[0-9]+' <<< "$verification_line" || true)"
            nonfinite_count="$(grep -oP 'nonfinite=\K[0-9]+' <<< "$verification_line" || true)"
        fi

        # Status determination
        if [[ "$run_rc" -eq 124 ]]; then
            status="timeout"
        elif [[ -n "$nonfinite_count" && "$nonfinite_count" -gt 0 ]]; then
            status="nonfinite_${nonfinite_count}"
        elif [[ -n "$err_count" && "$err_count" -gt 0 ]]; then
            status="correctness_errors_${err_count}"
        elif [[ "$run_rc" -ne 0 ]]; then
            status="process_exit_${run_rc}"
        elif [[ -z "$my" ]] || ! is_positive_number "$my"; then
            if grep -q 'OS call failed.*304' "${LOG_FILE}"; then
                status="gpu_busy_304"
            else
                status="no_timing"
            fi
        elif [[ -z "$cu" ]] || ! is_positive_number "$cu"; then
            status="no_cusparse_timing"
        elif [[ -z "$err_count" || -z "$nonfinite_count" ]]; then
            status="missing_verification"
        else
            status="ok"
        fi

        # Speedup
        speedup=""
        if [[ "$status" == "ok" && -n "$my" && -n "$cu" ]]; then
            speedup="$(python3 -c "print(f'{float($cu)/float($my):.4f}')" 2>/dev/null || echo "")"

            # Track best
            if python3 -c "exit(0 if float($my) < float($best_time) else 1)" 2>/dev/null; then
                best_time="$my"
                best_k="$k"
                best_cu="$cu"
            fi
        fi

        # Log result
        best_flag=""
        if [[ "$k" == "$best_k" ]]; then
            best_flag="*(best)"
        fi

        if [[ "$status" == "ok" ]]; then
            log2 "    K=${k}  my=${my}ms  cu=${cu}ms  speedup=${speedup}x  errors=${err_count}  ${best_flag}"
        else
            log2 "    K=${k}  status=${status}  exit=${run_rc}  my=${my:-NA}ms  errors=${err_count:-NA}  nonfinite=${nonfinite_count:-NA}"
        fi

        # Append to CSV
        echo "${ds},${M},${K_dim},${N_dim},${k},${my:-NA},${cu:-NA},${speedup:-NA},${status},${err_count:-NA},${nonfinite_count:-NA},${run_rc},${INC_SHA256},${BIN_SHA256}" \
            >> "${CSV}"
    done

    # ── Dataset best ──
    best_speedup=""
    if [[ "$best_k" -gt 0 && "$best_time" != "999999.0" ]]; then
        best_speedup="$(python3 -c "print(f'{float($best_cu)/float($best_time):.4f}')" 2>/dev/null || echo "")"
        log "  >> BEST for ${ds}: K=${best_k}, ${best_time}ms, speedup=${best_speedup}x"
        echo "${ds},${M},${K_dim},${N_dim},${best_k},${best_time},${best_cu},${best_speedup}" \
            >> "${BEST_CSV}"
    else
        log "  >> ${ds}: no valid runs, all K values failed"
        echo "${ds},${M},${K_dim},${N_dim},0,NA,NA,NA" >> "${BEST_CSV}"
    fi
done

# ── 6. Summary table (markdown) ───────────────────────────────────────────
SUMMARY_MD="${RESULT_ROOT}/sweep_summary.md"
{
    echo "# TC_BLOCK_K Sweep Results"
    echo ""
    echo "**Sweep date**: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "**K values**: ${K_VALUES[*]}"
    echo "**Datasets**: ${#DATASET_PATHS[@]}"
    echo "**N (columns)**: ${N_COLS}"
    echo "**Warmup / Repeat**: ${WARMUP} / ${REPEAT}"
    echo "**Generated include SHA256**: ${INC_SHA256}"
    echo "**Generated TILE_B**: ${GENERATED_TILE_B}"
    echo ""
    echo "## Best TC_BLOCK_K per Dataset"
    echo ""
    echo "| dataset | M | K | N | best_K | mykernel_ms | cusparse_ms | speedup |"
    echo "|---|---:|---:|---:|---:|---:|---:|---:|"
} > "${SUMMARY_MD}"

ok_count=0
fail_count=0
total_speedup=0.0
faster_count=0
faster_speedup_sum=0.0
faster_datasets=""
max_speedup=0.0
max_speedup_dataset=""

while IFS=',' read -r ds M K_dim N_dim best_k best_time best_cu best_speedup; do
    [[ "$ds" == "dataset" ]] && continue
    if [[ "$best_k" == "0" || "$best_k" == "" ]]; then
        echo "| ${ds} | ${M} | ${K_dim} | ${N_dim} | — | — | — | — |" >> "${SUMMARY_MD}"
        fail_count=$((fail_count + 1))
    else
        echo "| ${ds} | ${M} | ${K_dim} | ${N_dim} | **${best_k}** | ${best_time} | ${best_cu} | ${best_speedup} |" >> "${SUMMARY_MD}"
        ok_count=$((ok_count + 1))
        total_speedup="$(python3 -c "print(${total_speedup} + ${best_speedup})" 2>/dev/null || echo "${total_speedup}")"
        if awk -v value="${best_speedup}" 'BEGIN { exit !(value > 1.0) }'; then
            faster_count=$((faster_count + 1))
            faster_speedup_sum="$(python3 -c "print(${faster_speedup_sum} + ${best_speedup})")"
            faster_datasets="${faster_datasets}${faster_datasets:+, }${ds}(${best_speedup}x)"
            if awk -v value="${best_speedup}" -v current="${max_speedup}" 'BEGIN { exit !(value > current) }'; then
                max_speedup="${best_speedup}"
                max_speedup_dataset="${ds}"
            fi
        fi
    fi
done < "${BEST_CSV}"

if [[ "$ok_count" -gt 0 ]]; then
    avg_speedup="$(python3 -c "print(round(float(${total_speedup}) / ${ok_count}, 4))" 2>/dev/null || echo "N/A")"
    {
        echo ""
        echo "**Summary**: ${ok_count}/${#DATASET_PATHS[@]} datasets OK"
        echo "- Average best speedup: ${avg_speedup}x"
    } >> "${SUMMARY_MD}"
fi
if [[ "$fail_count" -gt 0 ]]; then
    echo "- Failed: ${fail_count} datasets" >> "${SUMMARY_MD}"
fi

{
    echo ""
    echo "## Faster than cuSPARSE"
    echo ""
    if [[ "$ok_count" -eq 0 ]]; then
        echo "- No valid datasets were available for comparison."
    elif [[ "$faster_count" -eq 0 ]]; then
        echo "- 0/${ok_count} valid datasets outperform cuSPARSE."
    else
        faster_avg="$(python3 -c "print(round(float(${faster_speedup_sum}) / ${faster_count}, 4))")"
        faster_pct="$(python3 -c "print(round(100.0 * ${faster_count} / ${ok_count}, 2))")"
        echo "- Count: ${faster_count}/${ok_count} valid datasets (${faster_pct}%)"
        echo "- Average speedup among faster datasets: ${faster_avg}x"
        echo "- Maximum speedup: ${max_speedup}x (${max_speedup_dataset})"
        echo "- Datasets: ${faster_datasets}"
    fi
} >> "${SUMMARY_MD}"

# ── 7. Done ───────────────────────────────────────────────────────────────
log "============================================================"
log "Sweep complete."
log "  Per-dataset best:   ${BEST_CSV}"
log "  Full raw data:      ${CSV}"
log "  Markdown summary:   ${SUMMARY_MD}"
log "  Per-dataset logs:   ${RESULT_ROOT}/logs/<dataset>/"
log "  Provenance:         ${PROVENANCE_FILE}"
log "============================================================"

# Print the best table
cat "${BEST_CSV}"
