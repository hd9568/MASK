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
#   GPU_IDLE_THRESHOLD max GPU util % before launching (default 20)
#   N_COLS             B matrix columns (default 512)
#   WARMUP             warmup runs (default 20)
#   REPEAT             benchmark runs (default 100)
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
INC_PATH="${PY_DIR}/me_tcf_kernel3.inc"

# ── tunables ──────────────────────────────────────────────────────────────
K_VALUES=(8 16 32)
GPU_IDLE_THRESHOLD="${GPU_IDLE_THRESHOLD:-20}"
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
    while true; do
        local util
        util="$(gpu_util_pct)"
        if [[ -z "$util" ]]; then
            log "WARNING: nvidia-smi failed, waiting 10s..."
            sleep 10
            continue
        fi
        if [[ "$util" -lt "$GPU_IDLE_THRESHOLD" ]]; then
            log "GPU util=${util}% < ${GPU_IDLE_THRESHOLD}%, proceeding."
            return 0
        fi
        log "GPU util=${util}% >= ${GPU_IDLE_THRESHOLD}%, waiting 5s..."
        sleep 5
    done
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
    ( cd "${PY_DIR}" && python test/test_generate.py >/dev/null ) || {
        log "FATAL: .inc generation failed"
        exit 1
    }
    log "Generation OK"
else
    log "SKIP_GENERATE=1, reusing ${INC_PATH}"
fi

# ── 2. Compile binaries ───────────────────────────────────────────────────
BINARIES=()
for k in "${K_VALUES[@]}"; do
    BIN="${BUILD_DIR}/generated_tbk${k}"
    if [[ -f "${BIN}" ]]; then
        log "Binary for K=${k} already exists, skipping compile"
        BINARIES+=("${BIN}")
        continue
    fi
    log "Compiling K=${k} ..."
    nvcc -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo \
        -ftemplate-backtrace-limit=0 \
        -arch="${CUDA_ARCH:-sm_80}" \
        -ccbin "${NVCC_CCBIN:-/opt/rh/gcc-toolset-13/root/usr/bin/g++}" \
        -Xcompiler -fopenmp \
        -Df32 \
        "-DTC_BLOCK_K=${k}" \
        "-DTC_BLOCK_M=16" \
        "-DROW_WINDOW_M=16" \
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
        -o "${BIN}" 2>&1 | tail -3
    if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
        log "FATAL: compile failed for K=${k}"
        exit 1
    fi
    log "Compile K=${k} OK  →  ${BIN}"
    BINARIES+=("${BIN}")
done

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
    echo "dataset,M,K,N,tc_block_k,mykernel_ms,cusparse_ms,speedup_vs_cusparse,status,error_count,best_for_dataset"
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

        log2 "  [${combo_idx}/${total_combos}] K=${k}"

        # Wait for GPU to be idle
        wait_gpu_idle

        # Warmup first (to trigger any checkpoint writes), then benchmark
        log2 "    running benchmark (warmup=${WARMUP}, repeat=${REPEAT})..."

        env SPARSENE_DISABLE_MTX_CACHE=1 \
            "${BIN}" \
            -filename "${mtx_path}" \
            -mtx_flag 1 \
            -N "${N_COLS}" \
            -warmup "${WARMUP}" \
            -repeat "${REPEAT}" \
            > "${LOG_FILE}" 2>&1 || true

        # ── parse output ──
        my=""
        cu=""
        err_count=0
        nan_ok="UNKNOWN"
        M=0
        K_dim=0
        N_dim=0
        status="unknown"

        # Matrix dims
        dim_match="$(grep -oP 'Sparse A\[\d+,\d+\] x B\[\d+,\d+\]' "${LOG_FILE}" | head -1 || true)"
        if [[ -n "$dim_match" ]]; then
            M="$(echo "$dim_match" | grep -oP 'A\[\K\d+' || true)"
            K_dim="$(echo "$dim_match" | grep -oP 'A\[\d+,\K\d+' || true)"
            N_dim="$(echo "$dim_match" | grep -oP 'C\[\K\d+' || true)"
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

        # Errors
        err_count="$(grep -c 'Error(' "${LOG_FILE}" || true)"
        err_count="${err_count:-0}"

        # NaN check
        if grep -q 'check nan:.*cusp.*mykernel' "${LOG_FILE}" 2>/dev/null; then
            nan_cusp="$(grep -oP 'check nan:\s*cusp\s*\K[0-9.\-]+' "${LOG_FILE}" | head -1 || true)"
            nan_my="$(grep -oP 'check nan:\s*cusp\s*[0-9.\-]+\s*mykernel\s*\K[0-9.\-]+' "${LOG_FILE}" | head -1 || true)"
            if [[ "$nan_cusp" == "$nan_my" ]]; then
                nan_ok="MATCH"
            else
                nan_ok="MISMATCH"
            fi
        fi

        # Status determination
        if [[ -z "$my" || "$my" == "0.000000" ]]; then
            if grep -q 'OS call failed.*304' "${LOG_FILE}"; then
                status="gpu_busy_304"
            else
                status="no_timing"
            fi
        elif [[ "$err_count" -gt 0 ]]; then
            status="correctness_errors_${err_count}"
        elif [[ "$nan_ok" == "MISMATCH" ]]; then
            status="nan_mismatch"
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
            log2 "    K=${k}  status=${status}  my=${my}ms  errors=${err_count}"
        fi

        # Append to CSV
        echo "${ds},${M},${K_dim},${N_dim},${k},${my:-NA},${cu:-NA},${speedup:-NA},${status},${err_count}," \
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
    echo ""
    echo "## Best TC_BLOCK_K per Dataset"
    echo ""
    echo "| dataset | M | K | N | best_K | mykernel_ms | cusparse_ms | speedup |"
    echo "|---|---:|---:|---:|---:|---:|---:|---:|"
} > "${SUMMARY_MD}"

ok_count=0
fail_count=0
total_speedup=0.0

while IFS=',' read -r ds M K_dim N_dim best_k best_time best_cu best_speedup; do
    [[ "$ds" == "dataset" ]] && continue
    if [[ "$best_k" == "0" || "$best_k" == "" ]]; then
        echo "| ${ds} | ${M} | ${K_dim} | ${N_dim} | — | — | — | — |" >> "${SUMMARY_MD}"
        fail_count=$((fail_count + 1))
    else
        echo "| ${ds} | ${M} | ${K_dim} | ${N_dim} | **${best_k}** | ${best_time} | ${best_cu} | ${best_speedup} |" >> "${SUMMARY_MD}"
        ok_count=$((ok_count + 1))
        total_speedup="$(python3 -c "print(${total_speedup} + ${best_speedup})" 2>/dev/null || echo "${total_speedup}")"
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

# ── 7. Done ───────────────────────────────────────────────────────────────
log "============================================================"
log "Sweep complete."
log "  Per-dataset best:   ${BEST_CSV}"
log "  Full raw data:      ${CSV}"
log "  Markdown summary:   ${SUMMARY_MD}"
log "  Per-dataset logs:   ${RESULT_ROOT}/logs/<dataset>/"
log "============================================================"

# Print the best table
cat "${BEST_CSV}"
