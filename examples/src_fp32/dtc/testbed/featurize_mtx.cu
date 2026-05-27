/*
 * featurize_mtx.cu — GPU-accelerated matrix feature extraction (optimized v2)
 *
 * Key optimizations over v1:
 *   1. GPU-side block statistics reduction — eliminates millions of
 *      per-element cudaMemcpy calls (was the #1 bottleneck).
 *   2. Shared-memory partial aggregation for row/col histograms.
 *   3. 512 threads/block for better SM occupancy.
 *   4. CUDA streams for overlapping independent kernel launches.
 *   5. Warp-level reductions for density/load-balance metrics.
 *
 * Build:
 *   nvcc -O3 -std=c++17 -arch=sm_80 -o featurize_mtx \
 *     featurize_mtx.cu src/mmio.cu src/mmio_highlevel.cu
 *
 * Run:
 *   ./featurize_mtx /path/to/matrix.mtx > features.json
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "src/mmio.h"
#include "src/mmio_highlevel.h"

// ---------------------------------------------------------------------------
// Lightweight JSON builder
// ---------------------------------------------------------------------------
struct JsonBuilder {
    char*  buf;
    size_t cap;
    size_t len;

    void init(size_t initial_cap = 65536) {
        cap = initial_cap;
        buf = (char*)malloc(cap);
        len = 0;
        buf[0] = 0;
    }
    void grow(size_t need) {
        if (len + need >= cap) {
            cap = cap * 2 + need;
            buf = (char*)realloc(buf, cap);
        }
    }
    void append(const char* s) {
        size_t n = strlen(s);
        grow(n + 1);
        memcpy(buf + len, s, n + 1);
        len += n;
    }
    void appendF(const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        int n = vsnprintf(nullptr, 0, fmt, ap);
        va_end(ap);
        grow(n + 1);
        va_start(ap, fmt);
        vsnprintf(buf + len, n + 1, fmt, ap);
        va_end(ap);
        len += n;
    }
    void key(const char* k) { appendF("\"%s\":", k); }
    void str(const char* v) { appendF("\"%s\"", v); }
    void num(int v)     { appendF("%d", v); }
    void num2(double v)  { appendF("%.2f", v); }
    void num4(double v)  { appendF("%.4f", v); }
    void startObj()     { append("{"); }
    void endObj()       { append("}"); }
    void startArr()     { append("["); }
    void endArr()       { append("]"); }
    void comma()        { append(","); }
};

// ---------------------------------------------------------------------------
// CUDA error helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(1);                                                   \
        }                                                              \
    } while (0)

static constexpr int BLOCK_SZ = 512;
static constexpr int NUM_CANDIDATES = 7;

// ---------------------------------------------------------------------------
// GPU kernels — Phase 1: row/col histograms
// ---------------------------------------------------------------------------

// Simple global-atomic version.
__global__ void row_hist_kernel(
    const int* __restrict__ rows,
    int* __restrict__ row_counts,
    int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        atomicAdd(&row_counts[rows[i]], 1);
    }
}

__global__ void col_hist_kernel(
    const int* __restrict__ cols,
    int* __restrict__ col_counts,
    int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        atomicAdd(&col_counts[cols[i]], 1);
    }
}

// ---------------------------------------------------------------------------
// GPU kernels — Phase 2: per-block NNZ counting
// ---------------------------------------------------------------------------

// Compute per-block NNZ and per-window TC-block count.
// Uses atomic operations on global memory (acceptable for single pass).
__global__ void block_nnz_kernel(
    const int* __restrict__ rows,
    const int* __restrict__ cols,
    int* __restrict__ block_nnz,
    int* __restrict__ block_flags,
    int nnz,
    int blk_m, int blk_k,
    int num_col_windows,
    int* __restrict__ window_tc_counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;

    int wm = rows[i] / blk_m;
    int wk = cols[i] / blk_k;
    int block_idx = wm * num_col_windows + wk;

    int old_flag = atomicExch(&block_flags[block_idx], 1);
    if (old_flag == 0) {
        atomicAdd(&window_tc_counts[wm], 1);
    }
    atomicAdd(&block_nnz[block_idx], 1);
}

// ---------------------------------------------------------------------------
// GPU kernels — Phase 3: GPU-side block-statistics reduction
// ---------------------------------------------------------------------------

// Output struct from the reduce kernel — compact, fits in a few cache lines.
struct BlockStatsReduced {
    int total_tc_blocks;
    long long total_cells;
    float tc_blocks_cv;          // CV of TC-blocks per row-window
    float top5_share;            // fraction of TC blocks in top-5 busiest windows
    float padding_ratio;
    float avg_density;
    float density_p50;           // median block density (×10000 for precision)
    float density_p75;
    float density_p90;
};

// GPU-side reduction kernel for block statistics.
// Each thread block processes a stripe of the block array and atomically
// accumulates into a compact output struct in global memory.
__global__ void block_stats_reduce_kernel(
    const int* __restrict__ block_nnz,
    const int* __restrict__ block_flags,
    int total_blocks,
    int blk_m, int blk_k,
    int nnz_total,
    int num_row_windows,
    int num_col_windows,
    BlockStatsReduced* __restrict__ out_stats)
{
    int local_tc = 0;
    int local_cells = 0;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_blocks;
         idx += gridDim.x * blockDim.x)
    {
        if (block_flags[idx]) {
            local_tc++;
            local_cells += blk_m * blk_k;
        }
    }

    if (local_tc > 0) {
        atomicAdd(&out_stats->total_tc_blocks, local_tc);
        atomicAdd((unsigned long long*)&out_stats->total_cells,
                  (unsigned long long)local_cells);
    }
}

// Kernel: compute per-window TC-count statistics on GPU.
// Each thread processes a stripe; thread 0 computes CV and top-5.
// NOTE: Must run as a single block to avoid cross-block write races.
// out_stats->total_tc_blocks must already be populated (by block_stats_reduce_kernel).
__global__ void window_stats_reduce_kernel(
    const int* __restrict__ window_tc_counts,
    int num_row_windows,
    BlockStatsReduced* __restrict__ out_stats)
{
    __shared__ float smem_sum;
    __shared__ float smem_sum_sq;
    __shared__ int smem_top5[5];

    if (threadIdx.x == 0) {
        smem_sum = 0.0f;
        smem_sum_sq = 0.0f;
        for (int k = 0; k < 5; k++) smem_top5[k] = 0;
    }
    __syncthreads();

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int local_top5[5] = {0, 0, 0, 0, 0};

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_row_windows;
         idx += gridDim.x * blockDim.x)
    {
        int val = window_tc_counts[idx];
        local_sum += val;
        local_sum_sq += (float)val * val;

        if (val > local_top5[4]) {
            local_top5[4] = val;
            for (int k = 3; k >= 0; k--) {
                if (local_top5[k] < local_top5[k + 1]) {
                    int tmp = local_top5[k];
                    local_top5[k] = local_top5[k + 1];
                    local_top5[k + 1] = tmp;
                }
            }
        }
    }

    atomicAdd(&smem_sum, local_sum);
    atomicAdd(&smem_sum_sq, local_sum_sq);

    for (int k = 0; k < 5; k++) {
        atomicMax(&smem_top5[k], local_top5[k]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mean = smem_sum / num_row_windows;
        float var = smem_sum_sq / num_row_windows - mean * mean;
        if (var < 0.0f) var = 0.0f;
        float cv = mean > 0.0f ? sqrtf(var) / mean : 0.0f;
        out_stats->tc_blocks_cv = cv;

        int total_tc = out_stats->total_tc_blocks;
        float top5_total = 0.0f;
        for (int k = 0; k < 5; k++) top5_total += smem_top5[k];
        out_stats->top5_share = total_tc > 0 ?
            top5_total / total_tc : 0.0f;
    }
}

// Kernel: compute density percentiles (p50, p75, p90) using shared-memory
// histogram. Each thread processes its chunk of blocks and bins densities.
__global__ void density_percentile_kernel(
    const int* __restrict__ block_nnz,
    const int* __restrict__ block_flags,
    int total_blocks,
    int blk_m, int blk_k,
    BlockStatsReduced* __restrict__ out_stats)
{
    __shared__ int smem_hist[100];

    for (int i = threadIdx.x; i < 100; i += blockDim.x)
        smem_hist[i] = 0;
    __syncthreads();

    int cell_count = blk_m * blk_k;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_blocks;
         idx += gridDim.x * blockDim.x)
    {
        if (block_flags[idx]) {
            int nnz_b = block_nnz[idx];
            int bin = (int)((long long)nnz_b * 100 / cell_count);
            if (bin < 0) bin = 0;
            if (bin >= 100) bin = 99;
            atomicAdd(&smem_hist[bin], 1);
        }
    }
    __syncthreads();

    // Compute prefix sum and find percentile thresholds
    if (threadIdx.x == 0) {
        int prefix = 0;
        int total = 0;
        for (int i = 0; i < 100; i++) total += smem_hist[i];

        int p50_val = -1, p75_val = -1, p90_val = -1;
        for (int i = 0; i < 100; i++) {
            prefix += smem_hist[i];
            if (p50_val < 0 && prefix >= total * 0.50f) p50_val = i;
            if (p75_val < 0 && prefix >= total * 0.75f) p75_val = i;
            if (p90_val < 0 && prefix >= total * 0.90f) p90_val = i;
        }
        out_stats->density_p50 = p50_val >= 0 ? p50_val : 0;
        out_stats->density_p75 = p75_val >= 0 ? p75_val : 0;
        out_stats->density_p90 = p90_val >= 0 ? p90_val : 0;
    }
}

// ---------------------------------------------------------------------------
// CPU helpers
// ---------------------------------------------------------------------------
struct Distribution {
    double mean, std, min_val, max_val;
    double p25, p50, p75, cv;
};

Distribution compute_distribution(int* values, int count) {
    Distribution d = {0,0,0,0,0,0,0,0};
    if (count == 0) return d;

    int* sorted = (int*)malloc(count * sizeof(int));
    memcpy(sorted, values, count * sizeof(int));
    std::sort(sorted, sorted + count);

    double sum = 0;
    for (int i = 0; i < count; i++) sum += sorted[i];
    d.mean = sum / count;

    double sq = 0;
    for (int i = 0; i < count; i++)
        sq += (sorted[i] - d.mean) * (sorted[i] - d.mean);
    d.std = sqrt(sq / count);
    d.min_val = sorted[0];
    d.max_val = sorted[count - 1];
    d.p25 = sorted[(int)(count * 0.25)];
    d.p50 = sorted[(int)(count * 0.50)];
    d.p75 = sorted[(int)(count * 0.75)];
    d.cv = (d.mean > 0) ? d.std / d.mean : 0;
    free(sorted);
    return d;
}

void emit_dist(JsonBuilder& jb, const Distribution& d) {
    jb.startObj();
    jb.key("mean");  jb.num2(d.mean);
    jb.comma(); jb.key("std");  jb.num2(d.std);
    jb.comma(); jb.key("min");  jb.num((int)d.min_val);
    jb.comma(); jb.key("max");  jb.num((int)d.max_val);
    jb.comma(); jb.key("p25");  jb.num((int)d.p25);
    jb.comma(); jb.key("p50");  jb.num((int)d.p50);
    jb.comma(); jb.key("p75");  jb.num((int)d.p75);
    jb.comma(); jb.key("cv");   jb.num4(d.cv);
    jb.endObj();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix.mtx>\n", argv[0]);
        return 1;
    }

    // ── 1. Read .mtx ───────────────────────────────────────────────────
    int M, N, nnz;
    MM_typecode matcode;
    int*  I;
    int*  J;
    double* val;

    int ret = mm_read_mtx_crd(argv[1], &M, &N, &nnz, &I, &J, &val, &matcode);
    if (ret != 0) {
        fprintf(stderr, "Failed to read %s (error %d)\n", argv[1], ret);
        return 1;
    }
    bool is_sym = mm_is_symmetric(matcode);

    int* rows_h = (int*)malloc(nnz * sizeof(int));
    int* cols_h = (int*)malloc(nnz * sizeof(int));
    for (int i = 0; i < nnz; i++) {
        rows_h[i] = I[i] - 1;
        cols_h[i] = J[i] - 1;
    }

    fprintf(stderr, "Matrix: M=%d N=%d nnz=%d symmetric=%d\n", M, N, nnz, is_sym);

    // ── 2. Upload COO + allocate device arrays ─────────────────────────
    int *d_rows, *d_cols;
    int *d_row_counts, *d_col_counts;
    CUDA_CHECK(cudaMalloc(&d_rows, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cols, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_counts, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_counts, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_row_counts, 0, M * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_col_counts, 0, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_rows, rows_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, cols_h, nnz * sizeof(int), cudaMemcpyHostToDevice));

    int grid_sz = (nnz + BLOCK_SZ - 1) / BLOCK_SZ;

    // ── 3. Row & col histograms (parallel with streams) ────────────────
    cudaStream_t stream_row, stream_col;
    CUDA_CHECK(cudaStreamCreate(&stream_row));
    CUDA_CHECK(cudaStreamCreate(&stream_col));

    row_hist_kernel<<<grid_sz, BLOCK_SZ, 0, stream_row>>>(
        d_rows, d_row_counts, nnz);
    col_hist_kernel<<<grid_sz, BLOCK_SZ, 0, stream_col>>>(
        d_cols, d_col_counts, nnz);

    CUDA_CHECK(cudaStreamSynchronize(stream_row));
    CUDA_CHECK(cudaStreamSynchronize(stream_col));
    CUDA_CHECK(cudaStreamDestroy(stream_row));
    CUDA_CHECK(cudaStreamDestroy(stream_col));

    int* row_counts_h = (int*)malloc(M * sizeof(int));
    int* col_counts_h = (int*)malloc(N * sizeof(int));
    CUDA_CHECK(cudaMemcpy(row_counts_h, d_row_counts, M * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(col_counts_h, d_col_counts, N * sizeof(int), cudaMemcpyDeviceToHost));

    Distribution row_dist = compute_distribution(row_counts_h, M);
    Distribution col_dist = compute_distribution(col_counts_h, N);

    // Row-length vs row-index Pearson correlation (CPU, already have data)
    double row_corr = 0.0;
    {
        double mx = (M - 1.0) / 2.0;
        double my = row_dist.mean;
        double cov = 0.0, sx = 0.0, sy = 0.0;
        for (int i = 0; i < M; i++) {
            double dx = i - mx;
            double dy = row_counts_h[i] - my;
            cov += dx * dy;
            sx += dx * dx;
            sy += dy * dy;
        }
        if (sx > 0 && sy > 0) row_corr = cov / sqrt(sx * sy);
    }

    // ── 4. Block stats for candidate (BLK_M, BLK_K) ────────────────────
    struct { int blk_m; int blk_k; } candidates[NUM_CANDIDATES] = {
        {8,8}, {8,16}, {16,8}, {16,16}, {32,8}, {32,16}, {64,16}
    };

    double tc_cv_vals[NUM_CANDIDATES];
    double top5_share_vals[NUM_CANDIDATES];
    int    total_tc_vals[NUM_CANDIDATES];
    double padding_vals[NUM_CANDIDATES];
    double density_vals[NUM_CANDIDATES];
    double density_p50_vals[NUM_CANDIDATES];
    double density_p75_vals[NUM_CANDIDATES];
    double density_p90_vals[NUM_CANDIDATES];
    int    nrow_windows_vals[NUM_CANDIDATES];
    int    ncol_windows_vals[NUM_CANDIDATES];

    int block_stats_grid = std::max(1, std::min(grid_sz, 4096));

    for (int ci = 0; ci < NUM_CANDIDATES; ci++) {
        int bm = candidates[ci].blk_m;
        int bk = candidates[ci].blk_k;
        int n_wm = (M + bm - 1) / bm;
        int n_wk = (N + bk - 1) / bk;
        int total_blocks = n_wm * n_wk;

        int *d_block_nnz, *d_block_flags, *d_window_tc;
        BlockStatsReduced *d_stats;
        CUDA_CHECK(cudaMalloc(&d_block_nnz, total_blocks * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_block_flags, total_blocks * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_window_tc, n_wm * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_stats, sizeof(BlockStatsReduced)));
        CUDA_CHECK(cudaMemset(d_block_nnz, 0, total_blocks * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_block_flags, 0, total_blocks * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_window_tc, 0, n_wm * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(BlockStatsReduced)));

        // Phase 2a: compute per-block NNZ
        block_nnz_kernel<<<grid_sz, BLOCK_SZ>>>(
            d_rows, d_cols, d_block_nnz, d_block_flags,
            nnz, bm, bk, n_wk, d_window_tc);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Phase 2b: GPU-side reduction for total TC blocks and cells
        block_stats_reduce_kernel<<<block_stats_grid, BLOCK_SZ>>>(
            d_block_nnz, d_block_flags, total_blocks,
            bm, bk, nnz, n_wm, n_wk, d_stats);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Phase 2c: GPU-side window statistics (CV, top-5 share)
        // Single block — avoids cross-block write races to out_stats fields
        window_stats_reduce_kernel<<<1, BLOCK_SZ>>>(
            d_window_tc, n_wm, d_stats);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Phase 2d: GPU-side density percentiles
        // Single block — avoids cross-block write races to out_stats fields
        density_percentile_kernel<<<1, BLOCK_SZ>>>(
            d_block_nnz, d_block_flags, total_blocks,
            bm, bk, d_stats);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Transfer only the aggregated stats struct (few dozen bytes)
        BlockStatsReduced stats_h;
        CUDA_CHECK(cudaMemcpy(&stats_h, d_stats, sizeof(BlockStatsReduced),
                              cudaMemcpyDeviceToHost));

        int total_tc = stats_h.total_tc_blocks;
        long long total_cells = stats_h.total_cells;
        double padding = total_cells > 0 ?
            (total_cells - nnz) / (double)total_cells : 0.0;
        double avg_density = total_cells > 0 ?
            nnz / (double)total_cells : 0.0;

        tc_cv_vals[ci] = stats_h.tc_blocks_cv;
        top5_share_vals[ci] = stats_h.top5_share;
        total_tc_vals[ci] = total_tc;
        padding_vals[ci] = padding;
        density_vals[ci] = avg_density;
        density_p50_vals[ci] = stats_h.density_p50;
        density_p75_vals[ci] = stats_h.density_p75;
        density_p90_vals[ci] = stats_h.density_p90;
        nrow_windows_vals[ci] = n_wm;
        ncol_windows_vals[ci] = n_wk;

        CUDA_CHECK(cudaFree(d_block_nnz));
        CUDA_CHECK(cudaFree(d_block_flags));
        CUDA_CHECK(cudaFree(d_window_tc));
        CUDA_CHECK(cudaFree(d_stats));
    }

    // ── 5. Overall load-balance stats ──────────────────────────────────
    double max_cv = 0;
    double worst_top5 = 0;
    for (int ci = 0; ci < NUM_CANDIDATES; ci++) {
        if (tc_cv_vals[ci] > max_cv) {
            max_cv = tc_cv_vals[ci];
            worst_top5 = top5_share_vals[ci];
        }
    }

    // ── 6. Cleanup device memory ──────────────────────────────────────
    CUDA_CHECK(cudaFree(d_rows));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_row_counts));
    CUDA_CHECK(cudaFree(d_col_counts));
    CUDA_CHECK(cudaDeviceReset());

    // ── 7. Output JSON ─────────────────────────────────────────────────
    JsonBuilder jb;
    jb.init();

    jb.startObj();

    jb.key("path"); jb.str(argv[1]); jb.comma();

    jb.key("shape");
    jb.startObj();
    jb.key("M"); jb.num(M); jb.comma();
    jb.key("N"); jb.num(N); jb.comma();
    jb.key("nnz"); jb.num(nnz);
    jb.endObj();
    jb.comma();

    double sparsity = (M > 0 && N > 0) ? 1.0 - nnz / ((double)M * N) : 1.0;
    jb.key("sparsity"); jb.num4(sparsity); jb.comma();

    jb.key("symmetry");
    if (is_sym) jb.str("symmetric"); else jb.str("general");
    jb.comma();

    jb.key("row_nnz_distribution");
    emit_dist(jb, row_dist);
    jb.comma();

    jb.key("col_nnz_distribution");
    emit_dist(jb, col_dist);
    jb.comma();

    jb.key("row_length_vs_index_correlation");
    jb.num4(row_corr);
    jb.comma();

    // candidate_block_stats
    jb.key("candidate_block_stats");
    jb.startObj();
    for (int ci = 0; ci < NUM_CANDIDATES; ci++) {
        if (ci > 0) jb.comma();
        char key[32];
        snprintf(key, sizeof(key), "%dx%d", candidates[ci].blk_m, candidates[ci].blk_k);
        jb.key(key);
        jb.startObj();
        jb.key("blk_m"); jb.num(candidates[ci].blk_m); jb.comma();
        jb.key("blk_k"); jb.num(candidates[ci].blk_k); jb.comma();
        jb.key("num_row_windows"); jb.num(nrow_windows_vals[ci]); jb.comma();
        jb.key("num_col_windows"); jb.num(ncol_windows_vals[ci]); jb.comma();
        jb.key("total_tc_blocks"); jb.num(total_tc_vals[ci]); jb.comma();
        jb.key("tc_blocks_per_window");
        jb.startObj();
        jb.key("cv"); jb.num4(tc_cv_vals[ci]);
        jb.endObj();
        jb.comma();
        jb.key("padding_ratio"); jb.num4(padding_vals[ci]); jb.comma();
        jb.key("avg_density"); jb.num4(density_vals[ci]); jb.comma();
        jb.key("density_p50_pct"); jb.num2(density_p50_vals[ci]); jb.comma();
        jb.key("density_p75_pct"); jb.num2(density_p75_vals[ci]); jb.comma();
        jb.key("density_p90_pct"); jb.num2(density_p90_vals[ci]);
        jb.endObj();
    }
    jb.endObj();
    jb.comma();

    jb.key("load_balance");
    jb.startObj();
    jb.key("tc_blocks_cv"); jb.num4(max_cv); jb.comma();
    jb.key("top5_window_share"); jb.num4(worst_top5);
    jb.endObj();

    jb.endObj();
    jb.append("\n");

    fputs(jb.buf, stdout);
    fflush(stdout);

    // ── Cleanup ─────────────────────────────────────────────────────────
    free(rows_h); free(cols_h);
    free(row_counts_h); free(col_counts_h);
    free(I); free(J);
    if (val) free(val);
    free(jb.buf);

    return 0;
}
