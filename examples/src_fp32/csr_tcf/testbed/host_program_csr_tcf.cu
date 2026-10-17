#include <cassert>
#include <cstdlib>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime_api.h>

#include <cute/tensor.hpp>

#include "../../../../python/test/results/csr_tcf.inc"

using namespace cute;

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))

#ifndef ROW_WINDOW_M
#define ROW_WINDOW_M 16
#endif
#ifndef TC_BLOCK_K
#define TC_BLOCK_K 8
#endif
#ifndef TC_BLOCK_M
#define TC_BLOCK_M 16
#endif

int sparsene_generated_tile_b() {
    const char* env = std::getenv("SPARSENE_GENERATED_TILE_B");
    if (env != nullptr && env[0] != '\0') {
        const int parsed = std::atoi(env);
        if (parsed == 16 || parsed == 32 || parsed == 64) return parsed;
    }
    return 64;
}

namespace {

template <int TILE_B>
void run_csr_tcf_kernel(
    int M, int N, int K,
    int row_window_num, int TCBlock_num, int total_nnz,
    const int* val_sidx,
    const int* val_soff,
    const int* val_csr_idx,
    const float* val_csr_val,
    const int* val_csr_off,
    const float* B_val,
    float* C,
    int warmup_time, int execute_time)
{
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 8;
    constexpr int Mma_M = 16;
    constexpr int Mma_N =  8;
    constexpr int Mma_K =  8;

    constexpr auto blk_mnk = make_shape(Int<TILE_M>{}, Int<TILE_N>{}, Int<TILE_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(Int<Mma_M>{}, Int<Mma_N>{}, Int<Mma_K>{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk =
        make_shape(Int<TILE_M / Mma_M>{}, Int<TILE_N / Mma_N>{}, Int<TILE_K / Mma_K>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    // --- prepare CSR_TCF metadata ---
    std::vector<int> val_len_offset(row_window_num + 1, 0);
    for (int i = 0; i <= row_window_num; ++i)
        val_len_offset[i] = val_soff[i];

    std::vector<int> val_sidx_new(TC_BLOCK_K * TCBlock_num, 0);
    for (int b = 0; b < TCBlock_num; ++b)
        for (int k = 0; k < TC_BLOCK_K; ++k)
            val_sidx_new[b * TC_BLOCK_K + k] = val_sidx[b * TC_BLOCK_K + k];

    // CSR: build contiguous payload arrays from [start, end) ranges
    std::vector<int> val_csr_len_new_offset(TCBlock_num + 1, 0);
    std::vector<int> val_csr_row_ptr(TCBlock_num * (TC_BLOCK_M + 1), 0);
    std::vector<int> val_csr_idx_new_1;
    std::vector<float> val_csr_val_new_1;
    val_csr_idx_new_1.reserve(total_nnz);
    val_csr_val_new_1.reserve(total_nnz);

    int S = (TC_BLOCK_K == 8) ? 3 : 4;

    // Deswizzle: inverse of Swizzle<1,2,S>
    auto deswizzle = [S](int p) -> int {
        int b1 = ((p >> 1) & 1) ^ ((p >> (1 + S)) & 1);
        int b2 = ((p >> 2) & 1) ^ ((p >> (2 + S)) & 1);
        return (p & ~6) | (b1 << 1) | (b2 << 2);
    };

    for (int b = 0; b < TCBlock_num; ++b) {
        val_csr_len_new_offset[b] = (int)val_csr_idx_new_1.size();
        int block_start = val_csr_off[b * 2];
        int block_end   = val_csr_off[b * 2 + 1];
        // Group by row within tile
        int row_counts[TC_BLOCK_M] = {0};
        std::vector<int> row_idx[TC_BLOCK_M];
        std::vector<float> row_val[TC_BLOCK_M];
        for (int p = block_start; p < block_end; ++p) {
            int sw = val_csr_idx[p];
            int logical = deswizzle(sw);
            int row = logical / TC_BLOCK_K;
            if (row >= 0 && row < TC_BLOCK_M) {
                row_idx[row].push_back(sw);
                row_val[row].push_back(val_csr_val[p]);
                row_counts[row]++;
            }
        }
        int* rp = &val_csr_row_ptr[b * (TC_BLOCK_M + 1)];
        rp[0] = 0;
        for (int r = 0; r < TC_BLOCK_M; r++) {
            rp[r + 1] = rp[r] + row_counts[r];
            for (size_t j = 0; j < row_idx[r].size(); j++) {
                val_csr_idx_new_1.push_back(row_idx[r][j]);
                val_csr_val_new_1.push_back(row_val[r][j]);
            }
        }
    }
    val_csr_len_new_offset[TCBlock_num] = (int)val_csr_idx_new_1.size();

    int nnz_dim_K   = TCBlock_num;
    int nnz_dim_K_o = TCBlock_num + 1;
    int nnz_dim_nnz = (int)val_csr_idx_new_1.size();
    int Mo = row_window_num + 1;

    // --- device allocations ---
    float *dB_val=nullptr, *dC_val=nullptr;
    int *dval_csr_idx_new_1=nullptr, *dval_csr_len_new_offset=nullptr;
    int *dval_csr_row_ptr_new=nullptr;
    float *dval_csr_val_new_1=nullptr;
    int *dval_len_offset=nullptr, *dval_sidx_new_d=nullptr;

    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(float) * K * N));
    CHECK_CUDA2(cudaMalloc(&dC_val, sizeof(float) * M * N));
    CHECK_CUDA2(cudaMalloc(&dval_csr_idx_new_1, sizeof(int) * nnz_dim_nnz));
    CHECK_CUDA2(cudaMalloc(&dval_csr_len_new_offset, sizeof(int) * nnz_dim_K_o));
    CHECK_CUDA2(cudaMalloc(&dval_csr_row_ptr_new, sizeof(int) * TCBlock_num * (TC_BLOCK_M + 1)));
    CHECK_CUDA2(cudaMalloc(&dval_csr_val_new_1, sizeof(float) * nnz_dim_nnz));
    CHECK_CUDA2(cudaMalloc(&dval_len_offset, sizeof(int) * Mo));
    CHECK_CUDA2(cudaMalloc(&dval_sidx_new_d, sizeof(int) * TC_BLOCK_K * nnz_dim_K));

    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC_val, 0, sizeof(float) * M * N));
    CHECK_CUDA2(cudaMemcpy(dval_csr_idx_new_1, val_csr_idx_new_1.data(), sizeof(int) * nnz_dim_nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_csr_len_new_offset, val_csr_len_new_offset.data(), sizeof(int) * nnz_dim_K_o, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_csr_row_ptr_new, val_csr_row_ptr.data(), sizeof(int) * TCBlock_num * (TC_BLOCK_M + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_csr_val_new_1, val_csr_val_new_1.data(), sizeof(float) * nnz_dim_nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_len_offset, val_len_offset.data(), sizeof(int) * Mo, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_sidx_new_d, val_sidx_new.data(), sizeof(int) * TC_BLOCK_K * nnz_dim_K, cudaMemcpyHostToDevice));

    dim3 grid(row_window_num, N / TILE_N);
    dim3 block(32);

    auto launch = [&]() {
        generated_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK><<<grid, block>>>(
            dB_val, dC_val,
            dval_csr_idx_new_1, dval_csr_len_new_offset, dval_csr_row_ptr_new,
            dval_csr_val_new_1,
            dval_len_offset, dval_sidx_new_d,
            K, M, Mo, N,
            nnz_dim_K, nnz_dim_K_o, nnz_dim_nnz);
    };

    if (warmup_time > 0) {
        for (int i = 0; i < warmup_time; ++i) launch();
        CHECK_CUDA2(cudaDeviceSynchronize());
    }
    if (execute_time > 0) {
        cudaEvent_t s, e;
        CHECK_CUDA2(cudaEventCreate(&s));
        CHECK_CUDA2(cudaEventCreate(&e));
        CHECK_CUDA2(cudaEventRecord(s));
        for (int i = 0; i < execute_time; ++i) launch();
        CHECK_CUDA2(cudaEventRecord(e));
        CHECK_CUDA2(cudaEventSynchronize(e));
        float ms = 0;
        CHECK_CUDA2(cudaEventElapsedTime(&ms, s, e));
        printf("mykernel_time: %f ms\n", (double)ms / execute_time);
        CHECK_CUDA2(cudaEventDestroy(s));
        CHECK_CUDA2(cudaEventDestroy(e));
        CHECK_CUDA2(cudaMemset(dC_val, 0, sizeof(float) * M * N));
    }

    launch();
    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(cudaMemcpy(C, dC_val, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    CHECK_CUDA2(cudaFree(dB_val));
    CHECK_CUDA2(cudaFree(dC_val));
    CHECK_CUDA2(cudaFree(dval_csr_idx_new_1));
    CHECK_CUDA2(cudaFree(dval_csr_len_new_offset));
    CHECK_CUDA2(cudaFree(dval_csr_row_ptr_new));
    CHECK_CUDA2(cudaFree(dval_csr_val_new_1));
    CHECK_CUDA2(cudaFree(dval_len_offset));
    CHECK_CUDA2(cudaFree(dval_sidx_new_d));
}

}  // namespace


template <int TILE_B>
void csr_tcf_spmm_test(int M, int N, int K,
                       int TCBlock_num, int nnz,
                       int* val_sidx, int* val_soff,
                       float* B_val,
                       int* val_csr_idx, float* val_csr_val, int* val_csr_off,
                       float* C,
                       bool /*is_ncu_test*/, int warmup_time, int execute_time)
{
    int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;
    run_csr_tcf_kernel<TILE_B>(
        M, N, K, row_window_num, TCBlock_num, nnz,
        val_sidx, val_soff, val_csr_idx, val_csr_val, val_csr_off,
        B_val, C, warmup_time, execute_time);
}

template void csr_tcf_spmm_test<64>(int, int, int, int, int,
                                     int*, int*, float*, int*, float*, int*, float*, bool, int, int);
template void csr_tcf_spmm_test<32>(int, int, int, int, int,
                                     int*, int*, float*, int*, float*, int*, float*, bool, int, int);
template void csr_tcf_spmm_test<16>(int, int, int, int, int,
                                     int*, int*, float*, int*, float*, int*, float*, bool, int, int);
