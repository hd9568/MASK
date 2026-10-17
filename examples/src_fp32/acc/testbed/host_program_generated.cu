#include <cassert>
#include <cstdint>
#include <type_traits>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime_api.h>

#include <cute/tensor.hpp>

#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "../../../../python/test/results/acc.inc"

using namespace cute;

#define DIVUP(a, b) (((a) - 1) / (b) + 1)
#define ALIGN(a, b) (DIVUP(a, b) * (b))

template <class dtypeA,
          class dtypeB,
          class dtypeC,
          class dtypeMask,
          int Tile_M,
          int Tile_N,
          int Tile_K,
          int Mma_M,
          int Mma_N,
          int Mma_K>
void acc_spmm_test(int M,
                   int N,
                   int K,
                   dtypeB* B_val,
                   dtypeC* C_val,
                   int* val_sidx,
                   int* val_soff,
                   int* val_mco_off,
                   dtypeMask* val_mco_mask,
                   dtypeA* val_mco_val,
                   bool is_ncu_test,
                   int warmup_time,
                   int execute_time) {
    static_assert(std::is_same_v<dtypeA, float>);
    static_assert(std::is_same_v<dtypeB, float>);
    static_assert(std::is_same_v<dtypeC, float>);
    static_assert(std::is_same_v<dtypeMask, uint64_t>);

    assert(N % Tile_N == 0);

    const int row_window_num = DIVUP(M, Tile_M);
    const int row_window_offsets_len = row_window_num + 1;
    const int nnz_block = val_soff[row_window_num];
    const int nnz_block_offsets_len = nnz_block + 1;
    const int nnz_aligned = ALIGN(val_mco_off[nnz_block], 8);
    const int nmask_per_tile = Tile_M * Tile_K / 64;

    float* dB_val = nullptr;
    float* dC_val = nullptr;
    int* dval_len_offset = nullptr;
    int* dval_mco_len_new_offset = nullptr;
    int64_t* dval_mco_mask_new = nullptr;
    float* dval_mco_val_new_1 = nullptr;
    int* dval_sidx_new = nullptr;

    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(float) * K * N));
    CHECK_CUDA2(cudaMalloc(&dC_val, sizeof(float) * row_window_num * Tile_M * N));
    CHECK_CUDA2(cudaMalloc(&dval_len_offset, sizeof(int) * row_window_offsets_len));
    CHECK_CUDA2(cudaMalloc(&dval_mco_len_new_offset, sizeof(int) * nnz_block_offsets_len));
    CHECK_CUDA2(cudaMalloc(&dval_mco_mask_new, sizeof(int64_t) * nmask_per_tile * nnz_block));
    CHECK_CUDA2(cudaMalloc(&dval_mco_val_new_1, sizeof(float) * nnz_aligned));
    CHECK_CUDA2(cudaMalloc(&dval_sidx_new, sizeof(int) * Tile_K * nnz_block));

    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC_val, 0, sizeof(float) * row_window_num * Tile_M * N));
    CHECK_CUDA2(cudaMemcpy(
        dval_len_offset, val_soff, sizeof(int) * row_window_offsets_len, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_mco_len_new_offset,
                           val_mco_off,
                           sizeof(int) * nnz_block_offsets_len,
                           cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_mco_mask_new,
                           val_mco_mask,
                           sizeof(int64_t) * nmask_per_tile * nnz_block,
                           cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_mco_val_new_1,
                           val_mco_val,
                           sizeof(float) * nnz_aligned,
                           cudaMemcpyHostToDevice));
    CHECK_CUDA2(
        cudaMemcpy(dval_sidx_new, val_sidx, sizeof(int) * Tile_K * nnz_block, cudaMemcpyHostToDevice));

    dim3 grid(N / Tile_N, row_window_num);
    dim3 block(32);

    constexpr auto blk_mnk = make_shape(Int<Tile_M>{}, Int<Tile_N>{}, Int<Tile_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(Int<Mma_M>{}, Int<Mma_N>{}, Int<Mma_K>{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk =
        make_shape(Int<Tile_M / Mma_M>{}, Int<Tile_N / Mma_N>{}, Int<Tile_K / Mma_K>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    auto launch = [&]() {
        generated_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK><<<grid, block>>>(
            dB_val,
            dC_val,
            dval_len_offset,
            dval_mco_len_new_offset,
            dval_mco_mask_new,
            dval_mco_val_new_1,
            dval_sidx_new,
            K,
            M,
            row_window_offsets_len,
            N,
            nnz_block,
            nnz_block_offsets_len,
            nnz_aligned);
    };

    if (!is_ncu_test && execute_time > 0) {
        cudaEvent_t start_event = nullptr;
        cudaEvent_t stop_event = nullptr;
        CHECK_CUDA2(cudaEventCreate(&start_event));
        CHECK_CUDA2(cudaEventCreate(&stop_event));
        for (int i = 0; i < warmup_time; ++i) {
            launch();
        }
        CHECK_CUDA2(cudaDeviceSynchronize());
        CHECK_CUDA2(cudaEventRecord(start_event));
        for (int i = 0; i < execute_time; ++i) {
            launch();
        }
        CHECK_CUDA2(cudaEventRecord(stop_event));
        CHECK_CUDA2(cudaEventSynchronize(stop_event));
        float elapsed_ms = 0.0f;
        CHECK_CUDA2(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        printf("mykernel_time: %f ms\n", static_cast<double>(elapsed_ms) / execute_time);
        CHECK_CUDA2(cudaEventDestroy(start_event));
        CHECK_CUDA2(cudaEventDestroy(stop_event));
        CHECK_CUDA2(cudaMemset(dC_val, 0, sizeof(float) * row_window_num * Tile_M * N));
    }

    launch();
    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(cudaMemcpy(C_val, dC_val, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    CHECK_CUDA2(cudaFree(dB_val));
    CHECK_CUDA2(cudaFree(dC_val));
    CHECK_CUDA2(cudaFree(dval_len_offset));
    CHECK_CUDA2(cudaFree(dval_mco_len_new_offset));
    CHECK_CUDA2(cudaFree(dval_mco_mask_new));
    CHECK_CUDA2(cudaFree(dval_mco_val_new_1));
    CHECK_CUDA2(cudaFree(dval_sidx_new));
}

template void
acc_spmm_test<float, float, float, uint64_t, 16, 64, 8, 16, 8, 8>(int M,
                                                                   int N,
                                                                   int K,
                                                                   float* B_val,
                                                                   float* C_val,
                                                                   int* val_sidx,
                                                                   int* val_soff,
                                                                   int* val_mco_off,
                                                                   uint64_t* val_mco_mask,
                                                                   float* val_mco_val,
                                                                   bool is_ncu_test,
                                                                   int warmup_time,
                                                                   int execute_time);
