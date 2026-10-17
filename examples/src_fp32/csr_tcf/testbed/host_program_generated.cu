#include <cassert>
#include <type_traits>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime_api.h>

#include <cute/tensor.hpp>

#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "../../../../python/test/results/sr_bcrs.inc"

using namespace cute;

#define DIVUP(a, b) (((a) - 1) / (b) + 1)

template <int TILE_M, int TILE_N, int TILE_K>
void sr_bcrs_test_tf32(int M,
                       int N,
                       int K,
                       int TCBlock_num,
                       int* val_sidx,
                       int* val_soff,
                       float* B_val,
                       float* val_block_val,
                       float* C,
                       bool is_ncu_test,
                       int warmup_time,
                       int execute_time) {
    assert(N % TILE_N == 0);

    const int row_window_num = DIVUP(M, TILE_M);
    const int row_window_offsets_len = row_window_num + 1;

    float* dB_val = nullptr;
    float* dC_val = nullptr;
    int* dval_len_offset = nullptr;
    float* dval_new = nullptr;
    int* dval_sidx_new = nullptr;

    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(float) * K * N));
    CHECK_CUDA2(cudaMalloc(&dC_val, sizeof(float) * row_window_num * TILE_M * N));
    CHECK_CUDA2(cudaMalloc(&dval_len_offset, sizeof(int) * row_window_offsets_len));
    CHECK_CUDA2(cudaMalloc(&dval_new, sizeof(float) * TCBlock_num * TILE_K * TILE_M));
    CHECK_CUDA2(cudaMalloc(&dval_sidx_new, sizeof(int) * TCBlock_num * TILE_K));

    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC_val, 0, sizeof(float) * row_window_num * TILE_M * N));
    CHECK_CUDA2(cudaMemcpy(
        dval_len_offset, val_soff, sizeof(int) * row_window_offsets_len, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(
        dval_new, val_block_val, sizeof(float) * TCBlock_num * TILE_K * TILE_M, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(
        dval_sidx_new, val_sidx, sizeof(int) * TCBlock_num * TILE_K, cudaMemcpyHostToDevice));

    dim3 grid(row_window_num, N / TILE_N);
    dim3 block(32);

    constexpr auto blk_mnk = make_shape(Int<TILE_M>{}, Int<TILE_N>{}, Int<TILE_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(_16{}, _8{}, _8{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk =
        make_shape(Int<TILE_M / 16>{}, Int<TILE_N / 8>{}, Int<TILE_K / 8>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    auto launch = [&]() {
        generated_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK><<<grid, block>>>(
            dB_val,
            dC_val,
            dval_len_offset,
            dval_new,
            dval_sidx_new,
            K,
            M,
            row_window_offsets_len,
            N,
            TCBlock_num);
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
        CHECK_CUDA2(cudaMemset(dC_val, 0, sizeof(float) * row_window_num * TILE_M * N));
    }

    launch();
    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(cudaMemcpy(C, dC_val, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    CHECK_CUDA2(cudaFree(dB_val));
    CHECK_CUDA2(cudaFree(dC_val));
    CHECK_CUDA2(cudaFree(dval_len_offset));
    CHECK_CUDA2(cudaFree(dval_new));
    CHECK_CUDA2(cudaFree(dval_sidx_new));
}

template void sr_bcrs_test_tf32<32, 64, 32>(int M,
                                            int N,
                                            int K,
                                            int TCBlock_num,
                                            int* val_sidx,
                                            int* val_soff,
                                            float* B_val,
                                            float* val_block_val,
                                            float* C,
                                            bool is_ncu_test,
                                            int warmup_time,
                                            int execute_time);
