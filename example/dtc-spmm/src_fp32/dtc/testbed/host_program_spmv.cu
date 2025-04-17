#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "kernel_spmv.inc"

using namespace cute;
using namespace std;

#define ROW_WINDOW_M 16
#define TC_BLOCK_K 8
#define TC_BLOCK_M 16

//! mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
template <int TILE_B>
void ncu_test(int threadblock_num_x,
              int threadblock_num_y,
              int thread_num,
              int M,
              int N,
              int K,
              int Mo,
              int nnz_block,
              int nnz,
              int* dval_sidx,
              int* dval_soff,
              float* dB_val,
              float* dval_coo_val,
              int* dval_coo_idx,
              int* dval_coo_off,
              float* dC) {
    dim3 grid(threadblock_num_x, threadblock_num_y);
    dim3 block(thread_num);

    constexpr auto blk_mnk = make_shape(Int<TC_BLOCK_M>{}, Int<TILE_B>{}, Int<TC_BLOCK_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(_16{}, _8{}, _8{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk =
      make_shape(Int<TC_BLOCK_M / 16>{}, Int<TILE_B / 8>{}, Int<TC_BLOCK_K / 8>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    using tile_sidx_size = Int<TC_BLOCK_K>;
    using tile_coo_val_size = Int<TC_BLOCK_M * TC_BLOCK_K>;
    using tile_coo_val_size_no_pad = Int<TC_BLOCK_M * TC_BLOCK_K>;

    dtc_spmm_kernel_fp32_val_idx_bind_stage2<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK,
                                             tile_coo_val_size, tile_coo_val_size_no_pad,
                                             tile_sidx_size>
      <<<grid, block>>>(dB_val, dC, dval_coo_idx, dval_coo_off, dval_coo_val, dval_sidx, dval_soff,
                        K, M, Mo, N, nnz, nnz_block);
}

template <int TILE_B>
void repeat_test(int threadblock_num_x,
                 int threadblock_num_y,
                 int thread_num,
                 int M,
                 int N,
                 int K,
                 int Mo,
                 int nnz_block,
                 int nnz,
                 int* dval_sidx,
                 int* dval_soff,
                 float* dB_val,
                 float* dval_coo_val,
                 int* dval_coo_idx,
                 int* dval_coo_off,
                 float* dC,
                 int warmup_time,
                 int execute_time) {
    dim3 grid(threadblock_num_x, threadblock_num_y);
    dim3 block(thread_num);

    constexpr auto blk_mnk = make_shape(Int<TC_BLOCK_M>{}, Int<TILE_B>{}, Int<TC_BLOCK_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(_16{}, _8{}, _8{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk =
      make_shape(Int<TC_BLOCK_M / 16>{}, Int<TILE_B / 8>{}, Int<TC_BLOCK_K / 8>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    using tile_sidx_size = Int<TC_BLOCK_K>;
    using tile_coo_val_size = Int<TC_BLOCK_M * TC_BLOCK_K>;
    using tile_coo_val_size_no_pad = Int<TC_BLOCK_M * TC_BLOCK_K>;

    struct timeval t1, t2;
    {
        for (int i = 0; i < warmup_time; i++) {
            dtc_spmm_kernel_fp32_val_idx_bind_stage2<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK,
                                                     tile_coo_val_size, tile_coo_val_size_no_pad,
                                                     tile_sidx_size>
              <<<grid, block>>>(dB_val, dC, dval_coo_idx, dval_coo_off, dval_coo_val, dval_sidx,
                                dval_soff, K, M, Mo, N, nnz, nnz_block);
        }
        CHECK_CUDA2(cudaDeviceSynchronize());
        gettimeofday(&t1, NULL);
        for (int i = 0; i < execute_time; i++) {
            dtc_spmm_kernel_fp32_val_idx_bind_stage2<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK,
                                                     tile_coo_val_size, tile_coo_val_size_no_pad,
                                                     tile_sidx_size>
              <<<grid, block>>>(dB_val, dC, dval_coo_idx, dval_coo_off, dval_coo_val, dval_sidx,
                                dval_soff, K, M, Mo, N, nnz, nnz_block);
        }
        CHECK_CUDA2(cudaDeviceSynchronize());
        gettimeofday(&t2, NULL);
        double mykernel_time =
          ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / execute_time;
        printf("mykernel_time: %f ms\n", mykernel_time);
    }
}

//! mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
template <int TILE_B>
void dtc_spmm_test_val_sidx_bind_reorder(int M,
                                         int N,
                                         int K,
                                         int TCBlock_num,
                                         int nnz,
                                         int* val_sidx,
                                         int* val_soff,
                                         float* B_val,
                                         int* val_coo_idx,
                                         float* val_coo_val,
                                         int* val_coo_off,
                                         float* C,
                                         bool is_ncu_test,
                                         int warmup_time,
                                         int execute_time) {
    int* dval_sidx;
    int* dval_soff;
    float* dB_val;
    int* dval_coo_idx;
    float* dval_coo_val;
    int* dval_coo_off;
    float* dC;

    int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;
    // int pad_K = (K + 8 - 1) / 8 * 8;
    assert(N % TILE_B == 0);

    CHECK_CUDA2(cudaMalloc(&dval_sidx, sizeof(int) * TCBlock_num * TC_BLOCK_K));

    CHECK_CUDA2(cudaMalloc(&dval_soff, sizeof(int) * (row_window_num + 1)));
    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(float) * K * N));
    //! coo
    CHECK_CUDA2(cudaMalloc(&dval_coo_idx, (nnz + 8 - 1) / 8 * 8 * sizeof(int)));
    CHECK_CUDA2(cudaMemset(dval_coo_idx, 0, (nnz + 8 - 1) / 8 * 8 * sizeof(int)));
    CHECK_CUDA2(cudaMalloc(&dval_coo_val, (nnz + 8 - 1) / 8 * 8 * sizeof(float)));
    CHECK_CUDA2(cudaMemset(dval_coo_val, 0, (nnz + 8 - 1) / 8 * 8 * sizeof(float)));
    CHECK_CUDA2(cudaMalloc(&dval_coo_off, sizeof(int) * (2 * TCBlock_num + 1)));
    CHECK_CUDA2(cudaMalloc(&dC, sizeof(float) * (row_window_num * ROW_WINDOW_M) *
                                  N)); // C is row major (alloc more copy some)
    CHECK_CUDA2(cudaMemcpy(dval_sidx, val_sidx, sizeof(int) * TCBlock_num * TC_BLOCK_K,
                           cudaMemcpyHostToDevice));
    CHECK_CUDA2(
      cudaMemcpy(dval_soff, val_soff, sizeof(int) * (row_window_num + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    //! coo
    CHECK_CUDA2(cudaMemcpy(dval_coo_idx, val_coo_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_coo_val, val_coo_val, sizeof(float) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_coo_off, val_coo_off, sizeof(int) * (2 * TCBlock_num + 1),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * (row_window_num * ROW_WINDOW_M) * N));

    int threadblock_num_x = row_window_num;
    int threadblock_num_y = N / TILE_B;
    int thread_num = 32;

    const int Mo = DIVUP(M, ROW_WINDOW_M);
    const int nnz_block = val_soff[Mo];
    const int nnz_aligned = ALIGN(val_coo_off[nnz_block * 2 - 1], 8);

    if (!is_ncu_test) {
        repeat_test<TILE_B>(threadblock_num_x, threadblock_num_y, thread_num, M, N, K, Mo,
                            nnz_block, nnz_aligned, dval_sidx, dval_soff, dB_val, dval_coo_val,
                            dval_coo_idx, dval_coo_off, dC, warmup_time, execute_time);
    }

    // repeat test would cause C_val to accumulate results, so we need to reset it
    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * (row_window_num * ROW_WINDOW_M) * N));
    ncu_test<TILE_B>(threadblock_num_x, threadblock_num_y, thread_num, M, N, K, Mo, nnz_block,
                     nnz_aligned, dval_sidx, dval_soff, dB_val, dval_coo_val, dval_coo_idx,
                     dval_coo_off, dC);

    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
}

template void dtc_spmm_test_val_sidx_bind_reorder<8>(int M,
                                                      int N,
                                                      int K,
                                                      int TCBlock_num,
                                                      int nnz,
                                                      int* val_sidx,
                                                      int* val_soff,
                                                      float* B_val,
                                                      int* val_coo_idx,
                                                      float* val_coo_val,
                                                      int* val_coo_off,
                                                      float* C,
                                                      bool is_ncu_test,
                                                      int warmup_time,
                                                      int execute_time);