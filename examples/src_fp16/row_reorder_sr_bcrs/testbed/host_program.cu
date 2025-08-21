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
#include <time.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

// #include "example/dtc-spmm/src_fp16/row_majorB_vs_bind_reorder/kernel-gen.inc"
#include "kernel.inc"

#define ROW_WINDOW_M 16
#define TC_BLOCK_K 16
#define TC_BLOCK_M 16

using namespace cute;
using namespace std;

//! todo: update kernel

template <int TILE_M, int TILE_N, int TILE_K>
void ncu_test(int threadblock_num_x,
              int threadblock_num_y,
              int thread_num,
              int M,
              int N,
              int K,
              int Mo,
              int nnz_block,
              int* dval_sidx,
              int* dval_soff,
              int* drow_order,
              half* dB_val,
              half* dval_block_val,
              float* dC) {
    dim3 grid(threadblock_num_y, threadblock_num_x);
    dim3 block(thread_num);

    constexpr auto blk_mnk = make_shape(Int<TILE_M>{}, Int<TILE_N>{}, Int<TILE_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(_16{}, _8{}, _16{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk =
      make_shape(Int<TILE_M / 16>{}, Int<TILE_N / 8>{}, Int<TILE_K / 16>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    rr_sr_bcrs_spmm_kernel_fp16<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>
      <<<grid, block>>>(dB_val, dC, drow_order, dval_block_val, dval_sidx, dval_soff, K, M, Mo, N, nnz_block);
}


template <int TILE_M, int TILE_N, int TILE_K>
void rr_sr_bcrs_test_fp16(int M,
                       int N,
                       int K,
                       int TCBlock_num,
                       int row_window_num,
                       int* val_sidx,
                       int* val_soff,
                       int* row_order,
                       half* B_val,
                       half* val_block_val,
                       float* C,
                       bool is_ncu_test,
                       int warmup_time,
                       int execute_time) {
    int* dval_sidx;
    int* dval_soff;
    int* drow_order;
    half* dB_val;
    half* dval_block_val;
    float* dC;

    // int row_window_num = DIVUP(M, TILE_M);
    assert(N % TILE_N == 0);

    CHECK_CUDA2(cudaMalloc(&dval_sidx, sizeof(int) * TCBlock_num * TILE_K));
    CHECK_CUDA2(cudaMalloc(&dval_soff, sizeof(int) * (row_window_num + 1)));
    CHECK_CUDA2(cudaMalloc(&drow_order, sizeof(int) * (row_window_num * TILE_M)));
    CHECK_CUDA2(cudaMemset(drow_order, -1, sizeof(int) * row_window_num * TILE_M));
    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(half) * K * N));
    CHECK_CUDA2(cudaMalloc(&dval_block_val, sizeof(half) * TCBlock_num * TILE_M * TILE_K));
    //! change dC size to M x N
    int dC_row_max = max(row_window_num * TILE_M, M);
    CHECK_CUDA2(cudaMalloc(&dC, sizeof(float) * dC_row_max *
                                  N)); // C is row major (alloc more copy some)

    CHECK_CUDA2(
      cudaMemcpy(dval_sidx, val_sidx, sizeof(int) * TCBlock_num * TILE_K, cudaMemcpyHostToDevice));
    CHECK_CUDA2(
      cudaMemcpy(dval_soff, val_soff, sizeof(int) * (row_window_num + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(drow_order, row_order, sizeof(int) * (row_window_num * TILE_M), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(half) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_block_val, val_block_val,
                           sizeof(half) * TILE_M * TILE_K * TCBlock_num, cudaMemcpyHostToDevice));
    //! change dC size to M x N    
    // CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * (row_window_num * TILE_M) * N));
    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * dC_row_max * N));

    int threadblock_num_x = row_window_num;
    int threadblock_num_y = N / TILE_N;
    int thread_num = 32;

    //> update Mo to row_window_num
    // const int Mo = DIVUP(M, TILE_M);
    const int Mo = row_window_num;
    const int nnz_block = val_soff[Mo];

    // if (!is_ncu_test) {
    //     repeat_test<TILE_M, TILE_N, TILE_K>(threadblock_num_x, threadblock_num_y, thread_num, M, N,
    //                                         K, Mo, nnz_block, dval_sidx, dval_soff, drow_order, dB_val,
    //                                         dval_block_val, dC, warmup_time, execute_time);
    // }

    // repeat test would cause C_val to accumulate results, so we need to reset it
    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * (row_window_num * TILE_M) * N));
    ncu_test<TILE_M, TILE_N, TILE_K>(threadblock_num_x, threadblock_num_y, thread_num, M, N, K, Mo,
                                     nnz_block, dval_sidx, dval_soff, drow_order, dB_val, dval_block_val, dC);

    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
}

template void rr_sr_bcrs_test_fp16<16, 64, 16>(int M,
                                            int N,
                                            int K,
                                            int TCBlock_num,
                                            int row_window_num,
                                            int* val_sidx,
                                            int* val_soff,
                                            int* row_order,
                                            half* B_val,
                                            half* val_block_val,
                                            float* C,
                                            bool is_ncu_test,
                                            int warmup_time,
                                            int execute_time);