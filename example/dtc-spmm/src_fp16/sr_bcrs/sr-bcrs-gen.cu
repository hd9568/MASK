#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <cuda_pipeline.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

// #include "example/dtc-spmm/src_fp16/row_majorB_vs_bind_reorder/kernel-gen.inc"
#include "kernel-gen.inc"

#define ROW_WINDOW_M 16
#define TC_BLOCK_K 16
#define TC_BLOCK_M 16

using namespace cute;
using namespace std;

template <int TILE_B>
void ncu_test_all(
    int threadblock_num_x, int threadblock_num_y, int thread_num,
    int M, int N, int K,
    int Mo, int nnz_block,
    int* dval_sidx,
    int* dval_soff,
    half* dB_val,
    half* dval_block_val,
    float* dC
) {
    dim3 grid(threadblock_num_x, threadblock_num_y);
    dim3 block(thread_num);

    constexpr auto blk_mnk = make_shape(Int<TC_BLOCK_M>{}, Int<TILE_B>{}, Int<TC_BLOCK_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(_16{}, _8{}, _16{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk = make_shape(Int<TC_BLOCK_M / 16>{}, Int<TILE_B / 8>{}, Int<TC_BLOCK_K / 16>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    using tile_sidx_size = Int<TC_BLOCK_K>;

    sr_bcrs_spmm_kernel_fp16<
        BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK,
        tile_sidx_size
    ><<<grid, block>>>(dB_val, dC, dval_block_val, dval_sidx, dval_soff, K, M, Mo, N, nnz_block);
}

#define DIVUP(a, b) (((a) - 1) / (b) + 1)
#define ALIGN(a,b) (DIVUP(a,b)*(b))

template <int TILE_B>
void sr_bcrs_test_fp16(
    int M, int N, int K,
    int TCBlock_num,
    int* val_sidx,
    int* val_soff,
    half* B_val,
    half* val_block_val,
    float* C
) {
    int* dval_sidx;
    int* dval_soff;
    half* dB_val;
    half* dval_block_val;
    float* dC;

    int row_window_num = DIVUP(M, ROW_WINDOW_M);
    assert(N % TILE_B == 0);

    CHECK_CUDA2(cudaMalloc(&dval_sidx, sizeof(int) * TCBlock_num * TC_BLOCK_K));
    CHECK_CUDA2(cudaMalloc(&dval_soff, sizeof(int) * (row_window_num + 1)));
    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(half) * K * N));
    CHECK_CUDA2(cudaMalloc(&dval_block_val, sizeof(half) * TCBlock_num * ROW_WINDOW_M * TC_BLOCK_K));
    CHECK_CUDA2(cudaMalloc(&dC, sizeof(float) * (row_window_num * ROW_WINDOW_M) * N)); // C is row major (alloc more copy some)

    CHECK_CUDA2(cudaMemcpy(dval_sidx, val_sidx, sizeof(int) * TCBlock_num * TC_BLOCK_K, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_soff, val_soff, sizeof(int) * (row_window_num + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(half) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_block_val, val_block_val, sizeof(half) * ROW_WINDOW_M * TC_BLOCK_K, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * (row_window_num * ROW_WINDOW_M) * N));

    int threadblock_num_x = row_window_num;
    int threadblock_num_y = N / TILE_B;
    int thread_num = 32;

    const int Mo = DIVUP(M, ROW_WINDOW_M);
    const int nnz_block = val_soff[Mo];
    
    ncu_test_all<TILE_B>(
        threadblock_num_x, threadblock_num_y, thread_num,
        M, N, K, 
        Mo, nnz_block, 
        dval_sidx, dval_soff, dB_val, dval_block_val,
        dC
    );
    
    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
}

template 
void sr_bcrs_test_fp16<64>(
    int M, int N, int K,
    int TCBlock_num,
    int* val_sidx,
    int* val_soff,
    half* B_val,
    half* val_block_val,
    float* C
);