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

#include "kernel_lb.inc"

#define DTYPEAB float

using namespace cute;
using namespace std;

template <int TILE_M, int TILE_N, int TILE_K>
void ncu_test_strict_lb(int threadblock_num_x,
                        int threadblock_num_y,
                        int thread_num,
                        int M,
                        int N,
                        int K,
                        int Mo,
                        int nnz_block,
                        int row_block_num,
                        int* dval_sidx,
                        int* dval_soff,
                        DTYPEAB* dB_val,
                        DTYPEAB* dval_block_val,
                        int* drow_block_offset,
                        int* drow_block_to_row_window_id,
                        int* dslb_tb_offset,
                        float* dC) {
    dim3 grid(threadblock_num_x, threadblock_num_y);
    dim3 block(thread_num);

    constexpr auto blk_mnk = make_shape(Int<TILE_M>{}, Int<TILE_N>{}, Int<TILE_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(_16{}, _8{}, _8{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk =
      make_shape(Int<TILE_M / 16>{}, Int<TILE_N / 8>{}, Int<TILE_K / 8>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    sr_bcrs_spmm_kernel_tf32_strict_lb<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>
      <<<grid, block>>>(dB_val,
                        dC,
                        dval_block_val,
                        dval_sidx,
                        dval_soff,
                        drow_block_offset,
                        drow_block_to_row_window_id,
                        dslb_tb_offset,
                        row_block_num,
                        K,
                        M,
                        Mo,
                        N,
                        nnz_block);
}

template <int TILE_M, int TILE_N, int TILE_K>
void repeat_test_strict_lb(int threadblock_num_x,
                           int threadblock_num_y,
                           int thread_num,
                           int M,
                           int N,
                           int K,
                           int Mo,
                           int nnz_block,
                           int row_block_num,
                           int* dval_sidx,
                           int* dval_soff,
                           DTYPEAB* dB_val,
                           DTYPEAB* dval_block_val,
                           int* drow_block_offset,
                           int* drow_block_to_row_window_id,
                           int* dslb_tb_offset,
                           float* dC,
                           int warmup_time,
                           int execute_time) {
    dim3 grid(threadblock_num_x, threadblock_num_y);
    dim3 block(thread_num);

    constexpr auto blk_mnk = make_shape(Int<TILE_M>{}, Int<TILE_N>{}, Int<TILE_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(_16{}, _8{}, _8{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk =
      make_shape(Int<TILE_M / 16>{}, Int<TILE_N / 8>{}, Int<TILE_K / 8>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    cudaEvent_t start, stop;
    CHECK_CUDA2(cudaEventCreate(&start));
    CHECK_CUDA2(cudaEventCreate(&stop));

    for (int i = 0; i < warmup_time; ++i) {
        sr_bcrs_spmm_kernel_tf32_strict_lb<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>
          <<<grid, block>>>(dB_val,
                            dC,
                            dval_block_val,
                            dval_sidx,
                            dval_soff,
                            drow_block_offset,
                            drow_block_to_row_window_id,
                            dslb_tb_offset,
                            row_block_num,
                            K,
                            M,
                            Mo,
                            N,
                            nnz_block);
    }
    CHECK_CUDA2(cudaDeviceSynchronize());

    CHECK_CUDA2(cudaEventRecord(start));
    for (int i = 0; i < execute_time; ++i) {
        sr_bcrs_spmm_kernel_tf32_strict_lb<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>
          <<<grid, block>>>(dB_val,
                            dC,
                            dval_block_val,
                            dval_sidx,
                            dval_soff,
                            drow_block_offset,
                            drow_block_to_row_window_id,
                            dslb_tb_offset,
                            row_block_num,
                            K,
                            M,
                            Mo,
                            N,
                            nnz_block);
    }
    CHECK_CUDA2(cudaEventRecord(stop));
    CHECK_CUDA2(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA2(cudaEventElapsedTime(&milliseconds, start, stop));
    double mykernel_time = milliseconds / execute_time;
    printf("mykernel_time: %f ms\n", mykernel_time);

    CHECK_CUDA2(cudaEventDestroy(start));
    CHECK_CUDA2(cudaEventDestroy(stop));
}

template <int TILE_M, int TILE_N, int TILE_K>
void sr_bcrs_test_tf32_strict_lb(int M,
                                 int N,
                                 int K,
                                 int TCBlock_num,
                                 int* val_sidx,
                                 int* val_soff,
                                 DTYPEAB* B_val,
                                 DTYPEAB* val_block_val,
                                 const vector<int>& row_block_offset,
                                 const vector<int>& row_block_to_row_window_id,
                                 const vector<int>& slb_tb_offset,
                                 float* C,
                                 bool is_ncu_test,
                                 int warmup_time,
                                 int execute_time) {
    int* dval_sidx;
    int* dval_soff;
    int* drow_block_offset;
    int* drow_block_to_row_window_id;
    int* dslb_tb_offset;
    DTYPEAB* dB_val;
    DTYPEAB* dval_block_val;
    float* dC;

    const int row_window_num = DIVUP(M, TILE_M);
    const int row_block_num = static_cast<int>(row_block_to_row_window_id.size());
    const int slb_block_dim_size = static_cast<int>(slb_tb_offset.size()) - 1;

    assert(N % TILE_N == 0);
    assert(static_cast<int>(row_block_offset.size()) == row_block_num + 1);
    assert(slb_block_dim_size > 0);

    CHECK_CUDA2(cudaMalloc(&dval_sidx, sizeof(int) * TCBlock_num * TILE_K));
    CHECK_CUDA2(cudaMalloc(&dval_soff, sizeof(int) * (row_window_num + 1)));
    CHECK_CUDA2(cudaMalloc(&drow_block_offset, sizeof(int) * row_block_offset.size()));
    CHECK_CUDA2(
      cudaMalloc(&drow_block_to_row_window_id, sizeof(int) * row_block_to_row_window_id.size()));
    CHECK_CUDA2(cudaMalloc(&dslb_tb_offset, sizeof(int) * slb_tb_offset.size()));
    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(DTYPEAB) * K * N));
    CHECK_CUDA2(cudaMalloc(&dval_block_val, sizeof(DTYPEAB) * TCBlock_num * TILE_M * TILE_K));
    CHECK_CUDA2(cudaMalloc(&dC, sizeof(float) * row_window_num * TILE_M * N));

    CHECK_CUDA2(
      cudaMemcpy(dval_sidx, val_sidx, sizeof(int) * TCBlock_num * TILE_K, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(
      dval_soff, val_soff, sizeof(int) * (row_window_num + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(drow_block_offset,
                           row_block_offset.data(),
                           sizeof(int) * row_block_offset.size(),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(drow_block_to_row_window_id,
                           row_block_to_row_window_id.data(),
                           sizeof(int) * row_block_to_row_window_id.size(),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dslb_tb_offset,
                           slb_tb_offset.data(),
                           sizeof(int) * slb_tb_offset.size(),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(DTYPEAB) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_block_val,
                           val_block_val,
                           sizeof(DTYPEAB) * TILE_M * TILE_K * TCBlock_num,
                           cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * row_window_num * TILE_M * N));

    int threadblock_num_x = slb_block_dim_size;
    int threadblock_num_y = N / TILE_N;
    int thread_num = 32;

    const int Mo = DIVUP(M, TILE_M);
    const int nnz_block = TCBlock_num;

    if (!is_ncu_test) {
        repeat_test_strict_lb<TILE_M, TILE_N, TILE_K>(threadblock_num_x,
                                                      threadblock_num_y,
                                                      thread_num,
                                                      M,
                                                      N,
                                                      K,
                                                      Mo,
                                                      nnz_block,
                                                      row_block_num,
                                                      dval_sidx,
                                                      dval_soff,
                                                      dB_val,
                                                      dval_block_val,
                                                      drow_block_offset,
                                                      drow_block_to_row_window_id,
                                                      dslb_tb_offset,
                                                      dC,
                                                      warmup_time,
                                                      execute_time);
    }

    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * row_window_num * TILE_M * N));
    ncu_test_strict_lb<TILE_M, TILE_N, TILE_K>(threadblock_num_x,
                                               threadblock_num_y,
                                               thread_num,
                                               M,
                                               N,
                                               K,
                                               Mo,
                                               nnz_block,
                                               row_block_num,
                                               dval_sidx,
                                               dval_soff,
                                               dB_val,
                                               dval_block_val,
                                               drow_block_offset,
                                               drow_block_to_row_window_id,
                                               dslb_tb_offset,
                                               dC);

    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    CHECK_CUDA2(cudaFree(dval_sidx));
    CHECK_CUDA2(cudaFree(dval_soff));
    CHECK_CUDA2(cudaFree(drow_block_offset));
    CHECK_CUDA2(cudaFree(drow_block_to_row_window_id));
    CHECK_CUDA2(cudaFree(dslb_tb_offset));
    CHECK_CUDA2(cudaFree(dB_val));
    CHECK_CUDA2(cudaFree(dval_block_val));
    CHECK_CUDA2(cudaFree(dC));
}

template void sr_bcrs_test_tf32_strict_lb<32, 64, 32>(int M,
                                                       int N,
                                                       int K,
                                                       int TCBlock_num,
                                                       int* val_sidx,
                                                       int* val_soff,
                                                       DTYPEAB* B_val,
                                                       DTYPEAB* val_block_val,
                                                       const vector<int>& row_block_offset,
                                                       const vector<int>& row_block_to_row_window_id,
                                                       const vector<int>& slb_tb_offset,
                                                       float* C,
                                                       bool is_ncu_test,
                                                       int warmup_time,
                                                       int execute_time);

template void sr_bcrs_test_tf32_strict_lb<16, 64, 8>(int M,
                                                      int N,
                                                      int K,
                                                      int TCBlock_num,
                                                      int* val_sidx,
                                                      int* val_soff,
                                                      DTYPEAB* B_val,
                                                      DTYPEAB* val_block_val,
                                                      const vector<int>& row_block_offset,
                                                      const vector<int>& row_block_to_row_window_id,
                                                      const vector<int>& slb_tb_offset,
                                                      float* C,
                                                      bool is_ncu_test,
                                                      int warmup_time,
                                                      int execute_time);
