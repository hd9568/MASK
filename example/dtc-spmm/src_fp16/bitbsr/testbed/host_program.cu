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

#include "kernel.inc"

using namespace cute;
using namespace std;

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
void ncu_test_all(int threadblock_num_x,
                  int threadblock_num_y,
                  int thread_num,
                  int M,
                  int N,
                  int K,
                  int Mo,
                  int nnz_aligned,
                  int nnz_block,
                  dtypeB* dB_val,
                  dtypeC* dC_val,
                  int* dval_sidx,
                  int* dval_soff,
                  dtypeMask* dval_mco_mask,
                  int* dval_mco_off,
                  dtypeA* dval_mco_val) {
    dim3 grid(threadblock_num_x, threadblock_num_y);
    dim3 block(thread_num);

    constexpr auto blk_mnk = make_shape(Int<Tile_M>{}, Int<Tile_N>{}, Int<Tile_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(Int<Mma_M>{}, Int<Mma_N>{}, Int<Mma_K>{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk =
      make_shape(Int<Tile_M / Mma_M>{}, Int<Tile_N / Mma_N>{}, Int<Tile_K / Mma_K>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);

    using nmask_per_tile = Int<Tile_M * Tile_N / 64>;
    using tile_mco_val_size = Int<Tile_M * Tile_N>;

    bitbsr_spmm_kernel_fp16fp16fp32<BLK_MNK, MMA_MNK, BLK_MMA_MNK, nmask_per_tile,
                                    tile_mco_val_size>
      <<<grid, block>>>(dB_val, dC_val, dval_mco_mask, dval_mco_off, dval_mco_val, dval_sidx,
                        dval_soff, K, M, Mo, N, nnz_aligned, nnz_block);
}

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
void bitbsr_spmm_test(int M,
                      int N,
                      int K,
                      dtypeB* B_val,
                      dtypeC* C_val,
                      int* val_sidx,
                      int* val_soff,
                      int* val_mco_off,
                      dtypeMask* val_mco_mask,
                      dtypeA* val_mco_val) {

    dtypeB* dB_val;
    dtypeC* dC_val;
    int* dval_sidx;
    int* dval_soff;
    int* dval_mco_off;
    dtypeMask* dval_mco_mask;
    dtypeA* dval_mco_val;

    assert(N % Tile_N == 0);

    int Mo = (M + Tile_M - 1) / Tile_M;
    int nnz_block = val_soff[Mo];
    int nnz_aligned = ALIGN(val_mco_off[nnz_block], 8);
    constexpr int nmask_per_tile = Tile_M * Tile_N / 64;

    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(dtypeB) * K * N));
    CHECK_CUDA2(cudaMalloc(&dC_val, sizeof(dtypeC) * (Mo * Tile_M) * N)); // padded to Tile_M
    CHECK_CUDA2(cudaMalloc(&dval_soff, sizeof(int) * (Mo + 1)));
    CHECK_CUDA2(cudaMalloc(&dval_sidx, sizeof(int) * nnz_block));
    //! mco
    CHECK_CUDA2(cudaMalloc(&dval_mco_mask, sizeof(dtypeMask) * nmask_per_tile * nnz_block));
    CHECK_CUDA2(cudaMalloc(&dval_mco_off, sizeof(int) * (nnz_block + 1)));
    CHECK_CUDA2(cudaMalloc(&dval_mco_val, sizeof(dtypeA) * nnz_aligned));

    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(dtypeB) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC_val, 0, sizeof(dtypeC) * (Mo * Tile_M) * N));
    CHECK_CUDA2(cudaMemcpy(dval_soff, val_soff, sizeof(int) * (Mo + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_sidx, val_sidx, sizeof(int) * nnz_block, cudaMemcpyHostToDevice));
    //! coo
    CHECK_CUDA2(cudaMemcpy(dval_mco_mask, val_mco_mask,
                           sizeof(dtypeMask) * nmask_per_tile * nnz_block, cudaMemcpyHostToDevice));
    CHECK_CUDA2(
      cudaMemcpy(dval_mco_off, val_mco_off, sizeof(int) * (nnz_block + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(
      cudaMemcpy(dval_mco_val, val_mco_val, sizeof(dtypeA) * nnz_aligned, cudaMemcpyHostToDevice));

    int threadblock_num_x = Mo;
    int threadblock_num_y = N / Tile_N;
    int thread_num = 32;

    ncu_test_all<dtypeA, dtypeB, dtypeC, dtypeMask, Tile_M, Tile_N, Tile_K, Mma_M, Mma_N, Mma_K>(
      threadblock_num_x, threadblock_num_y, thread_num, M, N, K, Mo, nnz_aligned, nnz_block, dB_val,
      dC_val, dval_sidx, dval_soff, dval_mco_mask, dval_mco_off, dval_mco_val);

    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(
      cudaMemcpy(C_val, dC_val, sizeof(dtypeC) * (Mo * Tile_M) * N, cudaMemcpyDeviceToHost));
}

template void
bitbsr_spmm_test<half, half, float, uint64_t, 16, 64, 16, 16, 8, 16>(int M,
                                                                     int N,
                                                                     int K,
                                                                     half* B_val,
                                                                     float* C_val,
                                                                     int* val_sidx,
                                                                     int* val_soff,
                                                                     int* val_mco_off,
                                                                     uint64_t* val_mco_mask,
                                                                     half* val_mco_val);