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
using namespace std;

#define ROW_WINDOW_M 16
#define TC_BLOCK_K 16
#define TC_BLOCK_M 16

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))
#define OFFSET_COL(row, col, lda) ((col) * (lda) + (row))

#define CHECK_CUDA2(func)                                              \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess) {                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
        }                                                              \
    }

__device__ __forceinline__ void ldmatrix_m8n8k8_x4(uint32_t* fragment, void* ptr) {
    uint32_t values_tile_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 \t"
        " { %0, %1, %2, %3 }, \t"
        " [%4];"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
        : "r"(values_tile_int_ptr)
    );
}

__device__ __forceinline__ void ldmatrix_m8n8k8_x2(uint32_t* fragment, void* ptr) {
    uint32_t values_tile_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 \t"
        " { %0, %1 }, \t"
        " [%2];"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(values_tile_int_ptr)
    );
}

__device__ __forceinline__ void ldmatrix_m8n8k8_x2_trans(uint32_t* fragment, void* ptr) {
    uint32_t values_tile_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 \t"
        " { %0, %1 }, \t"
        " [%2];"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(values_tile_int_ptr)
    );
}

__device__ __forceinline__ void mma_m16n8k16_fp32(float* acc, uint32_t* A, half* half_B) {
    uint32_t* B = reinterpret_cast<uint32_t*>(half_B);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1])
    );
}

__device__ __forceinline__ void unpack_half_short(uint32_t packed, half& h, short & s) {
    uint16_t half_bits = static_cast<uint16_t>(packed >> 16);
    uint16_t short_bits = static_cast<uint16_t>(packed & 0xFFFF);
    h = *reinterpret_cast<half*>(&half_bits);
    s = static_cast<short>(short_bits);
}

template <int TILE_B>
__device__ __forceinline__ void op_memcpy_async_sparse_encoding_value_load(
    int i, int smem_st_ptr, int* dval_coo_off, uint32_t* dval_coo_val_idx_bind, int l, int r, int* coo_range_reg, int lane_id, uint32_t* tileA_coo_val_idx_bind_block, int tile_coo_val_size
) {
    int ll = dval_coo_off[2 * (l + i)];
    int rr = dval_coo_off[2 * (l + i) + 1];
    coo_range_reg[smem_st_ptr] = rr - ll;
    // if (lane_id * 8 + ll < rr) {
    //     int4* src = (int4*)(&dval_coo_val_idx_bind[ll + lane_id * 8]);
    //     int4* src2 = (int4*)(&dval_coo_val_idx_bind[ll + lane_id * 8 + 4]);
    //     int4* dst = (int4*)(&tileA_coo_val_idx_bind_block[smem_st_ptr * (16 * 16) + lane_id * 8]);
    //     int4* dst2 = (int4*)(&tileA_coo_val_idx_bind_block[smem_st_ptr * (16 * 16) + lane_id * 8 + 4]);
    //     __pipeline_memcpy_async(dst, src, sizeof(int4));
    //     __pipeline_memcpy_async(dst2, src2, sizeof(int4));
    // }
    for (int i_load = lane_id * 4; i_load + ll < rr; i_load += 32 * 4) {
        int4* src = (int4*)(&dval_coo_val_idx_bind[ll + i_load]);
        int4* dst = (int4*)(&tileA_coo_val_idx_bind_block[smem_st_ptr * (16 * 16) + i_load]);
        __pipeline_memcpy_async(dst, src, sizeof(int4));
    }
}

template <int TILE_B>
__device__ __forceinline__ void op_memcpy_async_sparse_index_load(
    int i, int smem_st_ptr, int lane_id, int* tile_sidx_block, int tile_sidx_size, int* dval_sidx, int l
) {
    if (lane_id < 4) {
        int4* dst = (int4*)(&tile_sidx_block[smem_st_ptr * tile_sidx_size + lane_id * 4]);
        int4* src = (int4*)(&dval_sidx[(l + i) * 16 + lane_id * 4]);
        __pipeline_memcpy_async(dst, src, sizeof(int4));
    }
}

template <int TILE_B>
__device__ __forceinline__ void op_restore_matrix_A(
    int i, int smem_st_ptr, int smem_ld_ptr, int lane_id, int* coo_range_reg, half* tile_coo_restore_val_block, int tile_coo_val_size, uint32_t* tileA_coo_val_idx_bind_block
) {
    int nnz_num = coo_range_reg[smem_ld_ptr];
    // 16 * 16 half = 32 * 8 half = 32 * 4 float
    for (int i_o2s = lane_id; i_o2s < tile_coo_val_size / 2; i_o2s += 32) {
        // *((float*)(&tile_coo_restore_val_block[smem_st_ptr * tile_coo_val_size]) + i_o2s) = 0;
    }
    *((float*)(&tile_coo_restore_val_block[smem_st_ptr * tile_coo_val_size]) + lane_id) = 0;
    *((float*)(&tile_coo_restore_val_block[smem_st_ptr * tile_coo_val_size]) + lane_id + 32) = 0;
    *((float*)(&tile_coo_restore_val_block[smem_st_ptr * tile_coo_val_size]) + lane_id + 32 * 2) = 0;
    *((float*)(&tile_coo_restore_val_block[smem_st_ptr * tile_coo_val_size]) + lane_id + 32 * 3) = 0;
    *((float*)(&tile_coo_restore_val_block[smem_st_ptr * tile_coo_val_size]) + lane_id + 32 * 4) = 0;
    *((float*)(&tile_coo_restore_val_block[smem_st_ptr * tile_coo_val_size]) + lane_id + 32 * 5) = 0;
    for (int i_o2s = lane_id * 4; i_o2s < tile_coo_val_size / 2 / 4; i_o2s += 32 * 4) {
        // *((float4*)(&tile_coo_restore_val_block[smem_st_ptr * tile_coo_val_size] + i_o2s)) = (float4)0;
    }
    __syncthreads();
    //! val_idx_bind
    for (int i_restore = lane_id; i_restore < nnz_num; i_restore += 32) {
        uint32_t val_idx_bind = tileA_coo_val_idx_bind_block[smem_ld_ptr * (16 * 16) + i_restore];
        half value;
        short idx;
        unpack_half_short(val_idx_bind, value, idx);
        int row = idx / 16;
        int col = idx % 16;
        tile_coo_restore_val_block[smem_st_ptr * tile_coo_val_size + OFFSET_ROW(row, col, 16 + 8)] = value;
    }
    __syncthreads();
    if (blockIdx.x == 0 && blockIdx.y == 0 && lane_id == 0) {
        // printf("cuda check tile A: i = %d\n", i);
        // for (int i = 0; i < 16; i++) {
        //     for (int j = 0; j < 16; j++) {
        //         printf("%.0f ", (float)tile_coo_restore_val_block[smem_st_ptr * tile_coo_val_size + OFFSET_ROW(i, j, 16)]);
        //     } printf("\n");
        // }
    }
}

template <int TILE_B>
__device__ __forceinline__ void op_memcpy_async_matrix_B_load(
    int i, int smem_st_ptr, int smem_ld_ptr, int lane_id, int* tile_sidx_block, int tile_sidx_size, half* tileB_block, int tile_B_size, int tile_B_row_num, half* dB_val, int N, int block_idx_y
) {
    int n_iter = 0;
    // for (n_iter = 0; n_iter + 1 < TILE_B / 8; n_iter += 2) {
    //     int row = lane_id % 16;
    //     int col = lane_id / 16 + n_iter;
    //     int sidx = tile_sidx_block[smem_ld_ptr * tile_sidx_size + row];
    //     int gmem_col = block_idx_y * TILE_B + n_iter * 8 + lane_id / 16 * 8;
    //     float4* dst = (float4*)(&tileB_block[smem_st_ptr * tile_B_size + OFFSET_ROW(row, col * 8, tile_B_row_num)]);
    //     float4* src = (float4*)(&dB_val[OFFSET_ROW(sidx, gmem_col, N)]);
    //     __pipeline_memcpy_async(dst, src, sizeof(float4));
    //     if (i == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //         // printf("[%d,%d][%d]tile B,i=%d, bank = %d\n", blockIdx.x, blockIdx.y, lane_id, lane_id / 8, (smem_st_ptr * tile_B_size + OFFSET_ROW(row, col * 8, tile_B_row_num)) / 8 % 8);
    //     }
    // }
    // for (; n_iter < TILE_B / 8; n_iter++) {
    //     if (lane_id < 16) {
    //         int row = lane_id % 16;
    //         int col = lane_id / 16 + n_iter;
    //         int sidx = tile_sidx_block[smem_ld_ptr * tile_sidx_size + row];
    //         int gmem_col = block_idx_y * TILE_B + n_iter * 8 + lane_id / 16 * 8;
    //         float4* dst = (float4*)(&tileB_block[smem_st_ptr * tile_B_size + OFFSET_ROW(row, col * 8, tile_B_row_num)]);
    //         // if (blockIdx.y == 1 && blockIdx.x == 0)
    //         // printf("[%d,%d][%d]sidx=%d,col=%d\n", blockIdx.x, blockIdx.y, threadIdx.x, sidx, gmem_col);
    //         float4* src = (float4*)(&dB_val[OFFSET_ROW(sidx, gmem_col, N)]);
    //         __pipeline_memcpy_async(dst, src, sizeof(float4));
    //     }
    // }
    for (int tileK_iter_inner = 0; tileK_iter_inner < 4; tileK_iter_inner++) {
        int row = tileK_iter_inner * 4 + lane_id / 8;
        int col = lane_id % 8 * 8;
        int gmem_col = block_idx_y * TILE_B + col;
        int sidx = tile_sidx_block[smem_ld_ptr * tile_sidx_size + row];
        float4* dst = (float4*)(&tileB_block[smem_st_ptr * tile_B_size + OFFSET_ROW(row, col, (64 + 8))]);
        float4* src = (float4*)(&dB_val[OFFSET_ROW(sidx, gmem_col, N)]);
        __pipeline_memcpy_async(dst, src, sizeof(float4));
    }
}

template <int TILE_B>
__device__ __forceinline__ void op_calculate(
    int i, int smem_st_ptr, int smem_ld_ptr, int lane_id, half* tile_coo_restore_val_block, int tile_coo_val_size, half* tileB_block, int tile_B_size, int tile_B_row_num, half* REGA, half* REGB, float* REGC
) {
    int row = lane_id % 16;
    int col = lane_id / 16;
    if (blockIdx.x == 0 && blockIdx.y == 0 && lane_id == 0) {
        // printf("check tile B: i = %d\n", i);
        // for (int i = 0; i < 16; i++) {
        //     for (int j = 0; j < 8; j++) {
        //         printf("%.0f ", (float)tileB_block[OFFSET_ROW(i, j, tile_B_row_num)]);
        //     } printf("\n");
        // }
    }
    ldmatrix_m8n8k8_x4((uint32_t*)&REGA[smem_st_ptr * 8], (void*)&tile_coo_restore_val_block[smem_ld_ptr * tile_coo_val_size + OFFSET_ROW(row, col * 8, 16 + 8)]);
    for (int n_iter = 0; n_iter < TILE_B / 8; n_iter++) {
        int row_b = lane_id % 16;
        int col_b = n_iter;
        ldmatrix_m8n8k8_x2_trans((uint32_t*)&REGB[smem_st_ptr * 4 * (TILE_B / 8) + n_iter * 4], (void*)&tileB_block[smem_ld_ptr * tile_B_size + OFFSET_ROW(row_b, col_b * 8, tile_B_row_num)]);
        mma_m16n8k16_fp32(&REGC[4 * n_iter], (uint32_t*)&REGA[smem_st_ptr * 8], &REGB[smem_st_ptr * 4 * (TILE_B / 8) + n_iter * 4]);
    }
}

template <int TILE_B>
__global__ void dtc_spmm_kernel_fp16_val_idx_bind_stage2(
    int M, int N, int K,
    int* dval_sidx,
    int* dval_soff,
    half* dB_val,
    uint32_t* dval_coo_val_idx_bind,
    int* dval_coo_off,
    float* dC_val
) {
    int block_idx_x = blockIdx.x;
    const int block_idx_y = blockIdx.y;
    const int lane_id = threadIdx.x % 32;
    constexpr const int tile_sidx_size = 16;
    constexpr const int tile_coo_val_size = 16 * (16 + 8);
    constexpr const int tile_B_row_num = TILE_B + (TILE_B / 8 % 2 == 1 ? 0 : 8);
    constexpr const int tile_B_size = 16 * tile_B_row_num;
    
    constexpr const int stages = 2;
    
    __shared__ int tile_sidx_block[tile_sidx_size * stages];
    //! val_idx_bind 不需要coo_val_block   
    __shared__ __align__(32) half tile_coo_restore_val_block[tile_coo_val_size * stages];
    __shared__ half tileB_block[tile_B_size * stages];
    __shared__ uint32_t tileA_coo_val_idx_bind_block[16 * 16 * stages];

    //! regs
    int coo_range_reg[stages];

    half REGB[4 * (TILE_B / 8) * stages] = {0};
    half REGA[8 * stages] = {0};
    float REGC[4 * (TILE_B / 8)] = {0};
    

    //! A tile : TC Block ll and rr, G2S, restore in S
    //! B tile: sparse index, load B to SMEM, SMEM to REG
    //! restore 开销可能特别大

    int l = dval_soff[block_idx_x];
    int r = dval_soff[block_idx_x + 1];
    int k = r - l;

    int i = 0;
    //>========================================================
    //! A: memcpy async sparse encoding + value load (0)
    op_memcpy_async_sparse_encoding_value_load<TILE_B>(0, 0 % stages, dval_coo_off, dval_coo_val_idx_bind, l, r, coo_range_reg, lane_id, tileA_coo_val_idx_bind_block, tile_coo_val_size);
    //! B: memcpy async sparse index load (0)
    op_memcpy_async_sparse_index_load<TILE_B>(0, 0 % stages, lane_id, tile_sidx_block, tile_sidx_size, dval_sidx, l);
    __pipeline_commit();
    //>========================================================
    //! A: memcpy async sparse encoding + value load (1)
    op_memcpy_async_sparse_encoding_value_load<TILE_B>(1, 1 % stages, dval_coo_off, dval_coo_val_idx_bind, l, r, coo_range_reg, lane_id, tileA_coo_val_idx_bind_block, tile_coo_val_size);
    //! B: memcpy async sparse index load (1)
    op_memcpy_async_sparse_index_load<TILE_B>(1, 1 % stages, lane_id, tile_sidx_block, tile_sidx_size, dval_sidx, l);
    __pipeline_wait_prior(0);
    __syncthreads();
    //! A: memcpy async restore matrix A (0)
    op_restore_matrix_A<TILE_B>(0, 0 % stages, 0 % stages, lane_id, coo_range_reg, tile_coo_restore_val_block, tile_coo_val_size, tileA_coo_val_idx_bind_block);
    //! B: memcpy async dense matrix B load (0)
    op_memcpy_async_matrix_B_load<TILE_B>(0, 0 % stages, 0 % stages, lane_id, tile_sidx_block, tile_sidx_size, tileB_block, tile_B_size, tile_B_row_num, dB_val, N, block_idx_y);
    __pipeline_commit();
    //>========================================================
    for (i = 0; i + 2 < k; i++) {  
        //! A: memcpy async sparse encoding + value load (i + 2)
        op_memcpy_async_sparse_encoding_value_load<TILE_B>(i + 2, (i + 2) % stages, dval_coo_off, dval_coo_val_idx_bind, l, r, coo_range_reg, lane_id, tileA_coo_val_idx_bind_block, tile_coo_val_size);
        //! B: memcpy async sparse index load (i + 2)
        op_memcpy_async_sparse_index_load<TILE_B>(i + 2, (i + 2) % stages, lane_id, tile_sidx_block, tile_sidx_size, dval_sidx, l);
        __pipeline_wait_prior(0);
        __pipeline_wait_prior(0);
        __syncthreads();
        //! A: memcpy async restore matrix A (i + 1)
        op_restore_matrix_A<TILE_B>(i + 1, (i + 1) % stages, (i + 1) % stages, lane_id, coo_range_reg, tile_coo_restore_val_block, tile_coo_val_size, tileA_coo_val_idx_bind_block);
        //! B: memcpy async dense matrix B load (i + 1)
        op_memcpy_async_matrix_B_load<TILE_B>(i + 1, (i + 1) % stages, (i + 1) % stages, lane_id, tile_sidx_block, tile_sidx_size, tileB_block, tile_B_size, tile_B_row_num, dB_val, N, block_idx_y);
        __pipeline_commit();
        //! ldmatrix A, ldmatrix B, and MMA (i)
        op_calculate<TILE_B>(i, i % stages, i % stages, lane_id, tile_coo_restore_val_block, tile_coo_val_size, tileB_block, tile_B_size, tile_B_row_num, REGA, REGB, REGC);
    }
    for (; i + 1 < k; i++) {
        __pipeline_wait_prior(0);
        __syncthreads();
        //! A: memcpy async restore matrix A (i + 1)
        op_restore_matrix_A<TILE_B>(i + 1, (i + 1) % stages, (i + 1) % stages, lane_id, coo_range_reg, tile_coo_restore_val_block, tile_coo_val_size, tileA_coo_val_idx_bind_block);
        //! B: memcpy async dense matrix B load (i + 1)
        op_memcpy_async_matrix_B_load<TILE_B>(i + 1, (i + 1) % stages, (i + 1) % stages, lane_id, tile_sidx_block, tile_sidx_size, tileB_block, tile_B_size, tile_B_row_num, dB_val, N, block_idx_y);
        __pipeline_commit();
        //! ldmatrix A, ldmatrix B, and MMA (i)
        op_calculate<TILE_B>(i, i % stages, i % stages, lane_id, tile_coo_restore_val_block, tile_coo_val_size, tileB_block, tile_B_size, tile_B_row_num, REGA, REGB, REGC);
    }
    for (; i < k; i++) {
        __pipeline_wait_prior(0);
        __syncthreads();
        //! ldmatrix A, ldmatrix B, and MMA (i)
        op_calculate<TILE_B>(i, i % stages, i % stages, lane_id, tile_coo_restore_val_block, tile_coo_val_size, tileB_block, tile_B_size, tile_B_row_num, REGA, REGB, REGC);
    }
    //>========================================================
    //! store back
    for (int i_tileN = 0; i_tileN < TILE_B / 8; i_tileN++) {
        int row = block_idx_x * 16 + lane_id / 4;
        int col = block_idx_y * TILE_B + i_tileN * 8 + lane_id % 4 * 2;
        dC_val[OFFSET_ROW(row, col, N)] = REGC[i_tileN * 4 + 0];
        dC_val[OFFSET_ROW(row, col + 1, N)] = REGC[i_tileN * 4 + 1];
        dC_val[OFFSET_ROW(row + 8, col, N)] = REGC[i_tileN * 4 + 2];
        dC_val[OFFSET_ROW(row + 8, col + 1, N)] = REGC[i_tileN * 4 + 3];
    }
}


template <int TILE_B>
void ncu_test_all(
    int threadblock_num_x, int threadblock_num_y, int thread_num,
    int M, int N, int K,
    int* dval_sidx,
    int* dval_soff,
    half* dB_val,
    uint32_t* dval_coo_val_idx_bind,
    int* dval_coo_off,
    float* dC
) {
    dim3 grid(threadblock_num_x, threadblock_num_y);
    dim3 block(thread_num);
    dtc_spmm_kernel_fp16_val_idx_bind_stage2<TILE_B><<<grid, block>>>(M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_val_idx_bind, dval_coo_off, dC);
}

template <int TILE_B>
void dtc_spmm_test_val_sidx_bind_reorder(
    int M, int N, int K,
    int TCBlock_num,
    int nnz, 
    int* val_sidx,
    int* val_soff,
    half* B_val,
    uint32_t* val_coo_val_idx_bind,
    int* val_coo_off,
    float* C
) {
    int* dval_sidx;
    int* dval_soff;
    half* dB_val;
    uint32_t* dval_coo_val_idx_bind;
    int* dval_coo_off;
    float* dC;

    int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;
    int pad_K = (K + 16 - 1) / 16 * 16;
    assert(N % TILE_B == 0);

    CHECK_CUDA2(cudaMalloc(&dval_sidx, sizeof(int) * TCBlock_num * TC_BLOCK_K));

    CHECK_CUDA2(cudaMalloc(&dval_soff, sizeof(int) * (row_window_num + 1)));
    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(half) * K * N));
    // CHECK_CUDA2(cudaMalloc(&dval_coo_idx, (nnz + 8 - 1) / 8 * 8 * sizeof(int)));
    // CHECK_CUDA2(cudaMemset(dval_coo_idx, 0, (nnz + 8 - 1) / 8 * 8 * sizeof(int)));
    // CHECK_CUDA2(cudaMalloc(&dval_coo_val, (nnz + 8 - 1) / 8 * 8 * sizeof(half)));
    // CHECK_CUDA2(cudaMemset(dval_coo_val, 0, (nnz + 8 - 1) / 8 * 8 * sizeof(half)));
    //! val_idx_bind
    CHECK_CUDA2(cudaMalloc(&dval_coo_val_idx_bind, (nnz + 8 - 1) / 8 * 8 * sizeof(uint32_t)));
    CHECK_CUDA2(cudaMemset(dval_coo_val_idx_bind, 0, (nnz + 8 - 1) / 8 * 8 * sizeof(uint32_t)));
    CHECK_CUDA2(cudaMalloc(&dval_coo_off, sizeof(int) * (2 * TCBlock_num + 1)));
    CHECK_CUDA2(cudaMalloc(&dC, sizeof(float) * (row_window_num * ROW_WINDOW_M) * N)); // C is row major (alloc more copy some)
    CHECK_CUDA2(cudaMemcpy(dval_sidx, val_sidx, sizeof(int) * TCBlock_num * TC_BLOCK_K, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_soff, val_soff, sizeof(int) * (row_window_num + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(half) * K * N, cudaMemcpyHostToDevice));
    // CHECK_CUDA2(cudaMemcpy(dval_coo_idx, val_coo_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    // CHECK_CUDA2(cudaMemcpy(dval_coo_val, val_coo_val, sizeof(half) * nnz, cudaMemcpyHostToDevice));
    //! val_idx_bind
    CHECK_CUDA2(cudaMemcpy(dval_coo_val_idx_bind, val_coo_val_idx_bind, sizeof(uint32_t) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_coo_off, val_coo_off, sizeof(int) * (2 * TCBlock_num + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * (row_window_num * ROW_WINDOW_M) * N));

    int threadblock_num_x = row_window_num;
    int threadblock_num_y = N / TILE_B;
    int thread_num = 32;

    ncu_test_all<TILE_B>(threadblock_num_x, threadblock_num_y, thread_num, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_val_idx_bind, dval_coo_off, dC);

    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
}

template void dtc_spmm_test_val_sidx_bind_reorder<64>(
    int M, int N, int K,
    int TCBlock_num, 
    int nnz,
    int* val_sidx,
    int* val_soff,
    half* B_val,
    uint32_t* val_coo_val_idx_bind,
    int* val_coo_off,
    float* C
);