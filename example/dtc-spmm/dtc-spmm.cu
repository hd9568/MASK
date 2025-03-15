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

/*
maxshift_smem = 4;
maxshift_reg  = 4;

-----sync
OP1: 0 -> reg[0]

-----sync
op1: 1 -> reg[1]

-----sync
op1: 2 -> reg[2]

-----sync
op1: 3 -> reg[3]
op2: gmem[0] -> smem[0]

-----sync
op1: 4 -> reg[0]
op2: gmem[1] -> smem[1]

-----sync
op1: 5 -> reg[1]
op2: gmem[2] -> smem[2]

-----sync
op1: 6 -> reg[2]
op2: gmem[3] -> smem[3]
op3: smem[0] -> reg[0]

-----sync
op1: 7 -> reg[3]
op2: gmem[4] -> smem[0]
op3: smem[1] -> reg[1]

-----sync
op1: 8 -> reg[0]
op2: gmem[5] -> smem[1]
op3: smem[2] -> reg[2]
op4: mma(0), reg[0]

-----sync
op1: 9 -> reg[1]
op2: gmem[6] -> smem[2]
op3: smem[3] -> reg[3]
op4: mma(1), reg[1]

-----sync
op1: 10 -> reg[2]
op2: gmem[7] -> smem[3]
op3: smme[0] -> reg[0]
op4: mma(2), reg[2]

----sync
op1: 11 -> reg[3]
op2: gmem[8] -> smem[0]
op3: smem[1] -> reg[1]
op4: mma(3), reg[3]

-----sync
op1: 12 -> reg[0]
op2: gmem[9] -> smem[1]
op3: smem[2] -> reg[2]
op4: mma(4), reg[0]

TODO: 如果shift1和shift2大于shift3（这种情况中SMEM和REG副本数量相同），这样做是对的，需要验证其他情况

shift1 = 1
shift2 = 2
shift3 = 3
maxshift_smem = (2 + 1)
maxshift_reg = (3 + 1)

op1: 0 -> reg[0]

-----sync
op1: 1 -> reg[1]


*/

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

__device__ __forceinline__ void mma_m16n8k16_fp32(float* acc, uint32_t* A, half* half_B) {
    uint32_t* B = reinterpret_cast<uint32_t*>(half_B);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1])
    );
}

// shift 1
template <int maxshift_smem, int maxshift_reg>
__device__ void op_G2S_val_sidx_load(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if (lane_id == 0)
    printf("op_G2S_val_sidx_load(%d): dval_sidx[%d] -> sval_sidx[%d]\n", i, i, i % maxshift_smem);
    // load block from l to r
    int smem_st_ptr = i % maxshift_smem;
    if (lane_id < 4) {
        int4* dst = (int4*)(&tile_sidx_block[smem_st_ptr * tile_sidx_size + lane_id * 4]);
        int4* src = (int4*)(&dval_sidx[(l + i) * 16 + lane_id * 4]);
        __pipeline_memcpy_async(dst, src, sizeof(int4));
    }
}

template <int maxshift_smem, int maxshift_reg, int r1>
__device__ void op_G2R_coo_atomic_format_load_off(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if (lane_id == 0)
    printf("op_G2R_coo_atomic_format_load_off(%d): dval_coo_off[%d] -> coo_scope[%d]\n", i, i, r1);
    if (lane_id == 0) {
        printf("i=%d, l=%d, dval_coo_off[l+i, l+i+1]=(%d,%d)\n", i, l, dval_coo_off[l + i], dval_coo_off[l + i + 1]);
    }
    coo_scope[2 * r1] = dval_coo_off[l + i];
    coo_scope[2 * r1 + 1] = dval_coo_off[l + i + 1];
}

// shift 2
template <int maxshift_smem, int maxshift_reg, int TILE_B, int r1, int wait_num>
__device__ void op_S2R_val_sidx_load(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if (lane_id == 0)
    printf("op_S2R_val_sidx_load(%d): wait(%d), sval_sidx[%d] -> val_sidx_reg[%d]\n", i, wait_num, i % maxshift_smem, r1);
    __pipeline_wait_prior(wait_num);
    __syncthreads();
    int smem_ld_ptr = i % maxshift_smem;
    if (lane_id == 0) {
        printf("check tile_sidx_block: ");
        for (int i = 0; i < 16; i++) {
            printf("%d ", tile_sidx_block[smem_ld_ptr * tile_sidx_size + i]);
        } printf("\n");
    }
    *(int4*)(&val_sidx_reg[0]) = *(int4*)(&tile_sidx_block[smem_ld_ptr * tile_sidx_size + 0]);
    *(int4*)(&val_sidx_reg[4]) = *(int4*)(&tile_sidx_block[smem_ld_ptr * tile_sidx_size + 4]);
    *(int4*)(&val_sidx_reg[8]) = *(int4*)(&tile_sidx_block[smem_ld_ptr * tile_sidx_size + 8]);
    *(int4*)(&val_sidx_reg[12]) = *(int4*)(&tile_sidx_block[smem_ld_ptr * tile_sidx_size + 12]);
}

template <int maxshift_smem, int maxshift_reg, int r1>
__device__ void op_G2S_coo_atomic_format_load_val(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if (lane_id == 0)
    printf("op_G2S_coo_atomic_format_load_val(%d): dval_coo_val[coo_scope[%d]] -> tile_coo_val_block[%d]\n", i, r1, i % maxshift_smem);
    int smem_st_ptr = i % maxshift_smem;
    
    int ll = coo_scope[2 * r1];
    int rr = coo_scope[2 * r1 + 1];
    if (lane_id == 0) {
        printf("i = %d, ll = %d, rr = %d\n", i, ll, rr);
    }
    if (lane_id * 8 + ll < rr) {
        float4* src = (float4*)(&dval_coo_val[ll + lane_id * 8]);
        float4* dst = (float4*)(&tile_coo_val_block[smem_st_ptr * tile_coo_val_size + lane_id * 8]);
        // __pipeline_memcpy_async(dst, src, sizeof(float4));
    }
    
}

template <int maxshift_smem, int maxshift_reg, int r1, int r2>
__device__ void op_G2R_coo_atomic_format_load_idx(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if (lane_id == 0)
    printf("op_G2R_coo_atomic_format_load_idx(%d): dval_coo_idx[coo_scope[%d]] -> coo_idx_reg[%d]\n", i, r1, r2);
    int ll = coo_scope[2 * r1];
    int rr = coo_scope[2 * r1 + 1];
    coo_range_reg[r2] = rr - ll;
    if (lane_id * 8 + ll < rr) {
        *(int4*)(&coo_idx_reg[r2 * 8]) = *(int4*)(&dval_coo_idx[ll + lane_id * 8]);
        *(int4*)(&coo_idx_reg[r2 * 8 + 4]) = *(int4*)(&dval_coo_idx[ll + lane_id * 8 + 4]);
    }
}

// shift 3
template <int maxshift_smem, int maxshift_reg, int r1, int r2, int TILE_B>
__device__ void op_G2R_B_val_load(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if (lane_id == 0)
    printf("op_G2R_B_val_load(%d): dB_val[val_sidx_reg[%d]] -> REGB[%d]\n", i, r1, r2);
    for (int n_iter = 0; n_iter < TILE_B / 8; n_iter++) {
        int row = lane_id % 4 * 2;
        int col = block_idx_y * TILE_B + n_iter * 8 + lane_id / 4;
        (((half2*)(&REGB[r2 * 2 * (TILE_B / 8) + 0])) -> x) = dB_val[OFFSET_COL(val_sidx_reg[row], col, K)];
        (((half2*)(&REGB[r2 * 2 * (TILE_B / 8) + 0])) -> y) = dB_val[OFFSET_COL(val_sidx_reg[row + 1], col, K)];
        (((half2*)(&REGB[r2 * 2 * (TILE_B / 8) + 1])) -> x) = dB_val[OFFSET_COL(val_sidx_reg[row + 8], col, K)];
        (((half2*)(&REGB[r2 * 2 * (TILE_B / 8) + 1])) -> y) = dB_val[OFFSET_COL(val_sidx_reg[row + 9], col, K)];
    }
}

template <int maxshift_smem, int maxshift_reg, int r1, int wait_num>
__device__ void op_S2R_coo_atomic_val_restore(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if (lane_id == 0)
    printf("op_S2R_coo_atomic_val_restore(%d): wait(%d), tile_coo_val_block[%d] -> restore[%d] -> REGA[%d]\n", i, wait_num, i % maxshift_smem, i % maxshift_smem, r1);
    __pipeline_wait_prior(wait_num);
    __syncthreads();
    int nnz_num = coo_range_reg[r1];
    int smem_ld_ptr = i % maxshift_smem;
    for (int i_o2s = 0; i_o2s < 4; i_o2s++) {
        *((int*)&tile_coo_restore_val_block[32 * 2 * i_o2s + lane_id * 2]) = 0;
    }
    for (int i_restore = 0; i_restore < 8; i_restore++) {
        int idx = coo_idx_reg[i_restore];
        if (lane_id * 8 + i_restore < nnz_num) {
            tile_coo_restore_val_block[smem_ld_ptr * tile_coo_val_size + idx] = tile_coo_val_block[smem_ld_ptr * tile_coo_val_size + lane_id * 8 + i_restore];
        }
    }
    __syncthreads();
    if (lane_id == 0) {
        printf("device check tile A: \n");
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                printf("%.0f ", (float)tile_coo_restore_val_block[smem_ld_ptr * tile_coo_val_size + i * 16 + j]);
            } printf("\n");
        } printf("\n");
    }
    int row = lane_id % 16;
    int col = lane_id / 16;
    ldmatrix_m8n8k8_x4((uint32_t*)&REGA[r1 * 4], (void*)&tile_coo_restore_val_block[(row) * (16 + 8) + (col * 8)]);
}

template <int maxshift_smem, int maxshift_reg, int r1, int r2>
__device__ void op_mma(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if (lane_id == 0)
    printf("op_mma(%d): REGA[%d] x REGB[%d] -> REGC\n", i, r1, r2);
    mma_m16n8k16_fp32(REGC, (uint32_t*)&REGA[r1 * 4], &REGB[r2 * 2]);
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_reg, int maxshift_smem, int TILE_B, int K_total, int i>
__device__ void pipeline(int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if (lane_id == 0)
    printf("small pipeline %d\n", i);
    if constexpr (i < K_total) {
        op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);

        op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, i % maxshift_reg>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        if constexpr (i < shift1) {
            __pipeline_commit();
        }
    }
    // (sstage > (K - 1 - (i - sstage)) ? K - 1 - (i - sstage) : sstage)
    if constexpr (0 <= i - shift1 && i - shift1 < K_total) {
        op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg, (i - shift1) % maxshift_reg>(i - shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        __pipeline_commit();
        op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, TILE_B, (i - shift1) % maxshift_reg, (K_total > shift1 ? shift1 : K_total)>(i - shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (i - shift1) % maxshift_reg, (i - shift1) % maxshift_reg>(i - shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    }
    if constexpr (0 <= i - shift2 - shift1 && i - shift2 - shift1 < K_total) {
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (i - shift1 - shift2) % maxshift_reg, (i - shift1 - shift2) % maxshift_reg, TILE_B>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (i - shift1 - shift2) % maxshift_reg, (shift2 > (K_total + shift1 + shift2 - 1 - i) ? (K_total + shift1 + shift2 - 1 - i) : shift2)>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    }
    if constexpr (0 <= i - shift1 - shift2 - shift3 && i - shift1 - shift2 - shift3 < K_total) {
        op_mma<maxshift_smem, maxshift_reg, (i - shift1 - shift2 - shift3) % maxshift_reg, (i - shift1 - shift2 - shift3) % maxshift_reg>(i - shift1 - shift2 - shift3, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    }
    if constexpr (i < K_total + shift1 + shift2 + shift3) {
        pipeline<shift1, shift2, shift3, maxshift_reg, maxshift_smem, TILE_B, K_total, i + 1>(M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    }

}

template <int shift1, int shift2, int shift3, int maxshift_reg, int maxshift_smem, int TILE_B, int K_total>
__device__ void pipeline_switch(int k, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if constexpr (K_total == 0) {
        return;
    } else {
        if (k == K_total) {
            pipeline<shift1, shift2, shift3, maxshift_reg, maxshift_smem, TILE_B, K_total, 0>(M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        } else {
            pipeline_switch<shift1, shift2, shift3, maxshift_reg, maxshift_smem, TILE_B, K_total - 1>(k, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        }
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int i>
__device__ void fill_shift1(int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if constexpr (i == shift1) {
        return;
    } else {
        printf("fill shift1: i = %d\n", i);
        op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, i % maxshift_reg>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        __pipeline_commit();
        fill_shift1<shift1, shift2, shift3, maxshift_smem, maxshift_reg, i + 1>(M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int i>
__device__ void fill_shift2(int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if constexpr (i == shift2) {
        return;
    } else {
        printf("fill shift2: i = %d\n", i + shift1);
        op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i + shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg, i % maxshift_reg>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        __pipeline_commit();
        op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, TILE_B, i % maxshift_reg, shift1>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        
        op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, (i + shift1) % maxshift_reg>(i + shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        
        op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, i % maxshift_reg, i % maxshift_reg>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);

        fill_shift2<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B,  i + 1>(M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int i> 
__device__ void fill_shift3(int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if constexpr (i == shift3) {
        return;
    } else {
        printf("fill shift3: i = %d\n", i + shift1 + shift2);
        op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i + shift1 + shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg, (i + shift2) % maxshift_reg>(i + shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        __pipeline_commit();
        op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, TILE_B, (i + shift2) % maxshift_reg, shift1>(i + shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, i % maxshift_reg, i % maxshift_reg, TILE_B>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);

        op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, (i + shift1 + shift2) % maxshift_reg>(i + shift1 + shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (i + shift2) % maxshift_reg, (i + shift2) % maxshift_reg>(i + shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_smem, i % maxshift_reg, shift2>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        
        fill_shift3<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, i + 1>(M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int I>
__device__ void loop_step(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    printf("loop step: i = %d, I = %d\n", i, I);
    op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg, (I + shift3 + shift2) % maxshift_reg>(i - shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    __pipeline_commit();
    op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, TILE_B, (I + shift3 + shift2) % maxshift_reg, shift1>(i - shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift3) % maxshift_reg, (I + shift3) % maxshift_reg, TILE_B>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);

    op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + shift1) % maxshift_reg>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    
    op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (I + shift3 + shift2) % maxshift_reg, (I + shift3 + shift2) % maxshift_reg>(i - shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (I + shift3) % maxshift_reg, shift2>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    op_mma<maxshift_smem, maxshift_reg, I % maxshift_reg, I % maxshift_reg>(i - shift1 - shift2 - shift3, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);

    if constexpr (I + 1 < maxshift_reg) {
        loop_step<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, I + 1>(i + 1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain, int I>
__device__ void remainder(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if constexpr (I == K_remain) {
        return;
    } else {
        printf("remainder: i = %d\n", i);
        op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg, (I + shift3 + shift2) % maxshift_reg>(i - shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        __pipeline_commit();
        op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, TILE_B, (I + shift3 + shift2) % maxshift_reg, shift1>(i - shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift3) % maxshift_reg, (I + shift3) % maxshift_reg, TILE_B>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);

        op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + shift1) % maxshift_reg>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        
        op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (I + shift3 + shift2) % maxshift_reg, (I + shift3 + shift2) % maxshift_reg>(i - shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (I + shift3) % maxshift_reg, shift2>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_mma<maxshift_smem, maxshift_reg, I % maxshift_reg, I % maxshift_reg>(i - shift1 - shift2 - shift3, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);

        remainder<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, I + 1>(i + 1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain>
__device__ void remainder_switch(int i, int k, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if constexpr (K_remain == 0) {
        return;
    } else {
        if (k - i == K_remain) {
            remainder<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, 0>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);        
        } else {
            remainder_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain - 1>(i, k, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        }
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain, int I>
__device__ void empty_shift1(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {  // K is the remain
    if constexpr (I == shift1) {
        return;
    } else {
        printf("empty_shift1: i = %d, I = %d\n", i, I);
        // op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i);
        op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + K_remain) % maxshift_reg>(i - shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        __pipeline_commit();
        op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, TILE_B, (I + shift3 + shift2 + K_remain) % maxshift_reg, shift1/*这里还有commit，所以还是shift1*/>(i - shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift3 + K_remain) % maxshift_reg, (I + shift3 + K_remain) % maxshift_reg, TILE_B>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);

        
        op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + K_remain) % maxshift_reg, (I + shift3 + shift2 + K_remain) % maxshift_reg>(i - shift1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (I + shift3 + K_remain) % maxshift_reg, shift2>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_mma<maxshift_smem, maxshift_reg, (I + K_remain) % maxshift_reg, (I + K_remain) % maxshift_reg>(i - shift1 - shift2 - shift3, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);

        empty_shift1<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, I + 1>(i + 1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain>
__device__ void empty_shift1_switch(int i, int remain, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if constexpr (K_remain < 0) {
        return;
    } else {
        if (K_remain == remain) {
            empty_shift1<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, 0>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        } else {
            empty_shift1_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain - 1>(i, remain, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        }
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain, int I>
__device__ void empty_shift2(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if constexpr (I == shift2) {
        return;
    } else {
        printf("empty shift2: i = %d, I = %d\n", i, I);
    
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift1 + shift3 + K_remain) % maxshift_reg, (I + shift1 + shift3 + K_remain) % maxshift_reg, TILE_B>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);

        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (I + shift1 + shift3 + K_remain) % maxshift_reg, shift2 - (I + 1)>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        op_mma<maxshift_smem, maxshift_reg, (I + shift1 + K_remain) % maxshift_reg, (I + shift1 + K_remain) % maxshift_reg>(i - shift1 - shift2 - shift3, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);

        empty_shift2<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, I + 1>(i + 1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain>
__device__ void empty_shift2_switch(int i, int remain, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if constexpr (K_remain < 0) {
        return;
        
    } else {
        if (remain == K_remain) {
            empty_shift2<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, 0>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        } else {
            empty_shift2_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain - 1>(i, remain, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        }
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int K_remain, int I>
__device__ void empty_shift3(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if constexpr (I == shift3) {
        return;
    } else {
        printf("empty shift3: i = %d, I = %d\n", i, I);

        op_mma<maxshift_smem, maxshift_reg, (I + shift1 + shift2 + K_remain) % maxshift_reg, (I + shift1 + shift2 + K_remain) % maxshift_reg>(i - shift1 - shift2 - shift3, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        empty_shift3<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K_remain, I + 1>(i + 1, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int K_remain>
__device__ void empty_shift3_switch(int i, int remain, int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size) {
    if constexpr (K_remain < 0) {
        return;
    } else {
        if (K_remain == remain) {
            empty_shift3<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K_remain, 0>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        } else {
            empty_shift3_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K_remain - 1>(i, remain, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        }
    }
}

// =====================================================================================


template <int shift1, int shift2, int shift3, int TILE_B>
__global__ void dtc_spmm_kernel_sstage_rstage(
    int M, int N, int K,
    int* dval_sidx,
    int* dval_soff,
    half* dB_val,
    int* dval_coo_idx,
    half* dval_coo_val,
    int* dval_coo_off,
    float* dC_val
) {
    const int block_idx_x = blockIdx.x;
    const int block_idx_y = blockIdx.y;
    const int lane_id = threadIdx.x % 32;
    constexpr const int tile_sidx_size = 16;
    constexpr const int tile_coo_val_size = 16 * 16;
    
    // constexpr const int maxshift_smem = (shift1 > shift2 ? shift1 : shift2);
    // constexpr const int maxshift_reg = (maxshift_smem > shift3 ? maxshift_smem : shift3) + 1;
    constexpr const int maxshift1_2 = (shift1 > shift2 ? shift1 : shift2);
    constexpr const int maxshift_reg = (maxshift1_2 > shift3 ? maxshift1_2 : shift3) + 1;
    constexpr const int maxshift_smem = maxshift1_2 + 1;

    __shared__ int tile_sidx_block[tile_sidx_size * (maxshift_smem)];
    __shared__ half tile_coo_val_block[tile_coo_val_size * (maxshift_smem)];
    __shared__ half tile_coo_restore_val_block[tile_coo_val_size * (maxshift_smem)];

    // shift 1
    int coo_scope[2 * maxshift_reg];
    // shift 2
    int val_sidx_reg[tile_sidx_size * maxshift_reg];
    int coo_idx_reg[16 * 16 * maxshift_reg];
    int coo_range_reg[maxshift_reg];
    // shift 3
    // tile A [16 * 16], tile B [16 * 32]
    half REGB[2 * (TILE_B / 8) * maxshift_reg];
    half REGA[4 * maxshift_reg];
    float REGC[4 * (TILE_B / 8)];

    if (threadIdx.x == 0) {
        printf("[%d,%d,%d][%d,%d,%d]block_idx_x=%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, block_idx_x);
    }
    int l = dval_soff[block_idx_x];
    int r = dval_soff[block_idx_x + 1];

    if (threadIdx.x == 0) {
        printf("[%d,%d,%d][%d,%d,%d]k=%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, r - l);
    }

    // return;

    int k = r - l;
    // , int M, int N, int K, int* dval_sidx, int* dval_soff, half* dB_val, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int coo_scope, int val_sidx_reg, int coo_idx_reg, half REGB, half REGA, float REGC
    if (k <= shift1 + shift2 + shift3) {
        pipeline_switch<shift1, shift2, shift3, maxshift_reg, maxshift_smem, TILE_B, shift1 + shift2 + shift3>(k, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    } else {
        fill_shift1<shift1, shift2, shift3, maxshift_smem, maxshift_reg, 0>(M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        fill_shift2<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, 0>(M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        fill_shift3<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, 0>(M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        int i;
        for (i = shift1 + shift2 + shift3; i + maxshift_reg <= k; i += maxshift_reg) {
            loop_step<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, 0>(i, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        }
        int remain = k - i;
        remainder_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, maxshift_reg - 1>(i, k, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        i = k;
        empty_shift1_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, maxshift_reg - 1>(i, remain, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        i = k + shift1;
        empty_shift2_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, maxshift_reg - 1>(i, remain, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
        i = k + shift1 + shift2;
        empty_shift3_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, maxshift_reg - 1>(i, remain, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size);
    }

    //! write back
    for (int i_tileN = 0; i_tileN < TILE_B / 8; i_tileN++) {
        int row = block_idx_x * 16 + lane_id / 4;
        int col = block_idx_y * TILE_B + lane_id % 4 * 2;
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
    int* dval_coo_idx,
    half* dval_coo_val,
    int* dval_coo_off,
    float* dC
) {
    dim3 grid(threadblock_num_x, threadblock_num_y);
    dim3 block(thread_num);
    //! （shift1=1,shift2=1,shift3=1)
    dtc_spmm_kernel_sstage_rstage<1, 1, 1, 32><<<grid, block>>>(M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC);

}


template <int TILE_B>
void dtc_spmm_test(
    int M, int N, int K,
    int TCBlock_num, 
    int nnz, 
    int* val_sidx,
    int* val_soff,
    half* B_val,
    int* val_coo_idx,
    half* val_coo_val,
    int* val_coo_off,
    float* C
) {
    int* dval_sidx;
    int* dval_soff;
    half* dB_val;
    int* dval_coo_idx;
    half* dval_coo_val;
    int* dval_coo_off;
    float* dC;

    int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;

    printf("dval_sidx: alloc TCBlock_num * TC_Block_K = %d\n", TCBlock_num * TC_BLOCK_K);
    printf("dval_soff: alloc row_window_num + 1 = %d\n", sizeof(int) * (row_window_num + 1));
    printf("dB_val alloc: K * N = %d\n", sizeof(half) * K * N);
    printf("dval_coo_idx alloc: nnz = %d\n", sizeof(int) * nnz);
    printf("dval_coo_val alloc: nnz = %d\n", sizeof(half) * nnz);
    printf("dval_coo_off alloc: TCBlock_num + 1 = %d\n", sizeof(int) * (TCBlock_num + 1));
    printf("dC alloc: M * N = %d\n", sizeof(float) * M * N);

    // exit(0);
    
    CHECK_CUDA2(cudaMalloc(&dval_sidx, sizeof(int) * TCBlock_num * TC_BLOCK_K));
    // exit(0);
    
    CHECK_CUDA2(cudaMalloc(&dval_soff, sizeof(int) * (row_window_num + 1)));
    // exit(0);
    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(half) * K * N));
    CHECK_CUDA2(cudaMalloc(&dval_coo_idx, sizeof(int) * nnz));
    // exit(0);
    CHECK_CUDA2(cudaMalloc(&dval_coo_val, sizeof(half) * nnz));
    CHECK_CUDA2(cudaMalloc(&dval_coo_off, sizeof(int) * (TCBlock_num + 1)));
    CHECK_CUDA2(cudaMalloc(&dC, sizeof(float) * M * N));
    // exit(0);
    CHECK_CUDA2(cudaMemcpy(dval_sidx, val_sidx, sizeof(int) * TCBlock_num * TC_BLOCK_K, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_soff, val_soff, sizeof(int) * (row_window_num + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(half) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_coo_idx, val_coo_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_coo_val, val_coo_val, sizeof(half) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_coo_off, val_coo_off, sizeof(int) * (TCBlock_num + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * M * N));

    int threadblock_num_x = row_window_num;
    int threadblock_num_y = N / TILE_B;
    int thread_num = 32;
    
    // exit(0);
    ncu_test_all<TILE_B>(threadblock_num_x, threadblock_num_y, thread_num, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC);
    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

}

template void dtc_spmm_test<32>(
    int, int, int, 
    int, int, 
    int*, int*, half*, 
    int*, half*, 
    int*, float*
);