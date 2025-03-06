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

using namespace std;


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

// shift 1
template <int maxshift_smem, int maxshift_reg>
void op_G2S_val_sidx_load(int i) {
    printf("op_G2S_val_sidx_load(%d): dval_sidx[%d] -> sval_sidx[%d]\n", i, i, i % maxshift_smem);
}

template <int maxshift_smem, int maxshift_reg, int r1>
void op_G2R_coo_atomic_format_load_off(int i) {
    printf("op_G2R_coo_atomic_format_load_off(%d): dval_coo_off[%d] -> coo_scope[%d]\n", i, i, r1);
}

// shift 2
template <int maxshift_smem, int maxshift_reg, int r1>
void op_S2R_val_sidx_load(int i) {
    printf("op_S2R_val_sidx_load(%d): sval_sidx[%d] -> val_sidx_reg[%d]\n", i, i % maxshift_smem, r1);
}

template <int maxshift_smem, int maxshift_reg>
void op_G2S_coo_atomic_format_load_val(int i) {
    printf("op_G2S_coo_atomic_format_load_val(%d): dval_coo_val[%d] -> tile_coo_val_block[%d]\n", i, i, i % maxshift_smem);
}

template <int maxshift_smem, int maxshift_reg, int r1>
void op_G2R_coo_atomic_format_load_idx(int i) {
    printf("op_G2R_coo_atomic_format_load_idx(%d): dval_coo_idx[%d] -> coo_idx_reg[%d]\n", i, i, r1);
}

// shift 3
template <int maxshift_smem, int maxshift_reg, int r1>
void op_G2R_B_val_load(int i) {
    printf("op_G2R_B_val_load(%d): dB_val[%d] -> REGB[%d]\n", i, i, r1);
}

template <int maxshift_smem, int maxshift_reg, int r1>
void op_S2R_coo_atomic_val_restore(int i) {
    printf("op_S2R_coo_atomic_val_restore(%d): tile_coo_val_block[%d] -> restore[%d] -> REGA[%d]\n", i, i % maxshift_smem, i % maxshift_smem, r1);
}

template <int maxshift_smem, int maxshift_reg, int r1, int r2>
void op_mma(int i) {
    printf("op_mma(%d): REGA[%d] x REGB[%d] -> REGC\n", i, r1, r2);
}

template <int shift1, int shift2, int shift3, int maxshift_reg, int maxshift_smem, int K, int i>
void pipeline() {
    if constexpr (i < K) {
        op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i);

        op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, i % maxshift_reg>(i);
    }
    if constexpr (0 <= i - shift1 && i - shift1 < K) {
        op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, (i - shift1) % maxshift_reg>(i - shift1);
        op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg>(i - shift1);
        op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (i - shift1) % maxshift_reg>(i - shift1);
    }
    if constexpr (0 <= i - shift2 - shift1 && i - shift2 - shift1 < K) {
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (i - shift1 - shift2) % maxshift_reg>(i - shift1 - shift2);
        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (i - shift1 - shift2) % maxshift_reg>(i - shift1 - shift2);
    }
    if constexpr (0 <= i - shift1 - shift2 - shift3 && i - shift1 - shift2 - shift3 < K) {
        op_mma<maxshift_smem, maxshift_reg, (i - shift1 - shift2 - shift3) % maxshift_reg, (i - shift1 - shift2 - shift3) % maxshift_reg>(i - shift1 - shift2 - shift3);
    }

}

template <int shift1, int shift2, int shift3, int maxshift_reg, int maxshift_smem, int K>
void pipeline_switch(int k) {
    if constexpr (K == 0) {
        return;
    } else {
        if (k == K) {
            pipeline<>();
        } else {
            pipeline_switch<shift1, shift2, shift3, maxshift_reg, maxshift_smem, K - 1>(k);
        }
    }
}

template <int shift1, int shift2, int shift3, int TILE_B>
__global__ void dtc_spmm_kernel_sstage_rstage(
    int M, int N, int K,
    int* dval_sidx,
    int* dval_soff,
    half* dB_val,
    int* dval_coo_idx,
    half* dval_coo_val,
    int* dval_coo_off
) {
    const int block_idx_x = blockIdx.x;
    const int block_idx_y = blockIdx.y;
    const int lane_id = threadIdx.x % 32;
    constexpr const int tile_sidx_size = 16;
    constexpr const int tile_coo_val_size = 16 * 16;
    
    constexpr const int maxshift_smem = (shift1 > shift2 ? shift1 : shift2);
    constexpr const int maxshift_reg = (maxshift_smem > shift3 ? maxshift_smem : shift3) + 1;

    __shared__ int tile_sidx_block[tile_sidx_size * (maxshift_smem + 1)];
    __shared__ half tile_coo_val_block[tile_coo_val_size * (maxshift_smem + 1)];
    __shared__ half tile_coo_restore_val_block[tile_coo_val_size * (maxshift_smem + 1)];

    // shift 1
    int coo_scope[2 * maxshift_reg];
    // shift 2
    int val_sidx_reg[tile_sidx_size * maxshift_reg];
    int coo_idx_reg[16 * 16 * maxshift_reg];
    // shift 3
    // tile A [16 * 16], tile B [16 * 32]
    half REGB[2 * (TILE_B / 8) * maxshift_reg];
    half REGA[4 * maxshift_reg];
    float REGC[4 * (TILE_B / 8)];

    int l = dval_soff[block_idx_x];
    int r = dval_soff[block_idx_x + 1];

    int k = l - r;

    if (k <= shift1 + shift2 + shift3) {

    } else {

    }

    
}

template <int v, int mma_m, int mma_n, int mma_k>
void dtc_spmm_test(
    int M, int N, int K,
    int* val_sidx,
    int* val_soff,
    half* B_val,
    int* val_coo_idx,
    half* val_coo_val,
    int* val_coo_off
) {

}