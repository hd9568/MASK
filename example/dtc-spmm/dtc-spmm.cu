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

#define ROW_WINDOW_M 16
#define TC_BLOCK_K 16
#define TC_BLOCK_M 16

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))
#define OFFSET_COL(row, col, lda) ((col) * (lda) + (row))

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

int l = dval_soff[block_idx_x];
int r = dval_soff[block_idx_x + 1];

int k = l - r;


// shift 1
template <int maxshift_smem, int maxshift_reg>
void op_G2S_val_sidx_load(int i) {
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
void op_G2R_coo_atomic_format_load_off(int i) {
    printf("op_G2R_coo_atomic_format_load_off(%d): dval_coo_off[%d] -> coo_scope[%d]\n", i, i, r1);
    coo_scope[2 * r1] = dval_coo_off[l + i];
    coo_scope[2 * r1 + 1] = dval_coo_off[l + i + 1];
}

// shift 2
template <int maxshift_smem, int maxshift_reg, int TILE_N, int r1, int wait_num>
void op_S2R_val_sidx_load(int i) {
    printf("op_S2R_val_sidx_load(%d): wait(%d), sval_sidx[%d] -> val_sidx_reg[%d]\n", i, wait_num, i % maxshift_smem, r1);
    __pipeline_wait_prior(wait_num);
    __syncthreads();
    int smem_ld_ptr = i % maxshift_smem;
    *(int4*)(val_sidx_reg[0]) = *(int4*)(&tile_sidx_block[smem_ld_ptr * tile_sidx_size + 0]);
    *(int4*)(val_sidx_reg[4]) = *(int4*)(&tile_sidx_block[smem_ld_ptr * tile_sidx_size + 4]);
    *(int4*)(val_sidx_reg[8]) = *(int4*)(&tile_sidx_block[smem_ld_ptr * tile_sidx_size + 8]);
    *(int4*)(val_sidx_reg[12]) = *(int4*)(&tile_sidx_block[smem_ld_ptr * tile_sidx_size + 12]);
}

template <int maxshift_smem, int maxshift_reg, int r1>
void op_G2S_coo_atomic_format_load_val(int i) {
    printf("op_G2S_coo_atomic_format_load_val(%d): dval_coo_val[coo_scope[%d]] -> tile_coo_val_block[%d]\n", i, r1, i % maxshift_smem);
    int smem_st_ptr = i % maxshift_smem;
    
    int ll = coo_scope[2 * r1];
    int rr = coo_scope[2 * r1 + 1];
    if (lane_id * 8 + ll < rr) {
        float4* src = (float4*)(&dval_coo_val[ll + lane_id * 8]);
        float4* dst = (float4*)(tile_coo_val_block[smem_st_ptr * tile_coo_val_size + lane_id * 8]);
        __pipeline_memcpy_async(dst, src, sizeof(float4));
    }
    
}

template <int maxshift_smem, int maxshift_reg, int r1, int r2>
void op_G2R_coo_atomic_format_load_idx(int i) {
    printf("op_G2R_coo_atomic_format_load_idx(%d): dval_coo_idx[coo_scope[%d]] -> coo_idx_reg[%d]\n", i, r1, r2);
    int ll = coo_scope[2 * r1];
    int rr = coo_scope[2 * r1 + 1];
    coo_range_reg[r2] = rr - ll;
    if (lane_id * 8 + ll < rr) {
        *(int4*)(&coo_idx_reg[r2 * 8]) = *(int4*)(&dval_coo_idx[ll + lane_id * 8]);
        *(int4*)(&coo_idx_reg[r2 * 8 + 4]) = *(int4)(&dval_coo_idx[ll + lane_id * 8 + 4]);
    }
}

// shift 3
template <int maxshift_smem, int maxshift_reg, int r1, int r2, int TILE_N>
void op_G2R_B_val_load(int i) {
    printf("op_G2R_B_val_load(%d): dB_val[val_sidx_reg[%d]] -> REGB[%d]\n", i, r1, r2);
    for (int n_iter = 0; n_iter < TILE_N / 8; n_iter++) {
        int row = lane_id % 4 * 2;
        int col = block_idx_y * TILE_N + n_iter * 8 + lane_id / 4;
        ((half2*)(&REGB[r2 * 2 * (TILE_B / 8) + 0]) -> x) = dB[OFFSET_COL(val_sidx_reg[row], col, K)];
        ((half2*)(&REGB[r2 * 2 * (TILE_B / 8) + 0]) -> y) = dB[OFFSET_COL(val_sidx_reg[row + 1], col, K)];
        ((half2*)(&REGB[r2 * 2 * (TILE_B / 8) + 1]) -> x) = dB[OFFSET_COL(val_sidx_reg[row + 8], col, K)];
        ((half2*)(&REGB[r2 * 2 * (TILE_B / 8) + 1]) -> y) = dB[OFFSET_COL(val_sidx_reg[row + 9], col, K)];
    }
}

template <int maxshift_smem, int maxshift_reg, int r1, int wait_num>
void op_S2R_coo_atomic_val_restore(int i) {
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
    int row = lane_id % 16;
    int col = lane_id / 16;
    ldmatrix_m8n8_x4(&REGA[r1 * 4], &tile_coo_restore_val_block[(row) * (16 + 8) + (col * 8)]);
}

template <int maxshift_smem, int maxshift_reg, int r1, int r2>
void op_mma(int i) {
    printf("op_mma(%d): REGA[%d] x REGB[%d] -> REGC\n", i, r1, r2);
    mma_m16n8k16_fp32(&REGA[r1 * 4], &REGB[r2 * 2], REGC);
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_reg, int maxshift_smem, int K, int i>
void pipeline() {
    printf("small pipeline %d\n", i);
    if constexpr (i < K) {
        op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i);

        op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, i % maxshift_reg>(i);
        if constexpr (i < shift1) {
            __pipeline_commit();
        }
    }
    // (sstage > (K - 1 - (i - sstage)) ? K - 1 - (i - sstage) : sstage)
    if constexpr (0 <= i - shift1 && i - shift1 < K) {
        op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg, (i - shift1) % maxshift_reg>(i - shift1);
        __pipeline_commit();
        op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, (i - shift1) % maxshift_reg, (K > shift1 ? shift1 : K)>(i - shift1);
        op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (i - shift1) % maxshift_reg, (i - shift1) % maxshift_reg>(i - shift1);
    }
    if constexpr (0 <= i - shift2 - shift1 && i - shift2 - shift1 < K) {
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (i - shift1 - shift2) % maxshift_reg, (i - shift1 - shift2) % maxshift_reg>(i - shift1 - shift2);
        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (i - shift1 - shift2) % maxshift_reg, (shift2 > (K + shift1 + shift2 - 1 - i) ? (K + shift1 + shift2 - 1 - i) : shift2)>(i - shift1 - shift2);
    }
    if constexpr (0 <= i - shift1 - shift2 - shift3 && i - shift1 - shift2 - shift3 < K) {
        op_mma<maxshift_smem, maxshift_reg, (i - shift1 - shift2 - shift3) % maxshift_reg, (i - shift1 - shift2 - shift3) % maxshift_reg>(i - shift1 - shift2 - shift3);
    }
    if constexpr (i < K + shift1 + shift2 + shift3) {
        pipeline<shift1, shift2, shift3, maxshift_reg, maxshift_smem, K, i + 1>();
    }

}

template <int shift1, int shift2, int shift3, int maxshift_reg, int maxshift_smem, int K>
void pipeline_switch(int k) {
    if constexpr (K == 0) {
        return;
    } else {
        if (k == K) {
            pipeline<shift1, shift2, shift3, maxshift_reg, maxshift_smem, K, 0>();
        } else {
            pipeline_switch<shift1, shift2, shift3, maxshift_reg, maxshift_smem, K - 1>(k);
        }
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int i>
void fill_shift1() {
    if constexpr (i == shift1) {
        return;
    } else {
        printf("fill shift1: i = %d\n", i);
        op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i);
        op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, i % maxshift_reg>(i);
        __pipeline_commit();
        fill_shift1<shift1, shift2, shift3, maxshift_smem, maxshift_reg, i + 1>();
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int i>
void fill_shift2() {
    if constexpr (i == shift2) {
        return;
    } else {
        printf("fill shift2: i = %d\n", i + shift1);
        op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i + shift1);
        op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg, i % maxshift_reg>(i);
        __pipeline_commit();
        op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, i % maxshift_reg, shift1>(i);
        
        op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, (i + shift1) % maxshift_reg>(i + shift1);
        
        op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, i % maxshift_reg, i % maxshift_reg>(i);

        fill_shift2<shift1, shift2, shift3, maxshift_smem, maxshift_reg, i + 1>();
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int i> 
void fill_shift3() {
    if constexpr (i == shift3) {
        return;
    } else {
        printf("fill shift3: i = %d\n", i + shift1 + shift2);
        op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i + shift1 + shift2);
        op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg, (i + shift2) % maxshift_reg>(i + shift2);
        __pipeline_commit();
        op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, (i + shift2) % maxshift_reg, shift1>(i + shift2);
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, i % maxshift_reg, i % maxshift_reg>(i);

        op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, (i + shift1 + shift2) % maxshift_reg>(i + shift1 + shift2);
        op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (i + shift2) % maxshift_reg, (i + shift2) % maxshift_reg>(i + shift2);
        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_smem, i % maxshift_reg, shift2>(i);
        
        fill_shift3<shift1, shift2, shift3, maxshift_smem, maxshift_reg, i + 1>();
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int I>
void loop_step(int i) {
    printf("loop step: i = %d, I = %d\n", i, I);
    op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i);
    op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg, (I + shift3 + shift2) % maxshift_reg>(i - shift1);
    __pipeline_commit();
    op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, (I + shift3 + shift2) % maxshift_reg, shift1>(i - shift1);
    op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift3) % maxshift_reg, (I + shift3) % maxshift_reg>(i - shift1 - shift2);

    op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + shift1) % maxshift_reg>(i);
    
    op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (I + shift3 + shift2) % maxshift_reg, (I + shift3 + shift2) % maxshift_reg>(i - shift1);
    op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (I + shift3) % maxshift_reg, shift2>(i - shift1 - shift2);
    op_mma<maxshift_smem, maxshift_reg, I % maxshift_reg, I % maxshift_reg>(i - shift1 - shift2 - shift3);

    if constexpr (I + 1 < maxshift_reg) {
        loop_step<shift1, shift2, shift3, maxshift_smem, maxshift_reg, I + 1>(i + 1);
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int K, int I>
void remainder(int i) {
    if constexpr (I == K) {
        return;
    } else {
        printf("remainder: i = %d\n", i);
        op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i);
        op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg, (I + shift3 + shift2) % maxshift_reg>(i - shift1);
        __pipeline_commit();
        op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, (I + shift3 + shift2) % maxshift_reg, shift1>(i - shift1);
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift3) % maxshift_reg, (I + shift3) % maxshift_reg>(i - shift1 - shift2);

        op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + shift1) % maxshift_reg>(i);
        
        op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (I + shift3 + shift2) % maxshift_reg, (I + shift3 + shift2) % maxshift_reg>(i - shift1);
        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (I + shift3) % maxshift_reg, shift2>(i - shift1 - shift2);
        op_mma<maxshift_smem, maxshift_reg, I % maxshift_reg, I % maxshift_reg>(i - shift1 - shift2 - shift3);

        remainder<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K, I + 1>(i + 1);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int K>
void remainder_switch(int i, int k) {
    if constexpr (K == 0) {
        return;
    } else {
        if (k - i == K) {
            remainder<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K, 0>(i);        
        } else {
            remainder_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K - 1>(i, k);
        }
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int K, int I>
void empty_shift1(int i) {  // K is the remain
    if constexpr (I == shift1) {
        return;
    } else {
        printf("empty_shift1: i = %d, I = %d\n", i, I);
        // op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i);
        op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + K) % maxshift_reg>(i - shift1);
        __pipeline_commit();
        op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + K) % maxshift_reg, shift1/*这里还有commit，所以还是shift1*/>(i - shift1);
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift3 + K) % maxshift_reg, (I + shift3 + K) % maxshift_reg>(i - shift1 - shift2);

        
        op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + K) % maxshift_reg, (I + shift3 + shift2 + K) % maxshift_reg>(i - shift1);
        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (I + shift3 + K) % maxshift_reg, shift2>(i - shift1 - shift2);
        op_mma<maxshift_smem, maxshift_reg, (I + K) % maxshift_reg, (I + K) % maxshift_reg>(i - shift1 - shift2 - shift3);

        empty_shift1<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K, I + 1>(i + 1);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int K>
void empty_shift1_switch(int i, int remain) {
    if constexpr (K < 0) {
        return;
    } else {
        if (K == remain) {
            empty_shift1<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K, 0>(i);
        } else {
            empty_shift1_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K - 1>(i, remain);
        }
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int K, int I>
void empty_shift2(int i) {
    if constexpr (I == shift2) {
        return;
    } else {
        printf("empty shift2: i = %d, I = %d\n", i, I);
    
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift1 + shift3 + K) % maxshift_reg, (I + shift1 + shift3 + K) % maxshift_reg>(i - shift1 - shift2);

        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (I + shift1 + shift3 + K) % maxshift_reg, shift2 - (I + 1)>(i - shift1 - shift2);
        op_mma<maxshift_smem, maxshift_reg, (I + shift1 + K) % maxshift_reg, (I + shift1 + K) % maxshift_reg>(i - shift1 - shift2 - shift3);

        empty_shift2<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K, I + 1>(i + 1);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int K>
void empty_shift2_switch(int i, int remain) {
    if constexpr (K < 0) {
        return;
        
    } else {
        if (remain == K) {
            empty_shift2<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K, 0>(i);
        } else {
            empty_shift2_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K - 1>(i, remain);
        }
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int K, int I>
void empty_shift3(int i) {
    if constexpr (I == shift3) {
        return;
    } else {
        printf("empty shift3: i = %d, I = %d\n", i, I);

        op_mma<maxshift_smem, maxshift_reg, (I + shift1 + shift2 + K) % maxshift_reg, (I + shift1 + shift2 + K) % maxshift_reg>(i - shift1 - shift2 - shift3);
        empty_shift3<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K, I + 1>(i + 1);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int K>
void empty_shift3_switch(int i, int remain) {
    if constexpr (K < 0) {
        return;
    } else {
        if (K == remain) {
            empty_shift3<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K, 0>(i);
        } else {
            empty_shift3_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, K - 1>(i, remain);
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
    // shift 3
    // tile A [16 * 16], tile B [16 * 32]
    half REGB[2 * (TILE_B / 8) * maxshift_reg];
    half REGA[4 * maxshift_reg];
    float REGC[4 * (TILE_B / 8)];

    int l = dval_soff[block_idx_x];
    int r = dval_soff[block_idx_x + 1];

    int k = l - r;

    if (k <= shift1 + shift2 + shift3) {
        pipeline_switch<shift1, shift2, shift3, maxshift_reg, maxshift_smem, shift1 + shift2 + shift3>(k);
    } else {
        fill_shift1<shift1, shift2, shift3, maxshift_smem, maxshift_reg, 0>();
        fill_shift2<shift1, shift2, shift3, maxshift_smem, maxshift_reg, 0>();
        fill_shift3<shift1, shift2, shift3, maxshift_smem, maxshift_reg, 0>();
        int i;
        for (i = shift1 + shift2 + shift3; i + maxshift_reg <= k; i += maxshift_reg) {
            loop_step<shift1, shift2, shift3, maxshift_smem, maxshift_reg, 0>(i);
        }
        int remain = k - i;
        remainder_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, maxshift_reg - 1>(i, k);
        i = k;
        empty_shift1_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, maxshift_reg - 1>(i, remain);
        i = k + shift1;
        empty_shift2_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, maxshift_reg - 1>(i, remain);
        i = k + shift1 + shift2;
        empty_shift3_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, maxshift_reg - 1>(i, remain);
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
    //! （shift1=1,shift2=1,shift3=1)
    dtc_spmm_kernel_sstage_rstage<1, 1, 1, 32>(M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, val_coo_val, dval_coo_off, dC);

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

    CHECK_CUDA2(cudaMalloc(&dval_sidx, sizeof(int) * TCBlock_num * TC_BLOCK_K));
    CHECK_CUDA2(cudaMalloc(&dval_soff, sizeof(int) * row_window_num));
    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(half) * K * N));
    CHECK_CUDA2(cudaMalloc(&dval_coo_idx, sizeof(int) * nnz));
    CHECK_CUDA2(cudaMalloc(&dval_coo_val, sizeof(half) * nnz));
    CHECK_CUDA2(cudaMalloc(&dval_coo_off, sizeof(int) * (TCBlock_num + 1)));
    CHECK_CUDA2(cudaMalloc(&dC, sizeof(float) * M * N));

    CHECK_CUDA2(cudaMemcpy(dval_sidx, val_sidx, sizeof(int) * TCBlock_num * TC_BLOCK_K, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_soff, val_soff, sizeof(int) * row_window_num, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(half) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_coo_idx, val_coo_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_coo_val, val_coo_val, sizeof(half) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_coo_off, val_coo_off, sizeof(int) * (TCBlock_num + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * M * N));

    int threadblock_num_x = row_window_num;
    int threadblock_num_y = N / TILE_B;
    int thread_num = 32;

    ncu_test_all<TILE_B>(threadblock_num_x, threadblock_num_y, thread_num, M, N, K, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off);

    CHECK_CUDA2(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

}