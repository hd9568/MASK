#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
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
template <int maxshift_smem, int maxshift_reg, int r1, int wait_num>
void op_S2R_val_sidx_load(int i) {
    printf("op_S2R_val_sidx_load(%d): wait(%d), sval_sidx[%d] -> val_sidx_reg[%d]\n", i, wait_num, i % maxshift_smem, r1);
}

template <int maxshift_smem, int maxshift_reg, int r1>
void op_G2S_coo_atomic_format_load_val(int i) {
    printf("op_G2S_coo_atomic_format_load_val(%d): dval_coo_val[coo_scope[%d]] -> tile_coo_val_block[%d]\n", i, r1, i % maxshift_smem);
}

template <int maxshift_smem, int maxshift_reg, int r1, int r2>
void op_G2R_coo_atomic_format_load_idx(int i) {
    printf("op_G2R_coo_atomic_format_load_idx(%d): dval_coo_idx[coo_scope[%d]] -> coo_idx_reg[%d]\n", i, r1, r2);
}

// shift 3
template <int maxshift_smem, int maxshift_reg, int r1, int r2>
void op_G2R_B_val_load(int i) {
    printf("op_G2R_B_val_load(%d): dB_val[val_sidx_reg[%d]] -> REGB[%d]\n", i, r1, r2);
}

template <int maxshift_smem, int maxshift_reg, int r1, int wait_num>
void op_S2R_coo_atomic_val_restore(int i) {
    printf("op_S2R_coo_atomic_val_restore(%d): wait(%d), tile_coo_val_block[%d] -> restore[%d] -> REGA[%d]\n", i, wait_num, i % maxshift_smem, i % maxshift_smem, r1);
}

template <int maxshift_smem, int maxshift_reg, int r1, int r2>
void op_mma(int i) {
    printf("op_mma(%d): REGA[%d] x REGB[%d] -> REGC\n", i, r1, r2);
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_reg, int maxshift_smem, int K, int i>
void pipeline() {
    printf("small pipeline %d\n", i);
    if constexpr (i < K) {
        op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i);

        op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, i % maxshift_reg>(i);
        if constexpr (i < shift1) {
            printf("__pipeline_commit();\n");
        }
    }
    // (sstage > (K - 1 - (i - sstage)) ? K - 1 - (i - sstage) : sstage)
    if constexpr (0 <= i - shift1 && i - shift1 < K) {
        op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg, (i - shift1) % maxshift_reg>(i - shift1);
        printf("__pipeline_commit();\n");
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
        printf("__pipeline_commit();\n");
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
        printf("__pipeline_commit();\n");
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
        printf("__pipeline_commit();\n");
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
    printf("__pipeline_commit();\n");
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
        printf("__pipeline_commit();\n");
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
        printf("__pipeline_commit();\n");
        op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + K) % maxshift_reg, shift1/*这里还有commit，所以还是shift1*/>(i - shift1);
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift3 + K) % maxshift_reg, (I + shift3 + K) % maxshift_reg>(i - shift1 - shift2);

        // op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + shift1) % maxshift_reg>(i);
        
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
        // op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i);
        // op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + K) % maxshift_reg>(i - shift1);
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift1 + shift3 + K) % maxshift_reg, (I + shift1 + shift3 + K) % maxshift_reg>(i - shift1 - shift2);

        // op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + shift1) % maxshift_reg>(i);
        // op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg>(i - shift1);
        // op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + K) % maxshift_reg>(i - shift1);
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

        // op_G2S_val_sidx_load<maxshift_smem, maxshift_reg>(i);
        // op_S2R_val_sidx_load<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + K) % maxshift_reg>(i - shift1);
        // op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift1 + shift3 + K) % maxshift_reg>(i - shift1 - shift2);

        // op_G2R_coo_atomic_format_load_off<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + shift1) % maxshift_reg>(i);
        // op_G2S_coo_atomic_format_load_val<maxshift_smem, maxshift_reg>(i - shift1);
        // op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + K) % maxshift_reg>(i - shift1);
        // op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (I + shift1 + shift3 + K) % maxshift_reg>(i - shift1 - shift2);
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

template <int shift1, int shift2, int shift3>
void spmm(int k) {
    constexpr const int maxshift1_2 = (shift1 > shift2 ? shift1 : shift2);
    constexpr const int maxshift_reg = (maxshift1_2 > shift3 ? maxshift1_2 : shift3) + 1;
    constexpr const int maxshift_smem = maxshift1_2 + 1;

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
}


int main() {
    for (int k = 0; k < 20; k++) {
        printf("k = %d\n", k);
        spmm<1, 1, 1>(k);
        printf("\n\n");
    }
    return 0;
}