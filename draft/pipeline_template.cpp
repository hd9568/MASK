void op_s(int i); // R1 -> R2
void op_r(int i); // R2 -> R3
void op_mma(int i); // R3, R4 -> R4
// eg: op_s(0): R1[0] -> R2[0]
//     op_s(1): R1[1] -> R2[1]
//     op_s(2): R1[2] -> R2[2]
//     op_s(3): R1[3] -> R2[3]; op_r(0): R2[0] -> R3[0]
//     op_s(4): R1[0] -> R2[0]; op_r(1): R2[1] -> R3[1]
//     op_s(5): R1[1] -> R2[1]; op_r(2): R2[2] -> R3[2]; op_mma: R3[0], R4 -> R4

template<size_t sstage, size_t rstage>
fill_sstage();

template<size_t sstage, size_t rstage>
fill_rstage();

// =====================================================================================

template<size_t sstage, size_t rstage, size_t K, size_t i> // assume sstage-- ,rstage--
pipeline() {
    if constexpr (i < K) {
        op_s(i);
    }
    if constexpr (0 <= i - sstage && i - sstage <= K) {
        op_r(i - sstage);
    }
    if constexpr (0 <= i - sstage - rstage && - i sstage - rstage <= K) {
        op_mma(i - sstage - rstage);
    }
    if constexpr (i < K) {
        pipeline<sstage, rstage, K, i + 1>();
    }
}

template<size_t sstage, size_t rstage, size_t K>
pipeline_switch(int k) {
    if constexpr (K == 0) {
        return;
    }
    if (k == K) {
        pipeline<sstage, rstage, K, 0>();
    } else {
        pipeline_switch<sstage, rstage, K - 1>(k);
    }

}

// =====================================================================================

template<size_t sstage, size_t rstage, size_t maxstage, size_t I>
loop_step(int i) {
    op_s(i, R1[(sstage+rstage+I)%maxstage], R2[(sstage+rstage+I)%maxstage]);
    op_r(i - sstage, R2[(rstage+I)%maxstage], R3[(rstage+I)%maxstage]);
    op_mma(i - sstage - rstage, R3[0+I], R4);

    if constexpr (I < maxstage) {
        loop_step<sstage, rstage, maxstage, I + 1>(i + 1);
    }
}

// =====================================================================================

template<size_t sstage, size_t rstage, size_t maxstage, size_t K, size_t I>
remainder(int i) {
    if constexpr (I == K) return;

    op_s(i, R1[(sstage+rstage+I)%maxstage], R2[(sstage+rstage+I)%maxstage]);
    op_r(i - sstage, R2[(rstage+I)%maxstage], R3[(rstage+I)%maxstage]);
    op_mma(i - sstage - rstage, R3[0+I], R4);

    remainder<sstage, rstage, maxstage, K, I + 1>(i + 1);
}

template<size_t sstage, size_t rstage, size_t maxstage, size_t K>
remainder_switch(int i, int k) {
    if constexpr (K == 0) return;

    if (k - i == K) {
        remainder<sstage, rstage, maxstage, K, 0>(i);
    } else {
        remainder_switch<sstage, rstage, maxstage, K - 1>(i, k);
    }
}

// =====================================================================================

template<size_t sstage, size_t rstage>
empty_sstage();

template<size_t sstage, size_t rstage>
empty_rstage();

// =====================================================================================

template<size_t sstage, size_t rstage>
spmm(int k) {
    if (k < sstage+rstage) {
        pipeline_switch<sstage, rstage, sstage+rstage>(k);
    } else {
        fill_sstage();
        fill_rstage();
        constexpr int maxstage = max(sstage, rstage);
        for (int i = sstage+rstage; i + maxstage <= k; i += maxstage) {
            loop_step<sstage, rstage, maxstage, 0>();
        }
        remainder_switch<sstage, rstage, maxstage, maxstage - 1>(i, k);
        empty_sstage();
        empty_rstage();
    }
}






for (int i = 0; i < m; i++) {
    op(ptr, i);
}

switch(i) {
    case 0: 
        break;
    case 1: 
        op(ptr, 0); 
        break;
    case 2:
        op(ptr, 0);
        op(ptr, 1);
        break;
    case 3:
        op(ptr, 0);
        op(ptr, 1);
        op(ptr, 2);
        break;
    case 4:
        op(ptr, 0);
        op(ptr, 1);
        op(ptr, 2);
        op(ptr, 3);
        break;
}