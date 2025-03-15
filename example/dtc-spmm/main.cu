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

template <int TILE_B>
extern void dtc_spmm_test(
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
);

extern template void dtc_spmm_test<32>(
    int, int, int, 
    int, int, 
    int*, int*, half*, 
    int*, half*, 
    int*, float*
);

void initVector(half* vec, int length) {
    for (int i = 0; i < length; i++) {
        vec[i] = (half)(rand() % 5 + 1);
    }
}

void initVector(half* vec, int length, float sparsity) {
    if (sparsity < 0.0f || sparsity > 1.0f) return; 
    auto seed = std::time(nullptr);
    std::srand(1741962880); 
    cout << seed << endl;
    // std::srand(1);
    for (int i = 0; i < length; ++i) {
        if (static_cast<float>(rand()) / RAND_MAX < sparsity) {
            vec[i] = half(0.0f); 
        } else {
            // vec[i] = half(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f); 
            vec[i] = half(rand() % 5 + 1);
        }
    }
}

#define ROW_WINDOW_M 16
#define TC_BLOCK_K 16
#define TC_BLOCK_M 16

int binarySearch(int* arr, int l, int r, int target) {
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) l = mid + 1;
        else r = mid - 1;
    }
    return -1; // 未找到
}

void convertToMETCF(
    int M, int N, int K, 
    int nnz, int* csrPtrA, int* csrColIdxA, half* csrValA,
    half* B, float* C
) {
    // convert to ME-TCF
    int row_window_num = (M + ROW_WINDOW_M- 1) / ROW_WINDOW_M;
    // ok: val_soff is the value that record TC Block in a Row Window, function just like a Ptr array;
    int* val_soff = (int*)malloc(sizeof(int) * (row_window_num + 1));
    memset(val_soff, 0, sizeof(int) * (row_window_num + 1));
    int TCBlock_num = 0;
    int* nnz_col = (int*)malloc(sizeof(int) * K);
    vector<vector<int>> vec_nnz_per_tc_block(row_window_num);
    for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
        val_soff[i_row_window] = TCBlock_num;
        memset(nnz_col, 0, sizeof(int) * K);
        vector<int>& nnz_per_tc_block = vec_nnz_per_tc_block[i_row_window];
        for (int i_row = 0; i_row < ROW_WINDOW_M; i_row++) {
            int row = i_row_window * ROW_WINDOW_M + i_row;
            for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                nnz_col[csrColIdxA[j]] = 1;
            }   
        }
        int nnz_col_num = 0;
        for (int i = 0, j = 0; i < K; i++) {
            if (nnz_col[i] == 1) {
                nnz_col[j] = i;
                j++;
                nnz_col_num++;
            }
        }
        TCBlock_num += (nnz_col_num + TC_BLOCK_K - 1) / TC_BLOCK_K;
        nnz_per_tc_block.resize(TCBlock_num, 0);
        for (int i_row = 0; i_row < ROW_WINDOW_M; i_row++) {
            int row = i_row_window * ROW_WINDOW_M + i_row;
            for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                int col = csrColIdxA[j];
                int condensed_col = binarySearch(nnz_col, 0, nnz_col_num - 1, col);
                int tc_block_id = condensed_col / TC_BLOCK_K;
                nnz_per_tc_block[tc_block_id]++;
            }
        }
    }
    val_soff[row_window_num] = TCBlock_num;
    printf("TCBlock_num = %d, nnz = %d\n", TCBlock_num);
    {//! debug
        printf("val_soff: ");
        for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
            printf("%d ", val_soff[i_row_window]);
        }
        printf("%d\n", val_soff[row_window_num]);
    }
    // ok: val_sidx -> 每个TC Block中非零列对应的column index
    int* val_sidx = (int*)malloc(sizeof(int) * TCBlock_num * TC_BLOCK_K);
    memset(val_sidx, 0, sizeof(int) * TCBlock_num * TC_BLOCK_K);
    // ok: val_coo_dix -> TC Block中每个非零元在TC Block中的位置
    int* val_coo_idx = (int*)malloc(sizeof(int) * nnz);
    memset(val_coo_idx, 0, sizeof(int) * nnz);
    // ok: val_coo_off -> TC Block中非零元个数，类似CSR的PTR array
    int* val_coo_off = (int*)malloc(sizeof(int) * (TCBlock_num + 1));
    memset(val_coo_off, 0, sizeof(int) * (TCBlock_num + 1));
    // ok: val_coo_val -> TC Block中的非零元
    half* val_coo_val = (half*)malloc(sizeof(half) * nnz);
    memset(val_coo_val, 0, sizeof(half) * nnz);
    // exit(0);
    // prefix sum 
    int cur_pos = 0;
    int i_tc_block_ptr = 0;
    for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
        vector<int> & nnz_per_tc_block = vec_nnz_per_tc_block[i_row_window];
        for (int i_tc_block = 0; i_tc_block < nnz_per_tc_block.size(); i_tc_block++) {
            int tmp = nnz_per_tc_block[i_tc_block];
            nnz_per_tc_block[i_tc_block] = cur_pos;
            val_coo_off[i_tc_block_ptr] = cur_pos;
            i_tc_block_ptr++;
            cur_pos += tmp;
        }
        nnz_per_tc_block.push_back(cur_pos);
    }
    {//! debug
        printf("final i_tc_block_ptr=%d, TCBlock_num=%d, cur_pos=%d\n", i_tc_block_ptr, TCBlock_num, cur_pos);
    }
    val_coo_off[i_tc_block_ptr] = cur_pos;
    {//! debug
        printf("nnz_per_tc_block: ");
        for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
            vector<int>& nnz_per_tc_block = vec_nnz_per_tc_block[i_row_window];
            for (int i_tc_block = 0; i_tc_block < nnz_per_tc_block.size(); i_tc_block++) {
                printf("%d ", nnz_per_tc_block[i_tc_block]);
            }
            printf("\n");
        }
    }
    // exit(0);
    int* tc_block_nnz_ptr = (int*)malloc(sizeof(int) * (K + TC_BLOCK_K - 1) / TC_BLOCK_K);
    for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
        int TC_block_offset = val_soff[i_row_window];
        int* tmp_val_sidx = val_sidx + TC_block_offset * TC_BLOCK_K;
        memset(nnz_col, 0, sizeof(int) * K);
        vector<int>& nnz_per_tc_block = vec_nnz_per_tc_block[i_row_window];
        for (int i_row = 0; i_row < ROW_WINDOW_M; i_row++) {
            int row = i_row_window * ROW_WINDOW_M + i_row;
            for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                nnz_col[csrColIdxA[j]] = 1;
            }
        }
        int nnz_col_num = 0;
        for (int i = 0, j = 0; i < K; i++) {
            if (nnz_col[i] == 1) {
                nnz_col[j] = i;
                j++;
                nnz_col_num++;
            }
        }
        {//! debug
            printf("nnz_col: ");
            for (int i = 0; i < nnz_col_num; i++) {
                printf("%d ", nnz_col[i]);                
            } printf("\n");
        }
        TCBlock_num = (nnz_col_num + TC_BLOCK_K - 1) / TC_BLOCK_K;
        for (int i_tc_block_col = 0; i_tc_block_col < nnz_col_num; i_tc_block_col++) {
            tmp_val_sidx[i_tc_block_col] = nnz_col[i_tc_block_col];
        }

        memset(tc_block_nnz_ptr, 0, sizeof(int) * (K + TC_BLOCK_K - 1) / TC_BLOCK_K);
        for (int i_tc_block = 0; i_tc_block < TCBlock_num; i_tc_block++) {
            tc_block_nnz_ptr[i_tc_block] = nnz_per_tc_block[i_tc_block];
        }
        
        for (int i_row = 0; i_row < ROW_WINDOW_M; i_row++) {
            int row = i_row_window * ROW_WINDOW_M + i_row;
            int col_start_pos = 0;
            for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                int col = csrColIdxA[j];
                float value = (float)csrValA[j];
                int condensed_pos = binarySearch(nnz_col, col_start_pos, nnz_col_num - 1, col);
                col_start_pos = condensed_pos + 1;
                int tc_block_id = condensed_pos / TC_BLOCK_K;
                val_coo_val[tc_block_nnz_ptr[tc_block_id]] = (half)value;
                val_coo_idx[tc_block_nnz_ptr[tc_block_id]] = i_row * TC_BLOCK_K + (condensed_pos % TC_BLOCK_K);
                tc_block_nnz_ptr[tc_block_id]++;
                printf("row=%d,col=%d,val=%.0f,idx=%d,condensed_pos=%d,tc_block_id=%d,col_start_pos=%d\n", row, col, value, i_row * TC_BLOCK_K + (condensed_pos % TC_BLOCK_K), condensed_pos, tc_block_id, col_start_pos);
            }
        }
    }
    // exit(0);
    // input to cuda func
    dtc_spmm_test<32>(M, N, K, TCBlock_num, nnz, val_sidx, val_soff, B, val_coo_idx, val_coo_val, val_coo_off, C);
}

int main() {
    int M = 16;
    int N = 32;
    int K = 32;
    half* B = (half*)malloc(sizeof(half) * K * N);
    half* A = (half*)malloc(sizeof(half) * M * K);
    
    memset(A, 0, sizeof(half) * M * K);
    memset(B, 0, sizeof(half) * K * N);
    float sparsity = 0.5;
    initVector(B, K * N);
    initVector(A, M * K, sparsity);
    printf("A: \n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%.0f ", (float)A[i * K + j]);
        } printf("\n");
    }
    printf("\n\nB: \n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.0f ", (float)B[j * K + i]);
        } printf("\n");
    }

    // convert A to CSR
    int nnz = 0;
    int* csrPtrA = (int*)malloc(sizeof(int) * (M + 1));
    
    for (int i = 0; i < M; i++) {
        csrPtrA[i] = nnz;
        for (int k = 0; k < K; k++) {
            float value = (float)A[i * K + k];
            if (value != 0) {
                nnz++;
            }
        }
    } 
    csrPtrA[M] = nnz;
    int* csrColIdxA = (int*)malloc(sizeof(int) * nnz);
    half* csrValA = (half*)malloc(sizeof(half) * nnz);
    memset(csrColIdxA, 0, sizeof(int) * nnz);
    memset(csrValA, 0, sizeof(half) * nnz);
    int ptr = 0;
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float value = (float)A[i * K + k];
            if (value != 0) {
                csrColIdxA[ptr] = k;
                csrValA[ptr] = (half)value;
                ptr++;
            }
        }
    }
    {//! debug
        printf("csrPtrA: ");
        for (int i = 0; i <= M; i++) {
            printf("%d ", csrPtrA[i]);
        }
        printf("\n");
        printf("csrColIdxA: ");
        for (int i = 0; i < nnz; i++) {
            printf("%d ", csrColIdxA[i]);
        } printf("\n");
        printf("csrValA: ");
        for (int i = 0; i < nnz; i++) {
            printf("%.0f ", (float)csrValA[i]);
        } printf("\n");
    }
    float* C_cuda = (float*)malloc(sizeof(float) * M * N);
    convertToMETCF(M, N, K, nnz, csrPtrA, csrColIdxA, csrValA, B, C_cuda);

    //host check
    float* C_cpu = (float*)malloc(sizeof(float) * M * N);
    memset(C_cpu, 0, sizeof(float) * M * N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = csrPtrA[i]; k < csrPtrA[i + 1]; k++) {
                int col = csrColIdxA[k];
                float value = (float)csrValA[k];
                C_cpu[i * N + j] += value * (float)B[k * N + j];
            }
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(C_cpu[i * N + j] - C_cuda[i * N + j]) > 0.01) {
                printf("Error(%d, %d): %f %f\n", i, j, C_cpu[i * N + j], C_cuda[i * N + j]);
            }
        }
    }
    return 0;
}