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

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))
#define OFFSET_COL(row, col, lda) ((col) * (lda) + (row))

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
    int total_TCBlock_num = 0;
    int* nnz_col = (int*)malloc(sizeof(int) * K);
    vector<vector<int>> vec_nnz_per_tc_block(row_window_num);
    int nnz_pad = 0;
    for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
        val_soff[i_row_window] = total_TCBlock_num;
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
        total_TCBlock_num += (nnz_col_num + TC_BLOCK_K - 1) / TC_BLOCK_K;
        int TCBlock_num = (nnz_col_num + TC_BLOCK_K - 1) / TC_BLOCK_K;
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
        //! coo val and idx load in 8 elements group
        for (int i_tc_block = 0; i_tc_block < TCBlock_num; i_tc_block++) {
            int nnz_block = nnz_per_tc_block[i_tc_block];
            // nnz_per_tc_block[i_tc_block] = (nnz_block + 8 - 1) / 8;
            nnz_pad += (nnz_block + 8 - 1) / 8 * 8;
        }
    }
    val_soff[row_window_num] = total_TCBlock_num;
    printf("total_TCBlock_num = %d, nnz = %d, nnz_pad=%d\n", total_TCBlock_num, nnz, nnz_pad);

    // ok: val_sidx -> 每个TC Block中非零列对应的column index
    int* val_sidx = (int*)malloc(sizeof(int) * total_TCBlock_num * TC_BLOCK_K);
    memset(val_sidx, 0, sizeof(int) * total_TCBlock_num * TC_BLOCK_K);
    // ok: val_coo_dix -> TC Block中每个非零元在TC Block中的位置
    int* val_coo_idx = (int*)malloc(sizeof(int) * nnz_pad);
    memset(val_coo_idx, 0, sizeof(int) * nnz_pad);
    // val_coo_off -> TC Block 中非零元个数，每个block有两个范围，第一个范围记录起始index，第二个位置记录实际长度，但是每个TC Block都pad0了
    int* val_coo_off = (int*)malloc(sizeof(int) * (total_TCBlock_num * 2 + 1));
    memset(val_coo_off, 0, sizeof(int) * (total_TCBlock_num * 2 + 1));
    // ok: val_coo_val -> TC Block中的非零元
    half* val_coo_val = (half*)malloc(sizeof(half) * nnz_pad);
    memset(val_coo_val, 0, sizeof(half) * nnz_pad);

    // prefix sum 
    int cur_pos = 0;
    int i_tc_block_ptr = 0;
    for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
        vector<int> & nnz_per_tc_block = vec_nnz_per_tc_block[i_row_window];
        for (int i_tc_block = 0; i_tc_block < nnz_per_tc_block.size(); i_tc_block++) {
            int tmp = nnz_per_tc_block[i_tc_block];
            nnz_per_tc_block[i_tc_block] = cur_pos;
            val_coo_off[i_tc_block_ptr * 2] = cur_pos;
            val_coo_off[i_tc_block_ptr * 2 + 1] = cur_pos + tmp;
            i_tc_block_ptr++;
            cur_pos += (tmp + 8 - 1) / 8 * 8;
        }
        nnz_per_tc_block.push_back(cur_pos);
    }
    val_coo_off[2 * total_TCBlock_num] = cur_pos;

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

        int tc_block_num = (nnz_col_num + TC_BLOCK_K - 1) / TC_BLOCK_K;
        for (int i_tc_block_col = 0; i_tc_block_col < nnz_col_num; i_tc_block_col++) {
            tmp_val_sidx[i_tc_block_col] = nnz_col[i_tc_block_col];
        }

        memset(tc_block_nnz_ptr, 0, sizeof(int) * (K + TC_BLOCK_K - 1) / TC_BLOCK_K);
        for (int i_tc_block = 0; i_tc_block < tc_block_num; i_tc_block++) {
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
            }
        }
    }
    // exit(0);
    // input to cuda func
    dtc_spmm_test<32>(M, N, K, total_TCBlock_num, nnz_pad, val_sidx, val_soff, B, val_coo_idx, val_coo_val, val_coo_off, C);
}

void test(int M, int N, int K, bool check) {
    half* B = (half*)malloc(sizeof(half) * K * N);
    half* A = (half*)malloc(sizeof(half) * M * K);
    
    memset(A, 0, sizeof(half) * M * K);
    memset(B, 0, sizeof(half) * K * N);
    float sparsity = 0.5;
    initVector(B, K * N);
    initVector(A, M * K, sparsity);
    // printf("A: \n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < K; j++) {
    //         if (j % 16 == 0)
    //             printf("| %.0f ", (float)A[i * K + j]);
    //         else
    //             printf("%.0f ", (float)A[i * K + j]);
    //     } printf("\n");
    // }
    // printf("\n\nB: \n");
    // for (int i = 0; i < K; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%.0f ", (float)B[j * K + i]);
    //     } printf("\n");
    // }
    {//! debug check C
        // float* C_check = (float*)malloc(sizeof(float) * 16 * 8);
        // memset(C_check, 0, sizeof(float) * 16 * 8);
        // for (int k_ptr = 0; k_ptr < K / 16; k_ptr++) {
        //     printf("k_ptr = %d\n", k_ptr);
        //     for (int i = 0; i < 16; i++) {
        //         for (int j = 0; j < 8; j++) {
        //             for (int k = 0; k < 16; k++) {
        //                 C_check[OFFSET_ROW(i, j, 8)] += (float)A[i * K + (k_ptr * 16 + k)] * (float)B[j * K + (k_ptr * 16 + k)];
        //             }
        //         }
        //     }
        //     for (int i = 0; i < 16; i++) {
        //         for (int j = 0; j < 8; j++) {
        //             printf("%.0f ", C_check[OFFSET_ROW(i, j, 8)]);
        //         } printf("\n");
        //     }
        // }

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
        // printf("csrPtrA: ");
        // for (int i = 0; i <= M; i++) {
        //     printf("%d ", csrPtrA[i]);
        // }
        // printf("\n");
        // printf("csrColIdxA: ");
        // for (int i = 0; i < nnz; i++) {
        //     printf("%d ", csrColIdxA[i]);
        // } printf("\n");
        // printf("csrValA: ");
        // for (int i = 0; i < nnz; i++) {
        //     printf("%.0f ", (float)csrValA[i]);
        // } printf("\n");
    }
    float* C_cuda = (float*)malloc(sizeof(float) * M * N);
    convertToMETCF(M, N, K, nnz, csrPtrA, csrColIdxA, csrValA, B, C_cuda);

    //host check
    if (check) {
        float* C_cpu = (float*)malloc(sizeof(float) * M * N);
        memset(C_cpu, 0, sizeof(float) * M * N);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = csrPtrA[i]; k < csrPtrA[i + 1]; k++) {
                    int col = csrColIdxA[k];
                    float value = (float)csrValA[k];
                    C_cpu[i * N + j] += value * (float)B[j * K + col];
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
    }
}

int main() {
    test(2304, 1536, 16384, false);
    test(1024, 1024, 1024, true);
    return 0;
}