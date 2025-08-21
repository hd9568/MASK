#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "common.h"
#include "utils.h"

#include <cute/tensor.hpp>

using namespace std;
using namespace cute;

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))
#define OFFSET_COL(row, col, lda) ((col) * (lda) + (row))

#define ROW_WINDOW_M 16
#define TC_BLOCK_K 8
#define TC_BLOCK_M 16

template <int TILE_B>
extern void dtc_spmm_test_val_sidx_bind_reorder(int M,
                                                int N,
                                                int K,
                                                int TCBlock_num,
                                                int nnz,
                                                int* val_sidx,
                                                int* val_soff,
                                                float* B_val,
                                                int* val_coo_idx,
                                                float* val_coo_val,
                                                int* val_coo_off,

                                                int row_block_num,
                                                int slb_block_dim_size,
                                                vector<int> row_block_offset,
                                                vector<int> row_block_to_row_window_id,
                                                vector<int> slb_tb_offset,

                                                float* C,
                                                bool is_ncu_test,
                                                int warmup_time,
                                                int execute_time);

extern template void dtc_spmm_test_val_sidx_bind_reorder<64>(int M,
                                                             int N,
                                                             int K,
                                                             int TCBlock_num,
                                                             int nnz,
                                                             int* val_sidx,
                                                             int* val_soff,
                                                             float* B_val,
                                                             int* val_coo_idx,
                                                             float* val_coo_val,
                                                             int* val_coo_off,

                                                             int row_block_num,
                                                             int slb_block_dim_size,
                                                             vector<int> row_block_offset,
                                                             vector<int> row_block_to_row_window_id,
                                                             vector<int> slb_tb_offset,

                                                             float* C,
                                                             bool is_ncu_test,
                                                             int warmup_time,
                                                             int execute_time);

int binarySearch(int* arr, int l, int r, int target) {
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == target)
            return mid;
        if (arr[mid] < target)
            l = mid + 1;
        else
            r = mid - 1;
    }
    return -1; // 未找到
}

void initVector(float* vec, int length, float sparsity) {
    if (sparsity < 0.0f || sparsity > 1.0f)
        return;
    auto seed = std::time(nullptr);
    std::srand(1234);
    cout << seed << endl;
    // std::srand(1);
    for (int i = 0; i < length; ++i) {
        if (static_cast<float>(rand()) / RAND_MAX < sparsity) {
            vec[i] = float(0.0f);
        } else {
            // vec[i] = half(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
            vec[i] = float(rand() % 5 + 1);
        }
    }
}

void strict_load_balance(
    int M, int N, int K, 
    int total_TCBlock_num, 
    int* val_soff, 
    int& row_block_num, 
    int& slb_block_dim_size, 
    vector<int>& row_block_offset,
    vector<int>& row_block_to_row_window_id,
    vector<int>& slb_tb_offset
) {
    int thread_block_workload = 64;
    printf("total_tcblock_num = %d\n", total_TCBlock_num);
    slb_block_dim_size = (total_TCBlock_num + thread_block_workload - 1) / thread_block_workload;

    // slb_tb_offset.resize(slb_block_dim_size + 1, 0);

    int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;
    int row_window_ptr = 0;
    
    int current_workload = 0;
    // current row block start position
    int row_block_start = -1;
    // ptr for update slb_tb_off
    int current_row_block_ptr = 0;
    int tc_block_ptr = 0;
    
    slb_tb_offset.push_back(current_row_block_ptr);

    //! DEBUG
    // for (int row_window_i = 0; row_window_i < 20; row_window_i++) {
    //     printf("row_window_i = %d, tc block = %d\n", row_window_i, val_soff[row_window_i + 1] - val_soff[row_window_i]);
    // }
    
    
    for (int row_window_i = 0; row_window_i < row_window_num; row_window_i++) {
        int current_row_window_workload = val_soff[row_window_i + 1] - val_soff[row_window_i];
        if (current_row_window_workload == 0) continue;
        if (row_block_start == -1) {
            // current row block not start
            row_block_start = 0;
        }

        while (current_row_window_workload > 0) {
            if (current_workload + current_row_window_workload < thread_block_workload) {
                // current row window generate a row block
                {
                    //! debug
                    // printf("---row window %d, ptr = %d\n", row_window_i, tc_block_ptr);
                }
                row_block_offset.push_back(tc_block_ptr);
                row_block_to_row_window_id.push_back(row_window_i);

                // update ptr
                tc_block_ptr += current_row_window_workload;

                // update row block num
                current_row_block_ptr++;
                // new row block start in next row window
                row_block_start = -1;
                // current_workload += current_row_window_workload
                current_workload += current_row_window_workload;
                
                
                current_row_window_workload = 0;
            } else if (current_workload + current_row_window_workload == thread_block_workload) {
                // current row window generate a row block
                {
                    //! debug
                    // printf("---row window %d, ptr = %d\n", row_window_i, tc_block_ptr);
                }
                row_block_offset.push_back(tc_block_ptr);
                row_block_to_row_window_id.push_back(row_window_i);

                // update ptr
                tc_block_ptr += current_row_window_workload;
                
                // update row block num
                current_row_block_ptr++;
                // new row block start in next row window
                row_block_start = -1;
                // current_workload turn to 0
                current_workload = 0;
                current_row_window_workload = 0;
                // turn to next thread block
                slb_tb_offset.push_back(current_row_block_ptr);
                {//! DEBUG
                    // int current_slb_tb_offset_size = slb_tb_offset.size();
                    // printf("slb_tb_offset(%d) push a new value %d: current thread block from %d to %d\n", 
                    //     current_slb_tb_offset_size,
                    //     current_row_block_ptr, 
                    //     slb_tb_offset[current_slb_tb_offset_size - 2], 
                    //     slb_tb_offset[current_slb_tb_offset_size - 1]);
                    // for (int row_block_id = slb_tb_offset[current_slb_tb_offset_size - 2]; row_block_id < slb_tb_offset[current_slb_tb_offset_size - 1]; row_block_id++) {
                    //     if (row_block_id == slb_tb_offset[current_slb_tb_offset_size - 1] - 1) {
                    //         printf("\trow block %5d, tc block num: %d\n", row_block_id, tc_block_ptr - row_block_offset[row_block_id]);
                    //     }
                    //     printf("\trow block %5d, tc block num: %d\n", row_block_id, row_block_offset[row_block_id + 1] - row_block_offset[row_block_id]);
                    // }
                }
                
            } else {
                //> current_workload + current_row_window_workload > thread_block_workload
                // current row window split, generate a row block
                {
                    //! debug
                    // printf("---row window %d, ptr = %d\n", row_window_i, tc_block_ptr);
                }
                row_block_offset.push_back(tc_block_ptr);
                row_block_to_row_window_id.push_back(row_window_i);
                // update row block num
                current_row_block_ptr++;
                // new row block start at the split point
                int consume_tc_block_num = (thread_block_workload - current_workload);
                row_block_start += consume_tc_block_num;
                current_row_window_workload -= consume_tc_block_num;
                // current_workload turn to 0
                current_workload = 0;
                // turn to next thread block
                slb_tb_offset.push_back(current_row_block_ptr);
                {//! DEBUG
                    // int current_slb_tb_offset_size = slb_tb_offset.size();
                    // printf("slb_tb_offset(%d) push a new value %d: current thread block from %d to %d\n", 
                    //     current_slb_tb_offset_size,
                    //     current_row_block_ptr, 
                    //     slb_tb_offset[current_slb_tb_offset_size - 2], 
                    //     slb_tb_offset[current_slb_tb_offset_size - 1]);
                    // for (int row_block_id = slb_tb_offset[current_slb_tb_offset_size - 2]; row_block_id < slb_tb_offset[current_slb_tb_offset_size - 1]; row_block_id++) {
                    //     if (row_block_id == slb_tb_offset[current_slb_tb_offset_size - 1] - 1) {
                    //         printf("\trow block %5d, tc block num: %d\n", row_block_id, tc_block_ptr - row_block_offset[row_block_id]);
                    //     }
                    //     printf("\trow block %5d, tc block num: %d\n", row_block_id, row_block_offset[row_block_id + 1] - row_block_offset[row_block_id]);
                    // }
                }
                tc_block_ptr += consume_tc_block_num;
            }
        }        
    }
    // row block end
    row_block_offset.push_back(tc_block_ptr);
    // row_block_to_row_window_id.push_back(row_window_num - 1);
    if (total_TCBlock_num % thread_block_workload != 0) {
        current_row_block_ptr++;
        slb_tb_offset.push_back(current_row_block_ptr);
    }
    {//! DEBUG
        int current_slb_tb_offset_size = slb_tb_offset.size();
        printf("slb_tb_offset(%d) push a new value %d: current thread block from %d to %d\n", 
            current_slb_tb_offset_size,
            current_row_block_ptr, 
            slb_tb_offset[current_slb_tb_offset_size - 2], 
            slb_tb_offset[current_slb_tb_offset_size - 1]);
    }

    printf("slb_tb_offset size = %d\n", slb_tb_offset.size());
    printf("slb_block_dim_size = %d\n", slb_block_dim_size);
    assert(slb_tb_offset.size() == slb_block_dim_size + 1);
    printf("row_block_offset size = %d\n", row_block_offset.size());
    printf("row_block_to_row_window_id size = %d\n", row_block_to_row_window_id.size());
    assert(row_block_to_row_window_id.size() + 1 == row_block_offset.size());
    row_block_num = row_block_to_row_window_id.size();
    assert(row_block_offset[row_block_num] == val_soff[row_window_num]);
}

void convertToMETCF(int M,
                    int N,
                    int K,
                    int nnz,
                    int* csrPtrA,
                    int* csrColIdxA,
                    float* csrValA,
                    float* B,
                    float* C,
                    bool is_ncu_test,
                    int warmup_time,
                    int execute_time) {
    // convert to ME-TCF
    int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;
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
            if (row >= M)
                continue;
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
            if (row >= M)
                continue;
            for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                int col = csrColIdxA[j];
                int condensed_col = binarySearch(nnz_col, 0, nnz_col_num - 1, col);
                int tc_block_id = condensed_col / TC_BLOCK_K;
                nnz_per_tc_block[tc_block_id]++;
            }
        }
        //! coo val and idx load in 8 element group
        for (int i_tc_block = 0; i_tc_block < TCBlock_num; i_tc_block++) {
            int nnz_block = nnz_per_tc_block[i_tc_block];
            nnz_pad += (nnz_block + 8 - 1) / 8 * 8;
        }
    }
    val_soff[row_window_num] = total_TCBlock_num;
    printf("total_TCBlock_num = %d, nnz = %d, nnz_pad=%d\n", total_TCBlock_num, nnz, nnz_pad);

    //> val_sidx -> 每个 TC Block 中非零列对应的column index
    int* val_sidx = (int*)malloc(sizeof(int) * total_TCBlock_num * TC_BLOCK_K);
    memset(val_sidx, 0, sizeof(int) * total_TCBlock_num * TC_BLOCK_K);
    //> val_coo_off -> TC Block 中非零元个数，每个block有两个范围（起始index，结束index），TC
    // Block可能pad0
    int* val_coo_off = (int*)malloc(sizeof(int) * (total_TCBlock_num * 2 + 1));
    memset(val_coo_off, 0, sizeof(int) * (total_TCBlock_num * 2 + 1));

    //> val_coo_sidx
    int* val_coo_idx = (int*)malloc(sizeof(int) * nnz_pad);
    memset(val_coo_idx, 0, sizeof(int) * nnz_pad);
    float* val_coo_val = (float*)malloc(sizeof(float) * nnz_pad);
    memset(val_coo_val, 0, sizeof(float) * nnz_pad);

    //> prefix sum
    int cur_pos = 0;
    int i_tc_block_ptr = 0;
    for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
        vector<int>& nnz_per_tc_block = vec_nnz_per_tc_block[i_row_window];
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

    auto swizzle_layout =
      composition(Swizzle<1, 2, 3>{},
                  make_layout(make_shape(ROW_WINDOW_M, TC_BLOCK_K), make_stride(TC_BLOCK_K, 1)));
    int* tc_block_nnz_ptr = (int*)malloc(sizeof(int) * (K + TC_BLOCK_K - 1) / TC_BLOCK_K);
    for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
        int TC_block_offset = val_soff[i_row_window];
        int* tmp_val_sidx = val_sidx + TC_block_offset * TC_BLOCK_K;
        memset(nnz_col, 0, sizeof(int) * K);
        vector<int>& nnz_per_tc_block = vec_nnz_per_tc_block[i_row_window];
        for (int i_row = 0; i_row < ROW_WINDOW_M; i_row++) {
            int row = i_row_window * ROW_WINDOW_M + i_row;
            if (row >= M)
                continue;
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
            if (row >= M)
                continue;
            for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                int col = csrColIdxA[j];
                float value = (float)csrValA[j];
                int condensed_pos = binarySearch(nnz_col, col_start_pos, nnz_col_num - 1, col);
                col_start_pos = condensed_pos + 1;
                int tc_block_id = condensed_pos / TC_BLOCK_K;
                val_coo_val[tc_block_nnz_ptr[tc_block_id]] = value;
                //! origin idx
                // int coo_idx = i_row * TC_BLOCK_K + (condensed_pos % TC_BLOCK_K);
                // val_coo_idx[tc_block_nnz_ptr[tc_block_id]] = coo_idx;
                //! manual swizzle
                auto coord = make_coord(i_row, condensed_pos % TC_BLOCK_K);
                auto swizzled_idx = swizzle_layout(coord);
                val_coo_idx[tc_block_nnz_ptr[tc_block_id]] = swizzled_idx;
                //! end
                tc_block_nnz_ptr[tc_block_id]++;
            }
        }
    }

    { //! check sidx
      // printf("check sparse AtoB array: ");
      // for (int i = 0; i < total_TCBlock_num * TC_BLOCK_K; i++) {
      //     printf("%d ", val_sidx[i]);
      // } printf("\n");
    }

    //! strict load balance
    int row_block_num = 0;
    int slb_block_dim_size = 0;
    vector<int> row_block_offset;
    vector<int> row_block_to_row_window_id;
    vector<int> slb_tb_offset;

    strict_load_balance(M, N, K, total_TCBlock_num, val_soff, row_block_num, slb_block_dim_size, row_block_offset, row_block_to_row_window_id, slb_tb_offset);

    // TODO: need to change according to swizzle
    //! reorder

    dtc_spmm_test_val_sidx_bind_reorder<64>(M, N, K, total_TCBlock_num, nnz_pad, val_sidx, val_soff,
                                            B, val_coo_idx, val_coo_val, val_coo_off, 
                                            row_block_num, slb_block_dim_size, row_block_offset, row_block_to_row_window_id, slb_tb_offset, 
                                            C, is_ncu_test, warmup_time, execute_time);
}

void verify_new(float* C_cpu, float* C_cuda, int M, int N) {
    int flag = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float cusparse_val = C_cpu[OFFSET_ROW(i, j, N)];
            float mykernel_val = C_cuda[OFFSET_ROW(i, j, N)];
            if (fabs(cusparse_val - mykernel_val) > 0.01) {
                flag++;
                if (flag < 200)
                printf("Error(%d, %d): cusp(%.1f) mykernel(%.1f)\n", i, j, cusparse_val,
                       mykernel_val);
            }
        }
    }
}

void cusparse_spmm_all(float* cu_ValA,
                       int* cu_RowPtrA,
                       int* cu_ColIdxA,
                       float* cu_B,
                       float* cu_C,
                       int M,
                       int N,
                       int K,
                       int nnz) {
    int A_num_rows = M;
    int A_num_cols = K;
    int B_num_rows = K;
    int B_num_cols = N;
    int ldb = B_num_cols; // B: row first
    int B_size = B_num_rows * ldb;
    int C_num_rows = M;
    int C_num_cols = N;
    int ldc = C_num_cols; // C: row first
    int C_size = C_num_rows * ldc;
    float alpha = 1.0f;
    float beta = 0.0f;
    // device memory management
    int* dA_rpt;
    int* dA_cid;
    float* dA_val;
    float* dB;
    float* dC;
    CHECK_CUDA(cudaMalloc(&dA_rpt, sizeof(int) * (M + 1)));
    CHECK_CUDA(cudaMalloc(&dA_cid, sizeof(int) * nnz));
    CHECK_CUDA(cudaMalloc(&dA_val, sizeof(float) * nnz));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * B_size));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * C_size));

    CHECK_CUDA(cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_val, cu_ValA, sizeof(float) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, cu_B, sizeof(float) * B_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * C_size));

    // cuSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, nnz, dA_rpt, dA_cid, dA_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(
      cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW))
    CHECK_CUSPARSE(
      cusparseCreateDnMat(&matC, C_num_rows, C_num_cols, ldc, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
      matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute preprocess (optional)
    CHECK_CUSPARSE(cusparseSpMM_preprocess(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
      matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))

    // execute SpMM
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
                                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))

    CHECK_CUDA(cudaMemcpy(cu_C, dC, sizeof(float) * C_size, cudaMemcpyDeviceToHost));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(dA_cid))
    CHECK_CUDA(cudaFree(dA_rpt))
    CHECK_CUDA(cudaFree(dA_val))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(dC))
    return;
}

int main(int argc, char** argv) {
    string filename;
    int N = 1024;
    int M = 1024;
    int K = 1024;
    float sparsity = 0.9;
    int mtx_flag = 1;
    int is_ncu_test = 0;
    int warmup_time = 100;
    int execute_time = 1000;
    parseInput(argc, argv, filename, "-filename", N, "-N", M, "-M", K, "-K", mtx_flag, "-mtx_flag",
               sparsity, "-sparsity", is_ncu_test, "-ncu", warmup_time, "-warmup", execute_time,
               "-repeat");

    cout << "Filename: " << filename << endl;
    cout << "N: " << N << endl;

    int rowA, colA;
    int nnz = 0;
    int isSymmetricA;
    int* csrPtrA;
    int* csrColIdxA;
    float* csrValA;
    float* A;
    //! input from file
    if (mtx_flag == 1) {
        mmio_allinone(&rowA, &colA, &nnz, &isSymmetricA, &csrPtrA, &csrColIdxA, &csrValA,
                      filename.c_str());
        M = rowA;
        K = colA;
    } else { //! sparse matrix init
        A = (float*)malloc(sizeof(float) * M * K);
        initVector(A, M * K, sparsity);
        //> convert to csr
        csrPtrA = (int*)malloc(sizeof(int) * (M + 1));
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
        csrColIdxA = (int*)malloc(sizeof(int) * nnz);
        csrValA = (float*)malloc(sizeof(float) * nnz);
        memset(csrColIdxA, 0, sizeof(int) * nnz);
        memset(csrValA, 0, sizeof(float) * nnz);
        int ptr = 0;
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float value = A[i * K + k];
                if (value != 0) {
                    csrColIdxA[ptr] = k;
                    csrValA[ptr] = value;
                    ptr++;
                }
            }
        }
    }

    printf("FLOAT: Sparse A[%d,%d] x B[%d,%d] = C[%d,%d]\n", M, K, K, N, M, N);

    float* B = (float*)malloc(sizeof(float) * K * N);
    // initVec(B, K * N);
    initVec(csrValA, nnz);
    initVector(B, K * N, 0);

    float* C_cuda = (float*)malloc(sizeof(float) * M * N);
    convertToMETCF(M, N, K, nnz, csrPtrA, csrColIdxA, csrValA, B, C_cuda, (bool)is_ncu_test,
                   warmup_time, execute_time);

    // host check
    float* C_cpu = (float*)malloc(sizeof(float) * M * N);
    memset(C_cpu, 0, sizeof(float) * M * N);
    cusparse_spmm_all(csrValA, csrPtrA, csrColIdxA, B, C_cpu, M, N, K, nnz);
    verify_new(C_cpu, C_cuda, M, N);

    printf("check nan: cusp %.1f mykernel %.1f\n", C_cpu[0], C_cuda[0]);
    return 0;
}