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
#include <cusparse.h>

#include "common.h"
#include "utils.h"

using namespace std;

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))
#define OFFSET_COL(row, col, lda) ((col) * (lda) + (row))

#define ROW_WINDOW_M 16
#define TC_BLOCK_K 16
#define TC_BLOCK_M 16

template <int TILE_B>
extern void dtc_spmm_test_val_sidx_bind(
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

extern template void dtc_spmm_test_val_sidx_bind<32>(
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

int binarySearch(int* arr, int l, int r, int target) {
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) l = mid + 1;
        else r = mid - 1;
    }
    return -1; // 未找到
}

uint32_t pack_half_short(half h, short s) {
    uint16_t half_bits = *reinterpret_cast<uint16_t*>(&h);
    uint16_t short_bits = static_cast<uint16_t>(s);
    return (static_cast<uint32_t>(half_bits) << 16) | short_bits;
}

void debug_unpack_half_short(uint32_t packed, __half &h, short &s) {
    uint16_t half_bits = static_cast<uint16_t>(packed >> 16);
    uint16_t short_bits = static_cast<uint16_t>(packed & 0xFFFF);

    h = *reinterpret_cast<__half*>(&half_bits); // 还原 half
    s = static_cast<short>(short_bits);        // 还原 short
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

    //! val_sidx_bind
    uint32_t* val_coo_val_idx_bind = (uint32_t*)malloc(sizeof(uint32_t) * nnz_pad);
    memset(val_coo_val_idx_bind, 0, sizeof(uint32_t) * nnz_pad);

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
                //! bind the half and index together
                // val_coo_val[tc_block_nnz_ptr[tc_block_id]] = (half)value;
                // val_coo_idx[tc_block_nnz_ptr[tc_block_id]] = i_row * TC_BLOCK_K + (condensed_pos % TC_BLOCK_K);
                val_coo_val_idx_bind[tc_block_nnz_ptr[tc_block_id]] = pack_half_short((half)value, i_row * TC_BLOCK_K + (condensed_pos % TC_BLOCK_K));
                {
                    // half debug_value;
                    // short debug_idx;
                    // debug_unpack_half_short(val_coo_val_idx_bind[tc_block_nnz_ptr[tc_block_id]], debug_value, debug_idx);
                    // printf("debub val_idx_bind: %.0f+%d->(%.0f | %d)\n", value, i_row * TC_BLOCK_K + (condensed_pos % TC_BLOCK_K), (float)debug_value, (int)debug_idx);
                }
                tc_block_nnz_ptr[tc_block_id]++;
            }
        }
    }
    // exit(0);
    // input to cuda func
    // dtc_spmm_test<32>(M, N, K, total_TCBlock_num, nnz_pad, val_sidx, val_soff, B, val_coo_idx, val_coo_val, val_coo_off, C);
    dtc_spmm_test_val_sidx_bind<32>(M, N, K, total_TCBlock_num, nnz_pad, val_sidx, val_soff, B, val_coo_val_idx_bind, val_coo_off, C);

    {//! debug check sparse AtoB array
        // printf("check sparse AtoB array: ");
        // for (int i = 0; i < total_TCBlock_num * TC_BLOCK_K; i++) {
        //     printf("%d ", val_sidx[i]);
        // } printf("\n");
    }

}

void verify_new(float* C_cpu, float* C_cuda, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float cusparse_val = C_cpu[OFFSET_ROW(i, j, N)];
            float mykernel_val = C_cuda[OFFSET_ROW(i, j, N)];
            if (fabs(cusparse_val - mykernel_val) > 0.01) {
                printf("Error(%d, %d): cusp(%.1f) mykernel(%.1f)\n", i, j, cusparse_val, mykernel_val);
            }
            // if (i == 0)
            // printf("check(%d, %d): cusp(%.1f) mykernel(%.1f)\n", i, j, cusparse_val, mykernel_val);
        }
    }
}

void cusparse_spmm_all(
    half* cu_ValA, int* cu_RowPtrA, int* cu_ColIdxA,
    half* cu_B, float* cu_C, int M, int N, int K, int nnz
) {
    int A_num_rows = M;
    int A_num_cols = K;
    int B_num_rows = K;
    int B_num_cols = N;
    int ldb = B_num_cols;   // B: row first
    int B_size = B_num_rows * ldb;
    int C_num_rows = M;
    int C_num_cols = N;
    int ldc = C_num_cols;            // C: row first
    int C_size = C_num_rows * ldc;
    float alpha = 1.0f;
    float beta = 0.0f;
    // device memory management
    int* dA_rpt;
    int* dA_cid;
    half* dA_val;
    half* dB;
    float* dC;
    CHECK_CUDA(cudaMalloc(&dA_rpt, sizeof(int) * (M + 1)));
    CHECK_CUDA(cudaMalloc(&dA_cid, sizeof(int) * nnz));
    CHECK_CUDA(cudaMalloc(&dA_val, sizeof(half) * nnz));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(half) * B_size));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * C_size));

    CHECK_CUDA(cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_val, cu_ValA, sizeof(half) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, cu_B, sizeof(half) * B_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * C_size));

    // cuSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, nnz, 
                                    dA_rpt, dA_cid, dA_val, 
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB,
                                    CUDA_R_16F, CUSPARSE_ORDER_ROW))
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, C_num_rows, C_num_cols, ldc, dC,
                                    CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute preprocess (optional)
    CHECK_CUSPARSE( cusparseSpMM_preprocess(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    CHECK_CUDA(cudaMemcpy(cu_C, dC, sizeof(float) * C_size, cudaMemcpyDeviceToHost));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_cid) )
    CHECK_CUDA( cudaFree(dA_rpt) )
    CHECK_CUDA( cudaFree(dA_val) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    return;
}

int main(int argc, char** argv) {
    string filename;
    int N = 512;

    parseInput(argc, argv, filename, "-filename", N, "-num");

    cout << "Filename: " << filename << endl;
    cout << "N: " << N << endl;

    int rowA, colA;
    int nnz = 0;
    int isSymmetricA;
    int* csrPtrA;
    int* csrColIdxA;
    half* csrValA;

    mmio_allinone(&rowA, &colA, &nnz, &isSymmetricA, &csrPtrA, &csrColIdxA, &csrValA, filename.c_str());
    int M = rowA;
    int K = colA;
    printf("HALF: Sparse A[%d,%d] x B[%d,%d] = C[%d,%d]\n", M, K, K, N, M, N);

    half* B = (half*)malloc(sizeof(half) * K * N);
    initVec(B, K * N);
    initVec(csrValA, nnz);

    {//! debug convert to dense
        // float* restoreA = (float*)malloc(sizeof(float) * rowA * colA);
        // memset(restoreA, 0, sizeof(float) * rowA * colA);
        // for (int i_row = 0; i_row < rowA; i_row++) {
        //     for (int j = csrPtrA[i_row]; j < csrPtrA[i_row + 1]; j++) {
        //         int col = csrColIdxA[j];
        //         float value = (float)csrValA[j];
        //         restoreA[i_row * colA + col] = value;
        //     }
        // }

        // for (int k_ptr = 0; k_ptr < colA / 16; k_ptr++) {
        //     printf("host check tileA: i=%d\n", k_ptr);
        //     for (int i = 0; i < 16; i++) {
        //         for (int j = 0; j < 16; j++) {
        //             printf("%.0f ", restoreA[OFFSET_ROW(i, k_ptr * 16 + j, colA)]);
        //         } printf("\n");
        //     }
        // }

    }

    float* C_cuda = (float*)malloc(sizeof(float) * M * N);
    convertToMETCF(M, N, K, nnz, csrPtrA, csrColIdxA, csrValA, B, C_cuda);

    // host check
    float* C_cpu = (float*)malloc(sizeof(float) * M * N);
    memset(C_cpu, 0, sizeof(half) * M * N);
    cusparse_spmm_all(csrValA, csrPtrA, csrColIdxA, B, C_cpu, M, N, K, nnz);
    verify_new(C_cpu, C_cuda, M, N);

    printf("check nan: cusp %.1f mykernel %.1f\n", C_cpu[0], C_cuda[0]);


    // float* C_host = (float*)malloc(sizeof(float) * M * N);
    // memset(C_host, 0, sizeof(float) * M * N);
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         for (int k = csrPtrA[i]; k < csrPtrA[i + 1]; k++) {
    //             int col = csrColIdxA[k];
    //             float value = (float)csrValA[k];
    //             C_host[i * N + j] += value * (float)B[OFFSET_ROW(col, j, N)];
    //         }
    //     }
    // }
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         if (fabs(C_host[i * N + j] - C_cpu[i * N + j]) > 0.01) {
    //             printf("Error(%d, %d): %f %f\n", i, j, C_host[i * N + j], C_cpu[i * N + j]);
    //         }
    //     }
    // }
    return 0;
    
}