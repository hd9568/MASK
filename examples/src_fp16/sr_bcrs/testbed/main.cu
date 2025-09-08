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

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32

template <int T_M, int T_N, int T_K>
extern void sr_bcrs_test_fp16(int M,
                              int N,
                              int K,
                              int TCBlock_num,
                              int* val_sidx,
                              int* val_soff,
                              half* B_val,
                              half* val_block_val,
                              float* C,
                              bool is_ncu_test,
                              int warmup_time,
                              int execute_time);

// extern template
// void sr_bcrs_test_fp16<32, 64, 32>(
//     int M, int N, int K,
//     int TCBlock_num,
//     int* val_sidx,
//     int* val_soff,
//     half* B_val,
//     half* val_block_val,
//     float* C
// );

void initVector(half* vec, int length, float sparsity) {
    if (sparsity < 0.0f || sparsity > 1.0f)
        return;
    auto seed = std::time(nullptr);
    std::srand(1234);
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
            // if (i == 0)
            // printf("check(%d, %d): cusp(%.1f) mykernel(%.1f)\n", i, j, cusparse_val,
            // mykernel_val);
        }
    }
}

void cusparse_spmm_all(half* cu_ValA,
                       int* cu_RowPtrA,
                       int* cu_ColIdxA,
                       half* cu_B,
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
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, nnz, dA_rpt, dA_cid, dA_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
    CHECK_CUSPARSE(
      cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB, CUDA_R_16F, CUSPARSE_ORDER_ROW))
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

void convertToSRBCRS(int M,
                     int N,
                     int K,
                     int nnz,
                     int* csrPtrA,
                     int* csrColIdxA,
                     half* csrValA,
                     half* B,
                     float* C,
                     bool is_ncu_test,
                     int warmup_time,
                     int execute_time) {
    // convert to SR-BCRS
    int row_window_num = (M + TILE_M - 1) / TILE_M;
    int* val_soff = (int*)malloc(sizeof(int) * (row_window_num + 1));
    memset(val_soff, 0, sizeof(int) * (row_window_num + 1));
    int total_TCBlock_num = 0;
    int* nnz_col = (int*)malloc(sizeof(int) * K);
    for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
        val_soff[i_row_window] = total_TCBlock_num;
        memset(nnz_col, 0, sizeof(int) * K);
        for (int i_row = 0; i_row < TILE_M; i_row++) {
            int row = i_row_window * TILE_M + i_row;
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
        total_TCBlock_num += (nnz_col_num + TILE_K - 1) / TILE_K;
    }
    val_soff[row_window_num] = total_TCBlock_num;
    printf("total_BCRS_Block_num = %d\n", total_TCBlock_num);

    int* val_sidx = (int*)malloc(sizeof(int) * total_TCBlock_num * TILE_K);
    memset(val_sidx, 0, sizeof(int) * total_TCBlock_num * TILE_K);
    half* val_block_val = (half*)malloc(sizeof(half) * total_TCBlock_num * TILE_M * TILE_K);
    memset(val_block_val, 0, sizeof(half) * total_TCBlock_num * TILE_M * TILE_K);

    auto swizzle_layout = composition(
      Swizzle<2, 3, 3>{}, make_layout(make_shape(TILE_M, TILE_K), make_stride(TILE_K, 1)));
    for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
        int TC_block_offset = val_soff[i_row_window];
        int* tmp_val_sidx = val_sidx + TC_block_offset * TILE_K;
        half* tmp_val_block_val = val_block_val + TC_block_offset * TILE_M * TILE_K;
        memset(nnz_col, 0, sizeof(int) * K);
        for (int i_row = 0; i_row < TILE_M; i_row++) {
            int row = i_row_window * TILE_M + i_row;
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
        int tc_block_num = (nnz_col_num + TILE_K - 1) / TILE_K;
        for (int i_tc_block_col = 0; i_tc_block_col < tc_block_num * TILE_K; i_tc_block_col++) {
            if (i_tc_block_col >= nnz_col_num)
                tmp_val_sidx[i_tc_block_col] = nnz_col[nnz_col_num - 1];
            else
                tmp_val_sidx[i_tc_block_col] = nnz_col[i_tc_block_col];
        }
        for (int i_row = 0; i_row < TILE_M; i_row++) {
            int row = i_row_window * TILE_M + i_row;
            int col_start_pos = 0;
            if (row >= M)
                continue;
            for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                int col = csrColIdxA[j];
                float value = (float)csrValA[j];
                int condensed_pos = binarySearch(nnz_col, col_start_pos, nnz_col_num - 1, col);
                col_start_pos = condensed_pos + 1;
                //! without manual swizzle
                // tmp_val_block_val[
                //     (condensed_pos / TILE_K) * TILE_M * TILE_K +
                //     OFFSET_ROW(i_row, condensed_pos % TILE_K, TILE_K)] = (half)value;
                //! manual swizzle
                auto coord = make_coord(i_row, condensed_pos % TILE_K);
                auto idx = swizzle_layout(coord);
                tmp_val_block_val[(condensed_pos / TILE_K) * TILE_M * TILE_K + idx] = (half)value;
            }
        }
        //! debug
        {
            // printf("check s_off\n");
            // for (int i = 0; i < tc_block_num; i++) {
            //     printf("%d ", val_soff[i]);
            // } printf("\n");
            // for (int tc_block_ptr = 0; tc_block_ptr < tc_block_num; tc_block_ptr++) {
            //     printf("tc_block i = %d, j = %d\n", i_row_window, tc_block_ptr);
            //     for (int i = 0; i < TILE_M; i++) {
            //         for (int j = 0; j < TILE_K; j++) {
            //             printf("%.0f ", (float)tmp_val_block_val[tc_block_ptr * TILE_M * TILE_K +
            //             OFFSET_ROW(i, j, TILE_K)]);
            //         } printf("\n");
            //     }
            // }
        }
    }
    //! launch kernel
    sr_bcrs_test_fp16<TILE_M, TILE_N, TILE_K>(M, N, K, total_TCBlock_num, val_sidx, val_soff, B,
                                              val_block_val, C, is_ncu_test, warmup_time,
                                              execute_time);
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

    int rowA = M;
    int colA = K;
    int nnz = 0;
    int isSymmetricA;
    int* csrPtrA;
    int* csrColIdxA;
    half* csrValA;
    half* A;
    //! input from file
    if (mtx_flag == 1) {
        mmio_allinone(&rowA, &colA, &nnz, &isSymmetricA, &csrPtrA, &csrColIdxA, &csrValA,
                      filename.c_str());
        M = rowA;
        K = colA;
    } else { //! sparse matrix init
        A = (half*)malloc(sizeof(half) * M * K);
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
        csrValA = (half*)malloc(sizeof(half) * nnz);
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
    }

    printf("HALF: Sparse A[%d,%d] x B[%d,%d] = C[%d,%d]\n", M, K, K, N, M, N);

    half* B = (half*)malloc(sizeof(half) * K * N);
    // initVec(B, K * N);
    initVec(csrValA, nnz);
    initVector(B, K * N, 0);
    { //! debug matirx B
      // for (int tid = 0; tid < 32; tid++) {
      //     int row = tid / 8;
      //     int col = tid % 8 * 8;
      //     printf("tid=%d,load: ", tid);
      //     for (int i = 0; i < 8; i++) {
      //         printf("%.0f ", (float)B[OFFSET_ROW(row, col + i, N)]);
      //     }
      //     printf("\n");
      // }
    }
    { //! debug convert to dense
      // float* restoreA = (float*)malloc(sizeof(float) * rowA * colA);
      // memset(restoreA, 0, sizeof(float) * rowA * colA);
      // for (int i_row = 0; i_row < rowA; i_row++) {
      //     for (int j = csrPtrA[i_row]; j < csrPtrA[i_row + 1]; j++) {
      //         int col = csrColIdxA[j];
      //         float value = (float)csrValA[j];
      //         restoreA[i_row * colA + col] = value;
      //     }
      // }

        // for (int k_ptr = 0; k_ptr < colA / TILE_K; k_ptr++) {
        //     printf("host check tileA: i=%d\n", k_ptr);
        //     for (int i = 0; i < TILE_M; i++) {
        //         for (int j = 0; j < TILE_K; j++) {
        //             printf("%.0f ", restoreA[OFFSET_ROW(i, k_ptr * TILE_K + j, colA)]);
        //         } printf("\n");
        //     }
        // }
    }
    { //! debug tile B
      // for (int k_ptr = 0; k_ptr < colA / TILE_K; k_ptr++) {
      //     printf("host check tile B: i = %d\n", k_ptr);
      //     for (int i = 0; i < TILE_K; i++) {
      //         for (int j = 0; j < 16; j++) {
      //             printf("%.0f ", (float)B[OFFSET_ROW(i + k_ptr * TILE_K, j, N)]);
      //         } printf("\n");
      //     }
      // }
    }

    float* C_cuda = (float*)malloc(sizeof(float) * M * N);
    convertToSRBCRS(M, N, K, nnz, csrPtrA, csrColIdxA, csrValA, B, C_cuda, (bool)is_ncu_test,
                    warmup_time, execute_time);

    // host check
    float* C_cpu = (float*)malloc(sizeof(float) * M * N);
    memset(C_cpu, 0, sizeof(half) * M * N);
    cusparse_spmm_all(csrValA, csrPtrA, csrColIdxA, B, C_cpu, M, N, K, nnz);
    verify_new(C_cpu, C_cuda, M, N);

    printf("check nan: cusp %.1f mykernel %.1f\n", C_cpu[0], C_cuda[0]);
    return 0;
}