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
#include <omp.h>

#include "common.h"
#include "utils.h"

#include <cute/tensor.hpp>

using namespace std;
using namespace cute;

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))
#define OFFSET_COL(row, col, lda) ((col) * (lda) + (row))

#define TILE_M 16
#define TILE_N 64
#define TILE_K 8

#define DTYPEAB float

template <int T_M, int T_N, int T_K>
extern void sr_bcrs_test_tf32_multi_bind(int M,
                                         int N,
                                         int K,
                                         int TCBlock_num,
                                         int* val_sidx,
                                         const vector<int>& row_window_val_soff,
                                         DTYPEAB* B_val,
                                         DTYPEAB* val_block_val,
                                         const vector<int>& row_window_binding,
                                         const vector<int>& row_window_split,
                                         float* C,
                                         bool is_ncu_test,
                                         int warmup_time,
                                         int execute_time);

void initVector(DTYPEAB* vec, int length, float sparsity) {
    if (sparsity < 0.0f || sparsity > 1.0f)
        return;
    auto seed = std::time(nullptr);
    std::srand(1234);
    cout << seed << endl;
    for (int i = 0; i < length; ++i) {
        if (static_cast<float>(rand()) / RAND_MAX < sparsity) {
            vec[i] = DTYPEAB(0.0f);
        } else {
            vec[i] = DTYPEAB(rand() % 5 + 1);
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
                    printf("Error(%d, %d): cusp(%.1f) mykernel(%.1f)\n",
                           i,
                           j,
                           cusparse_val,
                           mykernel_val);
            }
        }
    }
}

void cusparse_spmm_all(DTYPEAB* cu_ValA,
                       int* cu_RowPtrA,
                       int* cu_ColIdxA,
                       DTYPEAB* cu_B,
                       float* cu_C,
                       int M,
                       int N,
                       int K,
                       int nnz) {
    int A_num_rows = M;
    int A_num_cols = K;
    int B_num_rows = K;
    int B_num_cols = N;
    int ldb = B_num_cols;
    int B_size = B_num_rows * ldb;
    int C_num_rows = M;
    int C_num_cols = N;
    int ldc = C_num_cols;
    int C_size = C_num_rows * ldc;
    float alpha = 1.0f;
    float beta = 0.0f;

    int* dA_rpt;
    int* dA_cid;
    DTYPEAB* dA_val;
    DTYPEAB* dB;
    float* dC;
    CHECK_CUDA(cudaMalloc(&dA_rpt, sizeof(int) * (M + 1)));
    CHECK_CUDA(cudaMalloc(&dA_cid, sizeof(int) * nnz));
    CHECK_CUDA(cudaMalloc(&dA_val, sizeof(DTYPEAB) * nnz));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(DTYPEAB) * B_size));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * C_size));

    CHECK_CUDA(cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_val, cu_ValA, sizeof(DTYPEAB) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, cu_B, sizeof(DTYPEAB) * B_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * C_size));

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
                                     A_num_rows,
                                     A_num_cols,
                                     nnz,
                                     dA_rpt,
                                     dA_cid,
                                     dA_val,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnMat(
      &matB, B_num_rows, B_num_cols, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW))
    CHECK_CUSPARSE(cusparseCreateDnMat(
      &matC, C_num_rows, C_num_cols, ldc, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           matA,
                                           matB,
                                           &beta,
                                           matC,
                                           CUDA_R_32F,
                                           CUSPARSE_SPMM_ALG_DEFAULT,
                                           &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMM_preprocess(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           matA,
                                           matB,
                                           &beta,
                                           matC,
                                           CUDA_R_32F,
                                           CUSPARSE_SPMM_ALG_DEFAULT,
                                           dBuffer))

    CHECK_CUSPARSE(cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                matA,
                                matB,
                                &beta,
                                matC,
                                CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT,
                                dBuffer))

    CHECK_CUDA(cudaMemcpy(cu_C, dC, sizeof(float) * C_size, cudaMemcpyDeviceToHost));

    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(dA_cid))
    CHECK_CUDA(cudaFree(dA_rpt))
    CHECK_CUDA(cudaFree(dA_val))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(dC))
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
    return -1;
}

void multi_binding_load_balance(int M,
                                int total_TCBlock_num,
                                int* val_soff,
                                int& row_window_blockDim_size,
                                vector<int>& row_window_binding,
                                vector<int>& row_window_split,
                                vector<int>& row_window_val_soff) {
    const double alpha = 2.0;
    const int max_splits = 9999;

    int row_window_num = (M + TILE_M - 1) / TILE_M;
    row_window_split.assign(row_window_num, 0);

    vector<int> workloads(row_window_num, 0);
    for (int i = 0; i < row_window_num; ++i) {
        workloads[i] = val_soff[i + 1] - val_soff[i];
    }

    double avg = 0.0;
    if (!workloads.empty()) {
        avg = accumulate(workloads.begin(), workloads.end(), 0.0) / workloads.size();
    }

    vector<int> split_workloads;
    split_workloads.reserve(row_window_num);
    row_window_binding.clear();

    for (int row_window_i = 0; row_window_i < row_window_num; ++row_window_i) {
        int x = workloads[row_window_i];
        if (x == 0) {
            continue;
        }

        if (avg > 0.0 && x > alpha * avg) {
            int parts = min((int)ceil(x / avg), max_splits);
            parts = max(parts, 1);
            if (parts > 1) {
                row_window_split[row_window_i] = 1;
            }

            int base = x / parts;
            int rem = x % parts;
            for (int i = 0; i < parts; ++i) {
                split_workloads.push_back(base + (i < rem ? 1 : 0));
                row_window_binding.push_back(row_window_i);
            }
        } else {
            split_workloads.push_back(x);
            row_window_binding.push_back(row_window_i);
        }
    }

    row_window_blockDim_size = (int)split_workloads.size();

    row_window_val_soff.clear();
    row_window_val_soff.reserve(row_window_blockDim_size + 1);
    row_window_val_soff.push_back(0);
    for (int i = 0; i < row_window_blockDim_size; ++i) {
        row_window_val_soff.push_back(row_window_val_soff.back() + split_workloads[i]);
    }

    assert((int)row_window_binding.size() == row_window_blockDim_size);
    assert((int)row_window_val_soff.size() == row_window_blockDim_size + 1);
    assert(row_window_val_soff.back() == total_TCBlock_num);
}

void convertToSRBCRS(int M,
                     int N,
                     int K,
                     int nnz,
                     int* csrPtrA,
                     int* csrColIdxA,
                     DTYPEAB* csrValA,
                     DTYPEAB* B,
                     float* C,
                     bool is_ncu_test,
                     int warmup_time,
                     int execute_time) {
    int row_window_num = (M + TILE_M - 1) / TILE_M;
    int* val_soff = (int*)malloc(sizeof(int) * (row_window_num + 1));
    memset(val_soff, 0, sizeof(int) * (row_window_num + 1));
    vector<int> tc_block_count(row_window_num, 0);

#pragma omp parallel
    {
        vector<unsigned char> col_mark(K, 0);
        vector<int> touched_cols;
        touched_cols.reserve(TILE_M * TILE_K);

#pragma omp for schedule(static)
        for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
            touched_cols.clear();
            for (int i_row = 0; i_row < TILE_M; i_row++) {
                int row = i_row_window * TILE_M + i_row;
                if (row >= M)
                    continue;
                for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                    int col = csrColIdxA[j];
                    if (!col_mark[col]) {
                        col_mark[col] = 1;
                        touched_cols.push_back(col);
                    }
                }
            }

            sort(touched_cols.begin(), touched_cols.end());
            tc_block_count[i_row_window] = ((int)touched_cols.size() + TILE_K - 1) / TILE_K;

            for (int col : touched_cols) {
                col_mark[col] = 0;
            }
        }
    }

    int total_TCBlock_num = 0;
    for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
        val_soff[i_row_window] = total_TCBlock_num;
        total_TCBlock_num += tc_block_count[i_row_window];
    }
    val_soff[row_window_num] = total_TCBlock_num;
    printf("total_BCRS_Block_num = %d\n", total_TCBlock_num);

    memset(C, 0, sizeof(float) * M * N);
    if (total_TCBlock_num == 0) {
        free(val_soff);
        return;
    }

    int* val_sidx = (int*)malloc(sizeof(int) * total_TCBlock_num * TILE_K);
    memset(val_sidx, 0, sizeof(int) * total_TCBlock_num * TILE_K);
    DTYPEAB* val_block_val =
      (DTYPEAB*)malloc(sizeof(DTYPEAB) * total_TCBlock_num * TILE_M * TILE_K);
    memset(val_block_val, 0, sizeof(DTYPEAB) * total_TCBlock_num * TILE_M * TILE_K);

    auto swizzle_layout = composition(
      Swizzle<2, 2, 3>{}, make_layout(make_shape(TILE_M, TILE_K), make_stride(TILE_K, 1)));

#pragma omp parallel
    {
        vector<unsigned char> col_mark(K, 0);
        vector<int> touched_cols;
        touched_cols.reserve(TILE_M * TILE_K);

#pragma omp for schedule(static)
        for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
            int TC_block_offset = val_soff[i_row_window];
            int* tmp_val_sidx = val_sidx + TC_block_offset * TILE_K;
            DTYPEAB* tmp_val_block_val = val_block_val + TC_block_offset * TILE_M * TILE_K;

            touched_cols.clear();
            for (int i_row = 0; i_row < TILE_M; i_row++) {
                int row = i_row_window * TILE_M + i_row;
                if (row >= M)
                    continue;
                for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                    int col = csrColIdxA[j];
                    if (!col_mark[col]) {
                        col_mark[col] = 1;
                        touched_cols.push_back(col);
                    }
                }
            }

            sort(touched_cols.begin(), touched_cols.end());
            int nnz_col_num = (int)touched_cols.size();
            int tc_block_num = (nnz_col_num + TILE_K - 1) / TILE_K;

            for (int i_tc_block_col = 0; i_tc_block_col < tc_block_num * TILE_K; i_tc_block_col++) {
                if (i_tc_block_col >= nnz_col_num)
                    tmp_val_sidx[i_tc_block_col] = touched_cols[nnz_col_num - 1];
                else
                    tmp_val_sidx[i_tc_block_col] = touched_cols[i_tc_block_col];
            }

            for (int i_row = 0; i_row < TILE_M; i_row++) {
                int row = i_row_window * TILE_M + i_row;
                int col_start_pos = 0;
                if (row >= M)
                    continue;
                for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                    int col = csrColIdxA[j];
                    float value = (float)csrValA[j];
                    int condensed_pos = (int)(lower_bound(touched_cols.begin() + col_start_pos,
                                                          touched_cols.end(),
                                                          col) -
                                             touched_cols.begin());
                    col_start_pos = condensed_pos + 1;

                    auto coord = make_coord(i_row, condensed_pos % TILE_K);
                    auto idx = swizzle_layout(coord);
                    tmp_val_block_val[(condensed_pos / TILE_K) * TILE_M * TILE_K + idx] =
                      DTYPEAB(value);
                }
            }

            for (int col : touched_cols) {
                col_mark[col] = 0;
            }
        }
    }

    int row_window_blockDim_size = 0;
    vector<int> row_window_binding;
    vector<int> row_window_split;
    vector<int> row_window_val_soff;
    multi_binding_load_balance(M,
                               total_TCBlock_num,
                               val_soff,
                               row_window_blockDim_size,
                               row_window_binding,
                               row_window_split,
                               row_window_val_soff);

    sr_bcrs_test_tf32_multi_bind<TILE_M, TILE_N, TILE_K>(M,
                                                          N,
                                                          K,
                                                          total_TCBlock_num,
                                                          val_sidx,
                                                          row_window_val_soff,
                                                          B,
                                                          val_block_val,
                                                          row_window_binding,
                                                          row_window_split,
                                                          C,
                                                          is_ncu_test,
                                                          warmup_time,
                                                          execute_time);

    free(val_sidx);
    free(val_block_val);
    free(val_soff);
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
    parseInput(argc,
               argv,
               filename,
               "-filename",
               N,
               "-N",
               M,
               "-M",
               K,
               "-K",
               mtx_flag,
               "-mtx_flag",
               sparsity,
               "-sparsity",
               is_ncu_test,
               "-ncu",
               warmup_time,
               "-warmup",
               execute_time,
               "-repeat");

    cout << "Filename: " << filename << endl;
    cout << "N: " << N << endl;

    int rowA = M;
    int colA = K;
    int nnz = 0;
    int isSymmetricA;
    int* csrPtrA;
    int* csrColIdxA;
    DTYPEAB* csrValA;
    DTYPEAB* A;

    if (mtx_flag == 1) {
        mmio_allinone(
          &rowA, &colA, &nnz, &isSymmetricA, &csrPtrA, &csrColIdxA, &csrValA, filename.c_str());
        M = rowA;
        K = colA;
    } else {
        A = (DTYPEAB*)malloc(sizeof(DTYPEAB) * M * K);
        initVector(A, M * K, sparsity);
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
        csrValA = (DTYPEAB*)malloc(sizeof(DTYPEAB) * nnz);
        memset(csrColIdxA, 0, sizeof(int) * nnz);
        memset(csrValA, 0, sizeof(DTYPEAB) * nnz);
        int ptr = 0;
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float value = (float)A[i * K + k];
                if (value != 0) {
                    csrColIdxA[ptr] = k;
                    csrValA[ptr] = DTYPEAB(value);
                    ptr++;
                }
            }
        }
    }

    printf("FLOAT: Sparse A[%d,%d] x B[%d,%d] = C[%d,%d]\n", M, K, K, N, M, N);

    DTYPEAB* B = (DTYPEAB*)malloc(sizeof(DTYPEAB) * K * N);
    initVec(csrValA, nnz);
    initVector(B, K * N, 0);

    float* C_cuda = (float*)malloc(sizeof(float) * M * N);
    convertToSRBCRS(M,
                    N,
                    K,
                    nnz,
                    csrPtrA,
                    csrColIdxA,
                    csrValA,
                    B,
                    C_cuda,
                    (bool)is_ncu_test,
                    warmup_time,
                    execute_time);

    float* C_cpu = (float*)malloc(sizeof(float) * M * N);
    memset(C_cpu, 0, sizeof(float) * M * N);
    cusparse_spmm_all(csrValA, csrPtrA, csrColIdxA, B, C_cpu, M, N, K, nnz);
    verify_new(C_cpu, C_cuda, M, N);

    printf("check nan: cusp %.1f mykernel %.1f\n", C_cpu[0], C_cuda[0]);
    return 0;
}
