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

#define CHECK_CUDA2(func)                                                                          \
    {                                                                                              \
        cudaError_t status = (func);                                                               \
        if (status != cudaSuccess) {                                                               \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,                   \
                   cudaGetErrorString(status), status);                                            \
        }                                                                                          \
    }

using namespace std;

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
            // vec[i] = float(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
            vec[i] = float(rand() % 5 + 1);
        }
    }
}

void cusparse_spmv(float* cu_ValA,
                   int* cu_RowPtrA,
                   int* cu_ColIdxA,
                   float* cu_B,
                   float* cu_C,
                   int M,
                   int K,
                   int nnz,
                   bool is_ncu_test,
                   int warmup_time,
                   int execute_time) {
    int A_num_rows = M;
    int A_num_cols = K;
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
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * K));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * M));

    CHECK_CUDA(cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_val, cu_ValA, sizeof(float) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, cu_B, sizeof(float) * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * M));

    // cuSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecB, vecC;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, nnz, dA_rpt, dA_cid, dA_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Create dense vector descriptors
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecB, K, dB, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecC, M, dC, CUDA_R_32F));

    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                           vecB, &beta, vecC, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                           &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    if (!is_ncu_test) {
        struct timeval t1, t2;
        {
            for (int i = 0; i < warmup_time; i++) {
                // execute SpMV
                CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                            vecB, &beta, vecC, CUDA_R_32F,
                                            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
            }
            CHECK_CUDA2(cudaDeviceSynchronize());
            gettimeofday(&t1, NULL);
            for (int i = 0; i < execute_time; i++) {
                // execute SpMV
                CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                            vecB, &beta, vecC, CUDA_R_32F,
                                            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
            }
            CHECK_CUDA2(cudaDeviceSynchronize());
            gettimeofday(&t2, NULL);
            double mykernel_time =
              ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) /
              execute_time;
            printf("mykernel_time: %f ms\n", mykernel_time);
        }
    } else {
        // execute SpMV
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecB,
                                    &beta, vecC, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        CHECK_CUDA2(cudaDeviceSynchronize());
    }

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecB))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecC))
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
    int N = 1; // Set N to 1 for vector case
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

    N = 1;

    printf("FLOAT: Sparse A[%d,%d] x B[%d,%d] = C[%d,%d]\n", M, K, K, N, M, N);

    float* B = (float*)malloc(sizeof(float) * K);
    initVec(csrValA, nnz);
    initVector(B, K, 0);

    cusparse_spmv(csrValA, csrPtrA, csrColIdxA, B, nullptr, M, K, nnz, (bool)is_ncu_test,
                  warmup_time, execute_time);

    return 0;
}