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

#include <cute/tensor.hpp>

using namespace std;
using namespace cute;

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))
#define OFFSET_COL(row, col, lda) ((col) * (lda) + (row))

template <int TILE_M, int TILE_N, int TILE_K>
extern void flash_llm_test(
    int M, int N, int K,
    int NumOffsets,
    int NNZ,
    int* TileOffsets,
    uint32_t* Compressed_A,
    half* B_val,
    float* C
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

#define TC_BLOCK_K 16
#define TC_BLOCK_M 16

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

int binarySearch(int* arr, int l, int r, int target) {
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) l = mid + 1;
        else r = mid - 1;
    }
    return -1; // 未找到
}

int flashLLMPreprocess_full_matrixA(
    half* A_h,
    int M, int N, int K,
    uint32_t** Compressed_A,    // CPU PTR
    int** TileOffsets           // CPU PTR
) {
    const int TILE_M                       = 128;
    const int TILE_K                       = 32;
    const int TILE_N                       = 128;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;

    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / TILE_M;
    int   NumCol_offsets = K / TILE_K;

    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
    
    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
    //
    *Compressed_A = (uint32_t*)malloc(NNZ_AfterPadding * sizeof(uint32_t));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }
    // Generating compressed format for A Matrix
    auto swizzle_layout = composition(Swizzle<3,3,3>{}, make_layout(make_shape(TILE_M, TILE_K), make_stride(TILE_K, 1)));
    assert(M % TILE_M == 0 && K % TILE_K == 0);
    int TotalNZCount = 0;
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR    = A_h + (i * TILE_M) * K + (j * TILE_K);
            int   TileNZCount       = 0;
            int   remainingPaddings = PaddingForEachTile[i * (K / TILE_K) + j];
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                        half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                        *half_ptr        = value;
                        short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                        // Row permutation for bank-conflict-free shared memory layout
                        int      row            = m;
                        int      col            = n;
                        //! swizzle
                        // uint32_t mask           = (row % 8) << 3;
                        // int      col_permutated = col ^ mask;
                        // *short_ptr              = static_cast<short>(row * TILE_K + col_permutated);
                        //! not swizzle
                        // *short_ptr = static_cast<short>(row * TILE_K + col);
                        //! swizzle cute
                        auto coord = make_coord(row, col);
                        auto idx = swizzle_layout(coord);
                        if (row >= 64) {
                            // printf("m = %d, n = %d, idx = %d\n", m, n, idx);
                        }
                        *short_ptr = static_cast<short>(idx);
                        //
                        TileNZCount++;
                        TotalNZCount++;
                    }
                    else {
                        if (remainingPaddings > 0) {
                            remainingPaddings--;
                            half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                            *half_ptr        = value;  // zero
                            short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                            // Row permutation for bank-conflict-free shared memory layout
                            int      row            = m;
                            int      col            = n;
                            uint32_t mask           = (row % 8) << 3;
                            int      col_permutated = col ^ mask;
                            //! swizzle
                            // *short_ptr              = static_cast<short>(row * TILE_K + col_permutated);
                            //! not swizzle
                            // *short_ptr = static_cast<short>(row * TILE_K + col_permutated);
                            //! swizzle cute
                            auto coord = make_coord(row, col);
                            auto idx = swizzle_layout(coord);
                            *short_ptr = static_cast<short>(idx);
                            //
                            TileNZCount++;
                            TotalNZCount++;
                        }
                    }
                }
            }
            //
            assert(TileNZCount % VECTOR_SIZE == 0);
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / VECTOR_SIZE;
        }
    }
    assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / VECTOR_SIZE;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding // adding
                                     // an empty tile at last
    //

    //
    return (M / TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}

void test(int M, int N, int K, bool check) {
    half* B = (half*)malloc(sizeof(half) * K * N);
    half* A = (half*)malloc(sizeof(half) * M * K);
    
    memset(A, 0, sizeof(half) * M * K);
    memset(B, 0, sizeof(half) * K * N);
    float sparsity = 0.9;
    initVector(B, K * N);
    initVector(A, M * K, sparsity);
    
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
    
    float* C_cuda = (float*)malloc(sizeof(float) * M * N);

    {   //! debug
        // printf("host check A\n");
        // for (int k_ptr = 0; k_ptr < K / 16; k_ptr++) {
        //     printf("tile A, k_ptr = %d\n", k_ptr);
        //     for (int row = 0; row < 16; row++) {
        //         for (int col = 0; col < 16; col++) {
        //             printf("%.0f ", (float)A[OFFSET_ROW(row, col + k_ptr * 16, K)]);
        //         } printf("\n");
        //     }
        // }
    }

    {
        //! debug
        // printf("host check B\n");
        // for (int k_ptr = 0; k_ptr < K / 16; k_ptr++) {
        //     printf("tile B, k_ptr = %d\n", k_ptr);
        //     for (int row = 0; row < 16; row++) {
        //         for (int col = 0; col < 8; col++) {
        //             printf("%.0f ", (float)B[OFFSET_ROW(row + k_ptr * 16, col, N)]);
        //         } printf("\n");
        //     }
        // }
    }
    
    // convert to flash-llm
    uint32_t* Compressed_A = NULL;
    int* TileOffsets = NULL;
    int NumOffsets = flashLLMPreprocess_full_matrixA(A, M, N, K, &Compressed_A, &TileOffsets);
    int NNZ = TileOffsets[NumOffsets - 1] * 4;
    printf("NumOffsets: %d, NNZ: %d\n", NumOffsets, NNZ);
    // flashLLMPreprocess_full_matrixA(A, )
    // convertToMETCF(M, N, K, nnz, csrPtrA, csrColIdxA, csrValA, B, C_cuda);
    // flash_llm_test<64, 128, 32>(M, N, K, NumOffsets, NNZ, TileOffsets, Compressed_A, B, C_cuda);
    flash_llm_test<128, 128, 32>(M, N, K, NumOffsets, NNZ, TileOffsets, Compressed_A, B, C_cuda);
    //host check
    if (check) {
        float* C_cpu = (float*)malloc(sizeof(float) * M * N);
        memset(C_cpu, 0, sizeof(float) * M * N);
        cusparse_spmm_all(csrValA, csrPtrA, csrColIdxA, B, C_cpu, M, N, K, nnz);
        verify_new(C_cpu, C_cuda, M, N);
        printf("check nan: cusp %.1f mykernel %.1f\n", C_cpu[0], C_cuda[0]);
    }
}

int main(int argc, char** argv) {
    string filename;
    int N = 1024;
    int M = 1024;
    int K = 1024;
    float sparsity = 0.9;
    int mtx_flag = 1;
    parseInput(argc, argv, filename, "-filename", N, "-N", M, "-M", K, "-K", mtx_flag, "-mtx_flag", sparsity, "-sparsity");
    cout << "Filename: " << filename << endl;
    cout << "N: " << N << endl;

    // convert A to CSR
    int rowA, colA;
    int nnz = 0;
    int isSymmetricA;
    int* csrPtrA;
    int* csrColIdxA;
    half* csrValA;

    if (mtx_flag == 1) {
        // mmio_allinone(&rowA, &colA, &nnz, &isSymmetricA, &csrPtrA, &csrColIdxA, &csrValA, filename.c_str());
        // M = rowA;
        // K = colA;
        // printf("HALF: Sparse A[%d,%d] x B[%d,%d] = C[%d,%d]\n", M, K, K, N, M, N);
        // half* B = (half*)malloc(sizeof(half) * K * N);
        // initVector(B, K * N);
        // initVector(csrValA, nnz);
        // float* C_cuda = (float*)malloc(sizeof(float) * M * N);
        // convertToMETCF(M, N, K, nnz, csrPtrA, csrColIdxA, csrValA, B, C_cuda);

        // // host check
        // float* C_cpu = (float*)malloc(sizeof(float) * M * N);
        // memset(C_cpu, 0, sizeof(half) * M * N);
        // cusparse_spmm_all(csrValA, csrPtrA, csrColIdxA, B, C_cpu, M, N, K, nnz);
        // verify_new(C_cpu, C_cuda, M, N);

        // printf("check nan: cusp %.1f mykernel %.1f\n", C_cpu[0], C_cuda[0]);
    } else {
        // printf("==============================(%d,%d,%d)\n", 2304, 1536, 16384);
        // test(2304, 1536, 16384, false);
        printf("==============================(%d,%d,%d)\n", 1024, 1024, 1024);
        test(1024, 1024, 1024, true);
        // printf("==============================(%d,%d,%d)\n", 128, 128, 128);
        // test(128, 128, 128, true);
        // printf("==============================(%d,%d,%d)\n", 64, 64, 64);
        // test(64, 64, 64, true);
        // printf("==============================(%d,%d,%d)\n", 64, 64, 64);
        // test(64, 64, 64, true);
        // printf("==============================(%d,%d,%d)\n", 32, 32, 32);
        // test(32, 32, 32, true);
        // printf("==============================(%d,%d,%d)\n", 16, 32, 16);
        // test(16, 32, 16, true);
        printf("==============================\n"); 
    }

    return 0;
}