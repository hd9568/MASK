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

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))
#define OFFSET_COL(row, col, lda) ((col) * (lda) + (row))

#define DIVUP(a, b) (((a) - 1) / (b) + 1)
#define ALIGN(a, b) (DIVUP(a, b) * (b))

template <class dtypeA,
          class dtypeB,
          class dtypeC,
          class dtypeMask,
          int Tile_M,
          int Tile_N,
          int Tile_K,
          int Mma_M,
          int Mma_N,
          int Mma_K>
extern void acc_spmm_test(int M,
                          int N,
                          int K,
                          dtypeB* B_val,
                          dtypeC* C_val,
                          int* val_sidx,
                          int* val_soff,
                          int* val_mco_off,
                          dtypeMask* val_mco_mask,
                          dtypeA* val_mco_val,
                          bool is_ncu_test,
                          int warmup_time,
                          int execute_time);

extern template void
acc_spmm_test<half, half, float, uint64_t, 16, 64, 16, 16, 8, 16>(int M,
                                                                  int N,
                                                                  int K,
                                                                  half* B_val,
                                                                  float* C_val,
                                                                  int* val_sidx,
                                                                  int* val_soff,
                                                                  int* val_mco_off,
                                                                  uint64_t* val_mco_mask,
                                                                  half* val_mco_val,
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

vector<uint64_t> convertToUInt64Vector(const vector<bool>& bitset) {
    vector<uint64_t> uint64Vector;
    uint64_t current = 0; // Holds the current 64-bit chunk
    int bitIndex = 0;     // Tracks the bit position in the current uint64_t

    for (size_t i = 0; i < bitset.size(); ++i) {
        // Set the corresponding bit in the current uint64_t
        current |= (static_cast<uint64_t>(bitset[i]) << bitIndex);

        // Move to the next bit in the uint64_t
        ++bitIndex;

        // If we've filled 64 bits, push the value to the result and reset
        if (bitIndex == 64) {
            uint64Vector.push_back(current);
            current = 0;
            bitIndex = 0;
        }
    }

    // If there are remaining bits, push the last uint64_t
    if (bitIndex > 0) {
        uint64Vector.push_back(current);
    }

    return uint64Vector;
}

void convertToAcc(int M,
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
    using dtypeA = half;
    using dtypeB = half;
    using dtypeC = float;
    using dtypeMask = uint64_t;

    // 1. Create ptrs
    dtypeB* B_val;
    dtypeC* C_val;
    int* val_sidx;
    int* val_soff;
    int* val_mco_off;
    dtypeMask* val_mco_mask;
    dtypeA* val_mco_val;

    B_val = B;
    C_val = C;

    const int Blk_M = 16;
    const int Blk_K = 16;

    // 2. Convert
    // auto swizzler =
    //   cute::composition(cute::Swizzle<1, 3, 3>{}, cute::make_layout(cute::make_shape(Blk_M, Blk_K),
    //                                             cute::make_stride(Blk_K, 1)));
    auto swizzler =
            cute::make_layout(cute::make_shape(Blk_M, Blk_K),
            cute::make_stride(Blk_K, 1));
    int num_row_windows = DIVUP(M, Blk_M);

    vector<vector<int>> nz_blk_nz_cols(num_row_windows);
    vector<vector<uint64_t>> nz_blk_mask;
    vector<vector<dtypeA>> nz_blk_val;

    for (int rw_idx = 0; rw_idx < num_row_windows; rw_idx++) {
        int row_start = rw_idx * Blk_M;
        int row_end = min(row_start + Blk_M, M);

        // all nz cols in the row window
        map<int, map<int, dtypeA>> nz_cols;
        for (int row = row_start; row < row_end; row++) {
            for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                int col = csrColIdxA[j];
                dtypeA val = csrValA[j];
                nz_cols[col][row] = val;
            }
        }

        // construct nz blocks by condensing nz cols
        vector<map<int, dtypeA>> nz_blks;
        nz_blks.emplace_back();
        int condensed_col = 0;
        for (auto& [col, row_vals] : nz_cols) {
            nz_blk_nz_cols[rw_idx].push_back(col);
            auto& current_blk = nz_blks.back();
            for (auto& [row, val] : row_vals) {
                //! manual swizzle
                int idx = swizzler(cute::make_coord(row % Blk_M, condensed_col % Blk_K));
                //! no pre
                // int idx = (row % Blk_M) * Blk_K + condensed_col % Blk_K;
                current_blk[idx] = val;
            }
            condensed_col++;
            if (condensed_col == Blk_K) {
                nz_blks.emplace_back();
                condensed_col = 0;
            }
        }
        if (nz_blks.back().size() == 0) {
            nz_blks.pop_back();
        }
        while (nz_blk_nz_cols[rw_idx].size() % Blk_K != 0) {
            nz_blk_nz_cols[rw_idx].push_back(nz_blk_nz_cols[rw_idx].back());
        }

        // extract bitmasks and values from nz blocks
        for (auto& nz_blk : nz_blks) {
            vector<bool> bitmask(Blk_M * Blk_K, false);
            vector<dtypeA> vals;
            for (auto& [idx, val] : nz_blk) {
                bitmask[idx] = true;
                vals.push_back(val);
            }
            nz_blk_mask.push_back(convertToUInt64Vector(bitmask));
            while (vals.size() % 8 != 0) {
                // padding with 0 to align with 8*fp16 (16 bytes)
                vals.push_back(half(0.0f));
            }
            nz_blk_val.push_back(vals);
        }
    }

    // 3. Accumulate offsets and flatten values
    vector<int> vval_soff;
    vector<int> vval_sidx;
    vector<int> vval_mco_off;
    vector<uint64_t> vval_mco_mask;
    vector<dtypeA> vval_mco_val;

    int soff_acc = 0;
    int mco_off_acc = 0;
    int blk_ptr = 0;
    for (int rw_idx = 0; rw_idx < num_row_windows; rw_idx++) {
        vval_soff.push_back(soff_acc);
        vval_sidx.insert(vval_sidx.end(), nz_blk_nz_cols[rw_idx].begin(),
                         nz_blk_nz_cols[rw_idx].end());
        int n_blks = DIVUP(nz_blk_nz_cols[rw_idx].size(), Blk_K);
        soff_acc += n_blks;
        for (int i = 0; i < n_blks; i++) {
            vval_mco_off.push_back(mco_off_acc);
            vval_mco_val.insert(vval_mco_val.end(), nz_blk_val[blk_ptr].begin(),
                                nz_blk_val[blk_ptr].end());
            mco_off_acc += nz_blk_val[blk_ptr].size();
            vval_mco_mask.insert(vval_mco_mask.end(), nz_blk_mask[blk_ptr].begin(),
                                 nz_blk_mask[blk_ptr].end());
            blk_ptr++;
        }
    }
    vval_soff.push_back(soff_acc);
    vval_mco_off.push_back(mco_off_acc);

    val_soff = vval_soff.data();
    val_sidx = vval_sidx.data();
    val_mco_off = vval_mco_off.data();
    val_mco_mask = vval_mco_mask.data();
    val_mco_val = vval_mco_val.data();

    // 4. Call the kernel
    acc_spmm_test<dtypeA, dtypeB, dtypeC, dtypeMask, 16, 64, 16, 16, 8, 16>(
      M, N, K, B_val, C_val, val_sidx, val_soff, val_mco_off, val_mco_mask, val_mco_val,
      is_ncu_test, warmup_time, execute_time);
}

void verify_new(float* C_cpu, float* C_cuda, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float cusparse_val = C_cpu[OFFSET_ROW(i, j, N)];
            float mykernel_val = C_cuda[OFFSET_ROW(i, j, N)];
            if (fabs(cusparse_val - mykernel_val) > 0.01) {
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
               sparsity, "-sparsity", is_ncu_test, "-ncu", warmup_time, "-warmup",
               execute_time, "-repeat");

    cout << "Filename: " << filename << endl;
    cout << "N: " << N << endl;

    int rowA, colA;
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

    float* C_cuda = (float*)malloc(sizeof(float) * M * N);
    convertToAcc(M, N, K, nnz, csrPtrA, csrColIdxA, csrValA, B, C_cuda, (bool)is_ncu_test,
                 warmup_time, execute_time);

    // host check
    float* C_cpu = (float*)malloc(sizeof(float) * M * N);
    memset(C_cpu, 0, sizeof(half) * M * N);
    cusparse_spmm_all(csrValA, csrPtrA, csrColIdxA, B, C_cpu, M, N, K, nnz);
    verify_new(C_cpu, C_cuda, M, N);

    printf("check nan: cusp %.1f mykernel %.1f\n", C_cpu[0], C_cuda[0]);
    return 0;
}