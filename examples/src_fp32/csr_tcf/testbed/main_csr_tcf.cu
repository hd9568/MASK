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
#include <fstream>
#include <sstream>
#include <filesystem>

#include "common.h"
#include "utils.h"

#include <cute/tensor.hpp>

using namespace std;
using namespace cute;
namespace fs = std::filesystem;

#define OFFSET_ROW(row, col, lda) ((row)*(lda)+(col))
#define ROW_WINDOW_M 16
#define TC_BLOCK_K 8
#define TC_BLOCK_M 16

// Simple MTX reader
static int readMtx(const char* fname, int* M, int* K, int* nnz,
    int** csrRowPtr, int** csrColIdx, float** csrVal, int mtx_flag) {
    ifstream fin(fname);
    if (!fin) { cerr << "Cannot open " << fname << endl; return 0; }
    string line;
    // Skip comments
    while (getline(fin, line)) {
        if (line[0] != '%') break;
    }
    // First non-comment line: M K nnz
    stringstream ss(line);
    int m, k, nz;
    ss >> m >> k >> nz;
    *M = m; *K = k; *nnz = nz;

    vector<int> row_idx(nz), col_idx(nz);
    vector<float> vals(nz);
    for (int i = 0; i < nz; i++) {
        fin >> row_idx[i] >> col_idx[i] >> vals[i];
        row_idx[i]--; col_idx[i]--;  // 1-based to 0-based
    }
    fin.close();

    // Build CSR
    int* rp = new int[m + 1]();
    int* ci = new int[nz];
    float* cv = new float[nz];
    for (int i = 0; i < nz; i++) rp[row_idx[i] + 1]++;
    for (int i = 1; i <= m; i++) rp[i] += rp[i-1];
    vector<int> pos(rp, rp + m + 1);
    for (int i = 0; i < nz; i++) {
        int r = row_idx[i];
        ci[pos[r]] = col_idx[i];
        cv[pos[r]] = vals[i];
        pos[r]++;
    }
    *csrRowPtr = rp;
    *csrColIdx = ci;
    *csrVal = cv;
    return 1;
}

template <int TILE_B>
extern void csr_tcf_spmm_test(int M, int N, int K,
    int TCBlock_num, int nnz,
    int* val_sidx, int* val_soff,
    float* B_val,
    int* val_csr_idx, float* val_csr_val, int* val_csr_off,
    float* C,
    bool is_ncu_test, int warmup_time, int execute_time);

extern template void csr_tcf_spmm_test<64>(int,int,int,int,int,int*,int*,float*,int*,float*,int*,float*,bool,int,int);

extern int sparsene_generated_tile_b();

// Direct wrapper: we only need TILE_B=64
void run_my_kernel(int M, int N, int K, int TCBlock_num, int nnz,
    int* val_sidx, int* val_soff, float* B_val,
    int* val_csr_idx, float* val_csr_val, int* val_csr_off,
    float* C, int warmup, int repeat)
{
    csr_tcf_spmm_test<64>(M, N, K, TCBlock_num, nnz,
        val_sidx, val_soff, B_val,
        val_csr_idx, val_csr_val, val_csr_off,
        C, false, warmup, repeat);
}

// --- simplified ME-TCF preprocessing (reuse from DTC) ---
struct METCFData {
    vector<int> val_sidx;
    vector<int> val_soff;
    vector<int> val_csr_idx;
    vector<float> val_csr_val;
    vector<int> val_csr_off;
    int total_TCBlock_num = 0;
    int nnz_pad = 0;
};

METCFData buildMETCF(int M, int K, const int* csrPtrA, const int* csrColIdxA, const float* csrValA) {
    METCFData meta;
    int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;
    meta.val_soff.assign(row_window_num + 1, 0);
    vector<vector<int>> vec_nnz_per_tc_block(row_window_num);
    vector<int> tc_block_count(row_window_num, 0);
    vector<int> nnz_pad_per_window(row_window_num, 0);

    #pragma omp parallel
    {
        vector<unsigned char> col_mark(K, 0);
        vector<int> unique_cols;
        unique_cols.reserve(ROW_WINDOW_M * TC_BLOCK_K);

        #pragma omp for schedule(static)
        for (int i_rw = 0; i_rw < row_window_num; i_rw++) {
            unique_cols.clear();
            for (int i_row = i_rw * ROW_WINDOW_M; i_row < min((i_rw+1)*ROW_WINDOW_M, M); i_row++) {
                for (int j = csrPtrA[i_row]; j < csrPtrA[i_row+1]; j++) {
                    int col = csrColIdxA[j];
                    if (!col_mark[col]) {
                        col_mark[col] = 1;
                        unique_cols.push_back(col);
                    }
                }
            }
            sort(unique_cols.begin(), unique_cols.end());
            tc_block_count[i_rw] = (unique_cols.size() + TC_BLOCK_K - 1) / TC_BLOCK_K;
            vec_nnz_per_tc_block[i_rw].assign(tc_block_count[i_rw], 0);
            for (int i_row = i_rw * ROW_WINDOW_M; i_row < min((i_rw+1)*ROW_WINDOW_M, M); i_row++) {
                for (int j = csrPtrA[i_row]; j < csrPtrA[i_row+1]; j++) {
                    col_mark[csrColIdxA[j]] = 0;
                }
            }
        }
    }

    int total_blocks = 0;
    for (int i = 0; i < row_window_num; i++) {
        meta.val_soff[i] = total_blocks;
        total_blocks += tc_block_count[i];
    }
    meta.val_soff[row_window_num] = total_blocks;
    meta.total_TCBlock_num = total_blocks;

    meta.val_sidx.assign(TC_BLOCK_K * total_blocks, 0);
    meta.val_csr_off.assign(total_blocks * 2, 0);
    vector<int> tmp_idx;
    vector<float> tmp_val;
    tmp_idx.reserve(total_blocks * ROW_WINDOW_M * TC_BLOCK_K);
    tmp_val.reserve(total_blocks * ROW_WINDOW_M * TC_BLOCK_K);

    int block_id = 0;
    for (int i_rw = 0; i_rw < row_window_num; i_rw++) {
        vector<unsigned char> col_mark(K, 0);
        vector<int> unique_cols;
        for (int i_row = i_rw * ROW_WINDOW_M; i_row < min((i_rw+1)*ROW_WINDOW_M, M); i_row++) {
            for (int j = csrPtrA[i_row]; j < csrPtrA[i_row+1]; j++) {
                int col = csrColIdxA[j];
                if (!col_mark[col]) {
                    col_mark[col] = 1;
                    unique_cols.push_back(col);
                }
            }
        }
        sort(unique_cols.begin(), unique_cols.end());

        for (int b = 0; b < tc_block_count[i_rw]; b++) {
            int col_start = b * TC_BLOCK_K;
            int col_end = min((b+1)*TC_BLOCK_K, (int)unique_cols.size());
            int block_cols = col_end - col_start;
            for (int k = 0; k < block_cols; k++) {
                meta.val_sidx[block_id * TC_BLOCK_K + k] = unique_cols[col_start + k];
            }

            meta.val_csr_off[block_id * 2] = (int)tmp_idx.size();
            int nnz_count = 0;
            for (int i_row = i_rw * ROW_WINDOW_M; i_row < min((i_rw+1)*ROW_WINDOW_M, M); i_row++) {
                for (int j = csrPtrA[i_row]; j < csrPtrA[i_row+1]; j++) {
                    int col = csrColIdxA[j];
                    for (int k = 0; k < block_cols; k++) {
                        if (col == unique_cols[col_start + k]) {
                            // swizzled index
                            int row = i_row - i_rw * ROW_WINDOW_M;
                            int col_in_block = k;
                            int swizzled = row * TC_BLOCK_K + col_in_block;
                            int S = (TC_BLOCK_K == 8) ? 3 : 4;
                            int b1 = ((swizzled >> 1) & 1) ^ ((swizzled >> (1+S)) & 1);
                            int b2 = ((swizzled >> 2) & 1) ^ ((swizzled >> (2+S)) & 1);
                            swizzled = (swizzled & ~6) | (b1<<1) | (b2<<2);
                            tmp_idx.push_back(swizzled);
                            tmp_val.push_back(csrValA[j]);
                            nnz_count++;
                        }
                    }
                }
            }
            meta.val_csr_off[block_id * 2 + 1] = (int)tmp_idx.size();
            vec_nnz_per_tc_block[i_rw][b] = nnz_count;
            block_id++;
        }
        for (int i_row = i_rw * ROW_WINDOW_M; i_row < min((i_rw+1)*ROW_WINDOW_M, M); i_row++) {
            for (int j = csrPtrA[i_row]; j < csrPtrA[i_row+1]; j++) {
                col_mark[csrColIdxA[j]] = 0;
            }
        }
    }

    meta.val_csr_idx.swap(tmp_idx);
    meta.val_csr_val.swap(tmp_val);
    meta.nnz_pad = (int)meta.val_csr_idx.size();
    return meta;
}

int main(int argc, char* argv[]) {
    string filename;
    int N = 512;
    int mtx_flag = 0;
    int warmup = 5;
    int repeat = 20;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-filename") == 0) filename = argv[++i];
        else if (strcmp(argv[i], "-N") == 0) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-mtx_flag") == 0) mtx_flag = atoi(argv[++i]);
        else if (strcmp(argv[i], "-warmup") == 0) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "-repeat") == 0) repeat = atoi(argv[++i]);
    }

    if (filename.empty()) { cerr << "Usage: -filename <mtx>" << endl; return 1; }

    // Read matrix
    int M, K, nnzA;
    int *csrRowPtrA = nullptr, *csrColIdxA = nullptr;
    float *csrValA = nullptr;

    cout << "Reading " << filename << endl;
    readMtx(filename.c_str(), &M, &K, &nnzA, &csrRowPtrA, &csrColIdxA, &csrValA, mtx_flag);
    cout << "M=" << M << " K=" << K << " nnz=" << nnzA << endl;

    // Setup dense B and C
    float *B_val = new float[K * N];
    float *C_cpu = new float[M * N];
    float *C_cuda = new float[M * N];
    for (int i = 0; i < K * N; i++) B_val[i] = (float)(rand() % 5 + 1);

    // cuSPARSE baseline - use device memory
    float *dB_val, *dC_cpu;
    cudaMalloc(&dB_val, sizeof(float) * K * N);
    cudaMalloc(&dC_cpu, sizeof(float) * M * N);
    cudaMemcpy(dB_val, B_val, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    cudaMemset(dC_cpu, 0, sizeof(float) * M * N);

    int *dCsrRowPtrA, *dCsrColIdxA;
    float *dCsrValA;
    cudaMalloc(&dCsrRowPtrA, sizeof(int) * (M + 1));
    cudaMalloc(&dCsrColIdxA, sizeof(int) * nnzA);
    cudaMalloc(&dCsrValA, sizeof(float) * nnzA);
    cudaMemcpy(dCsrRowPtrA, csrRowPtrA, sizeof(int) * (M + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dCsrColIdxA, csrColIdxA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dCsrValA, csrValA, sizeof(float) * nnzA, cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    cusparseCreateCsr(&matA, M, K, nnzA, dCsrRowPtrA, dCsrColIdxA, dCsrValA,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matB, K, N, K, dB_val, CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matC, M, N, M, dC_cpu, CUDA_R_32F, CUSPARSE_ORDER_COL);

    void* dBuffer = nullptr;
    size_t bufferSize = 0;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    cudaEvent_t cu_start, cu_stop;
    cudaEventCreate(&cu_start);
    cudaEventCreate(&cu_stop);
    cudaEventRecord(cu_start);
    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    cudaEventRecord(cu_stop);
    cudaEventSynchronize(cu_stop);
    float cu_ms = 0;
    cudaEventElapsedTime(&cu_ms, cu_start, cu_stop);
    printf("cusparse_time: %f ms\n", cu_ms);

    cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);

    // Copy cuSPARSE result back
    cudaMemcpy(C_cpu, dC_cpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Build ME-TCF metadata
    auto meta = buildMETCF(M, K, csrRowPtrA, csrColIdxA, csrValA);
    cout << "TCBlock_num=" << meta.total_TCBlock_num << " nnz=" << meta.nnz_pad << endl;

    // Run CSR_TCF kernel
    run_my_kernel(M, N, K, meta.total_TCBlock_num, meta.nnz_pad,
        meta.val_sidx.data(), meta.val_soff.data(), B_val,
        meta.val_csr_idx.data(), meta.val_csr_val.data(), meta.val_csr_off.data(),
        C_cuda, warmup, repeat);

    // Verify
    int errors = 0;
    float max_diff = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float cusp_v = C_cpu[OFFSET_ROW(i, j, N)];
            float my_v = C_cuda[OFFSET_ROW(i, j, N)];
            float diff = fabs(cusp_v - my_v);
            if (diff > 0.01) {
                errors++;
                max_diff = max(max_diff, diff);
                if (errors <= 10)
                    printf("Error(%d,%d): cusp(%.1f) mykernel(%.1f)\n", i, j, cusp_v, my_v);
            }
        }
    }
    float sum_cusp = 0, sum_my = 0;
    for (int i = 0; i < M * N; i++) { sum_cusp += fabs(C_cpu[i]); sum_my += fabs(C_cuda[i]); }
    printf("check nan: cusp %.1f mykernel %.1f\n", sum_cusp, sum_my);
    if (errors == 0) printf("PASS: No errors\n");
    else printf("FAIL: %d errors, max_diff=%.3f\n", errors, max_diff);

    delete[] csrRowPtrA; delete[] csrColIdxA; delete[] csrValA;
    delete[] B_val; delete[] C_cpu; delete[] C_cuda;
    return errors ? 1 : 0;
}
