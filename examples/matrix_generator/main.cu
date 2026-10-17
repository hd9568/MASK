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
    for (int i = 0; i < length; ++i) {
        if (static_cast<float>(rand()) / RAND_MAX < sparsity) {
            vec[i] = float(0.0f);
        } else {
            vec[i] = float(rand() % 5 + 1);
        }
    }
}

int main(int argc, char** argv) {
    string filename;
    int M = 1024;
    int K = 1024;
    float sparsity = 0.9;
    parseInput(argc, argv, filename, "-filename", M, "-M", K, "-K", sparsity, "-sparsity");

    cout << "Filename: " << filename << endl;

    int nnz = 0;

    float* A = (float*)malloc(sizeof(float) * M * K);
    initVector(A, M * K, sparsity);

    std::vector<int> rows(M * K);
    std::vector<int> cols(M * K);
    std::vector<float> vals(M * K);
    //> convert to coo
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float value = (float)A[i * K + k];
            if (value != 0) {
                rows[nnz] = i + 1;
                cols[nnz] = k + 1;
                vals[nnz] = value;
                nnz++;
            }
        }
    }

    printf("Generated Sparse A[%d,%d] with %d non-zero elements (sparsity: %.2f)\n", M, K, nnz,
           sparsity);

    //> write to file
    FILE* f = fopen(filename.c_str(), "w");
    fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%d %d %d\n", M, K, nnz);
    for (int i = 0; i < nnz; i++) {
        fprintf(f, "%d %d %.9f\n", rows[i], cols[i], vals[i]);
    }
    fclose(f);
    return 0;
}