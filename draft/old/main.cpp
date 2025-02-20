#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

int main() {
    int M, N, K;
    std::cin >> M >> N >> K;

    float* A = cudaMallocHost(sizeof(float) * M * K);
    float* B = cudaMallocHost(sizeof(float) * K * N);
    float* C = cudaMallocHost(sizeof(float) * M * N);
    float* dA = cudaMalloc(sizeof(float) * M * K);
    float* dB = cudaMalloc(sizeof(float) * K * K);
    float* dC = cudaMalloc(sizeof(float) * M * N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
        }
    }

    return 0;
}