#include <cctype>
#include <cstdint>
#include <cstdlib>

#include "mma/mma_f32.cuh"

template <typename T,
          size_t TILE_M, size_t TILE_N, size_t TILE_K,
          size_t MMA_M, size_t MMA_N, size_t MMA_K,
          size_t BLOCK_SIZE, size_t WARP_COUNT, size_t WARP_SIZE>
__device__ static void load_A_gmem_to_smem(T *smem_A, T *gmem_A, size_t m_tile, size_t k_tile) {

}

template <typename T,
          size_t TILE_M, size_t TILE_N, size_t TILE_K,
          size_t MMA_M, size_t MMA_N, size_t MMA_K,
          size_t M_MMA_COUNT, size_t N_MMA_COUNT,
          size_t A_FRAG_COUNT, size_t B_FRAG_COUNT, size_t C_FRAG_COUNT>
__device__ static void load_A_smem_to_reg(T (&frag_a)[M_MMA_COUNT][A_FRAG_COUNT], T *smem_A, size_t m_mma, size_t k_mma) {

}

template <typename T,
          size_t TILE_M, size_t TILE_N, size_t TILE_K,
          size_t MMA_M, size_t MMA_N, size_t MMA_K,
          size_t BLOCK_SIZE, size_t WARP_COUNT, size_t WARP_SIZE>
__device__ static void load_B_gmem_to_smem(T *smem_A, T *gmem_A, size_t n_tile, size_t k_tile) {

}

template <typename T,
          size_t TILE_M, size_t TILE_N, size_t TILE_K,
          size_t MMA_M, size_t MMA_N, size_t MMA_K,
          size_t M_MMA_COUNT, size_t N_MMA_COUNT,
          size_t A_FRAG_COUNT, size_t B_FRAG_COUNT, size_t C_FRAG_COUNT>
__device__ static void load_B_smem_to_reg(T (&frag_b)[N_MMA_COUNT][B_FRAG_COUNT], T *smem_B, size_t n_mma, size_t k_mma) {

}

template <typename T,
          size_t TILE_M, size_t TILE_N, size_t TILE_K,
          size_t MMA_M, size_t MMA_N, size_t MMA_K,
          size_t M_MMA_COUNT, size_t N_MMA_COUNT, size_t C_FRAG_COUNT,
          size_t BLOCK_SIZE, size_t WARP_COUNT, size_t WARP_SIZE>
__device__ static void store_C_reg_to_gmem(T *gmem_C, T (&frag_c)[M_MMA_COUNT][N_MMA_COUNT][C_FRAG_COUNT], size_t m_tile, size_t n_tile) {

}


template <typename T,
          size_t TILE_M, size_t TILE_N, size_t TILE_K,
          size_t MMA_M, size_t MMA_N, size_t MMA_K,
          size_t BLOCK_SIZE, size_t WARP_COUNT, size_t WARP_SIZE>
__device__ static void load_idx(size_t k_tile) {
}

template <typename T, typename MMA,
          size_t M_MMA_COUNT, size_t N_MMA_COUNT,
          size_t A_FRAG_COUNT, size_t B_FRAG_COUNT, size_t C_FRAG_COUNT>
__device__ static void mma(T (&frag_c)[M_MMA_COUNT][N_MMA_COUNT][C_FRAG_COUNT],
                           T (&frag_a)[M_MMA_COUNT][A_FRAG_COUNT],
                           T (&frag_b)[N_MMA_COUNT][B_FRAG_COUNT]) {
#pragma unroll(M_MMA_COUNT)
    for (int i = 0; i < M_MMA_COUNT; i++) {
#pragma unroll(N_MMA_COUNT)
        for (int j = 0; j < N_MMA_COUNT; j++) {
            MMA::mma(frag_c[i][j], frag_a[i], frag_b[j], frag_c[i][j]);
        }
    }
}

// TF32 m16n8k4
// TILE_M = 16, MMA_M = 16
// TILE_N = 32, MMA_N = 8
// TILE_K = 32, MMA_K = 4
template <size_t TILE_M, size_t TILE_N, size_t TILE_K,
          size_t MMA_M, size_t MMA_N, size_t MMA_K,
          size_t BLOCK_SIZE, size_t WARP_COUNT, size_t WARP_SIZE = 32>
__global__ void mm_fp32(float *a, float *b, float *c, int M, int N, int K,
                        size_t lda, size_t ldb, size_t ldc) {
    float A_smem[TILE_M * TILE_K];
    float B_smem[TILE_K * TILE_N];

    const size_t K_TILE_COUNT = K / TILE_K;

    load_aux_op(); // load index / aux (a)
    preload_op(); // fill zeros (b1)
    load_sparse_op(); // load compressed (b2)
    load_dense_op_1(); // decompress (c1)
    load_dense_op_2(); // load decompressed (c2)
    compute_op(); // compute (d)

    for (size_t k_tile = 0; k_tile < K_TILE_COUNT; k_tile++) {
        load_A_gmem_to_smem<float, TILE_M, TILE_N, TILE_K, MMA_M, MMA_N, MMA_K, BLOCK_SIZE, WARP_COUNT, WARP_SIZE>(A_smem, a, 0, k_tile);
        load_B_gmem_to_smem<float, TILE_M, TILE_N, TILE_K, MMA_M, MMA_N, MMA_K, BLOCK_SIZE, WARP_COUNT, WARP_SIZE>(B_smem, b, 0, k_tile);

        uint32_t frag_A[MMA_M][2];
        uint32_t frag_B[MMA_N][1];
        uint32_t frag_C[MMA_M][MMA_N][4];

        mma(frag_C, frag_A, frag_B);
    }

    store_op(); // store (e)
}