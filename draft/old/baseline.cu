#include <cctype>
#include <cstdint>
#include <cstdlib>

// TF32 m16n8k4
// TILE_M = 16, MMA_M = 16
// TILE_N = 32, MMA_N = 8
// TILE_K = 32, MMA_K = 4
template <size_t TILE_M, size_t TILE_N, size_t TILE_K,
          size_t MMA_M, size_t MMA_N, size_t MMA_K,
          size_t BLOCK_SIZE, size_t WARP_COUNT, size_t WARP_SIZE = 32>
__global__ void mm_fp32_dense(float* a, float* b, float* c, int M, int N, int K,
                              size_t lda, size_t ldb, size_t ldc) {
    __shared__ float* s_a[TILE_M * TILE_K];
    __shared__ float* s_b[TILE_K * TILE_N];
    const size_t bx = blockIdx.x;
    const size_t by = blockIdx.y;
    const size_t tid = threadIdx.x;
    const size_t lid = tid % WARP_SIZE;
    const size_t wid = tid / WARP_SIZE;

    const size_t m_base = TILE_M * bx;
    const size_t n_base = TILE_N * by;

    for (size_t k_base = 0; k_base < K; k_base += TILE_K) {
        // 1.1 load A to shm
        // shared mem A[TILE_M, TILE_K]
        {
            const size_t k_offset = tid;
            const size_t k = k_base + k_offset;
            const size_t m_end = m_base + TILE_M;
            for (size_t m_offset = 0; m_offset < TILE_M; m_offset++) {
                const size_t m = m_base + m_offset;
                s_a[m_offset * TILE_K + k_offset] = a[m * lda + k];
            }
        }

        // 1.2 load B to shm
        // shared mem B[TILE_K, TILE_N]
        {
            const size_t n_offset = tid % TILE_N;
            const size_t n = n_base + n_offset;
            for (size_t k_offset = tid / TILE_N; k_offset < K; k_offset += BLOCK_SIZE / TILE_N) {
                const size_t k = k_base + k_offset;
                s_b[k_offset * TILE_N + n_offset] = b[k * ldb + n];
            }
        }

        // 2.1 alloc reg
        uint32_t frag_A[2];  // 16 * 4  / 32 = 2
        uint32_t frag_B[4];  // 32 * 4  / 32 = 4

        for (size_t k_mma_base = 0; k_mma_base < TILE_K; k_mma_base += MMA_K) {
            // 2.2 laod to reg (f32 -> tf32 here)
            asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(s_a[(lid / 4 + 0 * 8) * TILE_K + tid % 4 + k_base + k_mma_base]));
            asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(s_a[(lid / 4 + 1 * 8) * TILE_K + tid % 4 + k_base + k_mma_base]));
            asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(s_b[(wid * 8) * TILE_N + tid % 4 + k_base + k_mma_base]));
        }

        // 3. mma
        {
            asm volatile(
                "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
                : "r"(frag_A[2]), "r"(frag_A[3]),
                  "r"(frag_B[2]),
                  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
        }

        // 4. write back to global mem
    }
}