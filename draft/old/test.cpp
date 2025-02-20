#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

template <typename T, size_t TILE_M, size_t TILE_N>
void print_matrix(T (&matrix)[TILE_M][TILE_N]) {
    printf("Matrix size: (%zd x %zd)\n", TILE_M, TILE_N);
    // Example: Print the matrix
#pragma unroll(TILE_M)
    for (int i = 0; i < TILE_M; ++i) {
#pragma unroll(TILE_N)
        for (int j = 0; j < TILE_N; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

struct MmaM16N8K4RowColF32Tf32 {
    static void mma(uint32_t (&frag_d)[4], uint32_t (&frag_a)[2], uint32_t (&frag_b)[1], uint32_t (&frag_c)[4]) {
        printf("D: %u, %u, %u, %u\n", frag_d[0], frag_d[1], frag_d[2], frag_d[3]);
        printf("A: %u, %u\n", frag_a[0], frag_a[1]);
        printf("B: %u\n", frag_b[0]);
        printf("C: %u, %u, %u, %u\n", frag_c[0], frag_c[1], frag_c[2], frag_c[3]);
    }
};

template <typename MMA, typename T,
          // template <typename T,
          size_t M_MMA_COUNT, size_t N_MMA_COUNT,
          size_t A_FRAG_COUNT, size_t B_FRAG_COUNT, size_t C_FRAG_COUNT>
static void mma(T (&frag_c)[M_MMA_COUNT][N_MMA_COUNT][C_FRAG_COUNT],
                T (&frag_a)[M_MMA_COUNT][A_FRAG_COUNT],
                T (&frag_b)[N_MMA_COUNT][B_FRAG_COUNT]) {
#pragma unroll(M_MMA_COUNT)
    for (int i = 0; i < M_MMA_COUNT; i++) {
#pragma unroll(N_MMA_COUNT)
        for (int j = 0; j < N_MMA_COUNT; j++) {
            MMA::mma(
                frag_c[i][j],
                frag_a[i],
                frag_b[j],
                frag_c[i][j]);
        }
    }
}

int main() {
    constexpr size_t TILE_M = 14;
    constexpr size_t TILE_N = 4;
    std::vector<float> mat_data(TILE_M * TILE_N, 3);
    float mat[TILE_M][TILE_N];
    memcpy(mat, mat_data.data(), sizeof(float) * TILE_M * TILE_N);
    print_matrix(mat);

    uint32_t frag_c[4][4][4];
    uint32_t frag_a[4][2];
    uint32_t frag_b[4][1];
    // mma<uint32_t, 4, 4, 2, 1, 4, >(frag_c, frag_a, frag_b);
    mma<MmaM16N8K4RowColF32Tf32>(frag_c, frag_a, frag_b);

    return 0;
}