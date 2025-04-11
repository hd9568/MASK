#include "kernel-gen.inc"

template <int TILE_M, int TILE_N, int TILE_K>
void ncu_test_all(
    int threadblock_num_x, int threadblock_num_y, int thread_num,
    int M, int N, int K, int NumOffsets, int NNZ,
    int* dTileOffsets, 
    uint32_t* dCompressed_A,
    half* dB_val,
    float* dC
) {
    dim3 grid(threadblock_num_x, threadblock_num_y);
    dim3 block(thread_num);

    constexpr auto blk_mnk = make_shape(Int<TILE_M>{}, Int<TILE_N>{}, Int<TILE_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(_16{}, _8{}, _16{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk = make_shape(Int<TILE_M / 16>{}, Int<TILE_N / 8>{}, Int<TILE_K / 16>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_2{}, _2{}, _1{});
    using WARP_MNK = decltype(warp_mnk);


    // constexpr auto compressed_A_reg_size_ins = make_shape(Int<(TILE_M * TILE_K + 32 - 1) / 32>{});
    // using compressed_A_reg_size = decltype(compressed_A_reg_size_ins);
    // constexpr auto tile_val_idx_bind_size_ins = make_shape(Int<get<0>(BLK_MNK{}) * get<2>(BLK_MNK{})>{});
    // using tile_val_idx_bind_size = decltype(tile_val_idx_bind_size_ins);
    // class REGA_RMEM_SET, class REGB_RMEM_SET, class compressed_A_reg_set
    constexpr auto compressed_A_reg_set_ins = make_shape(Int<DIVUP(DIVUP(Int<TILE_M>{} * Int<TILE_K>{}, 128), 4)>{});
    using compressed_A_reg_set = decltype(compressed_A_reg_set_ins);
    constexpr auto REGA_RMEM_SET_INS = make_shape(Int<get<0>(BLK_MMA_MNK{}) / get<0>(WARP_MNK{})>{});
    using REGA_RMEM_SET = decltype(REGA_RMEM_SET_INS);
    constexpr auto REGB_RMEM_SET_INS = make_shape(Int<get<1>(BLK_MMA_MNK{}) / get<1>(WARP_MNK{})>{});
    using REGB_RMEM_SET = decltype(REGB_RMEM_SET_INS);

    flash_llm_spmm_kernel_sstage_rstage<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, REGA_RMEM_SET, REGB_RMEM_SET, compressed_A_reg_set>
    <<<grid, block>>>(dB_val, dC, dCompressed_A, dTileOffsets, K, M, N, NumOffsets, NNZ);
}

template <int TILE_M, int TILE_N, int TILE_K>
void flash_llm_test(
    int M, int N, int K,
    int NumOffsets,
    int NNZ,
    int* TileOffsets,
    uint32_t* Compressed_A,
    half* B_val,
    float* C
) {
    int* dTileOffsets;
    uint32_t* dCompressed_A;
    half* dB_val;
    float* dC;

    {//! debug
        // for (int i = 0; i < 120 * 4; i++) {
        //     // if (i % 16 == 0) printf("\n");
        //     printf("%x ", Compressed_A[i]);
            
        // }
        // printf("\n");
        // for (int i = 120 * 4; i < 240 * 4; i++) {
        //     // if (i % 16 == 0) printf("\n");
        //     printf("%x ", Compressed_A[i]);
            
        // }
        // printf("\n");
    }

    CHECK_CUDA2(cudaMalloc(&dTileOffsets, sizeof(int) * NumOffsets));
    CHECK_CUDA2(cudaMalloc(&dCompressed_A, sizeof(uint32_t) * NNZ));
    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(half) * K * N));
    CHECK_CUDA2(cudaMalloc(&dC, sizeof(float) * M * N));

    CHECK_CUDA2(cudaMemcpy(dTileOffsets, TileOffsets, sizeof(int) * NumOffsets, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dCompressed_A, Compressed_A, sizeof(uint32_t) * NNZ, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(half) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * M * N));

    int threadblock_num_x = M / TILE_M;
    int threadblock_num_y = N / TILE_N;
    int thread_num = 128;

    ncu_test_all<TILE_M, TILE_N, TILE_K>(threadblock_num_x, threadblock_num_y, thread_num, M, N, K, NumOffsets, NNZ, dTileOffsets, dCompressed_A, dB_val, dC);

    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
}

template void flash_llm_test<64, 64, 32>(
    int M, int N, int K,
    int NumOffsets,
    int NNZ,
    int* TileOffsets,
    uint32_t* Compressed_A,
    half* B_val,
    float* C
);