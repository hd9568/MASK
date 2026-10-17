#include "kernel-gen.inc"

template <int TILE_B>
void ncu_test_all(
    int threadblock_num_x, int threadblock_num_y, int thread_num,
    int M, int N, int K, int Mo, int nnz_block, int nnz,
    int* dval_sidx,
    int* dval_soff,
    half* dB_val,
    int* dval_coo_idx,
    half* dval_coo_val,
    int* dval_coo_off,
    float* dC
) {
    dim3 grid(threadblock_num_x, threadblock_num_y);
    dim3 block(thread_num);


    // constexpr int ROW_WINDOW_M = 16;
    constexpr int TC_BLOCK_K = 16;
    constexpr int TC_BLOCK_M = 16;

    constexpr auto blk_mnk = make_shape(Int<TC_BLOCK_M>{}, Int<TILE_B>{}, Int<TC_BLOCK_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(_16{}, _8{}, _16{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk = make_shape(Int<TC_BLOCK_M / 16>{}, Int<TILE_B / 8>{}, Int<TC_BLOCK_K / 16>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    using tile_sidx_size = Int<TC_BLOCK_K>;
    using tile_coo_val_size = Int<TC_BLOCK_M * TC_BLOCK_K>;

    // constexpr int shift1 = 1;
    // constexpr int shift2 = 1;
    // constexpr int shift3 = 1;
    // constexpr int maxshift1_2 = (shift1 > shift2 ? shift1 : shift2);
    // constexpr int nbuf_rmem = (maxshift1_2 > shift3 ? maxshift1_2 : shift3) + 1;
    // constexpr int nbuf_smem = nbuf_rmem;
    // constexpr int nbuf_smem = maxshift1_2 + 1;

    //! （shift1=1,shift2=1,shift3=1)
    dtc_spmm_kernel_sstage_rstage<
        BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK,
        tile_coo_val_size, tile_sidx_size
    ><<<grid, block>>>(dB_val, dC, dval_coo_idx, dval_coo_off, dval_coo_val, dval_sidx, dval_soff, K, M, Mo, N, nnz, nnz_block);
}

#define DIVUP(a, b) (((a) - 1) / (b) + 1)
#define ALIGN(a,b) (DIVUP(a,b)*(b))


template <int TILE_B>
void dtc_spmm_test(
    int M, int N, int K,
    int TCBlock_num,
    int nnz,
    int* val_sidx,
    int* val_soff,
    half* B_val,
    int* val_coo_idx,
    half* val_coo_val,
    int* val_coo_off,
    float* C
) {
    int* dval_sidx;
    int* dval_soff;
    half* dB_val;
    int* dval_coo_idx;
    half* dval_coo_val;
    int* dval_coo_off;
    float* dC;

    constexpr int ROW_WINDOW_M = 16;
    constexpr int TC_BLOCK_K = 16;
    // constexpr int TC_BLOCK_M = 16;

    int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;

    printf("dval_sidx(load as int4): alloc TCBlock_num * TC_Block_K = %d\n", TCBlock_num * TC_BLOCK_K);
    printf("dval_soff: alloc row_window_num + 1 = %d, alloc size = %d\n", row_window_num + 1, sizeof(int) * (row_window_num + 1));
    printf("dB_val alloc: K * N = %d, alloc size = %d\n", K * N, sizeof(half) * K * N);
    printf("dval_coo_idx(load 8 int) alloc: nnz = %d, ceil(8, nnz)=%d, alloc size = %d\n", nnz, (nnz + 8 - 1) / 8 * 8, (nnz + 8 - 1) / 8 * 8 * sizeof(int));
    printf("dval_coo_val(load 8 half) alloc: nnz = %d, ceil(8, nnz)=%d, alloc size = %d\n", nnz, (nnz + 8 - 1) / 8 * 8, (nnz + 8 - 1) / 8 * 8 * sizeof(half));
    printf("dval_coo_off alloc: TCBlock_num + 1 = %d, alloc size = %d\n", 2 * TCBlock_num + 1,  sizeof(int) * (2 * TCBlock_num + 1));
    printf("dC alloc: M * N = %d, alloc size=%d\n", M * N, sizeof(float) * M * N);


    CHECK_CUDA2(cudaMalloc(&dval_sidx, sizeof(int) * TCBlock_num * TC_BLOCK_K));

    CHECK_CUDA2(cudaMalloc(&dval_soff, sizeof(int) * (row_window_num + 1)));
    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(half) * K * N));
    CHECK_CUDA2(cudaMalloc(&dval_coo_idx, (nnz + 8 - 1) / 8 * 8 * sizeof(int)));
    CHECK_CUDA2(cudaMemset(dval_coo_idx, 0, (nnz + 8 - 1) / 8 * 8 * sizeof(int)));
    CHECK_CUDA2(cudaMalloc(&dval_coo_val, (nnz + 8 - 1) / 8 * 8 * sizeof(half)));
    CHECK_CUDA2(cudaMemset(dval_coo_val, 0, (nnz + 8 - 1) / 8 * 8 * sizeof(half)));
    CHECK_CUDA2(cudaMalloc(&dval_coo_off, sizeof(int) * (2 * TCBlock_num + 1)));
    CHECK_CUDA2(cudaMalloc(&dC, sizeof(float) * M * N));
    CHECK_CUDA2(cudaMemcpy(dval_sidx, val_sidx, sizeof(int) * TCBlock_num * TC_BLOCK_K, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_soff, val_soff, sizeof(int) * (row_window_num + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(half) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_coo_idx, val_coo_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_coo_val, val_coo_val, sizeof(half) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_coo_off, val_coo_off, sizeof(int) * (2 * TCBlock_num + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * M * N));

    int threadblock_num_x = row_window_num;
    int threadblock_num_y = N / TILE_B;
    int thread_num = 32;

    const int Mo = DIVUP(M, ROW_WINDOW_M);
    const int nnz_block = val_soff[Mo];
    const int nnz_aligned = ALIGN(val_coo_off[nnz_block * 2 - 1], 8);

    ncu_test_all<TILE_B>(threadblock_num_x, threadblock_num_y, thread_num, M, N, K, Mo, nnz_block, nnz_aligned, dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC);
    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

}

template void dtc_spmm_test<32>(
    int, int, int,
    int, int,
    int*, int*, half*,
    int*, half*,
    int*, float*
);
