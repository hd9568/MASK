#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime_api.h>

#include <cute/tensor.hpp>

#if defined(SPARSENE_GENERATED_INC_PATH)
#include SPARSENE_GENERATED_INC_PATH
#elif defined(USE_PATCHED_INC) && __has_include("../../../../python/me_tcf_kernel3_patched.inc")
#include "../../../../python/me_tcf_kernel3_patched.inc"
#else
#include "../../../../python/test/results/dtc.inc"
#endif

#ifndef SPARSENE_GENERATED_TILE_B
#define SPARSENE_GENERATED_TILE_B 64
#endif

#ifndef SPARSENE_GENERATED_B_SMEM_SWIZZLE
#define SPARSENE_GENERATED_B_SMEM_SWIZZLE "swizzle_323"
#endif

#ifndef SPARSENE_GENERATED_COO_RESTORE_SWIZZLE
#define SPARSENE_GENERATED_COO_RESTORE_SWIZZLE "swizzle_123"
#endif

using namespace cute;

#ifndef ROW_WINDOW_M
#define ROW_WINDOW_M 16
#endif
#ifndef TC_BLOCK_K
#define TC_BLOCK_K 8
#endif
#ifndef TC_BLOCK_M
#define TC_BLOCK_M 16
#endif

int sparsene_generated_tile_b() {
    const char* env = std::getenv("SPARSENE_GENERATED_TILE_B");
    if (env != nullptr && env[0] != '\0') {
        const int parsed = std::atoi(env);
        if (parsed == 16 || parsed == 32 || parsed == 64) {
            return parsed;
        }
    }
    return SPARSENE_GENERATED_TILE_B;
}

namespace {

bool should_print_generated_meta() {
    const char* env = std::getenv("SPARSENE_PRINT_GENERATED_META");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

struct GeneratedKernelMetadata {
    std::vector<int> val_sidx_new_t;
    std::vector<int> val_len_offset;
};

GeneratedKernelMetadata convert_old_metadata_to_generated(
    int M,
    int TCBlock_num,
    int nnz,
    const int* val_sidx,
    const int* val_soff,
    const int* /*val_coo_idx*/,
    const float* /*val_coo_val*/,
    const int* /*val_coo_off*/) {
    const int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;

    GeneratedKernelMetadata meta;
    meta.val_sidx_new_t.assign(TC_BLOCK_K * TCBlock_num, 0);
    meta.val_len_offset.assign(row_window_num + 1, 0);

    for (int row_window = 0; row_window <= row_window_num; ++row_window) {
        meta.val_len_offset[row_window] = val_soff[row_window];
    }

    // Keep the original block-major storage. Both reference and generated
    // kernels materialize the tensor as shape [BLK_K, nnz_dim_K] with the
    // default compact layout, so block `j` occupies a contiguous 8-element
    // slice at `base + 8 * j`.
    for (int block_id = 0; block_id < TCBlock_num; ++block_id) {
        for (int lane = 0; lane < TC_BLOCK_K; ++lane) {
            meta.val_sidx_new_t[block_id * TC_BLOCK_K + lane] =
                val_sidx[block_id * TC_BLOCK_K + lane];
        }
    }

    return meta;
}

template <int TILE_B>
void launch_generated_kernel_once(
    int M,
    int N,
    int K,
    int TCBlock_num,
    int nnz,
    int* val_sidx,
    int* val_soff,
    float* B_val,
    int* val_coo_idx,
    float* val_coo_val,
    int* val_coo_off,
    float* C) {
    const int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;
    assert(N % TILE_B == 0);

    GeneratedKernelMetadata meta = convert_old_metadata_to_generated(
        M, TCBlock_num, nnz, val_sidx, val_soff, val_coo_idx, val_coo_val, val_coo_off);
    static bool printed_meta_summary = false;
    if (!printed_meta_summary && should_print_generated_meta()) {
        printed_meta_summary = true;
        printf(
            "[generated-meta] val_len_offset[0..2]=(%d,%d,%d) first_row_blocks=%d "
            "coo_off[0..2]=(%d,%d,%d) sidx[0..3]=(%d,%d,%d,%d)\n",
            meta.val_len_offset.size() > 0 ? meta.val_len_offset[0] : -1,
            meta.val_len_offset.size() > 1 ? meta.val_len_offset[1] : -1,
            meta.val_len_offset.size() > 2 ? meta.val_len_offset[2] : -1,
            meta.val_len_offset.size() > 1 ? (meta.val_len_offset[1] - meta.val_len_offset[0]) : -1,
            TCBlock_num > 0 ? val_coo_off[0] : -1,
            TCBlock_num > 0 ? val_coo_off[1] : -1,
            TCBlock_num > 1 ? val_coo_off[2] : -1,
            meta.val_sidx_new_t.size() > 0 ? meta.val_sidx_new_t[0] : -1,
            meta.val_sidx_new_t.size() > 1 ? meta.val_sidx_new_t[1] : -1,
            meta.val_sidx_new_t.size() > 2 ? meta.val_sidx_new_t[2] : -1,
            meta.val_sidx_new_t.size() > 3 ? meta.val_sidx_new_t[3] : -1);
    }
    int* dval_sidx_new = nullptr;
    int* dval_len_offset = nullptr;
    int* dval_coo_off_new = nullptr;
    int* dval_coo_idx_new = nullptr;
    float* dval_coo_val_new = nullptr;
    float* dB_val = nullptr;
    float* dC = nullptr;

    CHECK_CUDA2(cudaMalloc(&dval_sidx_new, sizeof(int) * meta.val_sidx_new_t.size()));
    CHECK_CUDA2(cudaMalloc(&dval_len_offset, sizeof(int) * meta.val_len_offset.size()));
    CHECK_CUDA2(cudaMalloc(&dval_coo_off_new, sizeof(int) * (2 * TCBlock_num + 1)));
    CHECK_CUDA2(cudaMalloc(&dval_coo_idx_new, sizeof(int) * nnz));
    CHECK_CUDA2(cudaMalloc(&dval_coo_val_new, sizeof(float) * nnz));
    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(float) * K * N));
    CHECK_CUDA2(cudaMalloc(&dC, sizeof(float) * row_window_num * ROW_WINDOW_M * N));

    CHECK_CUDA2(cudaMemcpy(
        dval_sidx_new,
        meta.val_sidx_new_t.data(),
        sizeof(int) * meta.val_sidx_new_t.size(),
        cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(
        dval_len_offset,
        meta.val_len_offset.data(),
        sizeof(int) * meta.val_len_offset.size(),
        cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(
        dval_coo_off_new,
        val_coo_off,
        sizeof(int) * (2 * TCBlock_num + 1),
        cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(
        dval_coo_idx_new,
        val_coo_idx,
        sizeof(int) * nnz,
        cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(
        dval_coo_val_new,
        val_coo_val,
        sizeof(float) * nnz,
        cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * row_window_num * ROW_WINDOW_M * N));

    dim3 grid(row_window_num, N / TILE_B);
    dim3 block(32);

    constexpr auto blk_mnk = make_shape(Int<TC_BLOCK_M>{}, Int<TILE_B>{}, Int<TC_BLOCK_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(_16{}, _8{}, _8{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk =
        make_shape(Int<TC_BLOCK_M / 16>{}, Int<TILE_B / 8>{}, Int<TC_BLOCK_K / 8>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    const int row_window_offsets_len = row_window_num + 1;
    generated_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>
        <<<grid, block>>>(
            dB_val,
            dC,
            dval_coo_idx_new,
            dval_coo_off_new,
            dval_coo_val_new,
            dval_len_offset,
            dval_sidx_new,
            K,
            M,
            row_window_offsets_len,
            N,
            nnz,
            TCBlock_num
        );
}

template <int TILE_B>
void run_generated_kernel(
    int M,
    int N,
    int K,
    int TCBlock_num,
    int nnz,
    int* val_sidx,
    int* val_soff,
    float* B_val,
    int* val_coo_idx,
    float* val_coo_val,
    int* val_coo_off,
    float* C,
    int warmup_time,
    int execute_time) {
    const int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;
    assert(N % TILE_B == 0);

    GeneratedKernelMetadata meta = convert_old_metadata_to_generated(
        M, TCBlock_num, nnz, val_sidx, val_soff, val_coo_idx, val_coo_val, val_coo_off);
    static bool printed_meta_summary = false;
    if (!printed_meta_summary && should_print_generated_meta()) {
        printed_meta_summary = true;
        printf(
            "[generated-meta] val_len_offset[0..2]=(%d,%d,%d) first_row_blocks=%d "
            "coo_off[0..2]=(%d,%d,%d) sidx[0..3]=(%d,%d,%d,%d)\n",
            meta.val_len_offset.size() > 0 ? meta.val_len_offset[0] : -1,
            meta.val_len_offset.size() > 1 ? meta.val_len_offset[1] : -1,
            meta.val_len_offset.size() > 2 ? meta.val_len_offset[2] : -1,
            meta.val_len_offset.size() > 1 ? (meta.val_len_offset[1] - meta.val_len_offset[0]) : -1,
            TCBlock_num > 0 ? val_coo_off[0] : -1,
            TCBlock_num > 0 ? val_coo_off[1] : -1,
            TCBlock_num > 1 ? val_coo_off[2] : -1,
            meta.val_sidx_new_t.size() > 0 ? meta.val_sidx_new_t[0] : -1,
            meta.val_sidx_new_t.size() > 1 ? meta.val_sidx_new_t[1] : -1,
            meta.val_sidx_new_t.size() > 2 ? meta.val_sidx_new_t[2] : -1,
            meta.val_sidx_new_t.size() > 3 ? meta.val_sidx_new_t[3] : -1);
    }
    int* dval_sidx_new = nullptr;
    int* dval_len_offset = nullptr;
    int* dval_coo_off_new = nullptr;
    int* dval_coo_idx_new = nullptr;
    float* dval_coo_val_new = nullptr;
    float* dB_val = nullptr;
    float* dC = nullptr;

    CHECK_CUDA2(cudaMalloc(&dval_sidx_new, sizeof(int) * meta.val_sidx_new_t.size()));
    CHECK_CUDA2(cudaMalloc(&dval_len_offset, sizeof(int) * meta.val_len_offset.size()));
    CHECK_CUDA2(cudaMalloc(&dval_coo_off_new, sizeof(int) * (2 * TCBlock_num + 1)));
    CHECK_CUDA2(cudaMalloc(&dval_coo_idx_new, sizeof(int) * nnz));
    CHECK_CUDA2(cudaMalloc(&dval_coo_val_new, sizeof(float) * nnz));
    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(float) * K * N));
    CHECK_CUDA2(cudaMalloc(&dC, sizeof(float) * row_window_num * ROW_WINDOW_M * N));

    CHECK_CUDA2(cudaMemcpy(
        dval_sidx_new,
        meta.val_sidx_new_t.data(),
        sizeof(int) * meta.val_sidx_new_t.size(),
        cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(
        dval_len_offset,
        meta.val_len_offset.data(),
        sizeof(int) * meta.val_len_offset.size(),
        cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(
        dval_coo_off_new,
        val_coo_off,
        sizeof(int) * (2 * TCBlock_num + 1),
        cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(
        dval_coo_idx_new,
        val_coo_idx,
        sizeof(int) * nnz,
        cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(
        dval_coo_val_new,
        val_coo_val,
        sizeof(float) * nnz,
        cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * row_window_num * ROW_WINDOW_M * N));

    dim3 grid(row_window_num, N / TILE_B);
    dim3 block(32);

    constexpr auto blk_mnk = make_shape(Int<TC_BLOCK_M>{}, Int<TILE_B>{}, Int<TC_BLOCK_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(_16{}, _8{}, _8{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk =
        make_shape(Int<TC_BLOCK_M / 16>{}, Int<TILE_B / 8>{}, Int<TC_BLOCK_K / 8>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    const int row_window_offsets_len = row_window_num + 1;
    auto launch = [&]() {
        generated_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>
            <<<grid, block>>>(
                dB_val,
                dC,
                dval_coo_idx_new,
                dval_coo_off_new,
                dval_coo_val_new,
                dval_len_offset,
                dval_sidx_new,
                K,
                M,
                row_window_offsets_len,
                N,
                nnz,
                TCBlock_num
            );
    };

    if (execute_time > 0) {
        cudaEvent_t start_event = nullptr;
        cudaEvent_t stop_event = nullptr;
        CHECK_CUDA2(cudaEventCreate(&start_event));
        CHECK_CUDA2(cudaEventCreate(&stop_event));
        for (int i = 0; i < warmup_time; ++i) {
            launch();
        }
        CHECK_CUDA2(cudaDeviceSynchronize());

        CHECK_CUDA2(cudaEventRecord(start_event));
        for (int i = 0; i < execute_time; ++i) {
            launch();
        }
        CHECK_CUDA2(cudaEventRecord(stop_event));
        CHECK_CUDA2(cudaEventSynchronize(stop_event));

        float elapsed_ms = 0.0f;
        CHECK_CUDA2(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        double mykernel_time = static_cast<double>(elapsed_ms) / execute_time;
        printf("mykernel_time: %f ms\n", mykernel_time);
        CHECK_CUDA2(cudaEventDestroy(start_event));
        CHECK_CUDA2(cudaEventDestroy(stop_event));
        CHECK_CUDA2(cudaMemset(dC, 0, sizeof(float) * row_window_num * ROW_WINDOW_M * N));
    }

    launch();
    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    CHECK_CUDA2(cudaFree(dval_sidx_new));
    CHECK_CUDA2(cudaFree(dval_len_offset));
    CHECK_CUDA2(cudaFree(dval_coo_off_new));
    CHECK_CUDA2(cudaFree(dval_coo_idx_new));
    CHECK_CUDA2(cudaFree(dval_coo_val_new));
    CHECK_CUDA2(cudaFree(dB_val));
    CHECK_CUDA2(cudaFree(dC));
}

}  // namespace

template <int TILE_B>
void dtc_spmm_test_val_sidx_bind_reorder(int M,
                                         int N,
                                         int K,
                                         int TCBlock_num,
                                         int nnz,
                                         int* val_sidx,
                                         int* val_soff,
                                         float* B_val,
                                         int* val_coo_idx,
                                         float* val_coo_val,
                                         int* val_coo_off,
                                         float* C,
                                         bool /*is_ncu_test*/,
                                         int warmup_time,
                                         int execute_time) {
    run_generated_kernel<TILE_B>(
        M,
        N,
        K,
        TCBlock_num,
        nnz,
        val_sidx,
        val_soff,
        B_val,
        val_coo_idx,
        val_coo_val,
        val_coo_off,
        C,
        warmup_time,
        execute_time);
}

template void dtc_spmm_test_val_sidx_bind_reorder<64>(int M,
                                                      int N,
                                                      int K,
                                                      int TCBlock_num,
                                                      int nnz,
                                                      int* val_sidx,
                                                      int* val_soff,
                                                      float* B_val,
                                                      int* val_coo_idx,
                                                      float* val_coo_val,
                                                      int* val_coo_off,
                                                      float* C,
                                                      bool is_ncu_test,
                                                      int warmup_time,
                                                      int execute_time);

template void dtc_spmm_test_val_sidx_bind_reorder<16>(int M,
                                                      int N,
                                                      int K,
                                                      int TCBlock_num,
                                                      int nnz,
                                                      int* val_sidx,
                                                      int* val_soff,
                                                      float* B_val,
                                                      int* val_coo_idx,
                                                      float* val_coo_val,
                                                      int* val_coo_off,
                                                      float* C,
                                                      bool is_ncu_test,
                                                      int warmup_time,
                                                      int execute_time);

template void dtc_spmm_test_val_sidx_bind_reorder<32>(int M,
                                                      int N,
                                                      int K,
                                                      int TCBlock_num,
                                                      int nnz,
                                                      int* val_sidx,
                                                      int* val_soff,
                                                      float* B_val,
                                                      int* val_coo_idx,
                                                      float* val_coo_val,
                                                      int* val_coo_off,
                                                      float* C,
                                                      bool is_ncu_test,
                                                      int warmup_time,
                                                      int execute_time);
