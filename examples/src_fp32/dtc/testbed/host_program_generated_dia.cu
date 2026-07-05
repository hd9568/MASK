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

#include "../../../../python/me_tcf_kernel3_dia.inc"

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

struct GeneratedKernelMetadata {
    std::vector<int> val_sidx_new_t;
    std::vector<int> val_len_offset;
    std::vector<int> val_dia_len_new_offset;
    std::vector<int> val_dia_idx_new_1;   // same length as val_dia_val_new_1
    std::vector<float> val_dia_val_new_1;
};

inline int dia_deswizzle_idx(int p, int S) {
    int b1 = ((p >> 1) & 1) ^ ((p >> (1 + S)) & 1);
    int b2 = ((p >> 2) & 1) ^ ((p >> (2 + S)) & 1);
    int bits = (b1 << 1) | (b2 << 2);
    return (p & ~6) | bits;
}

GeneratedKernelMetadata convert_old_metadata_to_generated(
    int M,
    int TCBlock_num,
    int nnz,
    const int* val_sidx,
    const int* val_soff,
    const int* val_coo_idx,
    const float* val_coo_val,
    const int* val_coo_off) {
    const int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;

    GeneratedKernelMetadata meta;
    meta.val_sidx_new_t.assign(TC_BLOCK_K * TCBlock_num, 0);
    meta.val_len_offset.assign(row_window_num + 1, 0);
    meta.val_dia_len_new_offset.assign(TCBlock_num + 1, 0);
    meta.val_dia_idx_new_1.reserve(nnz);
    meta.val_dia_val_new_1.reserve(nnz);

    for (int row_window = 0; row_window <= row_window_num; ++row_window) {
        meta.val_len_offset[row_window] = val_soff[row_window];
    }

    for (int block_id = 0; block_id < TCBlock_num; ++block_id) {
        for (int lane = 0; lane < TC_BLOCK_K; ++lane) {
            meta.val_sidx_new_t[block_id * TC_BLOCK_K + lane] =
                val_sidx[block_id * TC_BLOCK_K + lane];
        }
    }

    const int blk_k = (TC_BLOCK_K == 8) ? 8 : ((TC_BLOCK_K == 16) ? 16 : 32);
    const int S = (TC_BLOCK_K == 8) ? 3 : ((TC_BLOCK_K == 16) ? 4 : 5);
    const int diag_slot_count = TC_BLOCK_M + blk_k - 1;
    const int diag_bias = TC_BLOCK_M - 1;

    int compact_pos = 0;
    for (int block_id = 0; block_id < TCBlock_num; ++block_id) {
        const int block_start = val_coo_off[block_id * 2];
        const int block_real_end = val_coo_off[block_id * 2 + 1];
        meta.val_dia_len_new_offset[block_id] = compact_pos;

        bool active_diag[TC_BLOCK_M + 32] = {false};
        std::vector<float> diag_values(diag_slot_count * blk_k, 0.0f);

        for (int pos = block_start; pos < block_real_end; ++pos) {
            int swizzled = val_coo_idx[pos];
            int logical = dia_deswizzle_idx(swizzled, S);
            int row = logical / blk_k;
            int col = logical % blk_k;
            int diag = col - row;
            int diag_slot = diag + diag_bias;
            if (diag_slot >= 0 && diag_slot < diag_slot_count) {
                active_diag[diag_slot] = true;
                diag_values[diag_slot * blk_k + col] = val_coo_val[pos];
            }
        }

        for (int diag_slot = 0; diag_slot < diag_slot_count; ++diag_slot) {
            if (!active_diag[diag_slot]) {
                continue;
            }
            const int diag = diag_slot - diag_bias;
            for (int col = 0; col < blk_k; ++col) {
                meta.val_dia_idx_new_1.push_back(diag);
                meta.val_dia_val_new_1.push_back(diag_values[diag_slot * blk_k + col]);
                ++compact_pos;
            }
        }
    }
    meta.val_dia_len_new_offset[TCBlock_num] = compact_pos;
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

    int* dval_sidx_new = nullptr;
    int* dval_len_offset = nullptr;
    int* dval_dia_len_new_offset = nullptr;
    int* dval_dia_idx_new_1 = nullptr;
    float* dval_dia_val_new_1 = nullptr;
    float* dB_val = nullptr;
    float* dC = nullptr;

    CHECK_CUDA2(cudaMalloc(&dval_sidx_new, sizeof(int) * meta.val_sidx_new_t.size()));
    CHECK_CUDA2(cudaMalloc(&dval_len_offset, sizeof(int) * meta.val_len_offset.size()));
    CHECK_CUDA2(cudaMalloc(&dval_dia_len_new_offset, sizeof(int) * meta.val_dia_len_new_offset.size()));
    CHECK_CUDA2(cudaMalloc(&dval_dia_idx_new_1, sizeof(int) * meta.val_dia_idx_new_1.size()));
    CHECK_CUDA2(cudaMalloc(&dval_dia_val_new_1, sizeof(float) * meta.val_dia_val_new_1.size()));
    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(float) * K * N));
    CHECK_CUDA2(cudaMalloc(&dC, sizeof(float) * row_window_num * ROW_WINDOW_M * N));

    CHECK_CUDA2(cudaMemcpy(dval_sidx_new, meta.val_sidx_new_t.data(), sizeof(int) * meta.val_sidx_new_t.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_len_offset, meta.val_len_offset.data(), sizeof(int) * meta.val_len_offset.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_dia_len_new_offset, meta.val_dia_len_new_offset.data(), sizeof(int) * meta.val_dia_len_new_offset.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_dia_idx_new_1, meta.val_dia_idx_new_1.data(), sizeof(int) * meta.val_dia_idx_new_1.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_dia_val_new_1, meta.val_dia_val_new_1.data(), sizeof(float) * meta.val_dia_val_new_1.size(), cudaMemcpyHostToDevice));
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
    const int tcblock_offsets_len = TCBlock_num + 1;

    generated_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>
        <<<grid, block>>>(
            dB_val,
            dC,
            dval_dia_idx_new_1,
            dval_dia_len_new_offset,
            dval_dia_val_new_1,
            dval_len_offset,
            dval_sidx_new,
            K,
            M,
            row_window_offsets_len,
            N,
            TCBlock_num,
            tcblock_offsets_len,
            static_cast<int>(meta.val_dia_val_new_1.size()));
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

    int* dval_sidx_new = nullptr;
    int* dval_len_offset = nullptr;
    int* dval_dia_len_new_offset = nullptr;
    int* dval_dia_idx_new_1 = nullptr;
    float* dval_dia_val_new_1 = nullptr;
    float* dB_val = nullptr;
    float* dC = nullptr;

    CHECK_CUDA2(cudaMalloc(&dval_sidx_new, sizeof(int) * meta.val_sidx_new_t.size()));
    CHECK_CUDA2(cudaMalloc(&dval_len_offset, sizeof(int) * meta.val_len_offset.size()));
    CHECK_CUDA2(cudaMalloc(&dval_dia_len_new_offset, sizeof(int) * meta.val_dia_len_new_offset.size()));
    CHECK_CUDA2(cudaMalloc(&dval_dia_idx_new_1, sizeof(int) * meta.val_dia_idx_new_1.size()));
    CHECK_CUDA2(cudaMalloc(&dval_dia_val_new_1, sizeof(float) * meta.val_dia_val_new_1.size()));
    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(float) * K * N));
    CHECK_CUDA2(cudaMalloc(&dC, sizeof(float) * row_window_num * ROW_WINDOW_M * N));

    CHECK_CUDA2(cudaMemcpy(dval_sidx_new, meta.val_sidx_new_t.data(), sizeof(int) * meta.val_sidx_new_t.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_len_offset, meta.val_len_offset.data(), sizeof(int) * meta.val_len_offset.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_dia_len_new_offset, meta.val_dia_len_new_offset.data(), sizeof(int) * meta.val_dia_len_new_offset.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_dia_idx_new_1, meta.val_dia_idx_new_1.data(), sizeof(int) * meta.val_dia_idx_new_1.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_dia_val_new_1, meta.val_dia_val_new_1.data(), sizeof(float) * meta.val_dia_val_new_1.size(), cudaMemcpyHostToDevice));
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
    const int tcblock_offsets_len = TCBlock_num + 1;

    auto launch = [&]() {
        generated_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>
            <<<grid, block>>>(
                dB_val,
                dC,
                dval_dia_idx_new_1,
                dval_dia_len_new_offset,
                dval_dia_val_new_1,
                dval_len_offset,
                dval_sidx_new,
                K,
                M,
                row_window_offsets_len,
                N,
                TCBlock_num,
                tcblock_offsets_len,
                static_cast<int>(meta.val_dia_val_new_1.size()));
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
    CHECK_CUDA2(cudaFree(dval_dia_len_new_offset));
    CHECK_CUDA2(cudaFree(dval_dia_idx_new_1));
    CHECK_CUDA2(cudaFree(dval_dia_val_new_1));
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
        M, N, K, TCBlock_num, nnz,
        val_sidx, val_soff, B_val,
        val_coo_idx, val_coo_val, val_coo_off,
        C, warmup_time, execute_time);
}

template void dtc_spmm_test_val_sidx_bind_reorder<64>(int M, int N, int K, int TCBlock_num, int nnz, int* val_sidx, int* val_soff, float* B_val, int* val_coo_idx, float* val_coo_val, int* val_coo_off, float* C, bool is_ncu_test, int warmup_time, int execute_time);
template void dtc_spmm_test_val_sidx_bind_reorder<16>(int M, int N, int K, int TCBlock_num, int nnz, int* val_sidx, int* val_soff, float* B_val, int* val_coo_idx, float* val_coo_val, int* val_coo_off, float* C, bool is_ncu_test, int warmup_time, int execute_time);
template void dtc_spmm_test_val_sidx_bind_reorder<32>(int M, int N, int K, int TCBlock_num, int nnz, int* val_sidx, int* val_soff, float* B_val, int* val_coo_idx, float* val_coo_val, int* val_coo_off, float* C, bool is_ncu_test, int warmup_time, int execute_time);
