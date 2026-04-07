#pragma once

#include "kernel.inc"

using namespace cute;

namespace sr_bcrs_lb {

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TensorREGC>
__device__ __forceinline__ void store_regc_to_row_window(float* dC_val,
                                                          int N,
                                                          int M,
                                                          int row_window_id,
                                                          int lid,
                                                          TensorREGC REGC,
                                                          bool use_atomic) {
    constexpr int tile_m = static_cast<int>(get<0>(BLK_MNK{}));
    constexpr int tile_n = static_cast<int>(get<1>(BLK_MNK{}));

    int col_tile_base = static_cast<int>(blockIdx.y) * tile_n;
    int row_window_base = row_window_id * tile_m;

    for (int i_tileM = 0; i_tileM < get<0>(BLK_MMA_MNK{}); ++i_tileM) {
        for (int i_tileN = 0; i_tileN < get<1>(BLK_MMA_MNK{}); ++i_tileN) {
            int row = lid / 4 + i_tileM * get<0>(MMA_MNK{});
            int col = i_tileN * 8 + (lid % 4) * 2;

            int g_row0 = row_window_base + row;
            int g_row1 = g_row0 + 8;
            int g_col0 = col_tile_base + col;
            int g_col1 = g_col0 + 1;

            if (g_row0 < M) {
                float* p00 = dC_val + g_row0 * N + g_col0;
                float* p01 = dC_val + g_row0 * N + g_col1;
                if (use_atomic) {
                    atomicAdd(p00, REGC(0, make_coord(i_tileM, i_tileN)));
                    atomicAdd(p01, REGC(1, make_coord(i_tileM, i_tileN)));
                } else {
                    *p00 = REGC(0, make_coord(i_tileM, i_tileN));
                    *p01 = REGC(1, make_coord(i_tileM, i_tileN));
                }
            }

            if (g_row1 < M) {
                float* p10 = dC_val + g_row1 * N + g_col0;
                float* p11 = dC_val + g_row1 * N + g_col1;
                if (use_atomic) {
                    atomicAdd(p10, REGC(2, make_coord(i_tileM, i_tileN)));
                    atomicAdd(p11, REGC(3, make_coord(i_tileM, i_tileN)));
                } else {
                    *p10 = REGC(2, make_coord(i_tileM, i_tileN));
                    *p11 = REGC(3, make_coord(i_tileM, i_tileN));
                }
            }
        }
    }
}

} // namespace sr_bcrs_lb

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK>
__global__ void sr_bcrs_spmm_kernel_tf32_multi_bind(float* dB_val,
                                                     float* dC_val,
                                                     float* dval_block_val,
                                                     int* dval_sidx,
                                                     int* dval_soff,
                                                     int* drow_window_binding,
                                                     int* drow_window_split,
                                                     int K,
                                                     int M,
                                                     int Mo,
                                                     int N,
                                                     int nnz_block) {
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int wid = tid / 32;

    using B_valOp_t = B_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    B_valOp_t B_valOp_v{tid, lid, wid, K, N, dB_val};
    using C_valOp_t = C_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    C_valOp_t C_valOp_v{tid, lid, wid, N, M, dC_val};
    using val_block_valOp_t = val_block_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_block_valOp_t val_block_valOp_v{tid, lid, wid, nnz_block, dval_block_val};
    using val_sidxOp_t = val_sidxOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_sidxOp_t val_sidxOp_v{tid, lid, wid, nnz_block, dval_sidx};
    using val_soffOp_t = val_soffOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_soffOp_t val_soffOp_v{tid, lid, wid, Mo, dval_soff};
    using BlkYLoopZeroOp_t = BlkYLoopZeroOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlkYLoopZeroOp_layout_o0 = typename BlkYLoopZeroOp_t::Layout_o0;
    int BlkYLoopZeroOp_tensor_o0[cosize_v<BlkYLoopZeroOp_layout_o0>];
    BlkYLoopZeroOp_t BlkYLoopZeroOp_v{tid, lid, wid, BlkYLoopZeroOp_tensor_o0};
    using BlockDimYOp_t = BlockDimYOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlockDimYOp_layout_o0 = typename BlockDimYOp_t::Layout_o0;
    int BlockDimYOp_tensor_o0[cosize_v<BlockDimYOp_layout_o0>];
    BlockDimYOp_t BlockDimYOp_v{tid, lid, wid, BlockDimYOp_tensor_o0};
    using BlkXLoopZeroOp_t = BlkXLoopZeroOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlkXLoopZeroOp_layout_o0 = typename BlkXLoopZeroOp_t::Layout_o0;
    int BlkXLoopZeroOp_tensor_o0[cosize_v<BlkXLoopZeroOp_layout_o0>];
    BlkXLoopZeroOp_t BlkXLoopZeroOp_v{tid, lid, wid, BlkXLoopZeroOp_tensor_o0};
    using BlockDimXOp_t = BlockDimXOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlockDimXOp_layout_o0 = typename BlockDimXOp_t::Layout_o0;
    int BlockDimXOp_tensor_o0[cosize_v<BlockDimXOp_layout_o0>];
    BlockDimXOp_t BlockDimXOp_v{tid, lid, wid, BlockDimXOp_tensor_o0};
    using G2sSparseOffsetLoadOp_t = G2sSparseOffsetLoadOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sSparseOffsetLoadOp_layout_o0 = typename G2sSparseOffsetLoadOp_t::Layout_o0;
    using G2sSparseOffsetLoadOp_layout_o1 = typename G2sSparseOffsetLoadOp_t::Layout_o1;
    int G2sSparseOffsetLoadOp_tensor_o0[cosize_v<G2sSparseOffsetLoadOp_layout_o0>];
    int G2sSparseOffsetLoadOp_tensor_o1[cosize_v<G2sSparseOffsetLoadOp_layout_o1>];
    G2sSparseOffsetLoadOp_t G2sSparseOffsetLoadOp_v{tid, lid, wid,
                                                     G2sSparseOffsetLoadOp_tensor_o0,
                                                     G2sSparseOffsetLoadOp_tensor_o1,
                                                     Mo};
    using ZerosOp_t = ZerosOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using ZerosOp_layout_o0 = typename ZerosOp_t::Layout_o0;
    float ZerosOp_tensor_o0[cosize_v<ZerosOp_layout_o0>];
    ZerosOp_t ZerosOp_v{tid, lid, wid, ZerosOp_tensor_o0};
    using G2sSparseIndexLoadOp_t =
      G2sSparseIndexLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sSparseIndexLoadOp_layout_o0 = typename G2sSparseIndexLoadOp_t::Layout_o0;
    __shared__ int G2sSparseIndexLoadOp_tensor_o0[cosize_v<G2sSparseIndexLoadOp_layout_o0>];
    G2sSparseIndexLoadOp_t G2sSparseIndexLoadOp_v{tid, lid, wid,
                                                   G2sSparseIndexLoadOp_tensor_o0,
                                                   nnz_block};
    using G2sSparseValBlockValLoadOp_t =
      G2sSparseValBlockValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sSparseValBlockValLoadOp_layout_o0 = typename G2sSparseValBlockValLoadOp_t::Layout_o0;
    __shared__ float G2sSparseValBlockValLoadOp_tensor_o0[cosize_v<G2sSparseValBlockValLoadOp_layout_o0>];
    G2sSparseValBlockValLoadOp_t G2sSparseValBlockValLoadOp_v{tid,
                                                               lid,
                                                               wid,
                                                               G2sSparseValBlockValLoadOp_tensor_o0,
                                                               nnz_block};
    using G2sMatrixBLoadOp_t = G2sMatrixBLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sMatrixBLoadOp_layout_o0 = typename G2sMatrixBLoadOp_t::Layout_o0;
    __shared__ float G2sMatrixBLoadOp_tensor_o0[cosize_v<G2sMatrixBLoadOp_layout_o0>];
    G2sMatrixBLoadOp_t G2sMatrixBLoadOp_v{tid, lid, wid, G2sMatrixBLoadOp_tensor_o0, K, N};
    using S2rAValLoadOp_t = S2rAValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using S2rAValLoadOp_layout_o0 = typename S2rAValLoadOp_t::Layout_o0;
    float S2rAValLoadOp_tensor_o0[cosize_v<S2rAValLoadOp_layout_o0>];
    S2rAValLoadOp_t S2rAValLoadOp_v{tid, lid, wid, S2rAValLoadOp_tensor_o0};
    using S2rBValLoadOp_t = S2rBValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using S2rBValLoadOp_layout_o0 = typename S2rBValLoadOp_t::Layout_o0;
    float S2rBValLoadOp_tensor_o0[cosize_v<S2rBValLoadOp_layout_o0>];
    S2rBValLoadOp_t S2rBValLoadOp_v{tid, lid, wid, S2rBValLoadOp_tensor_o0};
    using CalculateOp_t = CalculateOp<Int<2>,
                                      BLK_MNK,
                                      MMA_MNK,
                                      BLK_MMA_MNK,
                                      WARP_MNK,
                                      std::decay_t<decltype(ZerosOp_v.template output<0>())>>;
    CalculateOp_t CalculateOp_v{tid, lid, wid, ZerosOp_v.template output<0>()};
    using SRBCRSMainLoopOp_t = SRBCRSMainLoopOp<BLK_MNK,
                                                MMA_MNK,
                                                BLK_MMA_MNK,
                                                WARP_MNK,
                                                B_valOp_t,
                                                C_valOp_t,
                                                val_block_valOp_t,
                                                val_sidxOp_t,
                                                val_soffOp_t,
                                                BlkYLoopZeroOp_t,
                                                BlockDimYOp_t,
                                                BlkXLoopZeroOp_t,
                                                BlockDimXOp_t,
                                                G2sSparseOffsetLoadOp_t,
                                                ZerosOp_t,
                                                G2sSparseIndexLoadOp_t,
                                                G2sSparseValBlockValLoadOp_t,
                                                G2sMatrixBLoadOp_t,
                                                S2rAValLoadOp_t,
                                                S2rBValLoadOp_t,
                                                CalculateOp_t,
                                                std::decay_t<decltype(CalculateOp_v.template output<0>())>>;
    SRBCRSMainLoopOp_t SRBCRSMainLoopOp_v{tid,
                                          lid,
                                          wid,
                                          B_valOp_v,
                                          C_valOp_v,
                                          val_block_valOp_v,
                                          val_sidxOp_v,
                                          val_soffOp_v,
                                          BlkYLoopZeroOp_v,
                                          BlockDimYOp_v,
                                          BlkXLoopZeroOp_v,
                                          BlockDimXOp_v,
                                          G2sSparseOffsetLoadOp_v,
                                          ZerosOp_v,
                                          G2sSparseIndexLoadOp_v,
                                          G2sSparseValBlockValLoadOp_v,
                                          G2sMatrixBLoadOp_v,
                                          S2rAValLoadOp_v,
                                          S2rBValLoadOp_v,
                                          CalculateOp_v,
                                          CalculateOp_v.template output<0>()};

    int bind_i = drow_window_binding[blockIdx.x];
    int l = dval_soff[blockIdx.x];
    int r = dval_soff[blockIdx.x + 1];

    if (l < r) {
        ZerosOp_v.template f<>();
        SRBCRSMainLoopOp_v.template f<>(l, r);
        bool use_atomic = (drow_window_split[bind_i] != 0);
        sr_bcrs_lb::store_regc_to_row_window<BLK_MNK, MMA_MNK, BLK_MMA_MNK>(
          dC_val,
          N,
          M,
          bind_i,
          lid,
          SRBCRSMainLoopOp_v.template output<0>(),
          use_atomic);
    }
}

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK>
__global__ void sr_bcrs_spmm_kernel_tf32_strict_lb(float* dB_val,
                                                    float* dC_val,
                                                    float* dval_block_val,
                                                    int* dval_sidx,
                                                    int* dval_soff,
                                                    int* drow_block_offset,
                                                    int* drow_block_to_row_window_id,
                                                    int* dslb_tb_offset,
                                                    int row_block_num,
                                                    int K,
                                                    int M,
                                                    int Mo,
                                                    int N,
                                                    int nnz_block) {
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int wid = tid / 32;

    using B_valOp_t = B_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    B_valOp_t B_valOp_v{tid, lid, wid, K, N, dB_val};
    using C_valOp_t = C_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    C_valOp_t C_valOp_v{tid, lid, wid, N, M, dC_val};
    using val_block_valOp_t = val_block_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_block_valOp_t val_block_valOp_v{tid, lid, wid, nnz_block, dval_block_val};
    using val_sidxOp_t = val_sidxOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_sidxOp_t val_sidxOp_v{tid, lid, wid, nnz_block, dval_sidx};
    using val_soffOp_t = val_soffOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_soffOp_t val_soffOp_v{tid, lid, wid, Mo, dval_soff};
    using BlkYLoopZeroOp_t = BlkYLoopZeroOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlkYLoopZeroOp_layout_o0 = typename BlkYLoopZeroOp_t::Layout_o0;
    int BlkYLoopZeroOp_tensor_o0[cosize_v<BlkYLoopZeroOp_layout_o0>];
    BlkYLoopZeroOp_t BlkYLoopZeroOp_v{tid, lid, wid, BlkYLoopZeroOp_tensor_o0};
    using BlockDimYOp_t = BlockDimYOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlockDimYOp_layout_o0 = typename BlockDimYOp_t::Layout_o0;
    int BlockDimYOp_tensor_o0[cosize_v<BlockDimYOp_layout_o0>];
    BlockDimYOp_t BlockDimYOp_v{tid, lid, wid, BlockDimYOp_tensor_o0};
    using BlkXLoopZeroOp_t = BlkXLoopZeroOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlkXLoopZeroOp_layout_o0 = typename BlkXLoopZeroOp_t::Layout_o0;
    int BlkXLoopZeroOp_tensor_o0[cosize_v<BlkXLoopZeroOp_layout_o0>];
    BlkXLoopZeroOp_t BlkXLoopZeroOp_v{tid, lid, wid, BlkXLoopZeroOp_tensor_o0};
    using BlockDimXOp_t = BlockDimXOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlockDimXOp_layout_o0 = typename BlockDimXOp_t::Layout_o0;
    int BlockDimXOp_tensor_o0[cosize_v<BlockDimXOp_layout_o0>];
    BlockDimXOp_t BlockDimXOp_v{tid, lid, wid, BlockDimXOp_tensor_o0};
    using G2sSparseOffsetLoadOp_t = G2sSparseOffsetLoadOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sSparseOffsetLoadOp_layout_o0 = typename G2sSparseOffsetLoadOp_t::Layout_o0;
    using G2sSparseOffsetLoadOp_layout_o1 = typename G2sSparseOffsetLoadOp_t::Layout_o1;
    int G2sSparseOffsetLoadOp_tensor_o0[cosize_v<G2sSparseOffsetLoadOp_layout_o0>];
    int G2sSparseOffsetLoadOp_tensor_o1[cosize_v<G2sSparseOffsetLoadOp_layout_o1>];
    G2sSparseOffsetLoadOp_t G2sSparseOffsetLoadOp_v{tid, lid, wid,
                                                     G2sSparseOffsetLoadOp_tensor_o0,
                                                     G2sSparseOffsetLoadOp_tensor_o1,
                                                     Mo};
    using ZerosOp_t = ZerosOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using ZerosOp_layout_o0 = typename ZerosOp_t::Layout_o0;
    float ZerosOp_tensor_o0[cosize_v<ZerosOp_layout_o0>];
    ZerosOp_t ZerosOp_v{tid, lid, wid, ZerosOp_tensor_o0};
    using G2sSparseIndexLoadOp_t =
      G2sSparseIndexLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sSparseIndexLoadOp_layout_o0 = typename G2sSparseIndexLoadOp_t::Layout_o0;
    __shared__ int G2sSparseIndexLoadOp_tensor_o0[cosize_v<G2sSparseIndexLoadOp_layout_o0>];
    G2sSparseIndexLoadOp_t G2sSparseIndexLoadOp_v{tid, lid, wid,
                                                   G2sSparseIndexLoadOp_tensor_o0,
                                                   nnz_block};
    using G2sSparseValBlockValLoadOp_t =
      G2sSparseValBlockValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sSparseValBlockValLoadOp_layout_o0 = typename G2sSparseValBlockValLoadOp_t::Layout_o0;
    __shared__ float G2sSparseValBlockValLoadOp_tensor_o0[cosize_v<G2sSparseValBlockValLoadOp_layout_o0>];
    G2sSparseValBlockValLoadOp_t G2sSparseValBlockValLoadOp_v{tid,
                                                               lid,
                                                               wid,
                                                               G2sSparseValBlockValLoadOp_tensor_o0,
                                                               nnz_block};
    using G2sMatrixBLoadOp_t = G2sMatrixBLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sMatrixBLoadOp_layout_o0 = typename G2sMatrixBLoadOp_t::Layout_o0;
    __shared__ float G2sMatrixBLoadOp_tensor_o0[cosize_v<G2sMatrixBLoadOp_layout_o0>];
    G2sMatrixBLoadOp_t G2sMatrixBLoadOp_v{tid, lid, wid, G2sMatrixBLoadOp_tensor_o0, K, N};
    using S2rAValLoadOp_t = S2rAValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using S2rAValLoadOp_layout_o0 = typename S2rAValLoadOp_t::Layout_o0;
    float S2rAValLoadOp_tensor_o0[cosize_v<S2rAValLoadOp_layout_o0>];
    S2rAValLoadOp_t S2rAValLoadOp_v{tid, lid, wid, S2rAValLoadOp_tensor_o0};
    using S2rBValLoadOp_t = S2rBValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using S2rBValLoadOp_layout_o0 = typename S2rBValLoadOp_t::Layout_o0;
    float S2rBValLoadOp_tensor_o0[cosize_v<S2rBValLoadOp_layout_o0>];
    S2rBValLoadOp_t S2rBValLoadOp_v{tid, lid, wid, S2rBValLoadOp_tensor_o0};
    using CalculateOp_t = CalculateOp<Int<2>,
                                      BLK_MNK,
                                      MMA_MNK,
                                      BLK_MMA_MNK,
                                      WARP_MNK,
                                      std::decay_t<decltype(ZerosOp_v.template output<0>())>>;
    CalculateOp_t CalculateOp_v{tid, lid, wid, ZerosOp_v.template output<0>()};
    using SRBCRSMainLoopOp_t = SRBCRSMainLoopOp<BLK_MNK,
                                                MMA_MNK,
                                                BLK_MMA_MNK,
                                                WARP_MNK,
                                                B_valOp_t,
                                                C_valOp_t,
                                                val_block_valOp_t,
                                                val_sidxOp_t,
                                                val_soffOp_t,
                                                BlkYLoopZeroOp_t,
                                                BlockDimYOp_t,
                                                BlkXLoopZeroOp_t,
                                                BlockDimXOp_t,
                                                G2sSparseOffsetLoadOp_t,
                                                ZerosOp_t,
                                                G2sSparseIndexLoadOp_t,
                                                G2sSparseValBlockValLoadOp_t,
                                                G2sMatrixBLoadOp_t,
                                                S2rAValLoadOp_t,
                                                S2rBValLoadOp_t,
                                                CalculateOp_t,
                                                std::decay_t<decltype(CalculateOp_v.template output<0>())>>;
    SRBCRSMainLoopOp_t SRBCRSMainLoopOp_v{tid,
                                          lid,
                                          wid,
                                          B_valOp_v,
                                          C_valOp_v,
                                          val_block_valOp_v,
                                          val_sidxOp_v,
                                          val_soffOp_v,
                                          BlkYLoopZeroOp_v,
                                          BlockDimYOp_v,
                                          BlkXLoopZeroOp_v,
                                          BlockDimXOp_v,
                                          G2sSparseOffsetLoadOp_v,
                                          ZerosOp_v,
                                          G2sSparseIndexLoadOp_v,
                                          G2sSparseValBlockValLoadOp_v,
                                          G2sMatrixBLoadOp_v,
                                          S2rAValLoadOp_v,
                                          S2rBValLoadOp_v,
                                          CalculateOp_v,
                                          CalculateOp_v.template output<0>()};

    int row_block_l = dslb_tb_offset[blockIdx.x];
    int row_block_r = dslb_tb_offset[blockIdx.x + 1];

    for (int row_block_id = row_block_l; row_block_id < row_block_r; ++row_block_id) {
        if (row_block_id < 0 || row_block_id >= row_block_num) {
            continue;
        }

        int l = drow_block_offset[row_block_id];
        int r = drow_block_offset[row_block_id + 1];
        if (l >= r) {
            continue;
        }

        int row_window_id = drow_block_to_row_window_id[row_block_id];
        bool use_atomic = (row_block_id == row_block_l) || (row_block_id == row_block_r - 1);

        ZerosOp_v.template f<>();
        SRBCRSMainLoopOp_v.template f<>(l, r);
        sr_bcrs_lb::store_regc_to_row_window<BLK_MNK, MMA_MNK, BLK_MMA_MNK>(
          dC_val,
          N,
          M,
          row_window_id,
          lid,
          SRBCRSMainLoopOp_v.template output<0>(),
          use_atomic);
    }
}
