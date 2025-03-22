#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <cuda_pipeline.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

using namespace std;
using namespace cute;

#define ROW_WINDOW_M 16
#define TC_BLOCK_K 16
#define TC_BLOCK_M 16

#define VARLEN 0

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))
#define OFFSET_COL(row, col, lda) ((col) * (lda) + (row))

#define CHECK_CUDA2(func)                                              \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess) {                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
        }                                                              \
    }

__device__ __forceinline__ void ldmatrix_m8n8k8_x4(uint32_t* fragment, void* ptr) {
    uint32_t values_tile_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 \t"
        " { %0, %1, %2, %3 }, \t"
        " [%4];"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
        : "r"(values_tile_int_ptr)
    );
}

__device__ __forceinline__ void mma_m16n8k16_fp32(float* acc, uint32_t* A, half* half_B) {
    uint32_t* B = reinterpret_cast<uint32_t*>(half_B);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1])
    );
}

// shift 1
template<class nbuf, class tile_sidx_size>
class G2sValSidxLoadOp {
    public:
    // Inputs
    // !! shape is unknown until runtime, how to handle this?
    // Solution: since the shape is runtile defined, we have to pass them into the contructor of op
    // during the codegen, we record all unknown runtime dimensions in a sequential list in order
    // and pass them into the constructor of op
    //     they are named mode0, mode1, ...
    // We only define the type of shapes here because we do not know the actual shape when runtime lengths are used,
    // and the actual value of shape would be determined on class construction.
    using Shape_i0 = decltype(make_shape(tile_sidx_size{}, VARLEN));
    using Tensor_i0 = decltype(make_tensor(make_gmem_ptr((int*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 dval_sidx;

    // Outputs
    using Shape_o0 = decltype(make_shape(tile_sidx_size{}, nbuf{}));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((int*)nullptr), Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 tile_sidx_block;

    // Hardware parameters
    int tid, lid, wid;
    decltype(make_coord(VARLEN)) thr_coord;
    decltype(make_shape(Int<4>{})) thr_tiler;

    CUTE_DEVICE G2sValSidxLoadOp(int tid, int lid, int wid, int* tile_sidx_block, int mode0)
      : tid(tid), lid(lid), wid(wid),
      thr_coord(make_coord(lid)),
        // dval_sidx(make_tensor(make_gmem_ptr(dval_sidx), Shape_i0{})),
        shape_i0(make_shape(tile_sidx_size{}, mode0)),
        tile_sidx_block(make_tensor(make_smem_ptr(tile_sidx_block), shape_o0)) {}

    template<size_t buf_idx, size_t output_idx=0>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_sidx_block(_, buf_idx);
        }
    }

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 dval_sidx, int i) {
        if (lid < 4) {
            auto src = local_tile(dval_sidx(_, i), thr_tiler, thr_coord);
            auto dst = local_tile(tile_sidx_block(_, buf_idx), thr_tiler, thr_coord);
            copy(Copy_Atom<UniversalCopy<uint128_t>, int>{}, src, dst);
        }
    }
};

template<class nbuf>
class G2rCooAtomicFormatLoadOffOp {
    public:

    // Inputs
    using Shape_i0 = decltype(make_shape(_2{}, VARLEN));
    using Tensor_i0 = decltype(make_tensor(make_gmem_ptr((int*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_io dval_coo_off

    // Outputs
    using Shape_o0 = decltype(make_shape(nbuf{}));
    using Shape_o1 = decltype(make_shape(nbuf{}));
    using Tensor_o0 = decltype(make_tensor<int>(Shape_o0{}));
    using Tensor_o1 = decltype(make_tensor<int>(Shape_o1{}));
    Tensor_o0 l;
    Tensor_o1 r;

    int tid, lid, wid;

    CUTE_HOST_DEVICE G2rCooAtomicFormatLoadOffOp(int tid, int lid, int wid, int mode0)
    : tid(tid), lid(lid), wid(wid),
    shape_i0(make_shape(_2{}, mode0)) {}

    template<size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return l(buf_idx);
        } else if constexpr (output_idx == 1) {
            return r(buf_idx);
        }
    }

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 dval_coo_off, int i) {
        l(buf_idx) = dval_coo_off(0, i);
        r(buf_idx) = dval_coo_off(1, i);
    }
};

// shift 2
template<class nbuf, class tile_sidx_size>
class S2rValSidxLoadOp {
    public:

    // Inputs
    using Shape_i0 = decltype(make_shape(tile_sidx_size{}));
    using Tensor_i0 = decltype(make_tensor(make_smem_ptr((int*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 tile_sidx_block;

    // Outputs
    using Shape_o0 = decltype(make_shape(tile_sidx_size{}, nbuf{}));
    using Tensor_o0 = decltype(make_tensor<int>(Shape_o0{}));
    Tensor_o0 val_sidx_reg;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE S2rValSidxLoadOp(int tid, int lid, int wid)
      : tid(tid), lid(lid), wid(wid) {}

    template<size_t buf_idx, size_t output_idx=0>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_sidx_reg(_, buf_idx);
        }
    }

    template<size_t buf_idx, size_t wait_num>
    CUTE_DEVICE void f(Tensor_i0 tile_sidx_block) {
        __pipeline_wait_prior(wait_num);
        __syncthreads();
        auto src = tile_sidx_block;
        auto dst = val_sidx_reg(_, buf_idx);
        copy(src, dst);
    }
};

template <class nbuf, class tile_coo_val_size, class DT=half>
class G2sCooAtomicFormatLoadValOp {
    public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    using Tensor_i0 = decltype(make_tensor(make_gmem_ptr((DT*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 val_coo_val;
    // int ll;
    // int rr;

    // Outputs
    using Shape_o0 = decltype(make_shape(tile_coo_val_size{}, nbuf{}));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((DT*)nullptr), Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 tile_coo_val_block;

    // Hardware parameters
    int tid, lid, wid;
    decltype(make_coord(VARLEN)) thr_coord;
    decltype(make_shape(_8{})) thr_tiler;

    CUTE_DEVICE G2sCooAtomicFormatLoadValOp(int tid, int lid, int wid, DT* tile_coo_val_block, int mode0)
    : tid(tid), lid(lid), wid(wid),
      thr_coord(make_coord(lid)),
      shape_i0(make_shape(mode0)),
      tile_coo_val_block(make_tensor(make_smem_ptr(tile_coo_val_block), shape_o0)) {}

    template<size_t buf_idx, size_t output_idx=0>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_coo_val_block(_, buf_idx);
        }
    }

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 val_coo_val, int ll, int rr) {
        auto input = make_tensor(val_coo_val.data() + ll, make_shape(tile_coo_val_size{}));
        auto output = tile_coo_val_block(_, buf_idx);

        if (lid * 8 < rr - ll) {
            auto src = local_tile(input, thr_tiler, thr_coord);
            auto dst = local_tile(output, thr_tiler, thr_coord);
            copy(Copy_Atom<UniversalCopy<uint128_t>, half>{}, src, dst);
        }
    }
};

template <class nbuf, class tile_coo_val_size>
class G2rCooAtomicFormatLoadIdxOp {
    public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    using Tensor_i0 = decltype(make_tensor(make_gmem_ptr((int*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 val_coo_idx;
    // int ll;
    // int rr;

    // Outputs
    using Shape_o0 = decltype(make_shape(_8{}, nbuf{}));
    using Tensor_o0 = decltype(make_tensor<int>(Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 coo_idx_reg;
    using Shape_o1 = decltype(make_shape(nbuf{}));
    using Tensor_o1 = decltype(make_tensor<int>(Shape_o1{}));
    Shape_o1 shape_o1;
    Tensor_o1 coo_range_reg;

    // Hardware parameters
    int tid, lid, wid;
    decltype(make_coord(VARLEN)) thr_coord;
    decltype(make_shape(_8{})) thr_tiler;

    CUTE_DEVICE G2rCooAtomicFormatLoadIdxOp(int tid, int lid, int wid, int mode0)
    : tid(tid), lid(lid), wid(wid),
      thr_coord(make_coord(lid)),
      shape_i0(make_shape(mode0)) {}

    template<size_t buf_idx, size_t output_idx=0>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return coo_idx_reg(_, buf_idx);
        } else if constexpr (output_idx == 1) {
            return coo_range_reg(buf_idx);
        }
    }

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 val_coo_idx, int ll, int rr) {
        auto input = make_tensor(val_coo_idx.data() + ll, make_shape(tile_coo_val_size{}));
        coo_range_reg(buf_idx) = rr - ll;
        if (lid * 8 < rr - ll) {
            auto src = local_tile(input, thr_tiler, thr_coord);
            auto dst = coo_idx_reg(_, buf_idx);
            copy(src, dst);
        }
    }
};

// template <int maxshift_smem, int maxshift_reg, int r1, int r2, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
// __device__ void op_G2R_coo_atomic_format_load_idx(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
//     g2r_coo_atomic_format_load_idx_op.f<r2>(val_coo_idx,
//         g2r_coo_atomic_format_load_off_op.template output<r1, 0>(),
//         g2r_coo_atomic_format_load_off_op.template output<r1, 1>()
//     );
// }

// shift 3
template <int maxshift_smem, int maxshift_reg, int r1, int r2, int TILE_B, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void op_G2R_B_val_load(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    for (int n_iter = 0; n_iter < TILE_B / 8; n_iter++) {
        int row = lane_id % 4 * 2;
        int col = block_idx_y * TILE_B + n_iter * 8 + lane_id / 4;

        REGB[r2 * 4 * (TILE_B / 8) + n_iter * 4 + 0] = B_val(s2r_val_sidx_load_op.template output<r1, 0>()(row + 0), col);
        REGB[r2 * 4 * (TILE_B / 8) + n_iter * 4 + 1] = B_val(s2r_val_sidx_load_op.template output<r1, 0>()(row + 1), col);
        REGB[r2 * 4 * (TILE_B / 8) + n_iter * 4 + 2] = B_val(s2r_val_sidx_load_op.template output<r1, 0>()(row + 8), col);
        REGB[r2 * 4 * (TILE_B / 8) + n_iter * 4 + 3] = B_val(s2r_val_sidx_load_op.template output<r1, 0>()(row + 9), col);

    }
}

template <int maxshift_smem, int maxshift_reg, int r1, int wait_num, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void op_S2R_coo_atomic_val_restore(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    __pipeline_wait_prior(wait_num);
    __syncthreads();
    int nnz_num = g2r_coo_atomic_format_load_idx_op.template output<r1, 1>();
    int smem_ld_ptr = i % maxshift_smem;
    for (int i_o2s = 0; i_o2s < 4; i_o2s++) {
        *((int*)&tile_coo_restore_val_block[smem_ld_ptr * tile_coo_val_size + 32 * 2 * i_o2s + lane_id * 2]) = 0;
    }
    __syncthreads();

    for (int i_restore = 0; i_restore < 8; i_restore++) {
        int idx = g2r_coo_atomic_format_load_idx_op.template output<r1, 0>()(i_restore);
        if (lane_id * 8 + i_restore < nnz_num) {
            tile_coo_restore_val_block[smem_ld_ptr * tile_coo_val_size + idx] = tile_coo_val_block[smem_ld_ptr * tile_coo_val_size + lane_id * 8 + i_restore];
        }
    }
    __syncthreads();

    int row = lane_id % 16;
    int col = lane_id / 16;
    // TODO bank conflict
    ldmatrix_m8n8k8_x4((uint32_t*)&REGA[r1 * 8], (void*)&tile_coo_restore_val_block[smem_ld_ptr * tile_coo_val_size + (row) * 16 + (col * 8)]);
}

template <int maxshift_smem, int maxshift_reg, int r1, int r2, int TILE_B, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void op_mma(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    for (int n_iter = 0; n_iter < (TILE_B / 8); n_iter++) {
        mma_m16n8k16_fp32(&REGC[4 * n_iter], (uint32_t*)&REGA[r1 * 8], &REGB[r2 * 4 * (TILE_B / 8) + n_iter * 4]);
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_reg, int maxshift_smem, int TILE_B, int K_total, int i, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void pipeline(int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    if constexpr (i < K_total) {
        g2s_val_sidx_load_op.f<i % maxshift_smem>(val_sidx, l + i);

        g2r_coo_atomic_format_load_off_op.f<i % maxshift_reg>(val_coo_off, l + i);
        if constexpr (i < shift1) {
            __pipeline_commit();
        }
    }
    // (sstage > (K - 1 - (i - sstage)) ? K - 1 - (i - sstage) : sstage)
    if constexpr (0 <= i - shift1 && i - shift1 < K_total) {
        g2s_coo_atomic_format_load_val_op.template f<(i - shift1) % maxshift_reg>(val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % maxshift_reg, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % maxshift_reg, 1>()
        );
        __pipeline_commit();
        s2r_val_sidx_load_op.f<(i - shift1) % maxshift_reg, (K_total > shift1 ? shift1 : K_total)>(g2s_val_sidx_load_op.template output<(i - shift1) % maxshift_reg, 0>());
        g2r_coo_atomic_format_load_idx_op.f<(i - shift1) % maxshift_reg>(val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % maxshift_reg, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % maxshift_reg, 1>()
        );
        // op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (i - shift1) % maxshift_reg, (i - shift1) % maxshift_reg>(i - shift1, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    }
    if constexpr (0 <= i - shift2 - shift1 && i - shift2 - shift1 < K_total) {
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (i - shift1 - shift2) % maxshift_reg, (i - shift1 - shift2) % maxshift_reg, TILE_B>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (i - shift1 - shift2) % maxshift_reg, (shift2 > (K_total + shift1 + shift2 - 1 - i) ? (K_total + shift1 + shift2 - 1 - i) : shift2)>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    }
    if constexpr (0 <= i - shift1 - shift2 - shift3 && i - shift1 - shift2 - shift3 < K_total) {
        op_mma<maxshift_smem, maxshift_reg, (i - shift1 - shift2 - shift3) % maxshift_reg, (i - shift1 - shift2 - shift3) % maxshift_reg, TILE_B>(i - shift1 - shift2 - shift3, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    }
    if constexpr (i < K_total + shift1 + shift2 + shift3) {
        pipeline<shift1, shift2, shift3, maxshift_reg, maxshift_smem, TILE_B, K_total, i + 1>(M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    }

}

template <int shift1, int shift2, int shift3, int maxshift_reg, int maxshift_smem, int TILE_B, int K_total, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void pipeline_switch(int k, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    if constexpr (K_total == 0) {
        return;
    } else {
        if (k == K_total) {
            pipeline<shift1, shift2, shift3, maxshift_reg, maxshift_smem, TILE_B, K_total, 0>(M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        } else {
            pipeline_switch<shift1, shift2, shift3, maxshift_reg, maxshift_smem, TILE_B, K_total - 1>(k, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        }
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int i, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void fill_shift1(int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    if constexpr (i == shift1) {
        return;
    } else {
        g2s_val_sidx_load_op.f<i % maxshift_smem>(val_sidx, l + i);
        g2r_coo_atomic_format_load_off_op.f<i % maxshift_reg>(val_coo_off, l + i);
        __pipeline_commit();
        fill_shift1<shift1, shift2, shift3, maxshift_smem, maxshift_reg, i + 1>(M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int i, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void fill_shift2(int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    if constexpr (i == shift2) {
        return;
    } else {
        g2s_val_sidx_load_op.f<(i + shift1) % maxshift_smem>(val_sidx, l + i + shift1);
        g2s_coo_atomic_format_load_val_op.template f<i % maxshift_reg>(val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<i % maxshift_reg, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<i % maxshift_reg, 1>()
        );
        __pipeline_commit();
        s2r_val_sidx_load_op.f<i % maxshift_reg, shift1>(g2s_val_sidx_load_op.template output<i % maxshift_reg, 0>());

        g2r_coo_atomic_format_load_off_op.f<(i + shift1) % maxshift_reg>(val_coo_off, l + i + shift1);

        g2r_coo_atomic_format_load_idx_op.f<i % maxshift_reg>(val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<i % maxshift_reg, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<i % maxshift_reg, 1>()
        );
        // op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, i % maxshift_reg, i % maxshift_reg>(i, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);

        fill_shift2<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B,  i + 1>(M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int i, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void fill_shift3(int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    if constexpr (i == shift3) {
        return;
    } else {
        g2s_val_sidx_load_op.f<(i + shift1 + shift2) % maxshift_smem>(val_sidx, l + i + shift1 + shift2);
        g2s_coo_atomic_format_load_val_op.template f<(i + shift2) % maxshift_reg>(val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<(i + shift2) % maxshift_reg, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(i + shift2) % maxshift_reg, 1>()
        );
        __pipeline_commit();
        s2r_val_sidx_load_op.f<(i + shift2) % maxshift_reg, shift1>(g2s_val_sidx_load_op.template output<(i + shift2) % maxshift_reg, 0>());
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, i % maxshift_reg, i % maxshift_reg, TILE_B>(i, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);

        g2r_coo_atomic_format_load_off_op.f<(i + shift1 + shift2) % maxshift_reg>(val_coo_off, l + i + shift1 + shift2);
        g2r_coo_atomic_format_load_idx_op.f<(i + shift2) % maxshift_reg>(val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<(i + shift2) % maxshift_reg, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(i + shift2) % maxshift_reg, 1>()
        );
        // op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (i + shift2) % maxshift_reg, (i + shift2) % maxshift_reg>(i + shift2, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_smem, i % maxshift_reg, shift2>(i, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);

        fill_shift3<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, i + 1>(M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int I, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void loop_step(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    g2s_val_sidx_load_op.f<(I + shift1 + shift2 + shift3) % maxshift_smem>(val_sidx, l + i);
    g2s_coo_atomic_format_load_val_op.template f<(I + shift3 + shift2) % maxshift_reg>(val_coo_val,
        g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % maxshift_reg, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % maxshift_reg, 1>()
    );
    __pipeline_commit();
    s2r_val_sidx_load_op.f<(I + shift3 + shift2) % maxshift_reg, shift1>(g2s_val_sidx_load_op.template output<(I + shift3 + shift2) % maxshift_reg, 0>());
    op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift3) % maxshift_reg, (I + shift3) % maxshift_reg, TILE_B>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);

    g2r_coo_atomic_format_load_off_op.f<(I + shift3 + shift1 + shift2) % maxshift_reg>(val_coo_off, l + i);

    g2r_coo_atomic_format_load_idx_op.f<(I + shift3 + shift2) % maxshift_reg>(val_coo_idx,
        g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % maxshift_reg, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % maxshift_reg, 1>()
    );
    // op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (I + shift3 + shift2) % maxshift_reg, (I + shift3 + shift2) % maxshift_reg>(i - shift1, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (I + shift3) % maxshift_reg, shift2>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    op_mma<maxshift_smem, maxshift_reg, I % maxshift_reg, I % maxshift_reg, TILE_B>(i - shift1 - shift2 - shift3, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);

    if constexpr (I + 1 < maxshift_reg) {
        loop_step<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, I + 1>(i + 1, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain, int I, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void remainder(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    if constexpr (I == K_remain) {
        return;
    } else {
        g2s_val_sidx_load_op.f<(I + shift1 + shift2 + shift3) % maxshift_smem>(val_sidx, l + i);
        g2s_coo_atomic_format_load_val_op.template f<(I + shift3 + shift2) % maxshift_reg>(val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % maxshift_reg, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % maxshift_reg, 1>()
        );
        __pipeline_commit();
        s2r_val_sidx_load_op.f<(I + shift3 + shift2) % maxshift_reg, shift1>(g2s_val_sidx_load_op.template output<(I + shift3 + shift2) % maxshift_reg, 0>());
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift3) % maxshift_reg, (I + shift3) % maxshift_reg, TILE_B>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);

        g2r_coo_atomic_format_load_off_op.f<(I + shift3 + shift1 + shift2) % maxshift_reg>(val_coo_off, l + i);

        g2r_coo_atomic_format_load_idx_op.f<(I + shift3 + shift2) % maxshift_reg>(val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % maxshift_reg, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % maxshift_reg, 1>()
        );
        // op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (I + shift3 + shift2) % maxshift_reg, (I + shift3 + shift2) % maxshift_reg>(i - shift1, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (I + shift3) % maxshift_reg, shift2>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        op_mma<maxshift_smem, maxshift_reg, I % maxshift_reg, I % maxshift_reg, TILE_B>(i - shift1 - shift2 - shift3, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);

        remainder<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, I + 1>(i + 1, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void remainder_switch(int i, int k, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    if constexpr (K_remain == 0) {
        return;
    } else {
        if (k - i == K_remain) {
            remainder<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, 0>(i, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        } else {
            remainder_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain - 1>(i, k, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        }
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain, int I, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void empty_shift1(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {  // K is the remain
    if constexpr (I == shift1) {
        return;
    } else {
        g2s_coo_atomic_format_load_val_op.template f<(I + shift3 + shift2 + K_remain) % maxshift_reg>(val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2 + K_remain) % maxshift_reg, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2 + K_remain) % maxshift_reg, 1>()
        );
        __pipeline_commit();
        s2r_val_sidx_load_op.f<(I + shift3 + shift2 + K_remain) % maxshift_reg, shift1>(g2s_val_sidx_load_op.template output<(I + shift3 + shift2 + K_remain) % maxshift_reg, 0>());
        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift3 + K_remain) % maxshift_reg, (I + shift3 + K_remain) % maxshift_reg, TILE_B>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);


        g2r_coo_atomic_format_load_idx_op.f<(I + shift3 + shift2 + K_remain) % maxshift_reg>(val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2 + K_remain) % maxshift_reg, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2 + K_remain) % maxshift_reg, 1>()
        );
        // op_G2R_coo_atomic_format_load_idx<maxshift_smem, maxshift_reg, (I + shift3 + shift2 + K_remain) % maxshift_reg, (I + shift3 + shift2 + K_remain) % maxshift_reg>(i - shift1, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (I + shift3 + K_remain) % maxshift_reg, shift2>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        op_mma<maxshift_smem, maxshift_reg, (I + K_remain) % maxshift_reg, (I + K_remain) % maxshift_reg, TILE_B>(i - shift1 - shift2 - shift3, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);

        empty_shift1<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, I + 1>(i + 1, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void empty_shift1_switch(int i, int remain, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    if constexpr (K_remain < 0) {
        return;
    } else {
        if (K_remain == remain) {
            empty_shift1<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, 0>(i, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        } else {
            empty_shift1_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain - 1>(i, remain, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        }
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain, int I, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void empty_shift2(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    if constexpr (I == shift2) {
        return;
    } else {

        op_G2R_B_val_load<maxshift_smem, maxshift_reg, (I + shift1 + shift3 + K_remain) % maxshift_reg, (I + shift1 + shift3 + K_remain) % maxshift_reg, TILE_B>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);

        op_S2R_coo_atomic_val_restore<maxshift_smem, maxshift_reg, (I + shift1 + shift3 + K_remain) % maxshift_reg, shift2 - (I + 1)>(i - shift1 - shift2, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        op_mma<maxshift_smem, maxshift_reg, (I + shift1 + K_remain) % maxshift_reg, (I + shift1 + K_remain) % maxshift_reg, TILE_B>(i - shift1 - shift2 - shift3, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);

        empty_shift2<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, I + 1>(i + 1, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void empty_shift2_switch(int i, int remain, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    if constexpr (K_remain < 0) {
        return;

    } else {
        if (remain == K_remain) {
            empty_shift2<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, 0>(i, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        } else {
            empty_shift2_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain - 1>(i, remain, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        }
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain, int I, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void empty_shift3(int i, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    if constexpr (I == shift3) {
        return;
    } else {

        op_mma<maxshift_smem, maxshift_reg, (I + shift1 + shift2 + K_remain) % maxshift_reg, (I + shift1 + shift2 + K_remain) % maxshift_reg, TILE_B>(i - shift1 - shift2 - shift3, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        empty_shift3<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, I + 1>(i + 1, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    }
}

template <int shift1, int shift2, int shift3, int maxshift_smem, int maxshift_reg, int TILE_B, int K_remain, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off>
__device__ void empty_shift3_switch(int i, int remain, int M, int N, int K, int* dval_sidx, int* dval_soff, int* dval_coo_idx, half* dval_coo_val, int* dval_coo_off, float* dC_val, const int block_idx_x, const int block_idx_y, const int lane_id, int* tile_sidx_block, half* tile_coo_val_block, half* tile_coo_restore_val_block, int* coo_scope, int* val_sidx_reg, int* coo_idx_reg, int* coo_range_reg, half* REGB, half* REGA, float* REGC, int l, int r, const int tile_sidx_size, const int tile_coo_val_size, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, int nnz_block) {
    if constexpr (K_remain < 0) {
        return;
    } else {
        if (K_remain == remain) {
            empty_shift3<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain, 0>(i, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        } else {
            empty_shift3_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, K_remain - 1>(i, remain, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        }
    }
}

// =====================================================================================

#define DIVUP(a, b) (((a) - 1) / (b) + 1)
#define ALIGN(a,b) (DIVUP(a,b)*(b))

template <int shift1, int shift2, int shift3, int TILE_B>
__global__ void dtc_spmm_kernel_sstage_rstage(
    int M, int N, int K,
    int* dval_sidx,
    int* dval_soff,
    half* dB_val,
    int* dval_coo_idx,
    half* dval_coo_val,
    int* dval_coo_off,
    float* dC_val
) {
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int wid = tid / 32;

    const int Mo = DIVUP(M, ROW_WINDOW_M);
    const int nnz_block = dval_soff[Mo];
    const int nnz = ALIGN(dval_coo_off[nnz_block * 2 - 1], 8);

    const int block_idx_x = blockIdx.x;
    const int block_idx_y = blockIdx.y;
    const int lane_id = tid % 32;
    constexpr auto tile_sidx_size = Int<TC_BLOCK_K>{};
    constexpr auto tile_coo_val_size = Int<TC_BLOCK_M>{} * Int<TC_BLOCK_K>{};

    // constexpr const int maxshift_smem = (shift1 > shift2 ? shift1 : shift2);
    // constexpr const int maxshift_reg = (maxshift_smem > shift3 ? maxshift_smem : shift3) + 1;
    constexpr const int maxshift1_2 = (shift1 > shift2 ? shift1 : shift2);
    constexpr const int maxshift_reg = (maxshift1_2 > shift3 ? maxshift1_2 : shift3) + 1;
    constexpr const int maxshift_smem = maxshift1_2 + 1;

    __shared__ half tile_coo_val_block[tile_coo_val_size * (maxshift_smem)];
    __shared__ half tile_coo_restore_val_block[tile_coo_val_size * (maxshift_smem)];

    auto val_sidx = make_tensor(make_gmem_ptr(dval_sidx), make_shape(tile_sidx_size, nnz_block));
    auto val_soff = make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo));
    auto B_val = make_tensor(make_gmem_ptr(dB_val), make_shape(K, N));
    auto val_coo_idx = make_tensor(make_gmem_ptr(dval_coo_idx), make_shape(nnz));
    auto val_coo_val = make_tensor(make_gmem_ptr(dval_coo_val), make_shape(nnz));
    auto val_coo_off = make_tensor(make_gmem_ptr(dval_coo_off), make_shape(_2{}, nnz_block));

    __shared__ int tile_sidx_block[tile_sidx_size * (maxshift_smem)];
    G2sValSidxLoadOp<Int<maxshift_smem>, decltype(tile_sidx_size)> g2s_val_sidx_load_op{
        tid, lid, wid,
        tile_sidx_block,
        nnz_block,
    };
    G2rCooAtomicFormatLoadOffOp<Int<maxshift_smem>> g2r_coo_atomic_format_load_off_op{
        tid, lid, wid,
        nnz_block,
    };
    S2rValSidxLoadOp<Int<maxshift_reg>, Int<tile_sidx_size>> s2r_val_sidx_load_op{
        tid, lid, wid,
    };
    G2sCooAtomicFormatLoadValOp<Int<maxshift_smem>, Int<tile_coo_val_size>> g2s_coo_atomic_format_load_val_op{
        tid, lid, wid,
        tile_coo_val_block,
        nnz,
    };
    G2rCooAtomicFormatLoadIdxOp<Int<maxshift_reg>, Int<tile_coo_val_size>> g2r_coo_atomic_format_load_idx_op{
        tid, lid, wid,
        nnz,
    };

    // shift 1
    int coo_scope[2 * maxshift_reg];
    // shift 2
    int val_sidx_reg[tile_sidx_size * maxshift_reg];
    int coo_idx_reg[16 * 16 * maxshift_reg];
    int coo_range_reg[maxshift_reg];
    // shift 3
    // tile A [16 * 16], tile B [16 * 32]
    half REGB[4 * (TILE_B / 8) * maxshift_reg] = {0};
    half REGA[8 * maxshift_reg] = {0};
    float REGC[4 * (TILE_B / 8)] = {0};

    int l = dval_soff[block_idx_x];
    int r = dval_soff[block_idx_x + 1];

    int k = r - l;

    if (k <= shift1 + shift2 + shift3) {
        pipeline_switch<shift1, shift2, shift3, maxshift_reg, maxshift_smem, TILE_B, shift1 + shift2 + shift3>(k, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    } else {
        fill_shift1<shift1, shift2, shift3, maxshift_smem, maxshift_reg, 0>(M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        fill_shift2<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, 0>(M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        fill_shift3<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, 0>(M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        int i;
        for (i = shift1 + shift2 + shift3; i + maxshift_reg <= k; i += maxshift_reg) {
            loop_step<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, 0>(i, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        }
        int remain = k - i;
        remainder_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, maxshift_reg - 1>(i, k, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        i = k;
        empty_shift1_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, maxshift_reg - 1>(i, remain, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        i = k + shift1;
        empty_shift2_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, maxshift_reg - 1>(i, remain, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
        i = k + shift1 + shift2;
        empty_shift3_switch<shift1, shift2, shift3, maxshift_smem, maxshift_reg, TILE_B, maxshift_reg - 1>(i, remain, M, N, K, dval_sidx, dval_soff, dval_coo_idx, dval_coo_val, dval_coo_off, dC_val, block_idx_x, block_idx_y, lane_id, tile_sidx_block, tile_coo_val_block, tile_coo_restore_val_block, coo_scope, val_sidx_reg, coo_idx_reg, coo_range_reg, REGB, REGA, REGC, l, r, tile_sidx_size, tile_coo_val_size, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, nnz_block);
    }

    //! write back
    for (int i_tileN = 0; i_tileN < TILE_B / 8; i_tileN++) {
        int row = block_idx_x * 16 + lane_id / 4;
        int col = block_idx_y * TILE_B + i_tileN * 8 + lane_id % 4 * 2;
        dC_val[OFFSET_ROW(row, col, N)] = REGC[i_tileN * 4 + 0];
        dC_val[OFFSET_ROW(row, col + 1, N)] = REGC[i_tileN * 4 + 1];
        dC_val[OFFSET_ROW(row + 8, col, N)] = REGC[i_tileN * 4 + 2];
        dC_val[OFFSET_ROW(row + 8, col + 1, N)] = REGC[i_tileN * 4 + 3];
    }
}


template <int TILE_B>
void ncu_test_all(
    int threadblock_num_x, int threadblock_num_y, int thread_num,
    int M, int N, int K,
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
    //! （shift1=1,shift2=1,shift3=1)
    dtc_spmm_kernel_sstage_rstage<1, 1, 1, 32><<<grid, block>>>(M, N, K,dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC);

}


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

    ncu_test_all<TILE_B>(threadblock_num_x, threadblock_num_y, thread_num, M, N, K,dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC);
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
