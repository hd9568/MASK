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

#define VARLEN 0

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
template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_sidx_size = Int<16>, class TI0 = int, class TI1 = int, class TO0 = int>
class G2sValSidxLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(tile_sidx_size{}, VARLEN));
    using Tensor_i0 = decltype(make_tensor(make_gmem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 dval_sidx;
    // TI1 i;

    // Outputs
    using Shape_o0 = decltype(make_shape(tile_sidx_size{}, NBUF{}));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 tile_sidx_block;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2sValSidxLoadOp(int tid, int lid, int wid, int *tile_sidx_block, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(tile_sidx_size{}, mode_i0_0)), tile_sidx_block(make_tensor(make_smem_ptr(tile_sidx_block), Shape_o0{})) {}

    template<size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_sidx_block(_, buf_idx);
        }
    }

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 dval_sidx, TI1 i) {
        if (lid < 4) {
        auto thr_tiler = make_shape(Int<4>{});
        auto thr_coord = make_coord(lid);
        auto src = local_tile(dval_sidx(_, i), thr_tiler, thr_coord);
        auto dst = local_tile(tile_sidx_block(_, buf_idx), thr_tiler, thr_coord);
        copy(Copy_Atom<UniversalCopy<uint128_t>, int>{}, src, dst);
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TI0 = int, class TI1 = int, class TO0 = int, class TO1 = int>
class G2rCooAtomicFormatLoadOffOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(Int<2>{}, VARLEN));
    using Tensor_i0 = decltype(make_tensor(make_gmem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 dval_coo_off;
    // TI1 i;

    // Outputs
    using Shape_o0 = decltype(make_shape(NBUF{}));
    using Tensor_o0 = decltype(make_tensor<TO0>(Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 l;
    using Shape_o1 = decltype(make_shape(NBUF{}));
    using Tensor_o1 = decltype(make_tensor<TO1>(Shape_o1{}));
    Shape_o1 shape_o1;
    Tensor_o1 r;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rCooAtomicFormatLoadOffOp(int tid, int lid, int wid, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(Int<2>{}, mode_i0_0)) {}

    template<size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return l(buf_idx);
        }
        if constexpr (output_idx == 1) {
            return r(buf_idx);
        }
    }

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 dval_coo_off, TI1 i) {
        l(buf_idx) = dval_coo_off(0, i);
        r(buf_idx) = dval_coo_off(1, i);
    }
};

// shift 2
template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_sidx_size = Int<16>, class TI0 = int, class TO0 = int>
class S2rValSidxLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(tile_sidx_size{}));
    using Tensor_i0 = decltype(make_tensor(make_smem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 tile_sidx_block;

    // Outputs
    using Shape_o0 = decltype(make_shape(tile_sidx_size{}, NBUF{}));
    using Tensor_o0 = decltype(make_tensor<TO0>(Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 val_sidx_reg;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE S2rValSidxLoadOp(int tid, int lid, int wid)
    : tid(tid), lid(lid), wid(wid) {}

    template<size_t buf_idx, size_t output_idx>
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
    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 tile_sidx_block) {
        // __pipeline_wait_prior(wait_num);
        // __syncthreads();
        auto src = tile_sidx_block;
        auto dst = val_sidx_reg(_, buf_idx);
        copy(src, dst);
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_coo_val_size = Int<16>, class TI0 = half, class TI1 = int, class TI2 = int, class TO0 = half>
class G2sCooAtomicFormatLoadValOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    using Tensor_i0 = decltype(make_tensor(make_gmem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 dval_coo_val;
    // TI1 ll;
    // TI2 rr;

    // Outputs
    using Shape_o0 = decltype(make_shape(tile_coo_val_size{}, NBUF{}));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 tile_coo_val_block;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2sCooAtomicFormatLoadValOp(int tid, int lid, int wid, half *tile_coo_val_block, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)), tile_coo_val_block(make_tensor(make_smem_ptr(tile_coo_val_block), Shape_o0{})) {}

    template<size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_coo_val_block(_, buf_idx);
        }
    }

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 dval_coo_val, TI1 ll, TI2 rr) {
        auto thr_tiler = make_shape(_8{});
        auto thr_coord = make_coord(lid);
        auto input = make_tensor(dval_coo_val.data() + ll, make_shape(tile_coo_val_size{}));
        auto output = tile_coo_val_block(_, buf_idx);
        
        if (lid * 8 < rr - ll) {
            auto src = local_tile(input, thr_tiler, thr_coord);
            auto dst = local_tile(output, thr_tiler, thr_coord);
            copy(Copy_Atom<UniversalCopy<uint128_t>, half>{}, src, dst);
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_coo_val_size = Int<16>, class TI0 = int, class TI1 = int, class TI2 = int, class TO0 = int, class TO1 = int>
class G2rCooAtomicFormatLoadIdxOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    using Tensor_i0 = decltype(make_tensor(make_gmem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 dval_coo_idx;
    // TI1 ll;
    // TI2 rr;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<8>{}, NBUF{}));
    using Tensor_o0 = decltype(make_tensor<TO0>(Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 coo_idx_reg;
    using Shape_o1 = decltype(make_shape(NBUF{}));
    using Tensor_o1 = decltype(make_tensor<TO1>(Shape_o1{}));
    Shape_o1 shape_o1;
    Tensor_o1 coo_range_reg;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rCooAtomicFormatLoadIdxOp(int tid, int lid, int wid, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)) {}

    template<size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return coo_idx_reg(_, buf_idx);
        }
        if constexpr (output_idx == 1) {
            return coo_range_reg(buf_idx);
        }
    }

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 dval_coo_idx, TI1 ll, TI2 rr) {
        auto thr_tiler = make_shape(_8{});
        auto thr_coord = make_coord(lid);
        auto input = make_tensor(dval_coo_idx.data() + ll, make_shape(tile_coo_val_size{}));
        coo_range_reg(buf_idx) = rr - ll;
        if (lid * 8 < rr - ll) {
            auto src = local_tile(input, thr_tiler, thr_coord);
            auto dst = coo_idx_reg(_, buf_idx);
            copy(src, dst);
        }
    }
};

// shift 3
template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_coo_val_size = Int<16>, class TI0 = half, class TI1 = int, class TO0 = half>
class G2rBValLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN, VARLEN));
    using Tensor_i0 = decltype(make_tensor(make_gmem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 B_val;
    using Shape_i1 = decltype(make_shape(get<2>(BLK_MNK{})));
    using Tensor_i1 = decltype(make_tensor(make_rmem_ptr((TI1*)nullptr), Shape_i1{}));
    Shape_i1 shape_i1;
    // Tensor_i1 val_sidx_reg;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, get<1>(BLK_MMA_MNK{}), NBUF{}));
    using Tensor_o0 = decltype(make_tensor<TO0>(Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 REGB;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rBValLoadOp(int tid, int lid, int wid, int mode_i0_0, int mode_i0_1)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0, mode_i0_1)) {}

    template<size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGB(_, _, buf_idx);
        }
    }

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 B_val, Tensor_i1 val_sidx_reg) {
        for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
            int row = lid % 4 * 2;
            int col = blockIdx.y * get<1>(BLK_MNK{}) + n_iter * 8 + lid / 4;
            REGB(0, n_iter, buf_idx) = B_val(val_sidx_reg(row + 0), col);
            REGB(1, n_iter, buf_idx) = B_val(val_sidx_reg(row + 1), col);
            REGB(2, n_iter, buf_idx) = B_val(val_sidx_reg(row + 8), col);
            REGB(3, n_iter, buf_idx) = B_val(val_sidx_reg(row + 9), col);
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_coo_val_size = Int<16>, class TI0 = half, class TI1 = int, class TI2 = int, class TO0 = half>
class S2sCooAtomicValRestoreOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(tile_coo_val_size{}));
    using Tensor_i0 = decltype(make_tensor(make_smem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 tile_coo_val_block;
    using Shape_i1 = decltype(make_shape(Int<8>{}));
    using Tensor_i1 = decltype(make_tensor(make_rmem_ptr((TI1*)nullptr), Shape_i1{}));
    Shape_i1 shape_i1;
    // Tensor_i1 coo_idx_reg;
    // TI2 nnz_num;

    // Outputs
    using Shape_o0 = decltype(make_shape(get<0>(BLK_MNK{}), get<2>(BLK_MNK{}), NBUF{}));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 tile_coo_restore_val_block;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE S2sCooAtomicValRestoreOp(int tid, int lid, int wid, half *tile_coo_restore_val_block)
    : tid(tid), lid(lid), wid(wid), tile_coo_restore_val_block(make_tensor(make_smem_ptr(tile_coo_restore_val_block), Shape_o0{})) {}

    template<size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_coo_restore_val_block(_, _, buf_idx);
        }
    }

    template<size_t buf_idx, size_t wait_num>
    CUTE_DEVICE void f(Tensor_i0 tile_coo_val_block, Tensor_i1 coo_idx_reg, TI2 nnz_num) {
        __pipeline_wait_prior(wait_num);
        __syncthreads();
        // TODO: use tiler to distribute the tensor and use clear(Tensor) to set zero
        for (int i_o2s = 0; i_o2s < 4; i_o2s++) {
            *((int*)(tile_coo_restore_val_block(_, _, buf_idx).data().get() + 32 * 2 * i_o2s + lid * 2)) = 0;
        }
        __syncthreads();
        for (int i_restore = 0; i_restore < 8; i_restore++) {
            int idx = coo_idx_reg(i_restore);
            if (lid * 8 + i_restore < nnz_num) {
                tile_coo_restore_val_block(_, _, buf_idx).data()[(idx)] = tile_coo_val_block(lid * 8 + i_restore);
            }
        }
    }

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 tile_coo_val_block, Tensor_i1 coo_idx_reg, TI2 nnz_num) {
        // __pipeline_wait_prior(wait_num);
        // __syncthreads();
        // TODO: use tiler to distribute the tensor and use clear(Tensor) to set zero
        for (int i_o2s = 0; i_o2s < 4; i_o2s++) {
            *((int*)(tile_coo_restore_val_block(_, _, buf_idx).data().get() + 32 * 2 * i_o2s + lid * 2)) = 0;
        }
        __syncthreads();
        for (int i_restore = 0; i_restore < 8; i_restore++) {
            int idx = coo_idx_reg(i_restore);
            if (lid * 8 + i_restore < nnz_num) {
                tile_coo_restore_val_block(_, _, buf_idx).data()[(idx)] = tile_coo_val_block(lid * 8 + i_restore);
            }
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TI0 = half, class TO0 = half>
class S2rAValLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(get<0>(BLK_MNK{}), get<2>(BLK_MNK{})));
    using Tensor_i0 = decltype(make_tensor(make_smem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 tile_coo_restore_val_block;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<8>{}, get<0>(BLK_MMA_MNK{}), NBUF{}));
    using Tensor_o0 = decltype(make_tensor<TO0>(Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 REGA;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE S2rAValLoadOp(int tid, int lid, int wid)
    : tid(tid), lid(lid), wid(wid) {}

    template<size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGA(_, _, buf_idx);
        }
    }

    template<size_t buf_idx, size_t wait_num>
    CUTE_DEVICE void f(Tensor_i0 tile_coo_restore_val_block) {
        // no need to __pipeline_wait_prior here because the prior op s2s_coo_atomic_val_restore does not use cp.async
        __syncthreads();
        int row = lid % 16;
        int col = lid / 16;
        // TODO: bank conflict
        for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) { // now the m_iter would always be zero, which means the for-loop body would run only once
            ldmatrix_m8n8k8_x4(
                (uint32_t*)(REGA(_, _0{}, buf_idx).data()),
                (void*)&tile_coo_restore_val_block((row) * 16 + (col * 8))
            );
        }
    }

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 tile_coo_restore_val_block) {
        // no need to __pipeline_wait_prior here because the prior op s2s_coo_atomic_val_restore does not use cp.async
        __syncthreads();
        int row = lid % 16;
        int col = lid / 16;
        // TODO: bank conflict
        for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) { // now the m_iter would always be zero, which means the for-loop body would run only once
            ldmatrix_m8n8k8_x4(
                (uint32_t*)(REGA(_, _0{}, buf_idx).data()),
                (void*)&tile_coo_restore_val_block((row) * 16 + (col * 8))
            );
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TI0 = half, class TI1 = half, class TO0 = float>
class MmaOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(Int<8>{}, get<0>(BLK_MMA_MNK{})));
    using Tensor_i0 = decltype(make_tensor(make_rmem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 REGA;
    using Shape_i1 = decltype(make_shape(Int<4>{}, get<1>(BLK_MMA_MNK{})));
    using Tensor_i1 = decltype(make_tensor(make_rmem_ptr((TI1*)nullptr), Shape_i1{}));
    Shape_i1 shape_i1;
    // Tensor_i1 REGB;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{})), NBUF{}));
    using Tensor_o0 = decltype(make_tensor<TO0>(Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 REGC;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE MmaOp(int tid, int lid, int wid)
    : tid(tid), lid(lid), wid(wid) {}

    template<size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGC(_, _, buf_idx);
        }
    }

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 REGA, Tensor_i1 REGB) {
        for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) {
            for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
                mma_m16n8k16_fp32(REGC(_, (m_iter, n_iter), buf_idx).data(), (uint32_t*)(REGA(_, m_iter).data()), REGB(_, n_iter).data());
            }
        }
    }
};
// =====================================================================================

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int K_total, int i, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void pipeline(int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {
    if constexpr (i < K_total) {
        g2s_val_sidx_load_op.f<i % nbuf_smem>(val_sidx, l + i);

        g2r_coo_atomic_format_load_off_op.f<i % nbuf_rmem>(val_coo_off, l + i);
        if constexpr (i < shift1) {
            __pipeline_commit();
        }
    }
    // (sstage > (K - 1 - (i - sstage)) ? K - 1 - (i - sstage) : sstage)
    if constexpr (0 <= i - shift1 && i - shift1 < K_total) {
        g2s_coo_atomic_format_load_val_op.template f<(i - shift1) % nbuf_rmem>(val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % nbuf_rmem, 1>()
        );
        __pipeline_commit();
        s2r_val_sidx_load_op.f<(i - shift1) % nbuf_rmem, (K_total > shift1 ? shift1 : K_total)>(g2s_val_sidx_load_op.template output<(i - shift1) % nbuf_rmem, 0>());
        g2r_coo_atomic_format_load_idx_op.f<(i - shift1) % nbuf_rmem>(val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % nbuf_rmem, 1>()
        );
    }
    if constexpr (0 <= i - shift2 - shift1 && i - shift2 - shift1 < K_total) {
        g2r_b_val_load_op.template f<(i - shift1 - shift2) % nbuf_rmem>(B_val, s2r_val_sidx_load_op.template output<(i - shift1 - shift2) % nbuf_rmem, 0>());
        s2s_coo_atomic_val_restore_op.f<(i - shift1 - shift2) % nbuf_rmem, (shift2 > (K_total + shift1 + shift2 - 1 - i) ? (K_total + shift1 + shift2 - 1 - i) : shift2)>(
            g2s_coo_atomic_format_load_val_op.template output<(i - shift1 - shift2) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<(i - shift1 - shift2) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<(i - shift1 - shift2) % nbuf_rmem, 1>()
        );
        s2r_a_val_load_op.f<(i - shift1 - shift2) % nbuf_rmem, (shift2 > (K_total + shift1 + shift2 - 1 - i) ? (K_total + shift1 + shift2 - 1 - i) : shift2)>(
            s2s_coo_atomic_val_restore_op.template output<(i - shift1 - shift2) % nbuf_rmem, 0>()
        );
    }
    if constexpr (0 <= i - shift1 - shift2 - shift3 && i - shift1 - shift2 - shift3 < K_total) {
        mma_op.f<0>(
            s2r_a_val_load_op.template output<(i - shift1 - shift2 - shift3) % nbuf_rmem, 0>(),
            g2r_b_val_load_op.template output<(i - shift1 - shift2 - shift3) % nbuf_rmem, 0>()
        );
    }
    if constexpr (i < K_total + shift1 + shift2 + shift3) {
        pipeline<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, K_total, i + 1>(l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
    }

}

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int K_total, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void pipeline_switch(int k, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {
    if (k == 0) {
        // i = 0
        // i = 1
        // i = 2
        
    }
    else if (k == 1) {
        // i = 0
        g2s_val_sidx_load_op.template f<0>(    val_sidx,l + 0);
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<0>(    val_coo_off,l + 0);
        // i = 1
        g2s_coo_atomic_format_load_val_op.template f<0>(    val_coo_val,g2r_coo_atomic_format_load_off_op.template output<0, 0>(),g2r_coo_atomic_format_load_off_op.template output<0, 1>());
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<0>(    g2s_val_sidx_load_op.template output<0, 0>());
        g2r_coo_atomic_format_load_idx_op.template f<0>(    val_coo_idx,g2r_coo_atomic_format_load_off_op.template output<0, 0>(),g2r_coo_atomic_format_load_off_op.template output<0, 1>());
        // i = 2
        g2r_b_val_load_op.template f<0>(    B_val,s2r_val_sidx_load_op.template output<0, 0>());
        __pipeline_wait_prior(0);
        __syncthreads();
        s2s_coo_atomic_val_restore_op.template f<0>(    g2s_coo_atomic_format_load_val_op.template output<0, 0>(),g2r_coo_atomic_format_load_idx_op.template output<0, 0>(),g2r_coo_atomic_format_load_idx_op.template output<0, 1>());
        s2r_a_val_load_op.template f<0>(    s2s_coo_atomic_val_restore_op.template output<0, 0>());
        // i = 3
        mma_op.template f<0>(    s2r_a_val_load_op.template output<0, 0>(),g2r_b_val_load_op.template output<0, 0>());
        
    }
    else if (k == 2) {
        // i = 0
        g2s_val_sidx_load_op.template f<0>(    val_sidx,l + 0);
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<0>(    val_coo_off,l + 0);
        // i = 1
        g2s_val_sidx_load_op.template f<1>(    val_sidx,l + 1);
        g2s_coo_atomic_format_load_val_op.template f<0>(    val_coo_val,g2r_coo_atomic_format_load_off_op.template output<0, 0>(),g2r_coo_atomic_format_load_off_op.template output<0, 1>());
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<1>(    val_coo_off,l + 1);
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<0>(    g2s_val_sidx_load_op.template output<0, 0>());
        g2r_coo_atomic_format_load_idx_op.template f<0>(    val_coo_idx,g2r_coo_atomic_format_load_off_op.template output<0, 0>(),g2r_coo_atomic_format_load_off_op.template output<0, 1>());
        // i = 2
        g2s_coo_atomic_format_load_val_op.template f<1>(    val_coo_val,g2r_coo_atomic_format_load_off_op.template output<1, 0>(),g2r_coo_atomic_format_load_off_op.template output<1, 1>());
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<1>(    g2s_val_sidx_load_op.template output<1, 0>());
        g2r_coo_atomic_format_load_idx_op.template f<1>(    val_coo_idx,g2r_coo_atomic_format_load_off_op.template output<1, 0>(),g2r_coo_atomic_format_load_off_op.template output<1, 1>());
        g2r_b_val_load_op.template f<0>(    B_val,s2r_val_sidx_load_op.template output<0, 0>());
        __pipeline_wait_prior(1);
        __syncthreads();
        s2s_coo_atomic_val_restore_op.template f<0>(    g2s_coo_atomic_format_load_val_op.template output<0, 0>(),g2r_coo_atomic_format_load_idx_op.template output<0, 0>(),g2r_coo_atomic_format_load_idx_op.template output<0, 1>());
        s2r_a_val_load_op.template f<0>(    s2s_coo_atomic_val_restore_op.template output<0, 0>());
        // i = 3
        g2r_b_val_load_op.template f<1>(    B_val,s2r_val_sidx_load_op.template output<1, 0>());
        __pipeline_wait_prior(0);
        __syncthreads();
        s2s_coo_atomic_val_restore_op.template f<1>(    g2s_coo_atomic_format_load_val_op.template output<1, 0>(),g2r_coo_atomic_format_load_idx_op.template output<1, 0>(),g2r_coo_atomic_format_load_idx_op.template output<1, 1>());
        s2r_a_val_load_op.template f<1>(    s2s_coo_atomic_val_restore_op.template output<1, 0>());
        mma_op.template f<0>(    s2r_a_val_load_op.template output<0, 0>(),g2r_b_val_load_op.template output<0, 0>());
        // i = 4
        mma_op.template f<1>(    s2r_a_val_load_op.template output<1, 0>(),g2r_b_val_load_op.template output<1, 0>());
        
    }
    else if (k == 3) {
        // i = 0
        g2s_val_sidx_load_op.template f<0>(    val_sidx,l + 0);
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<0>(    val_coo_off,l + 0);
        // i = 1
        g2s_val_sidx_load_op.template f<1>(    val_sidx,l + 1);
        g2s_coo_atomic_format_load_val_op.template f<0>(    val_coo_val,g2r_coo_atomic_format_load_off_op.template output<0, 0>(),g2r_coo_atomic_format_load_off_op.template output<0, 1>());
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<1>(    val_coo_off,l + 1);
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<0>(    g2s_val_sidx_load_op.template output<0, 0>());
        g2r_coo_atomic_format_load_idx_op.template f<0>(    val_coo_idx,g2r_coo_atomic_format_load_off_op.template output<0, 0>(),g2r_coo_atomic_format_load_off_op.template output<0, 1>());
        // i = 2
        g2s_val_sidx_load_op.template f<0>(    val_sidx,l + 2);
        g2s_coo_atomic_format_load_val_op.template f<1>(    val_coo_val,g2r_coo_atomic_format_load_off_op.template output<1, 0>(),g2r_coo_atomic_format_load_off_op.template output<1, 1>());
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<0>(    val_coo_off,l + 2);
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<1>(    g2s_val_sidx_load_op.template output<1, 0>());
        g2r_coo_atomic_format_load_idx_op.template f<1>(    val_coo_idx,g2r_coo_atomic_format_load_off_op.template output<1, 0>(),g2r_coo_atomic_format_load_off_op.template output<1, 1>());
        g2r_b_val_load_op.template f<0>(    B_val,s2r_val_sidx_load_op.template output<0, 0>());
        __pipeline_wait_prior(1);
        __syncthreads();
        s2s_coo_atomic_val_restore_op.template f<0>(    g2s_coo_atomic_format_load_val_op.template output<0, 0>(),g2r_coo_atomic_format_load_idx_op.template output<0, 0>(),g2r_coo_atomic_format_load_idx_op.template output<0, 1>());
        s2r_a_val_load_op.template f<0>(    s2s_coo_atomic_val_restore_op.template output<0, 0>());
        // i = 3
        g2s_coo_atomic_format_load_val_op.template f<0>(    val_coo_val,g2r_coo_atomic_format_load_off_op.template output<0, 0>(),g2r_coo_atomic_format_load_off_op.template output<0, 1>());
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<0>(    g2s_val_sidx_load_op.template output<0, 0>());
        g2r_coo_atomic_format_load_idx_op.template f<0>(    val_coo_idx,g2r_coo_atomic_format_load_off_op.template output<0, 0>(),g2r_coo_atomic_format_load_off_op.template output<0, 1>());
        g2r_b_val_load_op.template f<1>(    B_val,s2r_val_sidx_load_op.template output<1, 0>());
        __pipeline_wait_prior(1);
        __syncthreads();
        s2s_coo_atomic_val_restore_op.template f<1>(    g2s_coo_atomic_format_load_val_op.template output<1, 0>(),g2r_coo_atomic_format_load_idx_op.template output<1, 0>(),g2r_coo_atomic_format_load_idx_op.template output<1, 1>());
        s2r_a_val_load_op.template f<1>(    s2s_coo_atomic_val_restore_op.template output<1, 0>());
        mma_op.template f<0>(    s2r_a_val_load_op.template output<0, 0>(),g2r_b_val_load_op.template output<0, 0>());
        // i = 4
        g2r_b_val_load_op.template f<0>(    B_val,s2r_val_sidx_load_op.template output<0, 0>());
        __pipeline_wait_prior(0);
        __syncthreads();
        s2s_coo_atomic_val_restore_op.template f<0>(    g2s_coo_atomic_format_load_val_op.template output<0, 0>(),g2r_coo_atomic_format_load_idx_op.template output<0, 0>(),g2r_coo_atomic_format_load_idx_op.template output<0, 1>());
        s2r_a_val_load_op.template f<0>(    s2s_coo_atomic_val_restore_op.template output<0, 0>());
        mma_op.template f<1>(    s2r_a_val_load_op.template output<1, 0>(),g2r_b_val_load_op.template output<1, 0>());
        // i = 5
        mma_op.template f<0>(    s2r_a_val_load_op.template output<0, 0>(),g2r_b_val_load_op.template output<0, 0>());
        
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int i, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void fill_shift1(int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {
    if constexpr (i == shift1) {
        return;
    } else {
        g2s_val_sidx_load_op.f<i % nbuf_smem>(val_sidx, l + i);
        g2r_coo_atomic_format_load_off_op.f<i % nbuf_rmem>(val_coo_off, l + i);
        __pipeline_commit();
        fill_shift1<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, i + 1>(l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
    }
}

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int i, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void fill_shift2(int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {
    if constexpr (i == shift2) {
        return;
    } else {
        g2s_val_sidx_load_op.f<(i + shift1) % nbuf_smem>(val_sidx, l + i + shift1);
        g2s_coo_atomic_format_load_val_op.template f<i % nbuf_rmem>(val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<i % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<i % nbuf_rmem, 1>()
        );
        __pipeline_commit();
        s2r_val_sidx_load_op.f<i % nbuf_rmem, shift1>(g2s_val_sidx_load_op.template output<i % nbuf_rmem, 0>());

        g2r_coo_atomic_format_load_off_op.f<(i + shift1) % nbuf_rmem>(val_coo_off, l + i + shift1);

        g2r_coo_atomic_format_load_idx_op.f<i % nbuf_rmem>(val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<i % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<i % nbuf_rmem, 1>()
        );

        fill_shift2<shift1, shift2, shift3, nbuf_smem, nbuf_rmem,  i + 1>(l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
    }
}

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int i, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void fill_shift3(int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {
    if constexpr (i == shift3) {
        return;
    } else {
        g2s_val_sidx_load_op.f<(i + shift1 + shift2) % nbuf_smem>(val_sidx, l + i + shift1 + shift2);
        g2s_coo_atomic_format_load_val_op.template f<(i + shift2) % nbuf_rmem>(val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<(i + shift2) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(i + shift2) % nbuf_rmem, 1>()
        );
        __pipeline_commit();
        s2r_val_sidx_load_op.f<(i + shift2) % nbuf_rmem, shift1>(g2s_val_sidx_load_op.template output<(i + shift2) % nbuf_rmem, 0>());
        g2r_b_val_load_op.template f<i % nbuf_rmem>(B_val, s2r_val_sidx_load_op.template output<i % nbuf_rmem, 0>());

        g2r_coo_atomic_format_load_off_op.f<(i + shift1 + shift2) % nbuf_rmem>(val_coo_off, l + i + shift1 + shift2);
        g2r_coo_atomic_format_load_idx_op.f<(i + shift2) % nbuf_rmem>(val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<(i + shift2) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(i + shift2) % nbuf_rmem, 1>()
        );
        s2s_coo_atomic_val_restore_op.f<i % nbuf_rmem, shift2>(
            g2s_coo_atomic_format_load_val_op.template output<i % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<i % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<i % nbuf_rmem, 1>()
        );
        s2r_a_val_load_op.f<i % nbuf_rmem, shift2>(
            s2s_coo_atomic_val_restore_op.template output<i % nbuf_rmem, 0>()
        );

        fill_shift3<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, i + 1>(l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int I, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void loop_step(int i, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {
    g2s_val_sidx_load_op.f<(I + shift1 + shift2 + shift3) % nbuf_smem>(val_sidx, l + i);
    g2s_coo_atomic_format_load_val_op.template f<(I + shift3 + shift2) % nbuf_rmem>(val_coo_val,
        g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % nbuf_rmem, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % nbuf_rmem, 1>()
    );
    __pipeline_commit();
    s2r_val_sidx_load_op.f<(I + shift3 + shift2) % nbuf_rmem, shift1>(g2s_val_sidx_load_op.template output<(I + shift3 + shift2) % nbuf_rmem, 0>());
        g2r_b_val_load_op.template f<(I + shift3) % nbuf_rmem>(B_val, s2r_val_sidx_load_op.template output<(I + shift3) % nbuf_rmem, 0>());

    g2r_coo_atomic_format_load_off_op.f<(I + shift3 + shift1 + shift2) % nbuf_rmem>(val_coo_off, l + i);

    g2r_coo_atomic_format_load_idx_op.f<(I + shift3 + shift2) % nbuf_rmem>(val_coo_idx,
        g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % nbuf_rmem, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % nbuf_rmem, 1>()
    );
    s2s_coo_atomic_val_restore_op.f<(I + shift3) % nbuf_rmem, shift2>(
        g2s_coo_atomic_format_load_val_op.template output<(I + shift3) % nbuf_rmem, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<(I + shift3) % nbuf_rmem, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<(I + shift3) % nbuf_rmem, 1>()
    );
    s2r_a_val_load_op.f<(I + shift3) % nbuf_rmem, shift2>(
        s2s_coo_atomic_val_restore_op.template output<(I + shift3) % nbuf_rmem, 0>()
    );
    mma_op.f<0>(
        s2r_a_val_load_op.template output<I % nbuf_rmem, 0>(),
        g2r_b_val_load_op.template output<I % nbuf_rmem, 0>()
    );

    if constexpr (I + 1 < nbuf_rmem) {
        loop_step<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, I + 1>(i + 1, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int K_remain, int I, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void remainder(int i, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {
    if constexpr (I == K_remain) {
        return;
    } else {
        g2s_val_sidx_load_op.f<(I + shift1 + shift2 + shift3) % nbuf_smem>(val_sidx, l + i);
        g2s_coo_atomic_format_load_val_op.template f<(I + shift3 + shift2) % nbuf_rmem>(val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % nbuf_rmem, 1>()
        );
        __pipeline_commit();
        s2r_val_sidx_load_op.f<(I + shift3 + shift2) % nbuf_rmem, shift1>(g2s_val_sidx_load_op.template output<(I + shift3 + shift2) % nbuf_rmem, 0>());
        g2r_b_val_load_op.template f<(I + shift3) % nbuf_rmem>(B_val, s2r_val_sidx_load_op.template output<(I + shift3) % nbuf_rmem, 0>());

        g2r_coo_atomic_format_load_off_op.f<(I + shift3 + shift1 + shift2) % nbuf_rmem>(val_coo_off, l + i);

        g2r_coo_atomic_format_load_idx_op.f<(I + shift3 + shift2) % nbuf_rmem>(val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % nbuf_rmem, 1>()
        );
        s2s_coo_atomic_val_restore_op.f<(I + shift3) % nbuf_rmem, shift2>(
            g2s_coo_atomic_format_load_val_op.template output<(I + shift3) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<(I + shift3) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<(I + shift3) % nbuf_rmem, 1>()
        );
        s2r_a_val_load_op.f<(I + shift3) % nbuf_rmem, shift2>(
            s2s_coo_atomic_val_restore_op.template output<(I + shift3) % nbuf_rmem, 0>()
        );
        mma_op.f<0>(
            s2r_a_val_load_op.template output<I % nbuf_rmem, 0>(),
            g2r_b_val_load_op.template output<I % nbuf_rmem, 0>()
        );

        remainder<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, K_remain, I + 1>(i + 1, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
    }
}

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int K_remain, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void remainder_switch(int i, int k, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {
    if constexpr (K_remain == 0) {
        return;
    } else {
        if (k - i == K_remain) {
            remainder<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, K_remain, 0>(i, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        } else {
            remainder_switch<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, K_remain - 1>(i, k, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        }
    }
}

// =====================================================================================

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int K_remain, int I, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void empty_shift1(int i, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {  // K is the remain
    if constexpr (I == shift1) {
        return;
    } else {
        g2s_coo_atomic_format_load_val_op.template f<(I + shift3 + shift2 + K_remain) % nbuf_rmem>(val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2 + K_remain) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2 + K_remain) % nbuf_rmem, 1>()
        );
        __pipeline_commit();
        s2r_val_sidx_load_op.f<(I + shift3 + shift2 + K_remain) % nbuf_rmem, shift1>(g2s_val_sidx_load_op.template output<(I + shift3 + shift2 + K_remain) % nbuf_rmem, 0>());
        g2r_b_val_load_op.template f<(I + shift3 + K_remain) % nbuf_rmem>(B_val, s2r_val_sidx_load_op.template output<(I + shift3 + K_remain) % nbuf_rmem, 0>());


        g2r_coo_atomic_format_load_idx_op.template f<(I + shift3 + shift2 + K_remain) % nbuf_rmem>(val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2 + K_remain) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2 + K_remain) % nbuf_rmem, 1>()
        );
        s2s_coo_atomic_val_restore_op.f<(I + shift3 + K_remain) % nbuf_rmem, shift2>(
            g2s_coo_atomic_format_load_val_op.template output<(I + shift3 + K_remain) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<(I + shift3 + K_remain) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<(I + shift3 + K_remain) % nbuf_rmem, 1>()
        );
        s2r_a_val_load_op.f<(I + shift3 + K_remain) % nbuf_rmem, shift2>(
            s2s_coo_atomic_val_restore_op.template output<(I + shift3 + K_remain) % nbuf_rmem, 0>()
        );
        mma_op.f<0>(
            s2r_a_val_load_op.template output<(I + K_remain) % nbuf_rmem, 0>(),
            g2r_b_val_load_op.template output<(I + K_remain) % nbuf_rmem, 0>()
        );

        empty_shift1<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, K_remain, I + 1>(i + 1, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
    }
}

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int K_remain, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void empty_shift1_switch(int i, int remain, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {
    if constexpr (K_remain < 0) {
        return;
    } else {
        if (K_remain == remain) {
            empty_shift1<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, K_remain, 0>(i, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        } else {
            empty_shift1_switch<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, K_remain - 1>(i, remain, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        }
    }
}

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int K_remain, int I, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void empty_shift2(int i, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {
    if constexpr (I == shift2) {
        return;
    } else {

        g2r_b_val_load_op.template f<(I + shift1 + shift3 + K_remain) % nbuf_rmem>(B_val, s2r_val_sidx_load_op.template output<(I + shift1 + shift3 + K_remain) % nbuf_rmem, 0>());

        s2s_coo_atomic_val_restore_op.f<(I + shift1 + shift3 + K_remain) % nbuf_rmem, shift2 - (I + 1)>(
            g2s_coo_atomic_format_load_val_op.template output<(I + shift1 + shift3 + K_remain) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<(I + shift1 + shift3 + K_remain) % nbuf_rmem, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<(I + shift1 + shift3 + K_remain) % nbuf_rmem, 1>()
        );
        s2r_a_val_load_op.f<(I + shift1 + shift3 + K_remain) % nbuf_rmem, shift2 - (I + 1)>(
            s2s_coo_atomic_val_restore_op.template output<(I + shift1 + shift3 + K_remain) % nbuf_rmem, 0>()
        );
        mma_op.f<0>(
            s2r_a_val_load_op.template output<(I + shift1 + K_remain) % nbuf_rmem, 0>(),
            g2r_b_val_load_op.template output<(I + shift1 + K_remain) % nbuf_rmem, 0>()
        );

        empty_shift2<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, K_remain, I + 1>(i + 1, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
    }
}

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int K_remain, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void empty_shift2_switch(int i, int remain, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {
    if constexpr (K_remain < 0) {
        return;

    } else {
        if (remain == K_remain) {
            empty_shift2<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, K_remain, 0>(i, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        } else {
            empty_shift2_switch<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, K_remain - 1>(i, remain, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        }
    }
}

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int K_remain, int I, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void empty_shift3(int i, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {
    if constexpr (I == shift3) {
        return;
    } else {
        mma_op.f<0>(
            s2r_a_val_load_op.template output<(I + shift1 + shift2 + K_remain) % nbuf_rmem, 0>(),
            g2r_b_val_load_op.template output<(I + shift1 + shift2 + K_remain) % nbuf_rmem, 0>()
        );
        empty_shift3<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, K_remain, I + 1>(i + 1, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
    }
}

template <int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem, int K_remain, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOff, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_S2rValSidxLoadOp,  class T_val_sidx, class T_val_soff, class T_B_val, class T_val_coo_val, class T_val_coo_idx, class T_val_coo_off, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp>
__device__ void empty_shift3_switch(int i, int remain, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOff &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_val_sidx &val_sidx, T_val_soff &val_soff, T_B_val &B_val,  T_val_coo_val &val_coo_val,  T_val_coo_idx &val_coo_idx,  T_val_coo_off &val_coo_off, T_MmaOp &mma_op) {
    if constexpr (K_remain < 0) {
        return;
    } else {
        if (K_remain == remain) {
            empty_shift3<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, K_remain, 0>(i, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        } else {
            empty_shift3_switch<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, K_remain - 1>(i, remain, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        }
    }
}

// =====================================================================================

template <
    class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK,
    class tile_sidx_size, class tile_coo_val_size, // parameters
    int shift1, int shift2, int shift3, int nbuf_smem, int nbuf_rmem
>
__global__ void dtc_spmm_kernel_sstage_rstage(
    int* dval_sidx,
    int* dval_soff,
    half* dB_val,
    int* dval_coo_idx,
    half* dval_coo_val,
    int* dval_coo_off,
    float* dC_val,
    int M, int N, int K, int Mo, int nnz_block, int nnz
) {
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int wid = tid / 32;

    // gmem tensors
    auto val_sidx = make_tensor(make_gmem_ptr(dval_sidx), make_shape(tile_sidx_size{}, nnz_block));
    auto val_soff = make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo));
    auto B_val = make_tensor(make_gmem_ptr(dB_val), make_shape(K, N));
    auto val_coo_idx = make_tensor(make_gmem_ptr(dval_coo_idx), make_shape(nnz));
    auto val_coo_val = make_tensor(make_gmem_ptr(dval_coo_val), make_shape(nnz));
    auto val_coo_off = make_tensor(make_gmem_ptr(dval_coo_off), make_shape(_2{}, nnz_block));
    auto C_val = make_tensor(make_gmem_ptr(dC_val), make_layout(make_shape(M, N), make_stride(N, 1)));

    // shift 1
    using g2s_val_sidx_load_op_t = G2sValSidxLoadOp<Int<nbuf_smem>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_sidx_size>;
    using g2s_val_sidx_load_op_shape_o0 = typename g2s_val_sidx_load_op_t::Shape_o0;
    __shared__ int tile_sidx_block[cosize_v<Layout<g2s_val_sidx_load_op_shape_o0>>];
    g2s_val_sidx_load_op_t g2s_val_sidx_load_op{
        tid, lid, wid, 
        tile_sidx_block, 
        nnz_block,
    };

    using g2r_coo_atomic_format_load_off_op_t = G2rCooAtomicFormatLoadOffOp<Int<nbuf_rmem>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    g2r_coo_atomic_format_load_off_op_t g2r_coo_atomic_format_load_off_op{
        tid, lid, wid, 
        nnz_block,
    };

    // shift 2
    using s2r_val_sidx_load_op_t = S2rValSidxLoadOp<Int<nbuf_rmem>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_sidx_size>;
    s2r_val_sidx_load_op_t s2r_val_sidx_load_op{
        tid, lid, wid,
    };

    using g2s_coo_atomic_format_load_val_op_t = G2sCooAtomicFormatLoadValOp<Int<nbuf_smem>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_coo_val_size>;
    using g2s_coo_atomic_format_load_val_op_shape_o0 = typename g2s_coo_atomic_format_load_val_op_t::Shape_o0;
    __shared__ half tile_coo_val_block[cosize_v<Layout<g2s_coo_atomic_format_load_val_op_shape_o0>>];
    g2s_coo_atomic_format_load_val_op_t g2s_coo_atomic_format_load_val_op{
        tid, lid, wid, 
        tile_coo_val_block, 
        nnz,
    };

    using g2r_coo_atomic_format_load_idx_op_t = G2rCooAtomicFormatLoadIdxOp<Int<nbuf_rmem>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_coo_val_size>;
    g2r_coo_atomic_format_load_idx_op_t g2r_coo_atomic_format_load_idx_op{
        tid, lid, wid, 
        nnz,
    };

    // shift 3
    using g2r_b_val_load_op_t = G2rBValLoadOp<Int<nbuf_rmem>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    g2r_b_val_load_op_t g2r_b_val_load_op{
        tid, lid, wid, 
        K, N,
    };

    using s2s_coo_atomic_val_restore_op_t = S2sCooAtomicValRestoreOp<Int<nbuf_smem>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_coo_val_size>;
    using s2s_coo_atomic_val_restore_op_shape_o0 = typename s2s_coo_atomic_val_restore_op_t::Shape_o0;
    __shared__ half tile_coo_restore_val_block[cosize_v<Layout<s2s_coo_atomic_val_restore_op_shape_o0>>];
    s2s_coo_atomic_val_restore_op_t s2s_coo_atomic_val_restore_op{
        tid, lid, wid, 
        tile_coo_restore_val_block,
    };

    using s2r_a_val_load_op_t = S2rAValLoadOp<Int<nbuf_rmem>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    s2r_a_val_load_op_t s2r_a_val_load_op{
        tid, lid, wid,
    };

    using mma_op_t = MmaOp<Int<1> /* reduction */, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    mma_op_t mma_op{
        tid, lid, wid,
    };

    int l = val_soff(blockIdx.x);
    int r = val_soff(blockIdx.x + 1);
    int k = r - l;

    if (k <= shift1 + shift2 + shift3) {
        pipeline_switch<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, shift1 + shift2 + shift3>(k, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
    } else {
        fill_shift1<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, 0>(l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        fill_shift2<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, 0>(l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        fill_shift3<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, 0>(l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        int i;
        for (i = shift1 + shift2 + shift3; i + nbuf_rmem <= k; i += nbuf_rmem) {
            loop_step<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, 0>(i, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        }
        int remain = k - i;
        remainder_switch<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, nbuf_rmem - 1>(i, k, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        i = k;
        empty_shift1_switch<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, nbuf_rmem - 1>(i, remain, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        i = k + shift1;
        empty_shift2_switch<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, nbuf_rmem - 1>(i, remain, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
        i = k + shift1 + shift2;
        empty_shift3_switch<shift1, shift2, shift3, nbuf_smem, nbuf_rmem, nbuf_rmem - 1>(i, remain, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, val_sidx, val_soff, B_val, val_coo_val, val_coo_idx, val_coo_off, mma_op);
    }

    //! write back
    auto REGC = mma_op.template output<0, 0>();
    for (int i_tileN = 0; i_tileN < get<1>(BLK_MNK{}) / 8; i_tileN++) {
        int row = blockIdx.x * 16 + lid / 4;
        int col = blockIdx.y * get<1>(BLK_MNK{}) + i_tileN * 8 + lid % 4 * 2;
        C_val(row, col) = REGC(0, i_tileN);
        C_val(row, col + 1) = REGC(1, i_tileN);
        C_val(row + 8, col) = REGC(2, i_tileN);
        C_val(row + 8, col + 1) = REGC(3, i_tileN);
    }
}


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


    constexpr int ROW_WINDOW_M = 16;
    constexpr int TC_BLOCK_K = 16;
    constexpr int TC_BLOCK_M = 16;

    constexpr auto blk_mnk = make_shape(Int<TC_BLOCK_M>{}, Int<TILE_B>{}, Int<TC_BLOCK_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(_16{}, _8{}, _16{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk = make_shape(Int<TC_BLOCK_M / 16>{}, Int<TILE_B / 8>{}, Int<TC_BLOCK_K / 16>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);

    using tile_sidx_size = Int<TC_BLOCK_K>;
    using tile_coo_val_size = Int<TC_BLOCK_M * TC_BLOCK_K>;

    constexpr int shift1 = 1;
    constexpr int shift2 = 1;
    constexpr int shift3 = 1;
    constexpr int maxshift1_2 = (shift1 > shift2 ? shift1 : shift2);
    constexpr int nbuf_rmem = (maxshift1_2 > shift3 ? maxshift1_2 : shift3) + 1;
    constexpr int nbuf_smem = nbuf_rmem;
    // constexpr int nbuf_smem = maxshift1_2 + 1;

    //! （shift1=1,shift2=1,shift3=1)
    dtc_spmm_kernel_sstage_rstage<
        BLK_MNK, MMA_MNK, BLK_MMA_MNK,
        tile_sidx_size, tile_coo_val_size,
        shift1, shift2, shift3, nbuf_smem, nbuf_rmem
    ><<<grid, block>>>(dval_sidx, dval_soff, dB_val, dval_coo_idx, dval_coo_val, dval_coo_off, dC, M, N, K, Mo, nnz_block, nnz);

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
    constexpr int TC_BLOCK_M = 16;

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
