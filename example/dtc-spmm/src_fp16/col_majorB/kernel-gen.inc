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

// stage 0
template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_sidx_size = Int<16>, class TI0 = int, class TI1 = int, class TO0 = int>
class G2sValSidxLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(tile_sidx_size{}, VARLEN));
    using Tensor_i0 = decltype(make_tensor(make_gmem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // Tensor_i0 val_sidx;
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
    CUTE_DEVICE void f(Tensor_i0 val_sidx, TI1 i) {
        if (lid < 4) {
        auto thr_tiler = make_shape(Int<4>{});
        auto thr_coord = make_coord(lid);
        auto src = local_tile(val_sidx(_, i), thr_tiler, thr_coord);
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
    // Tensor_i0 val_coo_off;
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
    CUTE_DEVICE void f(Tensor_i0 val_coo_off, TI1 i) {
        l(buf_idx) = val_coo_off(0, i);
        r(buf_idx) = val_coo_off(1, i);
    }
};

// stage 1
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

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 tile_sidx_block) {
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
    // Tensor_i0 val_coo_val;
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
    CUTE_DEVICE void f(Tensor_i0 val_coo_val, TI1 ll, TI2 rr) {
        auto thr_tiler = make_shape(_8{});
        auto thr_coord = make_coord(lid);
        auto input = make_tensor(val_coo_val.data() + ll, make_shape(tile_coo_val_size{}));
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
    // Tensor_i0 val_coo_idx;
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
    CUTE_DEVICE void f(Tensor_i0 val_coo_idx, TI1 ll, TI2 rr) {
        auto thr_tiler = make_shape(_8{});
        auto thr_coord = make_coord(lid);
        auto input = make_tensor(val_coo_idx.data() + ll, make_shape(tile_coo_val_size{}));
        coo_range_reg(buf_idx) = rr - ll;
        if (lid * 8 < rr - ll) {
            auto src = local_tile(input, thr_tiler, thr_coord);
            auto dst = coo_idx_reg(_, buf_idx);
            copy(src, dst);
        }
    }
};

// stage 2
template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TI0 = half, class TI1 = int, class TO0 = half>
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

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 tile_coo_val_block, Tensor_i1 coo_idx_reg, TI2 nnz_num) {
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

    template<size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 tile_coo_restore_val_block) {
        
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

// stage 3
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

// short pipe
template <class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOffOp, class T_S2rValSidxLoadOp, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp, class T_B_val, class T_C_val, class T_val_coo_idx, class T_val_coo_off, class T_val_coo_val, class T_val_sidx, class T_val_soff>
__device__ void short_pipe(int k, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOffOp &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_MmaOp &mma_op, T_B_val &B_val, T_C_val &C_val, T_val_coo_idx &val_coo_idx, T_val_coo_off &val_coo_off, T_val_coo_val &val_coo_val, T_val_sidx &val_sidx, T_val_soff &val_soff) {
    if (k == 0) {
        // i = 0
        
        
        // i = 1
        
        
        // i = 2
        
    }
    else if (k == 1) {
        // i = 0
        g2s_val_sidx_load_op.template f<0>(
            val_sidx,
            l + 0
        );
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<0>(
            val_coo_off,
            l + 0
        );
        
        // i = 1
        g2s_coo_atomic_format_load_val_op.template f<0>(
            val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<0>(
            g2s_val_sidx_load_op.template output<0, 0>()
        );g2r_coo_atomic_format_load_idx_op.template f<0>(
            val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        
        // i = 2
        g2r_b_val_load_op.template f<0>(
            B_val,
            s2r_val_sidx_load_op.template output<0, 0>()
        );__pipeline_wait_prior(0);
        __syncthreads();
        s2s_coo_atomic_val_restore_op.template f<0>(
            g2s_coo_atomic_format_load_val_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<0, 1>()
        );s2r_a_val_load_op.template f<0>(
            s2s_coo_atomic_val_restore_op.template output<0, 0>()
        );
        
        // i = 3
        mma_op.template f<0>(
            s2r_a_val_load_op.template output<0, 0>(),
            g2r_b_val_load_op.template output<0, 0>()
        );
    }
    else if (k == 2) {
        // i = 0
        g2s_val_sidx_load_op.template f<0>(
            val_sidx,
            l + 0
        );
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<0>(
            val_coo_off,
            l + 0
        );
        
        // i = 1
        g2s_val_sidx_load_op.template f<1>(
            val_sidx,
            l + 1
        );
        g2s_coo_atomic_format_load_val_op.template f<0>(
            val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<1>(
            val_coo_off,
            l + 1
        );__pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<0>(
            g2s_val_sidx_load_op.template output<0, 0>()
        );g2r_coo_atomic_format_load_idx_op.template f<0>(
            val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        
        // i = 2
        g2s_coo_atomic_format_load_val_op.template f<1>(
            val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 1>()
        );
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<1>(
            g2s_val_sidx_load_op.template output<1, 0>()
        );g2r_coo_atomic_format_load_idx_op.template f<1>(
            val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 1>()
        );g2r_b_val_load_op.template f<0>(
            B_val,
            s2r_val_sidx_load_op.template output<0, 0>()
        );__pipeline_wait_prior(1);
        __syncthreads();
        s2s_coo_atomic_val_restore_op.template f<0>(
            g2s_coo_atomic_format_load_val_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<0, 1>()
        );s2r_a_val_load_op.template f<0>(
            s2s_coo_atomic_val_restore_op.template output<0, 0>()
        );
        
        // i = 3
        g2r_b_val_load_op.template f<1>(
            B_val,
            s2r_val_sidx_load_op.template output<1, 0>()
        );__pipeline_wait_prior(0);
        __syncthreads();
        s2s_coo_atomic_val_restore_op.template f<1>(
            g2s_coo_atomic_format_load_val_op.template output<1, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<1, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<1, 1>()
        );s2r_a_val_load_op.template f<1>(
            s2s_coo_atomic_val_restore_op.template output<1, 0>()
        );mma_op.template f<0>(
            s2r_a_val_load_op.template output<0, 0>(),
            g2r_b_val_load_op.template output<0, 0>()
        );
        
        // i = 4
        mma_op.template f<1>(
            s2r_a_val_load_op.template output<1, 0>(),
            g2r_b_val_load_op.template output<1, 0>()
        );
    }
    else if (k == 3) {
        // i = 0
        g2s_val_sidx_load_op.template f<0>(
            val_sidx,
            l + 0
        );
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<0>(
            val_coo_off,
            l + 0
        );
        
        // i = 1
        g2s_val_sidx_load_op.template f<1>(
            val_sidx,
            l + 1
        );
        g2s_coo_atomic_format_load_val_op.template f<0>(
            val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<1>(
            val_coo_off,
            l + 1
        );__pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<0>(
            g2s_val_sidx_load_op.template output<0, 0>()
        );g2r_coo_atomic_format_load_idx_op.template f<0>(
            val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        
        // i = 2
        g2s_val_sidx_load_op.template f<0>(
            val_sidx,
            l + 2
        );
        g2s_coo_atomic_format_load_val_op.template f<1>(
            val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 1>()
        );
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<0>(
            val_coo_off,
            l + 2
        );__pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<1>(
            g2s_val_sidx_load_op.template output<1, 0>()
        );g2r_coo_atomic_format_load_idx_op.template f<1>(
            val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 1>()
        );g2r_b_val_load_op.template f<0>(
            B_val,
            s2r_val_sidx_load_op.template output<0, 0>()
        );__pipeline_wait_prior(1);
        __syncthreads();
        s2s_coo_atomic_val_restore_op.template f<0>(
            g2s_coo_atomic_format_load_val_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<0, 1>()
        );s2r_a_val_load_op.template f<0>(
            s2s_coo_atomic_val_restore_op.template output<0, 0>()
        );
        
        // i = 3
        g2s_coo_atomic_format_load_val_op.template f<0>(
            val_coo_val,
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<0>(
            g2s_val_sidx_load_op.template output<0, 0>()
        );g2r_coo_atomic_format_load_idx_op.template f<0>(
            val_coo_idx,
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );g2r_b_val_load_op.template f<1>(
            B_val,
            s2r_val_sidx_load_op.template output<1, 0>()
        );__pipeline_wait_prior(1);
        __syncthreads();
        s2s_coo_atomic_val_restore_op.template f<1>(
            g2s_coo_atomic_format_load_val_op.template output<1, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<1, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<1, 1>()
        );s2r_a_val_load_op.template f<1>(
            s2s_coo_atomic_val_restore_op.template output<1, 0>()
        );mma_op.template f<0>(
            s2r_a_val_load_op.template output<0, 0>(),
            g2r_b_val_load_op.template output<0, 0>()
        );
        
        // i = 4
        g2r_b_val_load_op.template f<0>(
            B_val,
            s2r_val_sidx_load_op.template output<0, 0>()
        );__pipeline_wait_prior(0);
        __syncthreads();
        s2s_coo_atomic_val_restore_op.template f<0>(
            g2s_coo_atomic_format_load_val_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_idx_op.template output<0, 1>()
        );s2r_a_val_load_op.template f<0>(
            s2s_coo_atomic_val_restore_op.template output<0, 0>()
        );mma_op.template f<1>(
            s2r_a_val_load_op.template output<1, 0>(),
            g2r_b_val_load_op.template output<1, 0>()
        );
        
        // i = 5
        mma_op.template f<0>(
            s2r_a_val_load_op.template output<0, 0>(),
            g2r_b_val_load_op.template output<0, 0>()
        );
    }
    
}

// fill
template <class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOffOp, class T_S2rValSidxLoadOp, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp, class T_B_val, class T_C_val, class T_val_coo_idx, class T_val_coo_off, class T_val_coo_val, class T_val_sidx, class T_val_soff>
__device__ void fill(int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOffOp &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_MmaOp &mma_op, T_B_val &B_val, T_C_val &C_val, T_val_coo_idx &val_coo_idx, T_val_coo_off &val_coo_off, T_val_coo_val &val_coo_val, T_val_sidx &val_sidx, T_val_soff &val_soff) {
    // stage 0..0
    g2s_val_sidx_load_op.template f<0>(
        val_sidx,
        l + 0
    );
    __pipeline_commit();
    g2r_coo_atomic_format_load_off_op.template f<0>(
        val_coo_off,
        l + 0
    );
    // stage 0..1
    g2s_val_sidx_load_op.template f<1>(
        val_sidx,
        l + 1
    );
    g2s_coo_atomic_format_load_val_op.template f<0>(
        val_coo_val,
        g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<0, 1>()
    );
    __pipeline_commit();
    g2r_coo_atomic_format_load_off_op.template f<1>(
        val_coo_off,
        l + 1
    );
    __pipeline_wait_prior(1);
    __syncthreads();
    s2r_val_sidx_load_op.template f<0>(
        g2s_val_sidx_load_op.template output<0, 0>()
    );
    g2r_coo_atomic_format_load_idx_op.template f<0>(
        val_coo_idx,
        g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<0, 1>()
    );
    // stage 0..2
    g2s_val_sidx_load_op.template f<0>(
        val_sidx,
        l + 2
    );
    g2s_coo_atomic_format_load_val_op.template f<1>(
        val_coo_val,
        g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<1, 1>()
    );
    __pipeline_commit();
    g2r_coo_atomic_format_load_off_op.template f<0>(
        val_coo_off,
        l + 2
    );
    __pipeline_wait_prior(1);
    __syncthreads();
    s2r_val_sidx_load_op.template f<1>(
        g2s_val_sidx_load_op.template output<1, 0>()
    );
    g2r_coo_atomic_format_load_idx_op.template f<1>(
        val_coo_idx,
        g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<1, 1>()
    );
    g2r_b_val_load_op.template f<0>(
        B_val,
        s2r_val_sidx_load_op.template output<0, 0>()
    );
    __pipeline_wait_prior(1);
    __syncthreads();
    s2s_coo_atomic_val_restore_op.template f<0>(
        g2s_coo_atomic_format_load_val_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<0, 1>()
    );
    __syncthreads();
    s2r_a_val_load_op.template f<0>(
        s2s_coo_atomic_val_restore_op.template output<0, 0>()
    );
}

// loop_step
template <class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOffOp, class T_S2rValSidxLoadOp, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp, class T_B_val, class T_C_val, class T_val_coo_idx, class T_val_coo_off, class T_val_coo_val, class T_val_sidx, class T_val_soff>
__device__ void loop_step(int i, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOffOp &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_MmaOp &mma_op, T_B_val &B_val, T_C_val &C_val, T_val_coo_idx &val_coo_idx, T_val_coo_off &val_coo_off, T_val_coo_val &val_coo_val, T_val_sidx &val_sidx, T_val_soff &val_soff) {
    // step 0
    g2s_val_sidx_load_op.template f<1>(
        val_sidx,
        l + i + (0)
    );
    g2s_coo_atomic_format_load_val_op.template f<0>(
        val_coo_val,
        g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<0, 1>()
    );
    __pipeline_commit();
    g2r_coo_atomic_format_load_off_op.template f<1>(
        val_coo_off,
        l + i + (0)
    );
    __pipeline_wait_prior(1);
    __syncthreads();
    s2r_val_sidx_load_op.template f<0>(
        g2s_val_sidx_load_op.template output<0, 0>()
    );
    g2r_coo_atomic_format_load_idx_op.template f<0>(
        val_coo_idx,
        g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<0, 1>()
    );
    g2r_b_val_load_op.template f<1>(
        B_val,
        s2r_val_sidx_load_op.template output<1, 0>()
    );
    __pipeline_wait_prior(1);
    __syncthreads();
    s2s_coo_atomic_val_restore_op.template f<1>(
        g2s_coo_atomic_format_load_val_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<1, 1>()
    );
    __syncthreads();
    s2r_a_val_load_op.template f<1>(
        s2s_coo_atomic_val_restore_op.template output<1, 0>()
    );
    mma_op.template f<0>(
        s2r_a_val_load_op.template output<0, 0>(),
        g2r_b_val_load_op.template output<0, 0>()
    );
    // step 1
    g2s_val_sidx_load_op.template f<0>(
        val_sidx,
        l + i + (1)
    );
    g2s_coo_atomic_format_load_val_op.template f<1>(
        val_coo_val,
        g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<1, 1>()
    );
    __pipeline_commit();
    g2r_coo_atomic_format_load_off_op.template f<0>(
        val_coo_off,
        l + i + (1)
    );
    __pipeline_wait_prior(1);
    __syncthreads();
    s2r_val_sidx_load_op.template f<1>(
        g2s_val_sidx_load_op.template output<1, 0>()
    );
    g2r_coo_atomic_format_load_idx_op.template f<1>(
        val_coo_idx,
        g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<1, 1>()
    );
    g2r_b_val_load_op.template f<0>(
        B_val,
        s2r_val_sidx_load_op.template output<0, 0>()
    );
    __pipeline_wait_prior(1);
    __syncthreads();
    s2s_coo_atomic_val_restore_op.template f<0>(
        g2s_coo_atomic_format_load_val_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<0, 1>()
    );
    __syncthreads();
    s2r_a_val_load_op.template f<0>(
        s2s_coo_atomic_val_restore_op.template output<0, 0>()
    );
    mma_op.template f<1>(
        s2r_a_val_load_op.template output<1, 0>(),
        g2r_b_val_load_op.template output<1, 0>()
    );
}

// remainder
template <class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOffOp, class T_S2rValSidxLoadOp, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp, class T_B_val, class T_C_val, class T_val_coo_idx, class T_val_coo_off, class T_val_coo_val, class T_val_sidx, class T_val_soff>
__device__ void remains_1(int i, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOffOp &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_MmaOp &mma_op, T_B_val &B_val, T_C_val &C_val, T_val_coo_idx &val_coo_idx, T_val_coo_off &val_coo_off, T_val_coo_val &val_coo_val, T_val_sidx &val_sidx, T_val_soff &val_soff) {
    // total remains 1, current 0
    g2s_val_sidx_load_op.template f<1>(
        val_sidx,
        l + i + (0)
    );
    g2s_coo_atomic_format_load_val_op.template f<0>(
        val_coo_val,
        g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<0, 1>()
    );
    __pipeline_commit();
    g2r_coo_atomic_format_load_off_op.template f<1>(
        val_coo_off,
        l + i + (0)
    );
    __pipeline_wait_prior(1);
    __syncthreads();
    s2r_val_sidx_load_op.template f<0>(
        g2s_val_sidx_load_op.template output<0, 0>()
    );
    g2r_coo_atomic_format_load_idx_op.template f<0>(
        val_coo_idx,
        g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<0, 1>()
    );
    g2r_b_val_load_op.template f<1>(
        B_val,
        s2r_val_sidx_load_op.template output<1, 0>()
    );
    __pipeline_wait_prior(1);
    __syncthreads();
    s2s_coo_atomic_val_restore_op.template f<1>(
        g2s_coo_atomic_format_load_val_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<1, 1>()
    );
    __syncthreads();
    s2r_a_val_load_op.template f<1>(
        s2s_coo_atomic_val_restore_op.template output<1, 0>()
    );
    mma_op.template f<0>(
        s2r_a_val_load_op.template output<0, 0>(),
        g2r_b_val_load_op.template output<0, 0>()
    );
}

template <class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOffOp, class T_S2rValSidxLoadOp, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp, class T_B_val, class T_C_val, class T_val_coo_idx, class T_val_coo_off, class T_val_coo_val, class T_val_sidx, class T_val_soff>
__device__ void remainder(int i, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOffOp &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_MmaOp &mma_op, T_B_val &B_val, T_C_val &C_val, T_val_coo_idx &val_coo_idx, T_val_coo_off &val_coo_off, T_val_coo_val &val_coo_val, T_val_sidx &val_sidx, T_val_soff &val_soff) {
    if (r - l - i == 1) {
        remains_1(i, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, mma_op, B_val, C_val, val_coo_idx, val_coo_off, val_coo_val, val_sidx, val_soff);
    }
}

// empties
template <class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOffOp, class T_S2rValSidxLoadOp, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp, class T_B_val, class T_C_val, class T_val_coo_idx, class T_val_coo_off, class T_val_coo_val, class T_val_sidx, class T_val_soff>
__device__ void empty_0(int i, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOffOp &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_MmaOp &mma_op, T_B_val &B_val, T_C_val &C_val, T_val_coo_idx &val_coo_idx, T_val_coo_off &val_coo_off, T_val_coo_val &val_coo_val, T_val_sidx &val_sidx, T_val_soff &val_soff) {
    // stage 1..3
    g2s_coo_atomic_format_load_val_op.template f<1>(
        val_coo_val,
        g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<1, 1>()
    );
    __pipeline_commit();
    __pipeline_wait_prior(1);
    __syncthreads();
    s2r_val_sidx_load_op.template f<1>(
        g2s_val_sidx_load_op.template output<1, 0>()
    );
    g2r_coo_atomic_format_load_idx_op.template f<1>(
        val_coo_idx,
        g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<1, 1>()
    );
    g2r_b_val_load_op.template f<0>(
        B_val,
        s2r_val_sidx_load_op.template output<0, 0>()
    );
    __pipeline_wait_prior(1);
    __syncthreads();
    s2s_coo_atomic_val_restore_op.template f<0>(
        g2s_coo_atomic_format_load_val_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<0, 1>()
    );
    __syncthreads();
    s2r_a_val_load_op.template f<0>(
        s2s_coo_atomic_val_restore_op.template output<0, 0>()
    );
    mma_op.template f<1>(
        s2r_a_val_load_op.template output<1, 0>(),
        g2r_b_val_load_op.template output<1, 0>()
    );
    // stage 2..3
    g2r_b_val_load_op.template f<1>(
        B_val,
        s2r_val_sidx_load_op.template output<1, 0>()
    );
    __pipeline_wait_prior(0);
    __syncthreads();
    s2s_coo_atomic_val_restore_op.template f<1>(
        g2s_coo_atomic_format_load_val_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<1, 1>()
    );
    __syncthreads();
    s2r_a_val_load_op.template f<1>(
        s2s_coo_atomic_val_restore_op.template output<1, 0>()
    );
    mma_op.template f<0>(
        s2r_a_val_load_op.template output<0, 0>(),
        g2r_b_val_load_op.template output<0, 0>()
    );
    // stage 3..3
    mma_op.template f<1>(
        s2r_a_val_load_op.template output<1, 0>(),
        g2r_b_val_load_op.template output<1, 0>()
    );
}

template <class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOffOp, class T_S2rValSidxLoadOp, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp, class T_B_val, class T_C_val, class T_val_coo_idx, class T_val_coo_off, class T_val_coo_val, class T_val_sidx, class T_val_soff>
__device__ void empty_1(int i, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOffOp &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_MmaOp &mma_op, T_B_val &B_val, T_C_val &C_val, T_val_coo_idx &val_coo_idx, T_val_coo_off &val_coo_off, T_val_coo_val &val_coo_val, T_val_sidx &val_sidx, T_val_soff &val_soff) {
    // stage 1..3
    g2s_coo_atomic_format_load_val_op.template f<0>(
        val_coo_val,
        g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<0, 1>()
    );
    __pipeline_commit();
    __pipeline_wait_prior(1);
    __syncthreads();
    s2r_val_sidx_load_op.template f<0>(
        g2s_val_sidx_load_op.template output<0, 0>()
    );
    g2r_coo_atomic_format_load_idx_op.template f<0>(
        val_coo_idx,
        g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_off_op.template output<0, 1>()
    );
    g2r_b_val_load_op.template f<1>(
        B_val,
        s2r_val_sidx_load_op.template output<1, 0>()
    );
    __pipeline_wait_prior(1);
    __syncthreads();
    s2s_coo_atomic_val_restore_op.template f<1>(
        g2s_coo_atomic_format_load_val_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<1, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<1, 1>()
    );
    __syncthreads();
    s2r_a_val_load_op.template f<1>(
        s2s_coo_atomic_val_restore_op.template output<1, 0>()
    );
    mma_op.template f<0>(
        s2r_a_val_load_op.template output<0, 0>(),
        g2r_b_val_load_op.template output<0, 0>()
    );
    // stage 2..3
    g2r_b_val_load_op.template f<0>(
        B_val,
        s2r_val_sidx_load_op.template output<0, 0>()
    );
    __pipeline_wait_prior(0);
    __syncthreads();
    s2s_coo_atomic_val_restore_op.template f<0>(
        g2s_coo_atomic_format_load_val_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<0, 0>(),
        g2r_coo_atomic_format_load_idx_op.template output<0, 1>()
    );
    __syncthreads();
    s2r_a_val_load_op.template f<0>(
        s2s_coo_atomic_val_restore_op.template output<0, 0>()
    );
    mma_op.template f<1>(
        s2r_a_val_load_op.template output<1, 0>(),
        g2r_b_val_load_op.template output<1, 0>()
    );
    // stage 3..3
    mma_op.template f<0>(
        s2r_a_val_load_op.template output<0, 0>(),
        g2r_b_val_load_op.template output<0, 0>()
    );
}

template <class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOffOp, class T_S2rValSidxLoadOp, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp, class T_B_val, class T_C_val, class T_val_coo_idx, class T_val_coo_off, class T_val_coo_val, class T_val_sidx, class T_val_soff>
__device__ void empty(int i, int l, int r, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOffOp &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_MmaOp &mma_op, T_B_val &B_val, T_C_val &C_val, T_val_coo_idx &val_coo_idx, T_val_coo_off &val_coo_off, T_val_coo_val &val_coo_val, T_val_sidx &val_sidx, T_val_soff &val_soff) {
    if ((r - l - i + (3)) % 2 == 0) {
        empty_0(i, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, mma_op, B_val, C_val, val_coo_idx, val_coo_off, val_coo_val, val_sidx, val_soff);
    }
    else if ((r - l - i + (3)) % 2 == 1) {
        empty_1(i, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, mma_op, B_val, C_val, val_coo_idx, val_coo_off, val_coo_val, val_sidx, val_soff);
    }
}

// GLOBAL FUNCTION
template <
	class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK
	, class tile_coo_val_size, class tile_sidx_size
> __global__ void dtc_spmm_kernel_sstage_rstage(
    half *dB_val,
    float *dC_val,
    int *dval_coo_idx,
    int *dval_coo_off,
    half *dval_coo_val,
    int *dval_sidx,
    int *dval_soff,
    int K, int M, int Mo, int N, int nnz, int nnz_block
) {
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int wid = tid / 32;

    // gmem tensors
    auto B_val = make_tensor(make_gmem_ptr(dB_val), make_shape(K, N));
    auto C_val = make_tensor(make_gmem_ptr(dC_val), make_shape(N, M));
    auto val_coo_idx = make_tensor(make_gmem_ptr(dval_coo_idx), make_shape(nnz));
    auto val_coo_off = make_tensor(make_gmem_ptr(dval_coo_off), make_shape(Int<2>{}, nnz_block));
    auto val_coo_val = make_tensor(make_gmem_ptr(dval_coo_val), make_shape(nnz));
    auto val_sidx = make_tensor(make_gmem_ptr(dval_sidx), make_shape(tile_sidx_size{}, nnz_block));
    auto val_soff = make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo));

    // stage 0
    using g2s_val_sidx_load_op_t = G2sValSidxLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_sidx_size>;
    using g2s_val_sidx_load_op_shape_o0 = typename g2s_val_sidx_load_op_t::Shape_o0;
    __shared__ int tile_sidx_block[cosize_v<Layout<g2s_val_sidx_load_op_shape_o0>>];
    g2s_val_sidx_load_op_t g2s_val_sidx_load_op{
        tid, lid, wid, 
        tile_sidx_block, 
        nnz_block,
    };
    
    using g2r_coo_atomic_format_load_off_op_t = G2rCooAtomicFormatLoadOffOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    g2r_coo_atomic_format_load_off_op_t g2r_coo_atomic_format_load_off_op{
        tid, lid, wid, 
        nnz_block,
    };
    
    // stage 1
    using s2r_val_sidx_load_op_t = S2rValSidxLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_sidx_size>;
    s2r_val_sidx_load_op_t s2r_val_sidx_load_op{
        tid, lid, wid,
    };
    
    using g2s_coo_atomic_format_load_val_op_t = G2sCooAtomicFormatLoadValOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_coo_val_size>;
    using g2s_coo_atomic_format_load_val_op_shape_o0 = typename g2s_coo_atomic_format_load_val_op_t::Shape_o0;
    __shared__ half tile_coo_val_block[cosize_v<Layout<g2s_coo_atomic_format_load_val_op_shape_o0>>];
    g2s_coo_atomic_format_load_val_op_t g2s_coo_atomic_format_load_val_op{
        tid, lid, wid, 
        tile_coo_val_block, 
        nnz,
    };
    
    using g2r_coo_atomic_format_load_idx_op_t = G2rCooAtomicFormatLoadIdxOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_coo_val_size>;
    g2r_coo_atomic_format_load_idx_op_t g2r_coo_atomic_format_load_idx_op{
        tid, lid, wid, 
        nnz,
    };
    
    // stage 2
    using g2r_b_val_load_op_t = G2rBValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    g2r_b_val_load_op_t g2r_b_val_load_op{
        tid, lid, wid, 
        K, N,
    };
    
    using s2s_coo_atomic_val_restore_op_t = S2sCooAtomicValRestoreOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_coo_val_size>;
    using s2s_coo_atomic_val_restore_op_shape_o0 = typename s2s_coo_atomic_val_restore_op_t::Shape_o0;
    __shared__ half tile_coo_restore_val_block[cosize_v<Layout<s2s_coo_atomic_val_restore_op_shape_o0>>];
    s2s_coo_atomic_val_restore_op_t s2s_coo_atomic_val_restore_op{
        tid, lid, wid, 
        tile_coo_restore_val_block,
    };
    
    using s2r_a_val_load_op_t = S2rAValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    s2r_a_val_load_op_t s2r_a_val_load_op{
        tid, lid, wid,
    };
    
    // stage 3
    using mma_op_t = MmaOp<Int<1> /* reduction */, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    mma_op_t mma_op{
        tid, lid, wid,
    };

    int l = val_soff(blockIdx.x);
    int r = val_soff(blockIdx.x + 1);
    int k = r - l;

    if (k <= 3) {
        short_pipe(k, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, mma_op, B_val, C_val, val_coo_idx, val_coo_off, val_coo_val, val_sidx, val_soff);
    } else {
        fill(l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, mma_op, B_val, C_val, val_coo_idx, val_coo_off, val_coo_val, val_sidx, val_soff);
        int i;
        for (i = 3; i + 2 <= k; i += 2) {
            loop_step(i, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, mma_op, B_val, C_val, val_coo_idx, val_coo_off, val_coo_val, val_sidx, val_soff);
        }
        remainder(i, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, mma_op, B_val, C_val, val_coo_idx, val_coo_off, val_coo_val, val_sidx, val_soff);
        empty(i, l, r, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, mma_op, B_val, C_val, val_coo_idx, val_coo_off, val_coo_val, val_sidx, val_soff);
    }

    //! write back
    auto REGC = mma_op.template output<0, 0>();
    for (int i_tileN = 0; i_tileN < get<1>(BLK_MNK{}) / 8; i_tileN++) {
        int row = blockIdx.x * 16 + lid / 4;
        int col = blockIdx.y * get<1>(BLK_MNK{}) + i_tileN * 8 + lid % 4 * 2;
        C_val(col, row) = REGC(0, i_tileN);
        C_val(col + 1, row) = REGC(1, i_tileN);
        C_val(col, row + 8) = REGC(2, i_tileN);
        C_val(col + 1, row + 8) = REGC(3, i_tileN);
    }
}

