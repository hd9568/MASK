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

#define ROW_WINDOW_M 16
#define TC_BLOCK_K 16
#define TC_BLOCK_M 16

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

__device__ __forceinline__ void ldmatrix_m8n8k8_x2(uint32_t* fragment, void* ptr) {
    uint32_t values_tile_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 \t"
        " { %0, %1 }, \t"
        " [%2];"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(values_tile_int_ptr)
    );
}

__device__ __forceinline__ void ldmatrix_m8n8k8_x2_trans(uint32_t* fragment, void* ptr) {
    uint32_t values_tile_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 \t"
        " { %0, %1 }, \t"
        " [%2];"
        : "=r"(fragment[0]), "=r"(fragment[1])
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

__device__ __forceinline__ void unpack_half_short(uint32_t packed, half& h, short & s) {
    uint16_t half_bits = static_cast<uint16_t>(packed >> 16);
    uint16_t short_bits = static_cast<uint16_t>(packed & 0xFFFF);
    h = *reinterpret_cast<half*>(&half_bits);
    s = static_cast<short>(short_bits);
}

//! stage 0
template <class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, 
    class tile_sidx_size = Int<16>, class tile_coo_val_size = Int<16 * (16)>, class tile_coo_val_size_no_pad = Int<16 * 16>,  
    class TI0 = int, class TI1 = uint32_t, class TI2 = int, class TO0 = int, class TO1 = uint32_t>
class G2sSparseEncodingLoadOp {
    public:

    //> Inputs
    // I0: dval_coo_off
    using Shape_i0 = decltype(make_shape(Int<2>{}, VARLEN));
    using Tensor_i0 = decltype(make_tensor(make_gmem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // I1: dval_coo_val_idx_bind
    using Shape_i1 = decltype(make_shape(VARLEN));
    using Tensor_i1 = decltype(make_tensor(make_gmem_ptr((TI1*)nullptr), Shape_i1{}));
    Shape_i1 shape_i1;
    // I2: i
    

    //> outputs
    // O0: coo_range_reg
    using Shape_o0 = decltype(make_shape(NBUF{}));
    using Tensor_o0 = decltype(make_tensor<TO0>(Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 coo_range_reg;
    // O1: tileA_coo_val_idx_bind_block
    using Shape_o1 = decltype(make_shape(tile_coo_val_size_no_pad{}, NBUF{}));
    using Tensor_o1 = decltype(make_tensor(make_smem_ptr((TO1*)nullptr), Shape_o1{}));
    Shape_o1 shape_o1;
    Tensor_o1 tileA_coo_val_idx_bind_block;

    //> Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2sSparseEncodingLoadOp(int tid, int lid, int wid, uint32_t* tileA_coo_val_idx_bind_block, int mode_i0_0, int mode_i1_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(Int<2>{}, mode_i0_0)), shape_i1(make_shape(mode_i1_0)), tileA_coo_val_idx_bind_block(make_tensor(make_smem_ptr(tileA_coo_val_idx_bind_block), Shape_o1{})) {}

    template <size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return coo_range_reg(buf_idx); 
        } 
        if constexpr (output_idx == 1) {
            return tileA_coo_val_idx_bind_block(_, buf_idx); 
        }
    }

    template <size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 dval_coo_off, Tensor_i1 dval_coo_val_idx_bind, TI2 i) {
        int ll = dval_coo_off(0, i);
        int rr = dval_coo_off(1, i);
        coo_range_reg(buf_idx) = rr - ll;
        
        auto thr_tiler = make_shape(_4{});
        auto input = make_tensor(dval_coo_val_idx_bind.data() + ll, tile_coo_val_size_no_pad{});
        for (int i_load = lid; i_load * 4 + ll < rr; i_load += 32) {
            auto thr_coord = make_coord(i_load);
            auto src = local_tile(input, thr_tiler, thr_coord);
            auto dst = local_tile(tileA_coo_val_idx_bind_block(_, buf_idx), thr_tiler, thr_coord);
            copy(Copy_Atom<UniversalCopy<uint128_t>, uint32_t>{}, src, dst);
        }
    }
};

template <class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, 
    class tile_sidx_size = Int<16>, class tile_coo_val_size = Int<16 * (16)>, class tile_coo_val_size_no_pad = Int<16 * 16>, 
    class TI0 = int, class TI1 = int, class TO0 = int>
class G2sSparseIndexLoadOp {
    public:

    //> Inputs
    // I0: dval_sidx
    using Shape_i0 = decltype(make_shape(tile_sidx_size{}, VARLEN));
    using Tensor_i0 = decltype(make_tensor(make_gmem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // I1: i
    //> Outputs
    // O0: tile_sidx_block
    using Shape_o0 = decltype(make_shape(tile_sidx_size{}, NBUF{}));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 tile_sidx_block;

    //> Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2sSparseIndexLoadOp(int tid, int lid, int wid, int* tile_sidx_block, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(tile_sidx_size{}, mode_i0_0)), tile_sidx_block(make_tensor(make_smem_ptr(tile_sidx_block), Shape_o0{})) {}

    template <size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_sidx_block(_, buf_idx);
        }
    }

    template <size_t buf_idx>
    CUTE_DEVICE void f (Tensor_i0 dval_sidx, TI1 i) {
        if (lid < 4) {
            auto thr_tiler = make_shape(Int<4>{});
            auto thr_coord = make_coord(lid);
            auto src = local_tile(dval_sidx(_, i), thr_tiler, thr_coord);
            auto dst = local_tile(tile_sidx_block(_, buf_idx), thr_tiler, thr_coord);
            copy(Copy_Atom<UniversalCopy<uint128_t>, int>{}, src, dst);
        }
    }
};

//! stage 1
template <class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK,
    class tile_sidx_size = Int<16>, class tile_coo_val_size = Int<16 * (16)>, class tile_coo_val_size_no_pad = Int<16 * 16>,
    class TI0 = uint32_t, class TI1 = int, class TO0 = half>
class S2sRestoreMatrixAOp {
    public:

    //> Inputs
    // I0: tileA_coo_val_idx_bind_block
    using Shape_i0 = decltype(make_shape(tile_coo_val_size_no_pad{}));
    using Tensor_i0 = decltype(make_tensor(make_smem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // I1: nnz_num
    //> Outputs
    // O0: tile_coo_restore_val_block
    static constexpr int kShmLoadSwizzleM = 3;
    static constexpr int kShmLoadSwizzleS = 3;
    static constexpr int kShmLoadSwizzleB = 1;
    using SmemLayoutAtom = decltype(composition(
        Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
        make_layout(make_shape(get<0>(BLK_MNK{}), get<2>(BLK_MNK{}), NBUF{}),
                    make_stride(get<2>(BLK_MNK{}), Int<1>{}, get<0>(BLK_MNK{}) * get<2>(BLK_MNK{})))));
    using SmemLayoutAtom_2d = decltype(composition(
        Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
        make_layout(make_shape(get<0>(BLK_MNK{}), get<2>(BLK_MNK{})),
                    make_stride(get<2>(BLK_MNK{}), Int<1>{}))));
    using SmemLayoutValBlock = SmemLayoutAtom;
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), SmemLayoutValBlock{}));
    SmemLayoutValBlock shape_o0;
    Tensor_o0 tile_coo_restore_val_block;

    //> Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE S2sRestoreMatrixAOp(int tid, int lid, int wid, half* tile_coo_restore_val_block) 
    : tid(tid), lid(lid), wid(wid), tile_coo_restore_val_block(make_tensor(make_smem_ptr(tile_coo_restore_val_block), SmemLayoutValBlock{})) {}

    template <size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return make_tensor(tile_coo_restore_val_block(_, _, buf_idx).data(), SmemLayoutAtom_2d{});
        }
    }

    template <size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 tileA_coo_val_idx_bind_block, TI1 nnz_num) {
        for (int i_o2s = lid; i_o2s < tile_coo_val_size{} / 2; i_o2s += 32) {
            *((float*)(tile_coo_restore_val_block(_, _, buf_idx).data().get() + i_o2s * 2)) = 0;
        }
        __syncthreads();
        for (int i_restore = lid; i_restore < nnz_num; i_restore += 32) {
            uint32_t val_idx_bind = tileA_coo_val_idx_bind_block(_).data()[(i_restore)];
            half value;
            short idx;
            unpack_half_short(val_idx_bind, value, idx);
            int row = idx / 16;
            int col = idx % 16;
            tile_coo_restore_val_block(row, col, buf_idx) = value;
        }
    }
};



template <class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, 
    class tile_sidx_size = Int<16>, class tile_coo_val_size = Int<16 * (16)>, class tile_coo_val_size_no_pad = Int<16 * 16>,
    class TI0 = half, class TI1 = int, class TO0 = half>
class G2sMatrixBLoadOp {
    public:
    //> Inputs
    // I0: dB_val
    using Shape_i0 = decltype(make_layout(make_shape(VARLEN, VARLEN), make_stride(VARLEN, Int<1>{})));  // row major
    using Tensor_i0 = decltype(make_tensor(make_gmem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;
    // I1: tile_sidx_block
    using Shape_i1 = decltype(make_shape(get<2>(BLK_MNK{})));
    using Tensor_i1 = decltype(make_tensor(make_smem_ptr((TI1*)nullptr), Shape_i1{}));
    Shape_i1 shape_i1;
    
    //> Outputs
    // O0: tileB_block
    static constexpr int kShmLoadSwizzleM = 3;
    static constexpr int kShmLoadSwizzleS = 3;
    static constexpr int kShmLoadSwizzleB = 3;
    using SmemLayoutAtom = decltype(composition(
        Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{}, 
        make_layout(make_shape(get<2>(BLK_MNK{}), get<1>(BLK_MNK{}), NBUF{}),
                    make_stride(get<1>(BLK_MNK{}), Int<1>{}, get<1>(BLK_MNK{}) * get<2>(BLK_MNK{})))));    // row major
    using SmemLayoutAtom_2d = decltype(composition(
        Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{}, 
        make_layout(make_shape(get<2>(BLK_MNK{}), get<1>(BLK_MNK{})),
                    make_stride(get<1>(BLK_MNK{}), Int<1>{}))));    // row major

    using SmemLayoutB = SmemLayoutAtom;
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), SmemLayoutB{}));
    
    Tensor_o0 tileB_block;

    //> Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2sMatrixBLoadOp(int tid, int lid, int wid, half* tileB_block, int mode_i0_0, int mode_i0_1)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_layout(make_shape(mode_i0_0, mode_i0_1), make_stride(mode_i0_1, _1{}))), tileB_block(make_tensor(make_smem_ptr(tileB_block), SmemLayoutB{})) {}

    template <size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return make_tensor(tileB_block(_, _, buf_idx).data(), SmemLayoutAtom_2d{});
        }
    }

    template <size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 dB_val, Tensor_i1 tile_sidx_block) {
        // TODO: 有没有支持任意TILEB的函数模板？目前定死在TILEB=64上了
        auto thr_tiler = make_shape(_8{});
        auto thr_coord = make_coord(lid);
        for (int tileK_iter_inner = 0; tileK_iter_inner < 4; tileK_iter_inner++) {
            int row = tileK_iter_inner * 4 + lid / 8;
            int col = lid % 8;
            int gmem_col = blockIdx.y * get<1>(BLK_MNK{}) / 8 + col;
            int sidx = tile_sidx_block(row);
            auto thr_tiler_gmem = make_shape(_1{}, _8{});
            auto thr_coord_gmem = make_coord(sidx, gmem_col);
            auto B_val_thr = local_tile(dB_val, thr_tiler_gmem, thr_coord_gmem);
            auto thr_tiler_smem = make_shape(_1{}, _8{});
            auto thr_coord_smem = make_coord(row, col);
            auto dst = local_tile(tileB_block(_, _, buf_idx), thr_tiler_smem, thr_coord_smem);
            copy(Copy_Atom<UniversalCopy<uint128_t>, half>{}, B_val_thr(0, _), dst(0, _));
        }
    }
};

//! stage 2

//> 可以和最后的Calculate Op合在一起吗？（哪个Op的输出定义REGA & REGB？）
template <class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK,
    class tile_sidx_size = Int<16>, class tile_coo_val_size = Int<16 * (16)>, class tile_coo_val_size_no_pad = Int<16 * 16>,
    class TI0 = half, class TO0 = half>
class S2rAValLoadOp {
    public:
    //> Inputs
    // I0: tile_coo_restore_val_block
    static constexpr int kShmLoadSwizzleM = 3;
    static constexpr int kShmLoadSwizzleS = 3;
    static constexpr int kShmLoadSwizzleB = 1;
    using SmemLayoutAtom = decltype(composition(
        Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
        make_layout(make_shape(get<0>(BLK_MNK{}), get<2>(BLK_MNK{})),
                    make_stride(get<2>(BLK_MNK{}), Int<1>{}))));
    using Tensor_i0 = decltype(make_tensor(make_smem_ptr((TI0*)nullptr), SmemLayoutAtom{}));
    

    //> Outputs
    // O0: REGA 
    using Shape_o0 = decltype(make_shape(Int<8>{}, get<0>(BLK_MMA_MNK{}), NBUF{}));
    using Tensor_o0 = decltype(make_tensor<TO0>(Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 REGA;

    //> Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE S2rAValLoadOp(int tid, int lid, int wid)
    : tid(tid), lid(lid), wid(wid) {}

    template <size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGA(_, _, buf_idx);
        }
    }

    template <size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 tile_coo_restore_val_block) {
        __syncthreads();
        int row = lid % 16;
        int col = lid / 16;
        ldmatrix_m8n8k8_x4(
            (uint32_t*)(REGA(_, _0{}, buf_idx).data()),
            (void*)(&tile_coo_restore_val_block(row, col * 8))
        );
    }
};

template <class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, 
    class tile_sidx_size = Int<16>, class tile_coo_val_size = Int<16 * (16)>, class tile_coo_val_size_no_pad = Int<16 * 16>,
    class TI0 = half, class TO0 = half>
class S2rBValLoadOp {
    public:
    //> Inputs
    // I0: tileB_block
    static constexpr int kShmLoadSwizzleM = 3;
    static constexpr int kShmLoadSwizzleS = 3;
    static constexpr int kShmLoadSwizzleB = 3;
    using SmemLayoutAtom = decltype(composition(
        Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
        make_layout(make_shape(get<2>(BLK_MNK{}), get<1>(BLK_MNK{})),
                    make_stride(get<1>(BLK_MNK{}), Int<1>{}))));
    using Tensor_i0 = decltype(make_tensor(make_smem_ptr((TI0*)nullptr), SmemLayoutAtom{}));


    //> Outputs
    // O0: REGB
    using Shape_o0 = decltype(make_shape(Int<4>{}, get<1>(BLK_MMA_MNK{}), NBUF{}));
    using Tensor_o0 = decltype(make_tensor<TO0>(Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 REGB;
    
    //> Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE S2rBValLoadOp(int tid, int lid, int wid)
    : tid(tid), lid(lid), wid(wid) {}

    template <size_t buf_idx, size_t output_idx> 
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGB(_, _, buf_idx);
        }
    }

    template <size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 tileB_block) {
        __syncthreads();
        for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
            int row_b = lid % 16;
            int col_b = n_iter;
            // Tensor tileB_tensor = local_tile(tileB_block, make_shape());
            ldmatrix_m8n8k8_x2_trans(
                (uint32_t*)(REGB(_, n_iter, buf_idx).data()),
                (void*)(&tileB_block(row_b, col_b * 8))
            );
        }
    }
};

template <class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, 
    class tile_sidx_size = Int<16>, class tile_coo_val_size = Int<16 * (16)>, class tile_coo_val_size_no_pad = Int<16 * 16>,
    class TI0 = half, class TI1 = half, class TO0 = float>
class CalculateOp {
    public:
    //> Inputs
    // I0: REGA
    using Shape_i0 = decltype(make_shape(Int<8>{}, get<0>(BLK_MMA_MNK{})));
    using Tensor_i0 = decltype(make_tensor(make_rmem_ptr((TI0*)nullptr), Shape_i0{}));
    Shape_i0 shape_i0;

    // I1: REGB
    using Shape_i1 = decltype(make_shape(Int<4>{}, get<1>(BLK_MMA_MNK{})));
    using Tensor_i1 = decltype(make_tensor(make_rmem_ptr((TI0*)nullptr), Shape_i1{}));
    Shape_i1 shape_i1;


    //> Outputs
    // O0: REGC
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{})), NBUF{}));
    using Tensor_o0 = decltype(make_tensor<TO0>(Shape_o0{}));
    Shape_o0 shape_o0;
    Tensor_o0 REGC;

    //> Hardware parameters 
    int tid, lid, wid;

    CUTE_DEVICE CalculateOp(int tid, int lid, int wid)
    : tid(tid), lid(lid), wid(wid) {}

    template <size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGC(_, _, buf_idx);
        }
    }

    template <size_t buf_idx>
    CUTE_DEVICE void f(Tensor_i0 REGA, Tensor_i1 REGB) {
        for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) {
            for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
                mma_m16n8k16_fp32(REGC(_, (m_iter, n_iter), buf_idx).data(), (uint32_t*)(REGA(_, m_iter).data()), REGB(_, n_iter).data());
            }
        }
    }
};


template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK,
          class tile_coo_val_size, class tile_sidx_size>
__global__ void dtc_spmm_kernel_fp16_val_idx_bind_stage2(
    half* dB_val, 
    float* dC_val, 
    int* dval_sidx, 
    int* dval_soff, 
    uint32_t* dval_coo_val_idx_bind, 
    int* dval_coo_off, 
    int K, int M, int Mo/* row window num */, int N, int nnz, int nnz_block
) {
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int wid = tid / 32;

    //> gmem tensors
    // auto B_val = make_tensor(make_gmem_ptr(dB_val), make_shape(N, K));  // row major
    auto B_val = make_tensor(make_gmem_ptr(dB_val), make_layout(make_shape(K, N), make_stride(N, Int<1>{})));  // row major
    auto C_val = make_tensor(make_gmem_ptr(dC_val), make_shape(N, M));  // row major
    auto val_sidx = make_tensor(make_gmem_ptr(dval_sidx), make_shape(tile_sidx_size{}, nnz_block));
    auto val_soff = make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo));
    auto val_coo_val_idx_bind = make_tensor(make_gmem_ptr(dval_coo_val_idx_bind), make_shape(nnz));
    auto val_coo_off = make_tensor(make_gmem_ptr(dval_coo_off), make_shape(Int<2>{}, nnz_block));

    //> stage 0
    using g2s_sparse_encoding_load_op_t = G2sSparseEncodingLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using g2s_sparse_encoding_load_op_shape_o1 = typename g2s_sparse_encoding_load_op_t::Shape_o1;
    __shared__ uint32_t tileA_coo_val_idx_bind_block[cosize_v<Layout<g2s_sparse_encoding_load_op_shape_o1>>];
    g2s_sparse_encoding_load_op_t g2s_sparse_encoding_load_op{
        tid, lid, wid,
        tileA_coo_val_idx_bind_block,
        nnz_block, nnz
    };
    
    using g2s_sparse_index_load_op_t = G2sSparseIndexLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using g2s_sparse_index_load_op_shape_o0 = typename g2s_sparse_index_load_op_t::Shape_o0;
    __shared__ int tileA_sidx_block[cosize_v<Layout<g2s_sparse_index_load_op_shape_o0>>];
    g2s_sparse_index_load_op_t g2s_sparse_index_load_op {
        tid, lid, wid,
        tileA_sidx_block,
        nnz_block
    };
    //> stage 1
    using s2s_restore_matrix_a_op_t = S2sRestoreMatrixAOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using s2s_restore_matrix_a_op_shape_o0 = typename s2s_restore_matrix_a_op_t::SmemLayoutValBlock;
    __shared__ half tile_coo_restore_val_block[cosize_v<s2s_restore_matrix_a_op_shape_o0>];
    s2s_restore_matrix_a_op_t s2s_restore_matrix_a_op {
        tid, lid, wid,
        tile_coo_restore_val_block
    };

    using g2s_matrix_b_load_op_t = G2sMatrixBLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using g2s_matrix_b_load_op_shape_o0 = typename g2s_matrix_b_load_op_t::SmemLayoutB;
    __shared__ half tileB_block[cosize_v<g2s_matrix_b_load_op_shape_o0>];
    g2s_matrix_b_load_op_t g2s_matrix_b_load_op {
        tid, lid, wid,
        tileB_block,
        // N, K    // row major
        K, N    // row major
    };

    //> stage 2
    using s2r_a_val_load_op_t = S2rAValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    s2r_a_val_load_op_t s2r_a_val_load_op {
        tid, lid, wid
    };

    using s2r_b_val_load_op_t = S2rBValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    s2r_b_val_load_op_t s2r_b_val_load_op {
        tid, lid, wid
    };

    using calculate_op_t = CalculateOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    calculate_op_t calculate_op {
        tid, lid, wid
    };
    // TODO: change the swizzle col major to row major

    int l = val_soff(blockIdx.x);
    int r = val_soff(blockIdx.x + 1);
    int k = r - l;

    int i = 0;

    // TODO: change the invoke function below
    //>========================================================
    //! A: memcpy async sparse encoding + value load (0)
    g2s_sparse_encoding_load_op.template f<0>(val_coo_off, val_coo_val_idx_bind, l + 0/* i */);
    // op_memcpy_async_sparse_encoding_value_load<TILE_B>(0, 0 % stages, dval_coo_off, dval_coo_val_idx_bind, l, r, coo_range_reg, lane_id, tileA_coo_val_idx_bind_block, tile_coo_val_size, tile_coo_restore_val_block);
    //! B: memcpy async sparse index load (0)
    g2s_sparse_index_load_op.template f<0>(val_sidx, l + 0/* i */);
    // op_memcpy_async_sparse_index_load<TILE_B>(0, 0 % stages, lane_id, tile_sidx_block, tile_sidx_size, dval_sidx, l);
    __pipeline_commit();
    // return;
    //>========================================================
    //! A: memcpy async sparse encoding + value load (1)
    g2s_sparse_encoding_load_op.template f<1>(val_coo_off, val_coo_val_idx_bind, l + 1/* i */);
    // op_memcpy_async_sparse_encoding_value_load<TILE_B>(1, 1 % stages, dval_coo_off, dval_coo_val_idx_bind, l, r, coo_range_reg, lane_id, tileA_coo_val_idx_bind_block, tile_coo_val_size, tile_coo_restore_val_block);
    //! B: memcpy async sparse index load (1)
    g2s_sparse_index_load_op.template f<1>(val_sidx, l + 1/* i */);
    // op_memcpy_async_sparse_index_load<TILE_B>(1, 1 % stages, lane_id, tile_sidx_block, tile_sidx_size, dval_sidx, l);
    __pipeline_wait_prior(0);
    __syncthreads();
    //! A: memcpy async restore matrix A (0)
    s2s_restore_matrix_a_op.template f<0>(g2s_sparse_encoding_load_op.template output<0, 1>(), g2s_sparse_encoding_load_op.template output<0, 0>());
    // op_restore_matrix_A<TILE_B>(0, 0 % stages, 0 % stages, lane_id, coo_range_reg, tile_coo_restore_val_block, tile_coo_val_size, tileA_coo_val_idx_bind_block);
    
    //! B: memcpy async dense matrix B load (0)
    g2s_matrix_b_load_op.template f<0>(B_val, g2s_sparse_index_load_op.template output<0, 0>());
    // op_memcpy_async_matrix_B_load<TILE_B>(0, 0 % stages, 0 % stages, lane_id, tile_sidx_block, tile_sidx_size, tileB_block, tile_B_size, tile_B_row_num, dB_val, N, block_idx_y);
    __pipeline_commit();
    // return;
    //>========================================================
    for (i = 0; i + 2 < k; i++) {  
        if ((i + 2) % 2 == 0) {
            g2s_sparse_encoding_load_op.template f<0>(val_coo_off, val_coo_val_idx_bind, l + i + 2);
            g2s_sparse_index_load_op.template f<0>(val_sidx, l + i + 2);
        } else {
            g2s_sparse_encoding_load_op.template f<1>(val_coo_off, val_coo_val_idx_bind, l + i + 2);
            g2s_sparse_index_load_op.template f<1>(val_sidx, l + i + 2);
        }
        //! A: memcpy async sparse encoding + value load (i + 2)
        // op_memcpy_async_sparse_encoding_value_load<TILE_B>(i + 2, (i + 2) % stages, dval_coo_off, dval_coo_val_idx_bind, l, r, coo_range_reg, lane_id, tileA_coo_val_idx_bind_block, tile_coo_val_size, tile_coo_restore_val_block);
        //! B: memcpy async sparse index load (i + 2)
        // op_memcpy_async_sparse_index_load<TILE_B>(i + 2, (i + 2) % stages, lane_id, tile_sidx_block, tile_sidx_size, dval_sidx, l);
        __pipeline_wait_prior(0);
        __syncthreads();
        if ((i + 1) % 2 == 0) {
            s2s_restore_matrix_a_op.template f<0>(g2s_sparse_encoding_load_op.template output<0, 1>(), g2s_sparse_encoding_load_op.template output<0, 0>());
            g2s_matrix_b_load_op.template f<0>(B_val, g2s_sparse_index_load_op.template output<0, 0>());
        } else {
            s2s_restore_matrix_a_op.template f<1>(g2s_sparse_encoding_load_op.template output<1, 1>(), g2s_sparse_encoding_load_op.template output<1, 0>());
            g2s_matrix_b_load_op.template f<1>(B_val, g2s_sparse_index_load_op.template output<1, 0>());
        }
        //! A: memcpy async restore matrix A (i + 1)
        // op_restore_matrix_A<TILE_B>(i + 1, (i + 1) % stages, (i + 1) % stages, lane_id, coo_range_reg, tile_coo_restore_val_block, tile_coo_val_size, tileA_coo_val_idx_bind_block);
        //! B: memcpy async dense matrix B load (i + 1)
        // op_memcpy_async_matrix_B_load<TILE_B>(i + 1, (i + 1) % stages, (i + 1) % stages, lane_id, tile_sidx_block, tile_sidx_size, tileB_block, tile_B_size, tile_B_row_num, dB_val, N, block_idx_y);
        __pipeline_commit();
        //! ldmatrix A, ldmatrix B, and MMA (i)
        if (i % 2 == 0) {
            s2r_a_val_load_op.template f<0>(s2s_restore_matrix_a_op.template output<0, 0>());
            s2r_b_val_load_op.template f<0>(g2s_matrix_b_load_op.template output<0, 0>());
            calculate_op.template f<0>(s2r_a_val_load_op.template output<0, 0>(), s2r_b_val_load_op.template output<0, 0>());
        } else {
            s2r_a_val_load_op.template f<1>(s2s_restore_matrix_a_op.template output<1, 0>());
            s2r_b_val_load_op.template f<1>(g2s_matrix_b_load_op.template output<1, 0>());
            calculate_op.template f<0>(s2r_a_val_load_op.template output<1, 0>(), s2r_b_val_load_op.template output<1, 0>());
        }
        // op_calculate<TILE_B>(i, i % stages, i % stages, lane_id, tile_coo_restore_val_block, tile_coo_val_size, tileB_block, tile_B_size, tile_B_row_num, REGA, REGB, REGC);
    }
    for (; i + 1 < k; i++) {
        __pipeline_wait_prior(0);
        __syncthreads();
        if ((i + 1) % 2 == 0) {
            s2s_restore_matrix_a_op.template f<0>(g2s_sparse_encoding_load_op.template output<0, 1>(), g2s_sparse_encoding_load_op.template output<0, 0>());
            g2s_matrix_b_load_op.template f<0>(B_val, g2s_sparse_index_load_op.template output<0, 0>());
        } else {
            s2s_restore_matrix_a_op.template f<1>(g2s_sparse_encoding_load_op.template output<1, 1>(), g2s_sparse_encoding_load_op.template output<1, 0>());
            g2s_matrix_b_load_op.template f<1>(B_val, g2s_sparse_index_load_op.template output<1, 0>());
        }
        //! A: memcpy async restore matrix A (i + 1)
        // op_restore_matrix_A<TILE_B>(i + 1, (i + 1) % stages, (i + 1) % stages, lane_id, coo_range_reg, tile_coo_restore_val_block, tile_coo_val_size, tileA_coo_val_idx_bind_block);
        //! B: memcpy async dense matrix B load (i + 1)
        // op_memcpy_async_matrix_B_load<TILE_B>(i + 1, (i + 1) % stages, (i + 1) % stages, lane_id, tile_sidx_block, tile_sidx_size, tileB_block, tile_B_size, tile_B_row_num, dB_val, N, block_idx_y);
        __pipeline_commit();
        //! ldmatrix A, ldmatrix B, and MMA (i)
        if (i % 2 == 0) {
            s2r_a_val_load_op.template f<0>(s2s_restore_matrix_a_op.template output<0, 0>());
            s2r_b_val_load_op.template f<0>(g2s_matrix_b_load_op.template output<0, 0>());
            calculate_op.template f<0>(s2r_a_val_load_op.template output<0, 0>(), s2r_b_val_load_op.template output<0, 0>());
        } else {
            s2r_a_val_load_op.template f<1>(s2s_restore_matrix_a_op.template output<1, 0>());
            s2r_b_val_load_op.template f<1>(g2s_matrix_b_load_op.template output<1, 0>());
            calculate_op.template f<0>(s2r_a_val_load_op.template output<1, 0>(), s2r_b_val_load_op.template output<1, 0>());
        }
        // op_calculate<TILE_B>(i, i % stages, i % stages, lane_id, tile_coo_restore_val_block, tile_coo_val_size, tileB_block, tile_B_size, tile_B_row_num, REGA, REGB, REGC);
    }
    for (; i < k; i++) {
        __pipeline_wait_prior(0);
        __syncthreads();
        //! ldmatrix A, ldmatrix B, and MMA (i)
        if (i % 2 == 0) {
            s2r_a_val_load_op.template f<0>(s2s_restore_matrix_a_op.template output<0, 0>());
            s2r_b_val_load_op.template f<0>(g2s_matrix_b_load_op.template output<0, 0>());
            calculate_op.template f<0>(s2r_a_val_load_op.template output<0, 0>(), s2r_b_val_load_op.template output<0, 0>());
        } else {
            s2r_a_val_load_op.template f<1>(s2s_restore_matrix_a_op.template output<1, 0>());
            s2r_b_val_load_op.template f<1>(g2s_matrix_b_load_op.template output<1, 0>());
            calculate_op.template f<0>(s2r_a_val_load_op.template output<1, 0>(), s2r_b_val_load_op.template output<1, 0>());
        }
        // op_calculate<TILE_B>(i, i % stages, i % stages, lane_id, tile_coo_restore_val_block, tile_coo_val_size, tileB_block, tile_B_size, tile_B_row_num, REGA, REGB, REGC);
    }
    //>========================================================
    //! store back
    auto REGC = calculate_op.template output<0, 0>();
    for (int i_tileN = 0; i_tileN < get<1>(BLK_MNK{}) / 8; i_tileN++) {
        int row = blockIdx.x * 16 + lid / 4;
        int col = blockIdx.y * get<1>(BLK_MNK{}) + i_tileN * 8 + lid % 4 * 2;
        C_val(col, row) = REGC(0, i_tileN);
        C_val(col + 1, row) = REGC(1, i_tileN);
        C_val(col, row + 8) = REGC(2, i_tileN);
        C_val(col + 1, row + 8) = REGC(3, i_tileN);
    }
}