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

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))
#define OFFSET_COL(row, col, lda) ((col) * (lda) + (row))

#define DIVUP(a, b) (((a) - 1) / (b) + 1)
#define ALIGN(a,b) (DIVUP(a,b)*(b))

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

__device__ __forceinline__ void mma_m16n8k16_fp32(float* acc, uint32_t* A, half* half_B) {
    uint32_t* B = reinterpret_cast<uint32_t*>(half_B);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1])
    );
}

__device__ __forceinline__ void mma_m16n8k8_fp32_tf32_tf32_fp32(float* acc, uint32_t* frag_A, uint32_t* frag_B) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[1])
    );
}

__device__ __forceinline__ void unpack_half_short(uint32_t packed, half& h, short & s) {
    uint16_t half_bits = static_cast<uint16_t>(packed >> 16);
    uint16_t short_bits = static_cast<uint16_t>(packed & 0xFFFF);
    h = *reinterpret_cast<half*>(&half_bits);
    s = static_cast<short>(short_bits);
}


template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = float>
class B_valOp {
public:
    // Outputs
    using Tensor_o0 = decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_layout(make_shape(VARLEN, VARLEN), make_stride(VARLEN, _1{}))), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 B_val;

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE B_valOp(int tid, int lid, int wid, int K, int N, TO0 *dB_val)
    : tid(tid), lid(lid), wid(wid),
    shape_o0(logical_divide(make_tensor(make_gmem_ptr(dB_val), make_layout(make_shape(K, N), make_stride(N, _1{}))), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y)).shape()),
    layout_o0(logical_divide(make_tensor(make_gmem_ptr(dB_val), make_layout(make_shape(K, N), make_stride(N, _1{}))), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y)).layout()),
    B_val(logical_divide(make_tensor(make_gmem_ptr(dB_val), make_layout(make_shape(K, N), make_stride(N, _1{}))), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return B_val;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = float>
class C_valOp {
public:
    // Outputs
    using Tensor_o0 = decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_layout(make_shape(VARLEN, VARLEN), make_stride(VARLEN, _1{}))), select<0, 1>(BLK_MNK{}))(make_coord(_, VARLEN), make_coord(_, VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 C_val;

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE C_valOp(int tid, int lid, int wid, int M, int N, TO0 *dC_val)
    : tid(tid), lid(lid), wid(wid),
    shape_o0(logical_divide(make_tensor(make_gmem_ptr(dC_val), make_layout(make_shape(M, N), make_stride(N, _1{}))), select<0, 1>(BLK_MNK{}))(make_coord(_, blockIdx.x), make_coord(_, blockIdx.y)).shape()),
    layout_o0(logical_divide(make_tensor(make_gmem_ptr(dC_val), make_layout(make_shape(M, N), make_stride(N, _1{}))), select<0, 1>(BLK_MNK{}))(make_coord(_, blockIdx.x), make_coord(_, blockIdx.y)).layout()),
    C_val(logical_divide(make_tensor(make_gmem_ptr(dC_val), make_layout(make_shape(M, N), make_stride(N, _1{}))), select<0, 1>(BLK_MNK{}))(make_coord(_, blockIdx.x), make_coord(_, blockIdx.y))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return C_val;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class nmask_per_tile = Int<2>, class TO0 = uint64_t>
class val_mco_maskOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(nmask_per_tile{}, VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_mco_mask;

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE val_mco_maskOp(int tid, int lid, int wid, int nnz_block, TO0 *dval_mco_mask)
    : tid(tid), lid(lid), wid(wid),
    shape_o0(make_tensor(make_gmem_ptr(dval_mco_mask), make_shape(nmask_per_tile{}, nnz_block)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_mco_mask), make_shape(nmask_per_tile{}, nnz_block)).layout()),
    val_mco_mask(make_tensor(make_gmem_ptr(dval_mco_mask), make_shape(nmask_per_tile{}, nnz_block))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_mco_mask;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class nmask_per_tile = Int<2>, class TO0 = int>
class val_mco_offOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_mco_off;

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE val_mco_offOp(int tid, int lid, int wid, int nnz_block, TO0 *dval_mco_off)
    : tid(tid), lid(lid), wid(wid),
    shape_o0(make_tensor(make_gmem_ptr(dval_mco_off), make_shape(nnz_block)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_mco_off), make_shape(nnz_block)).layout()),
    val_mco_off(make_tensor(make_gmem_ptr(dval_mco_off), make_shape(nnz_block))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_mco_off;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = float>
class val_mco_valOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_mco_val;

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE val_mco_valOp(int tid, int lid, int wid, int nnz, TO0 *dval_mco_val)
    : tid(tid), lid(lid), wid(wid),
    shape_o0(make_tensor(make_gmem_ptr(dval_mco_val), make_shape(nnz)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_mco_val), make_shape(nnz)).layout()),
    val_mco_val(make_tensor(make_gmem_ptr(dval_mco_val), make_shape(nnz))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_mco_val;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class val_sidxOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_sidx;

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE val_sidxOp(int tid, int lid, int wid, int nnz_block, TO0 *dval_sidx)
    : tid(tid), lid(lid), wid(wid),
    shape_o0(make_tensor(make_gmem_ptr(dval_sidx), make_shape(nnz_block)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_sidx), make_shape(nnz_block)).layout()),
    val_sidx(make_tensor(make_gmem_ptr(dval_sidx), make_shape(nnz_block))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_sidx;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class val_soffOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_soff;

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE val_soffOp(int tid, int lid, int wid, int Mo, TO0 *dval_soff)
    : tid(tid), lid(lid), wid(wid),
    shape_o0(make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo)).layout()),
    val_soff(make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_soff;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class ZeroOp {
public:
    // No Inputs

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Stride_o0 = decltype(make_stride(_0{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 ZeroOp_out;
    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE ZeroOp(int tid, int lid, int wid, TO0 *ZeroOp_out)
    : tid(tid), lid(lid), wid(wid), ZeroOp_out(make_tensor(make_rmem_ptr(ZeroOp_out), Layout_o0{})) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return ZeroOp_out;
        }
    }
    
    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        clear(ZeroOp_out);
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class BlockDimYOp {
public:
    // No Inputs

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Stride_o0 = decltype(make_stride(_0{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 BlockDimYOp_out;
    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE BlockDimYOp(int tid, int lid, int wid, TO0 *BlockDimYOp_out)
    : tid(tid), lid(lid), wid(wid), BlockDimYOp_out(make_tensor(make_rmem_ptr(BlockDimYOp_out), Layout_o0{})) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return BlockDimYOp_out;
        }
    }
    
    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        for (int i = 0; i < BlockDimYOp_out.size(); i++) { BlockDimYOp_out(i) = blockDim.y; }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class BlockDimXOp {
public:
    // No Inputs

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Stride_o0 = decltype(make_stride(_0{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 BlockDimXOp_out;
    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE BlockDimXOp(int tid, int lid, int wid, TO0 *BlockDimXOp_out)
    : tid(tid), lid(lid), wid(wid), BlockDimXOp_out(make_tensor(make_rmem_ptr(BlockDimXOp_out), Layout_o0{})) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return BlockDimXOp_out;
        }
    }
    
    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        for (int i = 0; i < BlockDimXOp_out.size(); i++) { BlockDimXOp_out(i) = blockDim.x; }
    }
};

template<class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = int, class TI1 = int, class TO0 = int, class TO1 = int>
class G2rSparseOffsetLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 val_soff;
    // TI1 i;

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Stride_o0 = decltype(make_stride(_0{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 l;
    using Shape_o1 = decltype(make_shape(_1{}));
    using Stride_o1 = decltype(make_stride(_0{}));
    using Layout_o1 = decltype(make_layout(Shape_o1{}, Stride_o1{}));
    using Tensor_o1 = decltype(make_tensor(make_rmem_ptr((TO1*)nullptr), Layout_o1{}));
    Shape_o1 shape_o1;
    Layout_o1 layout_o1;
    Tensor_o1 r;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rSparseOffsetLoadOp(int tid, int lid, int wid, TO0 *l, TO1 *r, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)), l(make_tensor(make_rmem_ptr(l), Layout_o0{})), r(make_tensor(make_rmem_ptr(r), Layout_o1{})) {}

    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return l;
        } else if constexpr (output_idx == 1) {
            return r;
        }
    }

    template<class Tensor_i0, typename = void>
    CUTE_DEVICE void f(Tensor_i0 val_soff, TI1 i) {
        l(0) = val_soff(i);
        r(0) = val_soff(i + 1);
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = float>
class ZerosOp {
public:
    // No Inputs

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{}))));
    using Stride_o0 = decltype(make_stride(_1{}, make_stride(Int<4>{}, Int<4>{} * get<0>(BLK_MMA_MNK{}))));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 ZerosOp_out;
    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE ZerosOp(int tid, int lid, int wid, TO0 *ZerosOp_out)
    : tid(tid), lid(lid), wid(wid), ZerosOp_out(make_tensor(make_rmem_ptr(ZerosOp_out), Layout_o0{})) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return ZerosOp_out;
        }
    }
    
    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        clear(ZerosOp_out);
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = int, class TI1 = int, class TO0 = int>
class G2rSparseIndexLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 val_sidx;
    // TI1 i;

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}, NBUF{}));
    using Stride_o0 = decltype(make_stride(_0{}, _1{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 tile_sidx;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rSparseIndexLoadOp(int tid, int lid, int wid, TO0 *tile_sidx, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)), tile_sidx(make_tensor(make_rmem_ptr(tile_sidx), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_sidx(_, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_sidx;
        }
    }

    template<int buf_idx, class Tensor_i0, typename = void>
    CUTE_DEVICE void f(Tensor_i0 val_sidx, TI1 i) {
        tile_sidx(0, buf_idx) = val_sidx(i);
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class nmask_per_tile = Int<2>, class TI0 = int, class TI1 = int, class TO0 = int>
class G2rSparseMcoOffLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 val_mco_off;
    // TI1 i;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<2>{}, NBUF{}));
    using Stride_o0 = decltype(make_stride(_1{}, Int<2>{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 mco_off_range;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rSparseMcoOffLoadOp(int tid, int lid, int wid, TO0 *mco_off_range, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)), mco_off_range(make_tensor(make_rmem_ptr(mco_off_range), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return mco_off_range(_, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return mco_off_range;
        }
    }

    template<int buf_idx, class Tensor_i0, typename = void>
    CUTE_DEVICE void f(Tensor_i0 val_mco_off, TI1 i) {
        mco_off_range(0, buf_idx) = val_mco_off(i);
        mco_off_range(1, buf_idx) = val_mco_off(i + 1);
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class nmask_per_tile = Int<2>, class TI0 = uint64_t, class TI1 = int, class TO0 = uint64_t>
class G2rSparseMcoMaskLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(nmask_per_tile{}, VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 val_mco_mask;
    // TI1 i;

    // Outputs
    using Shape_o0 = decltype(make_shape(nmask_per_tile{}, NBUF{}));
    using Stride_o0 = decltype(make_stride(_1{}, nmask_per_tile{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 tile_mco_mask;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rSparseMcoMaskLoadOp(int tid, int lid, int wid, TO0 *tile_mco_mask, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(nmask_per_tile{}, mode_i0_0)), tile_mco_mask(make_tensor(make_rmem_ptr(tile_mco_mask), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_mco_mask(_, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_mco_mask;
        }
    }

    template<int buf_idx, class Tensor_i0, typename = void>
    CUTE_DEVICE void f(Tensor_i0 val_mco_mask, TI1 i) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(val_mco_mask))));
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(val_mco_mask))));
        auto tiler = make_shape(Int<sizeof(uint128_t) / sizeof(uint64_t)>{});
        auto src = flat_divide(val_mco_mask(_, i), tiler);
        auto dst = flat_divide(tile_mco_mask(_, buf_idx), tiler);
        copy(Copy_Atom<UniversalCopy<uint128_t>, uint64_t>{}, src, dst);
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class tile_mco_val_size = Int<128>, class TI0 = float, class TI1 = int, class TI2 = int, class TO0 = float>
class G2sSparseMcoValLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 val_mco_val;
    using Shape_i1 = decltype(make_shape(Int<2>{}));
    Shape_i1 shape_i1;
    // Tensor_i1 mco_off_range;
    // TI2 i;

    // Outputs
    using Shape_o0 = decltype(make_shape(tile_mco_val_size{}, NBUF{}));
    using Stride_o0 = decltype(make_stride(_1{}, tile_mco_val_size{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 tile_mco_val;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2sSparseMcoValLoadOp(int tid, int lid, int wid, TO0 *tile_mco_val, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)), tile_mco_val(make_tensor(make_smem_ptr(tile_mco_val), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_mco_val(_, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_mco_val;
        }
    }

    template<int buf_idx, class Tensor_i0, class Tensor_i1, typename = void>
    CUTE_DEVICE void f(Tensor_i0 val_mco_val, Tensor_i1 mco_off_range, TI2 i) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i1) == size<0>(shape(mco_off_range))));
        int ll = mco_off_range(0);
        int rr = mco_off_range(1);
        
        auto thr_tiler = make_shape(_4{});
        auto input = make_tensor(val_mco_val.data() + ll, tile_mco_val_size{});
        for (int i_load = lid; i_load * 4 + ll < rr; i_load += 32) {
            auto thr_coord = make_coord(i_load);
            auto src = local_tile(input, thr_tiler, thr_coord);
            auto dst = local_tile(tile_mco_val(_, buf_idx), thr_tiler, thr_coord);
            copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{}, src, dst);
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = float, class TI1 = int, class TO0 = float>
class G2sMatrixBLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN, VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 B_val;
    // TI1 tile_sidx;

    // Outputs
    using Shape_o0 = decltype(make_shape(get<2>(BLK_MNK{}), get<1>(BLK_MNK{}), NBUF{}));
    using Stride_o0 = decltype(make_stride(get<1>(BLK_MNK{}), _1{}, get<2>(BLK_MNK{}) * get<1>(BLK_MNK{})));
    using Layout_o0 = decltype(composition(Swizzle<1, 2, 3>{}, make_layout(Shape_o0{}, Stride_o0{})));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 tileB_block;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2sMatrixBLoadOp(int tid, int lid, int wid, TO0 *tileB_block, int mode_i0_0, int mode_i0_1)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0, mode_i0_1)), tileB_block(make_tensor(make_smem_ptr(tileB_block), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tileB_block(_, _, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tileB_block;
        }
    }

    template<int buf_idx, class Tensor_i0, typename = void>
    CUTE_DEVICE void f(Tensor_i0 B_val, TI1 tile_sidx) {
        auto load_tile_n = _4{};
        for (int iter_i = lid; iter_i < get<2>(BLK_MNK{}) * get<1>(BLK_MNK{}) / load_tile_n; iter_i += 32) {
            int row = iter_i / (get<1>(BLK_MNK{}) / load_tile_n);
            int col = iter_i % (get<1>(BLK_MNK{}) / load_tile_n);
            int gmem_row = tile_sidx * get<2>(BLK_MNK{}) + row;
            auto thr_tiler_gmem = make_shape(_1{}, _4{});
            auto thr_coord_gmem = make_coord(gmem_row, col);
            auto B_val_thr = local_tile(B_val, thr_tiler_gmem, thr_coord_gmem);
            auto thr_tiler_smem = make_shape(_1{}, _4{});
            auto thr_coord_smem = make_coord(row, col);
            auto dst = local_tile(tileB_block(_, _, buf_idx), thr_tiler_smem, thr_coord_smem);
            copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{}, B_val_thr(0, _), dst(0, _));   
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class tile_mco_val_size = Int<128>, class nmask_per_tile = Int<2>, class TI0 = float, class TI1 = uint64_t, class TO0 = float>
class S2sRestoreMatrixAOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(tile_mco_val_size{}));
    Shape_i0 shape_i0;
    // Tensor_i0 tile_mco_val;
    using Shape_i1 = decltype(make_shape(nmask_per_tile{}));
    Shape_i1 shape_i1;
    // Tensor_i1 tile_mco_mask;

    // Outputs
    using Shape_o0 = decltype(make_shape(get<0>(BLK_MNK{}), get<2>(BLK_MNK{}), NBUF{}));
    using Stride_o0 = decltype(make_stride(get<2>(BLK_MNK{}), _1{}, get<0>(BLK_MNK{}) * get<2>(BLK_MNK{})));
    using Layout_o0 = decltype(composition(Swizzle<1, 2, 3>{}, make_layout(Shape_o0{}, Stride_o0{})));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 tile_mco_restore_val_block;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE S2sRestoreMatrixAOp(int tid, int lid, int wid, TO0 *tile_mco_restore_val_block)
    : tid(tid), lid(lid), wid(wid), tile_mco_restore_val_block(make_tensor(make_smem_ptr(tile_mco_restore_val_block), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_mco_restore_val_block(_, _, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_mco_restore_val_block;
        }
    }

    template<int buf_idx, class Tensor_i0, class Tensor_i1, typename = void>
    CUTE_DEVICE void f(Tensor_i0 tile_mco_val, Tensor_i1 tile_mco_mask) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(tile_mco_val))));
        CUTE_STATIC_ASSERT_V((get<0>(shape_i1) == size<0>(shape(tile_mco_mask))));
        int off = 0;
        for (int i_mask = 0; i_mask < nmask_per_tile{}; i_mask++) {
            uint64_t mask = tile_mco_mask(i_mask);
            for (int local_vid = tid; local_vid < 64; local_vid += 32 /* num threads */) {
                int local_idx = 0;
                if (mask & (1ULL << local_vid)) {
                    local_idx = __popcll(mask << (63 - local_vid));
                }
                int vid = i_mask * 64 + local_vid;
                if (local_idx == 0) {
                    *((float*)tile_mco_restore_val_block(_, _, buf_idx).data().get() + vid) = 0;
                } else {
                    *((float*)tile_mco_restore_val_block(_, _, buf_idx).data().get() + vid) = tile_mco_val(off + local_idx - 1);
                }
            }
            off += __popcll(mask);
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = float, class TO0 = float>
class S2rAValLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(get<0>(BLK_MNK{}), get<2>(BLK_MNK{})));
    Shape_i0 shape_i0;
    // Tensor_i0 tile_mco_restore_val_block;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, get<0>(BLK_MMA_MNK{}), NBUF{}));
    using Stride_o0 = decltype(make_stride(_1{}, Int<4>{}, Int<4>{} * get<0>(BLK_MMA_MNK{})));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 REGA;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE S2rAValLoadOp(int tid, int lid, int wid, TO0 *REGA)
    : tid(tid), lid(lid), wid(wid), REGA(make_tensor(make_rmem_ptr(REGA), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGA(_, _, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGA;
        }
    }

    template<int buf_idx, class Tensor_i0, typename = void>
    CUTE_DEVICE void f(Tensor_i0 tile_mco_restore_val_block) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(tile_mco_restore_val_block))));
        CUTE_STATIC_ASSERT_V((get<1>(shape_i0) == size<1>(shape(tile_mco_restore_val_block))));
        int row = lid % 16;
        int col = lid / 16;
        ldmatrix_m8n8k8_x4(
            (uint32_t*)(REGA(_, _0{}, buf_idx).data()),
            (void*)(&tile_mco_restore_val_block(row, col * 4))
        );
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = float, class TO0 = float>
class S2rBValLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(get<2>(BLK_MNK{}), get<1>(BLK_MNK{})));
    Shape_i0 shape_i0;
    // Tensor_i0 tileB_block;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<2>{}, get<1>(BLK_MMA_MNK{}), NBUF{}));
    using Stride_o0 = decltype(make_stride(_1{}, Int<2>{}, Int<2>{} * get<1>(BLK_MMA_MNK{})));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 REGB;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE S2rBValLoadOp(int tid, int lid, int wid, TO0 *REGB)
    : tid(tid), lid(lid), wid(wid), REGB(make_tensor(make_rmem_ptr(REGB), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGB(_, _, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGB;
        }
    }

    template<int buf_idx, class Tensor_i0, typename = void>
    CUTE_DEVICE void f(Tensor_i0 tileB_block) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(tileB_block))));
        CUTE_STATIC_ASSERT_V((get<1>(shape_i0) == size<1>(shape(tileB_block))));
        for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
            int row_b = lid / 2;
            int col_b = lid % 2;
            ldmatrix_m8n8k8_x2(
                (uint32_t*)(REGB(_, n_iter, buf_idx).data()),
                (void*)(&tileB_block(row_b, col_b * 4 + n_iter * 8))
            );
            REGB(_0{}, n_iter, buf_idx) = __shfl_sync(0xffffffff, REGB(_0{}, n_iter, buf_idx), lid / 4 + lid % 4 * 8);
            REGB(_1{}, n_iter, buf_idx) = __shfl_sync(0xffffffff, REGB(_1{}, n_iter, buf_idx), lid / 4 + lid % 4 * 8);
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class Tensor_o0, class TI0 = float, class TI1 = float, class TO0 = float>
class CalculateOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(Int<4>{}, get<0>(BLK_MMA_MNK{})));
    Shape_i0 shape_i0;
    // Tensor_i0 REGA;
    using Shape_i1 = decltype(make_shape(Int<2>{}, get<1>(BLK_MMA_MNK{})));
    Shape_i1 shape_i1;
    // Tensor_i1 REGB;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{}))));
    Shape_o0 shape_o0;
    Tensor_o0 REGC;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE CalculateOp(int tid, int lid, int wid, Tensor_o0 REGC)
    : tid(tid), lid(lid), wid(wid), REGC(REGC) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGC;
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGC;
        }
    }

    template<int buf_idx, class Tensor_i0, class Tensor_i1, typename = void>
    CUTE_DEVICE void f(Tensor_i0 REGA, Tensor_i1 REGB) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(REGA))));
        CUTE_STATIC_ASSERT_V((get<1>(shape_i0) == size<1>(shape(REGA))));
        CUTE_STATIC_ASSERT_V((get<0>(shape_i1) == size<0>(shape(REGB))));
        CUTE_STATIC_ASSERT_V((get<1>(shape_i1) == size<1>(shape(REGB))));
        for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) {
            for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
                uint32_t frag_A[4];
                uint32_t frag_B[2];
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(REGA(0, m_iter)));
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(REGA(1, m_iter)));
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(REGA(2, m_iter)));
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(REGA(3, m_iter)));
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(REGB(0, n_iter)));
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(REGB(1, n_iter)));
        
                mma_m16n8k8_fp32_tf32_tf32_fp32(REGC(_, make_coord(m_iter, n_iter)).data(), frag_A, frag_B);
            }
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class T_B_valOp, class T_C_valOp, class T_val_mco_maskOp, class T_val_mco_offOp, class T_val_mco_valOp, class T_val_sidxOp, class T_val_soffOp, class T_ZeroOp, class T_BlockDimYOp, class T_BlockDimXOp, class T_G2rSparseOffsetLoadOp, class T_ZerosOp, class T_G2rSparseIndexLoadOp, class T_G2rSparseMcoOffLoadOp, class T_G2rSparseMcoMaskLoadOp, class T_G2sSparseMcoValLoadOp, class T_G2sMatrixBLoadOp, class T_S2sRestoreMatrixAOp, class T_S2rAValLoadOp, class T_S2rBValLoadOp, class T_CalculateOp, class Tensor_o0, class TI0 = int, class TI1 = int, class TO0 = float>
class BitBsrMainLoopOp {
public:
    // Inputs
    int _l, _r;
    int _idx; // i + _l
    int _iter; // i

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{}))));
    Shape_o0 shape_o0;
    Tensor_o0 REGC;

    // Hardware parameters
    int tid, lid, wid;

    // Body Ops
    T_B_valOp &B_valOp_v;
    T_C_valOp &C_valOp_v;
    T_val_mco_maskOp &val_mco_maskOp_v;
    T_val_mco_offOp &val_mco_offOp_v;
    T_val_mco_valOp &val_mco_valOp_v;
    T_val_sidxOp &val_sidxOp_v;
    T_val_soffOp &val_soffOp_v;
    T_ZeroOp &ZeroOp_v;
    T_BlockDimYOp &BlockDimYOp_v;
    T_BlockDimXOp &BlockDimXOp_v;
    T_G2rSparseOffsetLoadOp &G2rSparseOffsetLoadOp_v;
    T_ZerosOp &ZerosOp_v;
    T_G2rSparseIndexLoadOp &G2rSparseIndexLoadOp_v;
    T_G2rSparseMcoOffLoadOp &G2rSparseMcoOffLoadOp_v;
    T_G2rSparseMcoMaskLoadOp &G2rSparseMcoMaskLoadOp_v;
    T_G2sSparseMcoValLoadOp &G2sSparseMcoValLoadOp_v;
    T_G2sMatrixBLoadOp &G2sMatrixBLoadOp_v;
    T_S2sRestoreMatrixAOp &S2sRestoreMatrixAOp_v;
    T_S2rAValLoadOp &S2rAValLoadOp_v;
    T_S2rBValLoadOp &S2rBValLoadOp_v;
    T_CalculateOp &CalculateOp_v;

    // Constructor
    CUTE_DEVICE BitBsrMainLoopOp(int tid, int lid, int wid, T_B_valOp &B_valOp_v, T_C_valOp &C_valOp_v, T_val_mco_maskOp &val_mco_maskOp_v, T_val_mco_offOp &val_mco_offOp_v, T_val_mco_valOp &val_mco_valOp_v, T_val_sidxOp &val_sidxOp_v, T_val_soffOp &val_soffOp_v, T_ZeroOp &ZeroOp_v, T_BlockDimYOp &BlockDimYOp_v, T_BlockDimXOp &BlockDimXOp_v, T_G2rSparseOffsetLoadOp &G2rSparseOffsetLoadOp_v, T_ZerosOp &ZerosOp_v, T_G2rSparseIndexLoadOp &G2rSparseIndexLoadOp_v, T_G2rSparseMcoOffLoadOp &G2rSparseMcoOffLoadOp_v, T_G2rSparseMcoMaskLoadOp &G2rSparseMcoMaskLoadOp_v, T_G2sSparseMcoValLoadOp &G2sSparseMcoValLoadOp_v, T_G2sMatrixBLoadOp &G2sMatrixBLoadOp_v, T_S2sRestoreMatrixAOp &S2sRestoreMatrixAOp_v, T_S2rAValLoadOp &S2rAValLoadOp_v, T_S2rBValLoadOp &S2rBValLoadOp_v, T_CalculateOp &CalculateOp_v, Tensor_o0 REGC)
    : tid(tid), lid(lid), wid(wid), B_valOp_v(B_valOp_v), C_valOp_v(C_valOp_v), val_mco_maskOp_v(val_mco_maskOp_v), val_mco_offOp_v(val_mco_offOp_v), val_mco_valOp_v(val_mco_valOp_v), val_sidxOp_v(val_sidxOp_v), val_soffOp_v(val_soffOp_v), ZeroOp_v(ZeroOp_v), BlockDimYOp_v(BlockDimYOp_v), BlockDimXOp_v(BlockDimXOp_v), G2rSparseOffsetLoadOp_v(G2rSparseOffsetLoadOp_v), ZerosOp_v(ZerosOp_v), G2rSparseIndexLoadOp_v(G2rSparseIndexLoadOp_v), G2rSparseMcoOffLoadOp_v(G2rSparseMcoOffLoadOp_v), G2rSparseMcoMaskLoadOp_v(G2rSparseMcoMaskLoadOp_v), G2sSparseMcoValLoadOp_v(G2sSparseMcoValLoadOp_v), G2sMatrixBLoadOp_v(G2sMatrixBLoadOp_v), S2sRestoreMatrixAOp_v(S2sRestoreMatrixAOp_v), S2rAValLoadOp_v(S2rAValLoadOp_v), S2rBValLoadOp_v(S2rBValLoadOp_v), CalculateOp_v(CalculateOp_v), REGC(REGC) {}

    // Pipeline methods
    // short pipe
    CUTE_DEVICE void short_pipe(int k, int l, int r) {
        if (k == 0) {
            // i = 0
            
            
            // i = 1
            
        }
        else if (k == 1) {
            // i = 0
            G2rSparseIndexLoadOp_v.template f<0>(
                val_sidxOp_v.template output<0>(),
                l + 0
            );
            G2rSparseMcoOffLoadOp_v.template f<0>(
                val_mco_offOp_v.template output<0>(),
                l + 0
            );
            G2rSparseMcoMaskLoadOp_v.template f<0>(
                val_mco_maskOp_v.template output<0>(),
                l + 0
            );
            G2sSparseMcoValLoadOp_v.template f<0>(
                val_mco_valOp_v.template output<0>(),
                G2rSparseMcoOffLoadOp_v.template output<0, 0>(),
                l + 0
            );
            __pipeline_commit();
            
            
            // i = 1
            G2sMatrixBLoadOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                G2rSparseIndexLoadOp_v.template output<0, 0>()(0)
            );
            __pipeline_commit();
            __pipeline_wait_prior(1);
            __syncthreads();
            S2sRestoreMatrixAOp_v.template f<0>(
                G2sSparseMcoValLoadOp_v.template output<0, 0>(),
                G2rSparseMcoMaskLoadOp_v.template output<0, 0>()
            );
            __pipeline_commit();
            
            
            // i = 2
            __syncthreads();
            S2rAValLoadOp_v.template f<0>(
                S2sRestoreMatrixAOp_v.template output<0, 0>()
            );
            __pipeline_wait_prior(1);
            __syncthreads();
            S2rBValLoadOp_v.template f<0>(
                G2sMatrixBLoadOp_v.template output<0, 0>()
            );
            CalculateOp_v.template f<0>(
                S2rAValLoadOp_v.template output<0, 0>(),
                S2rBValLoadOp_v.template output<0, 0>()
            );
            
        }
        else if (k == 2) {
            // i = 0
            G2rSparseIndexLoadOp_v.template f<0>(
                val_sidxOp_v.template output<0>(),
                l + 0
            );
            G2rSparseMcoOffLoadOp_v.template f<0>(
                val_mco_offOp_v.template output<0>(),
                l + 0
            );
            G2rSparseMcoMaskLoadOp_v.template f<0>(
                val_mco_maskOp_v.template output<0>(),
                l + 0
            );
            G2sSparseMcoValLoadOp_v.template f<0>(
                val_mco_valOp_v.template output<0>(),
                G2rSparseMcoOffLoadOp_v.template output<0, 0>(),
                l + 0
            );
            __pipeline_commit();
            
            
            // i = 1
            G2rSparseIndexLoadOp_v.template f<1>(
                val_sidxOp_v.template output<0>(),
                l + 1
            );
            G2rSparseMcoOffLoadOp_v.template f<1>(
                val_mco_offOp_v.template output<0>(),
                l + 1
            );
            G2rSparseMcoMaskLoadOp_v.template f<1>(
                val_mco_maskOp_v.template output<0>(),
                l + 1
            );
            G2sSparseMcoValLoadOp_v.template f<1>(
                val_mco_valOp_v.template output<0>(),
                G2rSparseMcoOffLoadOp_v.template output<0, 1>(),
                l + 1
            );
            __pipeline_commit();
            G2sMatrixBLoadOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                G2rSparseIndexLoadOp_v.template output<0, 0>()(0)
            );
            __pipeline_commit();
            __pipeline_wait_prior(2);
            __syncthreads();
            S2sRestoreMatrixAOp_v.template f<0>(
                G2sSparseMcoValLoadOp_v.template output<0, 0>(),
                G2rSparseMcoMaskLoadOp_v.template output<0, 0>()
            );
            __pipeline_commit();
            
            
            // i = 2
            G2sMatrixBLoadOp_v.template f<1>(
                B_valOp_v.template output<0>(),
                G2rSparseIndexLoadOp_v.template output<0, 1>()(0)
            );
            __pipeline_commit();
            __pipeline_wait_prior(3);
            __syncthreads();
            S2sRestoreMatrixAOp_v.template f<1>(
                G2sSparseMcoValLoadOp_v.template output<0, 1>(),
                G2rSparseMcoMaskLoadOp_v.template output<0, 1>()
            );
            __pipeline_commit();
            __syncthreads();
            S2rAValLoadOp_v.template f<0>(
                S2sRestoreMatrixAOp_v.template output<0, 0>()
            );
            __pipeline_wait_prior(3);
            __syncthreads();
            S2rBValLoadOp_v.template f<0>(
                G2sMatrixBLoadOp_v.template output<0, 0>()
            );
            CalculateOp_v.template f<0>(
                S2rAValLoadOp_v.template output<0, 0>(),
                S2rBValLoadOp_v.template output<0, 0>()
            );
            
            
            // i = 3
            __syncthreads();
            S2rAValLoadOp_v.template f<1>(
                S2sRestoreMatrixAOp_v.template output<0, 1>()
            );
            __pipeline_wait_prior(1);
            __syncthreads();
            S2rBValLoadOp_v.template f<1>(
                G2sMatrixBLoadOp_v.template output<0, 1>()
            );
            CalculateOp_v.template f<1>(
                S2rAValLoadOp_v.template output<0, 1>(),
                S2rBValLoadOp_v.template output<0, 1>()
            );
            
        }
        else if (k == 3) {
            // i = 0
            G2rSparseIndexLoadOp_v.template f<0>(
                val_sidxOp_v.template output<0>(),
                l + 0
            );
            G2rSparseMcoOffLoadOp_v.template f<0>(
                val_mco_offOp_v.template output<0>(),
                l + 0
            );
            G2rSparseMcoMaskLoadOp_v.template f<0>(
                val_mco_maskOp_v.template output<0>(),
                l + 0
            );
            G2sSparseMcoValLoadOp_v.template f<0>(
                val_mco_valOp_v.template output<0>(),
                G2rSparseMcoOffLoadOp_v.template output<0, 0>(),
                l + 0
            );
            __pipeline_commit();
            
            
            // i = 1
            G2rSparseIndexLoadOp_v.template f<1>(
                val_sidxOp_v.template output<0>(),
                l + 1
            );
            G2rSparseMcoOffLoadOp_v.template f<1>(
                val_mco_offOp_v.template output<0>(),
                l + 1
            );
            G2rSparseMcoMaskLoadOp_v.template f<1>(
                val_mco_maskOp_v.template output<0>(),
                l + 1
            );
            G2sSparseMcoValLoadOp_v.template f<1>(
                val_mco_valOp_v.template output<0>(),
                G2rSparseMcoOffLoadOp_v.template output<0, 1>(),
                l + 1
            );
            __pipeline_commit();
            G2sMatrixBLoadOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                G2rSparseIndexLoadOp_v.template output<0, 0>()(0)
            );
            __pipeline_commit();
            __pipeline_wait_prior(2);
            __syncthreads();
            S2sRestoreMatrixAOp_v.template f<0>(
                G2sSparseMcoValLoadOp_v.template output<0, 0>(),
                G2rSparseMcoMaskLoadOp_v.template output<0, 0>()
            );
            __pipeline_commit();
            
            
            // i = 2
            G2rSparseIndexLoadOp_v.template f<0>(
                val_sidxOp_v.template output<0>(),
                l + 2
            );
            G2rSparseMcoOffLoadOp_v.template f<0>(
                val_mco_offOp_v.template output<0>(),
                l + 2
            );
            G2rSparseMcoMaskLoadOp_v.template f<0>(
                val_mco_maskOp_v.template output<0>(),
                l + 2
            );
            G2sSparseMcoValLoadOp_v.template f<0>(
                val_mco_valOp_v.template output<0>(),
                G2rSparseMcoOffLoadOp_v.template output<0, 0>(),
                l + 2
            );
            __pipeline_commit();
            G2sMatrixBLoadOp_v.template f<1>(
                B_valOp_v.template output<0>(),
                G2rSparseIndexLoadOp_v.template output<0, 1>()(0)
            );
            __pipeline_commit();
            __pipeline_wait_prior(4);
            __syncthreads();
            S2sRestoreMatrixAOp_v.template f<1>(
                G2sSparseMcoValLoadOp_v.template output<0, 1>(),
                G2rSparseMcoMaskLoadOp_v.template output<0, 1>()
            );
            __pipeline_commit();
            __syncthreads();
            S2rAValLoadOp_v.template f<0>(
                S2sRestoreMatrixAOp_v.template output<0, 0>()
            );
            __pipeline_wait_prior(4);
            __syncthreads();
            S2rBValLoadOp_v.template f<0>(
                G2sMatrixBLoadOp_v.template output<0, 0>()
            );
            CalculateOp_v.template f<0>(
                S2rAValLoadOp_v.template output<0, 0>(),
                S2rBValLoadOp_v.template output<0, 0>()
            );
            
            
            // i = 3
            G2sMatrixBLoadOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                G2rSparseIndexLoadOp_v.template output<0, 0>()(0)
            );
            __pipeline_commit();
            __pipeline_wait_prior(3);
            __syncthreads();
            S2sRestoreMatrixAOp_v.template f<0>(
                G2sSparseMcoValLoadOp_v.template output<0, 0>(),
                G2rSparseMcoMaskLoadOp_v.template output<0, 0>()
            );
            __pipeline_commit();
            __syncthreads();
            S2rAValLoadOp_v.template f<1>(
                S2sRestoreMatrixAOp_v.template output<0, 1>()
            );
            __pipeline_wait_prior(3);
            __syncthreads();
            S2rBValLoadOp_v.template f<1>(
                G2sMatrixBLoadOp_v.template output<0, 1>()
            );
            CalculateOp_v.template f<1>(
                S2rAValLoadOp_v.template output<0, 1>(),
                S2rBValLoadOp_v.template output<0, 1>()
            );
            
            
            // i = 4
            __syncthreads();
            S2rAValLoadOp_v.template f<0>(
                S2sRestoreMatrixAOp_v.template output<0, 0>()
            );
            __pipeline_wait_prior(1);
            __syncthreads();
            S2rBValLoadOp_v.template f<0>(
                G2sMatrixBLoadOp_v.template output<0, 0>()
            );
            CalculateOp_v.template f<0>(
                S2rAValLoadOp_v.template output<0, 0>(),
                S2rBValLoadOp_v.template output<0, 0>()
            );
            
        }
        
    }
    
    // fill
    CUTE_DEVICE void fill(int l, int r) {
        // stage 0..0
        // op_call |buf_idx=0| 0 G2rSparseIndexLoadOp 
        G2rSparseIndexLoadOp_v.template f<0>(
            val_sidxOp_v.template output<0>(),
            l + 0
        );
        // op_call |buf_idx=0| 0 G2rSparseMcoOffLoadOp 
        G2rSparseMcoOffLoadOp_v.template f<0>(
            val_mco_offOp_v.template output<0>(),
            l + 0
        );
        // op_call |buf_idx=0| 0 G2rSparseMcoMaskLoadOp 
        G2rSparseMcoMaskLoadOp_v.template f<0>(
            val_mco_maskOp_v.template output<0>(),
            l + 0
        );
        // op_call |buf_idx=0| 0 G2sSparseMcoValLoadOp 
        G2sSparseMcoValLoadOp_v.template f<0>(
            val_mco_valOp_v.template output<0>(),
            G2rSparseMcoOffLoadOp_v.template output<0, 0>(),
            l + 0
        );
        __pipeline_commit();
        // stage 0..1
        // op_call |buf_idx=1| 1 G2rSparseIndexLoadOp 
        G2rSparseIndexLoadOp_v.template f<1>(
            val_sidxOp_v.template output<0>(),
            l + 1
        );
        // op_call |buf_idx=1| 1 G2rSparseMcoOffLoadOp 
        G2rSparseMcoOffLoadOp_v.template f<1>(
            val_mco_offOp_v.template output<0>(),
            l + 1
        );
        // op_call |buf_idx=1| 1 G2rSparseMcoMaskLoadOp 
        G2rSparseMcoMaskLoadOp_v.template f<1>(
            val_mco_maskOp_v.template output<0>(),
            l + 1
        );
        // op_call |buf_idx=1| 1 G2sSparseMcoValLoadOp 
        G2sSparseMcoValLoadOp_v.template f<1>(
            val_mco_valOp_v.template output<0>(),
            G2rSparseMcoOffLoadOp_v.template output<0, 1>(),
            l + 1
        );
        __pipeline_commit();
        // op_call |buf_idx=0| 0 G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            G2rSparseIndexLoadOp_v.template output<0, 0>()(0)
        );
        __pipeline_commit();
        __pipeline_wait_prior(2);
        __syncthreads();
        // op_call |buf_idx=0| 0 S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<0>(
            G2sSparseMcoValLoadOp_v.template output<0, 0>(),
            G2rSparseMcoMaskLoadOp_v.template output<0, 0>()
        );
        __pipeline_commit();
        // stage 0..2
        // op_call |buf_idx=0| 2 G2rSparseIndexLoadOp 
        G2rSparseIndexLoadOp_v.template f<0>(
            val_sidxOp_v.template output<0>(),
            l + 2
        );
        // op_call |buf_idx=0| 2 G2rSparseMcoOffLoadOp 
        G2rSparseMcoOffLoadOp_v.template f<0>(
            val_mco_offOp_v.template output<0>(),
            l + 2
        );
        // op_call |buf_idx=0| 2 G2rSparseMcoMaskLoadOp 
        G2rSparseMcoMaskLoadOp_v.template f<0>(
            val_mco_maskOp_v.template output<0>(),
            l + 2
        );
        // op_call |buf_idx=0| 2 G2sSparseMcoValLoadOp 
        G2sSparseMcoValLoadOp_v.template f<0>(
            val_mco_valOp_v.template output<0>(),
            G2rSparseMcoOffLoadOp_v.template output<0, 0>(),
            l + 2
        );
        __pipeline_commit();
        // op_call |buf_idx=1| 1 G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<1>(
            B_valOp_v.template output<0>(),
            G2rSparseIndexLoadOp_v.template output<0, 1>()(0)
        );
        __pipeline_commit();
        __pipeline_wait_prior(4);
        __syncthreads();
        // op_call |buf_idx=1| 1 S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<1>(
            G2sSparseMcoValLoadOp_v.template output<0, 1>(),
            G2rSparseMcoMaskLoadOp_v.template output<0, 1>()
        );
        __pipeline_commit();
        __syncthreads();
        // op_call |buf_idx=0| 0 S2rAValLoadOp 
        S2rAValLoadOp_v.template f<0>(
            S2sRestoreMatrixAOp_v.template output<0, 0>()
        );
        __pipeline_wait_prior(4);
        __syncthreads();
        // op_call |buf_idx=0| 0 S2rBValLoadOp 
        S2rBValLoadOp_v.template f<0>(
            G2sMatrixBLoadOp_v.template output<0, 0>()
        );
        // op_call |buf_idx=0| 0 CalculateOp 
        CalculateOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            S2rBValLoadOp_v.template output<0, 0>()
        );
    }
    
    // loop_step
    CUTE_DEVICE void loop_step(int i, int l, int r) {
        // step 0
        // op_call |buf_idx=1| i + (0) G2rSparseIndexLoadOp 
        G2rSparseIndexLoadOp_v.template f<1>(
            val_sidxOp_v.template output<0>(),
            l + i + (0)
        );
        // op_call |buf_idx=1| i + (0) G2rSparseMcoOffLoadOp 
        G2rSparseMcoOffLoadOp_v.template f<1>(
            val_mco_offOp_v.template output<0>(),
            l + i + (0)
        );
        // op_call |buf_idx=1| i + (0) G2rSparseMcoMaskLoadOp 
        G2rSparseMcoMaskLoadOp_v.template f<1>(
            val_mco_maskOp_v.template output<0>(),
            l + i + (0)
        );
        // op_call |buf_idx=1| i + (0) G2sSparseMcoValLoadOp 
        G2sSparseMcoValLoadOp_v.template f<1>(
            val_mco_valOp_v.template output<0>(),
            G2rSparseMcoOffLoadOp_v.template output<0, 1>(),
            l + i + (0)
        );
        __pipeline_commit();
        // op_call |buf_idx=0| i + (-1) G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            G2rSparseIndexLoadOp_v.template output<0, 0>()(0)
        );
        __pipeline_commit();
        __pipeline_wait_prior(4);
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<0>(
            G2sSparseMcoValLoadOp_v.template output<0, 0>(),
            G2rSparseMcoMaskLoadOp_v.template output<0, 0>()
        );
        __pipeline_commit();
        __syncthreads();
        // op_call |buf_idx=1| i + (-2) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<1>(
            S2sRestoreMatrixAOp_v.template output<0, 1>()
        );
        __pipeline_wait_prior(4);
        __syncthreads();
        // op_call |buf_idx=1| i + (-2) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<1>(
            G2sMatrixBLoadOp_v.template output<0, 1>()
        );
        // op_call |buf_idx=1| i + (-2) CalculateOp 
        CalculateOp_v.template f<1>(
            S2rAValLoadOp_v.template output<0, 1>(),
            S2rBValLoadOp_v.template output<0, 1>()
        );
        // step 1
        // op_call |buf_idx=0| i + (1) G2rSparseIndexLoadOp 
        G2rSparseIndexLoadOp_v.template f<0>(
            val_sidxOp_v.template output<0>(),
            l + i + (1)
        );
        // op_call |buf_idx=0| i + (1) G2rSparseMcoOffLoadOp 
        G2rSparseMcoOffLoadOp_v.template f<0>(
            val_mco_offOp_v.template output<0>(),
            l + i + (1)
        );
        // op_call |buf_idx=0| i + (1) G2rSparseMcoMaskLoadOp 
        G2rSparseMcoMaskLoadOp_v.template f<0>(
            val_mco_maskOp_v.template output<0>(),
            l + i + (1)
        );
        // op_call |buf_idx=0| i + (1) G2sSparseMcoValLoadOp 
        G2sSparseMcoValLoadOp_v.template f<0>(
            val_mco_valOp_v.template output<0>(),
            G2rSparseMcoOffLoadOp_v.template output<0, 0>(),
            l + i + (1)
        );
        __pipeline_commit();
        // op_call |buf_idx=1| i + (0) G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<1>(
            B_valOp_v.template output<0>(),
            G2rSparseIndexLoadOp_v.template output<0, 1>()(0)
        );
        __pipeline_commit();
        __pipeline_wait_prior(4);
        __syncthreads();
        // op_call |buf_idx=1| i + (0) S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<1>(
            G2sSparseMcoValLoadOp_v.template output<0, 1>(),
            G2rSparseMcoMaskLoadOp_v.template output<0, 1>()
        );
        __pipeline_commit();
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<0>(
            S2sRestoreMatrixAOp_v.template output<0, 0>()
        );
        __pipeline_wait_prior(4);
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<0>(
            G2sMatrixBLoadOp_v.template output<0, 0>()
        );
        // op_call |buf_idx=0| i + (-1) CalculateOp 
        CalculateOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            S2rBValLoadOp_v.template output<0, 0>()
        );
    }
    
    // remainder
    CUTE_DEVICE void remains_1(int i, int l, int r) {
        // total remains 1, current 0
        // op_call |buf_idx=1| i + (0) G2rSparseIndexLoadOp 
        G2rSparseIndexLoadOp_v.template f<1>(
            val_sidxOp_v.template output<0>(),
            l + i + (0)
        );
        // op_call |buf_idx=1| i + (0) G2rSparseMcoOffLoadOp 
        G2rSparseMcoOffLoadOp_v.template f<1>(
            val_mco_offOp_v.template output<0>(),
            l + i + (0)
        );
        // op_call |buf_idx=1| i + (0) G2rSparseMcoMaskLoadOp 
        G2rSparseMcoMaskLoadOp_v.template f<1>(
            val_mco_maskOp_v.template output<0>(),
            l + i + (0)
        );
        // op_call |buf_idx=1| i + (0) G2sSparseMcoValLoadOp 
        G2sSparseMcoValLoadOp_v.template f<1>(
            val_mco_valOp_v.template output<0>(),
            G2rSparseMcoOffLoadOp_v.template output<0, 1>(),
            l + i + (0)
        );
        __pipeline_commit();
        // op_call |buf_idx=0| i + (-1) G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            G2rSparseIndexLoadOp_v.template output<0, 0>()(0)
        );
        __pipeline_commit();
        __pipeline_wait_prior(4);
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<0>(
            G2sSparseMcoValLoadOp_v.template output<0, 0>(),
            G2rSparseMcoMaskLoadOp_v.template output<0, 0>()
        );
        __pipeline_commit();
        __syncthreads();
        // op_call |buf_idx=1| i + (-2) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<1>(
            S2sRestoreMatrixAOp_v.template output<0, 1>()
        );
        __pipeline_wait_prior(4);
        __syncthreads();
        // op_call |buf_idx=1| i + (-2) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<1>(
            G2sMatrixBLoadOp_v.template output<0, 1>()
        );
        // op_call |buf_idx=1| i + (-2) CalculateOp 
        CalculateOp_v.template f<1>(
            S2rAValLoadOp_v.template output<0, 1>(),
            S2rBValLoadOp_v.template output<0, 1>()
        );
    }
    
    CUTE_DEVICE void remainder(int i, int l, int r) {
        if (r - l - i == 1) {
            remains_1(i, l, r);
        }
    }
    
    // empties
    CUTE_DEVICE void empty_after_remain_0(int i, int l, int r) {
        // stage 1..2
        // op_call |buf_idx=0| i + (-1) G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            G2rSparseIndexLoadOp_v.template output<0, 0>()(0)
        );
        __pipeline_commit();
        __pipeline_wait_prior(3);
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<0>(
            G2sSparseMcoValLoadOp_v.template output<0, 0>(),
            G2rSparseMcoMaskLoadOp_v.template output<0, 0>()
        );
        __pipeline_commit();
        __syncthreads();
        // op_call |buf_idx=1| i + (-2) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<1>(
            S2sRestoreMatrixAOp_v.template output<0, 1>()
        );
        __pipeline_wait_prior(3);
        __syncthreads();
        // op_call |buf_idx=1| i + (-2) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<1>(
            G2sMatrixBLoadOp_v.template output<0, 1>()
        );
        // op_call |buf_idx=1| i + (-2) CalculateOp 
        CalculateOp_v.template f<1>(
            S2rAValLoadOp_v.template output<0, 1>(),
            S2rBValLoadOp_v.template output<0, 1>()
        );
        // stage 2..2
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<0>(
            S2sRestoreMatrixAOp_v.template output<0, 0>()
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<0>(
            G2sMatrixBLoadOp_v.template output<0, 0>()
        );
        // op_call |buf_idx=0| i + (-1) CalculateOp 
        CalculateOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            S2rBValLoadOp_v.template output<0, 0>()
        );
    }
    
    CUTE_DEVICE void empty_after_remain_1(int i, int l, int r) {
        // stage 1..2
        // op_call |buf_idx=1| i + (0) G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<1>(
            B_valOp_v.template output<0>(),
            G2rSparseIndexLoadOp_v.template output<0, 1>()(0)
        );
        __pipeline_commit();
        __pipeline_wait_prior(3);
        __syncthreads();
        // op_call |buf_idx=1| i + (0) S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<1>(
            G2sSparseMcoValLoadOp_v.template output<0, 1>(),
            G2rSparseMcoMaskLoadOp_v.template output<0, 1>()
        );
        __pipeline_commit();
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<0>(
            S2sRestoreMatrixAOp_v.template output<0, 0>()
        );
        __pipeline_wait_prior(3);
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<0>(
            G2sMatrixBLoadOp_v.template output<0, 0>()
        );
        // op_call |buf_idx=0| i + (-1) CalculateOp 
        CalculateOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            S2rBValLoadOp_v.template output<0, 0>()
        );
        // stage 2..2
        __syncthreads();
        // op_call |buf_idx=1| i + (0) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<1>(
            S2sRestoreMatrixAOp_v.template output<0, 1>()
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        // op_call |buf_idx=1| i + (0) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<1>(
            G2sMatrixBLoadOp_v.template output<0, 1>()
        );
        // op_call |buf_idx=1| i + (0) CalculateOp 
        CalculateOp_v.template f<1>(
            S2rAValLoadOp_v.template output<0, 1>(),
            S2rBValLoadOp_v.template output<0, 1>()
        );
    }
    
    CUTE_DEVICE void empty(int i, int l, int r) {
        if (r - l - i == 0) {
            empty_after_remain_0(i, l, r);
        }
        else if (r - l - i == 1) {
            empty_after_remain_1(i, l, r);
        }
    }
    
    // dispatch
    CUTE_DEVICE void dispatch() {
        int l = this->_l;
        int r = this->_r;
        int k = r - l;
        if (k <= 3) {
            short_pipe(k, l, r);
        } else {
            fill(l, r);
            int i;
            for (i = 3; i + 2 <= k; i += 2) {
                loop_step(i, l, r);
            }
            remainder(i, l, r);
            empty(i, l, r);
        }
    }

    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGC;
        }
    }

    // f
    template<typename = void>
    CUTE_DEVICE void f(int l, int r) {
        _l = l;
        _r = r;
        dispatch();
    }
};

template<class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class Tensor_o0, class TI0 = float, class TO0 = float>
class R2gCValStoreOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{}))));
    Shape_i0 shape_i0;
    // Tensor_i0 REGC;

    // Outputs
    using Shape_o0 = decltype(make_shape(get<1>(BLK_MNK{}), get<0>(BLK_MNK{})));
    Shape_o0 shape_o0;
    Tensor_o0 C_val;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE R2gCValStoreOp(int tid, int lid, int wid, Tensor_o0 C_val)
    : tid(tid), lid(lid), wid(wid), C_val(C_val) {}

    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return C_val;
        }
    }

    template<class Tensor_i0, typename = void>
    CUTE_DEVICE void f(Tensor_i0 REGC) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(REGC))));
        CUTE_STATIC_ASSERT_V((get<0>(get<1>(shape_i0)) == size<0>(get<1>(shape(REGC)))));
        CUTE_STATIC_ASSERT_V((get<1>(get<1>(shape_i0)) == size<1>(get<1>(shape(REGC)))));
        //! write back
        for (int i_tileN = 0; i_tileN < get<1>(BLK_MNK{}) / 8; i_tileN++) {
            int row = lid / 4;
            int col = i_tileN * 8 + lid % 4 * 2;
            C_val(row, col) = REGC(0, i_tileN);
            C_val(row, col + 1) = REGC(1, i_tileN);
            C_val(row + 8, col) = REGC(2, i_tileN);
            C_val(row + 8, col + 1) = REGC(3, i_tileN);
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class T_B_valOp, class T_C_valOp, class T_val_mco_maskOp, class T_val_mco_offOp, class T_val_mco_valOp, class T_val_sidxOp, class T_val_soffOp, class T_ZeroOp, class T_BlockDimYOp, class T_BlockDimXOp, class T_G2rSparseOffsetLoadOp, class T_ZerosOp, class T_BitBsrMainLoopOp, class T_R2gCValStoreOp>
class bitbsr_spmm_kernel_blk_x_loop {
public:
    // Inputs
    int _idx;

    // Hardware parameters
    int tid, lid, wid;

    // Body Ops
    T_B_valOp &B_valOp_v;
    T_C_valOp &C_valOp_v;
    T_val_mco_maskOp &val_mco_maskOp_v;
    T_val_mco_offOp &val_mco_offOp_v;
    T_val_mco_valOp &val_mco_valOp_v;
    T_val_sidxOp &val_sidxOp_v;
    T_val_soffOp &val_soffOp_v;
    T_ZeroOp &ZeroOp_v;
    T_BlockDimYOp &BlockDimYOp_v;
    T_BlockDimXOp &BlockDimXOp_v;
    T_G2rSparseOffsetLoadOp &G2rSparseOffsetLoadOp_v;
    T_ZerosOp &ZerosOp_v;
    T_BitBsrMainLoopOp &BitBsrMainLoopOp_v;
    T_R2gCValStoreOp &R2gCValStoreOp_v;

    // Constructor
    CUTE_DEVICE bitbsr_spmm_kernel_blk_x_loop(int tid, int lid, int wid, T_B_valOp &B_valOp_v, T_C_valOp &C_valOp_v, T_val_mco_maskOp &val_mco_maskOp_v, T_val_mco_offOp &val_mco_offOp_v, T_val_mco_valOp &val_mco_valOp_v, T_val_sidxOp &val_sidxOp_v, T_val_soffOp &val_soffOp_v, T_ZeroOp &ZeroOp_v, T_BlockDimYOp &BlockDimYOp_v, T_BlockDimXOp &BlockDimXOp_v, T_G2rSparseOffsetLoadOp &G2rSparseOffsetLoadOp_v, T_ZerosOp &ZerosOp_v, T_BitBsrMainLoopOp &BitBsrMainLoopOp_v, T_R2gCValStoreOp &R2gCValStoreOp_v)
    : tid(tid), lid(lid), wid(wid), B_valOp_v(B_valOp_v), C_valOp_v(C_valOp_v), val_mco_maskOp_v(val_mco_maskOp_v), val_mco_offOp_v(val_mco_offOp_v), val_mco_valOp_v(val_mco_valOp_v), val_sidxOp_v(val_sidxOp_v), val_soffOp_v(val_soffOp_v), ZeroOp_v(ZeroOp_v), BlockDimYOp_v(BlockDimYOp_v), BlockDimXOp_v(BlockDimXOp_v), G2rSparseOffsetLoadOp_v(G2rSparseOffsetLoadOp_v), ZerosOp_v(ZerosOp_v), BitBsrMainLoopOp_v(BitBsrMainLoopOp_v), R2gCValStoreOp_v(R2gCValStoreOp_v) {}

    // Body
    CUTE_DEVICE void body() {
        G2rSparseOffsetLoadOp_v.template f<>(
            val_soffOp_v.template output<0>(),
            blockIdx.x
        );
        ZerosOp_v.template f<>();
        BitBsrMainLoopOp_v.template f<>(
            G2rSparseOffsetLoadOp_v.template output<0>()(0),
            G2rSparseOffsetLoadOp_v.template output<1>()(0)
        );
        R2gCValStoreOp_v.template f<>(
            BitBsrMainLoopOp_v.template output<0>()
        );
    }

    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        this->_idx = blockIdx.x;
        body();
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class T_B_valOp, class T_C_valOp, class T_val_mco_maskOp, class T_val_mco_offOp, class T_val_mco_valOp, class T_val_sidxOp, class T_val_soffOp, class T_ZeroOp, class T_BlockDimYOp, class T_BlockDimXOp, class T_bitbsr_spmm_kernel_blk_x_loop>
class bitbsr_spmm_kernel_blk_y_loop {
public:
    // Inputs
    int _idx;

    // Hardware parameters
    int tid, lid, wid;

    // Body Ops
    T_B_valOp &B_valOp_v;
    T_C_valOp &C_valOp_v;
    T_val_mco_maskOp &val_mco_maskOp_v;
    T_val_mco_offOp &val_mco_offOp_v;
    T_val_mco_valOp &val_mco_valOp_v;
    T_val_sidxOp &val_sidxOp_v;
    T_val_soffOp &val_soffOp_v;
    T_ZeroOp &ZeroOp_v;
    T_BlockDimYOp &BlockDimYOp_v;
    T_BlockDimXOp &BlockDimXOp_v;
    T_bitbsr_spmm_kernel_blk_x_loop &bitbsr_spmm_kernel_blk_x_loop_v;

    // Constructor
    CUTE_DEVICE bitbsr_spmm_kernel_blk_y_loop(int tid, int lid, int wid, T_B_valOp &B_valOp_v, T_C_valOp &C_valOp_v, T_val_mco_maskOp &val_mco_maskOp_v, T_val_mco_offOp &val_mco_offOp_v, T_val_mco_valOp &val_mco_valOp_v, T_val_sidxOp &val_sidxOp_v, T_val_soffOp &val_soffOp_v, T_ZeroOp &ZeroOp_v, T_BlockDimYOp &BlockDimYOp_v, T_BlockDimXOp &BlockDimXOp_v, T_bitbsr_spmm_kernel_blk_x_loop &bitbsr_spmm_kernel_blk_x_loop_v)
    : tid(tid), lid(lid), wid(wid), B_valOp_v(B_valOp_v), C_valOp_v(C_valOp_v), val_mco_maskOp_v(val_mco_maskOp_v), val_mco_offOp_v(val_mco_offOp_v), val_mco_valOp_v(val_mco_valOp_v), val_sidxOp_v(val_sidxOp_v), val_soffOp_v(val_soffOp_v), ZeroOp_v(ZeroOp_v), BlockDimYOp_v(BlockDimYOp_v), BlockDimXOp_v(BlockDimXOp_v), bitbsr_spmm_kernel_blk_x_loop_v(bitbsr_spmm_kernel_blk_x_loop_v) {}

    // Body
    CUTE_DEVICE void body() {
        bitbsr_spmm_kernel_blk_x_loop_v.template f<>();
    }

    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        this->_idx = blockIdx.y;
        body();
    }
};

// GLOBAL FUNCTION
template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class nmask_per_tile, class tile_mco_val_size>
__global__ void bitbsr_spmm_kernel_tf32tf32fp32(
    float *dB_val,
    float *dC_val,
    uint64_t *dval_mco_mask,
    int *dval_mco_off,
    float *dval_mco_val,
    int *dval_sidx,
    int *dval_soff,
    int K, int M, int Mo, int N, int nnz, int nnz_block
) {
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int wid = tid / 32;

    using B_valOp_t = B_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    B_valOp_t B_valOp_v{tid, lid, wid, K, N, dB_val};
    using C_valOp_t = C_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    C_valOp_t C_valOp_v{tid, lid, wid, M, N, dC_val};
    using val_mco_maskOp_t = val_mco_maskOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_mco_maskOp_t val_mco_maskOp_v{tid, lid, wid, nnz_block, dval_mco_mask};
    using val_mco_offOp_t = val_mco_offOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_mco_offOp_t val_mco_offOp_v{tid, lid, wid, nnz_block, dval_mco_off};
    using val_mco_valOp_t = val_mco_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_mco_valOp_t val_mco_valOp_v{tid, lid, wid, nnz, dval_mco_val};
    using val_sidxOp_t = val_sidxOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_sidxOp_t val_sidxOp_v{tid, lid, wid, nnz_block, dval_sidx};
    using val_soffOp_t = val_soffOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_soffOp_t val_soffOp_v{tid, lid, wid, Mo, dval_soff};
    using ZeroOp_t = ZeroOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using ZeroOp_layout_o0 = typename ZeroOp_t::Layout_o0;
    int ZeroOp_tensor_o0[cosize_v<ZeroOp_layout_o0>];
    ZeroOp_t ZeroOp_v{tid, lid, wid, ZeroOp_tensor_o0};
    using BlockDimYOp_t = BlockDimYOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlockDimYOp_layout_o0 = typename BlockDimYOp_t::Layout_o0;
    int BlockDimYOp_tensor_o0[cosize_v<BlockDimYOp_layout_o0>];
    BlockDimYOp_t BlockDimYOp_v{tid, lid, wid, BlockDimYOp_tensor_o0};
    using BlockDimXOp_t = BlockDimXOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlockDimXOp_layout_o0 = typename BlockDimXOp_t::Layout_o0;
    int BlockDimXOp_tensor_o0[cosize_v<BlockDimXOp_layout_o0>];
    BlockDimXOp_t BlockDimXOp_v{tid, lid, wid, BlockDimXOp_tensor_o0};
    using G2rSparseOffsetLoadOp_t = G2rSparseOffsetLoadOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2rSparseOffsetLoadOp_layout_o0 = typename G2rSparseOffsetLoadOp_t::Layout_o0;
    using G2rSparseOffsetLoadOp_layout_o1 = typename G2rSparseOffsetLoadOp_t::Layout_o1;
    int G2rSparseOffsetLoadOp_tensor_o0[cosize_v<G2rSparseOffsetLoadOp_layout_o0>];
    int G2rSparseOffsetLoadOp_tensor_o1[cosize_v<G2rSparseOffsetLoadOp_layout_o1>];
    G2rSparseOffsetLoadOp_t G2rSparseOffsetLoadOp_v{
        tid, lid, wid, 
        G2rSparseOffsetLoadOp_tensor_o0, G2rSparseOffsetLoadOp_tensor_o1, 
        Mo,
    };
    using ZerosOp_t = ZerosOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using ZerosOp_layout_o0 = typename ZerosOp_t::Layout_o0;
    float ZerosOp_tensor_o0[cosize_v<ZerosOp_layout_o0>];
    ZerosOp_t ZerosOp_v{tid, lid, wid, ZerosOp_tensor_o0};
    using G2rSparseIndexLoadOp_t = G2rSparseIndexLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2rSparseIndexLoadOp_layout_o0 = typename G2rSparseIndexLoadOp_t::Layout_o0;
    int G2rSparseIndexLoadOp_tensor_o0[cosize_v<G2rSparseIndexLoadOp_layout_o0>];
    G2rSparseIndexLoadOp_t G2rSparseIndexLoadOp_v{
        tid, lid, wid, 
        G2rSparseIndexLoadOp_tensor_o0, 
        nnz_block,
    };
    using G2rSparseMcoOffLoadOp_t = G2rSparseMcoOffLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2rSparseMcoOffLoadOp_layout_o0 = typename G2rSparseMcoOffLoadOp_t::Layout_o0;
    int G2rSparseMcoOffLoadOp_tensor_o0[cosize_v<G2rSparseMcoOffLoadOp_layout_o0>];
    G2rSparseMcoOffLoadOp_t G2rSparseMcoOffLoadOp_v{
        tid, lid, wid, 
        G2rSparseMcoOffLoadOp_tensor_o0, 
        nnz_block,
    };
    using G2rSparseMcoMaskLoadOp_t = G2rSparseMcoMaskLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2rSparseMcoMaskLoadOp_layout_o0 = typename G2rSparseMcoMaskLoadOp_t::Layout_o0;
    uint64_t G2rSparseMcoMaskLoadOp_tensor_o0[cosize_v<G2rSparseMcoMaskLoadOp_layout_o0>];
    G2rSparseMcoMaskLoadOp_t G2rSparseMcoMaskLoadOp_v{
        tid, lid, wid, 
        G2rSparseMcoMaskLoadOp_tensor_o0, 
        nnz_block,
    };
    using G2sSparseMcoValLoadOp_t = G2sSparseMcoValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sSparseMcoValLoadOp_layout_o0 = typename G2sSparseMcoValLoadOp_t::Layout_o0;
    __shared__ float G2sSparseMcoValLoadOp_tensor_o0[cosize_v<G2sSparseMcoValLoadOp_layout_o0>];
    G2sSparseMcoValLoadOp_t G2sSparseMcoValLoadOp_v{
        tid, lid, wid, 
        G2sSparseMcoValLoadOp_tensor_o0, 
        nnz,
    };
    using G2sMatrixBLoadOp_t = G2sMatrixBLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sMatrixBLoadOp_layout_o0 = typename G2sMatrixBLoadOp_t::Layout_o0;
    __shared__ float G2sMatrixBLoadOp_tensor_o0[cosize_v<G2sMatrixBLoadOp_layout_o0>];
    G2sMatrixBLoadOp_t G2sMatrixBLoadOp_v{
        tid, lid, wid, 
        G2sMatrixBLoadOp_tensor_o0, 
        K, N,
    };
    using S2sRestoreMatrixAOp_t = S2sRestoreMatrixAOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using S2sRestoreMatrixAOp_layout_o0 = typename S2sRestoreMatrixAOp_t::Layout_o0;
    __shared__ float S2sRestoreMatrixAOp_tensor_o0[cosize_v<S2sRestoreMatrixAOp_layout_o0>];
    S2sRestoreMatrixAOp_t S2sRestoreMatrixAOp_v{
        tid, lid, wid, 
        S2sRestoreMatrixAOp_tensor_o0,
    };
    using S2rAValLoadOp_t = S2rAValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using S2rAValLoadOp_layout_o0 = typename S2rAValLoadOp_t::Layout_o0;
    float S2rAValLoadOp_tensor_o0[cosize_v<S2rAValLoadOp_layout_o0>];
    S2rAValLoadOp_t S2rAValLoadOp_v{
        tid, lid, wid, 
        S2rAValLoadOp_tensor_o0,
    };
    using S2rBValLoadOp_t = S2rBValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using S2rBValLoadOp_layout_o0 = typename S2rBValLoadOp_t::Layout_o0;
    float S2rBValLoadOp_tensor_o0[cosize_v<S2rBValLoadOp_layout_o0>];
    S2rBValLoadOp_t S2rBValLoadOp_v{
        tid, lid, wid, 
        S2rBValLoadOp_tensor_o0,
    };
    using CalculateOp_t = CalculateOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, std::decay_t<decltype(ZerosOp_v.template output<0>())>>;
    CalculateOp_t CalculateOp_v{
        tid, lid, wid, 
        ZerosOp_v.template output<0>(),
    };
    using BitBsrMainLoopOp_t = BitBsrMainLoopOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, B_valOp_t, C_valOp_t, val_mco_maskOp_t, val_mco_offOp_t, val_mco_valOp_t, val_sidxOp_t, val_soffOp_t, ZeroOp_t, BlockDimYOp_t, BlockDimXOp_t, G2rSparseOffsetLoadOp_t, ZerosOp_t, G2rSparseIndexLoadOp_t, G2rSparseMcoOffLoadOp_t, G2rSparseMcoMaskLoadOp_t, G2sSparseMcoValLoadOp_t, G2sMatrixBLoadOp_t, S2sRestoreMatrixAOp_t, S2rAValLoadOp_t, S2rBValLoadOp_t, CalculateOp_t, std::decay_t<decltype(CalculateOp_v.template output<0>())>>;
    BitBsrMainLoopOp_t BitBsrMainLoopOp_v{tid, lid, wid, B_valOp_v, C_valOp_v, val_mco_maskOp_v, val_mco_offOp_v, val_mco_valOp_v, val_sidxOp_v, val_soffOp_v, ZeroOp_v, BlockDimYOp_v, BlockDimXOp_v, G2rSparseOffsetLoadOp_v, ZerosOp_v, G2rSparseIndexLoadOp_v, G2rSparseMcoOffLoadOp_v, G2rSparseMcoMaskLoadOp_v, G2sSparseMcoValLoadOp_v, G2sMatrixBLoadOp_v, S2sRestoreMatrixAOp_v, S2rAValLoadOp_v, S2rBValLoadOp_v, CalculateOp_v, CalculateOp_v.template output<0>()};
    using R2gCValStoreOp_t = R2gCValStoreOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, std::decay_t<decltype(C_valOp_v.template output<0>())>>;
    R2gCValStoreOp_t R2gCValStoreOp_v{
        tid, lid, wid, 
        C_valOp_v.template output<0>(),
    };
    using bitbsr_spmm_kernel_blk_x_loop_t = bitbsr_spmm_kernel_blk_x_loop<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, B_valOp_t, C_valOp_t, val_mco_maskOp_t, val_mco_offOp_t, val_mco_valOp_t, val_sidxOp_t, val_soffOp_t, ZeroOp_t, BlockDimYOp_t, BlockDimXOp_t, G2rSparseOffsetLoadOp_t, ZerosOp_t, BitBsrMainLoopOp_t, R2gCValStoreOp_t>;
    bitbsr_spmm_kernel_blk_x_loop_t bitbsr_spmm_kernel_blk_x_loop_v{tid, lid, wid, B_valOp_v, C_valOp_v, val_mco_maskOp_v, val_mco_offOp_v, val_mco_valOp_v, val_sidxOp_v, val_soffOp_v, ZeroOp_v, BlockDimYOp_v, BlockDimXOp_v, G2rSparseOffsetLoadOp_v, ZerosOp_v, BitBsrMainLoopOp_v, R2gCValStoreOp_v};
    using bitbsr_spmm_kernel_blk_y_loop_t = bitbsr_spmm_kernel_blk_y_loop<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, B_valOp_t, C_valOp_t, val_mco_maskOp_t, val_mco_offOp_t, val_mco_valOp_t, val_sidxOp_t, val_soffOp_t, ZeroOp_t, BlockDimYOp_t, BlockDimXOp_t, bitbsr_spmm_kernel_blk_x_loop_t>;
    bitbsr_spmm_kernel_blk_y_loop_t bitbsr_spmm_kernel_blk_y_loop_v{tid, lid, wid, B_valOp_v, C_valOp_v, val_mco_maskOp_v, val_mco_offOp_v, val_mco_valOp_v, val_sidxOp_v, val_soffOp_v, ZeroOp_v, BlockDimYOp_v, BlockDimXOp_v, bitbsr_spmm_kernel_blk_x_loop_v};

    ZeroOp_v.template f<>();
    BlockDimYOp_v.template f<>();
    BlockDimXOp_v.template f<>();
    bitbsr_spmm_kernel_blk_y_loop_v.template f<>();
};