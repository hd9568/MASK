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

#define SPARSENE_GENERATED_TILE_B 64
#define SPARSENE_GENERATED_B_SMEM_SWIZZLE "swizzle_323"
#define SPARSENE_GENERATED_COO_RESTORE_SWIZZLE "swizzle_123"

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

template <class TElem, class TSrc, class TDst>
__device__ __forceinline__ void sparsene_copy_g2s_128(TSrc&& src, TDst&& dst) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, TElem>{}, src, dst);
#else
    copy(src, dst);
#endif
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
    int lid;

    // Constructor
    CUTE_DEVICE B_valOp(int lid, int K, int N, TO0 *dB_val)
    : lid(lid),
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
    using Tensor_o0 = decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN, VARLEN)), select<1, 0>(BLK_MNK{}))(make_coord(_, VARLEN), make_coord(_, VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 C_val;

    // Hardware parameters
    int lid;

    // Constructor
    CUTE_DEVICE C_valOp(int lid, int N, int M, TO0 *dC_val)
    : lid(lid),
    shape_o0(logical_divide(make_tensor(make_gmem_ptr(dC_val), make_shape(N, M)), select<1, 0>(BLK_MNK{}))(make_coord(_, blockIdx.y), make_coord(_, blockIdx.x)).shape()),
    layout_o0(logical_divide(make_tensor(make_gmem_ptr(dC_val), make_shape(N, M)), select<1, 0>(BLK_MNK{}))(make_coord(_, blockIdx.y), make_coord(_, blockIdx.x)).layout()),
    C_val(logical_divide(make_tensor(make_gmem_ptr(dC_val), make_shape(N, M)), select<1, 0>(BLK_MNK{}))(make_coord(_, blockIdx.y), make_coord(_, blockIdx.x))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return C_val;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class val_coo_idx_newOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_coo_idx_new;

    // Hardware parameters
    int lid;

    // Constructor
    CUTE_DEVICE val_coo_idx_newOp(int lid, int nnz, TO0 *dval_coo_idx_new)
    : lid(lid),
    shape_o0(make_tensor(make_gmem_ptr(dval_coo_idx_new), make_shape(nnz)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_coo_idx_new), make_shape(nnz)).layout()),
    val_coo_idx_new(make_tensor(make_gmem_ptr(dval_coo_idx_new), make_shape(nnz))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_coo_idx_new;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class val_coo_off_newOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(Int<2>{}, VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_coo_off_new;

    // Hardware parameters
    int lid;

    // Constructor
    CUTE_DEVICE val_coo_off_newOp(int lid, int nnz_dim_K, TO0 *dval_coo_off_new)
    : lid(lid),
    shape_o0(make_tensor(make_gmem_ptr(dval_coo_off_new), make_shape(Int<2>{}, nnz_dim_K)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_coo_off_new), make_shape(Int<2>{}, nnz_dim_K)).layout()),
    val_coo_off_new(make_tensor(make_gmem_ptr(dval_coo_off_new), make_shape(Int<2>{}, nnz_dim_K))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_coo_off_new;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = float>
class val_coo_val_newOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_coo_val_new;

    // Hardware parameters
    int lid;

    // Constructor
    CUTE_DEVICE val_coo_val_newOp(int lid, int nnz, TO0 *dval_coo_val_new)
    : lid(lid),
    shape_o0(make_tensor(make_gmem_ptr(dval_coo_val_new), make_shape(nnz)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_coo_val_new), make_shape(nnz)).layout()),
    val_coo_val_new(make_tensor(make_gmem_ptr(dval_coo_val_new), make_shape(nnz))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_coo_val_new;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class val_len_offsetOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_len_offset;

    // Hardware parameters
    int lid;

    // Constructor
    CUTE_DEVICE val_len_offsetOp(int lid, int Mo, TO0 *dval_len_offset)
    : lid(lid),
    shape_o0(make_tensor(make_gmem_ptr(dval_len_offset), make_shape(Mo)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_len_offset), make_shape(Mo)).layout()),
    val_len_offset(make_tensor(make_gmem_ptr(dval_len_offset), make_shape(Mo))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_len_offset;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class val_sidx_newOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(get<2>(BLK_MNK{}), VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_sidx_new;

    // Hardware parameters
    int lid;

    // Constructor
    CUTE_DEVICE val_sidx_newOp(int lid, int nnz_dim_K, TO0 *dval_sidx_new)
    : lid(lid),
    shape_o0(make_tensor(make_gmem_ptr(dval_sidx_new), make_shape(get<2>(BLK_MNK{}), nnz_dim_K)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_sidx_new), make_shape(get<2>(BLK_MNK{}), nnz_dim_K)).layout()),
    val_sidx_new(make_tensor(make_gmem_ptr(dval_sidx_new), make_shape(get<2>(BLK_MNK{}), nnz_dim_K))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_sidx_new;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class BlkYLoopZeroOp {
public:
    // No Inputs

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Stride_o0 = decltype(make_stride(_0{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 BlkYLoopZeroOp_out;
    // Hardware parameters
    int lid;

    // Constructor
    CUTE_DEVICE BlkYLoopZeroOp(int lid, TO0 *BlkYLoopZeroOp_out)
    : lid(lid), BlkYLoopZeroOp_out(make_tensor(make_rmem_ptr(BlkYLoopZeroOp_out), Layout_o0{})) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return BlkYLoopZeroOp_out;
        }
    }
    
    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        clear(BlkYLoopZeroOp_out);
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
    int lid;

    // Constructor
    CUTE_DEVICE BlockDimYOp(int lid, TO0 *BlockDimYOp_out)
    : lid(lid), BlockDimYOp_out(make_tensor(make_rmem_ptr(BlockDimYOp_out), Layout_o0{})) {}

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
class BlkXLoopZeroOp {
public:
    // No Inputs

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Stride_o0 = decltype(make_stride(_0{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 BlkXLoopZeroOp_out;
    // Hardware parameters
    int lid;

    // Constructor
    CUTE_DEVICE BlkXLoopZeroOp(int lid, TO0 *BlkXLoopZeroOp_out)
    : lid(lid), BlkXLoopZeroOp_out(make_tensor(make_rmem_ptr(BlkXLoopZeroOp_out), Layout_o0{})) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return BlkXLoopZeroOp_out;
        }
    }
    
    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        clear(BlkXLoopZeroOp_out);
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
    int lid;

    // Constructor
    CUTE_DEVICE BlockDimXOp(int lid, TO0 *BlockDimXOp_out)
    : lid(lid), BlockDimXOp_out(make_tensor(make_rmem_ptr(BlockDimXOp_out), Layout_o0{})) {}

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
    int lid;

    // Constructor
    CUTE_DEVICE ZerosOp(int lid, TO0 *ZerosOp_out)
    : lid(lid), ZerosOp_out(make_tensor(make_rmem_ptr(ZerosOp_out), Layout_o0{})) {}

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

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class zero {
public:
    // No Inputs

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Stride_o0 = decltype(make_stride(_0{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 zero_out;
    // Hardware parameters
    int lid;

    // Constructor
    CUTE_DEVICE zero(int lid, TO0 *zero_out)
    : lid(lid), zero_out(make_tensor(make_rmem_ptr(zero_out), Layout_o0{})) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return zero_out;
        }
    }
    
    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        clear(zero_out);
    }
};

template<class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = int, class TI1 = int, class TO0 = int, class TO1 = int>
class G2rSparseOffsetLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 operand_0;
    // TI1 operand_1;

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
    int lid;

    CUTE_DEVICE G2rSparseOffsetLoadOp(int lid, TO0 *l, TO1 *r, int mode_i0_0)
    : lid(lid), shape_i0(make_shape(mode_i0_0)), l(make_tensor(make_rmem_ptr(l), Layout_o0{})), r(make_tensor(make_rmem_ptr(r), Layout_o1{})) {}

    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return l;
        } else if constexpr (output_idx == 1) {
            return r;
        }
    }

    template<class Tensor_i0, typename = void>
    CUTE_DEVICE void f(Tensor_i0 operand_0, TI1 operand_1) {
        l(0) = operand_0(operand_1);
        r(0) = operand_0(operand_1 + 1) - l(0);
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = int, class TI1 = int, class TI2 = int, class TO0 = int>
class G2sSparseIndexLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(get<2>(BLK_MNK{}), VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 operand_0;
    // TI1 off_l;
    // TI2 lc;

    // Outputs
    using Shape_o0 = decltype(make_shape(get<2>(BLK_MNK{}), NBUF{}));
    using Stride_o0 = decltype(make_stride(_1{}, get<2>(BLK_MNK{})));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 result_0;

    // Hardware parameters
    int lid;

    CUTE_DEVICE G2sSparseIndexLoadOp(int lid, TO0 *result_0, int mode_i0_0)
    : lid(lid), shape_i0(make_shape(get<2>(BLK_MNK{}), mode_i0_0)), result_0(make_tensor(make_smem_ptr(result_0), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return result_0(_, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return result_0;
        }
    }

    template<int buf_idx, class Tensor_i0, typename = void>
    CUTE_DEVICE void f(Tensor_i0 operand_0, TI1 off_l, TI2 lc) {
        int i = off_l + lc;
        if (lid < 2) {
            auto thr_tiler = make_shape(Int<4>{});
            auto thr_coord = make_coord(lid);
            auto src = local_tile(operand_0(_, i), thr_tiler, thr_coord);
            auto dst = local_tile(result_0(_, buf_idx), thr_tiler, thr_coord);
            copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, int>{}, src, dst);
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = int, class TI1 = int, class TI2 = int, class TI3 = int, class TO0 = int, class TO1 = int>
class G2sSparseCooIdxLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 operand_0;
    using Shape_i1 = decltype(make_shape(Int<2>{}, VARLEN));
    Shape_i1 shape_i1;
    // Tensor_i1 coo_off;
    // TI2 off_l;
    // TI3 lc;

    // Outputs
    using Shape_o0 = decltype(make_shape(get<0>(BLK_MNK{}) * get<2>(BLK_MNK{}), NBUF{}));
    using Stride_o0 = decltype(make_stride(_1{}, get<0>(BLK_MNK{}) * get<2>(BLK_MNK{})));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 coo_idx;
    using Shape_o1 = decltype(make_shape(_1{}, NBUF{}));
    using Stride_o1 = decltype(make_stride(_0{}, _1{}));
    using Layout_o1 = decltype(make_layout(Shape_o1{}, Stride_o1{}));
    using Tensor_o1 = decltype(make_tensor(make_rmem_ptr((TO1*)nullptr), Layout_o1{}));
    Shape_o1 shape_o1;
    Layout_o1 layout_o1;
    Tensor_o1 coo_range;

    // Hardware parameters
    int lid;

    CUTE_DEVICE G2sSparseCooIdxLoadOp(int lid, TO0 *coo_idx, TO1 *coo_range, int mode_i0_0, int mode_i1_0)
    : lid(lid), shape_i0(make_shape(mode_i0_0)), shape_i1(make_shape(Int<2>{}, mode_i1_0)), coo_idx(make_tensor(make_smem_ptr(coo_idx), Layout_o0{})), coo_range(make_tensor(make_rmem_ptr(coo_range), Layout_o1{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return coo_idx(_, buf_idx);
        } else if constexpr (output_idx == 1) {
            return coo_range(_, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return coo_idx;
        } else if constexpr (output_idx == 1) {
            return coo_range;
        }
    }

    template<int buf_idx, class Tensor_i0, class Tensor_i1, typename = void>
    CUTE_DEVICE void f(Tensor_i0 operand_0, Tensor_i1 coo_off, TI2 off_l, TI3 lc) {
        int idx = off_l + lc;
        int ll = coo_off(0, idx);
        int rr = coo_off(1, idx);
        coo_range(0, buf_idx) = rr - ll;
        
        auto thr_tiler = make_shape(_4{});
        auto input = make_tensor(operand_0.data() + ll, size<0>(shape(coo_idx)));
        for (int i_load = lid; i_load * 4 + ll < rr; i_load += 32) {
            auto thr_coord = make_coord(i_load);
            auto src = local_tile(input, thr_tiler, thr_coord);
            auto dst = local_tile(coo_idx(_, buf_idx), thr_tiler, thr_coord);
            copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, int>{}, src, dst);
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = float, class TI1 = int, class TI2 = int, class TI3 = int, class TO0 = float>
class G2sSparseCooValLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 operand_0;
    using Shape_i1 = decltype(make_shape(Int<2>{}, VARLEN));
    Shape_i1 shape_i1;
    // Tensor_i1 coo_off;
    // TI2 off_l;
    // TI3 lc;

    // Outputs
    using Shape_o0 = decltype(make_shape(get<0>(BLK_MNK{}) * get<2>(BLK_MNK{}), NBUF{}));
    using Stride_o0 = decltype(make_stride(_1{}, get<0>(BLK_MNK{}) * get<2>(BLK_MNK{})));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 coo_val;

    // Hardware parameters
    int lid;

    CUTE_DEVICE G2sSparseCooValLoadOp(int lid, TO0 *coo_val, int mode_i0_0, int mode_i1_0)
    : lid(lid), shape_i0(make_shape(mode_i0_0)), shape_i1(make_shape(Int<2>{}, mode_i1_0)), coo_val(make_tensor(make_smem_ptr(coo_val), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return coo_val(_, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return coo_val;
        }
    }

    template<int buf_idx, class Tensor_i0, class Tensor_i1, typename = void>
    CUTE_DEVICE void f(Tensor_i0 operand_0, Tensor_i1 coo_off, TI2 off_l, TI3 lc) {
        int idx = off_l + lc;
        int ll = coo_off(0, idx);
        int rr = coo_off(1, idx);
        
        auto thr_tiler = make_shape(_4{});
        auto input = make_tensor(operand_0.data() + ll, size<0>(shape(coo_val)));
        for (int i_load = lid; i_load * 4 + ll < rr; i_load += 32) {
            auto thr_coord = make_coord(i_load);
            auto src = local_tile(input, thr_tiler, thr_coord);
            auto dst = local_tile(coo_val(_, buf_idx), thr_tiler, thr_coord);
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
    // Tensor_i0 operand_0;
    using Shape_i1 = decltype(make_shape(get<2>(BLK_MNK{})));
    Shape_i1 shape_i1;
    // Tensor_i1 operand_1;

    // Outputs
    using Shape_o0 = decltype(make_shape(get<2>(BLK_MNK{}), get<1>(BLK_MNK{}), NBUF{}));
    using Stride_o0 = decltype(make_stride(get<1>(BLK_MNK{}), _1{}, get<2>(BLK_MNK{}) * get<1>(BLK_MNK{})));
    using Layout_o0 = decltype(composition(Swizzle<3, 2, 3>{}, make_layout(Shape_o0{}, Stride_o0{})));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 result_0;

    // Hardware parameters
    int lid;

    CUTE_DEVICE G2sMatrixBLoadOp(int lid, TO0 *result_0, int mode_i0_0, int mode_i0_1)
    : lid(lid), shape_i0(make_shape(mode_i0_0, mode_i0_1)), result_0(make_tensor(make_smem_ptr(result_0), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return result_0(_, _, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return result_0;
        }
    }

    template<int buf_idx, class Tensor_i0, class Tensor_i1, typename = void>
    CUTE_DEVICE void f(Tensor_i0 operand_0, Tensor_i1 operand_1) {
        __syncthreads();
        auto load_tile_n = _4{};
        for (int iter_i = lid; iter_i < get<2>(BLK_MNK{}) * get<1>(BLK_MNK{}) / load_tile_n; iter_i += 32) {
            int row = iter_i / (get<1>(BLK_MNK{}) / load_tile_n);
            int col = iter_i % (get<1>(BLK_MNK{}) / load_tile_n);
            int sidx = operand_1(row);
            auto thr_tiler_gmem = make_shape(_1{}, _4{});
            auto thr_coord_gmem = make_coord(sidx, col);
            auto B_val_thr = local_tile(operand_0, thr_tiler_gmem, thr_coord_gmem);
            auto thr_tiler_smem = make_shape(_1{}, _4{});
            auto thr_coord_smem = make_coord(row, col);
            auto dst = local_tile(result_0(_, _, buf_idx), thr_tiler_smem, thr_coord_smem);
            copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{}, B_val_thr(0, _), dst(0, _));
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = int, class TI1 = float, class TI2 = int, class TO0 = float>
class S2sRestoreMatrixAOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(get<0>(BLK_MNK{}) * get<2>(BLK_MNK{})));
    Shape_i0 shape_i0;
    // Tensor_i0 tileA_coo_idx;
    using Shape_i1 = decltype(make_shape(get<0>(BLK_MNK{}) * get<2>(BLK_MNK{})));
    Shape_i1 shape_i1;
    // Tensor_i1 tileA_coo_val;
    // TI2 nnz_num;

    // Outputs
    using Shape_o0 = decltype(make_shape(get<0>(BLK_MNK{}), get<2>(BLK_MNK{}), NBUF{}));
    using Stride_o0 = decltype(make_stride(get<2>(BLK_MNK{}), _1{}, get<0>(BLK_MNK{}) * get<2>(BLK_MNK{})));
    using Layout_o0 = decltype(composition(Swizzle<1, 2, (get<2>(BLK_MNK{}) == Int<8>{}) ? 3 : ((get<2>(BLK_MNK{}) == Int<16>{}) ? 4 : ((get<2>(BLK_MNK{}) == Int<32>{}) ? 5 : ((get<2>(BLK_MNK{}) == Int<64>{}) ? 6 : 7)))>{}, make_layout(Shape_o0{}, Stride_o0{})));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 A_val_tile;

    // Hardware parameters
    int lid;

    CUTE_DEVICE S2sRestoreMatrixAOp(int lid, TO0 *A_val_tile)
    : lid(lid), A_val_tile(make_tensor(make_smem_ptr(A_val_tile), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return A_val_tile(_, _, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return A_val_tile;
        }
    }

    template<int buf_idx, class Tensor_i0, class Tensor_i1, typename = void>
    CUTE_DEVICE void f(Tensor_i0 tileA_coo_idx, Tensor_i1 tileA_coo_val, TI2 nnz_num) {
        for (int i_o2s = lid; i_o2s < get<0>(shape(A_val_tile)) * get<1>(shape(A_val_tile)); i_o2s += 32) {
            *((float*)(A_val_tile(_, _, buf_idx).data().get() + i_o2s)) = float(0);
        }
        __syncthreads();
        for (int i_restore = lid; i_restore < nnz_num; i_restore += 32) {
            float value = tileA_coo_val(i_restore);
            int idx = tileA_coo_idx(i_restore);
            *((float*)(A_val_tile(_, _, buf_idx).data().get() + idx)) = value;
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = float, class TO0 = float>
class S2rAValLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(get<0>(BLK_MNK{}), get<2>(BLK_MNK{})));
    Shape_i0 shape_i0;
    // Tensor_i0 tileA;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<2>(BLK_MMA_MNK{})), NBUF{}));
    using Stride_o0 = decltype(make_stride(_1{}, make_stride(Int<4>{}, Int<4>{} * get<0>(BLK_MMA_MNK{})), Int<4>{} * _1{} * get<0>(BLK_MMA_MNK{}) * get<2>(BLK_MMA_MNK{})));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 REGA;

    // Hardware parameters
    int lid;

    CUTE_DEVICE S2rAValLoadOp(int lid, TO0 *REGA)
    : lid(lid), REGA(make_tensor(make_rmem_ptr(REGA), Layout_o0{})) {}

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
    CUTE_DEVICE void f(Tensor_i0 tileA) {
        __syncthreads();
        int row = lid % 16;
        int col = lid / 16;
        ldmatrix_m8n8k8_x4(
            (uint32_t*)(REGA(_, make_coord(_0{}, _0{}), buf_idx).data()),
            (void*)(&tileA(row, col * 4))
        );
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = float, class TO0 = float>
class S2rBValLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(get<2>(BLK_MNK{}), VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 tileB;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<2>{}, make_shape(get<2>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{})), NBUF{}));
    using Stride_o0 = decltype(make_stride(_1{}, make_stride(Int<2>{}, Int<2>{} * get<2>(BLK_MMA_MNK{})), Int<2>{} * _1{} * get<2>(BLK_MMA_MNK{}) * get<1>(BLK_MMA_MNK{})));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 REGB;

    // Hardware parameters
    int lid;

    CUTE_DEVICE S2rBValLoadOp(int lid, TO0 *REGB, int mode_i0_0)
    : lid(lid), shape_i0(make_shape(get<2>(BLK_MNK{}), mode_i0_0)), REGB(make_tensor(make_rmem_ptr(REGB), Layout_o0{})) {}

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
    CUTE_DEVICE void f(Tensor_i0 tileB) {
        __syncthreads();
        for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
            int row_b = lid / 2;
            int col_b = lid % 2;
            ldmatrix_m8n8k8_x2(
                (uint32_t*)(REGB(_, make_coord(_0{}, n_iter), buf_idx).data()),
                (void*)(&tileB(row_b, col_b * 4 + n_iter * 8))
            );
            REGB(_0{}, make_coord(_0{}, n_iter), buf_idx) = __shfl_sync(0xffffffff, REGB(_0{}, make_coord(_0{}, n_iter), buf_idx), lid / 4 + lid % 4 * 8);
            REGB(_1{}, make_coord(_0{}, n_iter), buf_idx) = __shfl_sync(0xffffffff, REGB(_1{}, make_coord(_0{}, n_iter), buf_idx), lid / 4 + lid % 4 * 8);
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class Tensor_o0, class TI0 = float, class TI1 = float, class TI2 = float, class TO0 = float>
class CalculateOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<2>(BLK_MMA_MNK{}))));
    Shape_i0 shape_i0;
    // Tensor_i0 operand_0;
    using Shape_i1 = decltype(make_shape(Int<2>{}, make_shape(get<2>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{}))));
    Shape_i1 shape_i1;
    // Tensor_i1 operand_1;
    using Shape_i2 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{}))));
    Shape_i2 shape_i2;
    // Tensor_i2 operand_2;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{}))));
    Shape_o0 shape_o0;
    Tensor_o0 C_val_tile;

    // Hardware parameters
    int lid;

    CUTE_DEVICE CalculateOp(int lid, Tensor_o0 C_val_tile)
    : lid(lid), C_val_tile(C_val_tile) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return C_val_tile;
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return C_val_tile;
        }
    }

    template<int buf_idx, class Tensor_i0, class Tensor_i1, class Tensor_i2, typename = void>
    CUTE_DEVICE void f(Tensor_i0 operand_0, Tensor_i1 operand_1, Tensor_i2 operand_2) {
        for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) {
            for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
                uint32_t frag_A[4];
                uint32_t frag_B[2];
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(operand_0(0, make_coord(m_iter, _0{}))));
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(operand_0(1, make_coord(m_iter, _0{}))));
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(operand_0(2, make_coord(m_iter, _0{}))));
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(operand_0(3, make_coord(m_iter, _0{}))));
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(operand_1(0, make_coord(_0{}, n_iter))));
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(operand_1(1, make_coord(_0{}, n_iter))));
        
                mma_m16n8k8_fp32_tf32_tf32_fp32(C_val_tile(_, make_coord(m_iter, n_iter)).data(), frag_A, frag_B);
            }
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class T_B_valOp, class T_C_valOp, class T_val_coo_idx_newOp, class T_val_coo_off_newOp, class T_val_coo_val_newOp, class T_val_len_offsetOp, class T_val_sidx_newOp, class T_BlkYLoopZeroOp, class T_BlockDimYOp, class T_BlkXLoopZeroOp, class T_BlockDimXOp, class T_ZerosOp, class T_zero, class T_G2rSparseOffsetLoadOp, class T_G2sSparseIndexLoadOp, class T_G2sSparseCooIdxLoadOp, class T_G2sSparseCooValLoadOp, class T_G2sMatrixBLoadOp, class T_S2sRestoreMatrixAOp, class T_S2rAValLoadOp, class T_S2rBValLoadOp, class T_CalculateOp, class Tensor_o0, class TI0 = int, class TI1 = int, class TI2 = int, class TO0 = float>
class for_loop {
public:
    // Inputs
    int _l, _r;
    int _idx; // i + _l
    int _iter; // i
    TI2 off_l;


    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{}))));
    Shape_o0 shape_o0;
    Tensor_o0 for_loop_result_0;

    // Hardware parameters
    int lid;

    // Body Ops
    T_B_valOp &B_valOp_v;
    T_C_valOp &C_valOp_v;
    T_val_coo_idx_newOp &val_coo_idx_newOp_v;
    T_val_coo_off_newOp &val_coo_off_newOp_v;
    T_val_coo_val_newOp &val_coo_val_newOp_v;
    T_val_len_offsetOp &val_len_offsetOp_v;
    T_val_sidx_newOp &val_sidx_newOp_v;
    T_BlkYLoopZeroOp &BlkYLoopZeroOp_v;
    T_BlockDimYOp &BlockDimYOp_v;
    T_BlkXLoopZeroOp &BlkXLoopZeroOp_v;
    T_BlockDimXOp &BlockDimXOp_v;
    T_ZerosOp &ZerosOp_v;
    T_zero &zero_v;
    T_G2rSparseOffsetLoadOp &G2rSparseOffsetLoadOp_v;
    T_G2sSparseIndexLoadOp &G2sSparseIndexLoadOp_v;
    T_G2sSparseCooIdxLoadOp &G2sSparseCooIdxLoadOp_v;
    T_G2sSparseCooValLoadOp &G2sSparseCooValLoadOp_v;
    T_G2sMatrixBLoadOp &G2sMatrixBLoadOp_v;
    T_S2sRestoreMatrixAOp &S2sRestoreMatrixAOp_v;
    T_S2rAValLoadOp &S2rAValLoadOp_v;
    T_S2rBValLoadOp &S2rBValLoadOp_v;
    T_CalculateOp &CalculateOp_v;

    // Constructor
    CUTE_DEVICE for_loop(int lid, T_B_valOp &B_valOp_v, T_C_valOp &C_valOp_v, T_val_coo_idx_newOp &val_coo_idx_newOp_v, T_val_coo_off_newOp &val_coo_off_newOp_v, T_val_coo_val_newOp &val_coo_val_newOp_v, T_val_len_offsetOp &val_len_offsetOp_v, T_val_sidx_newOp &val_sidx_newOp_v, T_BlkYLoopZeroOp &BlkYLoopZeroOp_v, T_BlockDimYOp &BlockDimYOp_v, T_BlkXLoopZeroOp &BlkXLoopZeroOp_v, T_BlockDimXOp &BlockDimXOp_v, T_ZerosOp &ZerosOp_v, T_zero &zero_v, T_G2rSparseOffsetLoadOp &G2rSparseOffsetLoadOp_v, T_G2sSparseIndexLoadOp &G2sSparseIndexLoadOp_v, T_G2sSparseCooIdxLoadOp &G2sSparseCooIdxLoadOp_v, T_G2sSparseCooValLoadOp &G2sSparseCooValLoadOp_v, T_G2sMatrixBLoadOp &G2sMatrixBLoadOp_v, T_S2sRestoreMatrixAOp &S2sRestoreMatrixAOp_v, T_S2rAValLoadOp &S2rAValLoadOp_v, T_S2rBValLoadOp &S2rBValLoadOp_v, T_CalculateOp &CalculateOp_v, Tensor_o0 for_loop_result_0)
    : lid(lid), B_valOp_v(B_valOp_v), C_valOp_v(C_valOp_v), val_coo_idx_newOp_v(val_coo_idx_newOp_v), val_coo_off_newOp_v(val_coo_off_newOp_v), val_coo_val_newOp_v(val_coo_val_newOp_v), val_len_offsetOp_v(val_len_offsetOp_v), val_sidx_newOp_v(val_sidx_newOp_v), BlkYLoopZeroOp_v(BlkYLoopZeroOp_v), BlockDimYOp_v(BlockDimYOp_v), BlkXLoopZeroOp_v(BlkXLoopZeroOp_v), BlockDimXOp_v(BlockDimXOp_v), ZerosOp_v(ZerosOp_v), zero_v(zero_v), G2rSparseOffsetLoadOp_v(G2rSparseOffsetLoadOp_v), G2sSparseIndexLoadOp_v(G2sSparseIndexLoadOp_v), G2sSparseCooIdxLoadOp_v(G2sSparseCooIdxLoadOp_v), G2sSparseCooValLoadOp_v(G2sSparseCooValLoadOp_v), G2sMatrixBLoadOp_v(G2sMatrixBLoadOp_v), S2sRestoreMatrixAOp_v(S2sRestoreMatrixAOp_v), S2rAValLoadOp_v(S2rAValLoadOp_v), S2rBValLoadOp_v(S2rBValLoadOp_v), CalculateOp_v(CalculateOp_v), for_loop_result_0(for_loop_result_0) {}

    // Pipeline methods
    // short pipe
    CUTE_DEVICE void short_pipe(int k, int l, int r) {
        if (k == 0) {
            // i = 0
            
            
            // i = 1
            
        }
        else if (k == 1) {
            // i = 0
            G2sSparseIndexLoadOp_v.template f<0>(
                val_sidx_newOp_v.template output<0>(),
                off_l,
                l + 0
            );
            __pipeline_commit();
            G2sSparseCooIdxLoadOp_v.template f<0>(
                val_coo_idx_newOp_v.template output<0>(),
                val_coo_off_newOp_v.template output<0>(),
                off_l,
                l + 0
            );
            __pipeline_commit();
            G2sSparseCooValLoadOp_v.template f<0>(
                val_coo_val_newOp_v.template output<0>(),
                val_coo_off_newOp_v.template output<0>(),
                off_l,
                l + 0
            );
            __pipeline_commit();
            
            
            // i = 1
            __pipeline_wait_prior(2);
            __syncthreads();
            G2sMatrixBLoadOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                G2sSparseIndexLoadOp_v.template output<0, 0>()
            );
            __pipeline_commit();
            __pipeline_wait_prior(1);
            __syncthreads();
            S2sRestoreMatrixAOp_v.template f<0>(
                G2sSparseCooIdxLoadOp_v.template output<0, 0>(),
                G2sSparseCooValLoadOp_v.template output<0, 0>(),
                G2sSparseCooIdxLoadOp_v.template output<1, 0>()(0)
            );
            
            
            // i = 2
            __syncthreads();
            S2rAValLoadOp_v.template f<0>(
                S2sRestoreMatrixAOp_v.template output<0, 0>()
            );
            __pipeline_wait_prior(0);
            __syncthreads();
            S2rBValLoadOp_v.template f<0>(
                G2sMatrixBLoadOp_v.template output<0, 0>()
            );
            CalculateOp_v.template f<0>(
                S2rAValLoadOp_v.template output<0, 0>(),
                S2rBValLoadOp_v.template output<0, 0>(),
                ZerosOp_v.template output<0>()
            );
            
        }
        else if (k == 2) {
            // i = 0
            G2sSparseIndexLoadOp_v.template f<0>(
                val_sidx_newOp_v.template output<0>(),
                off_l,
                l + 0
            );
            __pipeline_commit();
            G2sSparseCooIdxLoadOp_v.template f<0>(
                val_coo_idx_newOp_v.template output<0>(),
                val_coo_off_newOp_v.template output<0>(),
                off_l,
                l + 0
            );
            __pipeline_commit();
            G2sSparseCooValLoadOp_v.template f<0>(
                val_coo_val_newOp_v.template output<0>(),
                val_coo_off_newOp_v.template output<0>(),
                off_l,
                l + 0
            );
            __pipeline_commit();
            
            
            // i = 1
            G2sSparseIndexLoadOp_v.template f<1>(
                val_sidx_newOp_v.template output<0>(),
                off_l,
                l + 1
            );
            __pipeline_commit();
            G2sSparseCooIdxLoadOp_v.template f<1>(
                val_coo_idx_newOp_v.template output<0>(),
                val_coo_off_newOp_v.template output<0>(),
                off_l,
                l + 1
            );
            __pipeline_commit();
            G2sSparseCooValLoadOp_v.template f<1>(
                val_coo_val_newOp_v.template output<0>(),
                val_coo_off_newOp_v.template output<0>(),
                off_l,
                l + 1
            );
            __pipeline_commit();
            __pipeline_wait_prior(5);
            __syncthreads();
            G2sMatrixBLoadOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                G2sSparseIndexLoadOp_v.template output<0, 0>()
            );
            __pipeline_commit();
            __pipeline_wait_prior(4);
            __syncthreads();
            S2sRestoreMatrixAOp_v.template f<0>(
                G2sSparseCooIdxLoadOp_v.template output<0, 0>(),
                G2sSparseCooValLoadOp_v.template output<0, 0>(),
                G2sSparseCooIdxLoadOp_v.template output<1, 0>()(0)
            );
            
            
            // i = 2
            __pipeline_wait_prior(3);
            __syncthreads();
            G2sMatrixBLoadOp_v.template f<1>(
                B_valOp_v.template output<0>(),
                G2sSparseIndexLoadOp_v.template output<0, 1>()
            );
            __pipeline_commit();
            __pipeline_wait_prior(2);
            __syncthreads();
            S2sRestoreMatrixAOp_v.template f<1>(
                G2sSparseCooIdxLoadOp_v.template output<0, 1>(),
                G2sSparseCooValLoadOp_v.template output<0, 1>(),
                G2sSparseCooIdxLoadOp_v.template output<1, 1>()(0)
            );
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
                S2rBValLoadOp_v.template output<0, 0>(),
                ZerosOp_v.template output<0>()
            );
            
            
            // i = 3
            __syncthreads();
            S2rAValLoadOp_v.template f<1>(
                S2sRestoreMatrixAOp_v.template output<0, 1>()
            );
            __pipeline_wait_prior(0);
            __syncthreads();
            S2rBValLoadOp_v.template f<1>(
                G2sMatrixBLoadOp_v.template output<0, 1>()
            );
            CalculateOp_v.template f<1>(
                S2rAValLoadOp_v.template output<0, 1>(),
                S2rBValLoadOp_v.template output<0, 1>(),
                ZerosOp_v.template output<0>()
            );
            
        }
        else if (k == 3) {
            // i = 0
            G2sSparseIndexLoadOp_v.template f<0>(
                val_sidx_newOp_v.template output<0>(),
                off_l,
                l + 0
            );
            __pipeline_commit();
            G2sSparseCooIdxLoadOp_v.template f<0>(
                val_coo_idx_newOp_v.template output<0>(),
                val_coo_off_newOp_v.template output<0>(),
                off_l,
                l + 0
            );
            __pipeline_commit();
            G2sSparseCooValLoadOp_v.template f<0>(
                val_coo_val_newOp_v.template output<0>(),
                val_coo_off_newOp_v.template output<0>(),
                off_l,
                l + 0
            );
            __pipeline_commit();
            
            
            // i = 1
            G2sSparseIndexLoadOp_v.template f<1>(
                val_sidx_newOp_v.template output<0>(),
                off_l,
                l + 1
            );
            __pipeline_commit();
            G2sSparseCooIdxLoadOp_v.template f<1>(
                val_coo_idx_newOp_v.template output<0>(),
                val_coo_off_newOp_v.template output<0>(),
                off_l,
                l + 1
            );
            __pipeline_commit();
            G2sSparseCooValLoadOp_v.template f<1>(
                val_coo_val_newOp_v.template output<0>(),
                val_coo_off_newOp_v.template output<0>(),
                off_l,
                l + 1
            );
            __pipeline_commit();
            __pipeline_wait_prior(5);
            __syncthreads();
            G2sMatrixBLoadOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                G2sSparseIndexLoadOp_v.template output<0, 0>()
            );
            __pipeline_commit();
            __pipeline_wait_prior(4);
            __syncthreads();
            S2sRestoreMatrixAOp_v.template f<0>(
                G2sSparseCooIdxLoadOp_v.template output<0, 0>(),
                G2sSparseCooValLoadOp_v.template output<0, 0>(),
                G2sSparseCooIdxLoadOp_v.template output<1, 0>()(0)
            );
            
            
            // i = 2
            G2sSparseIndexLoadOp_v.template f<0>(
                val_sidx_newOp_v.template output<0>(),
                off_l,
                l + 2
            );
            __pipeline_commit();
            G2sSparseCooIdxLoadOp_v.template f<0>(
                val_coo_idx_newOp_v.template output<0>(),
                val_coo_off_newOp_v.template output<0>(),
                off_l,
                l + 2
            );
            __pipeline_commit();
            G2sSparseCooValLoadOp_v.template f<0>(
                val_coo_val_newOp_v.template output<0>(),
                val_coo_off_newOp_v.template output<0>(),
                off_l,
                l + 2
            );
            __pipeline_commit();
            __pipeline_wait_prior(6);
            __syncthreads();
            G2sMatrixBLoadOp_v.template f<1>(
                B_valOp_v.template output<0>(),
                G2sSparseIndexLoadOp_v.template output<0, 1>()
            );
            __pipeline_commit();
            __pipeline_wait_prior(5);
            __syncthreads();
            S2sRestoreMatrixAOp_v.template f<1>(
                G2sSparseCooIdxLoadOp_v.template output<0, 1>(),
                G2sSparseCooValLoadOp_v.template output<0, 1>(),
                G2sSparseCooIdxLoadOp_v.template output<1, 1>()(0)
            );
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
                S2rBValLoadOp_v.template output<0, 0>(),
                ZerosOp_v.template output<0>()
            );
            
            
            // i = 3
            __pipeline_wait_prior(3);
            __syncthreads();
            G2sMatrixBLoadOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                G2sSparseIndexLoadOp_v.template output<0, 0>()
            );
            __pipeline_commit();
            __pipeline_wait_prior(2);
            __syncthreads();
            S2sRestoreMatrixAOp_v.template f<0>(
                G2sSparseCooIdxLoadOp_v.template output<0, 0>(),
                G2sSparseCooValLoadOp_v.template output<0, 0>(),
                G2sSparseCooIdxLoadOp_v.template output<1, 0>()(0)
            );
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
                S2rBValLoadOp_v.template output<0, 1>(),
                ZerosOp_v.template output<0>()
            );
            
            
            // i = 4
            __syncthreads();
            S2rAValLoadOp_v.template f<0>(
                S2sRestoreMatrixAOp_v.template output<0, 0>()
            );
            __pipeline_wait_prior(0);
            __syncthreads();
            S2rBValLoadOp_v.template f<0>(
                G2sMatrixBLoadOp_v.template output<0, 0>()
            );
            CalculateOp_v.template f<0>(
                S2rAValLoadOp_v.template output<0, 0>(),
                S2rBValLoadOp_v.template output<0, 0>(),
                ZerosOp_v.template output<0>()
            );
            
        }
        
    }
    
    // fill
    CUTE_DEVICE void fill(int l, int r) {
        // stage 0..0
        // op_call |buf_idx=0| 0 G2sSparseIndexLoadOp 
        G2sSparseIndexLoadOp_v.template f<0>(
            val_sidx_newOp_v.template output<0>(),
            off_l,
            l + 0
        );
        __pipeline_commit();
        // op_call |buf_idx=0| 0 G2sSparseCooIdxLoadOp 
        G2sSparseCooIdxLoadOp_v.template f<0>(
            val_coo_idx_newOp_v.template output<0>(),
            val_coo_off_newOp_v.template output<0>(),
            off_l,
            l + 0
        );
        __pipeline_commit();
        // op_call |buf_idx=0| 0 G2sSparseCooValLoadOp 
        G2sSparseCooValLoadOp_v.template f<0>(
            val_coo_val_newOp_v.template output<0>(),
            val_coo_off_newOp_v.template output<0>(),
            off_l,
            l + 0
        );
        __pipeline_commit();
        // stage 0..1
        // op_call |buf_idx=1| 1 G2sSparseIndexLoadOp 
        G2sSparseIndexLoadOp_v.template f<1>(
            val_sidx_newOp_v.template output<0>(),
            off_l,
            l + 1
        );
        __pipeline_commit();
        // op_call |buf_idx=1| 1 G2sSparseCooIdxLoadOp 
        G2sSparseCooIdxLoadOp_v.template f<1>(
            val_coo_idx_newOp_v.template output<0>(),
            val_coo_off_newOp_v.template output<0>(),
            off_l,
            l + 1
        );
        __pipeline_commit();
        // op_call |buf_idx=1| 1 G2sSparseCooValLoadOp 
        G2sSparseCooValLoadOp_v.template f<1>(
            val_coo_val_newOp_v.template output<0>(),
            val_coo_off_newOp_v.template output<0>(),
            off_l,
            l + 1
        );
        __pipeline_commit();
        __pipeline_wait_prior(5);
        // op_call |buf_idx=0| 0 G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            G2sSparseIndexLoadOp_v.template output<0, 0>()
        );
        __pipeline_commit();
        __pipeline_wait_prior(4);
        // op_call |buf_idx=0| 0 S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<0>(
            G2sSparseCooIdxLoadOp_v.template output<0, 0>(),
            G2sSparseCooValLoadOp_v.template output<0, 0>(),
            G2sSparseCooIdxLoadOp_v.template output<1, 0>()(0)
        );
        // stage 0..2
        // op_call |buf_idx=0| 2 G2sSparseIndexLoadOp 
        G2sSparseIndexLoadOp_v.template f<0>(
            val_sidx_newOp_v.template output<0>(),
            off_l,
            l + 2
        );
        __pipeline_commit();
        // op_call |buf_idx=0| 2 G2sSparseCooIdxLoadOp 
        G2sSparseCooIdxLoadOp_v.template f<0>(
            val_coo_idx_newOp_v.template output<0>(),
            val_coo_off_newOp_v.template output<0>(),
            off_l,
            l + 2
        );
        __pipeline_commit();
        // op_call |buf_idx=0| 2 G2sSparseCooValLoadOp 
        G2sSparseCooValLoadOp_v.template f<0>(
            val_coo_val_newOp_v.template output<0>(),
            val_coo_off_newOp_v.template output<0>(),
            off_l,
            l + 2
        );
        __pipeline_commit();
        __pipeline_wait_prior(6);
        // op_call |buf_idx=1| 1 G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<1>(
            B_valOp_v.template output<0>(),
            G2sSparseIndexLoadOp_v.template output<0, 1>()
        );
        __pipeline_commit();
        __pipeline_wait_prior(5);
        // op_call |buf_idx=1| 1 S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<1>(
            G2sSparseCooIdxLoadOp_v.template output<0, 1>(),
            G2sSparseCooValLoadOp_v.template output<0, 1>(),
            G2sSparseCooIdxLoadOp_v.template output<1, 1>()(0)
        );
        __syncthreads();
        // op_call |buf_idx=0| 0 S2rAValLoadOp 
        S2rAValLoadOp_v.template f<0>(
            S2sRestoreMatrixAOp_v.template output<0, 0>()
        );
        __pipeline_wait_prior(4);
        // op_call |buf_idx=0| 0 S2rBValLoadOp 
        S2rBValLoadOp_v.template f<0>(
            G2sMatrixBLoadOp_v.template output<0, 0>()
        );
        // op_call |buf_idx=0| 0 CalculateOp 
        CalculateOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            S2rBValLoadOp_v.template output<0, 0>(),
            ZerosOp_v.template output<0>()
        );
    }
    
    // loop_step
    CUTE_DEVICE void loop_step(int i, int l, int r) {
        // step 0
        // op_call |buf_idx=1| i + (0) G2sSparseIndexLoadOp 
        G2sSparseIndexLoadOp_v.template f<1>(
            val_sidx_newOp_v.template output<0>(),
            off_l,
            l + i + (0)
        );
        __pipeline_commit();
        // op_call |buf_idx=1| i + (0) G2sSparseCooIdxLoadOp 
        G2sSparseCooIdxLoadOp_v.template f<1>(
            val_coo_idx_newOp_v.template output<0>(),
            val_coo_off_newOp_v.template output<0>(),
            off_l,
            l + i + (0)
        );
        __pipeline_commit();
        // op_call |buf_idx=1| i + (0) G2sSparseCooValLoadOp 
        G2sSparseCooValLoadOp_v.template f<1>(
            val_coo_val_newOp_v.template output<0>(),
            val_coo_off_newOp_v.template output<0>(),
            off_l,
            l + i + (0)
        );
        __pipeline_commit();
        __pipeline_wait_prior(6);
        // op_call |buf_idx=0| i + (-1) G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            G2sSparseIndexLoadOp_v.template output<0, 0>()
        );
        __pipeline_commit();
        __pipeline_wait_prior(5);
        // op_call |buf_idx=0| i + (-1) S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<0>(
            G2sSparseCooIdxLoadOp_v.template output<0, 0>(),
            G2sSparseCooValLoadOp_v.template output<0, 0>(),
            G2sSparseCooIdxLoadOp_v.template output<1, 0>()(0)
        );
        __syncthreads();
        // op_call |buf_idx=1| i + (-2) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<1>(
            S2sRestoreMatrixAOp_v.template output<0, 1>()
        );
        __pipeline_wait_prior(4);
        // op_call |buf_idx=1| i + (-2) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<1>(
            G2sMatrixBLoadOp_v.template output<0, 1>()
        );
        // op_call |buf_idx=1| i + (-2) CalculateOp 
        CalculateOp_v.template f<1>(
            S2rAValLoadOp_v.template output<0, 1>(),
            S2rBValLoadOp_v.template output<0, 1>(),
            ZerosOp_v.template output<0>()
        );
        // step 1
        // op_call |buf_idx=0| i + (1) G2sSparseIndexLoadOp 
        G2sSparseIndexLoadOp_v.template f<0>(
            val_sidx_newOp_v.template output<0>(),
            off_l,
            l + i + (1)
        );
        __pipeline_commit();
        // op_call |buf_idx=0| i + (1) G2sSparseCooIdxLoadOp 
        G2sSparseCooIdxLoadOp_v.template f<0>(
            val_coo_idx_newOp_v.template output<0>(),
            val_coo_off_newOp_v.template output<0>(),
            off_l,
            l + i + (1)
        );
        __pipeline_commit();
        // op_call |buf_idx=0| i + (1) G2sSparseCooValLoadOp 
        G2sSparseCooValLoadOp_v.template f<0>(
            val_coo_val_newOp_v.template output<0>(),
            val_coo_off_newOp_v.template output<0>(),
            off_l,
            l + i + (1)
        );
        __pipeline_commit();
        __pipeline_wait_prior(6);
        // op_call |buf_idx=1| i + (0) G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<1>(
            B_valOp_v.template output<0>(),
            G2sSparseIndexLoadOp_v.template output<0, 1>()
        );
        __pipeline_commit();
        __pipeline_wait_prior(5);
        // op_call |buf_idx=1| i + (0) S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<1>(
            G2sSparseCooIdxLoadOp_v.template output<0, 1>(),
            G2sSparseCooValLoadOp_v.template output<0, 1>(),
            G2sSparseCooIdxLoadOp_v.template output<1, 1>()(0)
        );
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<0>(
            S2sRestoreMatrixAOp_v.template output<0, 0>()
        );
        __pipeline_wait_prior(4);
        // op_call |buf_idx=0| i + (-1) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<0>(
            G2sMatrixBLoadOp_v.template output<0, 0>()
        );
        // op_call |buf_idx=0| i + (-1) CalculateOp 
        CalculateOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            S2rBValLoadOp_v.template output<0, 0>(),
            ZerosOp_v.template output<0>()
        );
    }
    
    // remainder
    CUTE_DEVICE void remains_1(int i, int l, int r) {
        // total remains 1, current 0
        // op_call |buf_idx=1| i + (0) G2sSparseIndexLoadOp 
        G2sSparseIndexLoadOp_v.template f<1>(
            val_sidx_newOp_v.template output<0>(),
            off_l,
            l + i + (0)
        );
        __pipeline_commit();
        // op_call |buf_idx=1| i + (0) G2sSparseCooIdxLoadOp 
        G2sSparseCooIdxLoadOp_v.template f<1>(
            val_coo_idx_newOp_v.template output<0>(),
            val_coo_off_newOp_v.template output<0>(),
            off_l,
            l + i + (0)
        );
        __pipeline_commit();
        // op_call |buf_idx=1| i + (0) G2sSparseCooValLoadOp 
        G2sSparseCooValLoadOp_v.template f<1>(
            val_coo_val_newOp_v.template output<0>(),
            val_coo_off_newOp_v.template output<0>(),
            off_l,
            l + i + (0)
        );
        __pipeline_commit();
        __pipeline_wait_prior(6);
        // op_call |buf_idx=0| i + (-1) G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            G2sSparseIndexLoadOp_v.template output<0, 0>()
        );
        __pipeline_commit();
        __pipeline_wait_prior(5);
        // op_call |buf_idx=0| i + (-1) S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<0>(
            G2sSparseCooIdxLoadOp_v.template output<0, 0>(),
            G2sSparseCooValLoadOp_v.template output<0, 0>(),
            G2sSparseCooIdxLoadOp_v.template output<1, 0>()(0)
        );
        __syncthreads();
        // op_call |buf_idx=1| i + (-2) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<1>(
            S2sRestoreMatrixAOp_v.template output<0, 1>()
        );
        __pipeline_wait_prior(4);
        // op_call |buf_idx=1| i + (-2) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<1>(
            G2sMatrixBLoadOp_v.template output<0, 1>()
        );
        // op_call |buf_idx=1| i + (-2) CalculateOp 
        CalculateOp_v.template f<1>(
            S2rAValLoadOp_v.template output<0, 1>(),
            S2rBValLoadOp_v.template output<0, 1>(),
            ZerosOp_v.template output<0>()
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
        __pipeline_wait_prior(3);
        // op_call |buf_idx=0| i + (-1) G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            G2sSparseIndexLoadOp_v.template output<0, 0>()
        );
        __pipeline_commit();
        __pipeline_wait_prior(2);
        // op_call |buf_idx=0| i + (-1) S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<0>(
            G2sSparseCooIdxLoadOp_v.template output<0, 0>(),
            G2sSparseCooValLoadOp_v.template output<0, 0>(),
            G2sSparseCooIdxLoadOp_v.template output<1, 0>()(0)
        );
        __syncthreads();
        // op_call |buf_idx=1| i + (-2) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<1>(
            S2sRestoreMatrixAOp_v.template output<0, 1>()
        );
        __pipeline_wait_prior(1);
        // op_call |buf_idx=1| i + (-2) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<1>(
            G2sMatrixBLoadOp_v.template output<0, 1>()
        );
        // op_call |buf_idx=1| i + (-2) CalculateOp 
        CalculateOp_v.template f<1>(
            S2rAValLoadOp_v.template output<0, 1>(),
            S2rBValLoadOp_v.template output<0, 1>(),
            ZerosOp_v.template output<0>()
        );
        // stage 2..2
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<0>(
            S2sRestoreMatrixAOp_v.template output<0, 0>()
        );
        __pipeline_wait_prior(0);
        // op_call |buf_idx=0| i + (-1) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<0>(
            G2sMatrixBLoadOp_v.template output<0, 0>()
        );
        // op_call |buf_idx=0| i + (-1) CalculateOp 
        CalculateOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            S2rBValLoadOp_v.template output<0, 0>(),
            ZerosOp_v.template output<0>()
        );
    }
    
    CUTE_DEVICE void empty_after_remain_1(int i, int l, int r) {
        // stage 1..2
        __pipeline_wait_prior(3);
        // op_call |buf_idx=1| i + (0) G2sMatrixBLoadOp 
        G2sMatrixBLoadOp_v.template f<1>(
            B_valOp_v.template output<0>(),
            G2sSparseIndexLoadOp_v.template output<0, 1>()
        );
        __pipeline_commit();
        __pipeline_wait_prior(2);
        // op_call |buf_idx=1| i + (0) S2sRestoreMatrixAOp 
        S2sRestoreMatrixAOp_v.template f<1>(
            G2sSparseCooIdxLoadOp_v.template output<0, 1>(),
            G2sSparseCooValLoadOp_v.template output<0, 1>(),
            G2sSparseCooIdxLoadOp_v.template output<1, 1>()(0)
        );
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<0>(
            S2sRestoreMatrixAOp_v.template output<0, 0>()
        );
        __pipeline_wait_prior(1);
        // op_call |buf_idx=0| i + (-1) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<0>(
            G2sMatrixBLoadOp_v.template output<0, 0>()
        );
        // op_call |buf_idx=0| i + (-1) CalculateOp 
        CalculateOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            S2rBValLoadOp_v.template output<0, 0>(),
            ZerosOp_v.template output<0>()
        );
        // stage 2..2
        __syncthreads();
        // op_call |buf_idx=1| i + (0) S2rAValLoadOp 
        S2rAValLoadOp_v.template f<1>(
            S2sRestoreMatrixAOp_v.template output<0, 1>()
        );
        __pipeline_wait_prior(0);
        // op_call |buf_idx=1| i + (0) S2rBValLoadOp 
        S2rBValLoadOp_v.template f<1>(
            G2sMatrixBLoadOp_v.template output<0, 1>()
        );
        // op_call |buf_idx=1| i + (0) CalculateOp 
        CalculateOp_v.template f<1>(
            S2rAValLoadOp_v.template output<0, 1>(),
            S2rBValLoadOp_v.template output<0, 1>(),
            ZerosOp_v.template output<0>()
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
            return for_loop_result_0;
        }
    }

    // f
    template<typename = void>
    CUTE_DEVICE void f(int l, int r, TI2 off_l) {
        _l = l;
        _r = r;
        this->off_l = off_l;
        
        dispatch();
    }
};

template<class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = float, class TI1 = float>
class R2gCValStoreOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{}))));
    Shape_i0 shape_i0;
    // Tensor_i0 operand_0;
    using Shape_i1 = decltype(make_shape(VARLEN, VARLEN));
    Shape_i1 shape_i1;
    // Tensor_i1 operand_1;

    // Outputs
    

    // Hardware parameters
    int lid;

    CUTE_DEVICE R2gCValStoreOp(int lid, int mode_i1_0, int mode_i1_1)
    : lid(lid), shape_i1(make_shape(mode_i1_0, mode_i1_1)) {}

    template<int output_idx>
    CUTE_DEVICE auto output() {
        
    }

    template<class Tensor_i0, class Tensor_i1, typename = void>
    CUTE_DEVICE void f(Tensor_i0 operand_0, Tensor_i1 operand_1) {
        for (int i_tileN = 0; i_tileN < get<1>(BLK_MNK{}) / 8; i_tileN++) {
            int row = lid / 4;
            int col = i_tileN * 8 + lid % 4 * 2;
            operand_1(col, row) = operand_0(0, make_coord(_0{}, i_tileN));
            operand_1(col + 1, row) = operand_0(1, make_coord(_0{}, i_tileN));
            operand_1(col, row + 8) = operand_0(2, make_coord(_0{}, i_tileN));
            operand_1(col + 1, row + 8) = operand_0(3, make_coord(_0{}, i_tileN));
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class T_B_valOp, class T_C_valOp, class T_val_coo_idx_newOp, class T_val_coo_off_newOp, class T_val_coo_val_newOp, class T_val_len_offsetOp, class T_val_sidx_newOp, class T_BlkYLoopZeroOp, class T_BlockDimYOp, class T_BlkXLoopZeroOp, class T_BlockDimXOp, class T_ZerosOp, class T_zero, class T_G2rSparseOffsetLoadOp, class T_for_loop, class T_R2gCValStoreOp>
class generated_kernel_blk_y_loop_blk_x_loop {
public:
    // Inputs
    int _idx;

    // Hardware parameters
    int lid;

    // Body Ops
    T_B_valOp &B_valOp_v;
    T_C_valOp &C_valOp_v;
    T_val_coo_idx_newOp &val_coo_idx_newOp_v;
    T_val_coo_off_newOp &val_coo_off_newOp_v;
    T_val_coo_val_newOp &val_coo_val_newOp_v;
    T_val_len_offsetOp &val_len_offsetOp_v;
    T_val_sidx_newOp &val_sidx_newOp_v;
    T_BlkYLoopZeroOp &BlkYLoopZeroOp_v;
    T_BlockDimYOp &BlockDimYOp_v;
    T_BlkXLoopZeroOp &BlkXLoopZeroOp_v;
    T_BlockDimXOp &BlockDimXOp_v;
    T_ZerosOp &ZerosOp_v;
    T_zero &zero_v;
    T_G2rSparseOffsetLoadOp &G2rSparseOffsetLoadOp_v;
    T_for_loop &for_loop_v;
    T_R2gCValStoreOp &R2gCValStoreOp_v;

    // Constructor
    CUTE_DEVICE generated_kernel_blk_y_loop_blk_x_loop(int lid, T_B_valOp &B_valOp_v, T_C_valOp &C_valOp_v, T_val_coo_idx_newOp &val_coo_idx_newOp_v, T_val_coo_off_newOp &val_coo_off_newOp_v, T_val_coo_val_newOp &val_coo_val_newOp_v, T_val_len_offsetOp &val_len_offsetOp_v, T_val_sidx_newOp &val_sidx_newOp_v, T_BlkYLoopZeroOp &BlkYLoopZeroOp_v, T_BlockDimYOp &BlockDimYOp_v, T_BlkXLoopZeroOp &BlkXLoopZeroOp_v, T_BlockDimXOp &BlockDimXOp_v, T_ZerosOp &ZerosOp_v, T_zero &zero_v, T_G2rSparseOffsetLoadOp &G2rSparseOffsetLoadOp_v, T_for_loop &for_loop_v, T_R2gCValStoreOp &R2gCValStoreOp_v)
    : lid(lid), B_valOp_v(B_valOp_v), C_valOp_v(C_valOp_v), val_coo_idx_newOp_v(val_coo_idx_newOp_v), val_coo_off_newOp_v(val_coo_off_newOp_v), val_coo_val_newOp_v(val_coo_val_newOp_v), val_len_offsetOp_v(val_len_offsetOp_v), val_sidx_newOp_v(val_sidx_newOp_v), BlkYLoopZeroOp_v(BlkYLoopZeroOp_v), BlockDimYOp_v(BlockDimYOp_v), BlkXLoopZeroOp_v(BlkXLoopZeroOp_v), BlockDimXOp_v(BlockDimXOp_v), ZerosOp_v(ZerosOp_v), zero_v(zero_v), G2rSparseOffsetLoadOp_v(G2rSparseOffsetLoadOp_v), for_loop_v(for_loop_v), R2gCValStoreOp_v(R2gCValStoreOp_v) {}

    // Body
    CUTE_DEVICE void body() {
        ZerosOp_v.template f<>();
        zero_v.template f<>();
        G2rSparseOffsetLoadOp_v.template f<>(
            val_len_offsetOp_v.template output<0>(),
            blockIdx.x
        );
        for_loop_v.template f<>(
            zero_v.template output<0>()(0),
            G2rSparseOffsetLoadOp_v.template output<1>()(0),
            G2rSparseOffsetLoadOp_v.template output<0>()(0)
        );
        R2gCValStoreOp_v.template f<>(
            for_loop_v.template output<0>(),
            C_valOp_v.template output<0>()
        );
    }

    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        this->_idx = blockIdx.x;
        body();
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class T_B_valOp, class T_C_valOp, class T_val_coo_idx_newOp, class T_val_coo_off_newOp, class T_val_coo_val_newOp, class T_val_len_offsetOp, class T_val_sidx_newOp, class T_BlkYLoopZeroOp, class T_BlockDimYOp, class T_BlkXLoopZeroOp, class T_BlockDimXOp, class T_generated_kernel_blk_y_loop_blk_x_loop>
class generated_kernel_blk_y_loop {
public:
    // Inputs
    int _idx;

    // Hardware parameters
    int lid;

    // Body Ops
    T_B_valOp &B_valOp_v;
    T_C_valOp &C_valOp_v;
    T_val_coo_idx_newOp &val_coo_idx_newOp_v;
    T_val_coo_off_newOp &val_coo_off_newOp_v;
    T_val_coo_val_newOp &val_coo_val_newOp_v;
    T_val_len_offsetOp &val_len_offsetOp_v;
    T_val_sidx_newOp &val_sidx_newOp_v;
    T_BlkYLoopZeroOp &BlkYLoopZeroOp_v;
    T_BlockDimYOp &BlockDimYOp_v;
    T_BlkXLoopZeroOp &BlkXLoopZeroOp_v;
    T_BlockDimXOp &BlockDimXOp_v;
    T_generated_kernel_blk_y_loop_blk_x_loop &generated_kernel_blk_y_loop_blk_x_loop_v;

    // Constructor
    CUTE_DEVICE generated_kernel_blk_y_loop(int lid, T_B_valOp &B_valOp_v, T_C_valOp &C_valOp_v, T_val_coo_idx_newOp &val_coo_idx_newOp_v, T_val_coo_off_newOp &val_coo_off_newOp_v, T_val_coo_val_newOp &val_coo_val_newOp_v, T_val_len_offsetOp &val_len_offsetOp_v, T_val_sidx_newOp &val_sidx_newOp_v, T_BlkYLoopZeroOp &BlkYLoopZeroOp_v, T_BlockDimYOp &BlockDimYOp_v, T_BlkXLoopZeroOp &BlkXLoopZeroOp_v, T_BlockDimXOp &BlockDimXOp_v, T_generated_kernel_blk_y_loop_blk_x_loop &generated_kernel_blk_y_loop_blk_x_loop_v)
    : lid(lid), B_valOp_v(B_valOp_v), C_valOp_v(C_valOp_v), val_coo_idx_newOp_v(val_coo_idx_newOp_v), val_coo_off_newOp_v(val_coo_off_newOp_v), val_coo_val_newOp_v(val_coo_val_newOp_v), val_len_offsetOp_v(val_len_offsetOp_v), val_sidx_newOp_v(val_sidx_newOp_v), BlkYLoopZeroOp_v(BlkYLoopZeroOp_v), BlockDimYOp_v(BlockDimYOp_v), BlkXLoopZeroOp_v(BlkXLoopZeroOp_v), BlockDimXOp_v(BlockDimXOp_v), generated_kernel_blk_y_loop_blk_x_loop_v(generated_kernel_blk_y_loop_blk_x_loop_v) {}

    // Body
    CUTE_DEVICE void body() {
        BlkXLoopZeroOp_v.template f<>();
        BlockDimXOp_v.template f<>();
        generated_kernel_blk_y_loop_blk_x_loop_v.template f<>();
    }

    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        this->_idx = blockIdx.y;
        body();
    }
};

// GLOBAL FUNCTION
template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK>
__global__ void generated_kernel(
    float *dB_val,
    float *dC_val,
    int *dval_coo_idx_new,
    int *dval_coo_off_new,
    float *dval_coo_val_new,
    int *dval_len_offset,
    int *dval_sidx_new,
    int K, int M, int Mo, int N, int nnz, int nnz_dim_K
) {
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int wid = tid / 32;

    using B_valOp_t = B_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    B_valOp_t B_valOp_v{lid, K, N, dB_val};
    using C_valOp_t = C_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    C_valOp_t C_valOp_v{lid, N, M, dC_val};
    using val_coo_idx_newOp_t = val_coo_idx_newOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_coo_idx_newOp_t val_coo_idx_newOp_v{lid, nnz, dval_coo_idx_new};
    using val_coo_off_newOp_t = val_coo_off_newOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_coo_off_newOp_t val_coo_off_newOp_v{lid, nnz_dim_K, dval_coo_off_new};
    using val_coo_val_newOp_t = val_coo_val_newOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_coo_val_newOp_t val_coo_val_newOp_v{lid, nnz, dval_coo_val_new};
    using val_len_offsetOp_t = val_len_offsetOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_len_offsetOp_t val_len_offsetOp_v{lid, Mo, dval_len_offset};
    using val_sidx_newOp_t = val_sidx_newOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_sidx_newOp_t val_sidx_newOp_v{lid, nnz_dim_K, dval_sidx_new};
    using BlkYLoopZeroOp_t = BlkYLoopZeroOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlkYLoopZeroOp_layout_o0 = typename BlkYLoopZeroOp_t::Layout_o0;
    int BlkYLoopZeroOp_tensor_o0[cosize_v<BlkYLoopZeroOp_layout_o0>];
    BlkYLoopZeroOp_t BlkYLoopZeroOp_v{lid, BlkYLoopZeroOp_tensor_o0};
    using BlockDimYOp_t = BlockDimYOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlockDimYOp_layout_o0 = typename BlockDimYOp_t::Layout_o0;
    int BlockDimYOp_tensor_o0[cosize_v<BlockDimYOp_layout_o0>];
    BlockDimYOp_t BlockDimYOp_v{lid, BlockDimYOp_tensor_o0};
    using BlkXLoopZeroOp_t = BlkXLoopZeroOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlkXLoopZeroOp_layout_o0 = typename BlkXLoopZeroOp_t::Layout_o0;
    int BlkXLoopZeroOp_tensor_o0[cosize_v<BlkXLoopZeroOp_layout_o0>];
    BlkXLoopZeroOp_t BlkXLoopZeroOp_v{lid, BlkXLoopZeroOp_tensor_o0};
    using BlockDimXOp_t = BlockDimXOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlockDimXOp_layout_o0 = typename BlockDimXOp_t::Layout_o0;
    int BlockDimXOp_tensor_o0[cosize_v<BlockDimXOp_layout_o0>];
    BlockDimXOp_t BlockDimXOp_v{lid, BlockDimXOp_tensor_o0};
    using ZerosOp_t = ZerosOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using ZerosOp_layout_o0 = typename ZerosOp_t::Layout_o0;
    float ZerosOp_tensor_o0[cosize_v<ZerosOp_layout_o0>];
    ZerosOp_t ZerosOp_v{lid, ZerosOp_tensor_o0};
    using zero_t = zero<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using zero_layout_o0 = typename zero_t::Layout_o0;
    int zero_tensor_o0[cosize_v<zero_layout_o0>];
    zero_t zero_v{lid, zero_tensor_o0};
    using G2rSparseOffsetLoadOp_t = G2rSparseOffsetLoadOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2rSparseOffsetLoadOp_layout_o0 = typename G2rSparseOffsetLoadOp_t::Layout_o0;
    using G2rSparseOffsetLoadOp_layout_o1 = typename G2rSparseOffsetLoadOp_t::Layout_o1;
    int G2rSparseOffsetLoadOp_tensor_o0[cosize_v<G2rSparseOffsetLoadOp_layout_o0>];
    int G2rSparseOffsetLoadOp_tensor_o1[cosize_v<G2rSparseOffsetLoadOp_layout_o1>];
    G2rSparseOffsetLoadOp_t G2rSparseOffsetLoadOp_v{
        lid, 
        G2rSparseOffsetLoadOp_tensor_o0, G2rSparseOffsetLoadOp_tensor_o1, 
        Mo,
    };
    using G2sSparseIndexLoadOp_t = G2sSparseIndexLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sSparseIndexLoadOp_layout_o0 = typename G2sSparseIndexLoadOp_t::Layout_o0;
    __align__(16) __shared__ int G2sSparseIndexLoadOp_tensor_o0[cosize_v<G2sSparseIndexLoadOp_layout_o0>];
    G2sSparseIndexLoadOp_t G2sSparseIndexLoadOp_v{
        lid, 
        G2sSparseIndexLoadOp_tensor_o0, 
        nnz_dim_K,
    };
    using G2sSparseCooIdxLoadOp_t = G2sSparseCooIdxLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sSparseCooIdxLoadOp_layout_o0 = typename G2sSparseCooIdxLoadOp_t::Layout_o0;
    __align__(16) __shared__ int G2sSparseCooIdxLoadOp_tensor_o0[cosize_v<G2sSparseCooIdxLoadOp_layout_o0>];
    using G2sSparseCooIdxLoadOp_layout_o1 = typename G2sSparseCooIdxLoadOp_t::Layout_o1;
    int G2sSparseCooIdxLoadOp_tensor_o1[cosize_v<G2sSparseCooIdxLoadOp_layout_o1>];
    G2sSparseCooIdxLoadOp_t G2sSparseCooIdxLoadOp_v{
        lid, 
        G2sSparseCooIdxLoadOp_tensor_o0, G2sSparseCooIdxLoadOp_tensor_o1, 
        nnz, nnz_dim_K,
    };
    using G2sSparseCooValLoadOp_t = G2sSparseCooValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sSparseCooValLoadOp_layout_o0 = typename G2sSparseCooValLoadOp_t::Layout_o0;
    __align__(16) __shared__ float G2sSparseCooValLoadOp_tensor_o0[cosize_v<G2sSparseCooValLoadOp_layout_o0>];
    G2sSparseCooValLoadOp_t G2sSparseCooValLoadOp_v{
        lid, 
        G2sSparseCooValLoadOp_tensor_o0, 
        nnz, nnz_dim_K,
    };
    using G2sMatrixBLoadOp_t = G2sMatrixBLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sMatrixBLoadOp_layout_o0 = typename G2sMatrixBLoadOp_t::Layout_o0;
    __align__(16) __shared__ float G2sMatrixBLoadOp_tensor_o0[cosize_v<G2sMatrixBLoadOp_layout_o0>];
    G2sMatrixBLoadOp_t G2sMatrixBLoadOp_v{
        lid, 
        G2sMatrixBLoadOp_tensor_o0, 
        K, N,
    };
    using S2sRestoreMatrixAOp_t = S2sRestoreMatrixAOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using S2sRestoreMatrixAOp_layout_o0 = typename S2sRestoreMatrixAOp_t::Layout_o0;
    __shared__ float S2sRestoreMatrixAOp_tensor_o0[cosize_v<S2sRestoreMatrixAOp_layout_o0>];
    S2sRestoreMatrixAOp_t S2sRestoreMatrixAOp_v{
        lid, 
        S2sRestoreMatrixAOp_tensor_o0,
    };
    using S2rAValLoadOp_t = S2rAValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using S2rAValLoadOp_layout_o0 = typename S2rAValLoadOp_t::Layout_o0;
    float S2rAValLoadOp_tensor_o0[cosize_v<S2rAValLoadOp_layout_o0>];
    S2rAValLoadOp_t S2rAValLoadOp_v{
        lid, 
        S2rAValLoadOp_tensor_o0,
    };
    using S2rBValLoadOp_t = S2rBValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using S2rBValLoadOp_layout_o0 = typename S2rBValLoadOp_t::Layout_o0;
    float S2rBValLoadOp_tensor_o0[cosize_v<S2rBValLoadOp_layout_o0>];
    S2rBValLoadOp_t S2rBValLoadOp_v{
        lid, 
        S2rBValLoadOp_tensor_o0, 
        N,
    };
    using CalculateOp_t = CalculateOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, std::decay_t<decltype(ZerosOp_v.template output<0>())>>;
    CalculateOp_t CalculateOp_v{
        lid, 
        ZerosOp_v.template output<0>(),
    };
    using for_loop_t = for_loop<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, B_valOp_t, C_valOp_t, val_coo_idx_newOp_t, val_coo_off_newOp_t, val_coo_val_newOp_t, val_len_offsetOp_t, val_sidx_newOp_t, BlkYLoopZeroOp_t, BlockDimYOp_t, BlkXLoopZeroOp_t, BlockDimXOp_t, ZerosOp_t, zero_t, G2rSparseOffsetLoadOp_t, G2sSparseIndexLoadOp_t, G2sSparseCooIdxLoadOp_t, G2sSparseCooValLoadOp_t, G2sMatrixBLoadOp_t, S2sRestoreMatrixAOp_t, S2rAValLoadOp_t, S2rBValLoadOp_t, CalculateOp_t, std::decay_t<decltype(CalculateOp_v.template output<0>())>>;
    for_loop_t for_loop_v{lid, B_valOp_v, C_valOp_v, val_coo_idx_newOp_v, val_coo_off_newOp_v, val_coo_val_newOp_v, val_len_offsetOp_v, val_sidx_newOp_v, BlkYLoopZeroOp_v, BlockDimYOp_v, BlkXLoopZeroOp_v, BlockDimXOp_v, ZerosOp_v, zero_v, G2rSparseOffsetLoadOp_v, G2sSparseIndexLoadOp_v, G2sSparseCooIdxLoadOp_v, G2sSparseCooValLoadOp_v, G2sMatrixBLoadOp_v, S2sRestoreMatrixAOp_v, S2rAValLoadOp_v, S2rBValLoadOp_v, CalculateOp_v, CalculateOp_v.template output<0>()};
    using R2gCValStoreOp_t = R2gCValStoreOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    R2gCValStoreOp_t R2gCValStoreOp_v{
        lid, 
        N, M,
    };
    using generated_kernel_blk_y_loop_blk_x_loop_t = generated_kernel_blk_y_loop_blk_x_loop<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, B_valOp_t, C_valOp_t, val_coo_idx_newOp_t, val_coo_off_newOp_t, val_coo_val_newOp_t, val_len_offsetOp_t, val_sidx_newOp_t, BlkYLoopZeroOp_t, BlockDimYOp_t, BlkXLoopZeroOp_t, BlockDimXOp_t, ZerosOp_t, zero_t, G2rSparseOffsetLoadOp_t, for_loop_t, R2gCValStoreOp_t>;
    generated_kernel_blk_y_loop_blk_x_loop_t generated_kernel_blk_y_loop_blk_x_loop_v{lid, B_valOp_v, C_valOp_v, val_coo_idx_newOp_v, val_coo_off_newOp_v, val_coo_val_newOp_v, val_len_offsetOp_v, val_sidx_newOp_v, BlkYLoopZeroOp_v, BlockDimYOp_v, BlkXLoopZeroOp_v, BlockDimXOp_v, ZerosOp_v, zero_v, G2rSparseOffsetLoadOp_v, for_loop_v, R2gCValStoreOp_v};
    using generated_kernel_blk_y_loop_t = generated_kernel_blk_y_loop<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, B_valOp_t, C_valOp_t, val_coo_idx_newOp_t, val_coo_off_newOp_t, val_coo_val_newOp_t, val_len_offsetOp_t, val_sidx_newOp_t, BlkYLoopZeroOp_t, BlockDimYOp_t, BlkXLoopZeroOp_t, BlockDimXOp_t, generated_kernel_blk_y_loop_blk_x_loop_t>;
    generated_kernel_blk_y_loop_t generated_kernel_blk_y_loop_v{lid, B_valOp_v, C_valOp_v, val_coo_idx_newOp_v, val_coo_off_newOp_v, val_coo_val_newOp_v, val_len_offsetOp_v, val_sidx_newOp_v, BlkYLoopZeroOp_v, BlockDimYOp_v, BlkXLoopZeroOp_v, BlockDimXOp_v, generated_kernel_blk_y_loop_blk_x_loop_v};

    BlkYLoopZeroOp_v.template f<>();
    BlockDimYOp_v.template f<>();
    generated_kernel_blk_y_loop_v.template f<>();
};