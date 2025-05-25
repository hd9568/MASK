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


template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = half>
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
    using Tensor_o0 = decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN, VARLEN)), select<1, 0>(BLK_MNK{}))(make_coord(_, VARLEN), make_coord(_, VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 C_val;

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE C_valOp(int tid, int lid, int wid, int N, int M, TO0 *dC_val)
    : tid(tid), lid(lid), wid(wid),
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

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = uint32_t>
class compressed_AOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 compressed_A;

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE compressed_AOp(int tid, int lid, int wid, int nnz, TO0 *dcompressed_A)
    : tid(tid), lid(lid), wid(wid),
    shape_o0(make_tensor(make_gmem_ptr(dcompressed_A), make_shape(nnz)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dcompressed_A), make_shape(nnz)).layout()),
    compressed_A(make_tensor(make_gmem_ptr(dcompressed_A), make_shape(nnz))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return compressed_A;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class tile_offsetsOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 tile_offsets;

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE tile_offsetsOp(int tid, int lid, int wid, int NumOffsets, TO0 *dtile_offsets)
    : tid(tid), lid(lid), wid(wid),
    shape_o0(make_tensor(make_gmem_ptr(dtile_offsets), make_shape(NumOffsets)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dtile_offsets), make_shape(NumOffsets)).layout()),
    tile_offsets(make_tensor(make_gmem_ptr(dtile_offsets), make_shape(NumOffsets))) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_offsets;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class FlashLLMLoopBoundZeroOp {
public:
    // No Inputs

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Stride_o0 = decltype(make_stride(_0{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 FlashLLMLoopBoundZeroOp_out;
    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE FlashLLMLoopBoundZeroOp(int tid, int lid, int wid, TO0 *FlashLLMLoopBoundZeroOp_out)
    : tid(tid), lid(lid), wid(wid), FlashLLMLoopBoundZeroOp_out(make_tensor(make_rmem_ptr(FlashLLMLoopBoundZeroOp_out), Layout_o0{})) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return FlashLLMLoopBoundZeroOp_out;
        }
    }
    
    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        clear(FlashLLMLoopBoundZeroOp_out);
    }
};

template<class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = int, class TO0 = int>
class FlashLLMLoopBoundUpOp {
public:
    // Inputs
    // TI0 ubi;

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Stride_o0 = decltype(make_stride(_0{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 ubo;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE FlashLLMLoopBoundUpOp(int tid, int lid, int wid, TO0 *ubo)
    : tid(tid), lid(lid), wid(wid), ubo(make_tensor(make_rmem_ptr(ubo), Layout_o0{})) {}

    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return ubo;
        }
    }

    template<typename = void>
    CUTE_DEVICE void f(TI0 ubi) {
        
        ubo(0) = ubi;
                    
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
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE BlkYLoopZeroOp(int tid, int lid, int wid, TO0 *BlkYLoopZeroOp_out)
    : tid(tid), lid(lid), wid(wid), BlkYLoopZeroOp_out(make_tensor(make_rmem_ptr(BlkYLoopZeroOp_out), Layout_o0{})) {}

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
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE BlkXLoopZeroOp(int tid, int lid, int wid, TO0 *BlkXLoopZeroOp_out)
    : tid(tid), lid(lid), wid(wid), BlkXLoopZeroOp_out(make_tensor(make_rmem_ptr(BlkXLoopZeroOp_out), Layout_o0{})) {}

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

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class FlashLLMLoadCompressedAVecSizeOp {
public:
    // No Inputs

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Stride_o0 = decltype(make_stride(_0{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 FlashLLMLoadCompressedAVecSizeOp_out;
    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE FlashLLMLoadCompressedAVecSizeOp(int tid, int lid, int wid, TO0 *FlashLLMLoadCompressedAVecSizeOp_out)
    : tid(tid), lid(lid), wid(wid), FlashLLMLoadCompressedAVecSizeOp_out(make_tensor(make_rmem_ptr(FlashLLMLoadCompressedAVecSizeOp_out), Layout_o0{})) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return FlashLLMLoadCompressedAVecSizeOp_out;
        }
    }
    
    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        for (int i = 0; i < FlashLLMLoadCompressedAVecSizeOp_out.size(); i++) { FlashLLMLoadCompressedAVecSizeOp_out(i) = 4; }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TO0 = int>
class FlashLLMLoadCompressedANumRegOp {
public:
    // No Inputs

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Stride_o0 = decltype(make_stride(_0{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 FlashLLMLoadCompressedANumRegOp_out;
    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE FlashLLMLoadCompressedANumRegOp(int tid, int lid, int wid, TO0 *FlashLLMLoadCompressedANumRegOp_out)
    : tid(tid), lid(lid), wid(wid), FlashLLMLoadCompressedANumRegOp_out(make_tensor(make_rmem_ptr(FlashLLMLoadCompressedANumRegOp_out), Layout_o0{})) {}

    // output
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return FlashLLMLoadCompressedANumRegOp_out;
        }
    }
    
    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        for (int i = 0; i < FlashLLMLoadCompressedANumRegOp_out.size(); i++) { FlashLLMLoadCompressedANumRegOp_out(i) = (get<0>(BLK_MNK{}) * get<2>(BLK_MNK{}) + blockDim.x - 1) / blockDim.x; }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class REGA_RMEM_SET = Int<get<0>(BLK_MMA_MNK{}) / get<0>(WARP_MNK{})>, class REGB_RMEM_SET = Int<get<1>(BLK_MMA_MNK{}) / get<1>(WARP_MNK{})>, class TO0 = float>
class ZerosOp {
public:
    // No Inputs

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(REGA_RMEM_SET{}, REGB_RMEM_SET{})));
    using Stride_o0 = decltype(make_stride(_1{}, make_stride(Int<4>{}, Int<4>{} * REGA_RMEM_SET{})));
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

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = int, class TI1 = int, class TI2 = int, class TO0 = int>
class G2rTileOffsetsLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 tile_offsets;
    // TI1 i;
    // TI2 num_tile_per_row_window;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<2>{}, NBUF{}));
    using Stride_o0 = decltype(make_stride(_1{}, Int<2>{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 tile_offsets_reg;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rTileOffsetsLoadOp(int tid, int lid, int wid, TO0 *tile_offsets_reg, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)), tile_offsets_reg(make_tensor(make_rmem_ptr(tile_offsets_reg), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_offsets_reg(_, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_offsets_reg;
        }
    }

    template<int buf_idx, class Tensor_i0, typename = void>
    CUTE_DEVICE void f(Tensor_i0 tile_offsets, TI1 i, TI2 num_tile_per_row_window) {
        
        tile_offsets_reg(0, buf_idx) = tile_offsets(blockIdx.x * num_tile_per_row_window + i);
        tile_offsets_reg(1, buf_idx) = tile_offsets(blockIdx.x * num_tile_per_row_window + i + 1);
                    
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class compressed_A_reg_set = Int<DIVUP(DIVUP(get<0>(BLK_MNK{}) * get<2>(BLK_MNK{}), get<0>(WARP_MNK{}) * get<1>(WARP_MNK{}) * get<2>(WARP_MNK{}) * 32/*blockDim*/), 4 /*vector size = 4 for a thread*/)>, class TI0 = uint32_t, class TI1 = int, class TO0 = uint32_t>
class G2rLoadCompressedTileAOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 compressed_A;
    using Shape_i1 = decltype(make_shape(Int<2>{}));
    Shape_i1 shape_i1;
    // Tensor_i1 tile_offsets_reg;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, compressed_A_reg_set{}, NBUF{}));
    using Stride_o0 = decltype(make_stride(_1{}, Int<4>{}, Int<4>{} * compressed_A_reg_set{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}, Stride_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 compressed_A_reg;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rLoadCompressedTileAOp(int tid, int lid, int wid, TO0 *compressed_A_reg, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)), compressed_A_reg(make_tensor(make_rmem_ptr(compressed_A_reg), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return compressed_A_reg(_, _, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return compressed_A_reg;
        }
    }

    template<int buf_idx, class Tensor_i0, class Tensor_i1, typename = void>
    CUTE_DEVICE void f(Tensor_i0 compressed_A, Tensor_i1 tile_offsets_reg) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i1) == size<0>(shape(tile_offsets_reg))));
        
        int max_nnz_vector_this_tile = tile_offsets_reg(1) - tile_offsets_reg(0);
        for (int i = 0; i < compressed_A_reg_set{}; i++) {
            //! single warp
            // int index = lid + i * (32);
            //! multi warps
            int index = threadIdx.x + i * (blockDim.x);
            if (index >= max_nnz_vector_this_tile) {
                break;
            }
            // TODO: support SMEM load
            // auto dst = compressed_A_reg(_, i, buf_idx).data();
            // auto src = &(compressed_A( (tile_offsets_reg(0) + index) * 4));
            // copy(Copy_Atom<UniversalCopy<uint128_t>, uint32_t>{}, src, dst);
            
            *((uint128_t*)(compressed_A_reg(_, i, buf_idx).data())) = *((uint128_t*)(&compressed_A( (tile_offsets_reg(0) + index) * 4)));
        }
                    
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class compressed_A_reg_set = Int<DIVUP(DIVUP(get<0>(BLK_MNK{}) * get<2>(BLK_MNK{}), get<0>(WARP_MNK{}) * get<1>(WARP_MNK{}) * get<2>(WARP_MNK{}) * 32/*blockDim*/), 4 /*vector size = 4 for a thread*/)>, class TI0 = uint32_t, class TI1 = int, class TO0 = half>
class R2sRestoreTileAOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(Int<4>{}, compressed_A_reg_set{}));
    Shape_i0 shape_i0;
    // Tensor_i0 compressed_A_reg;
    using Shape_i1 = decltype(make_shape(Int<2>{}));
    Shape_i1 shape_i1;
    // Tensor_i1 tile_offsets_reg;

    // Outputs
    using Shape_o0 = decltype(make_shape(get<0>(BLK_MNK{}), get<2>(BLK_MNK{}), NBUF{}));
    using Stride_o0 = decltype(make_stride(get<2>(BLK_MNK{}), _1{}, get<0>(BLK_MNK{}) * get<2>(BLK_MNK{})));
    using Layout_o0 = decltype(composition(Swizzle<3, 3, 3>{}, make_layout(Shape_o0{}, Stride_o0{})));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 tileA_block;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE R2sRestoreTileAOp(int tid, int lid, int wid, TO0 *tileA_block)
    : tid(tid), lid(lid), wid(wid), tileA_block(make_tensor(make_smem_ptr(tileA_block), Layout_o0{})) {}

    template<int output_idx, int buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tileA_block(_, _, buf_idx);
        }
    }
    
    template<int output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tileA_block;
        }
    }

    template<int buf_idx, class Tensor_i0, class Tensor_i1, typename = void>
    CUTE_DEVICE void f(Tensor_i0 compressed_A_reg, Tensor_i1 tile_offsets_reg) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(compressed_A_reg))));
        CUTE_STATIC_ASSERT_V((get<1>(shape_i0) == size<1>(shape(compressed_A_reg))));
        CUTE_STATIC_ASSERT_V((get<0>(shape_i1) == size<0>(shape(tile_offsets_reg))));
        
        for (int i_o2s = threadIdx.x; i_o2s < (get<0>(BLK_MNK{}) * get<2>(BLK_MNK{})) / 2; i_o2s += blockDim.x) {
            *((float*)(tileA_block(_, _, buf_idx).data().get()) + i_o2s) = 0;
        }
        __syncthreads();
        int max_nnz_vector_this_tile = tile_offsets_reg(1) - tile_offsets_reg(0);
        for (int i = 0; i < compressed_A_reg_set{}; i++) {
            int index = threadIdx.x + i * (blockDim.x);
            if (index >= max_nnz_vector_this_tile) {
                break;
            }
            for (int reg_ptr = 0; reg_ptr < 4; reg_ptr++) {
                uint32_t val_idx_bind = compressed_A_reg(reg_ptr, i);
                // half value;
                // short idx;
                // unpack_half_short(val_idx_bind, value, idx);
                half* half_ptr =
                            reinterpret_cast<half*>(&val_idx_bind);
                short* short_ptr  = reinterpret_cast<short*>(half_ptr + 1);
                half   value      = *half_ptr;
                short  idx      = *short_ptr;
                int row = idx / get<2>(BLK_MNK{});
                int col = idx % get<2>(BLK_MNK{});
                // tileA_block(col, row, buf_idx) = value; 
                // tileA_block(row, col, buf_idx) = value;
                *((half*)tileA_block(_, _, buf_idx).data().get() + idx) = value;
            }
        }
                    
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class TI0 = half, class TI1 = int, class TO0 = half>
class G2sLoadTileBOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN, get<1>(BLK_MNK{})));
    Shape_i0 shape_i0;
    // Tensor_i0 B_val;
    // TI1 tile_idx;

    // Outputs
    using Shape_o0 = decltype(make_shape(get<2>(BLK_MNK{}), get<1>(BLK_MNK{}), NBUF{}));
    using Stride_o0 = decltype(make_stride(get<1>(BLK_MNK{}), _1{}, get<2>(BLK_MNK{}) * get<1>(BLK_MNK{})));
    using Layout_o0 = decltype(composition(Swizzle<3, 3, 4>{}, make_layout(Shape_o0{}, Stride_o0{})));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 tileB_block;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2sLoadTileBOp(int tid, int lid, int wid, TO0 *tileB_block, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0, get<1>(BLK_MNK{}))), tileB_block(make_tensor(make_smem_ptr(tileB_block), Layout_o0{})) {}

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
    CUTE_DEVICE void f(Tensor_i0 B_val, TI1 tile_idx) {
        CUTE_STATIC_ASSERT_V((get<1>(shape_i0) == size<1>(shape(B_val))));
        
        __syncthreads();
        //! multi warps. check commit history for single warp version
        auto load_tile_n = _8{};
        for (int iter_i = threadIdx.x; iter_i < get<2>(BLK_MNK{}) * get<1>(BLK_MNK{}) / load_tile_n; iter_i += blockDim.x) {
            int row = iter_i / (get<1>(BLK_MNK{}) / load_tile_n);
            int col = iter_i % (get<1>(BLK_MNK{}) / load_tile_n);
            int gmem_row = tile_idx * get<2>(BLK_MNK{}) + row;
            auto thr_tiler_gmem = make_shape(_1{}, _8{});
            auto thr_coord_gmem = make_coord(gmem_row, col);
            auto B_val_thr = local_tile(B_val, thr_tiler_gmem, thr_coord_gmem);
            auto thr_tiler_smem = make_shape(_1{}, _8{});
            auto thr_coord_smem = make_coord(row, col);
            auto dst = local_tile(tileB_block(_, _, buf_idx), thr_tiler_smem, thr_coord_smem);
            copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{}, B_val_thr(0, _), dst(0, _));
        }
                    
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class Tensor_o0, class REGA_RMEM_SET = Int<get<0>(BLK_MMA_MNK{}) / get<0>(WARP_MNK{})>, class REGB_RMEM_SET = Int<get<1>(BLK_MMA_MNK{}) / get<1>(WARP_MNK{})>, class TI0 = half, class TI1 = half, class TO0 = float>
class MergedMMAOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(get<0>(BLK_MNK{}), get<2>(BLK_MNK{})));
    Shape_i0 shape_i0;
    // Tensor_i0 tileA_block;
    using Shape_i1 = decltype(make_shape(get<2>(BLK_MNK{}), get<1>(BLK_MNK{})));
    Shape_i1 shape_i1;
    // Tensor_i1 tileB_block;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(REGA_RMEM_SET{}, REGB_RMEM_SET{})));
    Shape_o0 shape_o0;
    Tensor_o0 REGC;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE MergedMMAOp(int tid, int lid, int wid, Tensor_o0 REGC)
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
    CUTE_DEVICE void f(Tensor_i0 tileA_block, Tensor_i1 tileB_block) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(tileA_block))));
        CUTE_STATIC_ASSERT_V((get<1>(shape_i0) == size<1>(shape(tileA_block))));
        CUTE_STATIC_ASSERT_V((get<0>(shape_i1) == size<0>(shape(tileB_block))));
        CUTE_STATIC_ASSERT_V((get<1>(shape_i1) == size<1>(shape(tileB_block))));
        
        __syncthreads();
        int WARP_M_ID = wid / get<1>(WARP_MNK{});
        int WARP_N_ID = wid % get<1>(WARP_MNK{});
        half REGA[2 * REGA_RMEM_SET{}][8];
        half REGB[2 * REGB_RMEM_SET{}][4];
        // load (0)
        int k_iter = 0;
        for (int m_iter = 0; m_iter < REGA_RMEM_SET{}; m_iter++) {
            int row = lid % 16 + m_iter * 16 + WARP_M_ID * REGA_RMEM_SET{} * 16;
            int col = lid / 16 + k_iter * 2;
            ldmatrix_m8n8k8_x4(
                (uint32_t*)(REGA[0 * REGA_RMEM_SET{} + m_iter]),
                (void*)&tileA_block(row, col * 8)
            );
        }
        for (int n_iter = 0; n_iter < REGB_RMEM_SET{}; n_iter++) {
            int row_b = lid % 16 + k_iter * 16;
            int col_b = n_iter + REGB_RMEM_SET{} * WARP_N_ID;
            ldmatrix_m8n8k8_x2_trans(
                (uint32_t*)(REGB[0 * REGB_RMEM_SET{} + n_iter]),
                (void*)(&tileB_block(row_b, col_b * 8))
            );
        }
        // overlap
        for (; k_iter + 1 < get<2>(BLK_MMA_MNK{}); k_iter++) {
            for (int m_iter = 0; m_iter < REGA_RMEM_SET{}; m_iter++) {
                int row = lid % 16 + m_iter * 16 + WARP_M_ID * REGA_RMEM_SET{} * 16;
                int col = lid / 16 + (k_iter + 1) * 2;
                ldmatrix_m8n8k8_x4(
                    (uint32_t*)(REGA[((k_iter + 1) % 2) * REGA_RMEM_SET{} + m_iter]),
                    (void*)&tileA_block(row, col * 8)
                );
            }
            for (int n_iter = 0; n_iter < REGB_RMEM_SET{}; n_iter++) {
                int row_b = lid % 16 + (k_iter + 1) * 16;
                int col_b = n_iter + REGB_RMEM_SET{} * WARP_N_ID;
                ldmatrix_m8n8k8_x2_trans(
                    (uint32_t*)(REGB[((k_iter + 1) % 2) * REGB_RMEM_SET{} + n_iter]),
                    (void*)(&tileB_block(row_b, col_b * 8))
                );
            }
            for (int m_iter = 0; m_iter < REGA_RMEM_SET{}; m_iter++) {
                for (int n_iter = 0; n_iter < REGB_RMEM_SET{}; n_iter++) {
                    mma_m16n8k16_fp32(
                        REGC(_, make_coord(m_iter, n_iter)).data(), 
                        (uint32_t*)(REGA[(k_iter % 2) * REGA_RMEM_SET{} + m_iter]), 
                        REGB[(k_iter % 2) * REGB_RMEM_SET{} + n_iter]);
                }
            }
        }
        // final compute
        for (int m_iter = 0; m_iter < REGA_RMEM_SET{}; m_iter++) {
            for (int n_iter = 0; n_iter < REGB_RMEM_SET{}; n_iter++) {
                mma_m16n8k16_fp32(
                    REGC(_, make_coord(m_iter, n_iter)).data(), 
                    (uint32_t*)(REGA[(k_iter % 2) * REGA_RMEM_SET{} + m_iter]), 
                    REGB[(k_iter % 2) * REGB_RMEM_SET{} + n_iter]);
            }
        }
                    
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class T_B_valOp, class T_C_valOp, class T_compressed_AOp, class T_tile_offsetsOp, class T_FlashLLMLoopBoundZeroOp, class T_FlashLLMLoopBoundUpOp, class T_BlkYLoopZeroOp, class T_BlockDimYOp, class T_BlkXLoopZeroOp, class T_BlockDimXOp, class T_FlashLLMLoadCompressedAVecSizeOp, class T_FlashLLMLoadCompressedANumRegOp, class T_ZerosOp, class T_G2rTileOffsetsLoadOp, class T_G2rLoadCompressedTileAOp, class T_R2sRestoreTileAOp, class T_G2sLoadTileBOp, class T_MergedMMAOp, class Tensor_o0, class TI0 = int, class TI1 = int, class TO0 = float, class REGA_RMEM_SET = Int<get<0>(BLK_MMA_MNK{}) / get<0>(WARP_MNK{})>, class REGB_RMEM_SET = Int<get<1>(BLK_MMA_MNK{}) / get<1>(WARP_MNK{})>>
class FlashLLMMainLoopOp {
public:
    // Inputs
    int _l, _r;
    int _idx; // i + _l
    int _iter; // i

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(REGA_RMEM_SET{}, REGB_RMEM_SET{})));
    Shape_o0 shape_o0;
    Tensor_o0 REGC;

    // Hardware parameters
    int tid, lid, wid;

    // Body Ops
    T_B_valOp &B_valOp_v;
    T_C_valOp &C_valOp_v;
    T_compressed_AOp &compressed_AOp_v;
    T_tile_offsetsOp &tile_offsetsOp_v;
    T_FlashLLMLoopBoundZeroOp &FlashLLMLoopBoundZeroOp_v;
    T_FlashLLMLoopBoundUpOp &FlashLLMLoopBoundUpOp_v;
    T_BlkYLoopZeroOp &BlkYLoopZeroOp_v;
    T_BlockDimYOp &BlockDimYOp_v;
    T_BlkXLoopZeroOp &BlkXLoopZeroOp_v;
    T_BlockDimXOp &BlockDimXOp_v;
    T_FlashLLMLoadCompressedAVecSizeOp &FlashLLMLoadCompressedAVecSizeOp_v;
    T_FlashLLMLoadCompressedANumRegOp &FlashLLMLoadCompressedANumRegOp_v;
    T_ZerosOp &ZerosOp_v;
    T_G2rTileOffsetsLoadOp &G2rTileOffsetsLoadOp_v;
    T_G2rLoadCompressedTileAOp &G2rLoadCompressedTileAOp_v;
    T_R2sRestoreTileAOp &R2sRestoreTileAOp_v;
    T_G2sLoadTileBOp &G2sLoadTileBOp_v;
    T_MergedMMAOp &MergedMMAOp_v;

    // Constructor
    CUTE_DEVICE FlashLLMMainLoopOp(int tid, int lid, int wid, T_B_valOp &B_valOp_v, T_C_valOp &C_valOp_v, T_compressed_AOp &compressed_AOp_v, T_tile_offsetsOp &tile_offsetsOp_v, T_FlashLLMLoopBoundZeroOp &FlashLLMLoopBoundZeroOp_v, T_FlashLLMLoopBoundUpOp &FlashLLMLoopBoundUpOp_v, T_BlkYLoopZeroOp &BlkYLoopZeroOp_v, T_BlockDimYOp &BlockDimYOp_v, T_BlkXLoopZeroOp &BlkXLoopZeroOp_v, T_BlockDimXOp &BlockDimXOp_v, T_FlashLLMLoadCompressedAVecSizeOp &FlashLLMLoadCompressedAVecSizeOp_v, T_FlashLLMLoadCompressedANumRegOp &FlashLLMLoadCompressedANumRegOp_v, T_ZerosOp &ZerosOp_v, T_G2rTileOffsetsLoadOp &G2rTileOffsetsLoadOp_v, T_G2rLoadCompressedTileAOp &G2rLoadCompressedTileAOp_v, T_R2sRestoreTileAOp &R2sRestoreTileAOp_v, T_G2sLoadTileBOp &G2sLoadTileBOp_v, T_MergedMMAOp &MergedMMAOp_v, Tensor_o0 REGC)
    : tid(tid), lid(lid), wid(wid), B_valOp_v(B_valOp_v), C_valOp_v(C_valOp_v), compressed_AOp_v(compressed_AOp_v), tile_offsetsOp_v(tile_offsetsOp_v), FlashLLMLoopBoundZeroOp_v(FlashLLMLoopBoundZeroOp_v), FlashLLMLoopBoundUpOp_v(FlashLLMLoopBoundUpOp_v), BlkYLoopZeroOp_v(BlkYLoopZeroOp_v), BlockDimYOp_v(BlockDimYOp_v), BlkXLoopZeroOp_v(BlkXLoopZeroOp_v), BlockDimXOp_v(BlockDimXOp_v), FlashLLMLoadCompressedAVecSizeOp_v(FlashLLMLoadCompressedAVecSizeOp_v), FlashLLMLoadCompressedANumRegOp_v(FlashLLMLoadCompressedANumRegOp_v), ZerosOp_v(ZerosOp_v), G2rTileOffsetsLoadOp_v(G2rTileOffsetsLoadOp_v), G2rLoadCompressedTileAOp_v(G2rLoadCompressedTileAOp_v), R2sRestoreTileAOp_v(R2sRestoreTileAOp_v), G2sLoadTileBOp_v(G2sLoadTileBOp_v), MergedMMAOp_v(MergedMMAOp_v), REGC(REGC) {}

    // Pipeline methods
    // short pipe
    CUTE_DEVICE void short_pipe(int k, int l, int r) {
        if (k == 0) {
            // i = 0
            
            
            // i = 1
            
        }
        else if (k == 1) {
            // i = 0
            G2rTileOffsetsLoadOp_v.template f<0>(
                tile_offsetsOp_v.template output<0>(),
                0,
                FlashLLMLoopBoundUpOp_v.template output<0>()(0)
            );
            
            
            // i = 1
            G2rLoadCompressedTileAOp_v.template f<0>(
                compressed_AOp_v.template output<0>(),
                G2rTileOffsetsLoadOp_v.template output<0, 0>()
            );
            R2sRestoreTileAOp_v.template f<0>(
                G2rLoadCompressedTileAOp_v.template output<0, 0>(),
                G2rTileOffsetsLoadOp_v.template output<0, 0>()
            );
            __pipeline_commit();
            __syncthreads();
            G2sLoadTileBOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                0
            );
            __pipeline_commit();
            
            
            // i = 2
            __pipeline_wait_prior(0);
            __syncthreads();
            MergedMMAOp_v.template f<0>(
                R2sRestoreTileAOp_v.template output<0, 0>(),
                G2sLoadTileBOp_v.template output<0, 0>()
            );
            
        }
        else if (k == 2) {
            // i = 0
            G2rTileOffsetsLoadOp_v.template f<0>(
                tile_offsetsOp_v.template output<0>(),
                0,
                FlashLLMLoopBoundUpOp_v.template output<0>()(0)
            );
            
            
            // i = 1
            G2rTileOffsetsLoadOp_v.template f<1>(
                tile_offsetsOp_v.template output<0>(),
                1,
                FlashLLMLoopBoundUpOp_v.template output<0>()(0)
            );
            G2rLoadCompressedTileAOp_v.template f<0>(
                compressed_AOp_v.template output<0>(),
                G2rTileOffsetsLoadOp_v.template output<0, 0>()
            );
            R2sRestoreTileAOp_v.template f<0>(
                G2rLoadCompressedTileAOp_v.template output<0, 0>(),
                G2rTileOffsetsLoadOp_v.template output<0, 0>()
            );
            __pipeline_commit();
            __syncthreads();
            G2sLoadTileBOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                0
            );
            __pipeline_commit();
            
            
            // i = 2
            G2rLoadCompressedTileAOp_v.template f<1>(
                compressed_AOp_v.template output<0>(),
                G2rTileOffsetsLoadOp_v.template output<0, 1>()
            );
            R2sRestoreTileAOp_v.template f<1>(
                G2rLoadCompressedTileAOp_v.template output<0, 1>(),
                G2rTileOffsetsLoadOp_v.template output<0, 1>()
            );
            __pipeline_commit();
            __syncthreads();
            G2sLoadTileBOp_v.template f<1>(
                B_valOp_v.template output<0>(),
                1
            );
            __pipeline_commit();
            __pipeline_wait_prior(2);
            __syncthreads();
            MergedMMAOp_v.template f<0>(
                R2sRestoreTileAOp_v.template output<0, 0>(),
                G2sLoadTileBOp_v.template output<0, 0>()
            );
            
            
            // i = 3
            __pipeline_wait_prior(0);
            __syncthreads();
            MergedMMAOp_v.template f<1>(
                R2sRestoreTileAOp_v.template output<0, 1>(),
                G2sLoadTileBOp_v.template output<0, 1>()
            );
            
        }
        else if (k == 3) {
            // i = 0
            G2rTileOffsetsLoadOp_v.template f<0>(
                tile_offsetsOp_v.template output<0>(),
                0,
                FlashLLMLoopBoundUpOp_v.template output<0>()(0)
            );
            
            
            // i = 1
            G2rTileOffsetsLoadOp_v.template f<1>(
                tile_offsetsOp_v.template output<0>(),
                1,
                FlashLLMLoopBoundUpOp_v.template output<0>()(0)
            );
            G2rLoadCompressedTileAOp_v.template f<0>(
                compressed_AOp_v.template output<0>(),
                G2rTileOffsetsLoadOp_v.template output<0, 0>()
            );
            R2sRestoreTileAOp_v.template f<0>(
                G2rLoadCompressedTileAOp_v.template output<0, 0>(),
                G2rTileOffsetsLoadOp_v.template output<0, 0>()
            );
            __pipeline_commit();
            __syncthreads();
            G2sLoadTileBOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                0
            );
            __pipeline_commit();
            
            
            // i = 2
            G2rTileOffsetsLoadOp_v.template f<0>(
                tile_offsetsOp_v.template output<0>(),
                2,
                FlashLLMLoopBoundUpOp_v.template output<0>()(0)
            );
            G2rLoadCompressedTileAOp_v.template f<1>(
                compressed_AOp_v.template output<0>(),
                G2rTileOffsetsLoadOp_v.template output<0, 1>()
            );
            R2sRestoreTileAOp_v.template f<1>(
                G2rLoadCompressedTileAOp_v.template output<0, 1>(),
                G2rTileOffsetsLoadOp_v.template output<0, 1>()
            );
            __pipeline_commit();
            __syncthreads();
            G2sLoadTileBOp_v.template f<1>(
                B_valOp_v.template output<0>(),
                1
            );
            __pipeline_commit();
            __pipeline_wait_prior(2);
            __syncthreads();
            MergedMMAOp_v.template f<0>(
                R2sRestoreTileAOp_v.template output<0, 0>(),
                G2sLoadTileBOp_v.template output<0, 0>()
            );
            
            
            // i = 3
            G2rLoadCompressedTileAOp_v.template f<0>(
                compressed_AOp_v.template output<0>(),
                G2rTileOffsetsLoadOp_v.template output<0, 0>()
            );
            R2sRestoreTileAOp_v.template f<0>(
                G2rLoadCompressedTileAOp_v.template output<0, 0>(),
                G2rTileOffsetsLoadOp_v.template output<0, 0>()
            );
            __pipeline_commit();
            __syncthreads();
            G2sLoadTileBOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                2
            );
            __pipeline_commit();
            __pipeline_wait_prior(2);
            __syncthreads();
            MergedMMAOp_v.template f<1>(
                R2sRestoreTileAOp_v.template output<0, 1>(),
                G2sLoadTileBOp_v.template output<0, 1>()
            );
            
            
            // i = 4
            __pipeline_wait_prior(0);
            __syncthreads();
            MergedMMAOp_v.template f<0>(
                R2sRestoreTileAOp_v.template output<0, 0>(),
                G2sLoadTileBOp_v.template output<0, 0>()
            );
            
        }
        
    }
    
    // fill
    CUTE_DEVICE void fill(int l, int r) {
        // stage 0..0
        // op_call |buf_idx=0| 0 G2rTileOffsetsLoadOp 
        G2rTileOffsetsLoadOp_v.template f<0>(
            tile_offsetsOp_v.template output<0>(),
            0,
            FlashLLMLoopBoundUpOp_v.template output<0>()(0)
        );
        // stage 0..1
        // op_call |buf_idx=1| 1 G2rTileOffsetsLoadOp 
        G2rTileOffsetsLoadOp_v.template f<1>(
            tile_offsetsOp_v.template output<0>(),
            1,
            FlashLLMLoopBoundUpOp_v.template output<0>()(0)
        );
        // op_call |buf_idx=0| 0 G2rLoadCompressedTileAOp 
        G2rLoadCompressedTileAOp_v.template f<0>(
            compressed_AOp_v.template output<0>(),
            G2rTileOffsetsLoadOp_v.template output<0, 0>()
        );
        // op_call |buf_idx=0| 0 R2sRestoreTileAOp 
        R2sRestoreTileAOp_v.template f<0>(
            G2rLoadCompressedTileAOp_v.template output<0, 0>(),
            G2rTileOffsetsLoadOp_v.template output<0, 0>()
        );
        __pipeline_commit();
        __syncthreads();
        // op_call |buf_idx=0| 0 G2sLoadTileBOp 
        G2sLoadTileBOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            0
        );
        __pipeline_commit();
        // stage 0..2
        // op_call |buf_idx=0| 2 G2rTileOffsetsLoadOp 
        G2rTileOffsetsLoadOp_v.template f<0>(
            tile_offsetsOp_v.template output<0>(),
            2,
            FlashLLMLoopBoundUpOp_v.template output<0>()(0)
        );
        // op_call |buf_idx=1| 1 G2rLoadCompressedTileAOp 
        G2rLoadCompressedTileAOp_v.template f<1>(
            compressed_AOp_v.template output<0>(),
            G2rTileOffsetsLoadOp_v.template output<0, 1>()
        );
        // op_call |buf_idx=1| 1 R2sRestoreTileAOp 
        R2sRestoreTileAOp_v.template f<1>(
            G2rLoadCompressedTileAOp_v.template output<0, 1>(),
            G2rTileOffsetsLoadOp_v.template output<0, 1>()
        );
        __pipeline_commit();
        __syncthreads();
        // op_call |buf_idx=1| 1 G2sLoadTileBOp 
        G2sLoadTileBOp_v.template f<1>(
            B_valOp_v.template output<0>(),
            1
        );
        __pipeline_commit();
        __pipeline_wait_prior(2);
        __syncthreads();
        // op_call |buf_idx=0| 0 MergedMMAOp 
        MergedMMAOp_v.template f<0>(
            R2sRestoreTileAOp_v.template output<0, 0>(),
            G2sLoadTileBOp_v.template output<0, 0>()
        );
    }
    
    // loop_step
    CUTE_DEVICE void loop_step(int i, int l, int r) {
        // step 0
        // op_call |buf_idx=1| i + (0) G2rTileOffsetsLoadOp 
        G2rTileOffsetsLoadOp_v.template f<1>(
            tile_offsetsOp_v.template output<0>(),
            i + (0),
            FlashLLMLoopBoundUpOp_v.template output<0>()(0)
        );
        // op_call |buf_idx=0| i + (-1) G2rLoadCompressedTileAOp 
        G2rLoadCompressedTileAOp_v.template f<0>(
            compressed_AOp_v.template output<0>(),
            G2rTileOffsetsLoadOp_v.template output<0, 0>()
        );
        // op_call |buf_idx=0| i + (-1) R2sRestoreTileAOp 
        R2sRestoreTileAOp_v.template f<0>(
            G2rLoadCompressedTileAOp_v.template output<0, 0>(),
            G2rTileOffsetsLoadOp_v.template output<0, 0>()
        );
        __pipeline_commit();
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) G2sLoadTileBOp 
        G2sLoadTileBOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            i + (-1)
        );
        __pipeline_commit();
        __pipeline_wait_prior(2);
        __syncthreads();
        // op_call |buf_idx=1| i + (-2) MergedMMAOp 
        MergedMMAOp_v.template f<1>(
            R2sRestoreTileAOp_v.template output<0, 1>(),
            G2sLoadTileBOp_v.template output<0, 1>()
        );
        // step 1
        // op_call |buf_idx=0| i + (1) G2rTileOffsetsLoadOp 
        G2rTileOffsetsLoadOp_v.template f<0>(
            tile_offsetsOp_v.template output<0>(),
            i + (1),
            FlashLLMLoopBoundUpOp_v.template output<0>()(0)
        );
        // op_call |buf_idx=1| i + (0) G2rLoadCompressedTileAOp 
        G2rLoadCompressedTileAOp_v.template f<1>(
            compressed_AOp_v.template output<0>(),
            G2rTileOffsetsLoadOp_v.template output<0, 1>()
        );
        // op_call |buf_idx=1| i + (0) R2sRestoreTileAOp 
        R2sRestoreTileAOp_v.template f<1>(
            G2rLoadCompressedTileAOp_v.template output<0, 1>(),
            G2rTileOffsetsLoadOp_v.template output<0, 1>()
        );
        __pipeline_commit();
        __syncthreads();
        // op_call |buf_idx=1| i + (0) G2sLoadTileBOp 
        G2sLoadTileBOp_v.template f<1>(
            B_valOp_v.template output<0>(),
            i + (0)
        );
        __pipeline_commit();
        __pipeline_wait_prior(2);
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) MergedMMAOp 
        MergedMMAOp_v.template f<0>(
            R2sRestoreTileAOp_v.template output<0, 0>(),
            G2sLoadTileBOp_v.template output<0, 0>()
        );
    }
    
    // remainder
    CUTE_DEVICE void remains_1(int i, int l, int r) {
        // total remains 1, current 0
        // op_call |buf_idx=1| i + (0) G2rTileOffsetsLoadOp 
        G2rTileOffsetsLoadOp_v.template f<1>(
            tile_offsetsOp_v.template output<0>(),
            i + (0),
            FlashLLMLoopBoundUpOp_v.template output<0>()(0)
        );
        // op_call |buf_idx=0| i + (-1) G2rLoadCompressedTileAOp 
        G2rLoadCompressedTileAOp_v.template f<0>(
            compressed_AOp_v.template output<0>(),
            G2rTileOffsetsLoadOp_v.template output<0, 0>()
        );
        // op_call |buf_idx=0| i + (-1) R2sRestoreTileAOp 
        R2sRestoreTileAOp_v.template f<0>(
            G2rLoadCompressedTileAOp_v.template output<0, 0>(),
            G2rTileOffsetsLoadOp_v.template output<0, 0>()
        );
        __pipeline_commit();
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) G2sLoadTileBOp 
        G2sLoadTileBOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            i + (-1)
        );
        __pipeline_commit();
        __pipeline_wait_prior(2);
        __syncthreads();
        // op_call |buf_idx=1| i + (-2) MergedMMAOp 
        MergedMMAOp_v.template f<1>(
            R2sRestoreTileAOp_v.template output<0, 1>(),
            G2sLoadTileBOp_v.template output<0, 1>()
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
        // op_call |buf_idx=0| i + (-1) G2rLoadCompressedTileAOp 
        G2rLoadCompressedTileAOp_v.template f<0>(
            compressed_AOp_v.template output<0>(),
            G2rTileOffsetsLoadOp_v.template output<0, 0>()
        );
        // op_call |buf_idx=0| i + (-1) R2sRestoreTileAOp 
        R2sRestoreTileAOp_v.template f<0>(
            G2rLoadCompressedTileAOp_v.template output<0, 0>(),
            G2rTileOffsetsLoadOp_v.template output<0, 0>()
        );
        __pipeline_commit();
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) G2sLoadTileBOp 
        G2sLoadTileBOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            i + (-1)
        );
        __pipeline_commit();
        __pipeline_wait_prior(2);
        __syncthreads();
        // op_call |buf_idx=1| i + (-2) MergedMMAOp 
        MergedMMAOp_v.template f<1>(
            R2sRestoreTileAOp_v.template output<0, 1>(),
            G2sLoadTileBOp_v.template output<0, 1>()
        );
        // stage 2..2
        __pipeline_wait_prior(0);
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) MergedMMAOp 
        MergedMMAOp_v.template f<0>(
            R2sRestoreTileAOp_v.template output<0, 0>(),
            G2sLoadTileBOp_v.template output<0, 0>()
        );
    }
    
    CUTE_DEVICE void empty_after_remain_1(int i, int l, int r) {
        // stage 1..2
        // op_call |buf_idx=1| i + (0) G2rLoadCompressedTileAOp 
        G2rLoadCompressedTileAOp_v.template f<1>(
            compressed_AOp_v.template output<0>(),
            G2rTileOffsetsLoadOp_v.template output<0, 1>()
        );
        // op_call |buf_idx=1| i + (0) R2sRestoreTileAOp 
        R2sRestoreTileAOp_v.template f<1>(
            G2rLoadCompressedTileAOp_v.template output<0, 1>(),
            G2rTileOffsetsLoadOp_v.template output<0, 1>()
        );
        __pipeline_commit();
        __syncthreads();
        // op_call |buf_idx=1| i + (0) G2sLoadTileBOp 
        G2sLoadTileBOp_v.template f<1>(
            B_valOp_v.template output<0>(),
            i + (0)
        );
        __pipeline_commit();
        __pipeline_wait_prior(2);
        __syncthreads();
        // op_call |buf_idx=0| i + (-1) MergedMMAOp 
        MergedMMAOp_v.template f<0>(
            R2sRestoreTileAOp_v.template output<0, 0>(),
            G2sLoadTileBOp_v.template output<0, 0>()
        );
        // stage 2..2
        __pipeline_wait_prior(0);
        __syncthreads();
        // op_call |buf_idx=1| i + (0) MergedMMAOp 
        MergedMMAOp_v.template f<1>(
            R2sRestoreTileAOp_v.template output<0, 1>(),
            G2sLoadTileBOp_v.template output<0, 1>()
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

template<class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class Tensor_o0, class REGA_RMEM_SET = Int<get<0>(BLK_MMA_MNK{}) / get<0>(WARP_MNK{})>, class REGB_RMEM_SET = Int<get<1>(BLK_MMA_MNK{}) / get<1>(WARP_MNK{})>, class TI0 = float, class TO0 = float>
class R2gCValStoreOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(Int<4>{}, make_shape(REGA_RMEM_SET{}, REGB_RMEM_SET{})));
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
        int WARP_M_ID = wid / get<1>(WARP_MNK{});
        int WARP_N_ID = wid % get<1>(WARP_MNK{});
        for (int i_tileM = 0; i_tileM < REGA_RMEM_SET{}; i_tileM++) {
            for (int i_tileN = 0; i_tileN < REGB_RMEM_SET{}; i_tileN++) {
                int row = lid / 4 + i_tileM * 16 + WARP_M_ID * REGA_RMEM_SET{} * 16;
                int col = lid % 4 * 2 + i_tileN * 8 + WARP_N_ID * REGB_RMEM_SET{} * 8;
                C_val(col, row) = REGC(0, make_coord(i_tileM, i_tileN));
                C_val(col + 1, row) = REGC(1, make_coord(i_tileM, i_tileN));
                C_val(col, row + 8) = REGC(2, make_coord(i_tileM, i_tileN));
                C_val(col + 1, row + 8) = REGC(3, make_coord(i_tileM, i_tileN));
            }
        }
                    
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class T_B_valOp, class T_C_valOp, class T_compressed_AOp, class T_tile_offsetsOp, class T_FlashLLMLoopBoundZeroOp, class T_FlashLLMLoopBoundUpOp, class T_BlkYLoopZeroOp, class T_BlockDimYOp, class T_BlkXLoopZeroOp, class T_BlockDimXOp, class T_FlashLLMLoadCompressedAVecSizeOp, class T_FlashLLMLoadCompressedANumRegOp, class T_ZerosOp, class T_FlashLLMMainLoopOp, class T_R2gCValStoreOp>
class flash_llm_spmm_kernel_blk_x_loop {
public:
    // Inputs
    int _idx;

    // Hardware parameters
    int tid, lid, wid;

    // Body Ops
    T_B_valOp &B_valOp_v;
    T_C_valOp &C_valOp_v;
    T_compressed_AOp &compressed_AOp_v;
    T_tile_offsetsOp &tile_offsetsOp_v;
    T_FlashLLMLoopBoundZeroOp &FlashLLMLoopBoundZeroOp_v;
    T_FlashLLMLoopBoundUpOp &FlashLLMLoopBoundUpOp_v;
    T_BlkYLoopZeroOp &BlkYLoopZeroOp_v;
    T_BlockDimYOp &BlockDimYOp_v;
    T_BlkXLoopZeroOp &BlkXLoopZeroOp_v;
    T_BlockDimXOp &BlockDimXOp_v;
    T_FlashLLMLoadCompressedAVecSizeOp &FlashLLMLoadCompressedAVecSizeOp_v;
    T_FlashLLMLoadCompressedANumRegOp &FlashLLMLoadCompressedANumRegOp_v;
    T_ZerosOp &ZerosOp_v;
    T_FlashLLMMainLoopOp &FlashLLMMainLoopOp_v;
    T_R2gCValStoreOp &R2gCValStoreOp_v;

    // Constructor
    CUTE_DEVICE flash_llm_spmm_kernel_blk_x_loop(int tid, int lid, int wid, T_B_valOp &B_valOp_v, T_C_valOp &C_valOp_v, T_compressed_AOp &compressed_AOp_v, T_tile_offsetsOp &tile_offsetsOp_v, T_FlashLLMLoopBoundZeroOp &FlashLLMLoopBoundZeroOp_v, T_FlashLLMLoopBoundUpOp &FlashLLMLoopBoundUpOp_v, T_BlkYLoopZeroOp &BlkYLoopZeroOp_v, T_BlockDimYOp &BlockDimYOp_v, T_BlkXLoopZeroOp &BlkXLoopZeroOp_v, T_BlockDimXOp &BlockDimXOp_v, T_FlashLLMLoadCompressedAVecSizeOp &FlashLLMLoadCompressedAVecSizeOp_v, T_FlashLLMLoadCompressedANumRegOp &FlashLLMLoadCompressedANumRegOp_v, T_ZerosOp &ZerosOp_v, T_FlashLLMMainLoopOp &FlashLLMMainLoopOp_v, T_R2gCValStoreOp &R2gCValStoreOp_v)
    : tid(tid), lid(lid), wid(wid), B_valOp_v(B_valOp_v), C_valOp_v(C_valOp_v), compressed_AOp_v(compressed_AOp_v), tile_offsetsOp_v(tile_offsetsOp_v), FlashLLMLoopBoundZeroOp_v(FlashLLMLoopBoundZeroOp_v), FlashLLMLoopBoundUpOp_v(FlashLLMLoopBoundUpOp_v), BlkYLoopZeroOp_v(BlkYLoopZeroOp_v), BlockDimYOp_v(BlockDimYOp_v), BlkXLoopZeroOp_v(BlkXLoopZeroOp_v), BlockDimXOp_v(BlockDimXOp_v), FlashLLMLoadCompressedAVecSizeOp_v(FlashLLMLoadCompressedAVecSizeOp_v), FlashLLMLoadCompressedANumRegOp_v(FlashLLMLoadCompressedANumRegOp_v), ZerosOp_v(ZerosOp_v), FlashLLMMainLoopOp_v(FlashLLMMainLoopOp_v), R2gCValStoreOp_v(R2gCValStoreOp_v) {}

    // Body
    CUTE_DEVICE void body() {
        FlashLLMLoadCompressedAVecSizeOp_v.template f<>();
        FlashLLMLoadCompressedANumRegOp_v.template f<>();
        ZerosOp_v.template f<>();
        FlashLLMMainLoopOp_v.template f<>(
            FlashLLMLoopBoundZeroOp_v.template output<0>()(0),
            FlashLLMLoopBoundUpOp_v.template output<0>()(0)
        );
        R2gCValStoreOp_v.template f<>(
            FlashLLMMainLoopOp_v.template output<0>()
        );
    }

    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        this->_idx = blockIdx.x;
        body();
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class T_B_valOp, class T_C_valOp, class T_compressed_AOp, class T_tile_offsetsOp, class T_FlashLLMLoopBoundZeroOp, class T_FlashLLMLoopBoundUpOp, class T_BlkYLoopZeroOp, class T_BlockDimYOp, class T_BlkXLoopZeroOp, class T_BlockDimXOp, class T_flash_llm_spmm_kernel_blk_x_loop>
class flash_spmm_kernel_blk_y_loop {
public:
    // Inputs
    int _idx;

    // Hardware parameters
    int tid, lid, wid;

    // Body Ops
    T_B_valOp &B_valOp_v;
    T_C_valOp &C_valOp_v;
    T_compressed_AOp &compressed_AOp_v;
    T_tile_offsetsOp &tile_offsetsOp_v;
    T_FlashLLMLoopBoundZeroOp &FlashLLMLoopBoundZeroOp_v;
    T_FlashLLMLoopBoundUpOp &FlashLLMLoopBoundUpOp_v;
    T_BlkYLoopZeroOp &BlkYLoopZeroOp_v;
    T_BlockDimYOp &BlockDimYOp_v;
    T_BlkXLoopZeroOp &BlkXLoopZeroOp_v;
    T_BlockDimXOp &BlockDimXOp_v;
    T_flash_llm_spmm_kernel_blk_x_loop &flash_llm_spmm_kernel_blk_x_loop_v;

    // Constructor
    CUTE_DEVICE flash_spmm_kernel_blk_y_loop(int tid, int lid, int wid, T_B_valOp &B_valOp_v, T_C_valOp &C_valOp_v, T_compressed_AOp &compressed_AOp_v, T_tile_offsetsOp &tile_offsetsOp_v, T_FlashLLMLoopBoundZeroOp &FlashLLMLoopBoundZeroOp_v, T_FlashLLMLoopBoundUpOp &FlashLLMLoopBoundUpOp_v, T_BlkYLoopZeroOp &BlkYLoopZeroOp_v, T_BlockDimYOp &BlockDimYOp_v, T_BlkXLoopZeroOp &BlkXLoopZeroOp_v, T_BlockDimXOp &BlockDimXOp_v, T_flash_llm_spmm_kernel_blk_x_loop &flash_llm_spmm_kernel_blk_x_loop_v)
    : tid(tid), lid(lid), wid(wid), B_valOp_v(B_valOp_v), C_valOp_v(C_valOp_v), compressed_AOp_v(compressed_AOp_v), tile_offsetsOp_v(tile_offsetsOp_v), FlashLLMLoopBoundZeroOp_v(FlashLLMLoopBoundZeroOp_v), FlashLLMLoopBoundUpOp_v(FlashLLMLoopBoundUpOp_v), BlkYLoopZeroOp_v(BlkYLoopZeroOp_v), BlockDimYOp_v(BlockDimYOp_v), BlkXLoopZeroOp_v(BlkXLoopZeroOp_v), BlockDimXOp_v(BlockDimXOp_v), flash_llm_spmm_kernel_blk_x_loop_v(flash_llm_spmm_kernel_blk_x_loop_v) {}

    // Body
    CUTE_DEVICE void body() {
        BlkXLoopZeroOp_v.template f<>();
        BlockDimXOp_v.template f<>();
        flash_llm_spmm_kernel_blk_x_loop_v.template f<>();
    }

    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        this->_idx = blockIdx.y;
        body();
    }
};

// GLOBAL FUNCTION
template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK, class REGA_RMEM_SET, class REGB_RMEM_SET, class compressed_A_reg_set>
__global__ void flash_llm_spmm_kernel_sstage_rstage(
    half *dB_val,
    float *dC_val,
    uint32_t *dcompressed_A,
    int *dtile_offsets,
    int K, int M, int N, int NumOffsets, int nnz
) {
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int wid = tid / 32;

    using B_valOp_t = B_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    B_valOp_t B_valOp_v{tid, lid, wid, K, N, dB_val};
    using C_valOp_t = C_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    C_valOp_t C_valOp_v{tid, lid, wid, N, M, dC_val};
    using compressed_AOp_t = compressed_AOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    compressed_AOp_t compressed_AOp_v{tid, lid, wid, nnz, dcompressed_A};
    using tile_offsetsOp_t = tile_offsetsOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    tile_offsetsOp_t tile_offsetsOp_v{tid, lid, wid, NumOffsets, dtile_offsets};
    using FlashLLMLoopBoundZeroOp_t = FlashLLMLoopBoundZeroOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using FlashLLMLoopBoundZeroOp_layout_o0 = typename FlashLLMLoopBoundZeroOp_t::Layout_o0;
    int FlashLLMLoopBoundZeroOp_tensor_o0[cosize_v<FlashLLMLoopBoundZeroOp_layout_o0>];
    FlashLLMLoopBoundZeroOp_t FlashLLMLoopBoundZeroOp_v{tid, lid, wid, FlashLLMLoopBoundZeroOp_tensor_o0};
    using FlashLLMLoopBoundUpOp_t = FlashLLMLoopBoundUpOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using FlashLLMLoopBoundUpOp_layout_o0 = typename FlashLLMLoopBoundUpOp_t::Layout_o0;
    int FlashLLMLoopBoundUpOp_tensor_o0[cosize_v<FlashLLMLoopBoundUpOp_layout_o0>];
    FlashLLMLoopBoundUpOp_t FlashLLMLoopBoundUpOp_v{
        tid, lid, wid, 
        FlashLLMLoopBoundUpOp_tensor_o0,
    };
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
    using FlashLLMLoadCompressedAVecSizeOp_t = FlashLLMLoadCompressedAVecSizeOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using FlashLLMLoadCompressedAVecSizeOp_layout_o0 = typename FlashLLMLoadCompressedAVecSizeOp_t::Layout_o0;
    int FlashLLMLoadCompressedAVecSizeOp_tensor_o0[cosize_v<FlashLLMLoadCompressedAVecSizeOp_layout_o0>];
    FlashLLMLoadCompressedAVecSizeOp_t FlashLLMLoadCompressedAVecSizeOp_v{tid, lid, wid, FlashLLMLoadCompressedAVecSizeOp_tensor_o0};
    using FlashLLMLoadCompressedANumRegOp_t = FlashLLMLoadCompressedANumRegOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using FlashLLMLoadCompressedANumRegOp_layout_o0 = typename FlashLLMLoadCompressedANumRegOp_t::Layout_o0;
    int FlashLLMLoadCompressedANumRegOp_tensor_o0[cosize_v<FlashLLMLoadCompressedANumRegOp_layout_o0>];
    FlashLLMLoadCompressedANumRegOp_t FlashLLMLoadCompressedANumRegOp_v{tid, lid, wid, FlashLLMLoadCompressedANumRegOp_tensor_o0};
    using ZerosOp_t = ZerosOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using ZerosOp_layout_o0 = typename ZerosOp_t::Layout_o0;
    float ZerosOp_tensor_o0[cosize_v<ZerosOp_layout_o0>];
    ZerosOp_t ZerosOp_v{tid, lid, wid, ZerosOp_tensor_o0};
    using G2rTileOffsetsLoadOp_t = G2rTileOffsetsLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2rTileOffsetsLoadOp_layout_o0 = typename G2rTileOffsetsLoadOp_t::Layout_o0;
    int G2rTileOffsetsLoadOp_tensor_o0[cosize_v<G2rTileOffsetsLoadOp_layout_o0>];
    G2rTileOffsetsLoadOp_t G2rTileOffsetsLoadOp_v{
        tid, lid, wid, 
        G2rTileOffsetsLoadOp_tensor_o0, 
        NumOffsets,
    };
    using G2rLoadCompressedTileAOp_t = G2rLoadCompressedTileAOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2rLoadCompressedTileAOp_layout_o0 = typename G2rLoadCompressedTileAOp_t::Layout_o0;
    uint32_t G2rLoadCompressedTileAOp_tensor_o0[cosize_v<G2rLoadCompressedTileAOp_layout_o0>];
    G2rLoadCompressedTileAOp_t G2rLoadCompressedTileAOp_v{
        tid, lid, wid, 
        G2rLoadCompressedTileAOp_tensor_o0, 
        nnz,
    };
    using R2sRestoreTileAOp_t = R2sRestoreTileAOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using R2sRestoreTileAOp_layout_o0 = typename R2sRestoreTileAOp_t::Layout_o0;
    __shared__ half R2sRestoreTileAOp_tensor_o0[cosize_v<R2sRestoreTileAOp_layout_o0>];
    R2sRestoreTileAOp_t R2sRestoreTileAOp_v{
        tid, lid, wid, 
        R2sRestoreTileAOp_tensor_o0,
    };
    using G2sLoadTileBOp_t = G2sLoadTileBOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sLoadTileBOp_layout_o0 = typename G2sLoadTileBOp_t::Layout_o0;
    __shared__ half G2sLoadTileBOp_tensor_o0[cosize_v<G2sLoadTileBOp_layout_o0>];
    G2sLoadTileBOp_t G2sLoadTileBOp_v{
        tid, lid, wid, 
        G2sLoadTileBOp_tensor_o0, 
        K,
    };
    using MergedMMAOp_t = MergedMMAOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, std::decay_t<decltype(ZerosOp_v.template output<0>())>>;
    MergedMMAOp_t MergedMMAOp_v{
        tid, lid, wid, 
        ZerosOp_v.template output<0>(),
    };
    using FlashLLMMainLoopOp_t = FlashLLMMainLoopOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, B_valOp_t, C_valOp_t, compressed_AOp_t, tile_offsetsOp_t, FlashLLMLoopBoundZeroOp_t, FlashLLMLoopBoundUpOp_t, BlkYLoopZeroOp_t, BlockDimYOp_t, BlkXLoopZeroOp_t, BlockDimXOp_t, FlashLLMLoadCompressedAVecSizeOp_t, FlashLLMLoadCompressedANumRegOp_t, ZerosOp_t, G2rTileOffsetsLoadOp_t, G2rLoadCompressedTileAOp_t, R2sRestoreTileAOp_t, G2sLoadTileBOp_t, MergedMMAOp_t, std::decay_t<decltype(MergedMMAOp_v.template output<0>())>>;
    FlashLLMMainLoopOp_t FlashLLMMainLoopOp_v{tid, lid, wid, B_valOp_v, C_valOp_v, compressed_AOp_v, tile_offsetsOp_v, FlashLLMLoopBoundZeroOp_v, FlashLLMLoopBoundUpOp_v, BlkYLoopZeroOp_v, BlockDimYOp_v, BlkXLoopZeroOp_v, BlockDimXOp_v, FlashLLMLoadCompressedAVecSizeOp_v, FlashLLMLoadCompressedANumRegOp_v, ZerosOp_v, G2rTileOffsetsLoadOp_v, G2rLoadCompressedTileAOp_v, R2sRestoreTileAOp_v, G2sLoadTileBOp_v, MergedMMAOp_v, MergedMMAOp_v.template output<0>()};
    using R2gCValStoreOp_t = R2gCValStoreOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, std::decay_t<decltype(C_valOp_v.template output<0>())>>;
    R2gCValStoreOp_t R2gCValStoreOp_v{
        tid, lid, wid, 
        C_valOp_v.template output<0>(),
    };
    using flash_llm_spmm_kernel_blk_x_loop_t = flash_llm_spmm_kernel_blk_x_loop<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, B_valOp_t, C_valOp_t, compressed_AOp_t, tile_offsetsOp_t, FlashLLMLoopBoundZeroOp_t, FlashLLMLoopBoundUpOp_t, BlkYLoopZeroOp_t, BlockDimYOp_t, BlkXLoopZeroOp_t, BlockDimXOp_t, FlashLLMLoadCompressedAVecSizeOp_t, FlashLLMLoadCompressedANumRegOp_t, ZerosOp_t, FlashLLMMainLoopOp_t, R2gCValStoreOp_t>;
    flash_llm_spmm_kernel_blk_x_loop_t flash_llm_spmm_kernel_blk_x_loop_v{tid, lid, wid, B_valOp_v, C_valOp_v, compressed_AOp_v, tile_offsetsOp_v, FlashLLMLoopBoundZeroOp_v, FlashLLMLoopBoundUpOp_v, BlkYLoopZeroOp_v, BlockDimYOp_v, BlkXLoopZeroOp_v, BlockDimXOp_v, FlashLLMLoadCompressedAVecSizeOp_v, FlashLLMLoadCompressedANumRegOp_v, ZerosOp_v, FlashLLMMainLoopOp_v, R2gCValStoreOp_v};
    using flash_spmm_kernel_blk_y_loop_t = flash_spmm_kernel_blk_y_loop<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK, B_valOp_t, C_valOp_t, compressed_AOp_t, tile_offsetsOp_t, FlashLLMLoopBoundZeroOp_t, FlashLLMLoopBoundUpOp_t, BlkYLoopZeroOp_t, BlockDimYOp_t, BlkXLoopZeroOp_t, BlockDimXOp_t, flash_llm_spmm_kernel_blk_x_loop_t>;
    flash_spmm_kernel_blk_y_loop_t flash_spmm_kernel_blk_y_loop_v{tid, lid, wid, B_valOp_v, C_valOp_v, compressed_AOp_v, tile_offsetsOp_v, FlashLLMLoopBoundZeroOp_v, FlashLLMLoopBoundUpOp_v, BlkYLoopZeroOp_v, BlockDimYOp_v, BlkXLoopZeroOp_v, BlockDimXOp_v, flash_llm_spmm_kernel_blk_x_loop_v};

    FlashLLMLoopBoundZeroOp_v.template f<>();
    FlashLLMLoopBoundUpOp_v.template f<>(
        ((K + get<2>(BLK_MNK{}) - 1) / get<2>(BLK_MNK{}))
    );
    BlkYLoopZeroOp_v.template f<>();
    BlockDimYOp_v.template f<>();
    flash_spmm_kernel_blk_y_loop_v.template f<>();
};