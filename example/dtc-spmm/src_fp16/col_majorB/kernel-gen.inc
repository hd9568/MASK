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

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = half>
class B_valOp {
public:
    // Outputs
    using Tensor_o0 = decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN, VARLEN)), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, VARLEN)));
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
    shape_o0(logical_divide(make_tensor(make_gmem_ptr(dB_val), make_shape(K, N)), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y)).shape()),
    layout_o0(logical_divide(make_tensor(make_gmem_ptr(dB_val), make_shape(K, N)), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y)).layout()),
    B_val(logical_divide(make_tensor(make_gmem_ptr(dB_val), make_shape(K, N)), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y))) {}

    // output
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return B_val;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = float>
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
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return C_val;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = int>
class val_coo_idxOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_coo_idx;

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE val_coo_idxOp(int tid, int lid, int wid, int nnz, TO0 *dval_coo_idx)
    : tid(tid), lid(lid), wid(wid),
    shape_o0(make_tensor(make_gmem_ptr(dval_coo_idx), make_shape(nnz)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_coo_idx), make_shape(nnz)).layout()),
    val_coo_idx(make_tensor(make_gmem_ptr(dval_coo_idx), make_shape(nnz))) {}

    // output
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_coo_idx;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = int>
class val_coo_offOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(Int<2>{}, VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_coo_off;

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE val_coo_offOp(int tid, int lid, int wid, int nnz_block, TO0 *dval_coo_off)
    : tid(tid), lid(lid), wid(wid),
    shape_o0(make_tensor(make_gmem_ptr(dval_coo_off), make_shape(Int<2>{}, nnz_block)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_coo_off), make_shape(Int<2>{}, nnz_block)).layout()),
    val_coo_off(make_tensor(make_gmem_ptr(dval_coo_off), make_shape(Int<2>{}, nnz_block))) {}

    // output
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_coo_off;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = half>
class val_coo_valOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_coo_val;

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE val_coo_valOp(int tid, int lid, int wid, int nnz, TO0 *dval_coo_val)
    : tid(tid), lid(lid), wid(wid),
    shape_o0(make_tensor(make_gmem_ptr(dval_coo_val), make_shape(nnz)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_coo_val), make_shape(nnz)).layout()),
    val_coo_val(make_tensor(make_gmem_ptr(dval_coo_val), make_shape(nnz))) {}

    // output
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_coo_val;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_sidx_size = Int<16>, class TO0 = int>
class val_sidxOp {
public:
    // Outputs
    using Tensor_o0 = decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(tile_sidx_size{}, VARLEN)));
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
    shape_o0(make_tensor(make_gmem_ptr(dval_sidx), make_shape(tile_sidx_size{}, nnz_block)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_sidx), make_shape(tile_sidx_size{}, nnz_block)).layout()),
    val_sidx(make_tensor(make_gmem_ptr(dval_sidx), make_shape(tile_sidx_size{}, nnz_block))) {}

    // output
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_sidx;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = int>
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
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_soff;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = int>
class BlkYLoopZeroOp {
public:
    // No Inputs
    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
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
    template<size_t output_idx>
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

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = int>
class BlockDimYOp {
public:
    // No Inputs
    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
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
    template<size_t output_idx>
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

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = int>
class BlkXLoopZeroOp {
public:
    // No Inputs
    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
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
    template<size_t output_idx>
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

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = int>
class BlockDimXOp {
public:
    // No Inputs
    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
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
    template<size_t output_idx>
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

template<class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TI0 = int, class TI1 = int, class TO0 = int, class TO1 = int>
class G2rSparseOffsetLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 val_soff;
    // TI1 i;

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 l;
    using Shape_o1 = decltype(make_shape(_1{}));
    using Layout_o1 = decltype(make_layout(Shape_o1{}));
    using Tensor_o1 = decltype(make_tensor(make_rmem_ptr((TO1*)nullptr), Layout_o1{}));
    Shape_o1 shape_o1;
    Layout_o1 layout_o1;
    Tensor_o1 r;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rSparseOffsetLoadOp(int tid, int lid, int wid, TO0 *l, TO1 *r, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)), l(make_tensor(make_rmem_ptr(l), Layout_o0{})), r(make_tensor(make_rmem_ptr(r), Layout_o1{})) {}

    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return l;
        }
        if constexpr (output_idx == 1) {
            return r;
        }
    }

    template<class Tensor_i0>
    CUTE_DEVICE void f(Tensor_i0 val_soff, TI1 i) {
        l(0) = val_soff(i);
        r(0) = val_soff(i + 1);
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = float>
class ZerosOp {
public:
    // No Inputs
    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{}))));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
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
    template<size_t output_idx>
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

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_sidx_size = Int<16>, class TI0 = int, class TI1 = int, class TO0 = int>
class G2sValSidxLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(tile_sidx_size{}, VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 val_sidx;
    // TI1 i;

    // Outputs
    using Shape_o0 = decltype(make_shape(tile_sidx_size{}, NBUF{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 tile_sidx_block;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2sValSidxLoadOp(int tid, int lid, int wid, TO0 *tile_sidx_block, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(tile_sidx_size{}, mode_i0_0)), tile_sidx_block(make_tensor(make_smem_ptr(tile_sidx_block), Layout_o0{})) {}

    template<size_t output_idx, size_t buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_sidx_block(_, buf_idx);
        }
    }
    
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_sidx_block;
        }
    }

    template<size_t buf_idx, class Tensor_i0>
    CUTE_DEVICE void f(Tensor_i0 val_sidx, TI1 i) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(val_sidx))));
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
    Shape_i0 shape_i0;
    // Tensor_i0 val_coo_off;
    // TI1 i;

    // Outputs
    using Shape_o0 = decltype(make_shape(_1{}, NBUF{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 l;
    using Shape_o1 = decltype(make_shape(_1{}, NBUF{}));
    using Layout_o1 = decltype(make_layout(Shape_o1{}));
    using Tensor_o1 = decltype(make_tensor(make_rmem_ptr((TO1*)nullptr), Layout_o1{}));
    Shape_o1 shape_o1;
    Layout_o1 layout_o1;
    Tensor_o1 r;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rCooAtomicFormatLoadOffOp(int tid, int lid, int wid, TO0 *l, TO1 *r, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(Int<2>{}, mode_i0_0)), l(make_tensor(make_rmem_ptr(l), Layout_o0{})), r(make_tensor(make_rmem_ptr(r), Layout_o1{})) {}

    template<size_t output_idx, size_t buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return l(_, buf_idx);
        }
        if constexpr (output_idx == 1) {
            return r(_, buf_idx);
        }
    }
    
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return l;
        }
        if constexpr (output_idx == 1) {
            return r;
        }
    }

    template<size_t buf_idx, class Tensor_i0>
    CUTE_DEVICE void f(Tensor_i0 val_coo_off, TI1 i) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(val_coo_off))));
        l(buf_idx) = val_coo_off(0, i);
        r(buf_idx) = val_coo_off(1, i);
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_sidx_size = Int<16>, class TI0 = int, class TO0 = int>
class S2rValSidxLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(tile_sidx_size{}));
    Shape_i0 shape_i0;
    // Tensor_i0 tile_sidx_block;

    // Outputs
    using Shape_o0 = decltype(make_shape(tile_sidx_size{}, NBUF{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 val_sidx_reg;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE S2rValSidxLoadOp(int tid, int lid, int wid, TO0 *val_sidx_reg)
    : tid(tid), lid(lid), wid(wid), val_sidx_reg(make_tensor(make_rmem_ptr(val_sidx_reg), Layout_o0{})) {}

    template<size_t output_idx, size_t buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_sidx_reg(_, buf_idx);
        }
    }
    
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_sidx_reg;
        }
    }

    template<size_t buf_idx, class Tensor_i0>
    CUTE_DEVICE void f(Tensor_i0 tile_sidx_block) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(tile_sidx_block))));
        auto src = tile_sidx_block;
        auto dst = val_sidx_reg(_, buf_idx);
        copy(src, dst);
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_coo_val_size = Int<256>, class TI0 = half, class TI1 = int, class TI2 = int, class TO0 = half>
class G2sCooAtomicFormatLoadValOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 val_coo_val;
    // TI1 ll;
    // TI2 rr;

    // Outputs
    using Shape_o0 = decltype(make_shape(tile_coo_val_size{}, NBUF{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 tile_coo_val_block;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2sCooAtomicFormatLoadValOp(int tid, int lid, int wid, TO0 *tile_coo_val_block, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)), tile_coo_val_block(make_tensor(make_smem_ptr(tile_coo_val_block), Layout_o0{})) {}

    template<size_t output_idx, size_t buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_coo_val_block(_, buf_idx);
        }
    }
    
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_coo_val_block;
        }
    }

    template<size_t buf_idx, class Tensor_i0>
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

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_coo_val_size = Int<256>, class TI0 = int, class TI1 = int, class TI2 = int, class TO0 = int, class TO1 = int>
class G2rCooAtomicFormatLoadIdxOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 val_coo_idx;
    // TI1 ll;
    // TI2 rr;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<8>{}, NBUF{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 coo_idx_reg;
    using Shape_o1 = decltype(make_shape(_1{}, NBUF{}));
    using Layout_o1 = decltype(make_layout(Shape_o1{}));
    using Tensor_o1 = decltype(make_tensor(make_rmem_ptr((TO1*)nullptr), Layout_o1{}));
    Shape_o1 shape_o1;
    Layout_o1 layout_o1;
    Tensor_o1 coo_range_reg;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rCooAtomicFormatLoadIdxOp(int tid, int lid, int wid, TO0 *coo_idx_reg, TO1 *coo_range_reg, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)), coo_idx_reg(make_tensor(make_rmem_ptr(coo_idx_reg), Layout_o0{})), coo_range_reg(make_tensor(make_rmem_ptr(coo_range_reg), Layout_o1{})) {}

    template<size_t output_idx, size_t buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return coo_idx_reg(_, buf_idx);
        }
        if constexpr (output_idx == 1) {
            return coo_range_reg(_, buf_idx);
        }
    }
    
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return coo_idx_reg;
        }
        if constexpr (output_idx == 1) {
            return coo_range_reg;
        }
    }

    template<size_t buf_idx, class Tensor_i0>
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

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TI0 = half, class TI1 = int, class TO0 = half>
class G2rBValLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN, get<1>(BLK_MNK{})));
    Shape_i0 shape_i0;
    // Tensor_i0 B_val;
    using Shape_i1 = decltype(make_shape(get<2>(BLK_MNK{})));
    Shape_i1 shape_i1;
    // Tensor_i1 val_sidx_reg;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, get<1>(BLK_MMA_MNK{}), NBUF{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 REGB;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rBValLoadOp(int tid, int lid, int wid, TO0 *REGB, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0, get<1>(BLK_MNK{}))), REGB(make_tensor(make_rmem_ptr(REGB), Layout_o0{})) {}

    template<size_t output_idx, size_t buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGB(_, _, buf_idx);
        }
    }
    
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGB;
        }
    }

    template<size_t buf_idx, class Tensor_i0, class Tensor_i1>
    CUTE_DEVICE void f(Tensor_i0 B_val, Tensor_i1 val_sidx_reg) {
        CUTE_STATIC_ASSERT_V((get<1>(shape_i0) == size<1>(shape(B_val))));
        CUTE_STATIC_ASSERT_V((get<0>(shape_i1) == size<0>(shape(val_sidx_reg))));
        for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
            int row = lid % 4 * 2;
            int col = n_iter * 8 + lid / 4;
            REGB(0, n_iter, buf_idx) = B_val(val_sidx_reg(row + 0), col);
            REGB(1, n_iter, buf_idx) = B_val(val_sidx_reg(row + 1), col);
            REGB(2, n_iter, buf_idx) = B_val(val_sidx_reg(row + 8), col);
            REGB(3, n_iter, buf_idx) = B_val(val_sidx_reg(row + 9), col);
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_coo_val_size = Int<256>, class TI0 = half, class TI1 = int, class TI2 = int, class TO0 = half>
class S2sCooAtomicValRestoreOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(tile_coo_val_size{}));
    Shape_i0 shape_i0;
    // Tensor_i0 tile_coo_val_block;
    using Shape_i1 = decltype(make_shape(Int<8>{}));
    Shape_i1 shape_i1;
    // Tensor_i1 coo_idx_reg;
    // TI2 nnz_num;

    // Outputs
    using Shape_o0 = decltype(make_shape(get<2>(BLK_MNK{}), get<0>(BLK_MNK{}), NBUF{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_smem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 tile_coo_restore_val_block;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE S2sCooAtomicValRestoreOp(int tid, int lid, int wid, TO0 *tile_coo_restore_val_block)
    : tid(tid), lid(lid), wid(wid), tile_coo_restore_val_block(make_tensor(make_smem_ptr(tile_coo_restore_val_block), Layout_o0{})) {}

    template<size_t output_idx, size_t buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_coo_restore_val_block(_, _, buf_idx);
        }
    }
    
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return tile_coo_restore_val_block;
        }
    }

    template<size_t buf_idx, class Tensor_i0, class Tensor_i1>
    CUTE_DEVICE void f(Tensor_i0 tile_coo_val_block, Tensor_i1 coo_idx_reg, TI2 nnz_num) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(tile_coo_val_block))));
        CUTE_STATIC_ASSERT_V((get<0>(shape_i1) == size<0>(shape(coo_idx_reg))));
        // TODO: use tiler to distribute the tensor and use clear(Tensor) to set zero
        for (int i_o2s = 0; i_o2s < 4; i_o2s++) {
            *((int*)(tile_coo_restore_val_block(_, _, buf_idx).data().get() + 32 * 2 * i_o2s + lid * 2)) = 0;
        }
        __syncthreads();
        for (int i_restore = 0; i_restore < 8; i_restore++) {
            int idx = coo_idx_reg(i_restore);
            if (lid * 8 + i_restore < nnz_num) {
                tile_coo_restore_val_block(_, _, buf_idx)(idx) = tile_coo_val_block(lid * 8 + i_restore);
            }
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TI0 = half, class TO0 = half>
class S2rAValLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(get<2>(BLK_MNK{}), get<0>(BLK_MNK{})));
    Shape_i0 shape_i0;
    // Tensor_i0 tile_coo_restore_val_block;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<8>{}, get<0>(BLK_MMA_MNK{}), NBUF{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 REGA;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE S2rAValLoadOp(int tid, int lid, int wid, TO0 *REGA)
    : tid(tid), lid(lid), wid(wid), REGA(make_tensor(make_rmem_ptr(REGA), Layout_o0{})) {}

    template<size_t output_idx, size_t buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGA(_, _, buf_idx);
        }
    }
    
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGA;
        }
    }

    template<size_t buf_idx, class Tensor_i0>
    CUTE_DEVICE void f(Tensor_i0 tile_coo_restore_val_block) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(tile_coo_restore_val_block))));
        CUTE_STATIC_ASSERT_V((get<1>(shape_i0) == size<1>(shape(tile_coo_restore_val_block))));
        
        __syncthreads();
        int row = lid % 16;
        int col = lid / 16;
        // TODO: bank conflict
        for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) { // now the m_iter would always be zero, which means the for-loop body would run only once
            ldmatrix_m8n8k8_x4(
                (uint32_t*)(REGA(_, _0{}, buf_idx).data()),
                (void*)&tile_coo_restore_val_block(col * 8, row)
            );
        }
    }
};

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class Tensor_o0, class TI0 = half, class TI1 = half, class TO0 = float>
class MmaOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(Int<8>{}, get<0>(BLK_MMA_MNK{})));
    Shape_i0 shape_i0;
    // Tensor_i0 REGA;
    using Shape_i1 = decltype(make_shape(Int<4>{}, get<1>(BLK_MMA_MNK{})));
    Shape_i1 shape_i1;
    // Tensor_i1 REGB;

    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{}))));
    Shape_o0 shape_o0;
    Tensor_o0 REGC;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE MmaOp(int tid, int lid, int wid, Tensor_o0 REGC)
    : tid(tid), lid(lid), wid(wid), REGC(REGC) {}

    template<size_t output_idx, size_t buf_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGC;
        }
    }
    
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return REGC;
        }
    }

    template<size_t buf_idx, class Tensor_i0, class Tensor_i1>
    CUTE_DEVICE void f(Tensor_i0 REGA, Tensor_i1 REGB) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(REGA))));
        CUTE_STATIC_ASSERT_V((get<1>(shape_i0) == size<1>(shape(REGA))));
        CUTE_STATIC_ASSERT_V((get<0>(shape_i1) == size<0>(shape(REGB))));
        CUTE_STATIC_ASSERT_V((get<1>(shape_i1) == size<1>(shape(REGB))));
        for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) {
            for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
                mma_m16n8k16_fp32(REGC(_, make_coord(m_iter, n_iter)).data(), (uint32_t*)(REGA(_, m_iter).data()), REGB(_, n_iter).data());
            }
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class T_B_valOp, class T_C_valOp, class T_val_coo_idxOp, class T_val_coo_offOp, class T_val_coo_valOp, class T_val_sidxOp, class T_val_soffOp, class T_BlkYLoopZeroOp, class T_BlockDimYOp, class T_BlkXLoopZeroOp, class T_BlockDimXOp, class T_G2rSparseOffsetLoadOp, class T_ZerosOp, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOffOp, class T_S2rValSidxLoadOp, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp, class Tensor_o0, class TI0 = int, class TI1 = int, class TO0 = float>
class DtcMainLoopOp {
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
    T_val_coo_idxOp &val_coo_idxOp_v;
    T_val_coo_offOp &val_coo_offOp_v;
    T_val_coo_valOp &val_coo_valOp_v;
    T_val_sidxOp &val_sidxOp_v;
    T_val_soffOp &val_soffOp_v;
    T_BlkYLoopZeroOp &BlkYLoopZeroOp_v;
    T_BlockDimYOp &BlockDimYOp_v;
    T_BlkXLoopZeroOp &BlkXLoopZeroOp_v;
    T_BlockDimXOp &BlockDimXOp_v;
    T_G2rSparseOffsetLoadOp &G2rSparseOffsetLoadOp_v;
    T_ZerosOp &ZerosOp_v;
    T_G2sValSidxLoadOp &G2sValSidxLoadOp_v;
    T_G2rCooAtomicFormatLoadOffOp &G2rCooAtomicFormatLoadOffOp_v;
    T_S2rValSidxLoadOp &S2rValSidxLoadOp_v;
    T_G2sCooAtomicFormatLoadValOp &G2sCooAtomicFormatLoadValOp_v;
    T_G2rCooAtomicFormatLoadIdxOp &G2rCooAtomicFormatLoadIdxOp_v;
    T_G2rBValLoadOp &G2rBValLoadOp_v;
    T_S2sCooAtomicValRestoreOp &S2sCooAtomicValRestoreOp_v;
    T_S2rAValLoadOp &S2rAValLoadOp_v;
    T_MmaOp &MmaOp_v;

    // Constructor
    CUTE_DEVICE DtcMainLoopOp(int tid, int lid, int wid, T_B_valOp &B_valOp_v, T_C_valOp &C_valOp_v, T_val_coo_idxOp &val_coo_idxOp_v, T_val_coo_offOp &val_coo_offOp_v, T_val_coo_valOp &val_coo_valOp_v, T_val_sidxOp &val_sidxOp_v, T_val_soffOp &val_soffOp_v, T_BlkYLoopZeroOp &BlkYLoopZeroOp_v, T_BlockDimYOp &BlockDimYOp_v, T_BlkXLoopZeroOp &BlkXLoopZeroOp_v, T_BlockDimXOp &BlockDimXOp_v, T_G2rSparseOffsetLoadOp &G2rSparseOffsetLoadOp_v, T_ZerosOp &ZerosOp_v, T_G2sValSidxLoadOp &G2sValSidxLoadOp_v, T_G2rCooAtomicFormatLoadOffOp &G2rCooAtomicFormatLoadOffOp_v, T_S2rValSidxLoadOp &S2rValSidxLoadOp_v, T_G2sCooAtomicFormatLoadValOp &G2sCooAtomicFormatLoadValOp_v, T_G2rCooAtomicFormatLoadIdxOp &G2rCooAtomicFormatLoadIdxOp_v, T_G2rBValLoadOp &G2rBValLoadOp_v, T_S2sCooAtomicValRestoreOp &S2sCooAtomicValRestoreOp_v, T_S2rAValLoadOp &S2rAValLoadOp_v, T_MmaOp &MmaOp_v, Tensor_o0 REGC)
    : tid(tid), lid(lid), wid(wid), B_valOp_v(B_valOp_v), C_valOp_v(C_valOp_v), val_coo_idxOp_v(val_coo_idxOp_v), val_coo_offOp_v(val_coo_offOp_v), val_coo_valOp_v(val_coo_valOp_v), val_sidxOp_v(val_sidxOp_v), val_soffOp_v(val_soffOp_v), BlkYLoopZeroOp_v(BlkYLoopZeroOp_v), BlockDimYOp_v(BlockDimYOp_v), BlkXLoopZeroOp_v(BlkXLoopZeroOp_v), BlockDimXOp_v(BlockDimXOp_v), G2rSparseOffsetLoadOp_v(G2rSparseOffsetLoadOp_v), ZerosOp_v(ZerosOp_v), G2sValSidxLoadOp_v(G2sValSidxLoadOp_v), G2rCooAtomicFormatLoadOffOp_v(G2rCooAtomicFormatLoadOffOp_v), S2rValSidxLoadOp_v(S2rValSidxLoadOp_v), G2sCooAtomicFormatLoadValOp_v(G2sCooAtomicFormatLoadValOp_v), G2rCooAtomicFormatLoadIdxOp_v(G2rCooAtomicFormatLoadIdxOp_v), G2rBValLoadOp_v(G2rBValLoadOp_v), S2sCooAtomicValRestoreOp_v(S2sCooAtomicValRestoreOp_v), S2rAValLoadOp_v(S2rAValLoadOp_v), MmaOp_v(MmaOp_v), REGC(REGC) {}

    // Pipeline methods
    // short pipe
    CUTE_DEVICE void short_pipe(int k, int l, int r) {
        if (k == 0) {
            // i = 0
            
            
            // i = 1
            
            
            // i = 2
            
        }
        else if (k == 1) {
            // i = 0
            G2sValSidxLoadOp_v.template f<0>(
                val_sidxOp_v.template output<0>(),
                l + 0
            );
            __pipeline_commit();
            G2rCooAtomicFormatLoadOffOp_v.template f<0>(
                val_coo_offOp_v.template output<0>(),
                l + 0
            );
            
            
            // i = 1
            G2sCooAtomicFormatLoadValOp_v.template f<0>(
                val_coo_valOp_v.template output<0>(),
                G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
                G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
            );
            __pipeline_commit();
            __pipeline_wait_prior(1);
            __syncthreads();
            S2rValSidxLoadOp_v.template f<0>(
                G2sValSidxLoadOp_v.template output<0, 0>()
            );
            G2rCooAtomicFormatLoadIdxOp_v.template f<0>(
                val_coo_idxOp_v.template output<0>(),
                G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
                G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
            );
            
            
            // i = 2
            G2rBValLoadOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                S2rValSidxLoadOp_v.template output<0, 0>()
            );
            __pipeline_wait_prior(0);
            __syncthreads();
            S2sCooAtomicValRestoreOp_v.template f<0>(
                G2sCooAtomicFormatLoadValOp_v.template output<0, 0>(),
                G2rCooAtomicFormatLoadIdxOp_v.template output<0, 0>(),
                G2rCooAtomicFormatLoadIdxOp_v.template output<1, 0>()(0)
            );
            S2rAValLoadOp_v.template f<0>(
                S2sCooAtomicValRestoreOp_v.template output<0, 0>()
            );
            
            
            // i = 3
            MmaOp_v.template f<0>(
                S2rAValLoadOp_v.template output<0, 0>(),
                G2rBValLoadOp_v.template output<0, 0>()
            );
            
        }
        else if (k == 2) {
            // i = 0
            G2sValSidxLoadOp_v.template f<0>(
                val_sidxOp_v.template output<0>(),
                l + 0
            );
            __pipeline_commit();
            G2rCooAtomicFormatLoadOffOp_v.template f<0>(
                val_coo_offOp_v.template output<0>(),
                l + 0
            );
            
            
            // i = 1
            G2sValSidxLoadOp_v.template f<1>(
                val_sidxOp_v.template output<0>(),
                l + 1
            );
            G2sCooAtomicFormatLoadValOp_v.template f<0>(
                val_coo_valOp_v.template output<0>(),
                G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
                G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
            );
            __pipeline_commit();
            G2rCooAtomicFormatLoadOffOp_v.template f<1>(
                val_coo_offOp_v.template output<0>(),
                l + 1
            );
            __pipeline_wait_prior(1);
            __syncthreads();
            S2rValSidxLoadOp_v.template f<0>(
                G2sValSidxLoadOp_v.template output<0, 0>()
            );
            G2rCooAtomicFormatLoadIdxOp_v.template f<0>(
                val_coo_idxOp_v.template output<0>(),
                G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
                G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
            );
            
            
            // i = 2
            G2sCooAtomicFormatLoadValOp_v.template f<1>(
                val_coo_valOp_v.template output<0>(),
                G2rCooAtomicFormatLoadOffOp_v.template output<0, 1>()(0),
                G2rCooAtomicFormatLoadOffOp_v.template output<1, 1>()(0)
            );
            __pipeline_commit();
            __pipeline_wait_prior(1);
            __syncthreads();
            S2rValSidxLoadOp_v.template f<1>(
                G2sValSidxLoadOp_v.template output<0, 1>()
            );
            G2rCooAtomicFormatLoadIdxOp_v.template f<1>(
                val_coo_idxOp_v.template output<0>(),
                G2rCooAtomicFormatLoadOffOp_v.template output<0, 1>()(0),
                G2rCooAtomicFormatLoadOffOp_v.template output<1, 1>()(0)
            );
            G2rBValLoadOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                S2rValSidxLoadOp_v.template output<0, 0>()
            );
            __pipeline_wait_prior(1);
            __syncthreads();
            S2sCooAtomicValRestoreOp_v.template f<0>(
                G2sCooAtomicFormatLoadValOp_v.template output<0, 0>(),
                G2rCooAtomicFormatLoadIdxOp_v.template output<0, 0>(),
                G2rCooAtomicFormatLoadIdxOp_v.template output<1, 0>()(0)
            );
            S2rAValLoadOp_v.template f<0>(
                S2sCooAtomicValRestoreOp_v.template output<0, 0>()
            );
            
            
            // i = 3
            G2rBValLoadOp_v.template f<1>(
                B_valOp_v.template output<0>(),
                S2rValSidxLoadOp_v.template output<0, 1>()
            );
            __pipeline_wait_prior(0);
            __syncthreads();
            S2sCooAtomicValRestoreOp_v.template f<1>(
                G2sCooAtomicFormatLoadValOp_v.template output<0, 1>(),
                G2rCooAtomicFormatLoadIdxOp_v.template output<0, 1>(),
                G2rCooAtomicFormatLoadIdxOp_v.template output<1, 1>()(0)
            );
            S2rAValLoadOp_v.template f<1>(
                S2sCooAtomicValRestoreOp_v.template output<0, 1>()
            );
            MmaOp_v.template f<0>(
                S2rAValLoadOp_v.template output<0, 0>(),
                G2rBValLoadOp_v.template output<0, 0>()
            );
            
            
            // i = 4
            MmaOp_v.template f<1>(
                S2rAValLoadOp_v.template output<0, 1>(),
                G2rBValLoadOp_v.template output<0, 1>()
            );
            
        }
        else if (k == 3) {
            // i = 0
            G2sValSidxLoadOp_v.template f<0>(
                val_sidxOp_v.template output<0>(),
                l + 0
            );
            __pipeline_commit();
            G2rCooAtomicFormatLoadOffOp_v.template f<0>(
                val_coo_offOp_v.template output<0>(),
                l + 0
            );
            
            
            // i = 1
            G2sValSidxLoadOp_v.template f<1>(
                val_sidxOp_v.template output<0>(),
                l + 1
            );
            G2sCooAtomicFormatLoadValOp_v.template f<0>(
                val_coo_valOp_v.template output<0>(),
                G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
                G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
            );
            __pipeline_commit();
            G2rCooAtomicFormatLoadOffOp_v.template f<1>(
                val_coo_offOp_v.template output<0>(),
                l + 1
            );
            __pipeline_wait_prior(1);
            __syncthreads();
            S2rValSidxLoadOp_v.template f<0>(
                G2sValSidxLoadOp_v.template output<0, 0>()
            );
            G2rCooAtomicFormatLoadIdxOp_v.template f<0>(
                val_coo_idxOp_v.template output<0>(),
                G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
                G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
            );
            
            
            // i = 2
            G2sValSidxLoadOp_v.template f<0>(
                val_sidxOp_v.template output<0>(),
                l + 2
            );
            G2sCooAtomicFormatLoadValOp_v.template f<1>(
                val_coo_valOp_v.template output<0>(),
                G2rCooAtomicFormatLoadOffOp_v.template output<0, 1>()(0),
                G2rCooAtomicFormatLoadOffOp_v.template output<1, 1>()(0)
            );
            __pipeline_commit();
            G2rCooAtomicFormatLoadOffOp_v.template f<0>(
                val_coo_offOp_v.template output<0>(),
                l + 2
            );
            __pipeline_wait_prior(1);
            __syncthreads();
            S2rValSidxLoadOp_v.template f<1>(
                G2sValSidxLoadOp_v.template output<0, 1>()
            );
            G2rCooAtomicFormatLoadIdxOp_v.template f<1>(
                val_coo_idxOp_v.template output<0>(),
                G2rCooAtomicFormatLoadOffOp_v.template output<0, 1>()(0),
                G2rCooAtomicFormatLoadOffOp_v.template output<1, 1>()(0)
            );
            G2rBValLoadOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                S2rValSidxLoadOp_v.template output<0, 0>()
            );
            __pipeline_wait_prior(1);
            __syncthreads();
            S2sCooAtomicValRestoreOp_v.template f<0>(
                G2sCooAtomicFormatLoadValOp_v.template output<0, 0>(),
                G2rCooAtomicFormatLoadIdxOp_v.template output<0, 0>(),
                G2rCooAtomicFormatLoadIdxOp_v.template output<1, 0>()(0)
            );
            S2rAValLoadOp_v.template f<0>(
                S2sCooAtomicValRestoreOp_v.template output<0, 0>()
            );
            
            
            // i = 3
            G2sCooAtomicFormatLoadValOp_v.template f<0>(
                val_coo_valOp_v.template output<0>(),
                G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
                G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
            );
            __pipeline_commit();
            __pipeline_wait_prior(1);
            __syncthreads();
            S2rValSidxLoadOp_v.template f<0>(
                G2sValSidxLoadOp_v.template output<0, 0>()
            );
            G2rCooAtomicFormatLoadIdxOp_v.template f<0>(
                val_coo_idxOp_v.template output<0>(),
                G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
                G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
            );
            G2rBValLoadOp_v.template f<1>(
                B_valOp_v.template output<0>(),
                S2rValSidxLoadOp_v.template output<0, 1>()
            );
            __pipeline_wait_prior(1);
            __syncthreads();
            S2sCooAtomicValRestoreOp_v.template f<1>(
                G2sCooAtomicFormatLoadValOp_v.template output<0, 1>(),
                G2rCooAtomicFormatLoadIdxOp_v.template output<0, 1>(),
                G2rCooAtomicFormatLoadIdxOp_v.template output<1, 1>()(0)
            );
            S2rAValLoadOp_v.template f<1>(
                S2sCooAtomicValRestoreOp_v.template output<0, 1>()
            );
            MmaOp_v.template f<0>(
                S2rAValLoadOp_v.template output<0, 0>(),
                G2rBValLoadOp_v.template output<0, 0>()
            );
            
            
            // i = 4
            G2rBValLoadOp_v.template f<0>(
                B_valOp_v.template output<0>(),
                S2rValSidxLoadOp_v.template output<0, 0>()
            );
            __pipeline_wait_prior(0);
            __syncthreads();
            S2sCooAtomicValRestoreOp_v.template f<0>(
                G2sCooAtomicFormatLoadValOp_v.template output<0, 0>(),
                G2rCooAtomicFormatLoadIdxOp_v.template output<0, 0>(),
                G2rCooAtomicFormatLoadIdxOp_v.template output<1, 0>()(0)
            );
            S2rAValLoadOp_v.template f<0>(
                S2sCooAtomicValRestoreOp_v.template output<0, 0>()
            );
            MmaOp_v.template f<1>(
                S2rAValLoadOp_v.template output<0, 1>(),
                G2rBValLoadOp_v.template output<0, 1>()
            );
            
            
            // i = 5
            MmaOp_v.template f<0>(
                S2rAValLoadOp_v.template output<0, 0>(),
                G2rBValLoadOp_v.template output<0, 0>()
            );
            
        }
        
    }
    
    // fill
    CUTE_DEVICE void fill(int l, int r) {
        // stage 0..0
        G2sValSidxLoadOp_v.template f<0>(
            val_sidxOp_v.template output<0>(),
            l + 0
        );
        __pipeline_commit();
        G2rCooAtomicFormatLoadOffOp_v.template f<0>(
            val_coo_offOp_v.template output<0>(),
            l + 0
        );
        // stage 0..1
        G2sValSidxLoadOp_v.template f<1>(
            val_sidxOp_v.template output<0>(),
            l + 1
        );
        G2sCooAtomicFormatLoadValOp_v.template f<0>(
            val_coo_valOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
        );
        __pipeline_commit();
        G2rCooAtomicFormatLoadOffOp_v.template f<1>(
            val_coo_offOp_v.template output<0>(),
            l + 1
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        S2rValSidxLoadOp_v.template f<0>(
            G2sValSidxLoadOp_v.template output<0, 0>()
        );
        G2rCooAtomicFormatLoadIdxOp_v.template f<0>(
            val_coo_idxOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
        );
        // stage 0..2
        G2sValSidxLoadOp_v.template f<0>(
            val_sidxOp_v.template output<0>(),
            l + 2
        );
        G2sCooAtomicFormatLoadValOp_v.template f<1>(
            val_coo_valOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 1>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 1>()(0)
        );
        __pipeline_commit();
        G2rCooAtomicFormatLoadOffOp_v.template f<0>(
            val_coo_offOp_v.template output<0>(),
            l + 2
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        S2rValSidxLoadOp_v.template f<1>(
            G2sValSidxLoadOp_v.template output<0, 1>()
        );
        G2rCooAtomicFormatLoadIdxOp_v.template f<1>(
            val_coo_idxOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 1>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 1>()(0)
        );
        G2rBValLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            S2rValSidxLoadOp_v.template output<0, 0>()
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        S2sCooAtomicValRestoreOp_v.template f<0>(
            G2sCooAtomicFormatLoadValOp_v.template output<0, 0>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<0, 0>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<1, 0>()(0)
        );
        __syncthreads();
        S2rAValLoadOp_v.template f<0>(
            S2sCooAtomicValRestoreOp_v.template output<0, 0>()
        );
    }
    
    // loop_step
    CUTE_DEVICE void loop_step(int i, int l, int r) {
        // step 0
        G2sValSidxLoadOp_v.template f<1>(
            val_sidxOp_v.template output<0>(),
            l + i + (0)
        );
        G2sCooAtomicFormatLoadValOp_v.template f<0>(
            val_coo_valOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
        );
        __pipeline_commit();
        G2rCooAtomicFormatLoadOffOp_v.template f<1>(
            val_coo_offOp_v.template output<0>(),
            l + i + (0)
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        S2rValSidxLoadOp_v.template f<0>(
            G2sValSidxLoadOp_v.template output<0, 0>()
        );
        G2rCooAtomicFormatLoadIdxOp_v.template f<0>(
            val_coo_idxOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
        );
        G2rBValLoadOp_v.template f<1>(
            B_valOp_v.template output<0>(),
            S2rValSidxLoadOp_v.template output<0, 1>()
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        S2sCooAtomicValRestoreOp_v.template f<1>(
            G2sCooAtomicFormatLoadValOp_v.template output<0, 1>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<0, 1>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<1, 1>()(0)
        );
        __syncthreads();
        S2rAValLoadOp_v.template f<1>(
            S2sCooAtomicValRestoreOp_v.template output<0, 1>()
        );
        MmaOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            G2rBValLoadOp_v.template output<0, 0>()
        );
        // step 1
        G2sValSidxLoadOp_v.template f<0>(
            val_sidxOp_v.template output<0>(),
            l + i + (1)
        );
        G2sCooAtomicFormatLoadValOp_v.template f<1>(
            val_coo_valOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 1>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 1>()(0)
        );
        __pipeline_commit();
        G2rCooAtomicFormatLoadOffOp_v.template f<0>(
            val_coo_offOp_v.template output<0>(),
            l + i + (1)
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        S2rValSidxLoadOp_v.template f<1>(
            G2sValSidxLoadOp_v.template output<0, 1>()
        );
        G2rCooAtomicFormatLoadIdxOp_v.template f<1>(
            val_coo_idxOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 1>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 1>()(0)
        );
        G2rBValLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            S2rValSidxLoadOp_v.template output<0, 0>()
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        S2sCooAtomicValRestoreOp_v.template f<0>(
            G2sCooAtomicFormatLoadValOp_v.template output<0, 0>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<0, 0>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<1, 0>()(0)
        );
        __syncthreads();
        S2rAValLoadOp_v.template f<0>(
            S2sCooAtomicValRestoreOp_v.template output<0, 0>()
        );
        MmaOp_v.template f<1>(
            S2rAValLoadOp_v.template output<0, 1>(),
            G2rBValLoadOp_v.template output<0, 1>()
        );
    }
    
    // remainder
    CUTE_DEVICE void remains_1(int i, int l, int r) {
        // total remains 1, current 0
        G2sValSidxLoadOp_v.template f<1>(
            val_sidxOp_v.template output<0>(),
            l + i + (0)
        );
        G2sCooAtomicFormatLoadValOp_v.template f<0>(
            val_coo_valOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
        );
        __pipeline_commit();
        G2rCooAtomicFormatLoadOffOp_v.template f<1>(
            val_coo_offOp_v.template output<0>(),
            l + i + (0)
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        S2rValSidxLoadOp_v.template f<0>(
            G2sValSidxLoadOp_v.template output<0, 0>()
        );
        G2rCooAtomicFormatLoadIdxOp_v.template f<0>(
            val_coo_idxOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
        );
        G2rBValLoadOp_v.template f<1>(
            B_valOp_v.template output<0>(),
            S2rValSidxLoadOp_v.template output<0, 1>()
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        S2sCooAtomicValRestoreOp_v.template f<1>(
            G2sCooAtomicFormatLoadValOp_v.template output<0, 1>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<0, 1>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<1, 1>()(0)
        );
        __syncthreads();
        S2rAValLoadOp_v.template f<1>(
            S2sCooAtomicValRestoreOp_v.template output<0, 1>()
        );
        MmaOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            G2rBValLoadOp_v.template output<0, 0>()
        );
    }
    
    CUTE_DEVICE void remainder(int i, int l, int r) {
        if (r - l - i == 1) {
            remains_1(i, l, r);
        }
    }
    
    // empties
    CUTE_DEVICE void empty_0(int i, int l, int r) {
        // stage 1..3
        G2sCooAtomicFormatLoadValOp_v.template f<1>(
            val_coo_valOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 1>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 1>()(0)
        );
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        S2rValSidxLoadOp_v.template f<1>(
            G2sValSidxLoadOp_v.template output<0, 1>()
        );
        G2rCooAtomicFormatLoadIdxOp_v.template f<1>(
            val_coo_idxOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 1>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 1>()(0)
        );
        G2rBValLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            S2rValSidxLoadOp_v.template output<0, 0>()
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        S2sCooAtomicValRestoreOp_v.template f<0>(
            G2sCooAtomicFormatLoadValOp_v.template output<0, 0>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<0, 0>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<1, 0>()(0)
        );
        __syncthreads();
        S2rAValLoadOp_v.template f<0>(
            S2sCooAtomicValRestoreOp_v.template output<0, 0>()
        );
        MmaOp_v.template f<1>(
            S2rAValLoadOp_v.template output<0, 1>(),
            G2rBValLoadOp_v.template output<0, 1>()
        );
        // stage 2..3
        G2rBValLoadOp_v.template f<1>(
            B_valOp_v.template output<0>(),
            S2rValSidxLoadOp_v.template output<0, 1>()
        );
        __pipeline_wait_prior(0);
        __syncthreads();
        S2sCooAtomicValRestoreOp_v.template f<1>(
            G2sCooAtomicFormatLoadValOp_v.template output<0, 1>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<0, 1>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<1, 1>()(0)
        );
        __syncthreads();
        S2rAValLoadOp_v.template f<1>(
            S2sCooAtomicValRestoreOp_v.template output<0, 1>()
        );
        MmaOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            G2rBValLoadOp_v.template output<0, 0>()
        );
        // stage 3..3
        MmaOp_v.template f<1>(
            S2rAValLoadOp_v.template output<0, 1>(),
            G2rBValLoadOp_v.template output<0, 1>()
        );
    }
    
    CUTE_DEVICE void empty_1(int i, int l, int r) {
        // stage 1..3
        G2sCooAtomicFormatLoadValOp_v.template f<0>(
            val_coo_valOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
        );
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        S2rValSidxLoadOp_v.template f<0>(
            G2sValSidxLoadOp_v.template output<0, 0>()
        );
        G2rCooAtomicFormatLoadIdxOp_v.template f<0>(
            val_coo_idxOp_v.template output<0>(),
            G2rCooAtomicFormatLoadOffOp_v.template output<0, 0>()(0),
            G2rCooAtomicFormatLoadOffOp_v.template output<1, 0>()(0)
        );
        G2rBValLoadOp_v.template f<1>(
            B_valOp_v.template output<0>(),
            S2rValSidxLoadOp_v.template output<0, 1>()
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        S2sCooAtomicValRestoreOp_v.template f<1>(
            G2sCooAtomicFormatLoadValOp_v.template output<0, 1>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<0, 1>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<1, 1>()(0)
        );
        __syncthreads();
        S2rAValLoadOp_v.template f<1>(
            S2sCooAtomicValRestoreOp_v.template output<0, 1>()
        );
        MmaOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            G2rBValLoadOp_v.template output<0, 0>()
        );
        // stage 2..3
        G2rBValLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            S2rValSidxLoadOp_v.template output<0, 0>()
        );
        __pipeline_wait_prior(0);
        __syncthreads();
        S2sCooAtomicValRestoreOp_v.template f<0>(
            G2sCooAtomicFormatLoadValOp_v.template output<0, 0>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<0, 0>(),
            G2rCooAtomicFormatLoadIdxOp_v.template output<1, 0>()(0)
        );
        __syncthreads();
        S2rAValLoadOp_v.template f<0>(
            S2sCooAtomicValRestoreOp_v.template output<0, 0>()
        );
        MmaOp_v.template f<1>(
            S2rAValLoadOp_v.template output<0, 1>(),
            G2rBValLoadOp_v.template output<0, 1>()
        );
        // stage 3..3
        MmaOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            G2rBValLoadOp_v.template output<0, 0>()
        );
    }
    
    CUTE_DEVICE void empty(int i, int l, int r) {
        if ((r - l - i + (3)) % 2 == 0) {
            empty_0(i, l, r);
        }
        else if ((r - l - i + (3)) % 2 == 1) {
            empty_1(i, l, r);
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

    template<size_t output_idx>
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

template<class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class Tensor_o0, class TI0 = float, class TO0 = float>
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

    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return C_val;
        }
    }

    template<class Tensor_i0>
    CUTE_DEVICE void f(Tensor_i0 REGC) {
        CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(REGC))));
        CUTE_STATIC_ASSERT_V((get<0>(get<1>(shape_i0)) == size<0>(get<1>(shape(REGC)))));
        CUTE_STATIC_ASSERT_V((get<1>(get<1>(shape_i0)) == size<1>(get<1>(shape(REGC)))));
        //! write back
        for (int i_tileN = 0; i_tileN < get<1>(BLK_MNK{}) / 8; i_tileN++) {
            int row = lid / 4;
            int col = i_tileN * 8 + lid % 4 * 2;
            C_val(col, row) = REGC(0, i_tileN);
            C_val(col + 1, row) = REGC(1, i_tileN);
            C_val(col, row + 8) = REGC(2, i_tileN);
            C_val(col + 1, row + 8) = REGC(3, i_tileN);
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class T_B_valOp, class T_C_valOp, class T_val_coo_idxOp, class T_val_coo_offOp, class T_val_coo_valOp, class T_val_sidxOp, class T_val_soffOp, class T_BlkYLoopZeroOp, class T_BlockDimYOp, class T_BlkXLoopZeroOp, class T_BlockDimXOp, class T_G2rSparseOffsetLoadOp, class T_ZerosOp, class T_DtcMainLoopOp, class T_R2gCValStoreOp>
class dtc_spmm_kernel_blk_x_loop {
public:
    // Inputs
    int _idx;

    // Hardware parameters
    int tid, lid, wid;

    // Body Ops
    T_B_valOp &B_valOp_v;
    T_C_valOp &C_valOp_v;
    T_val_coo_idxOp &val_coo_idxOp_v;
    T_val_coo_offOp &val_coo_offOp_v;
    T_val_coo_valOp &val_coo_valOp_v;
    T_val_sidxOp &val_sidxOp_v;
    T_val_soffOp &val_soffOp_v;
    T_BlkYLoopZeroOp &BlkYLoopZeroOp_v;
    T_BlockDimYOp &BlockDimYOp_v;
    T_BlkXLoopZeroOp &BlkXLoopZeroOp_v;
    T_BlockDimXOp &BlockDimXOp_v;
    T_G2rSparseOffsetLoadOp &G2rSparseOffsetLoadOp_v;
    T_ZerosOp &ZerosOp_v;
    T_DtcMainLoopOp &DtcMainLoopOp_v;
    T_R2gCValStoreOp &R2gCValStoreOp_v;

    // Constructor
    CUTE_DEVICE dtc_spmm_kernel_blk_x_loop(int tid, int lid, int wid, T_B_valOp &B_valOp_v, T_C_valOp &C_valOp_v, T_val_coo_idxOp &val_coo_idxOp_v, T_val_coo_offOp &val_coo_offOp_v, T_val_coo_valOp &val_coo_valOp_v, T_val_sidxOp &val_sidxOp_v, T_val_soffOp &val_soffOp_v, T_BlkYLoopZeroOp &BlkYLoopZeroOp_v, T_BlockDimYOp &BlockDimYOp_v, T_BlkXLoopZeroOp &BlkXLoopZeroOp_v, T_BlockDimXOp &BlockDimXOp_v, T_G2rSparseOffsetLoadOp &G2rSparseOffsetLoadOp_v, T_ZerosOp &ZerosOp_v, T_DtcMainLoopOp &DtcMainLoopOp_v, T_R2gCValStoreOp &R2gCValStoreOp_v)
    : tid(tid), lid(lid), wid(wid), B_valOp_v(B_valOp_v), C_valOp_v(C_valOp_v), val_coo_idxOp_v(val_coo_idxOp_v), val_coo_offOp_v(val_coo_offOp_v), val_coo_valOp_v(val_coo_valOp_v), val_sidxOp_v(val_sidxOp_v), val_soffOp_v(val_soffOp_v), BlkYLoopZeroOp_v(BlkYLoopZeroOp_v), BlockDimYOp_v(BlockDimYOp_v), BlkXLoopZeroOp_v(BlkXLoopZeroOp_v), BlockDimXOp_v(BlockDimXOp_v), G2rSparseOffsetLoadOp_v(G2rSparseOffsetLoadOp_v), ZerosOp_v(ZerosOp_v), DtcMainLoopOp_v(DtcMainLoopOp_v), R2gCValStoreOp_v(R2gCValStoreOp_v) {}

    // Body
    CUTE_DEVICE void body() {
        G2rSparseOffsetLoadOp_v.template f<>(
            val_soffOp_v.template output<0>(),
            blockIdx.x
        );
        ZerosOp_v.template f<>();
        DtcMainLoopOp_v.template f<>(
            G2rSparseOffsetLoadOp_v.template output<0>()(0),
            G2rSparseOffsetLoadOp_v.template output<1>()(0)
        );
        R2gCValStoreOp_v.template f<>(
            DtcMainLoopOp_v.template output<0>()
        );
    }

    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        this->_idx = blockIdx.x;
        body();
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class T_B_valOp, class T_C_valOp, class T_val_coo_idxOp, class T_val_coo_offOp, class T_val_coo_valOp, class T_val_sidxOp, class T_val_soffOp, class T_BlkYLoopZeroOp, class T_BlockDimYOp, class T_BlkXLoopZeroOp, class T_BlockDimXOp, class T_dtc_spmm_kernel_blk_x_loop>
class dtc_spmm_kernel_blk_y_loop {
public:
    // Inputs
    int _idx;

    // Hardware parameters
    int tid, lid, wid;

    // Body Ops
    T_B_valOp &B_valOp_v;
    T_C_valOp &C_valOp_v;
    T_val_coo_idxOp &val_coo_idxOp_v;
    T_val_coo_offOp &val_coo_offOp_v;
    T_val_coo_valOp &val_coo_valOp_v;
    T_val_sidxOp &val_sidxOp_v;
    T_val_soffOp &val_soffOp_v;
    T_BlkYLoopZeroOp &BlkYLoopZeroOp_v;
    T_BlockDimYOp &BlockDimYOp_v;
    T_BlkXLoopZeroOp &BlkXLoopZeroOp_v;
    T_BlockDimXOp &BlockDimXOp_v;
    T_dtc_spmm_kernel_blk_x_loop &dtc_spmm_kernel_blk_x_loop_v;

    // Constructor
    CUTE_DEVICE dtc_spmm_kernel_blk_y_loop(int tid, int lid, int wid, T_B_valOp &B_valOp_v, T_C_valOp &C_valOp_v, T_val_coo_idxOp &val_coo_idxOp_v, T_val_coo_offOp &val_coo_offOp_v, T_val_coo_valOp &val_coo_valOp_v, T_val_sidxOp &val_sidxOp_v, T_val_soffOp &val_soffOp_v, T_BlkYLoopZeroOp &BlkYLoopZeroOp_v, T_BlockDimYOp &BlockDimYOp_v, T_BlkXLoopZeroOp &BlkXLoopZeroOp_v, T_BlockDimXOp &BlockDimXOp_v, T_dtc_spmm_kernel_blk_x_loop &dtc_spmm_kernel_blk_x_loop_v)
    : tid(tid), lid(lid), wid(wid), B_valOp_v(B_valOp_v), C_valOp_v(C_valOp_v), val_coo_idxOp_v(val_coo_idxOp_v), val_coo_offOp_v(val_coo_offOp_v), val_coo_valOp_v(val_coo_valOp_v), val_sidxOp_v(val_sidxOp_v), val_soffOp_v(val_soffOp_v), BlkYLoopZeroOp_v(BlkYLoopZeroOp_v), BlockDimYOp_v(BlockDimYOp_v), BlkXLoopZeroOp_v(BlkXLoopZeroOp_v), BlockDimXOp_v(BlockDimXOp_v), dtc_spmm_kernel_blk_x_loop_v(dtc_spmm_kernel_blk_x_loop_v) {}

    // Body
    CUTE_DEVICE void body() {
        BlkXLoopZeroOp_v.template f<>();
        BlockDimXOp_v.template f<>();
        dtc_spmm_kernel_blk_x_loop_v.template f<>();
    }

    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        this->_idx = blockIdx.y;
        body();
    }
};

// GLOBAL FUNCTION
template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_coo_val_size, class tile_sidx_size>
__global__ void dtc_spmm_kernel_sstage_rstage(
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

    using B_valOp_t = B_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    B_valOp_t B_valOp_v{tid, lid, wid, K, N, dB_val};
    using C_valOp_t = C_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    C_valOp_t C_valOp_v{tid, lid, wid, N, M, dC_val};
    using val_coo_idxOp_t = val_coo_idxOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    val_coo_idxOp_t val_coo_idxOp_v{tid, lid, wid, nnz, dval_coo_idx};
    using val_coo_offOp_t = val_coo_offOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    val_coo_offOp_t val_coo_offOp_v{tid, lid, wid, nnz_block, dval_coo_off};
    using val_coo_valOp_t = val_coo_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    val_coo_valOp_t val_coo_valOp_v{tid, lid, wid, nnz, dval_coo_val};
    using val_sidxOp_t = val_sidxOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    val_sidxOp_t val_sidxOp_v{tid, lid, wid, nnz_block, dval_sidx};
    using val_soffOp_t = val_soffOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    val_soffOp_t val_soffOp_v{tid, lid, wid, Mo, dval_soff};
    using BlkYLoopZeroOp_t = BlkYLoopZeroOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using BlkYLoopZeroOp_layout_o0 = typename BlkYLoopZeroOp_t::Layout_o0;
    int BlkYLoopZeroOp_tensor_o0[cosize_v<BlkYLoopZeroOp_layout_o0>];
    BlkYLoopZeroOp_t BlkYLoopZeroOp_v{tid, lid, wid, BlkYLoopZeroOp_tensor_o0};
    using BlockDimYOp_t = BlockDimYOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using BlockDimYOp_layout_o0 = typename BlockDimYOp_t::Layout_o0;
    int BlockDimYOp_tensor_o0[cosize_v<BlockDimYOp_layout_o0>];
    BlockDimYOp_t BlockDimYOp_v{tid, lid, wid, BlockDimYOp_tensor_o0};
    using BlkXLoopZeroOp_t = BlkXLoopZeroOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using BlkXLoopZeroOp_layout_o0 = typename BlkXLoopZeroOp_t::Layout_o0;
    int BlkXLoopZeroOp_tensor_o0[cosize_v<BlkXLoopZeroOp_layout_o0>];
    BlkXLoopZeroOp_t BlkXLoopZeroOp_v{tid, lid, wid, BlkXLoopZeroOp_tensor_o0};
    using BlockDimXOp_t = BlockDimXOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using BlockDimXOp_layout_o0 = typename BlockDimXOp_t::Layout_o0;
    int BlockDimXOp_tensor_o0[cosize_v<BlockDimXOp_layout_o0>];
    BlockDimXOp_t BlockDimXOp_v{tid, lid, wid, BlockDimXOp_tensor_o0};
    using G2rSparseOffsetLoadOp_t = G2rSparseOffsetLoadOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using G2rSparseOffsetLoadOp_layout_o0 = typename G2rSparseOffsetLoadOp_t::Layout_o0;
    using G2rSparseOffsetLoadOp_layout_o1 = typename G2rSparseOffsetLoadOp_t::Layout_o1;
    int G2rSparseOffsetLoadOp_tensor_o0[cosize_v<G2rSparseOffsetLoadOp_layout_o0>];
    int G2rSparseOffsetLoadOp_tensor_o1[cosize_v<G2rSparseOffsetLoadOp_layout_o1>];
    G2rSparseOffsetLoadOp_t G2rSparseOffsetLoadOp_v{
        tid, lid, wid, 
        G2rSparseOffsetLoadOp_tensor_o0, G2rSparseOffsetLoadOp_tensor_o1, 
        Mo,
    };
    using ZerosOp_t = ZerosOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using ZerosOp_layout_o0 = typename ZerosOp_t::Layout_o0;
    float ZerosOp_tensor_o0[cosize_v<ZerosOp_layout_o0>];
    ZerosOp_t ZerosOp_v{tid, lid, wid, ZerosOp_tensor_o0};
    using G2sValSidxLoadOp_t = G2sValSidxLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using G2sValSidxLoadOp_layout_o0 = typename G2sValSidxLoadOp_t::Layout_o0;
    __shared__ int G2sValSidxLoadOp_tensor_o0[cosize_v<G2sValSidxLoadOp_layout_o0>];
    G2sValSidxLoadOp_t G2sValSidxLoadOp_v{
        tid, lid, wid, 
        G2sValSidxLoadOp_tensor_o0, 
        nnz_block,
    };
    using G2rCooAtomicFormatLoadOffOp_t = G2rCooAtomicFormatLoadOffOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using G2rCooAtomicFormatLoadOffOp_layout_o0 = typename G2rCooAtomicFormatLoadOffOp_t::Layout_o0;
    using G2rCooAtomicFormatLoadOffOp_layout_o1 = typename G2rCooAtomicFormatLoadOffOp_t::Layout_o1;
    int G2rCooAtomicFormatLoadOffOp_tensor_o0[cosize_v<G2rCooAtomicFormatLoadOffOp_layout_o0>];
    int G2rCooAtomicFormatLoadOffOp_tensor_o1[cosize_v<G2rCooAtomicFormatLoadOffOp_layout_o1>];
    G2rCooAtomicFormatLoadOffOp_t G2rCooAtomicFormatLoadOffOp_v{
        tid, lid, wid, 
        G2rCooAtomicFormatLoadOffOp_tensor_o0, G2rCooAtomicFormatLoadOffOp_tensor_o1, 
        nnz_block,
    };
    using S2rValSidxLoadOp_t = S2rValSidxLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using S2rValSidxLoadOp_layout_o0 = typename S2rValSidxLoadOp_t::Layout_o0;
    int S2rValSidxLoadOp_tensor_o0[cosize_v<S2rValSidxLoadOp_layout_o0>];
    S2rValSidxLoadOp_t S2rValSidxLoadOp_v{
        tid, lid, wid, 
        S2rValSidxLoadOp_tensor_o0,
    };
    using G2sCooAtomicFormatLoadValOp_t = G2sCooAtomicFormatLoadValOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using G2sCooAtomicFormatLoadValOp_layout_o0 = typename G2sCooAtomicFormatLoadValOp_t::Layout_o0;
    __shared__ half G2sCooAtomicFormatLoadValOp_tensor_o0[cosize_v<G2sCooAtomicFormatLoadValOp_layout_o0>];
    G2sCooAtomicFormatLoadValOp_t G2sCooAtomicFormatLoadValOp_v{
        tid, lid, wid, 
        G2sCooAtomicFormatLoadValOp_tensor_o0, 
        nnz,
    };
    using G2rCooAtomicFormatLoadIdxOp_t = G2rCooAtomicFormatLoadIdxOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using G2rCooAtomicFormatLoadIdxOp_layout_o0 = typename G2rCooAtomicFormatLoadIdxOp_t::Layout_o0;
    using G2rCooAtomicFormatLoadIdxOp_layout_o1 = typename G2rCooAtomicFormatLoadIdxOp_t::Layout_o1;
    int G2rCooAtomicFormatLoadIdxOp_tensor_o0[cosize_v<G2rCooAtomicFormatLoadIdxOp_layout_o0>];
    int G2rCooAtomicFormatLoadIdxOp_tensor_o1[cosize_v<G2rCooAtomicFormatLoadIdxOp_layout_o1>];
    G2rCooAtomicFormatLoadIdxOp_t G2rCooAtomicFormatLoadIdxOp_v{
        tid, lid, wid, 
        G2rCooAtomicFormatLoadIdxOp_tensor_o0, G2rCooAtomicFormatLoadIdxOp_tensor_o1, 
        nnz,
    };
    using G2rBValLoadOp_t = G2rBValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using G2rBValLoadOp_layout_o0 = typename G2rBValLoadOp_t::Layout_o0;
    half G2rBValLoadOp_tensor_o0[cosize_v<G2rBValLoadOp_layout_o0>];
    G2rBValLoadOp_t G2rBValLoadOp_v{
        tid, lid, wid, 
        G2rBValLoadOp_tensor_o0, 
        K,
    };
    using S2sCooAtomicValRestoreOp_t = S2sCooAtomicValRestoreOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using S2sCooAtomicValRestoreOp_layout_o0 = typename S2sCooAtomicValRestoreOp_t::Layout_o0;
    __shared__ half S2sCooAtomicValRestoreOp_tensor_o0[cosize_v<S2sCooAtomicValRestoreOp_layout_o0>];
    S2sCooAtomicValRestoreOp_t S2sCooAtomicValRestoreOp_v{
        tid, lid, wid, 
        S2sCooAtomicValRestoreOp_tensor_o0,
    };
    using S2rAValLoadOp_t = S2rAValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    using S2rAValLoadOp_layout_o0 = typename S2rAValLoadOp_t::Layout_o0;
    half S2rAValLoadOp_tensor_o0[cosize_v<S2rAValLoadOp_layout_o0>];
    S2rAValLoadOp_t S2rAValLoadOp_v{
        tid, lid, wid, 
        S2rAValLoadOp_tensor_o0,
    };
    using MmaOp_t = MmaOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, std::decay_t<decltype(ZerosOp_v.template output<0>())>>;
    MmaOp_t MmaOp_v{
        tid, lid, wid, 
        ZerosOp_v.template output<0>(),
    };
    using DtcMainLoopOp_t = DtcMainLoopOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, B_valOp_t, C_valOp_t, val_coo_idxOp_t, val_coo_offOp_t, val_coo_valOp_t, val_sidxOp_t, val_soffOp_t, BlkYLoopZeroOp_t, BlockDimYOp_t, BlkXLoopZeroOp_t, BlockDimXOp_t, G2rSparseOffsetLoadOp_t, ZerosOp_t, G2sValSidxLoadOp_t, G2rCooAtomicFormatLoadOffOp_t, S2rValSidxLoadOp_t, G2sCooAtomicFormatLoadValOp_t, G2rCooAtomicFormatLoadIdxOp_t, G2rBValLoadOp_t, S2sCooAtomicValRestoreOp_t, S2rAValLoadOp_t, MmaOp_t, std::decay_t<decltype(MmaOp_v.template output<0>())>>;
    DtcMainLoopOp_t DtcMainLoopOp_v{tid, lid, wid, B_valOp_v, C_valOp_v, val_coo_idxOp_v, val_coo_offOp_v, val_coo_valOp_v, val_sidxOp_v, val_soffOp_v, BlkYLoopZeroOp_v, BlockDimYOp_v, BlkXLoopZeroOp_v, BlockDimXOp_v, G2rSparseOffsetLoadOp_v, ZerosOp_v, G2sValSidxLoadOp_v, G2rCooAtomicFormatLoadOffOp_v, S2rValSidxLoadOp_v, G2sCooAtomicFormatLoadValOp_v, G2rCooAtomicFormatLoadIdxOp_v, G2rBValLoadOp_v, S2sCooAtomicValRestoreOp_v, S2rAValLoadOp_v, MmaOp_v, MmaOp_v.template output<0>()};
    using R2gCValStoreOp_t = R2gCValStoreOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, std::decay_t<decltype(C_valOp_v.template output<0>())>>;
    R2gCValStoreOp_t R2gCValStoreOp_v{
        tid, lid, wid, 
        C_valOp_v.template output<0>(),
    };
    using dtc_spmm_kernel_blk_x_loop_t = dtc_spmm_kernel_blk_x_loop<BLK_MNK, MMA_MNK, BLK_MMA_MNK, B_valOp_t, C_valOp_t, val_coo_idxOp_t, val_coo_offOp_t, val_coo_valOp_t, val_sidxOp_t, val_soffOp_t, BlkYLoopZeroOp_t, BlockDimYOp_t, BlkXLoopZeroOp_t, BlockDimXOp_t, G2rSparseOffsetLoadOp_t, ZerosOp_t, DtcMainLoopOp_t, R2gCValStoreOp_t>;
    dtc_spmm_kernel_blk_x_loop_t dtc_spmm_kernel_blk_x_loop_v{tid, lid, wid, B_valOp_v, C_valOp_v, val_coo_idxOp_v, val_coo_offOp_v, val_coo_valOp_v, val_sidxOp_v, val_soffOp_v, BlkYLoopZeroOp_v, BlockDimYOp_v, BlkXLoopZeroOp_v, BlockDimXOp_v, G2rSparseOffsetLoadOp_v, ZerosOp_v, DtcMainLoopOp_v, R2gCValStoreOp_v};
    using dtc_spmm_kernel_blk_y_loop_t = dtc_spmm_kernel_blk_y_loop<BLK_MNK, MMA_MNK, BLK_MMA_MNK, B_valOp_t, C_valOp_t, val_coo_idxOp_t, val_coo_offOp_t, val_coo_valOp_t, val_sidxOp_t, val_soffOp_t, BlkYLoopZeroOp_t, BlockDimYOp_t, BlkXLoopZeroOp_t, BlockDimXOp_t, dtc_spmm_kernel_blk_x_loop_t>;
    dtc_spmm_kernel_blk_y_loop_t dtc_spmm_kernel_blk_y_loop_v{tid, lid, wid, B_valOp_v, C_valOp_v, val_coo_idxOp_v, val_coo_offOp_v, val_coo_valOp_v, val_sidxOp_v, val_soffOp_v, BlkYLoopZeroOp_v, BlockDimYOp_v, BlkXLoopZeroOp_v, BlockDimXOp_v, dtc_spmm_kernel_blk_x_loop_v};

    BlkYLoopZeroOp_v.template f<>();
    BlockDimYOp_v.template f<>();
    dtc_spmm_kernel_blk_y_loop_v.template f<>();
};
