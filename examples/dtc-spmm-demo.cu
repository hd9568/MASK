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
class BValOp {
public:
    // Outputs
    // using Shape_o0 = decltype(make_shape(VARLEN, get<1>(BLK_MNK{})));
    using Tensor_o0 = decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN, VARLEN)), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, VARLEN)));
    using Layout_o0 = decltype(Tensor_o0{}.layout());
    using Shape_o0 = decltype(Layout_o0{}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 B_val;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE BValOp(int tid, int lid, int wid, int K, int N, TO0 *dB_val)
    : tid(tid), lid(lid), wid(wid), 
    shape_o0(logical_divide(make_tensor(make_gmem_ptr(dB_val), make_shape(K, N)), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y)).shape()),
    layout_o0(logical_divide(make_tensor(make_gmem_ptr(dB_val), make_shape(K, N)), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y)).layout()),
    B_val(logical_divide(make_tensor(make_gmem_ptr(dB_val), make_shape(K, N)), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y))) {}

    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return B_val;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = float>
class CValOp {
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

    CUTE_DEVICE CValOp(int tid, int lid, int wid, int N, int M, TO0 *dC_val)
    : tid(tid), lid(lid), wid(wid), 
    shape_o0(logical_divide(make_tensor(make_gmem_ptr(dC_val), make_shape(N, M)), select<1, 0>(BLK_MNK{}))(make_coord(_, blockIdx.y), make_coord(_, blockIdx.x)).shape()),
    layout_o0(logical_divide(make_tensor(make_gmem_ptr(dC_val), make_shape(N, M)), select<1, 0>(BLK_MNK{}))(make_coord(_, blockIdx.y), make_coord(_, blockIdx.x)).layout()),
    C_val(logical_divide(make_tensor(make_gmem_ptr(dC_val), make_shape(N, M)), select<1, 0>(BLK_MNK{}))(make_coord(_, blockIdx.y), make_coord(_, blockIdx.x))) {}
    
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return C_val;
        }
    }
};  

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = int>
class ValCooIdxOp {
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

    CUTE_DEVICE ValCooIdxOp(int tid, int lid, int wid, int nnz, TO0 *dval_coo_idx)
    : tid(tid), lid(lid), wid(wid), 
    shape_o0(make_tensor(make_gmem_ptr(dval_coo_idx), make_shape(nnz)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_coo_idx), make_shape(nnz)).layout()),
    val_coo_idx(make_tensor(make_gmem_ptr(dval_coo_idx), make_shape(nnz))) {}

    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_coo_idx;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = int>
class ValCooOffOp {
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

    CUTE_DEVICE ValCooOffOp(int tid, int lid, int wid, int nnz_block, TO0 *dval_coo_off)
    : tid(tid), lid(lid), wid(wid), 
    shape_o0(make_tensor(make_gmem_ptr(dval_coo_off), make_shape(Int<2>{}, nnz_block)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_coo_off), make_shape(Int<2>{}, nnz_block)).layout()),
    val_coo_off(make_tensor(make_gmem_ptr(dval_coo_off), make_shape(Int<2>{}, nnz_block))) {}
    
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_coo_off;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = half>
class ValCooValOp {
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

    CUTE_DEVICE ValCooValOp(int tid, int lid, int wid, int nnz, TO0 *dval_coo_val)
    : tid(tid), lid(lid), wid(wid), 
    shape_o0(make_tensor(make_gmem_ptr(dval_coo_val), make_shape(nnz)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_coo_val), make_shape(nnz)).layout()),
    val_coo_val(make_tensor(make_gmem_ptr(dval_coo_val), make_shape(nnz))) {}

    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_coo_val;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_sidx_size = Int<16>, class TO0 = int>
class ValSidxOp {
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

    CUTE_DEVICE ValSidxOp(int tid, int lid, int wid, int nnz_block, TO0 *dval_sidx)
    : tid(tid), lid(lid), wid(wid), 
    shape_o0(make_tensor(make_gmem_ptr(dval_sidx), make_shape(tile_sidx_size{}, nnz_block)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_sidx), make_shape(tile_sidx_size{}, nnz_block)).layout()),
    val_sidx(make_tensor(make_gmem_ptr(dval_sidx), make_shape(tile_sidx_size{}, nnz_block))) {}

    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_sidx;
        }
    }
};

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = int>
class ValSoffOp {
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

    CUTE_DEVICE ValSoffOp(int tid, int lid, int wid, int Mo, TO0 *dval_soff)
    : tid(tid), lid(lid), wid(wid), 
    shape_o0(make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo)).shape()),
    layout_o0(make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo)).layout()),
    val_soff(make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo))) {}

    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return val_soff;
        }
    }
};

// prologue
template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TI0 = int, class TI1 = int, class TO0 = int, class TO1 = int>
class G2rSparseOffsetLoadOp {
public:
    // Inputs
    using Shape_i0 = decltype(make_shape(VARLEN));
    Shape_i0 shape_i0;
    // Tensor_i0 val_soff;
    // TI1 i;

    // Outputs
    using Shape_o0 = decltype(make_shape(NBUF{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 l;
    using Shape_o1 = decltype(make_shape(NBUF{}));
    using Layout_o1 = decltype(make_layout(Shape_o1{}));
    using Tensor_o1 = decltype(make_tensor(make_rmem_ptr((TO1*)nullptr), Layout_o1{}));
    Shape_o1 shape_o1;
    Layout_o1 layout_o1;
    Tensor_o1 r;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rSparseOffsetLoadOp(int tid, int lid, int wid, TO0 *l, TO1 *r, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)), l(make_tensor(make_rmem_ptr(l), Layout_o0{})), r(make_tensor(make_rmem_ptr(r), Layout_o1{})) {}

    template<size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return l(buf_idx);
        }
        if constexpr (output_idx == 1) {
            return r(buf_idx);
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
    CUTE_DEVICE void f(Tensor_i0 val_soff, TI1 i) {
        l(buf_idx) = val_soff(i);
        r(buf_idx) = val_soff(i + 1);
    }
};

template<class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class TO0 = float>
class ZeroOp {
public:
    // Outputs
    using Shape_o0 = decltype(make_shape(Int<4>{}, make_shape(get<0>(BLK_MMA_MNK{}), get<1>(BLK_MMA_MNK{}))));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 REGC;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE ZeroOp(int tid, int lid, int wid, TO0 *REGC)
    : tid(tid), lid(lid), wid(wid), REGC(make_tensor(make_rmem_ptr(REGC), Layout_o0{})) {}

    template<size_t buf_idx, size_t output_idx>
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
    
    template<size_t buf_idx>
    CUTE_DEVICE void f() {
        clear(REGC);
    }

    template<typename = void>
    CUTE_DEVICE void f() {
        clear(REGC);
    }
    
};

// stage 0
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

    CUTE_DEVICE G2sValSidxLoadOp(int tid, int lid, int wid, int *tile_sidx_block, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(tile_sidx_size{}, mode_i0_0)), tile_sidx_block(make_tensor(make_smem_ptr(tile_sidx_block), Layout_o0{})) {}

    template<size_t buf_idx, size_t output_idx>
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
    using Shape_o0 = decltype(make_shape(NBUF{}));
    using Layout_o0 = decltype(make_layout(Shape_o0{}));
    using Tensor_o0 = decltype(make_tensor(make_rmem_ptr((TO0*)nullptr), Layout_o0{}));
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 l;
    using Shape_o1 = decltype(make_shape(NBUF{}));
    using Layout_o1 = decltype(make_layout(Shape_o1{}));
    using Tensor_o1 = decltype(make_tensor(make_rmem_ptr((TO1*)nullptr), Layout_o1{}));
    Shape_o1 shape_o1;
    Layout_o1 layout_o1;
    Tensor_o1 r;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rCooAtomicFormatLoadOffOp(int tid, int lid, int wid, TO0 *l, TO1 *r, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(Int<2>{}, mode_i0_0)), l(make_tensor(make_rmem_ptr(l), Layout_o0{})), r(make_tensor(make_rmem_ptr(r), Layout_o1{})) {}

    template<size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return l(buf_idx);
        }
        if constexpr (output_idx == 1) {
            return r(buf_idx);
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

    template<size_t buf_idx, size_t output_idx>
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

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_coo_val_size = Int<16>, class TI0 = half, class TI1 = int, class TI2 = int, class TO0 = half>
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

    CUTE_DEVICE G2sCooAtomicFormatLoadValOp(int tid, int lid, int wid, half *tile_coo_val_block, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)), tile_coo_val_block(make_tensor(make_smem_ptr(tile_coo_val_block), Layout_o0{})) {}

    template<size_t buf_idx, size_t output_idx>
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

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_coo_val_size = Int<16>, class TI0 = int, class TI1 = int, class TI2 = int, class TO0 = int, class TO1 = int>
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
    using Shape_o1 = decltype(make_shape(NBUF{}));
    using Layout_o1 = decltype(make_layout(Shape_o1{}));
    using Tensor_o1 = decltype(make_tensor(make_rmem_ptr((TO1*)nullptr), Layout_o1{}));
    Shape_o1 shape_o1;
    Layout_o1 layout_o1;
    Tensor_o1 coo_range_reg;

    // Hardware parameters
    int tid, lid, wid;

    CUTE_DEVICE G2rCooAtomicFormatLoadIdxOp(int tid, int lid, int wid, TO0 *coo_idx_reg, TO1 *coo_range_reg, int mode_i0_0)
    : tid(tid), lid(lid), wid(wid), shape_i0(make_shape(mode_i0_0)), coo_idx_reg(make_tensor(make_rmem_ptr(coo_idx_reg), Layout_o0{})), coo_range_reg(make_tensor(make_rmem_ptr(coo_range_reg), Layout_o1{})) {}

    template<size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return coo_idx_reg(_, buf_idx);
        }
        if constexpr (output_idx == 1) {
            return coo_range_reg(buf_idx);
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

// stage 2
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

    template<size_t buf_idx, size_t output_idx>
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

template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class tile_coo_val_size = Int<16>, class TI0 = half, class TI1 = int, class TI2 = int, class TO0 = half>
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

    CUTE_DEVICE S2sCooAtomicValRestoreOp(int tid, int lid, int wid, half *tile_coo_restore_val_block)
    : tid(tid), lid(lid), wid(wid), tile_coo_restore_val_block(make_tensor(make_smem_ptr(tile_coo_restore_val_block), Layout_o0{})) {}

    template<size_t buf_idx, size_t output_idx>
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

    template<size_t buf_idx, size_t output_idx>
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

// stage 3
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

    template<size_t buf_idx, size_t output_idx>
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

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class T_G2sValSidxLoadOp, class T_G2rCooAtomicFormatLoadOffOp, class T_S2rValSidxLoadOp, class T_G2sCooAtomicFormatLoadValOp, class T_G2rCooAtomicFormatLoadIdxOp, class T_G2rBValLoadOp, class T_S2sCooAtomicValRestoreOp, class T_S2rAValLoadOp, class T_MmaOp, class T_BValOp, class T_CValOp, class T_ValCooIdxOp, class T_ValCooOffOp, class T_ValCooValOp, class T_ValSidxOp, class T_ValSoffOp, class Tensor_o0, class TO0 = float>
class PipelinedForLoopOp {
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
    T_G2sValSidxLoadOp &g2s_val_sidx_load_op;
    T_G2rCooAtomicFormatLoadOffOp &g2r_coo_atomic_format_load_off_op;
    T_S2rValSidxLoadOp &s2r_val_sidx_load_op;
    T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op;
    T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op;
    T_G2rBValLoadOp &g2r_b_val_load_op;
    T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op;
    T_S2rAValLoadOp &s2r_a_val_load_op;
    T_MmaOp &mma_op;
    T_BValOp &B_val_op;
    T_CValOp &C_val_op;
    T_ValCooIdxOp &val_coo_idx_op;
    T_ValCooOffOp &val_coo_off_op;
    T_ValCooValOp &val_coo_val_op;
    T_ValSidxOp &val_sidx_op;
    T_ValSoffOp &val_soff_op;

    // Constructor
    CUTE_DEVICE PipelinedForLoopOp(int tid, int lid, int wid, T_G2sValSidxLoadOp &g2s_val_sidx_load_op, T_G2rCooAtomicFormatLoadOffOp &g2r_coo_atomic_format_load_off_op, T_S2rValSidxLoadOp &s2r_val_sidx_load_op, T_G2sCooAtomicFormatLoadValOp &g2s_coo_atomic_format_load_val_op, T_G2rCooAtomicFormatLoadIdxOp &g2r_coo_atomic_format_load_idx_op, T_G2rBValLoadOp &g2r_b_val_load_op, T_S2sCooAtomicValRestoreOp &s2s_coo_atomic_val_restore_op, T_S2rAValLoadOp &s2r_a_val_load_op, T_MmaOp &mma_op, T_BValOp &B_val_op, T_CValOp &C_val_op, T_ValCooIdxOp &val_coo_idx_op, T_ValCooOffOp &val_coo_off_op, T_ValCooValOp &val_coo_val_op, T_ValSidxOp &val_sidx_op, T_ValSoffOp &val_soff_op, Tensor_o0 REGC)
    : tid(tid), lid(lid), wid(wid), g2s_val_sidx_load_op(g2s_val_sidx_load_op), g2r_coo_atomic_format_load_off_op(g2r_coo_atomic_format_load_off_op), s2r_val_sidx_load_op(s2r_val_sidx_load_op), g2s_coo_atomic_format_load_val_op(g2s_coo_atomic_format_load_val_op), g2r_coo_atomic_format_load_idx_op(g2r_coo_atomic_format_load_idx_op), g2r_b_val_load_op(g2r_b_val_load_op), s2s_coo_atomic_val_restore_op(s2s_coo_atomic_val_restore_op), s2r_a_val_load_op(s2r_a_val_load_op), mma_op(mma_op), B_val_op(B_val_op), C_val_op(C_val_op), val_coo_idx_op(val_coo_idx_op), val_coo_off_op(val_coo_off_op), val_coo_val_op(val_coo_val_op), val_sidx_op(val_sidx_op), val_soff_op(val_soff_op), REGC(REGC) {}

    // short pipe
    __forceinline__ __device__ void short_pipe(int k, int l, int r) {
        if (k == 0) {
            // i = 0
            
            
            // i = 1
            
            
            // i = 2
            
        }
        else if (k == 1) {
            // i = 0
            g2s_val_sidx_load_op.template f<0>(
                val_sidx_op.template output<0>(),
                l + 0
            );
            __pipeline_commit();
            g2r_coo_atomic_format_load_off_op.template f<0>(
                val_coo_off_op.template output<0>(),
                l + 0
            );
            
            // i = 1
            g2s_coo_atomic_format_load_val_op.template f<0>(
                val_coo_val_op.template output<0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 1>()
            );
            __pipeline_commit();
            __pipeline_wait_prior(1);
            __syncthreads();
            s2r_val_sidx_load_op.template f<0>(
                g2s_val_sidx_load_op.template output<0, 0>()
            );g2r_coo_atomic_format_load_idx_op.template f<0>(
                val_coo_idx_op.template output<0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 1>()
            );
            
            // i = 2
            g2r_b_val_load_op.template f<0>(
                B_val_op.template output<0>(),
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
                val_sidx_op.template output<0>(),
                l + 0
            );
            __pipeline_commit();
            g2r_coo_atomic_format_load_off_op.template f<0>(
                val_coo_off_op.template output<0>(),
                l + 0
            );
            
            // i = 1
            g2s_val_sidx_load_op.template f<1>(
                val_sidx_op.template output<0>(),
                l + 1
            );
            g2s_coo_atomic_format_load_val_op.template f<0>(
                val_coo_val_op.template output<0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 1>()
            );
            __pipeline_commit();
            g2r_coo_atomic_format_load_off_op.template f<1>(
                val_coo_off_op.template output<0>(),
                l + 1
            );__pipeline_wait_prior(1);
            __syncthreads();
            s2r_val_sidx_load_op.template f<0>(
                g2s_val_sidx_load_op.template output<0, 0>()
            );g2r_coo_atomic_format_load_idx_op.template f<0>(
                val_coo_idx_op.template output<0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 1>()
            );
            
            // i = 2
            g2s_coo_atomic_format_load_val_op.template f<1>(
                val_coo_val_op.template output<0>(),
                g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
                g2r_coo_atomic_format_load_off_op.template output<1, 1>()
            );
            __pipeline_commit();
            __pipeline_wait_prior(1);
            __syncthreads();
            s2r_val_sidx_load_op.template f<1>(
                g2s_val_sidx_load_op.template output<1, 0>()
            );g2r_coo_atomic_format_load_idx_op.template f<1>(
                val_coo_idx_op.template output<0>(),
                g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
                g2r_coo_atomic_format_load_off_op.template output<1, 1>()
            );g2r_b_val_load_op.template f<0>(
                B_val_op.template output<0>(),
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
                B_val_op.template output<0>(),
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
                val_sidx_op.template output<0>(),
                l + 0
            );
            __pipeline_commit();
            g2r_coo_atomic_format_load_off_op.template f<0>(
                val_coo_off_op.template output<0>(),
                l + 0
            );
            
            // i = 1
            g2s_val_sidx_load_op.template f<1>(
                val_sidx_op.template output<0>(),
                l + 1
            );
            g2s_coo_atomic_format_load_val_op.template f<0>(
                val_coo_val_op.template output<0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 1>()
            );
            __pipeline_commit();
            g2r_coo_atomic_format_load_off_op.template f<1>(
                val_coo_off_op.template output<0>(),
                l + 1
            );__pipeline_wait_prior(1);
            __syncthreads();
            s2r_val_sidx_load_op.template f<0>(
                g2s_val_sidx_load_op.template output<0, 0>()
            );g2r_coo_atomic_format_load_idx_op.template f<0>(
                val_coo_idx_op.template output<0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 1>()
            );
            
            // i = 2
            g2s_val_sidx_load_op.template f<0>(
                val_sidx_op.template output<0>(),
                l + 2
            );
            g2s_coo_atomic_format_load_val_op.template f<1>(
                val_coo_val_op.template output<0>(),
                g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
                g2r_coo_atomic_format_load_off_op.template output<1, 1>()
            );
            __pipeline_commit();
            g2r_coo_atomic_format_load_off_op.template f<0>(
                val_coo_off_op.template output<0>(),
                l + 2
            );__pipeline_wait_prior(1);
            __syncthreads();
            s2r_val_sidx_load_op.template f<1>(
                g2s_val_sidx_load_op.template output<1, 0>()
            );g2r_coo_atomic_format_load_idx_op.template f<1>(
                val_coo_idx_op.template output<0>(),
                g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
                g2r_coo_atomic_format_load_off_op.template output<1, 1>()
            );g2r_b_val_load_op.template f<0>(
                B_val_op.template output<0>(),
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
                val_coo_val_op.template output<0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 1>()
            );
            __pipeline_commit();
            __pipeline_wait_prior(1);
            __syncthreads();
            s2r_val_sidx_load_op.template f<0>(
                g2s_val_sidx_load_op.template output<0, 0>()
            );g2r_coo_atomic_format_load_idx_op.template f<0>(
                val_coo_idx_op.template output<0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
                g2r_coo_atomic_format_load_off_op.template output<0, 1>()
            );g2r_b_val_load_op.template f<1>(
                B_val_op.template output<0>(),
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
                B_val_op.template output<0>(),
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
    __forceinline__ __device__ void fill(int l, int r) {
        // stage 0..0
        g2s_val_sidx_load_op.template f<0>(
            val_sidx_op.template output<0>(),
            l + 0
        );
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<0>(
            val_coo_off_op.template output<0>(),
            l + 0
        );
        // stage 0..1
        g2s_val_sidx_load_op.template f<1>(
            val_sidx_op.template output<0>(),
            l + 1
        );
        g2s_coo_atomic_format_load_val_op.template f<0>(
            val_coo_val_op.template output<0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<1>(
            val_coo_off_op.template output<0>(),
            l + 1
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<0>(
            g2s_val_sidx_load_op.template output<0, 0>()
        );
        g2r_coo_atomic_format_load_idx_op.template f<0>(
            val_coo_idx_op.template output<0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        // stage 0..2
        g2s_val_sidx_load_op.template f<0>(
            val_sidx_op.template output<0>(),
            l + 2
        );
        g2s_coo_atomic_format_load_val_op.template f<1>(
            val_coo_val_op.template output<0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 1>()
        );
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<0>(
            val_coo_off_op.template output<0>(),
            l + 2
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<1>(
            g2s_val_sidx_load_op.template output<1, 0>()
        );
        g2r_coo_atomic_format_load_idx_op.template f<1>(
            val_coo_idx_op.template output<0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 1>()
        );
        g2r_b_val_load_op.template f<0>(
            B_val_op.template output<0>(),
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
    __forceinline__ __device__ void loop_step(int i, int l, int r) {
        // step 0
        g2s_val_sidx_load_op.template f<1>(
            val_sidx_op.template output<0>(),
            l + i + (0)
        );
        g2s_coo_atomic_format_load_val_op.template f<0>(
            val_coo_val_op.template output<0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<1>(
            val_coo_off_op.template output<0>(),
            l + i + (0)
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<0>(
            g2s_val_sidx_load_op.template output<0, 0>()
        );
        g2r_coo_atomic_format_load_idx_op.template f<0>(
            val_coo_idx_op.template output<0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        g2r_b_val_load_op.template f<1>(
            B_val_op.template output<0>(),
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
            val_sidx_op.template output<0>(),
            l + i + (1)
        );
        g2s_coo_atomic_format_load_val_op.template f<1>(
            val_coo_val_op.template output<0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 1>()
        );
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<0>(
            val_coo_off_op.template output<0>(),
            l + i + (1)
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<1>(
            g2s_val_sidx_load_op.template output<1, 0>()
        );
        g2r_coo_atomic_format_load_idx_op.template f<1>(
            val_coo_idx_op.template output<0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 1>()
        );
        g2r_b_val_load_op.template f<0>(
            B_val_op.template output<0>(),
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
    __forceinline__ __device__ void remains_1(int i, int l, int r) {
        // total remains 1, current 0
        g2s_val_sidx_load_op.template f<1>(
            val_sidx_op.template output<0>(),
            l + i + (0)
        );
        g2s_coo_atomic_format_load_val_op.template f<0>(
            val_coo_val_op.template output<0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        __pipeline_commit();
        g2r_coo_atomic_format_load_off_op.template f<1>(
            val_coo_off_op.template output<0>(),
            l + i + (0)
        );
        __pipeline_wait_prior(1);
        __syncthreads();
        s2r_val_sidx_load_op.template f<0>(
            g2s_val_sidx_load_op.template output<0, 0>()
        );
        g2r_coo_atomic_format_load_idx_op.template f<0>(
            val_coo_idx_op.template output<0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        g2r_b_val_load_op.template f<1>(
            B_val_op.template output<0>(),
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

    __forceinline__ __device__ void remainder(int i, int l, int r) {
        if (r - l - i == 1) {
            remains_1(i, l, r);
        }
    }

    // empties
    __forceinline__ __device__ void empty_0(int i, int l, int r) {
        // stage 1..3
        g2s_coo_atomic_format_load_val_op.template f<1>(
            val_coo_val_op.template output<0>(),
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
            val_coo_idx_op.template output<0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<1, 1>()
        );
        g2r_b_val_load_op.template f<0>(
            B_val_op.template output<0>(),
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
            B_val_op.template output<0>(),
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

    __forceinline__ __device__ void empty_1(int i, int l, int r) {
        // stage 1..3
        g2s_coo_atomic_format_load_val_op.template f<0>(
            val_coo_val_op.template output<0>(),
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
            val_coo_idx_op.template output<0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 0>(),
            g2r_coo_atomic_format_load_off_op.template output<0, 1>()
        );
        g2r_b_val_load_op.template f<1>(
            B_val_op.template output<0>(),
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
            B_val_op.template output<0>(),
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

    __forceinline__ __device__ void empty(int i, int l, int r) {
        if ((r - l - i + (3)) % 2 == 0) {
            empty_0(i, l, r);
        }
        else if ((r - l - i + (3)) % 2 == 1) {
            empty_1(i, l, r);
        }
    }

    __forceinline__ __device__ void dispatch() {
        int l = this->_l;
        int r = this->_r;
        int k = this->_r - this->_l;
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

    template<size_t buf_idx, size_t output_idx>
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

    template<typename = void>
    CUTE_DEVICE void f(int l, int r) {
        this->_l = l;
        this->_r = r;
        dispatch();
    }
};

// epilogue
template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class Tensor_o0, class TI0 = float, class TO0 = float>
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

    template<size_t buf_idx, size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return C_val;
        }
    }
    
    template<size_t output_idx>
    CUTE_DEVICE auto output() {
        if constexpr (output_idx == 0) {
            return C_val;
        }
    }

    template<size_t buf_idx, class Tensor_i0>
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


template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class T_ValSoffOp, class T_ZeroOp, class T_G2rSparseOffsetLoadOp, class T_PipelinedForLoopOp, class T_R2gCValStoreOp>
class BlkXForLoopOp {
public:
    int _idx;

    // Hardware parameters
    int tid, lid, wid;

    // Body Ops
    T_ValSoffOp val_soff_op;
    T_ZeroOp zero_op;
    T_G2rSparseOffsetLoadOp g2r_sparse_offset_load_op;
    T_PipelinedForLoopOp pipelined_for_loop_op;
    T_R2gCValStoreOp r2g_c_val_store_op;

    // Constructor
    CUTE_DEVICE BlkXForLoopOp(int tid, int lid, int wid, T_ValSoffOp &val_soff_op, T_ZeroOp &zero_op, T_G2rSparseOffsetLoadOp &g2r_sparse_offset_load_op, T_PipelinedForLoopOp &pipelined_for_loop_op, T_R2gCValStoreOp &r2g_c_val_store_op) :
    tid(tid), lid(lid), wid(wid), val_soff_op(val_soff_op), zero_op(zero_op), g2r_sparse_offset_load_op(g2r_sparse_offset_load_op), pipelined_for_loop_op(pipelined_for_loop_op), r2g_c_val_store_op(r2g_c_val_store_op) {}

    // Body
    CUTE_DEVICE void body() {
        zero_op.template f<>();

        g2r_sparse_offset_load_op.template f<0>(
            val_soff_op.template output<0>(),
            this->_idx
        );

        
        pipelined_for_loop_op.template f<>(
            g2r_sparse_offset_load_op.template output<0, 0>(),
            g2r_sparse_offset_load_op.template output<0, 1>()
        );

        r2g_c_val_store_op.template f<0>(
            pipelined_for_loop_op.template output<0>()
        );
    }

    // f
    template<typename = void>
    CUTE_DEVICE void f() {
        this->_idx = blockIdx.x;
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

    // gmem tensors
    using B_val_op_t = BValOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    B_val_op_t B_val_op{ tid, lid, wid, K, N, dB_val, };
    using C_val_op_t = CValOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    C_val_op_t C_val_op{ tid, lid, wid, N, M, dC_val, };
    using val_coo_idx_op_t = ValCooIdxOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    val_coo_idx_op_t val_coo_idx_op{ tid, lid, wid, nnz, dval_coo_idx, };
    using val_coo_off_op_t = ValCooOffOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    val_coo_off_op_t val_coo_off_op{ tid, lid, wid, nnz_block, dval_coo_off, };
    using val_coo_val_op_t = ValCooValOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    val_coo_val_op_t val_coo_val_op{ tid, lid, wid, nnz, dval_coo_val, };
    using val_sidx_op_t = ValSidxOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    val_sidx_op_t val_sidx_op{ tid, lid, wid, nnz_block, dval_sidx, };
    using val_soff_op_t = ValSoffOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
    val_soff_op_t val_soff_op{ tid, lid, wid, Mo, dval_soff, };
        // prologue
        using g2r_sparse_offset_load_op_t = G2rSparseOffsetLoadOp<Int<1>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
        using g2r_sparse_offset_load_op_layout_o0 = typename g2r_sparse_offset_load_op_t::Layout_o0;
        using g2r_sparse_offset_load_op_layout_o1 = typename g2r_sparse_offset_load_op_t::Layout_o1;
        int g2r_sparse_offset_load_op_tensor_o0[cosize_v<g2r_sparse_offset_load_op_layout_o0>];
        int g2r_sparse_offset_load_op_tensor_o1[cosize_v<g2r_sparse_offset_load_op_layout_o1>];
        g2r_sparse_offset_load_op_t g2r_sparse_offset_load_op{
            tid, lid, wid, 
            g2r_sparse_offset_load_op_tensor_o0, g2r_sparse_offset_load_op_tensor_o1,
            Mo,
        };
        
        using zero_op_t = ZeroOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
        using zero_op_layout_o0 = typename zero_op_t::Layout_o0;
        float zero_op_tensor_o0[cosize_v<zero_op_layout_o0>];
        zero_op_t zero_op{ tid, lid, wid, zero_op_tensor_o0, };

            // stage 0
            using g2s_val_sidx_load_op_t = G2sValSidxLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_sidx_size>;
            using g2s_val_sidx_load_op_layout_o0 = typename g2s_val_sidx_load_op_t::Layout_o0;
            __shared__ int g2s_val_sidx_load_op_tensor_o0[cosize_v<g2s_val_sidx_load_op_layout_o0>];
            g2s_val_sidx_load_op_t g2s_val_sidx_load_op{
                tid, lid, wid, 
                g2s_val_sidx_load_op_tensor_o0,
                nnz_block,
            };
            
            using g2r_coo_atomic_format_load_off_op_t = G2rCooAtomicFormatLoadOffOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
            using g2r_coo_atomic_format_load_off_op_layout_o0 = typename g2r_coo_atomic_format_load_off_op_t::Layout_o0;
            using g2r_coo_atomic_format_load_off_op_layout_o1 = typename g2r_coo_atomic_format_load_off_op_t::Layout_o1;
            int g2r_coo_atomic_format_load_off_op_tensor_o0[cosize_v<g2r_coo_atomic_format_load_off_op_layout_o0>];
            int g2r_coo_atomic_format_load_off_op_tensor_o1[cosize_v<g2r_coo_atomic_format_load_off_op_layout_o1>];
            g2r_coo_atomic_format_load_off_op_t g2r_coo_atomic_format_load_off_op{
                tid, lid, wid, 
                g2r_coo_atomic_format_load_off_op_tensor_o0, g2r_coo_atomic_format_load_off_op_tensor_o1,
                nnz_block,
            };
            
            // stage 1
            using s2r_val_sidx_load_op_t = S2rValSidxLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_sidx_size>;
            using s2r_val_sidx_load_op_layout_o0 = typename s2r_val_sidx_load_op_t::Layout_o0;
            int s2r_val_sidx_load_op_tensor_o0[cosize_v<s2r_val_sidx_load_op_layout_o0>];
            s2r_val_sidx_load_op_t s2r_val_sidx_load_op{
                tid, lid, wid,
                s2r_val_sidx_load_op_tensor_o0,
            };
            
            using g2s_coo_atomic_format_load_val_op_t = G2sCooAtomicFormatLoadValOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_coo_val_size>;
            using g2s_coo_atomic_format_load_val_op_layout_o0 = typename g2s_coo_atomic_format_load_val_op_t::Layout_o0;
            __shared__ half g2s_coo_atomic_format_load_val_op_tensor_o0[cosize_v<g2s_coo_atomic_format_load_val_op_layout_o0>];
            g2s_coo_atomic_format_load_val_op_t g2s_coo_atomic_format_load_val_op{
                tid, lid, wid, 
                g2s_coo_atomic_format_load_val_op_tensor_o0,
                nnz,
            };
            
            using g2r_coo_atomic_format_load_idx_op_t = G2rCooAtomicFormatLoadIdxOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_coo_val_size>;
            using g2r_coo_atomic_format_load_idx_op_layout_o0 = typename g2r_coo_atomic_format_load_idx_op_t::Layout_o0;
            using g2r_coo_atomic_format_load_idx_op_layout_o1 = typename g2r_coo_atomic_format_load_idx_op_t::Layout_o1;
            int g2r_coo_atomic_format_load_idx_op_tensor_o0[cosize_v<g2r_coo_atomic_format_load_idx_op_layout_o0>];
            int g2r_coo_atomic_format_load_idx_op_tensor_o1[cosize_v<g2r_coo_atomic_format_load_idx_op_layout_o1>];
            g2r_coo_atomic_format_load_idx_op_t g2r_coo_atomic_format_load_idx_op{
                tid, lid, wid, 
                g2r_coo_atomic_format_load_idx_op_tensor_o0, g2r_coo_atomic_format_load_idx_op_tensor_o1,
                nnz,
            };
            
            // stage 2
            using g2r_b_val_load_op_t = G2rBValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
            using g2r_b_val_load_op_layout_o0 = typename g2r_b_val_load_op_t::Layout_o0;
            half g2r_b_val_load_op_tensor_o0[cosize_v<g2r_b_val_load_op_layout_o0>];
            g2r_b_val_load_op_t g2r_b_val_load_op{
                tid, lid, wid, 
                g2r_b_val_load_op_tensor_o0,
                K,
            };
            
            using s2s_coo_atomic_val_restore_op_t = S2sCooAtomicValRestoreOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, tile_coo_val_size>;
            using s2s_coo_atomic_val_restore_op_layout_o0 = typename s2s_coo_atomic_val_restore_op_t::Layout_o0;
            __shared__ half s2s_coo_atomic_val_restore_op_tensor_o0[cosize_v<s2s_coo_atomic_val_restore_op_layout_o0>];
            s2s_coo_atomic_val_restore_op_t s2s_coo_atomic_val_restore_op{
                tid, lid, wid, 
                s2s_coo_atomic_val_restore_op_tensor_o0,
            };
            
            using s2r_a_val_load_op_t = S2rAValLoadOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK>;
            using s2r_a_val_load_op_layout_o0 = typename s2r_a_val_load_op_t::Layout_o0;
            half s2r_a_val_load_op_tensor_o0[cosize_v<s2r_a_val_load_op_layout_o0>];
            s2r_a_val_load_op_t s2r_a_val_load_op{
                tid, lid, wid,
                s2r_a_val_load_op_tensor_o0,
            };
            
            // stage 3
            using mma_op_t = MmaOp<Int<2>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, std::decay_t<decltype(zero_op.template output<0>())>>;
            mma_op_t mma_op{
                tid, lid, wid,
                zero_op.template output<0>(),
            };

        // for_loop
        using pipelined_for_loop_op_t = PipelinedForLoopOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, g2s_val_sidx_load_op_t, g2r_coo_atomic_format_load_off_op_t, s2r_val_sidx_load_op_t, g2s_coo_atomic_format_load_val_op_t, g2r_coo_atomic_format_load_idx_op_t, g2r_b_val_load_op_t, s2s_coo_atomic_val_restore_op_t, s2r_a_val_load_op_t, mma_op_t, B_val_op_t, C_val_op_t, val_coo_idx_op_t, val_coo_off_op_t, val_coo_val_op_t, val_sidx_op_t, val_soff_op_t, std::decay_t<decltype(mma_op.template output<0>())> >;
        pipelined_for_loop_op_t pipelined_for_loop_op{ tid, lid, wid, g2s_val_sidx_load_op, g2r_coo_atomic_format_load_off_op, s2r_val_sidx_load_op, g2s_coo_atomic_format_load_val_op, g2r_coo_atomic_format_load_idx_op, g2r_b_val_load_op, s2s_coo_atomic_val_restore_op, s2r_a_val_load_op, mma_op, B_val_op, C_val_op, val_coo_idx_op, val_coo_off_op, val_coo_val_op, val_sidx_op, val_soff_op, mma_op.template output<0>()};
        
        // epilogue
        using r2g_c_val_store_op_t = R2gCValStoreOp<Int<1>, BLK_MNK, MMA_MNK, BLK_MMA_MNK, std::decay_t<decltype(C_val_op.template output<0>())>>;
        r2g_c_val_store_op_t r2g_c_val_store_op{
            tid, lid, wid, 
            C_val_op.template output<0>(),
        };

    using blk_x_for_loop_op_t = BlkXForLoopOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, val_soff_op_t, zero_op_t, g2r_sparse_offset_load_op_t, pipelined_for_loop_op_t, r2g_c_val_store_op_t>;
    blk_x_for_loop_op_t blk_x_for_loop_op{ tid, lid, wid, val_soff_op, zero_op, g2r_sparse_offset_load_op, pipelined_for_loop_op, r2g_c_val_store_op };

    blk_x_for_loop_op.template f<>();
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


    // constexpr int ROW_WINDOW_M = 16;
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

    // constexpr int shift1 = 1;
    // constexpr int shift2 = 1;
    // constexpr int shift3 = 1;
    // constexpr int maxshift1_2 = (shift1 > shift2 ? shift1 : shift2);
    // constexpr int nbuf_rmem = (maxshift1_2 > shift3 ? maxshift1_2 : shift3) + 1;
    // constexpr int nbuf_smem = nbuf_rmem;
    // constexpr int nbuf_smem = maxshift1_2 + 1;

    //! （shift1=1,shift2=1,shift3=1)
    dtc_spmm_kernel_sstage_rstage<
        BLK_MNK, MMA_MNK, BLK_MMA_MNK,
        tile_coo_val_size, tile_sidx_size
    ><<<grid, block>>>(dB_val, dC, dval_coo_idx, dval_coo_off, dval_coo_val, dval_sidx, dval_soff, K, M, Mo, N, nnz, nnz_block);
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
    // constexpr int TC_BLOCK_M = 16;

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
