DTC_HELPER_FUNCTIONS = r"""__device__ __forceinline__ void ldmatrix_m8n8k8_x4(uint32_t* fragment, void* ptr) {
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
"""

SHORT_PIPE_SKELETON = """CUTE_DEVICE void short_pipe(int k, int l, int r) {{
{short_pipe_body}
}}"""

FILL_SKELETON = """CUTE_DEVICE void fill(int l, int r) {{
{fill_body}
}}"""

EMPTY_AFTER_REMAIN_R_SKELETON = """CUTE_DEVICE void empty_after_remain_{r}(int i, int l, int r) {{
{empty_after_remain_r_body}
}}"""

EMPTY_SKELETON = """CUTE_DEVICE void empty(int i, int l, int r) {{
{empty_dispatch_body}
}}"""

LOOP_STEP_SKELETON = """CUTE_DEVICE void loop_step(int i, int l, int r) {{
{loop_step_body}
}}"""

REMAINS_R_SKELETON = """CUTE_DEVICE void remains_{r}(int i, int l, int r) {{
{remains_r_body}
}}"""

REMAINDER_SKELETON = """CUTE_DEVICE void remainder(int i, int l, int r) {{
{remainder_dispatch_body}
}}"""

PIPELINE_DISPATCH_SKELETON = """CUTE_DEVICE void dispatch() {{
    int l = this->_l;
    int r = this->_r;
    int k = r - l;
    if (k <= {fill_len}) {{
        {short_pipe_dispatch}
    }} else {{
        {fill_dispatch}
        int i;
        for (i = {fill_len}; i + {nbuf} <= k; i += {nbuf}) {{
            {loop_step_dispatch}
        }}
        {remainder_dispatch}
        {empty_dispatch}
    }}
}}"""

PIPELINED_FOR_LOOP_NVOP_F_SKELETON = """this->_l = l;
this->_r = r;
{pipeline_dispatch}
"""

PIPELINED_FOR_LOOP_NVOP_DEF_SKELETON = """template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK{visible_op_type_template_params}{non_owning_tensor_type_template_params}{io_dtype_template_params}{parameter_template_params}>
class {op_name} {{
public:
    // Inputs
    int _l, _r;
    int _idx; // i + _l
    int _iter; // i
{extra_inputs}

{outputs}

    // Hardware parameters
    int tid, lid, wid;

    // Body Ops
{visible_ops}

    // Constructor
    CUTE_DEVICE {op_name}(int tid, int lid, int wid{visible_op_as_params}{non_owning_tensor_as_params})
    : tid(tid), lid(lid), wid(wid){visible_op_inits}{non_owning_tensor_inits} {{}}

    // Pipeline methods
{pipeline_methods}

{get_output_method}

    // f
    template<typename = void>
    CUTE_DEVICE void f(int l, int r{extra_f_params}) {{
        _l = l;
        _r = r;
{extra_input_assignments}        
        dispatch();
    }}
}};"""

BLK_IDX_FOR_LOOP_NVOP_DEF_SKELETON = """template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK{visible_op_type_template_params}{parameter_template_params}>
class {op_name} {{
public:
    // Inputs
    int _idx;

    // Hardware parameters
    int tid, lid, wid;

    // Body Ops
{visible_ops}

    // Constructor
    CUTE_DEVICE {op_name}(int tid, int lid, int wid{visible_op_as_params})
    : tid(tid), lid(lid), wid(wid){visible_op_inits} {{}}

    // Body
    CUTE_DEVICE void body() {{
{body}
    }}

    // f
    template<typename = void>
    CUTE_DEVICE void f() {{
        this->_idx = blockIdx.{block_idx_dim};
        body();
    }}
}};"""

SEQUENTIAL_FOR_LOOP_NVOP_DEF_SKELETON = """template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK{visible_op_type_template_params}{non_owning_tensor_type_template_params}{io_dtype_template_params}{parameter_template_params}>
class {op_name} {{
public:
    // Inputs
    int _l, _r;
    int _idx; // i + _l
    int _iter; // i
{extra_inputs}

{loop_result_outputs}

    // Hardware parameters
    int tid, lid, wid;

    // Body Ops
{visible_ops}

    // Constructor
    CUTE_DEVICE {op_name}(int tid, int lid, int wid{visible_op_as_params}{non_owning_tensor_as_params})
    : tid(tid), lid(lid), wid(wid){visible_op_inits}{non_owning_tensor_inits} {{}}

{loop_result_outputs_method}

    // f
    template<class Tensor_i0, class Tensor_i1>
    CUTE_DEVICE void f(Tensor_i0 l, Tensor_i1 r{extra_f_params}) {{
        _l = l;
        _r = r;
{extra_input_assignments}
        for (int i = _l; i < _r; i++) {{
            _idx = i;
            _iter = i - _l;
{loop_body}
        }}
    }}
}};"""

CONSTANT_NVOP_CLASS_DEF_SKELETON = """template <{nbuf_type}class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK{parameter_types}{io_dtypes}>
class {op_name} {{
public:
    // No Inputs

{outputs}
    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE {op_name}(int tid, int lid, int wid, TO0 *{tensor_name})
    : tid(tid), lid(lid), wid(wid), {tensor_name}(make_tensor(make_{mem_type_str}_ptr({tensor_name}), Layout_o0{{}})) {{}}

    // output
{get_output_method}
    
    // f
    template<typename = void>
    CUTE_DEVICE void f() {{
        {f_body}
    }}
}};"""

GMEM_TENSOR_NVOP_CLASS_DEF_SKELETON = """template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK{parameter_types}{io_dtypes}>
class {op_name} {{
public:
    // Outputs
    using Tensor_o0 = {tensor_type_str};
    using Layout_o0 = decltype(Tensor_o0{{}}.layout());
    using Shape_o0 = decltype(Layout_o0{{}}.shape());
    Shape_o0 shape_o0;
    Layout_o0 layout_o0;
    Tensor_o0 {tensor_name};

    // Hardware parameters
    int tid, lid, wid;

    // Constructor
    CUTE_DEVICE {op_name}(int tid, int lid, int wid{varlen_modes_constructor_params}, TO0 *d{tensor_name})
    : tid(tid), lid(lid), wid(wid),
    shape_o0({tensor_str}.shape()),
    layout_o0({tensor_str}.layout()),
    {tensor_name}({tensor_str}) {{}}

    // output
{get_output_method}
}};"""

GLOBAL_FUNCTION_SKELETON = """template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK{parameter_types}>
__global__ void {kernel_name}(
{gmem_ptrs},
{varlens}
) {{
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int wid = tid / 32;

{nvop_inits}

{nvop_calls}
}};"""


CUDA_SOURCE_SKELETON = """#include <stdlib.h>
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

#define CHECK_CUDA2(func)                                              \\
    {{                                                                  \\
        cudaError_t status = (func);                                   \\
        if (status != cudaSuccess) {{                                   \\
            printf("CUDA API failed at line %d with error: %s (%d)\\n", \\
                   __LINE__, cudaGetErrorString(status), status);      \\
        }}                                                              \\
    }}

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))
#define OFFSET_COL(row, col, lda) ((col) * (lda) + (row))

#define DIVUP(a, b) (((a) - 1) / (b) + 1)
#define ALIGN(a,b) (DIVUP(a,b)*(b))

{device_helper_functions}

{nvop_defs}

// GLOBAL FUNCTION
{global_function}"""

GMEM_TENSOR_NVOP_CREATE_SKELETON = """using {op_name}_t = {op_name}<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
{op_name}_t {op_name}_v{{tid, lid, wid{varlen_modes_constructor_args}, d{tensor_name}}};"""

CONSTANT_NVOP_CREATE_SKELETON = """using {op_name}_t = {op_name}<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
using {op_name}_layout_o0 = typename {op_name}_t::Layout_o0;
{dtype} {op_name}_tensor_o0[cosize_v<{op_name}_layout_o0>];
{op_name}_t {op_name}_v{{tid, lid, wid, {op_name}_tensor_o0}};"""

FOR_LOOP_NVOP_CREATE_SKELETON = """using {op_name}_t = {op_name}<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK{visible_op_type_template_args}{non_owning_tensor_type_template_args}>;
{op_name}_t {op_name}_v{{tid, lid, wid{visible_op_as_args}{non_owning_tensor_as_args}}};"""