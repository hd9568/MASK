CUDA_SOURCE_SKELETON = """
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

{{device_helper_functions}}

{{nvops}}

{{short_pipe_device_functions}}

{{fill_device_functions}}

{{empty_device_functions}}

{{loop_step_device_functions}}

{{remainder_device_functions}}

{{global_function}}
"""

DTC_HELPER_FUNCTIONS = """
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
"""

GLOBAL_FUNCTION_SKELETON = """
template <
	class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK,
	{{all_parameters}}
	{{shifts}}
	int nbuf_smem, int nbuf_rmem>
__global__ void {{kernel_name}}(
	{{gmem_ptrs}}
	{{varlens}}
) {
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int wid = tid / 32;

    {{gmem_tensor_init}}

	{{op_init}}

	{{prologue}}

	{{dispatch}}

	{{epilogue}}
}
"""

GLOBAL_FUNCTION_DISPATCH_SKELETON = """
if (k <= {sum_shifts}) {{
    {short_pipe_dispatch}
}} else {{
    {fill_dispatch}

    int i;
    for (i = {sum_shifts}; i + {nbuf} <= k; i += {nbuf}) {{
        {loop_step_dispatch}
    }}

    {remainder_dispatch}

    {empty_dispatch}
}}
"""

FILL_DEVICE_FUNCTION_SKELETON = """
template <{op_types}>
__device__ void fill(int i, int l, int r, {op_as_params}) {{
    {fill_body}
}}
"""

EMPTY_FROM_IDX_DEVICE_FUNCTION_SKELETON = """
template <{op_types}>
__device__ void empty_{from_idx}(int i, int l, int r, {op_as_params}) {{
    {empty_from_idx_body}
}}
"""

EMPTY_DEVICE_FUNCTION_SKELETON = """
template <{op_types}>
__device__ void empty(int i, int l, int r, {op_as_params}) {{
    {empty_dispatch_body}
}}
"""

LOOP_STEP_DEVICE_FUNCTION_SKELETON = """
template <{op_types}>
__device__ void loop_step(int i, int l, int r, {op_as_params}) {{
    {loop_step_body}
}}
"""

REMAINS_R_DEVICE_FUNCTION_SKELETON = """
template <{op_types}>
__device__ void remains_{r}(int i, int l, int r, {op_as_params}) {{
    {remains_r_body}
}}
"""

REMAINDER_DEVICE_FUNCTION_SKELETON = """
template <{op_types}>
__device__ void remainder(int i, int l, int r, {op_as_params}) {{
    {remainder_dispatch_body}
}}
"""

