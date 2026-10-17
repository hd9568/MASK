cmake_minimum_required(VERSION 3.16)
project(flash_llm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 编译选项
set(CUDA_NVCC_FLAGS
    -g
    -lineinfo
    -O3
    -m64
    -Xcompiler -fopenmp-simd
    --ptxas-options=-v
)

# 头文件路径
set(SRCDIR ${CMAKE_SOURCE_DIR}/src)
set(CUTLASS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/cutlass/include)
set(CUTLASS_COMMON_DIR ${CMAKE_SOURCE_DIR}/cutlass/examples/common)
set(CUTLASS_UTIL_DIR ${CMAKE_SOURCE_DIR}/cutlass/tools/util/include)

include_directories(${CUTLASS_INCLUDE_DIR} ${CUTLASS_COMMON_DIR} ${CUTLASS_UTIL_DIR} ${SRCDIR})

# 编译 dtc-spmm-gen.cu 和 dtc-spmm.cu 为对象文件
add_library(flash_llm_gen_obj OBJECT flash-llm-gen.cu kernel-gen.inc)
target_compile_options(flash_llm_gen_obj PRIVATE ${CUDA_NVCC_FLAGS})
# 生成可执行文件 gen_col_majorB
file(GLOB_RECURSE SRC_CUDA_FILES ${SRCDIR}/*.cu)
add_executable(flash_llm_half_row_majorB_vs_bind_reorder main_flash_llm_row_major.cu ${SRC_CUDA_FILES}
    $<TARGET_OBJECTS:flash_llm_gen_obj>
)

target_compile_options(flash_llm_half_row_majorB_vs_bind_reorder PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR})
target_link_libraries(flash_llm_half_row_majorB_vs_bind_reorder PRIVATE -lcusparse -lcublas)
