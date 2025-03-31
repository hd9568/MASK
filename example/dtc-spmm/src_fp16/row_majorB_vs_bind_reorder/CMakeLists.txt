cmake_minimum_required(VERSION 3.16)
project(row_majorB_vs_bind_reorder LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 编译选项
set(CUDA_NVCC_FLAGS
    -g
    -G
    -lineinfo
    -O3
    -m64
    -Xcompiler -fopenmp-simd
)

# 头文件路径
set(SRCDIR ${CMAKE_SOURCE_DIR}/src)

set(CUTLASS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/cutlass/include)
set(CUTLASS_COMMON_DIR ${CMAKE_SOURCE_DIR}/cutlass/examples/common)
set(CUTLASS_UTIL_DIR ${CMAKE_SOURCE_DIR}/cutlass/tools/util/include)

include_directories(${CUTLASS_INCLUDE_DIR} ${CUTLASS_COMMON_DIR} ${CUTLASS_UTIL_DIR} ${SRCDIR})

# 编译 dtc-spmm-gen.cu 和 dtc-spmm.cu 为对象文件
add_library(dtc_spmm_row_majorB_vs_bind_reorder_obj OBJECT dtc-spmm-naive-vs-bind-reorder-gen.cu kernel-gen.inc)


# 生成可执行文件
file(GLOB_RECURSE SRC_CUDA_FILES ${SRCDIR}/*.cu)
add_executable(dtc_half_row_majorB_vs_bind_reorder main-row-majorB-vs-bind-reorder.cu ${SRC_CUDA_FILES}
    $<TARGET_OBJECTS:dtc_spmm_row_majorB_vs_bind_reorder_obj>
)

target_compile_options(dtc_half_row_majorB_vs_bind_reorder PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dtc_half_row_majorB_vs_bind_reorder PRIVATE -lcusparse -lcublas)
