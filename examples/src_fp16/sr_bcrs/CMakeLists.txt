cmake_minimum_required(VERSION 3.16)
project(sr_bcrs_fp16 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 编译选项
set(CUDA_NVCC_FLAGS
    -g
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
add_library(sr_bcrs_fp16_obj OBJECT sr-bcrs-gen.cu kernel-gen.inc)
target_compile_options(sr_bcrs_fp16_obj PRIVATE ${CUDA_NVCC_FLAGS})


# 生成可执行文件
file(GLOB_RECURSE SRC_CUDA_FILES ${SRCDIR}/*.cu)
add_executable(sr_bcrs_fp16 main-sr-bcrs.cu ${SRC_CUDA_FILES}
    $<TARGET_OBJECTS:sr_bcrs_fp16_obj>
)

target_compile_options(sr_bcrs_fp16 PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sr_bcrs_fp16 PRIVATE -lcusparse -lcublas)