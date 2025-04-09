cmake_minimum_required(VERSION 3.16)
project(row_majorB_vs_bind_reorder_float LANGUAGES CXX CUDA)

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
add_library(dtc_fp32_spmm_row_majorB_vs_bind_reorder_obj OBJECT dtc-spmm-naive-vs-bind-reorder-gen.cu kernel-gen.inc)
add_library(dtc_fp32_spmm_row_majorB_vs_bind_reorder_demo_obj OBJECT dtc-spmm-naive-vs-bind-reorder-demo.cu kernel-demo.inc)
target_compile_options(dtc_fp32_spmm_row_majorB_vs_bind_reorder_obj PRIVATE ${CUDA_NVCC_FLAGS})
target_compile_options(dtc_fp32_spmm_row_majorB_vs_bind_reorder_demo_obj PRIVATE ${CUDA_NVCC_FLAGS})

# 生成可执行文件
file(GLOB_RECURSE SRC_CUDA_FILES ${SRCDIR}/*.cu)
add_executable(dtc_float_row_majorB_vs_bind_reorder main-row-majorB-vs-bind-reorder.cu ${SRC_CUDA_FILES}
    $<TARGET_OBJECTS:dtc_fp32_spmm_row_majorB_vs_bind_reorder_obj>
)
add_executable(dtc_float_row_majorB_vs_bind_reorder_demo main-row-majorB-vs-bind-reorder.cu ${SRC_CUDA_FILES}
    $<TARGET_OBJECTS:dtc_fp32_spmm_row_majorB_vs_bind_reorder_demo_obj>
)

target_compile_options(dtc_float_row_majorB_vs_bind_reorder PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(dtc_float_row_majorB_vs_bind_reorder_demo PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dtc_float_row_majorB_vs_bind_reorder PRIVATE -lcusparse -lcublas)
target_link_libraries(dtc_float_row_majorB_vs_bind_reorder_demo PRIVATE -lcusparse -lcublas)
target_compile_definitions(dtc_float_row_majorB_vs_bind_reorder PRIVATE f32)
target_compile_definitions(dtc_float_row_majorB_vs_bind_reorder_demo PRIVATE f32)
target_include_directories(dtc_float_row_majorB_vs_bind_reorder PRIVATE  ${CUTLASS_INCLUDE_DIR}
${CUTLASS_COMMON_DIR}
${CUTLASS_UTIL_DIR}
${SRCDIR})
target_include_directories(dtc_float_row_majorB_vs_bind_reorder_demo PRIVATE  ${CUTLASS_INCLUDE_DIR}
${CUTLASS_COMMON_DIR}
${CUTLASS_UTIL_DIR}
${SRCDIR})
