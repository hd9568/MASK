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

file(GLOB_RECURSE SRC_CUDA_FILES ${SRCDIR}/*.cu)

add_library(flash_llm_gen_multi_warp_obj OBJECT flash-llm-gen.cu kernel-gen.inc)
target_compile_options(flash_llm_gen_multi_warp_obj PRIVATE ${CUDA_NVCC_FLAGS})
add_executable(flash_llm_half_row_majorB_vs_bind_reorder_mw main_flash_llm_row_major.cu ${SRC_CUDA_FILES}
    $<TARGET_OBJECTS:flash_llm_gen_multi_warp_obj>
)
target_compile_options(flash_llm_half_row_majorB_vs_bind_reorder_mw PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR})
target_link_libraries(flash_llm_half_row_majorB_vs_bind_reorder_mw PRIVATE -lcusparse -lcublas)

add_library(flash_llm_gen_multi_warp_obj_nofuse OBJECT flash-llm-gen-nofuse.cu kernel-gen-nofuse.inc)
target_compile_options(flash_llm_gen_multi_warp_obj_nofuse PRIVATE ${CUDA_NVCC_FLAGS})
add_executable(flash_llm_half_row_majorB_vs_bind_reorder_mw_nofuse main_flash_llm_row_major_nofuse.cu ${SRC_CUDA_FILES}
    $<TARGET_OBJECTS:flash_llm_gen_multi_warp_obj_nofuse>
)
target_compile_options(flash_llm_half_row_majorB_vs_bind_reorder_mw_nofuse PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR})
target_link_libraries(flash_llm_half_row_majorB_vs_bind_reorder_mw_nofuse PRIVATE -lcusparse -lcublas)
