cmake_minimum_required(VERSION 3.16)
project(testbed LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS
    -g -lineinfo -O3 -m64
    -arch=sm_80
    -Xcompiler -fopenmp
    --expt-relaxed-constexpr
    --ptxas-options=-v
    -ftemplate-backtrace-limit=0
)
set(CUDA_LIBS -lcusparse -lcublas)

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")

set(SRCDIR ${CMAKE_SOURCE_DIR}/src)

set(CUTLASS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/cutlass/include)
set(CUTLASS_COMMON_DIR ${CMAKE_SOURCE_DIR}/cutlass/examples/common)
set(CUTLASS_UTIL_DIR ${CMAKE_SOURCE_DIR}/cutlass/tools/util/include)

include_directories(${CUTLASS_INCLUDE_DIR} ${CUTLASS_COMMON_DIR} ${CUTLASS_UTIL_DIR} ${SRCDIR})

set(SRC_CUDA_FILES
    ${SRCDIR}/mmio.cu
    ${SRCDIR}/mmio_highlevel.cu
    ${SRCDIR}/utils.cu
)

add_executable(demo-csrtcf-generated-tf32
    ${PROJECT_SOURCE_DIR}/host_program_csr_tcf.cu
    ${PROJECT_SOURCE_DIR}/main_csr_tcf.cu
    ${SRC_CUDA_FILES}
)
target_compile_definitions(demo-csrtcf-generated-tf32 PRIVATE
    -Df32
    -DTC_BLOCK_K=8
    -DTC_BLOCK_M=16
    -DROW_WINDOW_M=16
    -DCSR_TCF
)
target_link_libraries(demo-csrtcf-generated-tf32 ${CUDA_LIBS})
set_target_properties(demo-csrtcf-generated-tf32 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

add_executable(test-csrtcf-checkpoint
    ${PROJECT_SOURCE_DIR}/test_checkpoint.cu
    ${SRC_CUDA_FILES}
)
target_compile_definitions(test-csrtcf-checkpoint PRIVATE
    -Df32 -DTC_BLOCK_K=8 -DTC_BLOCK_M=16 -DROW_WINDOW_M=16
)
target_include_directories(test-csrtcf-checkpoint PRIVATE
    ${CUTLASS_INCLUDE_DIR} ${CUTLASS_COMMON_DIR} ${CUTLASS_UTIL_DIR} ${SRCDIR}
)
target_compile_options(test-csrtcf-checkpoint PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>
)
target_link_libraries(test-csrtcf-checkpoint ${CUDA_LIBS})
