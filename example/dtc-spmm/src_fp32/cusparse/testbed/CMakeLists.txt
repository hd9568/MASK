cmake_minimum_required(VERSION 3.16)
project(testbed LANGUAGES CXX CUDA)

# CUDA compiler settings
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)



# CUDA flags
set(CUDA_NVCC_FLAGS
        -g
        -lineinfo
        -O3
        -m64
        -Xcompiler -fopenmp-simd
        --expt-relaxed-constexpr
        --ptxas-options=-v
        -ftemplate-backtrace-limit=0
)

set(CUDA_LIBS -lcusparse -lcublas)

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")

# Header file paths
set(SRCDIR ${CMAKE_SOURCE_DIR}/src)

set(CUTLASS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/cutlass/include)
set(CUTLASS_COMMON_DIR ${CMAKE_SOURCE_DIR}/cutlass/examples/common)
set(CUTLASS_UTIL_DIR ${CMAKE_SOURCE_DIR}/cutlass/tools/util/include)

include_directories(${CUTLASS_INCLUDE_DIR} ${CUTLASS_COMMON_DIR} ${CUTLASS_UTIL_DIR} ${SRCDIR})

file(GLOB_RECURSE SRC_CUDA_FILES ${SRCDIR}/*.cu)

add_executable(demo-cusp-tf32 ${PROJECT_SOURCE_DIR}/main.cu ${SRC_CUDA_FILES})

target_compile_options(demo-cusp-tf32 PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(demo-cusp-tf32 PRIVATE -lcusparse -lcublas)
target_compile_definitions(demo-cusp-tf32 PRIVATE f32)
target_include_directories(demo-cusp-tf32 PRIVATE 
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_COMMON_DIR}
    ${CUTLASS_UTIL_DIR}
    ${SRCDIR}
)

add_executable(demo-cusp-spmv-tf32 ${PROJECT_SOURCE_DIR}/main_spmv.cu ${SRC_CUDA_FILES})

target_compile_options(demo-cusp-spmv-tf32 PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(demo-cusp-spmv-tf32 PRIVATE -lcusparse -lcublas)
target_compile_definitions(demo-cusp-spmv-tf32 PRIVATE f32)
target_include_directories(demo-cusp-spmv-tf32 PRIVATE 
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_COMMON_DIR}
    ${CUTLASS_UTIL_DIR}
    ${SRCDIR}
)
