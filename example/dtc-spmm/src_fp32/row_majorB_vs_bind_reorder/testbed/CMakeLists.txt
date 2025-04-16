cmake_minimum_required(VERSION 3.16)
project(testbed LANGUAGES CXX CUDA)

# CUDA compiler settings
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80)

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

# Find all plan_xxx.inc files in the plans directory
file(GLOB PLAN_FILES "${CMAKE_SOURCE_DIR}/plans/plan_*.inc")
file(GLOB_RECURSE SRC_CUDA_FILES ${SRCDIR}/*.cu)

# Generate a target for each plan file
foreach(PLAN_FILE ${PLAN_FILES})
    # Extract the base name of the plan file (e.g., plan_0 from plan_0.inc)
    get_filename_component(PLAN_NAME ${PLAN_FILE} NAME_WE)

    # Define the output host_program.cu for this plan
    set(GENERATED_HOST_PROGRAM ${CMAKE_BINARY_DIR}/${PLAN_NAME}_host_program.cu)

    # Generate the modified host_program.cu
    add_custom_command(
        OUTPUT ${GENERATED_HOST_PROGRAM}
        COMMAND ${CMAKE_COMMAND} -E echo "Generating ${GENERATED_HOST_PROGRAM} for plan ${PLAN_NAME}"
        COMMAND python "${CMAKE_SOURCE_DIR}/scripts/replace_in_file.py"
            "${CMAKE_SOURCE_DIR}/host_program.cu"
            "#include \"kernel.inc\""
            "#include \"plans/${PLAN_NAME}.inc\""
            "${GENERATED_HOST_PROGRAM}"
        DEPENDS ${CMAKE_SOURCE_DIR}/host_program.cu ${PLAN_FILE}
        VERBATIM
    )

    # Add an executable target for this plan
    add_executable(${PLAN_NAME} ${GENERATED_HOST_PROGRAM} ${SRC_CUDA_FILES} ${CMAKE_SOURCE_DIR}/main.cu)

    target_compile_options(${PLAN_NAME} PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${PLAN_NAME} PRIVATE -lcusparse -lcublas)
    target_compile_definitions(${PLAN_NAME} PRIVATE f32)
    target_include_directories(${PLAN_NAME} PRIVATE 
        ${CUTLASS_INCLUDE_DIR}
        ${CUTLASS_COMMON_DIR}
        ${CUTLASS_UTIL_DIR}
        ${SRCDIR}
    )
endforeach()

add_executable(demo ${CMAKE_SOURCE_DIR}/host_program.cu ${CMAKE_SOURCE_DIR}/main.cu ${SRC_CUDA_FILES})

target_compile_options(demo PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(demo PRIVATE -lcusparse -lcublas)
target_compile_definitions(demo PRIVATE f32)
target_include_directories(demo PRIVATE 
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_COMMON_DIR}
    ${CUTLASS_UTIL_DIR}
    ${SRCDIR}
)
