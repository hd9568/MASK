cmake_minimum_required(VERSION 3.16)
project(testbed LANGUAGES CXX CUDA)

# CUDA compiler settings
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

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
    -Xcompiler -fopenmp
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
# file(GLOB PLAN_FILES "${PROJECT_SOURCE_DIR}/plans/plan_*.inc")
set(PLAN_FILES
    "${PROJECT_SOURCE_DIR}/plans/plan_0.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_9.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_18.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_27.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_36.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_45.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_54.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_63.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_72.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_81.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_90.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_99.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_108.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_117.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_126.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_135.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_144.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_153.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_162.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_171.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_180.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_189.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_198.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_207.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_216.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_225.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_234.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_243.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_252.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_261.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_270.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_279.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_288.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_297.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_306.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_315.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_324.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_333.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_342.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_351.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_360.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_369.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_378.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_387.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_396.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_405.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_414.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_423.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_432.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_441.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_450.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_459.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_468.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_477.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_486.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_495.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_504.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_513.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_522.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_531.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_540.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_549.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_558.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_567.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_576.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_585.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_594.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_603.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_612.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_621.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_630.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_639.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_648.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_657.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_666.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_675.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_684.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_693.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_702.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_711.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_720.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_729.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_738.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_747.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_756.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_765.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_774.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_783.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_792.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_801.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_810.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_819.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_828.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_837.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_846.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_855.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_864.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_873.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_882.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_891.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_900.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_909.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_918.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_927.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_936.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_945.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_954.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_963.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_972.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_981.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_990.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_999.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1008.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1017.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1026.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1035.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1044.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1053.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1062.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1071.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1080.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1089.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1098.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1107.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1116.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1125.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1134.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1143.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1152.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1161.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1170.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1179.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1188.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1197.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1206.inc"
    "${PROJECT_SOURCE_DIR}/plans/plan_1215.inc"
)
file(GLOB_RECURSE SRC_CUDA_FILES ${SRCDIR}/*.cu)
set(PLAN_HOST_TEMPLATE ${PROJECT_SOURCE_DIR}/host_program_plan.cu)

# # Generate a target for each plan file
foreach(PLAN_FILE ${PLAN_FILES})
    # Extract the base name of the plan file (e.g., plan_0 from plan_0.inc)
    get_filename_component(PLAN_NAME ${PLAN_FILE} NAME_WE)

    # Define the output host_program.cu for this plan
    set(GENERATED_HOST_PROGRAM ${CMAKE_BINARY_DIR}/${PLAN_NAME}_host_program.cu)

    # Generate the modified host_program.cu
    add_custom_command(
        OUTPUT ${GENERATED_HOST_PROGRAM}
        COMMAND ${CMAKE_COMMAND} -E echo "Generating ${GENERATED_HOST_PROGRAM} for plan ${PLAN_NAME}"
        COMMAND python "${PROJECT_SOURCE_DIR}/scripts/replace_in_file.py"
            "${PLAN_HOST_TEMPLATE}"
            "#include \"kernel.inc\""
            "#include \"plans/${PLAN_NAME}.inc\""
            "${GENERATED_HOST_PROGRAM}"
        DEPENDS ${PLAN_HOST_TEMPLATE} ${PLAN_FILE}
        VERBATIM
    )

    # Add an executable target for this plan
    add_executable(dtc-${PLAN_NAME}-tf32 ${GENERATED_HOST_PROGRAM} ${SRC_CUDA_FILES} ${PROJECT_SOURCE_DIR}/main.cu)

    target_compile_options(dtc-${PLAN_NAME}-tf32 PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(dtc-${PLAN_NAME}-tf32 PRIVATE -lcusparse -lcublas OpenMP::OpenMP_CXX)
    target_compile_definitions(dtc-${PLAN_NAME}-tf32 PRIVATE f32)
    target_include_directories(dtc-${PLAN_NAME}-tf32 PRIVATE 
        ${CUTLASS_INCLUDE_DIR}
        ${CUTLASS_COMMON_DIR}
        ${CUTLASS_UTIL_DIR}
        ${SRCDIR}
    )
endforeach()

add_executable(demo-dtc-tf32 ${PROJECT_SOURCE_DIR}/host_program.cu ${PROJECT_SOURCE_DIR}/main.cu ${SRC_CUDA_FILES})

target_compile_options(demo-dtc-tf32 PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(demo-dtc-tf32 PRIVATE -lcusparse -lcublas OpenMP::OpenMP_CXX)
target_compile_definitions(demo-dtc-tf32 PRIVATE f32)
target_include_directories(demo-dtc-tf32 PRIVATE 
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_COMMON_DIR}
    ${CUTLASS_UTIL_DIR}
    ${SRCDIR}
)

add_executable(demo-dtc-generated-tf32 ${PROJECT_SOURCE_DIR}/host_program_generated_results.cu ${PROJECT_SOURCE_DIR}/main.cu ${SRC_CUDA_FILES})

target_compile_options(demo-dtc-generated-tf32 PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(demo-dtc-generated-tf32 PRIVATE -lcusparse -lcublas OpenMP::OpenMP_CXX)
target_compile_definitions(demo-dtc-generated-tf32 PRIVATE f32)
target_include_directories(demo-dtc-generated-tf32 PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_COMMON_DIR}
    ${CUTLASS_UTIL_DIR}
    ${SRCDIR}
)

add_executable(demo-dtc-multi-binding-tf32 ${PROJECT_SOURCE_DIR}/host_program_multi_binding.cu ${PROJECT_SOURCE_DIR}/main_multi_binding.cu ${SRC_CUDA_FILES})

target_compile_options(demo-dtc-multi-binding-tf32 PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(demo-dtc-multi-binding-tf32 PRIVATE -lcusparse -lcublas OpenMP::OpenMP_CXX)
target_compile_definitions(demo-dtc-multi-binding-tf32 PRIVATE f32)
target_include_directories(demo-dtc-multi-binding-tf32 PRIVATE 
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_COMMON_DIR}
    ${CUTLASS_UTIL_DIR}
    ${SRCDIR}
)

add_executable(demo-dtc-strict-lb-tf32 ${PROJECT_SOURCE_DIR}/host_program_strict_lb.cu ${PROJECT_SOURCE_DIR}/main_strict_lb.cu ${SRC_CUDA_FILES})

target_compile_options(demo-dtc-strict-lb-tf32 PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(demo-dtc-strict-lb-tf32 PRIVATE -lcusparse -lcublas OpenMP::OpenMP_CXX)
target_compile_definitions(demo-dtc-strict-lb-tf32 PRIVATE f32)
target_include_directories(demo-dtc-strict-lb-tf32 PRIVATE 
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_COMMON_DIR}
    ${CUTLASS_UTIL_DIR}
    ${SRCDIR}
)

# add_executable(demo-dtc-spmv-tf32 ${PROJECT_SOURCE_DIR}/host_program_spmv.cu ${PROJECT_SOURCE_DIR}/main_spmv.cu ${SRC_CUDA_FILES})

# target_compile_options(demo-dtc-spmv-tf32 PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR} -I${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(demo-dtc-spmv-tf32 PRIVATE -lcusparse -lcublas)
# target_compile_definitions(demo-dtc-spmv-tf32 PRIVATE f32)
# target_include_directories(demo-dtc-spmv-tf32 PRIVATE 
#     ${CUTLASS_INCLUDE_DIR}
#     ${CUTLASS_COMMON_DIR}
#     ${CUTLASS_UTIL_DIR}
#     ${SRCDIR}
# )
