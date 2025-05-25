cmake_minimum_required(VERSION 3.16)
project(dtc LANGUAGES CXX CUDA)

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
)

# Link libraries
set(CUDA_LIBS -lcusparse -lcublas)

# Include source directory
set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB_RECURSE SRC_CUDA_FILES ${SRCDIR}/*.cu)
# Exclude specific files
# set(EXCLUDE_FILES
#         ${SRCDIR}/main_f16.cu
#         ${SRCDIR}/main_f64.cu
#         ${SRCDIR}/mySpMV_main_f16.cu
# )
# list(REMOVE_ITEM SRC_CUDA_FILES ${EXCLUDE_FILES})

# SRC_FP16 directory
# set(SRCDIR_FP16 ${CMAKE_CURRENT_SOURCE_DIR}/src_fp16)

# #> col major B
# add_subdirectory(${SRCDIR_FP16}/col_majorB/)
# file(GLOB_RECURSE SRC_FP16_COLMAJORB_FILES ${SRCDIR_FP16}/col_majorB/*.cu)
# add_executable(dtc_half_col_majorB ${SRC_CUDA_FILES} ${SRC_FP16_COLMAJORB_FILES})
# target_compile_options(dtc_half_col_majorB PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR})
# target_link_libraries(dtc_half_col_majorB PRIVATE ${CUDA_LIBS})


# #> row major B
# file(GLOB_RECURSE SRC_FP16_ROWMAJORB_FILES ${SRCDIR_FP16}/row_majorB/*.cu)
# add_executable(dtc_half_row_majorB ${SRC_CUDA_FILES} ${SRC_FP16_ROWMAJORB_FILES})
# target_compile_options(dtc_half_row_majorB PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR})
# target_link_libraries(dtc_half_row_majorB PRIVATE ${CUDA_LIBS})


# #> row major B val sidx bind
# file(GLOB_RECURSE SRC_FP16_ROWMAJORB_VS_BIND_FILES ${SRCDIR_FP16}/row_majorB_vs_bind/*.cu)
# add_executable(dtc_half_row_majorB_vs_bind ${SRC_CUDA_FILES} ${SRC_FP16_ROWMAJORB_VS_BIND_FILES})
# target_compile_options(dtc_half_row_majorB_vs_bind PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR})
# target_link_libraries(dtc_half_row_majorB_vs_bind PRIVATE ${CUDA_LIBS})


#> row major B val sidx bind reorder
# add_subdirectory(${SRCDIR_FP16}/row_majorB_vs_bind_reorder/)
# file(GLOB_RECURSE SRC_FP16_ROWMAJORB_VS_BIND_REORDER_FILES ${SRCDIR_FP16}/row_majorB_vs_bind_reorder/*.cu)
# add_executable(dtc_half_row_majorB_vs_bind_reorder ${SRC_CUDA_FILES} ${SRC_FP16_ROWMAJORB_VS_BIND_REORDER_FILES})
# target_compile_options(dtc_half_row_majorB_vs_bind_reorder PRIVATE ${CUDA_NVCC_FLAGS} -I${SRCDIR})
# target_link_libraries(dtc_half_row_majorB_vs_bind_reorder PRIVATE ${CUDA_LIBS})

# add_subdirectory(${SRCDIR_FP16}/flash_llm_vs_bind_reorder/)
# add_subdirectory(${SRCDIR_FP16}/flash_llm_vs_bind_reorder_multi_warp/)
# add_subdirectory(${SRCDIR_FP16}/sr_bcrs/)

# # SRC_FP32 directory
# set(SRCDIR_FP32 ${CMAKE_CURRENT_SOURCE_DIR}/src_fp32)
# add_subdirectory(${SRCDIR_FP32}/row_majorB_vs_bind_reorder/)

# file(GLOB_RECURSE SRC_FP32_CUDA_FILES ${SRCDIR_FP32}/*.cu)
# add_executable(dtc_float ${SRC_CUDA_FILES} ${SRC_FP32_CUDA_FILES})
# target_compile_options(dtc_float PRIVATE ${CUDA_NVCC_FLAGS})
# target_link_libraries(dtc_float PRIVATE ${CUDA_LIBS})
# target_compile_definitions(dtc_float PRIVATE f32)

# # SRC_FP64 directory
# set(SRCDIR_FP64 ${CMAKE_CURRENT_SOURCE_DIR}/src_fp64)
# file(GLOB_RECURSE SRC_FP64_CUDA_FILES ${SRCDIR_FP64}/*.cu)
# add_executable(dtc_double ${SRC_CUDA_FILES} ${SRC_FP64_CUDA_FILES})
# target_compile_options(dtc_double PRIVATE ${CUDA_NVCC_FLAGS})
# target_link_libraries(dtc_double PRIVATE ${CUDA_LIBS})
# target_compile_definitions(dtc_double PRIVATE f64)

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")

# Targets
# add_executable(spmv_double ${ALL_CUDA_FILES} ${SRCDIR}/main_f64.cu)
# target_compile_options(spmv_double PRIVATE ${CUDA_NVCC_FLAGS} -D f64)
# target_link_libraries(spmv_double PRIVATE ${CUDA_LIBS})

# add_executable(spmv_half ${ALL_CUDA_FILES} ${SRCDIR}/main_f16.cu)
# target_compile_options(spmv_half PRIVATE ${CUDA_NVCC_FLAGS})
# target_link_libraries(spmv_half PRIVATE ${CUDA_LIBS})

# add_executable(myhalf ${ALL_CUDA_FILES} ${SRCDIR}/mySpMV_main_f16.cu)
# target_compile_options(myhalf PRIVATE ${CUDA_NVCC_FLAGS})
# target_link_libraries(myhalf PRIVATE ${CUDA_LIBS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src_fp16/acc/testbed/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src_fp16/dtc/testbed/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src_fp16/bitbsr/testbed/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src_fp16/sr_bcrs/testbed/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src_fp16/cusparse/testbed/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src_fp32/acc/testbed/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src_fp32/dtc/testbed/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src_fp32/bitbsr/testbed/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src_fp32/sr_bcrs/testbed/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src_fp32/cusparse/testbed/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src_fp32/acc-ldg/testbed/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src_fp32/dtc-ldg/testbed/)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src_fp16/flash_llm_vs_bind_reorder_multi_warp/)
add_custom_target(demos DEPENDS 
    demo-acc-fp16 
    demo-acc-spmv-fp16 
    demo-acc-tf32 
    demo-acc-ldg-tf32 
    demo-acc-spmv-tf32 
    demo-bitbsr-fp16 
    demo-bitbsr-spmv-fp16 
    demo-bitbsr-tf32 
    demo-bitbsr-spmv-tf32 
    demo-cusp-fp16 
    demo-cusp-spmv-fp16 
    demo-cusp-tf32 
    demo-cusp-spmv-tf32 
    demo-dtc-fp16 
    demo-dtc-spmv-fp16 
    demo-dtc-tf32 
    demo-dtc-ldg-tf32 
    demo-dtc-spmv-tf32 
    demo-srbcrs-fp16 
    demo-srbcrs-tf32 
    demo-srbcrs-16x16-fp16
    demo-srbcrs-16x16-spmv-fp16
    demo-srbcrs-16x8-tf32
    demo-srbcrs-16x8-spmv-tf32
)

add_custom_target(demo-ablation DEPENDS
    demo-dtc-fp16
    demo-dtc-nopipe-fp16
    demo-dtc-nopipe-nopre-fp16
    demo-dtc-nopipe-nopre-noswz-fp16
    demo-acc-fp16
    demo-acc-nopipe-fp16
    demo-acc-nopipe-nopre-fp16
    demo-acc-nopipe-nopre-noswz-fp16
    demo-bitbsr-fp16
    demo-bitbsr-nopipe-fp16
    demo-bitbsr-nopipe-nopre-fp16
    demo-bitbsr-nopipe-nopre-noswz-fp16
    demo-srbcrs-fp16
    demo-srbcrs-nopipe-fp16
    demo-srbcrs-nopipe-nopre-fp16
    demo-srbcrs-nopipe-nopre-noswz-fp16
)

add_custom_target(demo-optexp DEPENDS
    demo-dtc-fp16
    demo-dtc-nobind-fp16
    demo-dtc-noreorder-fp16
)

add_custom_target(demo-fuse DEPENDS
    flash_llm_half_row_majorB_vs_bind_reorder_mw
    flash_llm_half_row_majorB_vs_bind_reorder_mw_nofuse
)

add_custom_target(demo-cusparse DEPENDS
    demo-cusp-fp16
    demo-cusp-spmv-fp16
    demo-cusp-tf32
    demo-cusp-spmv-tf32
)
