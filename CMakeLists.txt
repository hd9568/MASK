cmake_minimum_required(VERSION 3.16)
project(sparsene LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(draft/ChainSaw)

#add_library(kernel baseline.cu)
#
#add_executable(main main.cpp)
#
#target_link_libraries(main private CUDA::cudart)
#target_link_libraries(main private kernel)

