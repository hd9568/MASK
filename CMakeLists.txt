project(sparsene LANGUAGE C++ CUDA)

find_library(CUDAToolkit REQUIRED)

add_library(kernel baseline.cu)

add_executable(main main.cpp)

target_link_libraries(main private CUDA::cudart)
target_link_libraries(main private kernel)

