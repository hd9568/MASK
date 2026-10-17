#compilers
CC=nvcc
NVCC_FLAGS = -std=c++17 -g -lineinfo -O3 -m64 -gencode arch=compute_80,code=sm_80 --expt-relaxed-constexpr --ptxas-options=-v

# #ENVIRONMENT_PARAMETERS
# CUDA_INSTALL_PATH = /usr/local/cuda-12.0

CUDA_LIBS = -lcusparse -lcublas
LIBS =  -lineinfo $(CUDA_LIBS)

#options
OPTIONS = -Xcompiler -fopenmp-simd

CURDIR := $(PWD)

dtc-spmm.o: dtc-spmm.cu
	$(CC) $(NVCC_FLAGS) -Icutlass/include -Icutlass/examples/common -Icutlass/tools/util/include -c $^ -o $@ $(OPTIONS) $(LIBS) 

dtc-spmm-gen.o: dtc-spmm-gen.cu kernel-gen.inc
	$(CC) $(NVCC_FLAGS) -Icutlass/include -Icutlass/examples/common -Icutlass/tools/util/include -c $< -o $@ $(OPTIONS) $(LIBS) 

dtc-spmm-demo.o: dtc-spmm-demo.cu
	$(CC) $(NVCC_FLAGS) -Icutlass/include -Icutlass/examples/common -Icutlass/tools/util/include -c $< -o $@ $(OPTIONS) $(LIBS) 

dtc: main.cu dtc-spmm.o
	$(CC) $(NVCC_FLAGS) -Icutlass/include -Icutlass/examples/common -Icutlass/tools/util/include $^ -o $@ $(OPTIONS) $(LIBS)

demo: main.cu dtc-spmm-demo.o
	$(CC) $(NVCC_FLAGS) -Icutlass/include -Icutlass/examples/common -Icutlass/tools/util/include $^ -o $@ $(OPTIONS) $(LIBS)

gen: main.cu dtc-spmm-gen.o
	$(CC) $(NVCC_FLAGS) -Icutlass/include -Icutlass/examples/common -Icutlass/tools/util/include $^ -o $@ $(OPTIONS) $(LIBS)

test_cute: test_cute.cu layouts.hpp
	$(CC) $(NVCC_FLAGS) -Icutlass/include -Icutlass/examples/common -Icutlass/tools/util/include $< -o $@ $(OPTIONS) $(LIBS)
 
sm80_demo: sm80_demo.cu
	$(CC) $(NVCC_FLAGS) -Icutlass/include -Icutlass/examples/common -Icutlass/tools/util/include $< -o $@ $(OPTIONS) $(LIBS)

# Clean rule
clean:
	rm -rf *.o
	rm -rf dtc
	rm -rf test_cute
