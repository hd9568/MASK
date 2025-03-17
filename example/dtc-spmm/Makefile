#compilers
CC=nvcc
NVCC_FLAGS = -std=c++17 -g -lineinfo -O3 -m64 -gencode arch=compute_80,code=sm_80

# #ENVIRONMENT_PARAMETERS
# CUDA_INSTALL_PATH = /usr/local/cuda-12.0

CUDA_LIBS = -lcusparse -lcublas
LIBS =  -lineinfo $(CUDA_LIBS)

#options
OPTIONS = -Xcompiler -fopenmp-simd

CURDIR := $(PWD)

dtc-spmm.o: dtc-spmm.cu
	$(CC) $(NVCC_FLAGS) -c $^ -o $@ $(OPTIONS) $(LIBS)

dtc: main.cu dtc-spmm.o
	$(CC) $(NVCC_FLAGS) $^ -o $@ $(OPTIONS) $(LIBS) 

 

# Clean rule
clean:
	rm -rf *.o
	rm -rf dtc