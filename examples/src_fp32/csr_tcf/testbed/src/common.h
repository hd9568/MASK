#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

// #include <helper_cuda.h>
// #include <helper_functions.h>

#include <cusparse.h>
#include <cublas_v2.h>

#include "omp.h"
#include "mmio_highlevel.h"

#ifndef __DASP_COMMON_H__
#define __DASP_COMMON_H__

#ifdef f64
    #define MAT_VAL_TYPE double
#elif defined(f32)
    #define MAT_VAL_TYPE float
#else
    #define MAT_VAL_TYPE half
#endif


#define WARP_SIZE 32
#define BlockSize 8

#define MMA_M 8
#define MMA_N 8
#define MMA_K 4

#define MAT_PTR_TYPE int

#define NEW_CID_TYPE int


#define GET_BIT_REST(x)  ((unsigned int)(x << 2) >> 2)

#define SET_16_BIT(dst, src, index)  \
    dst &= ~(0xffff << (index << 4)); \
    dst |= (src << (index << 4))

#define SET_8_BIT(dst, src, index)  \
    dst &= ~(0xff << (index << 3)); \
    dst |= (src << (index << 3))

#define SET_4_BIT(dst, src, index) \
    dst &= ~(0xf << (index << 2)); \
    dst |= (src << (index << 2))

#define SET_2_BIT(dst, src) dst |= src << 30

#define GET_16_BIT(src, index) ((src >> (index << 4)) & 0xffff)
#define GET_8_BIT(src, index) ((src >> (index << 3)) & 0xff)
#define GET_4_BIT(src, index) ((src >> (index << 2)) & 0xf)
#define GET_2_BIT(src) ((src >> 30) & 0b11)
#define omp_valve 1e4

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(-1);                                                              \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(-1);                                                              \
    }                                                                          \
}

#endif