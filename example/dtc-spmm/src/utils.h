#ifndef __DASP_UTILS_H__
#define __DASP_UTILS_H__
#include "common.h"

int BinarySearch(int *arr, int len, int target);

void swap_key(int *a, int *b);

// quick sort key (child function)
int partition_key(int *key, int length, int pivot_index);

// quick sort key (child function)
int partition_key_idx(int *key, int *len, int length, int pivot_index);

// quick sort key (main function)
void quick_sort_key(int *key, int length);

void quick_sort_key_idx(int *key, int *len, int length);

void initVec(MAT_VAL_TYPE *vec, int length);

#ifdef f64
__device__ __forceinline__ void mma_m8n8k4(MAT_VAL_TYPE *acc, MAT_VAL_TYPE &frag_a, MAT_VAL_TYPE &frag_b);
#endif


int get_max(int *arr, int len);

void count_sort(int *arr, int *idx, int len, int exp);

void count_sort_asce(int *arr, int *idx, int len, int exp);

void radix_sort(int *arr, int *idx, int len);

void radix_sort_asce(int *arr, int *idx, int len);

#endif // __DASP_UTILS_H__