#ifndef __DASP_UTILS_H__
#define __DASP_UTILS_H__
#include "common.h"
#include <bits/stdc++.h>

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

using namespace std;
// 定义 always_false<T>，用于 static_assert
template <typename T>
struct always_false : std::false_type {};

// C++17+ 简写版本
template <typename T>
constexpr bool always_false_v = always_false<T>::value;

template <typename T>
void parseArgValue(const char* arg, T& val) {
    if constexpr (is_same<T, int>::value) {
        val = stoi(arg);
    } else if constexpr (is_same<T, float>::value) {
        val = stof(arg);
    } else if constexpr (is_same<T, double>::value) {
        val = stod(arg);
    } else if constexpr (is_same<T, string>::value) {
        val = string(arg);
    } else {
        static_assert(always_false_v<T>, "Unsupported type");
    }
}

template <typename T, typename... Args>
void parseInput(int argc, char** argv, T& val, const string& str_val, Args&... args) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], str_val.c_str()) == 0 && i + 1 < argc) {
            parseArgValue(argv[i + 1], val);
        } 
    }
    if constexpr (sizeof...(args) > 0) {
        parseInput(argc, argv, args...);
    }
}

#endif // __DASP_UTILS_H__