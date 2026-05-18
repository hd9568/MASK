#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <omp.h>

#include "common.h"
#include "utils.h"

#include <cute/tensor.hpp>

using namespace std;
using namespace cute;

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))
#define OFFSET_COL(row, col, lda) ((col) * (lda) + (row))

#define ROW_WINDOW_M 16
#define TC_BLOCK_K 8
#define TC_BLOCK_M 16

template <int TILE_B>
extern void dtc_spmm_test_val_sidx_bind_reorder(int M,
                                                int N,
                                                int K,
                                                int TCBlock_num,
                                                int nnz,
                                                int* val_sidx,
                                                int* val_soff,
                                                float* B_val,
                                                int* val_coo_idx,
                                                float* val_coo_val,
                                                int* val_coo_off,
                                                float* C,
                                                bool is_ncu_test,
                                                int warmup_time,
                                                int execute_time);

extern template void dtc_spmm_test_val_sidx_bind_reorder<64>(int M,
                                                             int N,
                                                             int K,
                                                             int TCBlock_num,
                                                             int nnz,
                                                             int* val_sidx,
                                                             int* val_soff,
                                                             float* B_val,
                                                             int* val_coo_idx,
                                                             float* val_coo_val,
                                                             int* val_coo_off,
                                                             float* C,
                                                             bool is_ncu_test,
                                                             int warmup_time,
                                                             int execute_time);
extern template void dtc_spmm_test_val_sidx_bind_reorder<32>(int M,
                                                             int N,
                                                             int K,
                                                             int TCBlock_num,
                                                             int nnz,
                                                             int* val_sidx,
                                                             int* val_soff,
                                                             float* B_val,
                                                             int* val_coo_idx,
                                                             float* val_coo_val,
                                                             int* val_coo_off,
                                                             float* C,
                                                             bool is_ncu_test,
                                                             int warmup_time,
                                                             int execute_time);
extern template void dtc_spmm_test_val_sidx_bind_reorder<16>(int M,
                                                             int N,
                                                             int K,
                                                             int TCBlock_num,
                                                             int nnz,
                                                             int* val_sidx,
                                                             int* val_soff,
                                                             float* B_val,
                                                             int* val_coo_idx,
                                                             float* val_coo_val,
                                                             int* val_coo_off,
                                                             float* C,
                                                             bool is_ncu_test,
                                                             int warmup_time,
                                                             int execute_time);
extern int sparsene_generated_tile_b();

int binarySearch(int* arr, int l, int r, int target) {
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == target)
            return mid;
        if (arr[mid] < target)
            l = mid + 1;
        else
            r = mid - 1;
    }
    return -1; // 未找到
}

void initVector(float* vec, int length, float sparsity) {
    if (sparsity < 0.0f || sparsity > 1.0f)
        return;
    auto seed = std::time(nullptr);
    std::srand(1234);
    cout << seed << endl;
    // std::srand(1);
    for (int i = 0; i < length; ++i) {
        if (static_cast<float>(rand()) / RAND_MAX < sparsity) {
            vec[i] = float(0.0f);
        } else {
            // vec[i] = half(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
            vec[i] = float(rand() % 5 + 1);
        }
    }
}

// int loadCheckpoint(
//     string file_path,
//     int M, int K, 
//     int& total_TCBlock_num,
//     int& nnz_pad,
//     vector<int>& val_sidx,
//     vector<int>& val_soff,
//     vector<int>& val_coo_idx,
//     vector<float>& val_coo_val,
//     vector<int>& val_coo_off
// ) {

// }

// int storeCheckpoint(
//     string file_path,
//     int M, int K, 
//     int& total_TCBlock_num,
//     int& nnz_pad,
//     vector<int>& val_sidx,
//     vector<int>& val_soff,
//     vector<int>& val_coo_idx,
//     vector<float>& val_coo_val,
//     vector<int>& val_coo_off
// ) {

// }

std::string extractMatrixName(const std::string& file_path) {
    // 使用 C++17 filesystem 处理路径
    std::filesystem::path p(file_path);
    std::string filename = p.stem().string(); // 去掉扩展名 .mtx
    return filename;
}

std::filesystem::path getCheckpointCacheDir() {
    if (const char* env_dir = std::getenv("SPARSENE_MTX_CACHE_DIR")) {
        if (env_dir[0] != '\0') {
            return std::filesystem::path(env_dir);
        }
    }
    return std::filesystem::path(__FILE__).parent_path() / "mtx_cache";
}

bool useCheckpointCache() {
    if (const char* env = std::getenv("SPARSENE_DISABLE_MTX_CACHE")) {
        if (env[0] != '\0' && env[0] != '0') {
            return false;
        }
    }
    return true;
}

std::string formatSparsityToken(float sparsity) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << sparsity;
    std::string token = oss.str();
    while (!token.empty() && token.back() == '0') {
        token.pop_back();
    }
    if (!token.empty() && token.back() == '.') {
        token.pop_back();
    }
    std::replace(token.begin(), token.end(), '.', 'p');
    return token.empty() ? "unknown" : token;
}

std::string buildCheckpointStem(const std::string& file_path, int M, int K, float sparsity) {
    std::string base = extractMatrixName(file_path);
    if (base.empty()) {
        base = "synthetic";
    }
    std::string stem = base + "_M" + to_string(M) + "_K" + to_string(K);
    if (file_path.empty()) {
        stem += "_sp" + formatSparsityToken(sparsity);
    }
    return stem + "_METCF.checkpoint";
}

// -------------------------
// storeCheckpoint for ME-TCF
// -------------------------
int storeCheckpoint(
    string file_path,
    int M, int K,
    float sparsity,
    vector<int>& val_sidx,
    vector<int>& val_soff,
    vector<int>& val_coo_idx,
    vector<float>& val_coo_val,
    vector<int>& val_coo_off,
    int& total_TCBlock_num,
    int& nnz_pad
) {
    if (!useCheckpointCache()) {
        return 0;
    }

    string checkpoint_filename = buildCheckpointStem(file_path, M, K, sparsity);
    std::filesystem::path cache_dir = getCheckpointCacheDir();
    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);
    string full_path = (cache_dir / checkpoint_filename).string();

    std::ofstream outputFile(full_path, std::ios::binary);
    if (!outputFile.is_open()) {
        std::cerr << "Fail to open file: " << full_path << std::endl;
        return 0;
    }

    // 写入基础信息
    outputFile.write(reinterpret_cast<char*>(&total_TCBlock_num), sizeof(int));
    outputFile.write(reinterpret_cast<char*>(&nnz_pad), sizeof(int));

    // 写入数组数据
    outputFile.write(reinterpret_cast<char*>(val_soff.data()), sizeof(int) * val_soff.size());
    outputFile.write(reinterpret_cast<char*>(val_sidx.data()), sizeof(int) * val_sidx.size());
    outputFile.write(reinterpret_cast<char*>(val_coo_off.data()), sizeof(int) * val_coo_off.size());
    outputFile.write(reinterpret_cast<char*>(val_coo_idx.data()), sizeof(int) * val_coo_idx.size());
    outputFile.write(reinterpret_cast<char*>(val_coo_val.data()), sizeof(float) * val_coo_val.size());

    if (outputFile.fail()) {
        std::cerr << "Failed to write checkpoint to file." << std::endl;
        return 0;
    }

    std::cout << "Checkpoint written successfully to: " << full_path << std::endl;
    outputFile.close();
    return 1;
}

// -------------------------
// loadCheckpoint for ME-TCF
// -------------------------
int loadCheckpoint(
    string file_path,
    int M, int K,
    float sparsity,
    vector<int>& val_sidx,
    vector<int>& val_soff,
    vector<int>& val_coo_idx,
    vector<float>& val_coo_val,
    vector<int>& val_coo_off,
    int& total_TCBlock_num,
    int& nnz_pad
) {
    if (!useCheckpointCache()) {
        return 0;
    }

    string checkpoint_filename = buildCheckpointStem(file_path, M, K, sparsity);
    std::filesystem::path cache_dir = getCheckpointCacheDir();
    string full_path = (cache_dir / checkpoint_filename).string();

    std::ifstream inputFile(full_path, std::ios::binary);
    if (!inputFile.is_open()) {
        return 0; // file not exist
    }

    // 读取基础信息
    inputFile.read(reinterpret_cast<char*>(&total_TCBlock_num), sizeof(int));
    inputFile.read(reinterpret_cast<char*>(&nnz_pad), sizeof(int));

    int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;

    // resize vectors
    val_soff.resize(row_window_num + 1);
    val_sidx.resize(total_TCBlock_num * TC_BLOCK_K);
    val_coo_off.resize(2 * total_TCBlock_num + 1);
    val_coo_idx.resize(nnz_pad);
    val_coo_val.resize(nnz_pad);

    // 读取数组
    inputFile.read(reinterpret_cast<char*>(val_soff.data()), sizeof(int) * val_soff.size());
    inputFile.read(reinterpret_cast<char*>(val_sidx.data()), sizeof(int) * val_sidx.size());
    inputFile.read(reinterpret_cast<char*>(val_coo_off.data()), sizeof(int) * val_coo_off.size());
    inputFile.read(reinterpret_cast<char*>(val_coo_idx.data()), sizeof(int) * val_coo_idx.size());
    inputFile.read(reinterpret_cast<char*>(val_coo_val.data()), sizeof(float) * val_coo_val.size());

    if (inputFile.fail()) {
        std::cerr << "Failed to read checkpoint from file." << std::endl;
        return 0;
    }

    std::cout << "Checkpoint loaded successfully from: " << full_path << std::endl;
    inputFile.close();
    return 1;
}
void convertToMETCF(
    int M,
    int N,
    int K,
    int nnz,
    int* csrPtrA,
    int* csrColIdxA,
    float* csrValA,
    float* B,
    float* C,
    bool is_ncu_test,
    int warmup_time,
    int execute_time
);

void convertToMETCF(
    int M,
    int N,
    int K,
    int nnz,
    int* csrPtrA,
    int* csrColIdxA,
    float* csrValA,
    float* B,
    float* C,
    bool is_ncu_test,
    int warmup_time,
    int execute_time,
    string file_path,
    float sparsity
) {
    using dtypeB = float;
    using dtypeC = float;

    if (!useCheckpointCache()) {
        convertToMETCF(
            M, N, K, nnz, csrPtrA, csrColIdxA, csrValA, B, C, is_ncu_test, warmup_time,
            execute_time);
        return;
    }

    dtypeB* B_val = B;
    dtypeC* C_val = C;

    // const int ROW_WINDOW_M = 16;  // 根据原来的定义
    // const int TC_BLOCK_K = 8;

    // -------------------------
    // 尝试加载 checkpoint
    // -------------------------
    vector<int> val_soff;
    vector<int> val_sidx;
    vector<int> val_coo_off;
    vector<int> val_coo_idx;
    vector<float> val_coo_val;
    int total_TCBlock_num = 0;
    int nnz_pad = 0;

    int loaded = loadCheckpoint(file_path, M, K, sparsity, val_sidx, val_soff, val_coo_idx, val_coo_val, val_coo_off, total_TCBlock_num, nnz_pad);
    // int loaded = 0;
    if (!loaded) {
        // -------------------------
        // 没有 checkpoint，需要重新计算
        // -------------------------
        int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;
        vector<vector<int>> vec_nnz_per_tc_block(row_window_num);
        int* nnz_col = new int[K];
        memset(nnz_col, 0, sizeof(int) * K);

        total_TCBlock_num = 0;
        nnz_pad = 0;

        // 1. 计算 total_TCBlock_num 和 nnz_pad
        for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
            val_soff.push_back(total_TCBlock_num);
            memset(nnz_col, 0, sizeof(int) * K);
            vector<int>& nnz_per_tc_block = vec_nnz_per_tc_block[i_row_window];

            for (int i_row = 0; i_row < ROW_WINDOW_M; i_row++) {
                int row = i_row_window * ROW_WINDOW_M + i_row;
                if (row >= M) continue;
                for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                    nnz_col[csrColIdxA[j]] = 1;
                }
            }

            int nnz_col_num = 0;
            for (int i = 0, j = 0; i < K; i++) {
                if (nnz_col[i] == 1) {
                    nnz_col[j++] = i;
                    nnz_col_num++;
                }
            }

            int TCBlock_num = (nnz_col_num + TC_BLOCK_K - 1) / TC_BLOCK_K;
            nnz_per_tc_block.resize(TCBlock_num, 0);

            for (int i_row = 0; i_row < ROW_WINDOW_M; i_row++) {
                int row = i_row_window * ROW_WINDOW_M + i_row;
                if (row >= M) continue;
                for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                    int col = csrColIdxA[j];
                    int condensed_col = binarySearch(nnz_col, 0, nnz_col_num - 1, col);
                    int tc_block_id = condensed_col / TC_BLOCK_K;
                    nnz_per_tc_block[tc_block_id]++;
                }
            }

            for (int i_tc_block = 0; i_tc_block < TCBlock_num; i_tc_block++) {
                int nnz_block = nnz_per_tc_block[i_tc_block];
                nnz_pad += (nnz_block + 8 - 1) / 8 * 8;
            }

            total_TCBlock_num += TCBlock_num;
        }
        val_soff.push_back(total_TCBlock_num);
        delete[] nnz_col;

        // 2. 分配数组
        val_sidx.resize(total_TCBlock_num * TC_BLOCK_K, 0);
        val_coo_off.resize(total_TCBlock_num * 2 + 1, 0);
        val_coo_idx.resize(nnz_pad, 0);
        val_coo_val.resize(nnz_pad, 0);

        // 3. prefix sum 与 coo_off 填充
        int cur_pos = 0;
        int i_tc_block_ptr = 0;
        for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
            vector<int>& nnz_per_tc_block = vec_nnz_per_tc_block[i_row_window];
            for (int i_tc_block = 0; i_tc_block < nnz_per_tc_block.size(); i_tc_block++) {
                int tmp = nnz_per_tc_block[i_tc_block];
                nnz_per_tc_block[i_tc_block] = cur_pos;
                val_coo_off[i_tc_block_ptr * 2] = cur_pos;
                val_coo_off[i_tc_block_ptr * 2 + 1] = cur_pos + tmp;
                i_tc_block_ptr++;
                cur_pos += (tmp + 8 - 1) / 8 * 8;
            }
            nnz_per_tc_block.push_back(cur_pos);
        }
        val_coo_off[2 * total_TCBlock_num] = cur_pos;

        // 4. 构建 val_sidx, val_coo_idx, val_coo_val
        auto swizzle_layout =
            composition(Swizzle<1, 2, 3>{},
                        make_layout(make_shape(ROW_WINDOW_M, TC_BLOCK_K), make_stride(TC_BLOCK_K, 1)));
        int* tc_block_nnz_ptr = new int[(K + TC_BLOCK_K - 1) / TC_BLOCK_K];

        for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
            int TC_block_offset = val_soff[i_row_window];
            int* tmp_val_sidx = val_sidx.data() + TC_block_offset * TC_BLOCK_K;
            memset(tc_block_nnz_ptr, 0, sizeof(int) * (K + TC_BLOCK_K - 1) / TC_BLOCK_K);

            nnz_col = new int[K];
            memset(nnz_col, 0, sizeof(int) * K);
            vector<int>& nnz_per_tc_block = vec_nnz_per_tc_block[i_row_window];

            for (int i_row = 0; i_row < ROW_WINDOW_M; i_row++) {
                int row = i_row_window * ROW_WINDOW_M + i_row;
                if (row >= M) continue;
                for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                    nnz_col[csrColIdxA[j]] = 1;
                }
            }

            int nnz_col_num = 0;
            for (int i = 0, j = 0; i < K; i++) {
                if (nnz_col[i] == 1) {
                    nnz_col[j++] = i;
                    nnz_col_num++;
                }
            }

            int tc_block_num = (nnz_col_num + TC_BLOCK_K - 1) / TC_BLOCK_K;
            for (int i_tc_block_col = 0; i_tc_block_col < nnz_col_num; i_tc_block_col++) {
                tmp_val_sidx[i_tc_block_col] = nnz_col[i_tc_block_col];
            }

            for (int i_tc_block = 0; i_tc_block < tc_block_num; i_tc_block++) {
                tc_block_nnz_ptr[i_tc_block] = nnz_per_tc_block[i_tc_block];
            }

            for (int i_row = 0; i_row < ROW_WINDOW_M; i_row++) {
                int row = i_row_window * ROW_WINDOW_M + i_row;
                int col_start_pos = 0;
                if (row >= M) continue;
                for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                    int col = csrColIdxA[j];
                    float value = (float)csrValA[j];
                    int condensed_pos = binarySearch(nnz_col, col_start_pos, nnz_col_num - 1, col);
                    col_start_pos = condensed_pos + 1;
                    int tc_block_id = condensed_pos / TC_BLOCK_K;
                    val_coo_val[tc_block_nnz_ptr[tc_block_id]] = value;
                    auto coord = make_coord(i_row, condensed_pos % TC_BLOCK_K);
                    int swizzled_idx = swizzle_layout(coord);
                    val_coo_idx[tc_block_nnz_ptr[tc_block_id]] = swizzled_idx;
                    tc_block_nnz_ptr[tc_block_id]++;
                }
            }
            delete[] nnz_col;
        }

        delete[] tc_block_nnz_ptr;

        // 5. 保存 checkpoint
        storeCheckpoint(file_path, M, K, sparsity, val_sidx, val_soff, val_coo_idx, val_coo_val, val_coo_off, total_TCBlock_num, nnz_pad);
    }

    // 6. 调用 kernel
    int requested_tile_b = sparsene_generated_tile_b();
    auto launch_with_tile_b = [&](int tile_b) {
        switch (tile_b) {
            case 16:
                dtc_spmm_test_val_sidx_bind_reorder<16>(
                    M, N, K, total_TCBlock_num, nnz_pad, val_sidx.data(), val_soff.data(),
                    B_val, val_coo_idx.data(), val_coo_val.data(), val_coo_off.data(),
                    C_val, is_ncu_test, warmup_time, execute_time);
                break;
            case 32:
                dtc_spmm_test_val_sidx_bind_reorder<32>(
                    M, N, K, total_TCBlock_num, nnz_pad, val_sidx.data(), val_soff.data(),
                    B_val, val_coo_idx.data(), val_coo_val.data(), val_coo_off.data(),
                    C_val, is_ncu_test, warmup_time, execute_time);
                break;
            default:
                dtc_spmm_test_val_sidx_bind_reorder<64>(
                    M, N, K, total_TCBlock_num, nnz_pad, val_sidx.data(), val_soff.data(),
                    B_val, val_coo_idx.data(), val_coo_val.data(), val_coo_off.data(),
                    C_val, is_ncu_test, warmup_time, execute_time);
                break;
        }
    };
    if (requested_tile_b > 0 && N % requested_tile_b == 0) {
        launch_with_tile_b(requested_tile_b);
    } else if (N % 64 == 0) {
        launch_with_tile_b(64);
    } else if (N % 32 == 0) {
        launch_with_tile_b(32);
    } else if (N % 16 == 0) {
        launch_with_tile_b(16);
    } else {
        throw std::runtime_error("Generated TILE_B does not divide N and no fallback tile is valid.");
    }
}



void convertToMETCF(
    int M,
    int N,
    int K,
    int nnz,
    int* csrPtrA,
    int* csrColIdxA,
    float* csrValA,
    float* B,
    float* C,
    bool is_ncu_test,
    int warmup_time,
    int execute_time
) {
    // convert to ME-TCF
    int row_window_num = (M + ROW_WINDOW_M - 1) / ROW_WINDOW_M;
    int* val_soff = (int*)malloc(sizeof(int) * (row_window_num + 1));
    memset(val_soff, 0, sizeof(int) * (row_window_num + 1));
    vector<vector<int>> vec_nnz_per_tc_block(row_window_num);
    vector<int> tc_block_count(row_window_num, 0);
    vector<int> nnz_pad_per_window(row_window_num, 0);

#pragma omp parallel
    {
        vector<unsigned char> col_mark(K, 0);
        vector<int> unique_cols;
        unique_cols.reserve(ROW_WINDOW_M * TC_BLOCK_K);

#pragma omp for schedule(static)
        for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
            unique_cols.clear();
            for (int i_row = 0; i_row < ROW_WINDOW_M; i_row++) {
                int row = i_row_window * ROW_WINDOW_M + i_row;
                if (row >= M)
                    continue;
                for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                    int col = csrColIdxA[j];
                    if (!col_mark[col]) {
                        col_mark[col] = 1;
                        unique_cols.push_back(col);
                    }
                }
            }

            sort(unique_cols.begin(), unique_cols.end());
            int nnz_col_num = (int)unique_cols.size();
            int TCBlock_num = (nnz_col_num + TC_BLOCK_K - 1) / TC_BLOCK_K;
            tc_block_count[i_row_window] = TCBlock_num;

            vector<int>& nnz_per_tc_block = vec_nnz_per_tc_block[i_row_window];
            nnz_per_tc_block.assign(TCBlock_num, 0);

            for (int i_row = 0; i_row < ROW_WINDOW_M; i_row++) {
                int row = i_row_window * ROW_WINDOW_M + i_row;
                int col_start_pos = 0;
                if (row >= M)
                    continue;
                for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                    int col = csrColIdxA[j];
                    int condensed_col =
                      (int)(lower_bound(unique_cols.begin() + col_start_pos, unique_cols.end(), col) -
                            unique_cols.begin());
                    col_start_pos = condensed_col + 1;
                    int tc_block_id = condensed_col / TC_BLOCK_K;
                    nnz_per_tc_block[tc_block_id]++;
                }
            }

            int pad_cnt = 0;
            for (int i_tc_block = 0; i_tc_block < TCBlock_num; i_tc_block++) {
                int nnz_block = nnz_per_tc_block[i_tc_block];
                pad_cnt += (nnz_block + 8 - 1) / 8 * 8;
            }
            nnz_pad_per_window[i_row_window] = pad_cnt;

            for (int col : unique_cols) {
                col_mark[col] = 0;
            }
        }
    }

    int total_TCBlock_num = 0;
    int nnz_pad = 0;
    for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
        val_soff[i_row_window] = total_TCBlock_num;
        total_TCBlock_num += tc_block_count[i_row_window];
        nnz_pad += nnz_pad_per_window[i_row_window];
    }
    val_soff[row_window_num] = total_TCBlock_num;
    printf("total_TCBlock_num = %d, nnz = %d, nnz_pad=%d\n", total_TCBlock_num, nnz, nnz_pad);

    //> val_sidx -> 每个 TC Block 中非零列对应的column index
    int* val_sidx = (int*)malloc(sizeof(int) * total_TCBlock_num * TC_BLOCK_K);
    memset(val_sidx, 0, sizeof(int) * total_TCBlock_num * TC_BLOCK_K);
    //> val_coo_off -> TC Block 中非零元个数，每个block有两个范围（起始index，结束index），TC
    // Block可能pad0
    int* val_coo_off = (int*)malloc(sizeof(int) * (total_TCBlock_num * 2 + 1));
    memset(val_coo_off, 0, sizeof(int) * (total_TCBlock_num * 2 + 1));

    //> val_coo_sidx
    int* val_coo_idx = (int*)malloc(sizeof(int) * nnz_pad);
    memset(val_coo_idx, 0, sizeof(int) * nnz_pad);
    float* val_coo_val = (float*)malloc(sizeof(float) * nnz_pad);
    memset(val_coo_val, 0, sizeof(float) * nnz_pad);

    //> prefix sum
    int cur_pos = 0;
    int i_tc_block_ptr = 0;
    for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
        vector<int>& nnz_per_tc_block = vec_nnz_per_tc_block[i_row_window];
        for (int i_tc_block = 0; i_tc_block < nnz_per_tc_block.size(); i_tc_block++) {
            int tmp = nnz_per_tc_block[i_tc_block];
            nnz_per_tc_block[i_tc_block] = cur_pos;
            val_coo_off[i_tc_block_ptr * 2] = cur_pos;
            val_coo_off[i_tc_block_ptr * 2 + 1] = cur_pos + tmp;
            i_tc_block_ptr++;
            cur_pos += (tmp + 8 - 1) / 8 * 8;
        }
        nnz_per_tc_block.push_back(cur_pos);
    }
    val_coo_off[2 * total_TCBlock_num] = cur_pos;

    auto swizzle_layout =
      composition(Swizzle<1, 2, 3>{},
                  make_layout(make_shape(ROW_WINDOW_M, TC_BLOCK_K), make_stride(TC_BLOCK_K, 1)));

#pragma omp parallel
    {
        vector<unsigned char> col_mark(K, 0);
        vector<int> unique_cols;
        unique_cols.reserve(ROW_WINDOW_M * TC_BLOCK_K);

#pragma omp for schedule(static)
        for (int i_row_window = 0; i_row_window < row_window_num; i_row_window++) {
            int TC_block_offset = val_soff[i_row_window];
            int* tmp_val_sidx = val_sidx + TC_block_offset * TC_BLOCK_K;
            int tc_block_num = tc_block_count[i_row_window];

            unique_cols.clear();
            for (int i_row = 0; i_row < ROW_WINDOW_M; i_row++) {
                int row = i_row_window * ROW_WINDOW_M + i_row;
                if (row >= M)
                    continue;
                for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                    int col = csrColIdxA[j];
                    if (!col_mark[col]) {
                        col_mark[col] = 1;
                        unique_cols.push_back(col);
                    }
                }
            }

            sort(unique_cols.begin(), unique_cols.end());
            int nnz_col_num = (int)unique_cols.size();
            for (int i_tc_block_col = 0; i_tc_block_col < nnz_col_num; i_tc_block_col++) {
                tmp_val_sidx[i_tc_block_col] = unique_cols[i_tc_block_col];
            }

            vector<int>& nnz_per_tc_block = vec_nnz_per_tc_block[i_row_window];
            vector<int> tc_block_nnz_ptr(tc_block_num, 0);
            for (int i_tc_block = 0; i_tc_block < tc_block_num; i_tc_block++) {
                tc_block_nnz_ptr[i_tc_block] = nnz_per_tc_block[i_tc_block];
            }

            for (int i_row = 0; i_row < ROW_WINDOW_M; i_row++) {
                int row = i_row_window * ROW_WINDOW_M + i_row;
                int col_start_pos = 0;
                if (row >= M)
                    continue;
                for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                    int col = csrColIdxA[j];
                    float value = (float)csrValA[j];
                    int condensed_pos =
                      (int)(lower_bound(unique_cols.begin() + col_start_pos, unique_cols.end(), col) -
                            unique_cols.begin());
                    col_start_pos = condensed_pos + 1;
                    int tc_block_id = condensed_pos / TC_BLOCK_K;
                    int write_pos = tc_block_nnz_ptr[tc_block_id]++;
                    val_coo_val[write_pos] = value;

                    auto coord = make_coord(i_row, condensed_pos % TC_BLOCK_K);
                    auto swizzled_idx = swizzle_layout(coord);
                    val_coo_idx[write_pos] = swizzled_idx;
                }
            }

            for (int col : unique_cols) {
                col_mark[col] = 0;
            }
        }
    }

    { //! check sidx
      // printf("check sparse AtoB array: ");
      // for (int i = 0; i < total_TCBlock_num * TC_BLOCK_K; i++) {
      //     printf("%d ", val_sidx[i]);
      // } printf("\n");
    }

    // TODO: need to change according to swizzle
    //! reorder

    dtc_spmm_test_val_sidx_bind_reorder<64>(M, N, K, total_TCBlock_num, nnz_pad, val_sidx, val_soff,
                                            B, val_coo_idx, val_coo_val, val_coo_off, C,
                                            is_ncu_test, warmup_time, execute_time);
}

void verify_new(float* C_cpu, float* C_cuda, int M, int N) {
    int flag = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float cusparse_val = C_cpu[OFFSET_ROW(i, j, N)];
            float mykernel_val = C_cuda[OFFSET_ROW(i, j, N)];
            if (fabs(cusparse_val - mykernel_val) > 0.01) {
                flag++;
                if (flag < 200)
                printf("Error(%d, %d): cusp(%.1f) mykernel(%.1f)\n", i, j, cusparse_val,
                       mykernel_val);
            }
        }
    }
}

double cusparse_spmm_all(float* cu_ValA,
                         int* cu_RowPtrA,
                         int* cu_ColIdxA,
                         float* cu_B,
                         float* cu_C,
                         int M,
                         int N,
                         int K,
                         int nnz,
                         int warmup_time,
                         int execute_time) {
    int A_num_rows = M;
    int A_num_cols = K;
    int B_num_rows = K;
    int B_num_cols = N;
    int ldb = B_num_cols; // B: row first
    int B_size = B_num_rows * ldb;
    int C_num_rows = M;
    int C_num_cols = N;
    int ldc = C_num_cols; // C: row first
    int C_size = C_num_rows * ldc;
    float alpha = 1.0f;
    float beta = 0.0f;
    // device memory management
    int* dA_rpt;
    int* dA_cid;
    float* dA_val;
    float* dB;
    float* dC;
    CHECK_CUDA(cudaMalloc(&dA_rpt, sizeof(int) * (M + 1)));
    CHECK_CUDA(cudaMalloc(&dA_cid, sizeof(int) * nnz));
    CHECK_CUDA(cudaMalloc(&dA_val, sizeof(float) * nnz));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * B_size));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * C_size));

    CHECK_CUDA(cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_val, cu_ValA, sizeof(float) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, cu_B, sizeof(float) * B_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * C_size));

    // cuSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, nnz, dA_rpt, dA_cid, dA_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(
      cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW))
    CHECK_CUSPARSE(
      cusparseCreateDnMat(&matC, C_num_rows, C_num_cols, ldc, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
      matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute preprocess (optional)
    CHECK_CUSPARSE(cusparseSpMM_preprocess(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
      matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    if (execute_time > 0) {
        CHECK_CUDA(cudaEventCreate(&start_event));
        CHECK_CUDA(cudaEventCreate(&stop_event));
        for (int i = 0; i < warmup_time; ++i) {
            CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                                        &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                        dBuffer))
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start_event));
        for (int i = 0; i < execute_time; ++i) {
            CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                                        &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                        dBuffer))
        }
        CHECK_CUDA(cudaEventRecord(stop_event));
        CHECK_CUDA(cudaEventSynchronize(stop_event));
    }

    // execute SpMM for the final host verification result
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * C_size));
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
                                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))

    CHECK_CUDA(cudaMemcpy(cu_C, dC, sizeof(float) * C_size, cudaMemcpyDeviceToHost));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(dA_cid))
    CHECK_CUDA(cudaFree(dA_rpt))
    CHECK_CUDA(cudaFree(dA_val))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(dC))
    if (execute_time > 0) {
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        CHECK_CUDA(cudaEventDestroy(start_event));
        CHECK_CUDA(cudaEventDestroy(stop_event));
        double cusparse_time = static_cast<double>(elapsed_ms) / execute_time;
        printf("cusparse_time: %f ms\n", cusparse_time);
        return cusparse_time;
    }
    return 0.0;
}

int main(int argc, char** argv) {
    string filename;
    int N = 1024;
    int M = 1024;
    int K = 1024;
    float sparsity = 0.9;
    int mtx_flag = 1;
    int is_ncu_test = 0;
    int warmup_time = 100;
    int execute_time = 1000;
    parseInput(argc, argv, filename, "-filename", N, "-N", M, "-M", K, "-K", mtx_flag, "-mtx_flag",
               sparsity, "-sparsity", is_ncu_test, "-ncu", warmup_time, "-warmup", execute_time,
               "-repeat");

    // Validation scripts often run without a matrix path. In that case,
    // fall back to the synthetic generator instead of dereferencing an
    // empty filename through the MatrixMarket loader.
    if (filename.empty()) {
        mtx_flag = 0;
    }

    cout << "Filename: " << filename << endl;
    cout << "N: " << N << endl;

    int rowA, colA;
    int nnz = 0;
    int isSymmetricA;
    int* csrPtrA;
    int* csrColIdxA;
    float* csrValA;
    float* A;
    //! input from file
    if (mtx_flag == 1) {
        mmio_allinone(&rowA, &colA, &nnz, &isSymmetricA, &csrPtrA, &csrColIdxA, &csrValA,
                      filename.c_str());
        M = rowA;
        K = colA;
    } else { //! sparse matrix init
        A = (float*)malloc(sizeof(float) * M * K);
        initVector(A, M * K, sparsity);
        //> convert to csr
        csrPtrA = (int*)malloc(sizeof(int) * (M + 1));
        for (int i = 0; i < M; i++) {
            csrPtrA[i] = nnz;
            for (int k = 0; k < K; k++) {
                float value = (float)A[i * K + k];
                if (value != 0) {
                    nnz++;
                }
            }
        }
        csrPtrA[M] = nnz;
        csrColIdxA = (int*)malloc(sizeof(int) * nnz);
        csrValA = (float*)malloc(sizeof(float) * nnz);
        memset(csrColIdxA, 0, sizeof(int) * nnz);
        memset(csrValA, 0, sizeof(float) * nnz);
        int ptr = 0;
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float value = A[i * K + k];
                if (value != 0) {
                    csrColIdxA[ptr] = k;
                    csrValA[ptr] = value;
                    ptr++;
                }
            }
        }
    }

    printf("FLOAT: Sparse A[%d,%d] x B[%d,%d] = C[%d,%d]\n", M, K, K, N, M, N);

    float* B = (float*)malloc(sizeof(float) * K * N);
    // initVec(B, K * N);
    initVec(csrValA, nnz);
    initVector(B, K * N, 0);

    float* C_cuda = (float*)malloc(sizeof(float) * M * N);
    convertToMETCF(M, N, K, nnz, csrPtrA, csrColIdxA, csrValA, B, C_cuda, (bool)is_ncu_test,
                   warmup_time, execute_time, filename, sparsity);

    // host check
    float* C_cpu = (float*)malloc(sizeof(float) * M * N);
    memset(C_cpu, 0, sizeof(float) * M * N);
    cusparse_spmm_all(csrValA, csrPtrA, csrColIdxA, B, C_cpu, M, N, K, nnz, warmup_time,
                      execute_time);
    verify_new(C_cpu, C_cuda, M, N);

    printf("check nan: cusp %.1f mykernel %.1f\n", C_cpu[0], C_cuda[0]);
    return 0;
}
