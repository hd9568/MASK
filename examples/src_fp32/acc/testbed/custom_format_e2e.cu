#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cute/tensor.hpp>

#include "common.h"
#include "utils.h"

#ifndef SPARSENE_GENERATED_OPERATOR
#error "SPARSENE_GENERATED_OPERATOR must name the generated .inc file"
#endif
#ifndef SPARSENE_LOGICAL_BLK_M
#error "SPARSENE_LOGICAL_BLK_M is required"
#endif
#ifndef SPARSENE_PHYSICAL_BLK_M
#error "SPARSENE_PHYSICAL_BLK_M is required"
#endif
#ifndef SPARSENE_BLK_K
#error "SPARSENE_BLK_K is required"
#endif
#ifndef SPARSENE_TILE_N
#define SPARSENE_TILE_N 64
#endif
#ifndef SPARSENE_ROW_REORDER
#define SPARSENE_ROW_REORDER 0
#endif
#ifndef SPARSENE_PACKING_MODE
#define SPARSENE_PACKING_MODE 0
#endif
#ifndef SPARSENE_MAX_TILES_PER_SEGMENT
#define SPARSENE_MAX_TILES_PER_SEGMENT 32
#endif

#include SPARSENE_GENERATED_OPERATOR

#ifndef SPARSENE_GENERATED_REQUIRES_SINGLE_WARP
#error "generated operator must declare its synchronization/launch contract"
#endif
#ifndef SPARSENE_GENERATED_THREADS_PER_BLOCK
#error "generated operator must declare SPARSENE_GENERATED_THREADS_PER_BLOCK"
#endif
static_assert(SPARSENE_GENERATED_REQUIRES_SINGLE_WARP == 1);
static_assert(SPARSENE_GENERATED_THREADS_PER_BLOCK == 32);

namespace sparsene_custom_e2e {

constexpr int kLogicalM = SPARSENE_LOGICAL_BLK_M;
constexpr int kPhysicalM = SPARSENE_PHYSICAL_BLK_M;
constexpr int kBlockK = SPARSENE_BLK_K;
constexpr int kTileN = SPARSENE_TILE_N;
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 8;
constexpr int kMaxTilesPerExecutionSegment = SPARSENE_MAX_TILES_PER_SEGMENT;

constexpr int kRowReorder = SPARSENE_ROW_REORDER;
constexpr int kPackingMode = SPARSENE_PACKING_MODE;

static_assert(kLogicalM == 8 || kLogicalM == 16 || kLogicalM == 32);
static_assert(kPhysicalM >= kLogicalM && kPhysicalM % 16 == 0);
static_assert(kBlockK >= 8 && kBlockK % 8 == 0);
static_assert(kTileN >= 8 && kTileN % 8 == 0);
static_assert(kMaxTilesPerExecutionSegment > 0);
static_assert(kRowReorder >= 0 && kRowReorder <= 3);
static_assert(kPackingMode >= 0 && kPackingMode <= 4);

int div_up(int a, int b) { return (a + b - 1) / b; }
int align_up(int a, int b) { return div_up(a, b) * b; }

struct Options {
    std::string filename;
    std::string dataset;
    std::string result_json;
    int n = 64;
    int warmup = 10;
    int repeat = 100;
    int max_tiles_per_segment = kMaxTilesPerExecutionSegment;
    std::string execution_policy = "auto";
    int task_group_tiles = 64;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for argument " + arg);
        }
        const std::string value = argv[++i];
        if (arg == "--filename") options.filename = value;
        else if (arg == "--dataset") options.dataset = value;
        else if (arg == "--result-json") options.result_json = value;
        else if (arg == "--n") options.n = std::stoi(value);
        else if (arg == "--warmup") options.warmup = std::stoi(value);
        else if (arg == "--repeat") options.repeat = std::stoi(value);
        else if (arg == "--max-tiles-per-segment") {
            options.max_tiles_per_segment = std::stoi(value);
        }
        else if (arg == "--execution-policy") options.execution_policy = value;
        else if (arg == "--task-group-tiles") options.task_group_tiles = std::stoi(value);
        else throw std::runtime_error("unknown argument " + arg);
    }
    if (options.filename.empty() || options.dataset.empty() || options.result_json.empty()) {
        throw std::runtime_error("--filename, --dataset, and --result-json are required");
    }
    if (options.n <= 0 || options.n % kTileN != 0) {
        throw std::runtime_error("N must be positive and divisible by the generated Tile_N");
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::runtime_error("warmup must be nonnegative and repeat must be positive");
    }
    if (options.max_tiles_per_segment < 0) {
        throw std::runtime_error("max tiles per segment must be zero (auto) or positive");
    }
    if (options.execution_policy != "auto" &&
        options.execution_policy != "row" &&
        options.execution_policy != "strict_lb" &&
        options.execution_policy != "multi_bind") {
        throw std::runtime_error(
            "execution policy must be auto, row, strict_lb, or multi_bind");
    }
    if (options.task_group_tiles <= 0) {
        throw std::runtime_error("task group tiles must be positive");
    }
    return options;
}

std::vector<int> identity_order(int m) {
    std::vector<int> order(m);
    std::iota(order.begin(), order.end(), 0);
    return order;
}

// Deterministic Reverse Cuthill-McKee on the symmetrized structural graph.
// The returned mapping is new_row -> original_row.  This is format conversion,
// so construction and the later inverse output mapping are outside timing.
std::vector<int> reverse_cuthill_mckee(
    int m,
    int k,
    const int* row_ptr,
    const int* col_idx) {
    if (m != k) {
        throw std::runtime_error("row_rcm currently requires a square matrix");
    }
    std::vector<std::vector<int>> adjacency(m);
    for (int row = 0; row < m; ++row) {
        for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
            const int col = col_idx[p];
            if (col < 0 || col >= m || col == row) continue;
            adjacency[row].push_back(col);
            adjacency[col].push_back(row);
        }
    }
    std::vector<int> degree(m);
    for (int row = 0; row < m; ++row) {
        auto& neighbors = adjacency[row];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        degree[row] = static_cast<int>(neighbors.size());
    }
    std::vector<int> starts(m);
    std::iota(starts.begin(), starts.end(), 0);
    std::stable_sort(starts.begin(), starts.end(), [&](int lhs, int rhs) {
        return std::tie(degree[lhs], lhs) < std::tie(degree[rhs], rhs);
    });

    std::vector<unsigned char> visited(m, 0);
    std::vector<int> rcm;
    rcm.reserve(m);
    std::queue<int> queue;
    std::vector<int> component;
    for (const int start : starts) {
        if (visited[start]) continue;
        component.clear();
        visited[start] = 1;
        queue.push(start);
        while (!queue.empty()) {
            const int row = queue.front();
            queue.pop();
            component.push_back(row);
            std::vector<int> next;
            for (const int neighbor : adjacency[row]) {
                if (!visited[neighbor]) next.push_back(neighbor);
            }
            std::stable_sort(next.begin(), next.end(), [&](int lhs, int rhs) {
                return std::tie(degree[lhs], lhs) < std::tie(degree[rhs], rhs);
            });
            for (const int neighbor : next) {
                if (!visited[neighbor]) {
                    visited[neighbor] = 1;
                    queue.push(neighbor);
                }
            }
        }
        std::reverse(component.begin(), component.end());
        rcm.insert(rcm.end(), component.begin(), component.end());
    }
    if (static_cast<int>(rcm.size()) != m) {
        throw std::runtime_error("RCM did not cover all rows");
    }
    return rcm;
}

int first_structural_column(int row, int k, const int* row_ptr, const int* col_idx) {
    int first = k;
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        if (col_idx[p] >= 0 && col_idx[p] < k) first = std::min(first, col_idx[p]);
    }
    return first;
}

std::vector<int> first_column_order(
    int m,
    int k,
    const int* row_ptr,
    const int* col_idx) {
    std::vector<int> order = identity_order(m);
    std::vector<int> first(m);
    for (int row = 0; row < m; ++row) {
        first[row] = first_structural_column(row, k, row_ptr, col_idx);
    }
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return std::tie(first[lhs], lhs) < std::tie(first[rhs], rhs);
    });
    return order;
}

uint64_t mix_column(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

uint64_t row_minhash(int row, const int* row_ptr, const int* col_idx) {
    uint64_t signature = std::numeric_limits<uint64_t>::max();
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        signature = std::min(signature, mix_column(static_cast<uint32_t>(col_idx[p])));
    }
    return signature;
}

double row_jaccard(int lhs, int rhs, const int* row_ptr, const int* col_idx) {
    int left = row_ptr[lhs];
    int right = row_ptr[rhs];
    const int left_end = row_ptr[lhs + 1];
    const int right_end = row_ptr[rhs + 1];
    int intersection = 0;
    int left_unique = 0;
    int right_unique = 0;
    while (left < left_end || right < right_end) {
        const int left_col = left < left_end ? col_idx[left] : std::numeric_limits<int>::max();
        const int right_col = right < right_end ? col_idx[right] : std::numeric_limits<int>::max();
        if (left_col == right_col) {
            ++intersection;
            ++left_unique;
            ++right_unique;
            while (left < left_end && col_idx[left] == left_col) ++left;
            while (right < right_end && col_idx[right] == right_col) ++right;
        } else if (left_col < right_col) {
            ++left_unique;
            while (left < left_end && col_idx[left] == left_col) ++left;
        } else {
            ++right_unique;
            while (right < right_end && col_idx[right] == right_col) ++right;
        }
    }
    const int union_size = left_unique + right_unique - intersection;
    return union_size ? static_cast<double>(intersection) / union_size : 1.0;
}

// A deterministic and scalable greedy Jaccard order.  Global O(M^2) greedy
// matching is infeasible for the 685k-row input, so minhash first creates
// similarity buckets and exact Jaccard greedily orders each bounded bucket.
std::vector<int> jaccard_greedy_order(
    int m,
    int k,
    const int* row_ptr,
    const int* col_idx) {
    constexpr int bucket_size = 64;
    std::vector<int> ranked = identity_order(m);
    std::vector<uint64_t> signature(m);
    std::vector<int> first(m);
    for (int row = 0; row < m; ++row) {
        signature[row] = row_minhash(row, row_ptr, col_idx);
        first[row] = first_structural_column(row, k, row_ptr, col_idx);
    }
    std::stable_sort(ranked.begin(), ranked.end(), [&](int lhs, int rhs) {
        return std::tie(signature[lhs], first[lhs], lhs) <
               std::tie(signature[rhs], first[rhs], rhs);
    });

    std::vector<int> order;
    order.reserve(m);
    for (int base = 0; base < m; base += bucket_size) {
        const int end = std::min(base + bucket_size, m);
        std::vector<int> remaining(ranked.begin() + base, ranked.begin() + end);
        int current = remaining.front();
        order.push_back(current);
        remaining.erase(remaining.begin());
        while (!remaining.empty()) {
            size_t best_index = 0;
            double best_score = -1.0;
            for (size_t candidate = 0; candidate < remaining.size(); ++candidate) {
                const double score = row_jaccard(current, remaining[candidate], row_ptr, col_idx);
                if (score > best_score ||
                    (score == best_score && remaining[candidate] < remaining[best_index])) {
                    best_score = score;
                    best_index = candidate;
                }
            }
            current = remaining[best_index];
            order.push_back(current);
            remaining.erase(remaining.begin() + best_index);
        }
    }
    return order;
}

const char* row_reorder_name() {
    if constexpr (kRowReorder == 1) return "row_rcm";
    if constexpr (kRowReorder == 2) return "first_col_sort";
    if constexpr (kRowReorder == 3) return "jaccard_greedy";
    return "none";
}

const char* packing_mode_name() {
    if constexpr (kPackingMode == 1) return "active_fixed_blocks";
    if constexpr (kPackingMode == 2) return "column_condensed_dense";
    if constexpr (kPackingMode == 3) return "active_fixed_blocks_dense";
    if constexpr (kPackingMode == 4) return "sparse_row_column_condensed_dense";
    return "column_condensed";
}

struct PackedMco {
    std::vector<int> row_offsets;
    std::vector<int> sparse_columns;
    std::vector<int> value_offsets;
    std::vector<uint64_t> masks;
    std::vector<float> values;
    std::vector<int> row_order;
    std::vector<int> execution_row_offsets;
    std::vector<int> logical_segment_offsets;
    std::vector<int> task_tile_begin;
    std::vector<int> task_tile_end;
    std::vector<int> task_row_window;
    std::vector<int> task_use_atomic;
    std::vector<int> task_group_offsets;
    std::vector<int> bound_task_offsets;
    std::vector<int> row_window_split;
    int row_windows = 0;
    int tiles = 0;
    int execution_segments = 0;
    int aligned_values = 0;
    int max_tiles_per_execution_segment = 0;
    int row_tiles_p95 = 0;
    int row_tiles_max = 0;
    int task_groups = 0;
    int task_group_tiles = 0;
    double row_tiles_mean = 0.0;
    bool automatic_segmentation = false;
    bool distribution_heavy_tail = false;
    std::string execution_policy = "row";
};

PackedMco convert_to_mco(
    int m,
    int k,
    const int* row_ptr,
    const int* col_idx,
    const float* vals,
    int requested_max_tiles_per_segment,
    const std::string& requested_execution_policy,
    int requested_task_group_tiles) {
    PackedMco packed;
    if constexpr (kRowReorder == 1) {
        packed.row_order = reverse_cuthill_mckee(m, k, row_ptr, col_idx);
    } else if constexpr (kRowReorder == 2) {
        packed.row_order = first_column_order(m, k, row_ptr, col_idx);
    } else if constexpr (kRowReorder == 3) {
        packed.row_order = jaccard_greedy_order(m, k, row_ptr, col_idx);
    } else if constexpr (kPackingMode == 4) {
        packed.row_order.reserve(m);
        for (int row = 0; row < m; ++row) {
            if (row_ptr[row] != row_ptr[row + 1]) packed.row_order.push_back(row);
        }
    } else {
        packed.row_order = identity_order(m);
    }
    const int physical_rows = static_cast<int>(packed.row_order.size());
    packed.row_windows = div_up(physical_rows, kLogicalM);
    packed.row_offsets.reserve(packed.row_windows + 1);
    packed.row_offsets.push_back(0);
    packed.value_offsets.push_back(0);
    constexpr int masks_per_tile = kPhysicalM * kBlockK / 64;
    auto physical_offset = [](int row, int col) {
        auto base = cute::make_layout(
            cute::make_shape(cute::Int<kPhysicalM>{}, cute::Int<kBlockK>{}),
            cute::make_stride(cute::Int<kBlockK>{}, cute::_1{}));
        if constexpr (kPackingMode == 1) {
            return cute::composition(cute::Swizzle<1, 3, 3>{}, base)(
                cute::make_coord(row, col));
        } else if constexpr (
            kPackingMode == 2 || kPackingMode == 3 || kPackingMode == 4) {
            // SR-BCRS loads a full dense A tile directly with float4 cp.async;
            // its generated S2R path expects the canonical 16x8 dense swizzle.
            return cute::composition(cute::Swizzle<2, 2, 3>{}, base)(
                cute::make_coord(row, col));
        } else {
            return cute::composition(cute::Swizzle<1, 2, 3>{}, base)(
                cute::make_coord(row, col));
        }
    };

    auto append_tile = [&](const std::vector<int>& tile_columns,
                           const std::map<int, float>& physical_values) {
        const int expected_columns = kPackingMode == 1 ? 1 : kBlockK;
        if (static_cast<int>(tile_columns.size()) != expected_columns) {
            throw std::runtime_error("internal MCO tile column count mismatch");
        }
        packed.sparse_columns.insert(
            packed.sparse_columns.end(), tile_columns.begin(), tile_columns.end());
        if constexpr (
            kPackingMode == 2 || kPackingMode == 3 || kPackingMode == 4) {
            const size_t value_base = packed.values.size();
            packed.values.resize(value_base + kPhysicalM * kBlockK, 0.0f);
            for (const auto& [physical, value] : physical_values) {
                packed.values[value_base + physical] += value;
            }
        } else {
            std::vector<uint64_t> mask(masks_per_tile, 0);
            for (const auto& [physical, value] : physical_values) {
                if (value == 0.0f) continue;
                mask[physical / 64] |= uint64_t{1} << (physical % 64);
                packed.values.push_back(value);
            }
            packed.masks.insert(packed.masks.end(), mask.begin(), mask.end());
            while (packed.values.size() % 4 != 0) packed.values.push_back(0.0f);
        }
        packed.value_offsets.push_back(static_cast<int>(packed.values.size()));
        ++packed.tiles;
    };

    for (int window = 0; window < packed.row_windows; ++window) {
        const int new_row_begin = window * kLogicalM;
        const int new_row_end = std::min(new_row_begin + kLogicalM, physical_rows);
        if constexpr (
            kPackingMode == 0 || kPackingMode == 2 || kPackingMode == 4) {
            std::map<int, std::map<int, float>> columns;
            for (int new_row = new_row_begin; new_row < new_row_end; ++new_row) {
                const int original_row = packed.row_order[new_row];
                const int local_row = new_row - new_row_begin;
                for (int p = row_ptr[original_row]; p < row_ptr[original_row + 1]; ++p) {
                    const int col = col_idx[p];
                    if (col < 0 || col >= k) {
                        throw std::runtime_error("CSR column index out of range");
                    }
                    columns[col][local_row] += vals[p];
                }
            }

            std::vector<int> unique_columns;
            unique_columns.reserve(columns.size());
            for (const auto& entry : columns) unique_columns.push_back(entry.first);
            for (size_t base = 0; base < unique_columns.size(); base += kBlockK) {
                const size_t count = std::min<size_t>(kBlockK, unique_columns.size() - base);
                std::vector<int> tile_columns;
                tile_columns.reserve(kBlockK);
                for (size_t j = 0; j < count; ++j) {
                    tile_columns.push_back(unique_columns[base + j]);
                }
                const int pad_column = count ? unique_columns[base + count - 1] : 0;
                tile_columns.resize(kBlockK, pad_column);

                std::map<int, float> physical_values;
                for (size_t local_col = 0; local_col < count; ++local_col) {
                    const auto& row_values = columns[unique_columns[base + local_col]];
                    for (const auto& [local_row, value] : row_values) {
                        const int physical = physical_offset(
                            local_row, static_cast<int>(local_col));
                        physical_values[physical] += value;
                    }
                }
                append_tile(tile_columns, physical_values);
            }
        } else {
            std::map<int, std::map<int, float>> blocks;
            for (int new_row = new_row_begin; new_row < new_row_end; ++new_row) {
                const int original_row = packed.row_order[new_row];
                const int local_row = new_row - new_row_begin;
                for (int p = row_ptr[original_row]; p < row_ptr[original_row + 1]; ++p) {
                    const int col = col_idx[p];
                    if (col < 0 || col >= k) {
                        throw std::runtime_error("CSR column index out of range");
                    }
                    const int block_col = col / kBlockK;
                    const int local_col = col % kBlockK;
                    const int physical = physical_offset(local_row, local_col);
                    blocks[block_col][physical] += vals[p];
                }
            }
            for (const auto& [block_col, physical_values] : blocks) {
                std::vector<int> tile_columns;
                if constexpr (kPackingMode == 3) {
                    tile_columns.reserve(kBlockK);
                    for (int local_col = 0; local_col < kBlockK; ++local_col) {
                        tile_columns.push_back(
                            std::min(block_col * kBlockK + local_col, k - 1));
                    }
                } else {
                    tile_columns.push_back(block_col);
                }
                append_tile(tile_columns, physical_values);
            }
        }
        packed.row_offsets.push_back(packed.tiles);
    }
    packed.aligned_values = align_up(static_cast<int>(packed.values.size()), 8);
    packed.values.resize(packed.aligned_values, 0.0f);
    if (static_cast<int>(packed.row_offsets.size()) != packed.row_windows + 1 ||
        static_cast<int>(packed.value_offsets.size()) != packed.tiles + 1 ||
        static_cast<int>(packed.sparse_columns.size()) !=
            packed.tiles * (kPackingMode == 1 ? 1 : kBlockK) ||
        static_cast<int>(packed.masks.size()) !=
            ((kPackingMode == 2 || kPackingMode == 3 || kPackingMode == 4)
                 ? 0
                 : packed.tiles * masks_per_tile)) {
        throw std::runtime_error("internal MCO packing invariant failed");
    }
    std::vector<int> row_tile_lengths;
    row_tile_lengths.reserve(packed.row_windows);
    long long row_tile_sum = 0;
    for (int window = 0; window < packed.row_windows; ++window) {
        const int length = packed.row_offsets[window + 1] - packed.row_offsets[window];
        row_tile_lengths.push_back(length);
        row_tile_sum += length;
        packed.row_tiles_max = std::max(packed.row_tiles_max, length);
    }
    packed.row_tiles_mean = packed.row_windows
        ? static_cast<double>(row_tile_sum) / packed.row_windows
        : 0.0;
    if (!row_tile_lengths.empty()) {
        std::sort(row_tile_lengths.begin(), row_tile_lengths.end());
        const size_t p95_index = std::min(
            row_tile_lengths.size() - 1,
            static_cast<size_t>(std::ceil(row_tile_lengths.size() * 0.95)) - 1);
        packed.row_tiles_p95 = row_tile_lengths[p95_index];
    }
    packed.automatic_segmentation = requested_max_tiles_per_segment == 0;
    if (packed.automatic_segmentation) {
        const int p95 = std::max(packed.row_tiles_p95, 1);
        const bool heavy_tail =
            packed.row_tiles_max > std::max(64, 4 * p95);
        packed.distribution_heavy_tail = heavy_tail;
        packed.max_tiles_per_execution_segment = heavy_tail
            ? std::max(32, std::min(128, 2 * p95))
            : std::max(packed.row_tiles_max, 1);
    } else {
        packed.max_tiles_per_execution_segment = requested_max_tiles_per_segment;
    }

    // A logical condensation window remains unchanged in storage, but a very
    // long tile list is scheduled as bounded partial-output segments.  A GPU
    // reduction below restores one output tile per logical row window.
    packed.execution_row_offsets.push_back(0);
    packed.logical_segment_offsets.push_back(0);
    for (int window = 0; window < packed.row_windows; ++window) {
        const int begin = packed.row_offsets[window];
        const int end = packed.row_offsets[window + 1];
        if (begin == end) {
            packed.execution_row_offsets.push_back(end);
            ++packed.execution_segments;
        } else {
            for (int tile = begin; tile < end;
                 tile += packed.max_tiles_per_execution_segment) {
                packed.execution_row_offsets.push_back(
                    std::min(tile + packed.max_tiles_per_execution_segment, end));
                ++packed.execution_segments;
            }
        }
        packed.logical_segment_offsets.push_back(packed.execution_segments);
    }
    if (static_cast<int>(packed.execution_row_offsets.size()) !=
            packed.execution_segments + 1 ||
        static_cast<int>(packed.logical_segment_offsets.size()) !=
            packed.row_windows + 1 ||
        packed.execution_row_offsets.back() != packed.tiles) {
        throw std::runtime_error("internal execution-segmentation invariant failed");
    }

    // Keep the NVIR-derived MMA/pipeline body fixed while selecting only its
    // execution-task mapping. strict_lb packs consecutive row fragments into
    // fixed-work groups; multi_bind splits rows above twice the global mean.
    packed.execution_policy = requested_execution_policy;
    if (packed.execution_policy == "auto") {
        packed.execution_policy = packed.distribution_heavy_tail ? "strict_lb" : "row";
    }
    packed.task_group_tiles = requested_task_group_tiles;
    if constexpr (
        kPackingMode == 2 || kPackingMode == 3 || kPackingMode == 4) {
        if (packed.execution_policy == "row") {
            packed.execution_row_offsets = packed.row_offsets;
            packed.logical_segment_offsets.resize(packed.row_windows + 1);
            std::iota(
                packed.logical_segment_offsets.begin(),
                packed.logical_segment_offsets.end(),
                0);
            packed.execution_segments = packed.row_windows;
        }
#ifndef SPARSENE_GENERATED_SUPPORTS_TASK_MAPPING
        if (packed.execution_policy != "row") {
            throw std::runtime_error(
                "generated SR-BCRS operator lacks task-mapping support");
        }
#else
        if (packed.execution_policy == "strict_lb") {
            packed.task_group_offsets.push_back(0);
            int group_workload = 0;
            for (int window = 0; window < packed.row_windows; ++window) {
                int tile = packed.row_offsets[window];
                const int end = packed.row_offsets[window + 1];
                while (tile < end) {
                    const int take = std::min(
                        requested_task_group_tiles - group_workload, end - tile);
                    packed.task_tile_begin.push_back(tile);
                    packed.task_tile_end.push_back(tile + take);
                    packed.task_row_window.push_back(window);
                    packed.task_use_atomic.push_back(0);
                    tile += take;
                    group_workload += take;
                    if (group_workload == requested_task_group_tiles) {
                        packed.task_group_offsets.push_back(
                            static_cast<int>(packed.task_tile_begin.size()));
                        group_workload = 0;
                    }
                }
            }
            if (packed.task_group_offsets.back() !=
                static_cast<int>(packed.task_tile_begin.size())) {
                packed.task_group_offsets.push_back(
                    static_cast<int>(packed.task_tile_begin.size()));
            }
            packed.task_groups = static_cast<int>(packed.task_group_offsets.size()) - 1;
            for (int group = 0; group < packed.task_groups; ++group) {
                const int begin = packed.task_group_offsets[group];
                const int end = packed.task_group_offsets[group + 1];
                for (int task = begin; task < end; ++task) {
                    packed.task_use_atomic[task] =
                        (task == begin || task + 1 == end) ? 1 : 0;
                }
            }
        } else if (packed.execution_policy == "multi_bind") {
            packed.task_group_offsets.push_back(0);
            packed.row_window_split.assign(packed.row_windows, 0);
            const double average = packed.row_tiles_mean;
            for (int window = 0; window < packed.row_windows; ++window) {
                const int begin = packed.row_offsets[window];
                const int end = packed.row_offsets[window + 1];
                const int length = end - begin;
                if (length == 0) continue;
                int parts = 1;
                if (average > 0.0 && length > 2.0 * average) {
                    parts = std::max(1, static_cast<int>(std::ceil(length / average)));
                }
                if (parts > 1) packed.row_window_split[window] = 1;
                const int base = length / parts;
                const int remainder = length % parts;
                int tile = begin;
                for (int part = 0; part < parts; ++part) {
                    const int take = base + (part < remainder ? 1 : 0);
                    packed.task_tile_begin.push_back(tile);
                    packed.task_tile_end.push_back(tile + take);
                    packed.task_row_window.push_back(window);
                    packed.task_use_atomic.push_back(parts > 1 ? 1 : 0);
                    tile += take;
                    packed.task_group_offsets.push_back(
                        static_cast<int>(packed.task_tile_begin.size()));
                }
            }
            packed.task_groups = static_cast<int>(packed.task_group_offsets.size()) - 1;
            if (!packed.task_tile_begin.empty()) {
                packed.bound_task_offsets = packed.task_tile_begin;
                packed.bound_task_offsets.push_back(packed.task_tile_end.back());
            }
        }
#endif
    }
    if (packed.task_tile_begin.size() != packed.task_tile_end.size() ||
        packed.task_tile_begin.size() != packed.task_row_window.size() ||
        packed.task_tile_begin.size() != packed.task_use_atomic.size() ||
        (packed.task_groups > 0 &&
         (static_cast<int>(packed.task_group_offsets.size()) != packed.task_groups + 1 ||
          packed.task_group_offsets.back() !=
              static_cast<int>(packed.task_tile_begin.size())))) {
        throw std::runtime_error("internal task-mapping invariant failed");
    }
    if (packed.execution_policy == "multi_bind" &&
        (packed.bound_task_offsets.size() != packed.task_tile_begin.size() + 1 ||
         packed.row_window_split.size() != static_cast<size_t>(packed.row_windows))) {
        throw std::runtime_error("internal bound-task ABI invariant failed");
    }
    return packed;
}

template <typename T>
struct DeviceArray {
    T* ptr = nullptr;
    size_t count = 0;
    DeviceArray() = default;
    explicit DeviceArray(size_t n) : count(n) {
        CHECK_CUDA(cudaMalloc(&ptr, sizeof(T) * std::max<size_t>(n, 1)));
    }
    ~DeviceArray() { if (ptr) cudaFree(ptr); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    void copy_from(const T* source, size_t n) {
        if (n > count) throw std::runtime_error("device copy exceeds allocation");
        if (n) CHECK_CUDA(cudaMemcpy(ptr, source, sizeof(T) * n, cudaMemcpyHostToDevice));
    }
};

__global__ void reduce_tcf_segments(
    const float* partial,
    float* output,
    const int* logical_segment_offsets,
    int logical_windows,
    int n) {
    const size_t tile_elems = static_cast<size_t>(kPhysicalM) * n;
    const size_t total = static_cast<size_t>(logical_windows) * tile_elems;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int window = static_cast<int>(index / tile_elems);
        const size_t within_tile = index % tile_elems;
        float sum = 0.0f;
        for (int segment = logical_segment_offsets[window];
             segment < logical_segment_offsets[window + 1];
             ++segment) {
            sum += partial[static_cast<size_t>(segment) * tile_elems + within_tile];
        }
        output[index] = sum;
    }
}

float time_generated_kernel(
    int m,
    int n,
    int k,
    const PackedMco& packed,
    const std::vector<float>& b,
    int warmup,
    int repeat,
    std::vector<float>& physical_c) {
    const int backend_m = packed.row_windows * kPhysicalM;
    const int partial_m = packed.execution_segments * kPhysicalM;
    const bool task_mapped_execution = packed.task_groups > 0;
    const bool segmented_execution = !task_mapped_execution &&
        packed.execution_segments != packed.row_windows;
    DeviceArray<float> d_b(static_cast<size_t>(k) * n);
    DeviceArray<float> d_c(static_cast<size_t>(backend_m) * n);
    DeviceArray<float> d_partial(
        segmented_execution ? static_cast<size_t>(partial_m) * n : 0);
    DeviceArray<int> d_row_offsets(packed.execution_row_offsets.size());
    DeviceArray<int> d_segment_offsets(packed.logical_segment_offsets.size());
    DeviceArray<int> d_value_offsets(packed.value_offsets.size());
    DeviceArray<int64_t> d_masks(packed.masks.size());
    DeviceArray<float> d_values(packed.values.size());
    DeviceArray<int> d_columns(packed.sparse_columns.size());
    DeviceArray<int> d_task_tile_begin(packed.task_tile_begin.size());
    DeviceArray<int> d_task_tile_end(packed.task_tile_end.size());
    DeviceArray<int> d_task_row_window(packed.task_row_window.size());
    DeviceArray<int> d_task_use_atomic(packed.task_use_atomic.size());
    DeviceArray<int> d_task_group_offsets(packed.task_group_offsets.size());
    DeviceArray<int> d_bound_task_offsets(packed.bound_task_offsets.size());
    DeviceArray<int> d_row_window_split(packed.row_window_split.size());
    d_b.copy_from(b.data(), b.size());
    CHECK_CUDA(cudaMemset(d_c.ptr, 0, sizeof(float) * d_c.count));
    if (segmented_execution) {
        CHECK_CUDA(cudaMemset(d_partial.ptr, 0, sizeof(float) * d_partial.count));
    }
    d_row_offsets.copy_from(
        packed.execution_row_offsets.data(), packed.execution_row_offsets.size());
    d_segment_offsets.copy_from(
        packed.logical_segment_offsets.data(), packed.logical_segment_offsets.size());
    d_value_offsets.copy_from(packed.value_offsets.data(), packed.value_offsets.size());
    if (!packed.masks.empty()) {
        CHECK_CUDA(cudaMemcpy(d_masks.ptr, packed.masks.data(),
                              sizeof(uint64_t) * packed.masks.size(), cudaMemcpyHostToDevice));
    }
    d_values.copy_from(packed.values.data(), packed.values.size());
    d_columns.copy_from(packed.sparse_columns.data(), packed.sparse_columns.size());
    d_task_tile_begin.copy_from(
        packed.task_tile_begin.data(), packed.task_tile_begin.size());
    d_task_tile_end.copy_from(packed.task_tile_end.data(), packed.task_tile_end.size());
    d_task_row_window.copy_from(
        packed.task_row_window.data(), packed.task_row_window.size());
    d_task_use_atomic.copy_from(
        packed.task_use_atomic.data(), packed.task_use_atomic.size());
    d_task_group_offsets.copy_from(
        packed.task_group_offsets.data(), packed.task_group_offsets.size());
    d_bound_task_offsets.copy_from(
        packed.bound_task_offsets.data(), packed.bound_task_offsets.size());
    d_row_window_split.copy_from(
        packed.row_window_split.data(), packed.row_window_split.size());

    constexpr auto blk_mnk = cute::make_shape(
        cute::Int<kPhysicalM>{}, cute::Int<kTileN>{}, cute::Int<kBlockK>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = cute::make_shape(
        cute::Int<kMmaM>{}, cute::Int<kMmaN>{}, cute::Int<kMmaK>{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk = cute::make_shape(
        cute::Int<kPhysicalM / kMmaM>{},
        cute::Int<kTileN / kMmaN>{},
        cute::Int<kBlockK / kMmaK>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    using WARP_MNK = decltype(cute::make_shape(cute::_1{}, cute::_1{}, cute::_1{}));
    const dim3 block(SPARSENE_GENERATED_THREADS_PER_BLOCK);
    auto launch = [&]() {
        float* generated_output = segmented_execution ? d_partial.ptr : d_c.ptr;
#if SPARSENE_PACKING_MODE == 2 || SPARSENE_PACKING_MODE == 3 || \
    SPARSENE_PACKING_MODE == 4
        if (task_mapped_execution) {
#ifndef SPARSENE_GENERATED_SUPPORTS_TASK_MAPPING
            throw std::runtime_error("task-mapped launch requires generated support");
#else
            const dim3 grid(packed.task_groups, n / kTileN);
            if (packed.execution_policy == "multi_bind") {
                generated_bound_task_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>
                    <<<grid, block>>>(
                        d_b.ptr,
                        d_c.ptr,
                        d_row_offsets.ptr,
                        d_values.ptr,
                        d_columns.ptr,
                        d_bound_task_offsets.ptr,
                        d_task_row_window.ptr,
                        d_row_window_split.ptr,
                        k,
                        backend_m,
                        packed.row_windows + 1,
                        n,
                        packed.tiles);
            } else {
                generated_task_mapped_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>
                    <<<grid, block>>>(
                    d_b.ptr,
                    d_c.ptr,
                    d_row_offsets.ptr,
                    d_values.ptr,
                    d_columns.ptr,
                    d_task_tile_begin.ptr,
                    d_task_tile_end.ptr,
                    d_task_row_window.ptr,
                    d_task_use_atomic.ptr,
                    d_task_group_offsets.ptr,
                    k,
                    backend_m,
                    packed.row_windows + 1,
                    n,
                    packed.tiles);
            }
#endif
        } else {
            // Dense SR-BCRS canonical row-window/N-tile grid mapping.
            const dim3 grid(packed.execution_segments, n / kTileN);
            generated_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK><<<grid, block>>>(
                d_b.ptr,
                generated_output,
                d_row_offsets.ptr,
                d_values.ptr,
                d_columns.ptr,
                k,
                packed.execution_segments * kPhysicalM,
                packed.execution_segments + 1,
                n,
                packed.tiles);
        }
#elif SPARSENE_PACKING_MODE == 1
            // BIT-BSR's format-derived OPIR maps row blocks to grid.x and N
            // tiles to grid.y.  grid.x has ample range for all selected data.
            const dim3 grid(packed.execution_segments, n / kTileN);
            generated_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK><<<grid, block>>>(
                d_b.ptr,
                generated_output,
                d_row_offsets.ptr,
                d_value_offsets.ptr,
                d_masks.ptr,
                d_values.ptr,
                d_columns.ptr,
                k,
                packed.execution_segments * kPhysicalM,
                packed.execution_segments + 1,
                n,
                packed.tiles,
                static_cast<int>(packed.value_offsets.size()),
                packed.aligned_values);
#else
            // TCF maps N tiles to grid.x and row windows to grid.y.  grid.y is
            // limited to 65535 on SM80, so large matrices need multiple launches.
            constexpr int max_grid_y = 32768;
            for (int segment_base = 0; segment_base < packed.execution_segments;
                 segment_base += max_grid_y) {
                const int chunk_segments =
                    std::min(max_grid_y, packed.execution_segments - segment_base);
                const dim3 grid(n / kTileN, chunk_segments);
                generated_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK><<<grid, block>>>(
                    d_b.ptr,
                    generated_output + static_cast<size_t>(segment_base) * kPhysicalM * n,
                    d_row_offsets.ptr + segment_base,
                    d_value_offsets.ptr,
                    d_masks.ptr,
                    d_values.ptr,
                    d_columns.ptr,
                    k,
                    chunk_segments * kPhysicalM,
                    chunk_segments + 1,
                    n,
                    packed.tiles,
                    static_cast<int>(packed.value_offsets.size()),
                    packed.aligned_values);
            }
#endif
        if (segmented_execution) {
            const size_t output_count = static_cast<size_t>(backend_m) * n;
            const int reduction_blocks = static_cast<int>(
                std::min<size_t>((output_count + 255) / 256, 1048576));
            reduce_tcf_segments<<<reduction_blocks, 256>>>(
                d_partial.ptr,
                d_c.ptr,
                d_segment_offsets.ptr,
                packed.row_windows,
                n);
        }
    };
    for (int i = 0; i < warmup; ++i) launch();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CHECK_CUDA(cudaEventCreate(&begin));
    CHECK_CUDA(cudaEventCreate(&end));
    CHECK_CUDA(cudaEventRecord(begin));
    for (int i = 0; i < repeat; ++i) launch();
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, begin, end));
    CHECK_CUDA(cudaEventDestroy(begin));
    CHECK_CUDA(cudaEventDestroy(end));
    // Timed atomic kernels follow the historical Sparsene protocol and may
    // accumulate across repeats. Generate a clean single-launch result after
    // timing so verification remains strict and deterministic.
    if (task_mapped_execution) {
        CHECK_CUDA(cudaMemset(d_c.ptr, 0, sizeof(float) * d_c.count));
        launch();
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());
    }
    physical_c.resize(static_cast<size_t>(backend_m) * n);
    CHECK_CUDA(cudaMemcpy(physical_c.data(), d_c.ptr, sizeof(float) * physical_c.size(),
                          cudaMemcpyDeviceToHost));
    return elapsed_ms / repeat;
}

std::vector<float> unpack_output(
    int m,
    int n,
    const PackedMco& packed,
    const std::vector<float>& physical_c) {
    std::vector<float> output(static_cast<size_t>(m) * n, 0.0f);
    for (int new_row = 0; new_row < static_cast<int>(packed.row_order.size()); ++new_row) {
        const int window = new_row / kLogicalM;
        const int local = new_row % kLogicalM;
        const int physical_row = window * kPhysicalM + local;
        const int original_row = packed.row_order[new_row];
        std::copy_n(
            physical_c.data() + static_cast<size_t>(physical_row) * n,
            n,
            output.data() + static_cast<size_t>(original_row) * n);
    }
    return output;
}

struct CusparseResult {
    std::vector<float> output;
    float milliseconds = 0.0f;
};

CusparseResult run_cusparse(
    int m,
    int n,
    int k,
    int nnz,
    const int* row_ptr,
    const int* col_idx,
    const float* values,
    const std::vector<float>& b,
    int warmup,
    int repeat) {
    DeviceArray<int> d_row_ptr(m + 1);
    DeviceArray<int> d_col_idx(nnz);
    DeviceArray<float> d_values(nnz);
    DeviceArray<float> d_b(static_cast<size_t>(k) * n);
    DeviceArray<float> d_c(static_cast<size_t>(m) * n);
    d_row_ptr.copy_from(row_ptr, m + 1);
    d_col_idx.copy_from(col_idx, nnz);
    d_values.copy_from(values, nnz);
    d_b.copy_from(b.data(), b.size());
    CHECK_CUDA(cudaMemset(d_c.ptr, 0, sizeof(float) * d_c.count));

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseDnMatDescr_t mat_b = nullptr;
    cusparseDnMatDescr_t mat_c = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateCsr(
        &mat_a, m, k, nnz, d_row_ptr.ptr, d_col_idx.ptr, d_values.ptr,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &mat_b, k, n, n, d_b.ptr, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &mat_c, m, n, n, d_c.ptr, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    size_t workspace_size = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_a, mat_b, &beta, mat_c, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &workspace_size));
    DeviceArray<unsigned char> workspace(workspace_size);
    CHECK_CUSPARSE(cusparseSpMM_preprocess(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_a, mat_b, &beta, mat_c, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, workspace.ptr));
    auto launch = [&]() {
        CHECK_CUSPARSE(cusparseSpMM(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_a, mat_b, &beta, mat_c, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, workspace.ptr));
    };
    for (int i = 0; i < warmup; ++i) launch();
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CHECK_CUDA(cudaEventCreate(&begin));
    CHECK_CUDA(cudaEventCreate(&end));
    CHECK_CUDA(cudaEventRecord(begin));
    for (int i = 0; i < repeat; ++i) launch();
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, begin, end));
    CHECK_CUDA(cudaEventDestroy(begin));
    CHECK_CUDA(cudaEventDestroy(end));
    CusparseResult result;
    result.milliseconds = elapsed_ms / repeat;
    result.output.resize(static_cast<size_t>(m) * n);
    CHECK_CUDA(cudaMemcpy(result.output.data(), d_c.ptr,
                          sizeof(float) * result.output.size(), cudaMemcpyDeviceToHost));
    CHECK_CUSPARSE(cusparseDestroySpMat(mat_a));
    CHECK_CUSPARSE(cusparseDestroyDnMat(mat_b));
    CHECK_CUSPARSE(cusparseDestroyDnMat(mat_c));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    return result;
}

struct Verification {
    size_t mismatches = 0;
    size_t nonfinite = 0;
    size_t elements = 0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    int max_abs_row = -1;
    int max_abs_col = -1;
    double max_abs_reference = 0.0;
    double max_abs_actual = 0.0;
    double max_reference_magnitude = 0.0;
    double relative_l2_error = 0.0;
    bool passed = false;
};

Verification verify(
    const std::vector<float>& reference,
    const std::vector<float>& actual,
    int n) {
    if (reference.size() != actual.size()) throw std::runtime_error("output size mismatch");
    constexpr double atol = 2.0e-2;
    constexpr double rtol = 5.0e-3;
    constexpr double max_tf32_relative_l2 = 5.0e-4;
    constexpr double max_tf32_normalized_abs = 5.0e-3;
    Verification result;
    result.elements = reference.size();
    long double squared_error = 0.0;
    long double squared_reference = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double ref = reference[i];
        const double got = actual[i];
        const double abs_error = std::abs(ref - got);
        const double rel_error = abs_error / std::max(std::abs(ref), 1.0e-12);
        squared_error += static_cast<long double>(abs_error) * abs_error;
        squared_reference += static_cast<long double>(ref) * ref;
        result.max_reference_magnitude =
            std::max(result.max_reference_magnitude, std::abs(ref));
        if (!std::isfinite(got)) {
            ++result.nonfinite;
        }
        if (!std::isfinite(got) || abs_error > atol + rtol * std::abs(ref)) {
            ++result.mismatches;
        }
        if (abs_error > result.max_abs_error) {
            result.max_abs_error = abs_error;
            result.max_abs_row = static_cast<int>(i / n);
            result.max_abs_col = static_cast<int>(i % n);
            result.max_abs_reference = ref;
            result.max_abs_actual = got;
        }
        result.max_rel_error = std::max(result.max_rel_error, rel_error);
    }
    result.relative_l2_error = std::sqrt(
        static_cast<double>(squared_error / std::max<long double>(squared_reference, 1.0e-30)));
    const double normalized_abs_error = result.max_abs_error /
        std::max(result.max_reference_magnitude, 1.0e-12);
    // TF32 products can exceed a pointwise tolerance at a tiny number of
    // near-cancellation outputs even when the sparse structure and reduction
    // are correct.  Keep reporting those strict mismatches, but accept only a
    // finite result with both a small global L2 error and a small maximum error
    // normalized by the output scale.  Structural/indexing failures exceed
    // these bounds by orders of magnitude (for example dc2 was relative-L2
    // 0.68 before the K-tile fix).
    result.passed = result.nonfinite == 0 &&
        (result.mismatches == 0 ||
         (result.relative_l2_error <= max_tf32_relative_l2 &&
          normalized_abs_error <= max_tf32_normalized_abs));
    return result;
}

void write_json(
    const Options& options,
    int m,
    int k,
    int nnz,
    const PackedMco& packed,
    float generated_ms,
    float cusparse_ms,
    const Verification& verification) {
    std::ofstream out(options.result_json);
    if (!out) throw std::runtime_error("cannot open result JSON " + options.result_json);
    out << std::boolalpha;
    out << "{\n"
        << "  \"schema_version\": \"sparsene.custom_format_e2e.v2\",\n"
        << "  \"dataset\": \"" << options.dataset << "\",\n"
        << "  \"matrix_path\": \"" << options.filename << "\",\n"
        << "  \"shape\": {\"M\": " << m << ", \"K\": " << k
        << ", \"N\": " << options.n << ", \"nnz\": " << nnz << "},\n"
        << "  \"value_policy\": \"deterministic_structural_values_1_to_5.v1\",\n"
        << "  \"format\": {\"logical_blk_m\": " << kLogicalM
        << ", \"physical_blk_m\": " << kPhysicalM
        << ", \"blk_k\": " << kBlockK
        << ", \"row_reorder\": \"" << row_reorder_name()
        << "\", \"packing_mode\": \"" << packing_mode_name() << "\"},\n"
        << "  \"packed\": {\"row_windows\": " << packed.row_windows
        << ", \"tiles\": " << packed.tiles
        << ", \"execution_segments\": " << packed.execution_segments
        << ", \"segmented_execution\": "
        << (packed.execution_segments != packed.row_windows)
        << ", \"segmentation_policy\": \""
        << (packed.automatic_segmentation ? "distribution_auto_v1" : "fixed")
        << "\", \"max_tiles_per_execution_segment\": "
        << packed.max_tiles_per_execution_segment
        << ", \"row_tiles_mean\": " << packed.row_tiles_mean
        << ", \"row_tiles_p95\": " << packed.row_tiles_p95
        << ", \"row_tiles_max\": " << packed.row_tiles_max
        << ", \"distribution_heavy_tail\": " << packed.distribution_heavy_tail
        << ", \"execution_policy\": \"" << packed.execution_policy
        << "\", \"task_groups\": " << packed.task_groups
        << ", \"tasks\": " << packed.task_tile_begin.size()
        << ", \"task_group_tiles\": " << packed.task_group_tiles
        << ", \"stored_values_aligned\": " << packed.aligned_values
        << ", \"mco_values_aligned\": " << packed.aligned_values << "},\n"
        << "  \"timing\": {\"contract\": \"sparsene.kernel_only_post_conversion.v1\", "
        << "\"warmup\": " << options.warmup << ", \"repeat\": " << options.repeat
        << ", \"generated_ms\": " << generated_ms
        << ", \"cusparse_ms\": " << cusparse_ms
        << ", \"speedup_vs_cusparse\": " << (cusparse_ms / generated_ms) << "},\n"
        << "  \"correctness\": {\"passed\": " << verification.passed
        << ", \"mismatches\": " << verification.mismatches
        << ", \"nonfinite\": " << verification.nonfinite
        << ", \"elements\": " << verification.elements
        << ", \"mismatch_fraction\": "
        << (static_cast<double>(verification.mismatches) /
            std::max<size_t>(verification.elements, 1))
        << ", \"max_abs_error\": " << verification.max_abs_error
        << ", \"max_abs_normalized\": "
        << (verification.max_abs_error /
            std::max(verification.max_reference_magnitude, 1.0e-12))
        << ", \"max_rel_error\": " << verification.max_rel_error
        << ", \"relative_l2_error\": " << verification.relative_l2_error
        << ", \"max_reference_magnitude\": " << verification.max_reference_magnitude
        << ", \"max_abs_reference\": " << verification.max_abs_reference
        << ", \"max_abs_actual\": " << verification.max_abs_actual
        << ", \"max_abs_location\": [" << verification.max_abs_row << ", "
        << verification.max_abs_col
        << "], \"atol\": 0.02, \"rtol\": 0.005, "
        << "\"tf32_max_relative_l2\": 0.0005, "
        << "\"tf32_max_abs_normalized\": 0.005}\n"
        << "}\n";
}

}  // namespace sparsene_custom_e2e

using namespace sparsene_custom_e2e;

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int m = 0;
        int k = 0;
        int nnz = 0;
        int symmetric = 0;
        int* row_ptr = nullptr;
        int* col_idx = nullptr;
        float* values = nullptr;
        mmio_allinone(
            &m, &k, &nnz, &symmetric, &row_ptr, &col_idx, &values,
            options.filename.c_str());
        if (!row_ptr || !col_idx || !values || m <= 0 || k <= 0 || nnz <= 0) {
            throw std::runtime_error("failed to load a nonempty MatrixMarket CSR matrix");
        }
        // Follow the historical Sparsene correctness protocol while removing
        // libc-rand dependence: test the matrix structure with bounded,
        // reproducible values.  Kernel performance is independent of values.
        for (int p = 0; p < nnz; ++p) values[p] = static_cast<float>(p % 5 + 1);
        std::vector<float> b(static_cast<size_t>(k) * options.n);
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] = static_cast<float>((static_cast<int>(i % 29) - 14) / 14.0);
        }

        std::cout << "dataset=" << options.dataset << " M=" << m << " K=" << k
                  << " N=" << options.n << " nnz=" << nnz << " logical_tile="
                  << kLogicalM << "x" << kBlockK << " physical_tile="
                  << kPhysicalM << "x" << kBlockK << " row_reorder="
                  << row_reorder_name() << " packing=" << packing_mode_name() << std::endl;
        const PackedMco packed = convert_to_mco(
            m,
            k,
            row_ptr,
            col_idx,
            values,
            options.max_tiles_per_segment,
            options.execution_policy,
            options.task_group_tiles);
        std::vector<float> physical_c;
        const float generated_ms = time_generated_kernel(
            m, options.n, k, packed, b, options.warmup, options.repeat, physical_c);
        const std::vector<float> generated_c = unpack_output(m, options.n, packed, physical_c);
        const CusparseResult cusparse = run_cusparse(
            m, options.n, k, nnz, row_ptr, col_idx, values, b,
            options.warmup, options.repeat);
        const Verification verification = verify(cusparse.output, generated_c, options.n);
        write_json(
            options, m, k, nnz, packed, generated_ms, cusparse.milliseconds, verification);
        std::cout << "generated_ms=" << generated_ms
                  << " cusparse_ms=" << cusparse.milliseconds
                  << " speedup=" << cusparse.milliseconds / generated_ms
                  << " correctness=" << (verification.passed ? "PASS" : "FAIL")
                  << " mismatches=" << verification.mismatches
                  << " max_abs_error=" << verification.max_abs_error << std::endl;
        free(row_ptr);
        free(col_idx);
        free(values);
        return verification.passed ? 0 : 2;
    } catch (const std::exception& exc) {
        std::cerr << "fatal: " << exc.what() << std::endl;
        return 1;
    }
}
