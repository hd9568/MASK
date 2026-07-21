/*
 * Sparsene matrix feature extractor, schema v3.
 *
 * CUDA computes expanded row/column histograms. Host-side analysis then
 * derives locality and exact ME-TCF row-window condensation features. The
 * latter intentionally mirrors buildMETCFMetadata() instead of measuring
 * regular blocks in the uncompressed coordinate space.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "src/mmio.h"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error_ = (call);                                            \
        if (error_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(error_));                  \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

namespace {

constexpr int kThreads = 256;
constexpr int kBlockMs[] = {8, 16, 32};
constexpr int kBlockKs[] = {8, 16, 32};

struct Json {
    std::string out;
    void add(const std::string& value) { out += value; }
    void key(const char* value) { string(value); out += ':'; }
    void string(const std::string& value) {
        out += '"';
        for (char ch : value) {
            switch (ch) {
                case '\\': out += "\\\\"; break;
                case '"': out += "\\\""; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default: out += ch; break;
            }
        }
        out += '"';
    }
    template <typename T> void integer(T value) { out += std::to_string(value); }
    void number(double value, int precision = 8) {
        char buffer[96];
        std::snprintf(buffer, sizeof(buffer), "%.*g", precision, value);
        out += buffer;
    }
    void boolean(bool value) { out += value ? "true" : "false"; }
    void comma() { out += ','; }
};

struct Distribution {
    long long count = 0;
    double mean = 0;
    double stddev = 0;
    double min = 0;
    double max = 0;
    double p25 = 0;
    double p50 = 0;
    double p75 = 0;
    double p90 = 0;
    double p95 = 0;
    double p99 = 0;
    double cv = 0;
    double gini = 0;
    double zero_fraction = 0;
    double top1pct_share = 0;
};

Distribution distribution(const std::vector<int>& input) {
    Distribution d;
    d.count = static_cast<long long>(input.size());
    if (input.empty()) return d;
    std::vector<int> values = input;
    std::sort(values.begin(), values.end());
    long double sum = std::accumulate(values.begin(), values.end(), 0.0L);
    d.mean = static_cast<double>(sum / values.size());
    long double squared = 0;
    long long zeros = 0;
    for (int value : values) {
        long double delta = value - d.mean;
        squared += delta * delta;
        zeros += value == 0;
    }
    d.stddev = std::sqrt(static_cast<double>(squared / values.size()));
    d.min = values.front();
    d.max = values.back();
    auto percentile = [&](double p) -> double {
        size_t index = static_cast<size_t>(std::floor(p * (values.size() - 1)));
        return values[index];
    };
    d.p25 = percentile(0.25); d.p50 = percentile(0.50); d.p75 = percentile(0.75);
    d.p90 = percentile(0.90); d.p95 = percentile(0.95); d.p99 = percentile(0.99);
    d.cv = d.mean > 0 ? d.stddev / d.mean : 0;
    d.zero_fraction = static_cast<double>(zeros) / values.size();
    if (sum > 0) {
        long double weighted = 0;
        for (size_t i = 0; i < values.size(); ++i) weighted += (i + 1) * values[i];
        d.gini = static_cast<double>(2 * weighted / (values.size() * sum) -
                                     (values.size() + 1.0) / values.size());
        size_t top_count = std::max<size_t>(1, (values.size() + 99) / 100);
        long double top_sum = 0;
        for (size_t i = values.size() - top_count; i < values.size(); ++i) top_sum += values[i];
        d.top1pct_share = static_cast<double>(top_sum / sum);
    }
    return d;
}

void emit_distribution(Json& j, const Distribution& d) {
    j.add("{");
    j.key("count"); j.integer(d.count); j.comma();
    j.key("mean"); j.number(d.mean); j.comma();
    j.key("std"); j.number(d.stddev); j.comma();
    j.key("min"); j.number(d.min); j.comma();
    j.key("max"); j.number(d.max); j.comma();
    j.key("p25"); j.number(d.p25); j.comma();
    j.key("p50"); j.number(d.p50); j.comma();
    j.key("p75"); j.number(d.p75); j.comma();
    j.key("p90"); j.number(d.p90); j.comma();
    j.key("p95"); j.number(d.p95); j.comma();
    j.key("p99"); j.number(d.p99); j.comma();
    j.key("cv"); j.number(d.cv); j.comma();
    j.key("gini"); j.number(d.gini); j.comma();
    j.key("zero_fraction"); j.number(d.zero_fraction); j.comma();
    j.key("top1pct_share"); j.number(d.top1pct_share);
    j.add("}");
}

__global__ void histogram_kernel(const int* indices, int* counts, int nnz) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nnz) atomicAdd(&counts[indices[index]], 1);
}

struct Csr {
    std::vector<int> offsets;
    std::vector<int> columns;
};

Csr make_csr(int rows, const std::vector<std::pair<int, int>>& coords) {
    Csr csr;
    csr.offsets.assign(rows + 1, 0);
    csr.columns.resize(coords.size());
    for (const auto& coord : coords) csr.offsets[coord.first + 1]++;
    std::partial_sum(csr.offsets.begin(), csr.offsets.end(), csr.offsets.begin());
    std::vector<int> cursor = csr.offsets;
    for (const auto& coord : coords) csr.columns[cursor[coord.first]++] = coord.second;
    for (int row = 0; row < rows; ++row) {
        std::sort(csr.columns.begin() + csr.offsets[row],
                  csr.columns.begin() + csr.offsets[row + 1]);
    }
    return csr;
}

double pearson_row_index(const std::vector<int>& counts) {
    if (counts.size() < 2) return 0;
    double mean_x = (counts.size() - 1) / 2.0;
    double mean_y = std::accumulate(counts.begin(), counts.end(), 0.0) / counts.size();
    long double cov = 0, sx = 0, sy = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        double dx = i - mean_x, dy = counts[i] - mean_y;
        cov += dx * dy; sx += dx * dx; sy += dy * dy;
    }
    return sx > 0 && sy > 0 ? static_cast<double>(cov / std::sqrt(sx * sy)) : 0;
}

double jaccard_rows(const Csr& csr, int lhs, int rhs) {
    int a = csr.offsets[lhs], ae = csr.offsets[lhs + 1];
    int b = csr.offsets[rhs], be = csr.offsets[rhs + 1];
    long long intersection = 0, union_count = 0;
    int last_a = -1, last_b = -1;
    while (a < ae || b < be) {
        while (a < ae && csr.columns[a] == last_a) ++a;
        while (b < be && csr.columns[b] == last_b) ++b;
        if (a >= ae && b >= be) break;
        if (b >= be || (a < ae && csr.columns[a] < csr.columns[b])) {
            last_a = csr.columns[a++]; union_count++;
        } else if (a >= ae || csr.columns[b] < csr.columns[a]) {
            last_b = csr.columns[b++]; union_count++;
        } else {
            last_a = csr.columns[a++]; last_b = csr.columns[b++];
            intersection++; union_count++;
        }
    }
    return union_count ? static_cast<double>(intersection) / union_count : 0;
}

struct ReorderProxy {
    double adjacent_jaccard = 0;
    double first_col_sorted_jaccard = 0;
    double gain = 0;
    double condensed_unique_col_reduction_bm16 = 0;
};

long long window_unique_sum(const Csr& csr, const std::vector<int>& order, int block_m) {
    long long total = 0;
    std::vector<int> cols;
    for (size_t base = 0; base < order.size(); base += block_m) {
        cols.clear();
        for (size_t i = base; i < std::min(order.size(), base + block_m); ++i) {
            int row = order[i];
            cols.insert(cols.end(), csr.columns.begin() + csr.offsets[row],
                        csr.columns.begin() + csr.offsets[row + 1]);
        }
        std::sort(cols.begin(), cols.end());
        total += std::unique(cols.begin(), cols.end()) - cols.begin();
    }
    return total;
}

ReorderProxy compute_reorder_proxy(const Csr& csr, int rows) {
    ReorderProxy result;
    std::vector<int> active;
    active.reserve(rows);
    for (int row = 0; row < rows; ++row) {
        if (csr.offsets[row] != csr.offsets[row + 1]) active.push_back(row);
    }
    auto adjacent_average = [&](const std::vector<int>& order) {
        if (order.size() < 2) return 0.0;
        long double sum = 0;
        for (size_t i = 1; i < order.size(); ++i) sum += jaccard_rows(csr, order[i - 1], order[i]);
        return static_cast<double>(sum / (order.size() - 1));
    };
    result.adjacent_jaccard = adjacent_average(active);
    std::vector<int> reordered = active;
    std::stable_sort(reordered.begin(), reordered.end(), [&](int lhs, int rhs) {
        int lhs_first = csr.columns[csr.offsets[lhs]];
        int rhs_first = csr.columns[csr.offsets[rhs]];
        if (lhs_first != rhs_first) return lhs_first < rhs_first;
        int lhs_size = csr.offsets[lhs + 1] - csr.offsets[lhs];
        int rhs_size = csr.offsets[rhs + 1] - csr.offsets[rhs];
        return lhs_size < rhs_size;
    });
    result.first_col_sorted_jaccard = adjacent_average(reordered);
    result.gain = result.first_col_sorted_jaccard - result.adjacent_jaccard;
    std::vector<int> original(rows);
    std::iota(original.begin(), original.end(), 0);
    for (int row = 0; row < rows; ++row) {
        if (csr.offsets[row] == csr.offsets[row + 1]) reordered.push_back(row);
    }
    long long before = window_unique_sum(csr, original, 16);
    long long after = window_unique_sum(csr, reordered, 16);
    result.condensed_unique_col_reduction_bm16 =
        before > 0 ? static_cast<double>(before - after) / before : 0;
    return result;
}

struct UniqueColumnFrequency {
    int entries = 0;
    int structural = 0;
};

struct WindowData {
    std::vector<UniqueColumnFrequency> columns;
};

std::vector<WindowData> build_windows(const Csr& csr, int rows, int block_m) {
    int window_count = (rows + block_m - 1) / block_m;
    std::vector<WindowData> windows(window_count);
    std::vector<std::pair<int, int>> coords;
    for (int window = 0; window < window_count; ++window) {
        coords.clear();
        int row_end = std::min(rows, (window + 1) * block_m);
        for (int row = window * block_m; row < row_end; ++row) {
            for (int pos = csr.offsets[row]; pos < csr.offsets[row + 1]; ++pos)
                coords.emplace_back(csr.columns[pos], row);
        }
        std::sort(coords.begin(), coords.end());
        for (size_t i = 0; i < coords.size();) {
            int col = coords[i].first;
            int entries = 0, structural = 0, previous_row = -1;
            while (i < coords.size() && coords[i].first == col) {
                entries++;
                if (coords[i].second != previous_row) { structural++; previous_row = coords[i].second; }
                ++i;
            }
            windows[window].columns.push_back({entries, structural});
        }
    }
    return windows;
}

struct Candidate {
    int block_m = 0, block_k = 0, row_windows = 0;
    long long total_tc_blocks = 0, unique_columns = 0;
    long long entry_nnz = 0, structural_nnz = 0, coo_padded_nnz = 0;
    long long coo_bytes = 0, mask_bytes = 0, dense_bytes = 0;
    Distribution unique_cols_per_window;
    Distribution tc_blocks_per_window;
    Distribution entry_nnz_per_tc_block;
    Distribution structural_nnz_per_tc_block;
    double structural_density = 0, tensorcore_zero_fraction = 0;
    double density_p50_pct = 0, density_p75_pct = 0, density_p90_pct = 0;
    double coo_padding_ratio = 0, b_index_padding_ratio = 0;
    double entries_per_unique_column = 0, metadata_bytes_per_entry = 0;
    double top5_window_share = 0;
    std::string smallest_storage;
};

Candidate simulate_candidate(const std::vector<WindowData>& windows, int block_m, int block_k) {
    Candidate c;
    c.block_m = block_m; c.block_k = block_k; c.row_windows = windows.size();
    std::vector<int> unique_per_window, blocks_per_window, block_entries, block_structural;
    unique_per_window.reserve(windows.size()); blocks_per_window.reserve(windows.size());
    for (const WindowData& window : windows) {
        int unique = window.columns.size();
        int blocks = (unique + block_k - 1) / block_k;
        unique_per_window.push_back(unique); blocks_per_window.push_back(blocks);
        c.unique_columns += unique; c.total_tc_blocks += blocks;
        for (int block = 0; block < blocks; ++block) {
            int entries = 0, structural = 0;
            int end = std::min(unique, (block + 1) * block_k);
            for (int col = block * block_k; col < end; ++col) {
                entries += window.columns[col].entries;
                structural += window.columns[col].structural;
            }
            block_entries.push_back(entries); block_structural.push_back(structural);
            c.entry_nnz += entries; c.structural_nnz += structural;
            c.coo_padded_nnz += ((entries + 7) / 8) * 8;
        }
    }
    c.unique_cols_per_window = distribution(unique_per_window);
    c.tc_blocks_per_window = distribution(blocks_per_window);
    c.entry_nnz_per_tc_block = distribution(block_entries);
    c.structural_nnz_per_tc_block = distribution(block_structural);
    long long cells = c.total_tc_blocks * block_m * block_k;
    c.structural_density = cells ? static_cast<double>(c.structural_nnz) / cells : 0;
    c.tensorcore_zero_fraction = 1.0 - c.structural_density;
    double block_cells = static_cast<double>(block_m) * block_k;
    c.density_p50_pct = block_cells ? 100.0 * c.structural_nnz_per_tc_block.p50 / block_cells : 0;
    c.density_p75_pct = block_cells ? 100.0 * c.structural_nnz_per_tc_block.p75 / block_cells : 0;
    c.density_p90_pct = block_cells ? 100.0 * c.structural_nnz_per_tc_block.p90 / block_cells : 0;
    c.coo_padding_ratio = c.coo_padded_nnz ?
        static_cast<double>(c.coo_padded_nnz - c.entry_nnz) / c.coo_padded_nnz : 0;
    long long gathered_rows = c.total_tc_blocks * block_k;
    c.b_index_padding_ratio = gathered_rows ?
        static_cast<double>(gathered_rows - c.unique_columns) / gathered_rows : 0;
    c.entries_per_unique_column = c.unique_columns ?
        static_cast<double>(c.entry_nnz) / c.unique_columns : 0;
    long long common = c.total_tc_blocks * block_k * 4 + (windows.size() + 1LL) * 4;
    c.coo_bytes = common + (2 * c.total_tc_blocks + 1) * 4 + c.coo_padded_nnz * 8;
    c.mask_bytes = common + (c.total_tc_blocks + 1) * 4 +
                   c.total_tc_blocks * ((block_m * block_k + 7) / 8) + c.entry_nnz * 4;
    c.dense_bytes = common + c.total_tc_blocks * block_m * block_k * 4;
    c.metadata_bytes_per_entry = c.entry_nnz ? static_cast<double>(c.coo_bytes) / c.entry_nnz : 0;
    std::vector<int> sorted_blocks = blocks_per_window;
    std::sort(sorted_blocks.rbegin(), sorted_blocks.rend());
    long long top5 = 0;
    for (size_t i = 0; i < std::min<size_t>(5, sorted_blocks.size()); ++i) top5 += sorted_blocks[i];
    c.top5_window_share = c.total_tc_blocks ? static_cast<double>(top5) / c.total_tc_blocks : 0;
    c.smallest_storage = c.coo_bytes <= c.mask_bytes && c.coo_bytes <= c.dense_bytes ? "coo" :
                         (c.mask_bytes <= c.dense_bytes ? "mask" : "dense");
    return c;
}

void emit_candidate(Json& j, const Candidate& c) {
    j.add("{");
    j.key("blk_m"); j.integer(c.block_m); j.comma();
    j.key("blk_k"); j.integer(c.block_k); j.comma();
    j.key("num_row_windows"); j.integer(c.row_windows); j.comma();
    j.key("total_tc_blocks"); j.integer(c.total_tc_blocks); j.comma();
    j.key("unique_columns_total"); j.integer(c.unique_columns); j.comma();
    j.key("unique_cols_per_window"); emit_distribution(j, c.unique_cols_per_window); j.comma();
    j.key("tc_blocks_per_window"); emit_distribution(j, c.tc_blocks_per_window); j.comma();
    j.key("entry_nnz_per_tc_block"); emit_distribution(j, c.entry_nnz_per_tc_block); j.comma();
    j.key("structural_nnz_per_tc_block"); emit_distribution(j, c.structural_nnz_per_tc_block); j.comma();
    j.key("structural_density"); j.number(c.structural_density); j.comma();
    j.key("avg_density"); j.number(c.structural_density); j.comma();
    j.key("density_p50_pct"); j.number(c.density_p50_pct); j.comma();
    j.key("density_p75_pct"); j.number(c.density_p75_pct); j.comma();
    j.key("density_p90_pct"); j.number(c.density_p90_pct); j.comma();
    j.key("padding_ratio"); j.number(c.tensorcore_zero_fraction); j.comma();
    j.key("tensorcore_zero_fraction"); j.number(c.tensorcore_zero_fraction); j.comma();
    j.key("coo_padded_nnz"); j.integer(c.coo_padded_nnz); j.comma();
    j.key("coo_padding_ratio"); j.number(c.coo_padding_ratio); j.comma();
    j.key("b_index_padding_ratio"); j.number(c.b_index_padding_ratio); j.comma();
    j.key("entries_per_unique_column"); j.number(c.entries_per_unique_column); j.comma();
    j.key("top5_window_share"); j.number(c.top5_window_share); j.comma();
    j.key("storage_estimate_bytes"); j.add("{");
    j.key("coo"); j.integer(c.coo_bytes); j.comma();
    j.key("mask"); j.integer(c.mask_bytes); j.comma();
    j.key("dense"); j.integer(c.dense_bytes); j.comma();
    j.key("smallest"); j.string(c.smallest_storage); j.add("}"); j.comma();
    j.key("coo_bytes_per_entry"); j.number(c.metadata_bytes_per_entry);
    j.add("}");
}

void emit_candidates(Json& j, const std::vector<Candidate>& candidates) {
    j.add("{");
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (i) j.comma();
        j.key((std::to_string(candidates[i].block_m) + "x" +
               std::to_string(candidates[i].block_k)).c_str());
        emit_candidate(j, candidates[i]);
    }
    j.add("}");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <matrix.mtx>\n", argv[0]);
        return 1;
    }

    int M = 0, K = 0, file_nnz = 0;
    int *I = nullptr, *J = nullptr;
    double* values = nullptr;
    MM_typecode matcode;
    int rc = mm_read_mtx_crd(argv[1], &M, &K, &file_nnz, &I, &J, &values, &matcode);
    if (rc != 0 || M < 0 || K < 0 || file_nnz < 0) {
        std::fprintf(stderr, "Failed to read %s (error %d)\n", argv[1], rc);
        return 1;
    }
    bool symmetric = mm_is_symmetric(matcode) || mm_is_hermitian(matcode);
    std::vector<std::pair<int, int>> coords;
    coords.reserve(static_cast<size_t>(file_nnz) * (symmetric ? 2 : 1));
    long long invalid_entries = 0;
    for (int i = 0; i < file_nnz; ++i) {
        int row = I[i] - 1, col = J[i] - 1;
        if (row < 0 || row >= M || col < 0 || col >= K) { invalid_entries++; continue; }
        coords.emplace_back(row, col);
        if (symmetric && row != col && col < M && row < K) coords.emplace_back(col, row);
    }
    std::free(I); std::free(J); std::free(values);
    std::sort(coords.begin(), coords.end());
    std::vector<std::pair<int, int>> structural_coords = coords;
    structural_coords.erase(std::unique(structural_coords.begin(), structural_coords.end()),
                            structural_coords.end());
    long long duplicate_entries = coords.size() - structural_coords.size();
    Csr csr = make_csr(M, coords);

    std::vector<int> rows_h(coords.size()), cols_h(coords.size());
    for (size_t i = 0; i < coords.size(); ++i) {
        rows_h[i] = coords[i].first; cols_h[i] = coords[i].second;
    }
    std::vector<int> row_counts(M, 0), col_counts(K, 0);
    if (!coords.empty()) {
        int *d_rows = nullptr, *d_cols = nullptr, *d_row_counts = nullptr, *d_col_counts = nullptr;
        CUDA_CHECK(cudaMalloc(&d_rows, rows_h.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cols, cols_h.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_row_counts, M * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_counts, K * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_rows, rows_h.data(), rows_h.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, cols_h.data(), cols_h.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_row_counts, 0, M * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_col_counts, 0, K * sizeof(int)));
        int grid = (coords.size() + kThreads - 1) / kThreads;
        histogram_kernel<<<grid, kThreads>>>(d_rows, d_row_counts, coords.size());
        histogram_kernel<<<grid, kThreads>>>(d_cols, d_col_counts, coords.size());
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(row_counts.data(), d_row_counts, M * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(col_counts.data(), d_col_counts, K * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_rows)); CUDA_CHECK(cudaFree(d_cols));
        CUDA_CHECK(cudaFree(d_row_counts)); CUDA_CHECK(cudaFree(d_col_counts));
    }

    std::vector<int> row_spans, column_gaps;
    row_spans.reserve(M);
    long long diagonal_band_entries = 0;
    double normalized_bandwidth = 0;
    for (int row = 0; row < M; ++row) {
        int begin = csr.offsets[row], end = csr.offsets[row + 1];
        if (begin == end) { row_spans.push_back(0); continue; }
        row_spans.push_back(csr.columns[end - 1] - csr.columns[begin] + 1);
        int previous = -1;
        for (int pos = begin; pos < end; ++pos) {
            int col = csr.columns[pos];
            if (col == previous) continue;
            if (previous >= 0) column_gaps.push_back(col - previous);
            previous = col;
        }
    }
    int diagonal_band = std::max(1, std::max(M, K) / 100);
    for (const auto& coord : structural_coords) {
        double row_norm = M > 1 ? static_cast<double>(coord.first) / (M - 1) : 0;
        double col_norm = K > 1 ? static_cast<double>(coord.second) / (K - 1) : 0;
        normalized_bandwidth = std::max(normalized_bandwidth, std::fabs(row_norm - col_norm));
        diagonal_band_entries += std::abs(coord.first - coord.second) <= diagonal_band;
    }
    ReorderProxy reorder = compute_reorder_proxy(csr, M);

    std::vector<Candidate> candidates;
    for (int block_m : kBlockMs) {
        std::vector<WindowData> windows = build_windows(csr, M, block_m);
        for (int block_k : kBlockKs) candidates.push_back(simulate_candidate(windows, block_m, block_k));
    }
    auto worst = std::max_element(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return lhs.tc_blocks_per_window.cv < rhs.tc_blocks_per_window.cv;
    });

    Json j;
    j.add("{");
    j.key("schema_version"); j.string("sparsene.matrix_features.v3"); j.comma();
    j.key("extractor"); j.add("{");
    j.key("name"); j.string("featurize_mtx_cuda"); j.comma();
    j.key("backend"); j.string("cuda_histogram+format_aware_host_analysis"); j.comma();
    j.key("symmetry_expanded"); j.boolean(symmetric); j.comma();
    j.key("values_used"); j.boolean(false); j.add("}"); j.comma();
    j.key("path"); j.string(argv[1]); j.comma();
    j.key("shape"); j.add("{");
    j.key("M"); j.integer(M); j.comma(); j.key("K"); j.integer(K); j.comma();
    j.key("N"); j.integer(K); j.comma();
    j.key("file_nnz"); j.integer(file_nnz); j.comma();
    j.key("expanded_nnz"); j.integer(coords.size()); j.comma();
    j.key("structural_nnz"); j.integer(structural_coords.size()); j.comma();
    j.key("nnz"); j.integer(coords.size()); j.add("}"); j.comma();
    double cells = static_cast<double>(M) * K;
    double density = cells > 0 ? structural_coords.size() / cells : 0;
    j.key("density"); j.number(density, 12); j.comma();
    j.key("sparsity"); j.number(1.0 - density, 12); j.comma();
    j.key("symmetry"); j.string(symmetric ? "symmetric_or_hermitian" : "general"); j.comma();
    j.key("data_quality"); j.add("{");
    j.key("invalid_entries"); j.integer(invalid_entries); j.comma();
    j.key("duplicate_entries_after_expansion"); j.integer(duplicate_entries); j.comma();
    j.key("has_duplicates"); j.boolean(duplicate_entries > 0); j.add("}"); j.comma();
    j.key("row_nnz_distribution"); emit_distribution(j, distribution(row_counts)); j.comma();
    j.key("col_nnz_distribution"); emit_distribution(j, distribution(col_counts)); j.comma();
    j.key("row_length_vs_index_correlation"); j.number(pearson_row_index(row_counts)); j.comma();
    j.key("locality"); j.add("{");
    j.key("row_span_distribution"); emit_distribution(j, distribution(row_spans)); j.comma();
    j.key("column_gap_distribution"); emit_distribution(j, distribution(column_gaps)); j.comma();
    j.key("normalized_bandwidth"); j.number(normalized_bandwidth); j.comma();
    j.key("diagonal_band_fraction_1pct");
    j.number(structural_coords.empty() ? 0.0 : static_cast<double>(diagonal_band_entries) / structural_coords.size());
    j.add("}"); j.comma();
    j.key("reorder_proxy"); j.add("{");
    j.key("adjacent_row_jaccard"); j.number(reorder.adjacent_jaccard); j.comma();
    j.key("first_col_sorted_jaccard"); j.number(reorder.first_col_sorted_jaccard); j.comma();
    j.key("jaccard_gain"); j.number(reorder.gain); j.comma();
    j.key("bm16_unique_col_reduction"); j.number(reorder.condensed_unique_col_reduction_bm16);
    j.add("}"); j.comma();
    j.key("hierarchical_candidates"); emit_candidates(j, candidates); j.comma();
    j.key("candidate_block_stats"); emit_candidates(j, candidates); j.comma();
    j.key("load_balance"); j.add("{");
    j.key("worst_candidate");
    j.string(worst == candidates.end() ? "none" : std::to_string(worst->block_m) + "x" + std::to_string(worst->block_k));
    j.comma(); j.key("tc_blocks_cv"); j.number(worst == candidates.end() ? 0 : worst->tc_blocks_per_window.cv);
    j.comma(); j.key("top5_window_share"); j.number(worst == candidates.end() ? 0 : worst->top5_window_share);
    j.add("}");
    j.add("}\n");
    std::fwrite(j.out.data(), 1, j.out.size(), stdout);
    return 0;
}
