#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cute/tensor.hpp>

#include "kernel.inc"
#include "srbcrs_runtime_api.h"

namespace srbcrs_runtime_internal {

struct VariantConfig {
  int tile_m;
  int tile_k;
  bool use_16x8;
};

struct CacheEntry {
  int num_nodes = 0;
  int num_edges = 0;
  int total_tcblock_num = 0;
  int tile_m = 0;
  int tile_k = 0;
  bool use_16x8 = false;

  torch::Tensor rowptr_cuda;
  torch::Tensor colind_cuda;
  torch::Tensor values_cuda;

  torch::Tensor val_sidx_cuda;
  torch::Tensor val_soff_cuda;
  torch::Tensor val_block_val_cuda;
};

std::mutex g_cache_mu;
std::unordered_map<int64_t, CacheEntry> g_cache;

torch::Tensor vector_i32_to_cuda(const std::vector<int>& data, const c10::Device& device) {
  auto opts_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  if (data.empty()) {
    return torch::empty({0}, opts_cpu).to(device);
  }
  auto t_cpu = torch::from_blob(const_cast<int*>(data.data()), {(int64_t)data.size()}, opts_cpu).clone();
  return t_cpu.to(device);
}

torch::Tensor vector_f32_to_cuda(const std::vector<float>& data, const c10::Device& device) {
  auto opts_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  if (data.empty()) {
    return torch::empty({0}, opts_cpu).to(device);
  }
  auto t_cpu =
    torch::from_blob(const_cast<float*>(data.data()), {(int64_t)data.size()}, opts_cpu).clone();
  return t_cpu.to(device);
}

__global__ void csr_spmm_value_kernel(const int* rowptr,
                                      const int* colind,
                                      const float* values,
                                      const float* dense,
                                      float* out,
                                      int m,
                                      int n) {
  const int row = blockIdx.x;
  const int feat = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= m || feat >= n) {
    return;
  }

  float acc = 0.0f;
  const int row_begin = rowptr[row];
  const int row_end = rowptr[row + 1];
  for (int e = row_begin; e < row_end; ++e) {
    const int col = colind[e];
    acc += values[e] * dense[col * n + feat];
  }
  out[row * n + feat] = acc;
}

torch::Tensor fallback_spmm_from_csr(const CacheEntry& entry, const torch::Tensor& dense) {
  const int64_t n = dense.size(1);
  auto out = torch::zeros({entry.num_nodes, n}, dense.options());

  const dim3 block(256);
  const dim3 grid(entry.num_nodes, static_cast<unsigned int>((n + block.x - 1) / block.x));
  auto stream = at::cuda::getDefaultCUDAStream();
  csr_spmm_value_kernel<<<grid, block, 0, stream>>>(entry.rowptr_cuda.data_ptr<int>(),
                                                     entry.colind_cuda.data_ptr<int>(),
                                                     entry.values_cuda.data_ptr<float>(),
                                                     dense.data_ptr<float>(),
                                                     out.data_ptr<float>(),
                                                     entry.num_nodes,
                                                     static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

int find_condensed(const std::vector<int>& cols, int target) {
  auto it = std::lower_bound(cols.begin(), cols.end(), target);
  if (it == cols.end() || *it != target) {
    return -1;
  }
  return static_cast<int>(it - cols.begin());
}

VariantConfig resolve_variant(bool use_16x8) {
  if (use_16x8) {
    return VariantConfig{16, 8, true};
  }
  return VariantConfig{32, 32, false};
}

template <int TILE_M, int TILE_N, int TILE_K>
void run_srbcrs_device(int M,
                       int N,
                       int K,
                       int nnz_block,
                       int* dval_sidx,
                       int* dval_soff,
                       float* dB_val,
                       float* dval_block_val,
                       float* dC,
                       cudaStream_t stream) {
  if (nnz_block == 0 || N == 0) {
    return;
  }

  const int row_window_num = (M + TILE_M - 1) / TILE_M;
  const int threadblock_num_x = row_window_num;
  const int threadblock_num_y = N / TILE_N;
  const int thread_num = 32;
  const int Mo = row_window_num;

  dim3 grid(threadblock_num_x, threadblock_num_y);
  dim3 block(thread_num);

  constexpr auto blk_mnk = cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_N>{}, cute::Int<TILE_K>{});
  using BLK_MNK = decltype(blk_mnk);
  constexpr auto mma_mnk = cute::make_shape(cute::_16{}, cute::_8{}, cute::_8{});
  using MMA_MNK = decltype(mma_mnk);
  constexpr auto blk_mma_mnk =
    cute::make_shape(cute::Int<TILE_M / 16>{}, cute::Int<TILE_N / 8>{}, cute::Int<TILE_K / 8>{});
  using BLK_MMA_MNK = decltype(blk_mma_mnk);
  constexpr auto warp_mnk = cute::make_shape(cute::_1{}, cute::_1{}, cute::_1{});
  using WARP_MNK = decltype(warp_mnk);

  sr_bcrs_spmm_kernel_tf32<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK><<<grid, block, 0, stream>>>(
    dB_val,
    dC,
    dval_block_val,
    dval_sidx,
    dval_soff,
    K,
    M,
    Mo,
    N,
    nnz_block);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace srbcrs_runtime_internal

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
preprocess_gpu_srcfp32_srbcrs(torch::Tensor colind_i32,
                              torch::Tensor rowptr_i32,
                              torch::Tensor values_f32,
                              int num_nodes,
                              int blk_h,
                              int blk_w,
                              torch::Tensor block_partition_i32,
                              torch::Tensor edge_to_col_i32,
                              torch::Tensor edge_to_row_i32,
                              bool use_16x8) {
  TORCH_CHECK(colind_i32.is_cuda(), "colind must be CUDA tensor");
  TORCH_CHECK(rowptr_i32.is_cuda(), "rowptr must be CUDA tensor");
  TORCH_CHECK(values_f32.is_cuda(), "values must be CUDA tensor");
  TORCH_CHECK(colind_i32.scalar_type() == torch::kInt32, "colind must be int32");
  TORCH_CHECK(rowptr_i32.scalar_type() == torch::kInt32, "rowptr must be int32");
  TORCH_CHECK(values_f32.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(num_nodes >= 1, "num_nodes must be positive");

  const auto variant = srbcrs_runtime_internal::resolve_variant(use_16x8);
  TORCH_CHECK(blk_h == variant.tile_m,
              "blk_h mismatch for SR-BCRS variant: expected ",
              variant.tile_m,
              " got ",
              blk_h);
  TORCH_CHECK(blk_w == variant.tile_k,
              "blk_w mismatch for SR-BCRS variant: expected ",
              variant.tile_k,
              " got ",
              blk_w);

  auto rowptr_cpu_t = rowptr_i32.to(torch::kCPU).contiguous();
  auto colind_cpu_t = colind_i32.to(torch::kCPU).contiguous();
  auto values_cpu_t = values_f32.to(torch::kCPU).contiguous();

  auto rowptr = rowptr_cpu_t.data_ptr<int>();
  auto colind = colind_cpu_t.data_ptr<int>();
  auto values = values_cpu_t.data_ptr<float>();

  const int nnz = static_cast<int>(colind_i32.numel());
  TORCH_CHECK(values_f32.numel() == nnz, "values length must equal colind length");

  const int row_window_num = (num_nodes + variant.tile_m - 1) / variant.tile_m;

  std::vector<int> val_soff(row_window_num + 1, 0);
  std::vector<int> block_partition_cpu(row_window_num, 0);
  std::vector<int> edge_to_row_cpu(nnz, 0);
  std::vector<int> edge_to_col_cpu(nnz, 0);

  std::vector<int> val_sidx;
  std::vector<float> val_block_val;

  int total_tcblock_num = 0;

  if (variant.use_16x8) {
    auto swizzle_layout = cute::composition(
      cute::Swizzle<2, 2, 3>{},
      cute::make_layout(cute::make_shape(16, 8), cute::make_stride(8, 1)));

    for (int w = 0; w < row_window_num; ++w) {
      val_soff[w] = total_tcblock_num;

      std::vector<int> unique_cols;
      unique_cols.reserve(256);
      for (int i_row = 0; i_row < 16; ++i_row) {
        const int row = w * 16 + i_row;
        if (row >= num_nodes) {
          continue;
        }
        for (int e = rowptr[row]; e < rowptr[row + 1]; ++e) {
          edge_to_row_cpu[e] = row;
          unique_cols.push_back(colind[e]);
        }
      }

      std::sort(unique_cols.begin(), unique_cols.end());
      unique_cols.erase(std::unique(unique_cols.begin(), unique_cols.end()), unique_cols.end());

      const int nnz_col_num = static_cast<int>(unique_cols.size());
      const int tcblock_num = (nnz_col_num + 8 - 1) / 8;
      block_partition_cpu[w] = tcblock_num;

      const int sidx_base = static_cast<int>(val_sidx.size());
      val_sidx.resize(sidx_base + tcblock_num * 8, unique_cols.empty() ? 0 : unique_cols.back());
      for (int i = 0; i < nnz_col_num; ++i) {
        val_sidx[sidx_base + i] = unique_cols[i];
      }

      const int block_base = static_cast<int>(val_block_val.size());
      val_block_val.resize(block_base + tcblock_num * 16 * 8, 0.0f);

      for (int i_row = 0; i_row < 16; ++i_row) {
        const int row = w * 16 + i_row;
        if (row >= num_nodes) {
          continue;
        }
        for (int e = rowptr[row]; e < rowptr[row + 1]; ++e) {
          const int condensed = srbcrs_runtime_internal::find_condensed(unique_cols, colind[e]);
          TORCH_CHECK(condensed >= 0, "Internal preprocess error: column not found");
          edge_to_col_cpu[e] = condensed;

          const int tcblock_id = condensed / 8;
          const auto coord = cute::make_coord(i_row, condensed % 8);
          const int swizzled_idx = swizzle_layout(coord);
          const int dst = block_base + tcblock_id * 16 * 8 + swizzled_idx;
          val_block_val[dst] = values[e];
        }
      }

      total_tcblock_num += tcblock_num;
    }
  } else {
    auto swizzle_layout = cute::composition(
      cute::Swizzle<2, 2, 3>{},
      cute::make_layout(cute::make_shape(32, 32), cute::make_stride(32, 1)));

    for (int w = 0; w < row_window_num; ++w) {
      val_soff[w] = total_tcblock_num;

      std::vector<int> unique_cols;
      unique_cols.reserve(256);
      for (int i_row = 0; i_row < 32; ++i_row) {
        const int row = w * 32 + i_row;
        if (row >= num_nodes) {
          continue;
        }
        for (int e = rowptr[row]; e < rowptr[row + 1]; ++e) {
          edge_to_row_cpu[e] = row;
          unique_cols.push_back(colind[e]);
        }
      }

      std::sort(unique_cols.begin(), unique_cols.end());
      unique_cols.erase(std::unique(unique_cols.begin(), unique_cols.end()), unique_cols.end());

      const int nnz_col_num = static_cast<int>(unique_cols.size());
      const int tcblock_num = (nnz_col_num + 32 - 1) / 32;
      block_partition_cpu[w] = tcblock_num;

      const int sidx_base = static_cast<int>(val_sidx.size());
      val_sidx.resize(sidx_base + tcblock_num * 32, unique_cols.empty() ? 0 : unique_cols.back());
      for (int i = 0; i < nnz_col_num; ++i) {
        val_sidx[sidx_base + i] = unique_cols[i];
      }

      const int block_base = static_cast<int>(val_block_val.size());
      val_block_val.resize(block_base + tcblock_num * 32 * 32, 0.0f);

      for (int i_row = 0; i_row < 32; ++i_row) {
        const int row = w * 32 + i_row;
        if (row >= num_nodes) {
          continue;
        }
        for (int e = rowptr[row]; e < rowptr[row + 1]; ++e) {
          const int condensed = srbcrs_runtime_internal::find_condensed(unique_cols, colind[e]);
          TORCH_CHECK(condensed >= 0, "Internal preprocess error: column not found");
          edge_to_col_cpu[e] = condensed;

          const int tcblock_id = condensed / 32;
          const auto coord = cute::make_coord(i_row, condensed % 32);
          const int swizzled_idx = swizzle_layout(coord);
          const int dst = block_base + tcblock_id * 32 * 32 + swizzled_idx;
          val_block_val[dst] = values[e];
        }
      }

      total_tcblock_num += tcblock_num;
    }
  }
  val_soff[row_window_num] = total_tcblock_num;

  auto opts_gpu_i32 = torch::TensorOptions().dtype(torch::kInt32).device(colind_i32.device());

  torch::Tensor row_window_offset =
    srbcrs_runtime_internal::vector_i32_to_cuda(val_soff, colind_i32.device());
  torch::Tensor tcblock_rowid = torch::zeros({total_tcblock_num}, opts_gpu_i32);
  torch::Tensor tcblocktile_id =
    srbcrs_runtime_internal::vector_i32_to_cuda(val_sidx, colind_i32.device());
  torch::Tensor tcblock_offset = torch::zeros({1}, opts_gpu_i32);
  torch::Tensor sparse_a_to_x_idx = torch::zeros({1}, opts_gpu_i32);
  torch::Tensor tcblock_val =
    srbcrs_runtime_internal::vector_f32_to_cuda(val_block_val, colind_i32.device());

  if (block_partition_i32.numel() >= row_window_num) {
    auto bp_cpu = torch::from_blob(block_partition_cpu.data(), {row_window_num}, torch::TensorOptions().dtype(torch::kInt32)).clone();
    block_partition_i32.narrow(0, 0, row_window_num).copy_(bp_cpu.to(block_partition_i32.device()));
  }
  if (edge_to_row_i32.numel() >= nnz) {
    auto er_cpu = torch::from_blob(edge_to_row_cpu.data(), {nnz}, torch::TensorOptions().dtype(torch::kInt32)).clone();
    edge_to_row_i32.narrow(0, 0, nnz).copy_(er_cpu.to(edge_to_row_i32.device()));
  }
  if (edge_to_col_i32.numel() >= nnz) {
    auto ec_cpu = torch::from_blob(edge_to_col_cpu.data(), {nnz}, torch::TensorOptions().dtype(torch::kInt32)).clone();
    edge_to_col_i32.narrow(0, 0, nnz).copy_(ec_cpu.to(edge_to_col_i32.device()));
  }

  const int64_t key = reinterpret_cast<int64_t>(row_window_offset.data_ptr<int>());
  {
    std::lock_guard<std::mutex> guard(srbcrs_runtime_internal::g_cache_mu);
    auto& entry = srbcrs_runtime_internal::g_cache[key];
    entry.num_nodes = num_nodes;
    entry.num_edges = nnz;
    entry.total_tcblock_num = total_tcblock_num;
    entry.tile_m = variant.tile_m;
    entry.tile_k = variant.tile_k;
    entry.use_16x8 = variant.use_16x8;

    entry.rowptr_cuda = rowptr_i32.contiguous();
    entry.colind_cuda = colind_i32.contiguous();
    entry.values_cuda = values_f32.contiguous();
    entry.val_sidx_cuda = tcblocktile_id;
    entry.val_soff_cuda = row_window_offset;
    entry.val_block_val_cuda = tcblock_val;
  }

  return std::make_tuple(row_window_offset,
                         tcblock_rowid,
                         tcblocktile_id,
                         tcblock_offset,
                         sparse_a_to_x_idx,
                         total_tcblock_num);
}

std::vector<torch::Tensor> run_spmm_srcfp32_srbcrs(torch::Tensor dense,
                                                    torch::Tensor row_window_offset,
                                                    torch::Tensor /*tcblocktile_id*/,
                                                    torch::Tensor /*tcblock_offset*/,
                                                    torch::Tensor /*sparse_a_to_x_idx*/,
                                                    int num_nodes,
                                                    int /*num_edges*/,
                                                    std::string /*exeplan*/,
                                                    bool use_16x8) {
  TORCH_CHECK(dense.is_cuda(), "dense must be CUDA tensor");
  TORCH_CHECK(dense.scalar_type() == torch::kFloat32, "dense must be float32");
  TORCH_CHECK(dense.dim() == 2, "dense must be 2-D");

  const int64_t key = reinterpret_cast<int64_t>(row_window_offset.data_ptr<int>());

  srbcrs_runtime_internal::CacheEntry entry;
  bool has_entry = false;
  {
    std::lock_guard<std::mutex> guard(srbcrs_runtime_internal::g_cache_mu);
    auto it = srbcrs_runtime_internal::g_cache.find(key);
    if (it != srbcrs_runtime_internal::g_cache.end()) {
      entry = it->second;
      has_entry = true;
    }
  }

  TORCH_CHECK(has_entry, "run_SRBCRS cache miss. Call preprocess_gpu before run_SRBCRS.");
  TORCH_CHECK(num_nodes == entry.num_nodes,
              "num_nodes mismatch between preprocess and run: ",
              num_nodes,
              " vs ",
              entry.num_nodes);
  TORCH_CHECK(use_16x8 == entry.use_16x8,
              "SR-BCRS variant mismatch between preprocess and run");

  const int n = static_cast<int>(dense.size(1));
  if (n <= 0) {
    return {torch::zeros({entry.num_nodes, 0}, dense.options())};
  }

  int tile_b = 0;
  const char* tile_env = std::getenv("SRBCRS_TILE_B");
  if (tile_env != nullptr) {
    const std::string forced_tile = tile_env;
    if (forced_tile == "64") {
      TORCH_CHECK(n % 64 == 0,
                  "SRBCRS_TILE_B=64 requires dense feature dim divisible by 64, got ",
                  n);
      tile_b = 64;
    } else if (forced_tile == "32") {
      TORCH_CHECK(n % 32 == 0,
                  "SRBCRS_TILE_B=32 requires dense feature dim divisible by 32, got ",
                  n);
      tile_b = 32;
    } else if (forced_tile == "16") {
      TORCH_CHECK(n % 16 == 0,
                  "SRBCRS_TILE_B=16 requires dense feature dim divisible by 16, got ",
                  n);
      tile_b = 16;
    } else {
      TORCH_CHECK(forced_tile == "auto",
                  "Invalid SRBCRS_TILE_B=",
                  forced_tile,
                  ". Supported values: auto, 64, 32, 16");
    }
  }

  if (tile_b == 0) {
    if (n % 64 == 0) {
      tile_b = 64;
    } else if (n % 32 == 0) {
      tile_b = 32;
    } else if (n % 16 == 0) {
      tile_b = 16;
    }
  }

  TORCH_CHECK(tile_b != 0,
              "SR-BCRS requires dense feature dim divisible by 16/32/64, got ",
              n,
              ". Please pad feature dim before run.");
  TORCH_CHECK(entry.total_tcblock_num > 0,
              "SR-BCRS has empty preprocessed blocks; CSR fallback disabled.");

  auto dense_contig = dense.contiguous();
  auto out = torch::zeros({entry.num_nodes, n}, dense.options());

  auto stream = at::cuda::getDefaultCUDAStream();
  if (entry.use_16x8) {
    if (tile_b == 64) {
      srbcrs_runtime_internal::run_srbcrs_device<16, 64, 8>(entry.num_nodes,
                                                            n,
                                                            static_cast<int>(dense_contig.size(0)),
                                                            entry.total_tcblock_num,
                                                            entry.val_sidx_cuda.data_ptr<int>(),
                                                            entry.val_soff_cuda.data_ptr<int>(),
                                                            dense_contig.data_ptr<float>(),
                                                            entry.val_block_val_cuda.data_ptr<float>(),
                                                            out.data_ptr<float>(),
                                                            stream.stream());
    } else if (tile_b == 32) {
      srbcrs_runtime_internal::run_srbcrs_device<16, 32, 8>(entry.num_nodes,
                                                            n,
                                                            static_cast<int>(dense_contig.size(0)),
                                                            entry.total_tcblock_num,
                                                            entry.val_sidx_cuda.data_ptr<int>(),
                                                            entry.val_soff_cuda.data_ptr<int>(),
                                                            dense_contig.data_ptr<float>(),
                                                            entry.val_block_val_cuda.data_ptr<float>(),
                                                            out.data_ptr<float>(),
                                                            stream.stream());
    } else {
      srbcrs_runtime_internal::run_srbcrs_device<16, 16, 8>(entry.num_nodes,
                                                            n,
                                                            static_cast<int>(dense_contig.size(0)),
                                                            entry.total_tcblock_num,
                                                            entry.val_sidx_cuda.data_ptr<int>(),
                                                            entry.val_soff_cuda.data_ptr<int>(),
                                                            dense_contig.data_ptr<float>(),
                                                            entry.val_block_val_cuda.data_ptr<float>(),
                                                            out.data_ptr<float>(),
                                                            stream.stream());
    }
  } else {
    if (tile_b == 64) {
      srbcrs_runtime_internal::run_srbcrs_device<32, 64, 32>(entry.num_nodes,
                                                             n,
                                                             static_cast<int>(dense_contig.size(0)),
                                                             entry.total_tcblock_num,
                                                             entry.val_sidx_cuda.data_ptr<int>(),
                                                             entry.val_soff_cuda.data_ptr<int>(),
                                                             dense_contig.data_ptr<float>(),
                                                             entry.val_block_val_cuda.data_ptr<float>(),
                                                             out.data_ptr<float>(),
                                                             stream.stream());
    } else if (tile_b == 32) {
      srbcrs_runtime_internal::run_srbcrs_device<32, 32, 32>(entry.num_nodes,
                                                             n,
                                                             static_cast<int>(dense_contig.size(0)),
                                                             entry.total_tcblock_num,
                                                             entry.val_sidx_cuda.data_ptr<int>(),
                                                             entry.val_soff_cuda.data_ptr<int>(),
                                                             dense_contig.data_ptr<float>(),
                                                             entry.val_block_val_cuda.data_ptr<float>(),
                                                             out.data_ptr<float>(),
                                                             stream.stream());
    } else {
      srbcrs_runtime_internal::run_srbcrs_device<32, 16, 32>(entry.num_nodes,
                                                             n,
                                                             static_cast<int>(dense_contig.size(0)),
                                                             entry.total_tcblock_num,
                                                             entry.val_sidx_cuda.data_ptr<int>(),
                                                             entry.val_soff_cuda.data_ptr<int>(),
                                                             dense_contig.data_ptr<float>(),
                                                             entry.val_block_val_cuda.data_ptr<float>(),
                                                             out.data_ptr<float>(),
                                                             stream.stream());
    }
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out};
}
