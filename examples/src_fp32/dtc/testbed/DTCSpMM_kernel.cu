#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cute/tensor.hpp>

#include "dtc_runtime_api.h"

template <int TILE_B>
void dtc_spmm_test_val_sidx_bind_reorder(int M,
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

void run_spmm_device_srcfp32_64(int M,
                                int N,
                                int K,
                                int TCBlock_num,
                                int nnz,
                                int* dval_sidx,
                                int* dval_soff,
                                float* dB_val,
                                int* dval_coo_idx,
                                float* dval_coo_val,
                                int* dval_coo_off,
                                float* dC,
                                cudaStream_t stream);

void run_spmm_device_srcfp32_16(int M,
                                int N,
                                int K,
                                int TCBlock_num,
                                int nnz,
                                int* dval_sidx,
                                int* dval_soff,
                                float* dB_val,
                                int* dval_coo_idx,
                                float* dval_coo_val,
                                int* dval_coo_off,
                                float* dC,
                                cudaStream_t stream);

void run_spmm_device_srcfp32_32(int M,
                                int N,
                                int K,
                                int TCBlock_num,
                                int nnz,
                                int* dval_sidx,
                                int* dval_soff,
                                float* dB_val,
                                int* dval_coo_idx,
                                float* dval_coo_val,
                                int* dval_coo_off,
                                float* dC,
                                cudaStream_t stream);

void run_spmm_device_srcfp32_64_strict_lb(int M,
                                          int N,
                                          int K,
                                          int TCBlock_num,
                                          int nnz,
                                          int* dval_sidx,
                                          int* dval_soff,
                                          float* dB_val,
                                          int* dval_coo_idx,
                                          float* dval_coo_val,
                                          int* dval_coo_off,
                                          int row_block_num,
                                          int slb_block_dim_size,
                                          int* drow_block_offset,
                                          int* drow_block_to_row_window_id,
                                          int* dslb_tb_offset,
                                          unsigned long long* dsm_cycles,
                                          int* dblock_smid,
                                          float* dC,
                                          cudaStream_t stream);

void run_spmm_device_srcfp32_16_strict_lb(int M,
                                          int N,
                                          int K,
                                          int TCBlock_num,
                                          int nnz,
                                          int* dval_sidx,
                                          int* dval_soff,
                                          float* dB_val,
                                          int* dval_coo_idx,
                                          float* dval_coo_val,
                                          int* dval_coo_off,
                                          int row_block_num,
                                          int slb_block_dim_size,
                                          int* drow_block_offset,
                                          int* drow_block_to_row_window_id,
                                          int* dslb_tb_offset,
                                          unsigned long long* dsm_cycles,
                                          int* dblock_smid,
                                          float* dC,
                                          cudaStream_t stream);

void run_spmm_device_srcfp32_32_strict_lb(int M,
                                          int N,
                                          int K,
                                          int TCBlock_num,
                                          int nnz,
                                          int* dval_sidx,
                                          int* dval_soff,
                                          float* dB_val,
                                          int* dval_coo_idx,
                                          float* dval_coo_val,
                                          int* dval_coo_off,
                                          int row_block_num,
                                          int slb_block_dim_size,
                                          int* drow_block_offset,
                                          int* drow_block_to_row_window_id,
                                          int* dslb_tb_offset,
                                          unsigned long long* dsm_cycles,
                                          int* dblock_smid,
                                          float* dC,
                                          cudaStream_t stream);

namespace dtc_runtime_internal {

constexpr int kRowWindowM = 16;
constexpr int kTcBlockK = 8;

struct CacheEntry {
  int num_nodes = 0;
  int nnz = 0;
  int total_tcblock_num = 0;
  int nnz_pad = 0;
  int row_block_num = 0;
  int slb_block_dim_size = 0;
  bool strict_lb_ready = false;

  torch::Tensor rowptr_cuda;
  torch::Tensor colind_cuda;

  torch::Tensor val_sidx_cuda;
  torch::Tensor val_soff_cuda;
  torch::Tensor val_coo_idx_cuda;
  torch::Tensor val_coo_val_cuda;
  torch::Tensor val_coo_off_cuda;

  torch::Tensor row_block_offset_cuda;
  torch::Tensor row_block_to_row_window_id_cuda;
  torch::Tensor slb_tb_offset_cuda;
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

int binary_search_vec(const std::vector<int>& arr, int target) {
  int l = 0;
  int r = static_cast<int>(arr.size()) - 1;
  while (l <= r) {
    const int mid = l + (r - l) / 2;
    if (arr[mid] == target) {
      return mid;
    }
    if (arr[mid] < target) {
      l = mid + 1;
    } else {
      r = mid - 1;
    }
  }
  return -1;
}

void strict_load_balance_from_soff(const std::vector<int>& val_soff,
                                   int total_tcblock_num,
                                   int& row_block_num,
                                   int& slb_block_dim_size,
                                   std::vector<int>& row_block_offset,
                                   std::vector<int>& row_block_to_row_window_id,
                                   std::vector<int>& slb_tb_offset) {
  constexpr int thread_block_workload = 64;
  const int row_window_num = static_cast<int>(val_soff.size()) - 1;

  TORCH_CHECK(row_window_num >= 0, "Invalid val_soff for strict lb");
  slb_block_dim_size = (total_tcblock_num + thread_block_workload - 1) / thread_block_workload;

  int current_workload = 0;
  int row_block_start = -1;
  int current_row_block_ptr = 0;
  int tc_block_ptr = 0;

  slb_tb_offset.clear();
  row_block_offset.clear();
  row_block_to_row_window_id.clear();
  slb_tb_offset.push_back(current_row_block_ptr);

  for (int row_window_i = 0; row_window_i < row_window_num; ++row_window_i) {
    int current_row_window_workload = val_soff[row_window_i + 1] - val_soff[row_window_i];
    if (current_row_window_workload == 0) {
      continue;
    }
    if (row_block_start == -1) {
      row_block_start = 0;
    }

    while (current_row_window_workload > 0) {
      if (current_workload + current_row_window_workload < thread_block_workload) {
        row_block_offset.push_back(tc_block_ptr);
        row_block_to_row_window_id.push_back(row_window_i);

        tc_block_ptr += current_row_window_workload;
        current_row_block_ptr++;
        row_block_start = -1;
        current_workload += current_row_window_workload;
        current_row_window_workload = 0;
      } else if (current_workload + current_row_window_workload == thread_block_workload) {
        row_block_offset.push_back(tc_block_ptr);
        row_block_to_row_window_id.push_back(row_window_i);

        tc_block_ptr += current_row_window_workload;
        current_row_block_ptr++;
        row_block_start = -1;
        current_workload = 0;
        current_row_window_workload = 0;
        slb_tb_offset.push_back(current_row_block_ptr);
      } else {
        row_block_offset.push_back(tc_block_ptr);
        row_block_to_row_window_id.push_back(row_window_i);

        current_row_block_ptr++;
        const int consume_tc_block_num = thread_block_workload - current_workload;
        row_block_start += consume_tc_block_num;
        current_row_window_workload -= consume_tc_block_num;
        current_workload = 0;
        slb_tb_offset.push_back(current_row_block_ptr);
        tc_block_ptr += consume_tc_block_num;
      }
    }
  }

  row_block_offset.push_back(tc_block_ptr);
  if (total_tcblock_num % thread_block_workload != 0) {
    current_row_block_ptr++;
    slb_tb_offset.push_back(current_row_block_ptr);
  }

  row_block_num = static_cast<int>(row_block_to_row_window_id.size());
  TORCH_CHECK(static_cast<int>(slb_tb_offset.size()) == slb_block_dim_size + 1,
              "strict lb slb_tb_offset size mismatch");
  TORCH_CHECK(static_cast<int>(row_block_offset.size()) == row_block_num + 1,
              "strict lb row_block_offset size mismatch");
  TORCH_CHECK(row_block_offset[row_block_num] == val_soff[row_window_num],
              "strict lb total tcblock mismatch");
}

__global__ void csr_spmm_unit_kernel(const int* rowptr,
                                     const int* colind,
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
    acc += dense[col * n + feat];
  }
  out[row * n + feat] = acc;
}

torch::Tensor fallback_spmm_from_csr(const CacheEntry& entry, const torch::Tensor& dense) {
  const int64_t n = dense.size(1);
  auto out = torch::zeros({entry.num_nodes, n}, dense.options());

  const dim3 block(256);
  const dim3 grid(entry.num_nodes, static_cast<unsigned int>((n + block.x - 1) / block.x));
  auto stream = at::cuda::getDefaultCUDAStream();
  csr_spmm_unit_kernel<<<grid, block, 0, stream>>>(entry.rowptr_cuda.data_ptr<int>(),
                                                   entry.colind_cuda.data_ptr<int>(),
                                                   dense.data_ptr<float>(),
                                                   out.data_ptr<float>(),
                                                   entry.num_nodes,
                                                   static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace dtc_runtime_internal

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
preprocess_gpu_srcfp32(torch::Tensor colind_i32,
                       torch::Tensor rowptr_i32,
                       int num_nodes,
                       int blk_h,
                       int blk_w,
                       torch::Tensor block_partition_i32,
                       torch::Tensor edge_to_col_i32,
                       torch::Tensor edge_to_row_i32) {
  TORCH_CHECK(colind_i32.is_cuda(), "colind must be CUDA tensor");
  TORCH_CHECK(rowptr_i32.is_cuda(), "rowptr must be CUDA tensor");
  TORCH_CHECK(colind_i32.scalar_type() == torch::kInt32, "colind must be int32");
  TORCH_CHECK(rowptr_i32.scalar_type() == torch::kInt32, "rowptr must be int32");
  TORCH_CHECK(blk_h == dtc_runtime_internal::kRowWindowM,
              "Only blk_h=16 is supported by current src_fp32 kernel");
  TORCH_CHECK(blk_w == dtc_runtime_internal::kTcBlockK,
              "Only blk_w=8 is supported by current src_fp32 kernel");
  TORCH_CHECK(num_nodes >= 1, "num_nodes must be positive");

  auto rowptr_cpu_t = rowptr_i32.to(torch::kCPU).contiguous();
  auto colind_cpu_t = colind_i32.to(torch::kCPU).contiguous();
  auto rowptr = rowptr_cpu_t.data_ptr<int>();
  auto colind = colind_cpu_t.data_ptr<int>();

  const int nnz = static_cast<int>(colind_i32.numel());
  const int row_window_num =
      (num_nodes + dtc_runtime_internal::kRowWindowM - 1) / dtc_runtime_internal::kRowWindowM;

  std::vector<int> val_soff(row_window_num + 1, 0);
  std::vector<int> block_partition_cpu(row_window_num, 0);
  std::vector<int> edge_to_row_cpu(nnz, 0);
  std::vector<int> edge_to_col_cpu(nnz, 0);

  std::vector<int> val_sidx;
  std::vector<int> val_coo_off;
  std::vector<int> val_coo_idx;
  std::vector<float> val_coo_val;

  int total_tcblock_num = 0;
  int nnz_pad = 0;

  auto swizzle_layout =
      cute::composition(cute::Swizzle<1, 2, 3>{},
                        cute::make_layout(cute::make_shape(dtc_runtime_internal::kRowWindowM,
                                                           dtc_runtime_internal::kTcBlockK),
                                          cute::make_stride(dtc_runtime_internal::kTcBlockK, 1)));

  for (int w = 0; w < row_window_num; ++w) {
    val_soff[w] = total_tcblock_num;

    std::vector<int> unique_cols;
    unique_cols.reserve(256);
    for (int i_row = 0; i_row < dtc_runtime_internal::kRowWindowM; ++i_row) {
      const int row = w * dtc_runtime_internal::kRowWindowM + i_row;
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
    const int tcblock_num =
      (nnz_col_num + dtc_runtime_internal::kTcBlockK - 1) / dtc_runtime_internal::kTcBlockK;
    block_partition_cpu[w] = tcblock_num;

    std::vector<int> nnz_per_tcblock(tcblock_num, 0);
    for (int i_row = 0; i_row < dtc_runtime_internal::kRowWindowM; ++i_row) {
      const int row = w * dtc_runtime_internal::kRowWindowM + i_row;
      if (row >= num_nodes) {
        continue;
      }
      for (int e = rowptr[row]; e < rowptr[row + 1]; ++e) {
        const int condensed = dtc_runtime_internal::binary_search_vec(unique_cols, colind[e]);
        TORCH_CHECK(condensed >= 0, "Internal preprocess error: column not found in unique set");
        edge_to_col_cpu[e] = condensed;
        nnz_per_tcblock[condensed / dtc_runtime_internal::kTcBlockK] += 1;
      }
    }

    const int sidx_base = static_cast<int>(val_sidx.size());
    val_sidx.resize(sidx_base + tcblock_num * dtc_runtime_internal::kTcBlockK, 0);
    for (int i = 0; i < nnz_col_num; ++i) {
      val_sidx[sidx_base + i] = unique_cols[i];
    }

    std::vector<int> block_begin(tcblock_num, 0);
    std::vector<int> block_end(tcblock_num, 0);
    std::vector<int> block_ptr(tcblock_num, 0);
    for (int b = 0; b < tcblock_num; ++b) {
      block_begin[b] = static_cast<int>(val_coo_idx.size());
      const int padded = ((nnz_per_tcblock[b] + 7) / 8) * 8;
      block_end[b] = block_begin[b] + padded;
      block_ptr[b] = block_begin[b];

      val_coo_idx.resize(block_end[b], 0);
      val_coo_val.resize(block_end[b], 0.0f);
      nnz_pad += padded;
    }

    for (int i_row = 0; i_row < dtc_runtime_internal::kRowWindowM; ++i_row) {
      const int row = w * dtc_runtime_internal::kRowWindowM + i_row;
      if (row >= num_nodes) {
        continue;
      }
      for (int e = rowptr[row]; e < rowptr[row + 1]; ++e) {
        const int condensed = edge_to_col_cpu[e];
        const int tcblock_id = condensed / dtc_runtime_internal::kTcBlockK;
        const auto coord = cute::make_coord(i_row, condensed % dtc_runtime_internal::kTcBlockK);
        const int swizzled_idx = swizzle_layout(coord);

        const int dst = block_ptr[tcblock_id]++;
        val_coo_idx[dst] = swizzled_idx;
        val_coo_val[dst] = 1.0f;
      }
    }

    for (int b = 0; b < tcblock_num; ++b) {
      val_coo_off.push_back(block_begin[b]);
      val_coo_off.push_back(block_ptr[b]);
    }

    total_tcblock_num += tcblock_num;
  }
  val_soff[row_window_num] = total_tcblock_num;
  val_coo_off.push_back(static_cast<int>(val_coo_idx.size()));

  auto opts_gpu_i32 = torch::TensorOptions().dtype(torch::kInt32).device(colind_i32.device());

  torch::Tensor row_window_offset =
    dtc_runtime_internal::vector_i32_to_cuda(val_soff, colind_i32.device());
  torch::Tensor tcblock_rowid = torch::zeros({total_tcblock_num}, opts_gpu_i32);
  torch::Tensor tcblocktile_id =
    dtc_runtime_internal::vector_i32_to_cuda(val_sidx, colind_i32.device());
  torch::Tensor tcblock_offset =
    dtc_runtime_internal::vector_i32_to_cuda(val_coo_off, colind_i32.device());
  torch::Tensor sparse_a_to_x_idx =
    dtc_runtime_internal::vector_i32_to_cuda(val_coo_idx, colind_i32.device());
  torch::Tensor tcblock_val =
    dtc_runtime_internal::vector_f32_to_cuda(val_coo_val, colind_i32.device());

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
    std::lock_guard<std::mutex> guard(dtc_runtime_internal::g_cache_mu);
    auto& entry = dtc_runtime_internal::g_cache[key];
    entry.num_nodes = num_nodes;
    entry.nnz = nnz;
    entry.total_tcblock_num = total_tcblock_num;
    entry.nnz_pad = nnz_pad;
    entry.rowptr_cuda = rowptr_i32.contiguous();
    entry.colind_cuda = colind_i32.contiguous();
    entry.val_sidx_cuda = tcblocktile_id;
    entry.val_soff_cuda = row_window_offset;
    entry.val_coo_idx_cuda = sparse_a_to_x_idx;
    entry.val_coo_val_cuda = tcblock_val;
    entry.val_coo_off_cuda = tcblock_offset;
    entry.row_block_num = 0;
    entry.slb_block_dim_size = 0;
    entry.strict_lb_ready = false;
    entry.row_block_offset_cuda = torch::Tensor();
    entry.row_block_to_row_window_id_cuda = torch::Tensor();
    entry.slb_tb_offset_cuda = torch::Tensor();
  }

  return std::make_tuple(row_window_offset,
                         tcblock_rowid,
                         tcblocktile_id,
                         tcblock_offset,
                         sparse_a_to_x_idx,
                         total_tcblock_num);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
preprocess_gpu_srcfp32_strict_lb(torch::Tensor colind_i32,
                                 torch::Tensor rowptr_i32,
                                 int num_nodes,
                                 int blk_h,
                                 int blk_w,
                                 torch::Tensor block_partition_i32,
                                 torch::Tensor edge_to_col_i32,
                                 torch::Tensor edge_to_row_i32) {
  auto prep = preprocess_gpu_srcfp32(colind_i32,
                                     rowptr_i32,
                                     num_nodes,
                                     blk_h,
                                     blk_w,
                                     block_partition_i32,
                                     edge_to_col_i32,
                                     edge_to_row_i32);

  auto row_window_offset = std::get<0>(prep);
  const int total_tcblock_num = std::get<5>(prep);

  auto val_soff_cpu_t = row_window_offset.to(torch::kCPU).contiguous();
  const int* val_soff_ptr = val_soff_cpu_t.data_ptr<int>();
  std::vector<int> val_soff(val_soff_ptr, val_soff_ptr + val_soff_cpu_t.numel());

  int row_block_num = 0;
  int slb_block_dim_size = 0;
  std::vector<int> row_block_offset;
  std::vector<int> row_block_to_row_window_id;
  std::vector<int> slb_tb_offset;
  dtc_runtime_internal::strict_load_balance_from_soff(val_soff,
                                                      total_tcblock_num,
                                                      row_block_num,
                                                      slb_block_dim_size,
                                                      row_block_offset,
                                                      row_block_to_row_window_id,
                                                      slb_tb_offset);

  auto row_block_offset_cuda =
      dtc_runtime_internal::vector_i32_to_cuda(row_block_offset, colind_i32.device());
  auto row_block_to_row_window_id_cuda =
      dtc_runtime_internal::vector_i32_to_cuda(row_block_to_row_window_id, colind_i32.device());
  auto slb_tb_offset_cuda =
      dtc_runtime_internal::vector_i32_to_cuda(slb_tb_offset, colind_i32.device());

  const int64_t key = reinterpret_cast<int64_t>(row_window_offset.data_ptr<int>());
  {
    std::lock_guard<std::mutex> guard(dtc_runtime_internal::g_cache_mu);
    auto it = dtc_runtime_internal::g_cache.find(key);
    TORCH_CHECK(it != dtc_runtime_internal::g_cache.end(),
                "strict lb preprocess cache entry missing");
    auto& entry = it->second;
    entry.row_block_num = row_block_num;
    entry.slb_block_dim_size = slb_block_dim_size;
    entry.strict_lb_ready = true;
    entry.row_block_offset_cuda = row_block_offset_cuda;
    entry.row_block_to_row_window_id_cuda = row_block_to_row_window_id_cuda;
    entry.slb_tb_offset_cuda = slb_tb_offset_cuda;
  }

  return prep;
}

std::vector<torch::Tensor> run_spmm_srcfp32(torch::Tensor dense,
                                             torch::Tensor row_window_offset,
                                             torch::Tensor tcblocktile_id,
                                             torch::Tensor tcblock_offset,
                                             torch::Tensor sparse_a_to_x_idx,
                                             int num_nodes,
                                             int /*num_edges*/,
                                             std::string /*exeplan*/) {
  TORCH_CHECK(dense.is_cuda(), "dense must be CUDA tensor");
  TORCH_CHECK(dense.scalar_type() == torch::kFloat32, "dense must be float32");
  TORCH_CHECK(dense.dim() == 2, "dense must be 2-D");

  const int64_t key = reinterpret_cast<int64_t>(row_window_offset.data_ptr<int>());

  dtc_runtime_internal::CacheEntry entry;
  bool has_entry = false;
  {
    std::lock_guard<std::mutex> guard(dtc_runtime_internal::g_cache_mu);
    auto it = dtc_runtime_internal::g_cache.find(key);
    if (it != dtc_runtime_internal::g_cache.end()) {
      entry = it->second;
      has_entry = true;
    }
  }

  TORCH_CHECK(has_entry, "run_DTCSpMM cache miss. Call preprocess_gpu before run_DTCSpMM.");
  TORCH_CHECK(num_nodes == entry.num_nodes,
              "num_nodes mismatch between preprocess and run: ",
              num_nodes,
              " vs ",
              entry.num_nodes);

  const int n = static_cast<int>(dense.size(1));
  if (n <= 0) {
    return {torch::zeros({entry.num_nodes, 0}, dense.options())};
  }

  int tile_b = 0;
  const char* tile_env = std::getenv("DTC_TILE_B");
  if (tile_env != nullptr) {
    const std::string forced_tile = tile_env;
    if (forced_tile == "64") {
      TORCH_CHECK(n % 64 == 0,
                  "DTC_TILE_B=64 requires dense feature dim divisible by 64, got ",
                  n);
      tile_b = 64;
    } else if (forced_tile == "32") {
      TORCH_CHECK(n % 32 == 0,
                  "DTC_TILE_B=32 requires dense feature dim divisible by 32, got ",
                  n);
      tile_b = 32;
    } else if (forced_tile == "16") {
      TORCH_CHECK(n % 16 == 0,
                  "DTC_TILE_B=16 requires dense feature dim divisible by 16, got ",
                  n);
      tile_b = 16;
    } else {
      TORCH_CHECK(forced_tile == "auto",
                  "Invalid DTC_TILE_B=",
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
              "DTCSpMM requires dense feature dim divisible by 16/32/64, got ",
              n,
              ". Please pad feature dim before run.");
  TORCH_CHECK(entry.total_tcblock_num > 0 && entry.nnz_pad > 0,
              "DTCSpMM has empty preprocessed blocks; CSR fallback disabled.");

  TORCH_CHECK(entry.val_sidx_cuda.is_cuda() && entry.val_soff_cuda.is_cuda() &&
                entry.val_coo_idx_cuda.is_cuda() && entry.val_coo_val_cuda.is_cuda() &&
                entry.val_coo_off_cuda.is_cuda(),
              "Cached preprocess tensors must reside on CUDA device");

  auto dense_contig = dense.contiguous();
  auto out = torch::zeros({entry.num_nodes, n}, dense.options());

  auto stream = at::cuda::getDefaultCUDAStream();
  if (tile_b == 64) {
    run_spmm_device_srcfp32_64(entry.num_nodes,
                               n,
                               entry.num_nodes,
                               entry.total_tcblock_num,
                               entry.nnz_pad,
                               entry.val_sidx_cuda.data_ptr<int>(),
                               entry.val_soff_cuda.data_ptr<int>(),
                               dense_contig.data_ptr<float>(),
                               entry.val_coo_idx_cuda.data_ptr<int>(),
                               entry.val_coo_val_cuda.data_ptr<float>(),
                               entry.val_coo_off_cuda.data_ptr<int>(),
                               out.data_ptr<float>(),
                               stream.stream());
  } else if (tile_b == 32) {
    run_spmm_device_srcfp32_32(entry.num_nodes,
                               n,
                               entry.num_nodes,
                               entry.total_tcblock_num,
                               entry.nnz_pad,
                               entry.val_sidx_cuda.data_ptr<int>(),
                               entry.val_soff_cuda.data_ptr<int>(),
                               dense_contig.data_ptr<float>(),
                               entry.val_coo_idx_cuda.data_ptr<int>(),
                               entry.val_coo_val_cuda.data_ptr<float>(),
                               entry.val_coo_off_cuda.data_ptr<int>(),
                               out.data_ptr<float>(),
                               stream.stream());
  } else {
    run_spmm_device_srcfp32_16(entry.num_nodes,
                               n,
                               entry.num_nodes,
                               entry.total_tcblock_num,
                               entry.nnz_pad,
                               entry.val_sidx_cuda.data_ptr<int>(),
                               entry.val_soff_cuda.data_ptr<int>(),
                               dense_contig.data_ptr<float>(),
                               entry.val_coo_idx_cuda.data_ptr<int>(),
                               entry.val_coo_val_cuda.data_ptr<float>(),
                               entry.val_coo_off_cuda.data_ptr<int>(),
                               out.data_ptr<float>(),
                               stream.stream());
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out};
}

std::vector<torch::Tensor> run_spmm_srcfp32_strict_lb(torch::Tensor dense,
                                                       torch::Tensor row_window_offset,
                                                       torch::Tensor tcblocktile_id,
                                                       torch::Tensor tcblock_offset,
                                                       torch::Tensor sparse_a_to_x_idx,
                                                       int num_nodes,
                                                       int num_edges,
                                                       std::string exeplan) {
  TORCH_CHECK(dense.is_cuda(), "dense must be CUDA tensor");
  TORCH_CHECK(dense.scalar_type() == torch::kFloat32, "dense must be float32");
  TORCH_CHECK(dense.dim() == 2, "dense must be 2-D");

  const int64_t key = reinterpret_cast<int64_t>(row_window_offset.data_ptr<int>());

  dtc_runtime_internal::CacheEntry entry;
  bool has_entry = false;
  {
    std::lock_guard<std::mutex> guard(dtc_runtime_internal::g_cache_mu);
    auto it = dtc_runtime_internal::g_cache.find(key);
    if (it != dtc_runtime_internal::g_cache.end()) {
      entry = it->second;
      has_entry = true;
    }
  }

  TORCH_CHECK(has_entry, "run_DTCSpMM_strict_lb cache miss. Call preprocess first.");
  TORCH_CHECK(num_nodes == entry.num_nodes,
              "num_nodes mismatch between preprocess and strict_lb run: ",
              num_nodes,
              " vs ",
              entry.num_nodes);

  if (!entry.strict_lb_ready) {
    return run_spmm_srcfp32(dense,
                            row_window_offset,
                            tcblocktile_id,
                            tcblock_offset,
                            sparse_a_to_x_idx,
                            num_nodes,
                            num_edges,
                            exeplan);
  }

  const int n = static_cast<int>(dense.size(1));
  if (n <= 0) {
    return {torch::zeros({entry.num_nodes, 0}, dense.options())};
  }

  int tile_b = 0;
  const char* strict_lb_tile_env = std::getenv("DTC_STRICT_LB_TILE_B");
  if (strict_lb_tile_env == nullptr) {
    strict_lb_tile_env = std::getenv("DTC_TILE_B");
  }
  if (strict_lb_tile_env != nullptr) {
    const std::string forced_tile = strict_lb_tile_env;
    if (forced_tile == "64") {
      TORCH_CHECK(n % 64 == 0,
                  "DTC_STRICT_LB_TILE_B=64 requires dense feature dim divisible by 64, got ",
                  n);
      tile_b = 64;
    } else if (forced_tile == "32") {
      TORCH_CHECK(n % 32 == 0,
                  "DTC_STRICT_LB_TILE_B=32 requires dense feature dim divisible by 32, got ",
                  n);
      tile_b = 32;
    } else if (forced_tile == "16") {
      TORCH_CHECK(n % 16 == 0,
                  "DTC_STRICT_LB_TILE_B=16 requires dense feature dim divisible by 16, got ",
                  n);
      tile_b = 16;
    } else {
      TORCH_CHECK(forced_tile == "auto",
                  "Invalid DTC_STRICT_LB_TILE_B=",
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
              "DTC strict-lb requires dense feature dim divisible by 16/32/64, got ",
              n,
              ". Please pad feature dim before run.");
  TORCH_CHECK(entry.total_tcblock_num > 0 && entry.nnz_pad > 0,
              "DTC strict-lb has empty preprocessed blocks; CSR fallback disabled.");
  TORCH_CHECK(entry.row_block_num > 0 && entry.slb_block_dim_size > 0,
              "DTC strict-lb metadata invalid; CSR fallback disabled.");

  TORCH_CHECK(entry.row_block_offset_cuda.is_cuda() &&
                entry.row_block_to_row_window_id_cuda.is_cuda() &&
                entry.slb_tb_offset_cuda.is_cuda(),
              "Strict lb metadata must reside on CUDA device");

  auto dense_contig = dense.contiguous();
  const int row_window_num =
      (entry.num_nodes + dtc_runtime_internal::kRowWindowM - 1) / dtc_runtime_internal::kRowWindowM;
  const int padded_m = row_window_num * dtc_runtime_internal::kRowWindowM;
  auto out_padded = torch::zeros({padded_m, n}, dense.options());
  auto sm_cycles = torch::zeros({256}, torch::TensorOptions().dtype(torch::kInt64).device(dense.device()));
  auto block_smid = torch::full({entry.slb_block_dim_size * (n / tile_b)},
                                -1,
                                torch::TensorOptions().dtype(torch::kInt32).device(dense.device()));

  auto stream = at::cuda::getDefaultCUDAStream();
  if (tile_b == 64) {
    run_spmm_device_srcfp32_64_strict_lb(entry.num_nodes,
                                         n,
                                         entry.num_nodes,
                                         entry.total_tcblock_num,
                                         entry.nnz_pad,
                                         entry.val_sidx_cuda.data_ptr<int>(),
                                         entry.val_soff_cuda.data_ptr<int>(),
                                         dense_contig.data_ptr<float>(),
                                         entry.val_coo_idx_cuda.data_ptr<int>(),
                                         entry.val_coo_val_cuda.data_ptr<float>(),
                                         entry.val_coo_off_cuda.data_ptr<int>(),
                                         entry.row_block_num,
                                         entry.slb_block_dim_size,
                                         entry.row_block_offset_cuda.data_ptr<int>(),
                                         entry.row_block_to_row_window_id_cuda.data_ptr<int>(),
                                         entry.slb_tb_offset_cuda.data_ptr<int>(),
                                         reinterpret_cast<unsigned long long*>(sm_cycles.data_ptr<int64_t>()),
                                         block_smid.data_ptr<int>(),
                                         out_padded.data_ptr<float>(),
                                         stream.stream());
  } else if (tile_b == 32) {
    run_spmm_device_srcfp32_32_strict_lb(entry.num_nodes,
                                         n,
                                         entry.num_nodes,
                                         entry.total_tcblock_num,
                                         entry.nnz_pad,
                                         entry.val_sidx_cuda.data_ptr<int>(),
                                         entry.val_soff_cuda.data_ptr<int>(),
                                         dense_contig.data_ptr<float>(),
                                         entry.val_coo_idx_cuda.data_ptr<int>(),
                                         entry.val_coo_val_cuda.data_ptr<float>(),
                                         entry.val_coo_off_cuda.data_ptr<int>(),
                                         entry.row_block_num,
                                         entry.slb_block_dim_size,
                                         entry.row_block_offset_cuda.data_ptr<int>(),
                                         entry.row_block_to_row_window_id_cuda.data_ptr<int>(),
                                         entry.slb_tb_offset_cuda.data_ptr<int>(),
                                         reinterpret_cast<unsigned long long*>(sm_cycles.data_ptr<int64_t>()),
                                         block_smid.data_ptr<int>(),
                                         out_padded.data_ptr<float>(),
                                         stream.stream());
  } else {
    run_spmm_device_srcfp32_16_strict_lb(entry.num_nodes,
                                         n,
                                         entry.num_nodes,
                                         entry.total_tcblock_num,
                                         entry.nnz_pad,
                                         entry.val_sidx_cuda.data_ptr<int>(),
                                         entry.val_soff_cuda.data_ptr<int>(),
                                         dense_contig.data_ptr<float>(),
                                         entry.val_coo_idx_cuda.data_ptr<int>(),
                                         entry.val_coo_val_cuda.data_ptr<float>(),
                                         entry.val_coo_off_cuda.data_ptr<int>(),
                                         entry.row_block_num,
                                         entry.slb_block_dim_size,
                                         entry.row_block_offset_cuda.data_ptr<int>(),
                                         entry.row_block_to_row_window_id_cuda.data_ptr<int>(),
                                         entry.slb_tb_offset_cuda.data_ptr<int>(),
                                         reinterpret_cast<unsigned long long*>(sm_cycles.data_ptr<int64_t>()),
                                         block_smid.data_ptr<int>(),
                                         out_padded.data_ptr<float>(),
                                         stream.stream());
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto out = out_padded.narrow(0, 0, entry.num_nodes).contiguous();
  return {out};
}
