#include <pybind11/stl.h>
#include <torch/extension.h>

#include "srbcrs_runtime_api.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
preprocess_common(torch::Tensor colind,
                  torch::Tensor rowptr,
                  torch::Tensor values,
                  int num_nodes,
                  int blk_h,
                  int blk_w,
                  torch::Tensor block_partition,
                  torch::Tensor edge_to_col,
                  torch::Tensor edge_to_row,
                  bool use_16x8) {
  CHECK_INPUT(colind);
  CHECK_INPUT(rowptr);
  CHECK_INPUT(values);
  CHECK_INPUT(block_partition);
  CHECK_INPUT(edge_to_col);
  CHECK_INPUT(edge_to_row);
  TORCH_CHECK(colind.scalar_type() == torch::kInt32, "colind must be int32");
  TORCH_CHECK(rowptr.scalar_type() == torch::kInt32, "rowptr must be int32");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");

  return preprocess_gpu_srcfp32_srbcrs(colind,
                                       rowptr,
                                       values,
                                       num_nodes,
                                       blk_h,
                                       blk_w,
                                       block_partition,
                                       edge_to_col,
                                       edge_to_row,
                                       use_16x8);
}

std::vector<torch::Tensor> run_common(torch::Tensor dense,
                                      torch::Tensor row_window_offset,
                                      torch::Tensor tcblocktile_id,
                                      torch::Tensor tcblock_offset,
                                      torch::Tensor sparse_a_to_x_idx,
                                      int num_nodes,
                                      int num_edges,
                                      std::string exeplan,
                                      bool use_16x8) {
  CHECK_INPUT(dense);
  CHECK_INPUT(row_window_offset);
  CHECK_INPUT(tcblocktile_id);
  CHECK_INPUT(tcblock_offset);
  CHECK_INPUT(sparse_a_to_x_idx);
  TORCH_CHECK(dense.scalar_type() == torch::kFloat32, "dense must be float32");

  return run_spmm_srcfp32_srbcrs(dense,
                                 row_window_offset,
                                 tcblocktile_id,
                                 tcblock_offset,
                                 sparse_a_to_x_idx,
                                 num_nodes,
                                 num_edges,
                                 exeplan,
                                 use_16x8);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
preprocess_gpu(torch::Tensor colind,
               torch::Tensor rowptr,
               torch::Tensor values,
               int num_nodes,
               int blk_h,
               int blk_w,
               torch::Tensor block_partition,
               torch::Tensor edge_to_col,
               torch::Tensor edge_to_row) {
  return preprocess_common(colind,
                           rowptr,
                           values,
                           num_nodes,
                           blk_h,
                           blk_w,
                           block_partition,
                           edge_to_col,
                           edge_to_row,
                           false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
preprocess_gpu_16x8(torch::Tensor colind,
                    torch::Tensor rowptr,
                    torch::Tensor values,
                    int num_nodes,
                    int blk_h,
                    int blk_w,
                    torch::Tensor block_partition,
                    torch::Tensor edge_to_col,
                    torch::Tensor edge_to_row) {
  return preprocess_common(colind,
                           rowptr,
                           values,
                           num_nodes,
                           blk_h,
                           blk_w,
                           block_partition,
                           edge_to_col,
                           edge_to_row,
                           true);
}

std::vector<torch::Tensor> run_SRBCRS(torch::Tensor dense,
                                      torch::Tensor row_window_offset,
                                      torch::Tensor tcblocktile_id,
                                      torch::Tensor tcblock_offset,
                                      torch::Tensor sparse_a_to_x_idx,
                                      int num_nodes,
                                      int num_edges,
                                      std::string exeplan) {
  return run_common(dense,
                    row_window_offset,
                    tcblocktile_id,
                    tcblock_offset,
                    sparse_a_to_x_idx,
                    num_nodes,
                    num_edges,
                    exeplan,
                    false);
}

std::vector<torch::Tensor> run_SRBCRS_balance(torch::Tensor dense,
                                              torch::Tensor row_window_offset,
                                              torch::Tensor tcblocktile_id,
                                              torch::Tensor tcblock_offset,
                                              torch::Tensor sparse_a_to_x_idx,
                                              int num_nodes,
                                              std::string exeplan) {
  return run_common(dense,
                    row_window_offset,
                    tcblocktile_id,
                    tcblock_offset,
                    sparse_a_to_x_idx,
                    num_nodes,
                    static_cast<int>(sparse_a_to_x_idx.numel()),
                    exeplan,
                    false);
}

std::vector<torch::Tensor> run_SRBCRS_16x8(torch::Tensor dense,
                                           torch::Tensor row_window_offset,
                                           torch::Tensor tcblocktile_id,
                                           torch::Tensor tcblock_offset,
                                           torch::Tensor sparse_a_to_x_idx,
                                           int num_nodes,
                                           int num_edges,
                                           std::string exeplan) {
  return run_common(dense,
                    row_window_offset,
                    tcblocktile_id,
                    tcblock_offset,
                    sparse_a_to_x_idx,
                    num_nodes,
                    num_edges,
                    exeplan,
                    true);
}

std::vector<torch::Tensor> run_SRBCRS_16x8_balance(torch::Tensor dense,
                                                   torch::Tensor row_window_offset,
                                                   torch::Tensor tcblocktile_id,
                                                   torch::Tensor tcblock_offset,
                                                   torch::Tensor sparse_a_to_x_idx,
                                                   int num_nodes,
                                                   std::string exeplan) {
  return run_common(dense,
                    row_window_offset,
                    tcblocktile_id,
                    tcblock_offset,
                    sparse_a_to_x_idx,
                    num_nodes,
                    static_cast<int>(sparse_a_to_x_idx.numel()),
                    exeplan,
                    true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_gpu", &preprocess_gpu, "SR-BCRS preprocess base (CUDA)");
  m.def("preprocess_gpu_16x8", &preprocess_gpu_16x8, "SR-BCRS preprocess 16x8 (CUDA)");

  m.def("run_SRBCRS", &run_SRBCRS, "SR-BCRS run base (CUDA)");
  m.def("run_SRBCRS_balance", &run_SRBCRS_balance, "SR-BCRS run base balance alias (CUDA)");
  m.def("run_SRBCRS_16x8", &run_SRBCRS_16x8, "SR-BCRS run 16x8 (CUDA)");
  m.def("run_SRBCRS_16x8_balance",
        &run_SRBCRS_16x8_balance,
        "SR-BCRS run 16x8 balance alias (CUDA)");
}
