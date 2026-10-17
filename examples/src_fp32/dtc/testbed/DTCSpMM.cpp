#include <pybind11/stl.h>
#include <torch/extension.h>

#include "dtc_runtime_api.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
preprocess_gpu(torch::Tensor colind,
               torch::Tensor rowptr,
               int num_nodes,
               int blk_h,
               int blk_w,
               torch::Tensor block_partition,
               torch::Tensor edge_to_col,
               torch::Tensor edge_to_row) {
  CHECK_INPUT(colind);
  CHECK_INPUT(rowptr);
  CHECK_INPUT(block_partition);
  CHECK_INPUT(edge_to_col);
  CHECK_INPUT(edge_to_row);
  TORCH_CHECK(colind.scalar_type() == torch::kInt32, "colind must be int32");
  TORCH_CHECK(rowptr.scalar_type() == torch::kInt32, "rowptr must be int32");

  return preprocess_gpu_srcfp32(colind,
                                rowptr,
                                num_nodes,
                                blk_h,
                                blk_w,
                                block_partition,
                                edge_to_col,
                                edge_to_row);
}

std::vector<torch::Tensor> run_DTCSpMM(torch::Tensor dense,
                                        torch::Tensor row_window_offset,
                                        torch::Tensor tcblocktile_id,
                                        torch::Tensor tcblock_offset,
                                        torch::Tensor sparse_a_to_x_idx,
                                        int num_nodes,
                                        int num_edges,
                                        std::string exeplan) {
  CHECK_INPUT(dense);
  CHECK_INPUT(row_window_offset);
  CHECK_INPUT(tcblocktile_id);
  CHECK_INPUT(tcblock_offset);
  CHECK_INPUT(sparse_a_to_x_idx);
  TORCH_CHECK(dense.scalar_type() == torch::kFloat32, "dense must be float32");

  return run_spmm_srcfp32(dense,
                          row_window_offset,
                          tcblocktile_id,
                          tcblock_offset,
                          sparse_a_to_x_idx,
                          num_nodes,
                          num_edges,
                          exeplan);
}

std::vector<torch::Tensor> run_DTCSpMM_balance(torch::Tensor dense,
                                                torch::Tensor row_window_offset,
                                                torch::Tensor tcblocktile_id,
                                                torch::Tensor tcblock_offset,
                                                torch::Tensor sparse_a_to_x_idx,
                                                int num_nodes,
                                                std::string exeplan) {
  CHECK_INPUT(dense);
  CHECK_INPUT(row_window_offset);
  CHECK_INPUT(tcblocktile_id);
  CHECK_INPUT(tcblock_offset);
  CHECK_INPUT(sparse_a_to_x_idx);
  TORCH_CHECK(dense.scalar_type() == torch::kFloat32, "dense must be float32");

  // The src_fp32 runtime currently uses the same launch path for balance mode.
  return run_spmm_srcfp32(dense,
                          row_window_offset,
                          tcblocktile_id,
                          tcblock_offset,
                          sparse_a_to_x_idx,
                          num_nodes,
                          static_cast<int>(sparse_a_to_x_idx.numel()),
                          exeplan);
}

// Keep variant names available so the external router can dispatch without warnings.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
preprocess_gpu_strict_lb(torch::Tensor colind,
                         torch::Tensor rowptr,
                         int num_nodes,
                         int blk_h,
                         int blk_w,
                         torch::Tensor block_partition,
                         torch::Tensor edge_to_col,
                         torch::Tensor edge_to_row) {
  CHECK_INPUT(colind);
  CHECK_INPUT(rowptr);
  CHECK_INPUT(block_partition);
  CHECK_INPUT(edge_to_col);
  CHECK_INPUT(edge_to_row);
  TORCH_CHECK(colind.scalar_type() == torch::kInt32, "colind must be int32");
  TORCH_CHECK(rowptr.scalar_type() == torch::kInt32, "rowptr must be int32");

  return preprocess_gpu_srcfp32_strict_lb(colind,
                                          rowptr,
                                          num_nodes,
                                          blk_h,
                                          blk_w,
                                          block_partition,
                                          edge_to_col,
                                          edge_to_row);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
preprocess_gpu_multi_binding(torch::Tensor colind,
                             torch::Tensor rowptr,
                             int num_nodes,
                             int blk_h,
                             int blk_w,
                             torch::Tensor block_partition,
                             torch::Tensor edge_to_col,
                             torch::Tensor edge_to_row) {
  return preprocess_gpu(colind,
                        rowptr,
                        num_nodes,
                        blk_h,
                        blk_w,
                        block_partition,
                        edge_to_col,
                        edge_to_row);
}

std::vector<torch::Tensor> run_DTCSpMM_strict_lb(torch::Tensor dense,
                                                  torch::Tensor row_window_offset,
                                                  torch::Tensor tcblocktile_id,
                                                  torch::Tensor tcblock_offset,
                                                  torch::Tensor sparse_a_to_x_idx,
                                                  int num_nodes,
                                                  int num_edges,
                                                  std::string exeplan) {
  CHECK_INPUT(dense);
  CHECK_INPUT(row_window_offset);
  CHECK_INPUT(tcblocktile_id);
  CHECK_INPUT(tcblock_offset);
  CHECK_INPUT(sparse_a_to_x_idx);
  TORCH_CHECK(dense.scalar_type() == torch::kFloat32, "dense must be float32");

  return run_spmm_srcfp32_strict_lb(dense,
                                    row_window_offset,
                                    tcblocktile_id,
                                    tcblock_offset,
                                    sparse_a_to_x_idx,
                                    num_nodes,
                                    num_edges,
                                    exeplan);
}

std::vector<torch::Tensor> run_DTCSpMM_strict_lb_balance(torch::Tensor dense,
                                                          torch::Tensor row_window_offset,
                                                          torch::Tensor tcblocktile_id,
                                                          torch::Tensor tcblock_offset,
                                                          torch::Tensor sparse_a_to_x_idx,
                                                          int num_nodes,
                                                          std::string exeplan) {
  CHECK_INPUT(dense);
  CHECK_INPUT(row_window_offset);
  CHECK_INPUT(tcblocktile_id);
  CHECK_INPUT(tcblock_offset);
  CHECK_INPUT(sparse_a_to_x_idx);
  TORCH_CHECK(dense.scalar_type() == torch::kFloat32, "dense must be float32");

  return run_spmm_srcfp32_strict_lb(dense,
                                    row_window_offset,
                                    tcblocktile_id,
                                    tcblock_offset,
                                    sparse_a_to_x_idx,
                                    num_nodes,
                                    static_cast<int>(sparse_a_to_x_idx.numel()),
                                    exeplan);
}

std::vector<torch::Tensor> run_DTCSpMM_multi_binding(torch::Tensor dense,
                                                      torch::Tensor row_window_offset,
                                                      torch::Tensor tcblocktile_id,
                                                      torch::Tensor tcblock_offset,
                                                      torch::Tensor sparse_a_to_x_idx,
                                                      int num_nodes,
                                                      int num_edges,
                                                      std::string exeplan) {
  return run_DTCSpMM(dense,
                     row_window_offset,
                     tcblocktile_id,
                     tcblock_offset,
                     sparse_a_to_x_idx,
                     num_nodes,
                     num_edges,
                     exeplan);
}

std::vector<torch::Tensor> run_DTCSpMM_multi_binding_balance(torch::Tensor dense,
                                                              torch::Tensor row_window_offset,
                                                              torch::Tensor tcblocktile_id,
                                                              torch::Tensor tcblock_offset,
                                                              torch::Tensor sparse_a_to_x_idx,
                                                              int num_nodes,
                                                              std::string exeplan) {
  return run_DTCSpMM_balance(dense,
                             row_window_offset,
                             tcblocktile_id,
                             tcblock_offset,
                             sparse_a_to_x_idx,
                             num_nodes,
                             exeplan);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_gpu", &preprocess_gpu, "src_fp32 preprocess (CUDA)");
  m.def("run_DTCSpMM", &run_DTCSpMM, "src_fp32 spmm (CUDA)");

  m.def("preprocess_gpu_strict_lb", &preprocess_gpu_strict_lb, "src_fp32 strict_lb preprocess (CUDA)");
  m.def("preprocess_gpu_multi_binding", &preprocess_gpu_multi_binding, "src_fp32 multi_binding preprocess (CUDA)");
  m.def("run_DTCSpMM_strict_lb", &run_DTCSpMM_strict_lb, "src_fp32 strict_lb spmm (CUDA)");
  m.def("run_DTCSpMM_multi_binding", &run_DTCSpMM_multi_binding, "src_fp32 multi_binding spmm (CUDA)");
  m.def("run_DTCSpMM_balance", &run_DTCSpMM_balance, "src_fp32 balance spmm (CUDA)");
  m.def("run_DTCSpMM_strict_lb_balance", &run_DTCSpMM_strict_lb_balance, "src_fp32 strict_lb balance spmm (CUDA)");
  m.def("run_DTCSpMM_multi_binding_balance", &run_DTCSpMM_multi_binding_balance, "src_fp32 multi_binding balance spmm (CUDA)");
}