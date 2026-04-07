#pragma once

#include <torch/extension.h>

#include <string>
#include <tuple>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
preprocess_gpu_srcfp32(torch::Tensor colind_i32,
                       torch::Tensor rowptr_i32,
                       int num_nodes,
                       int blk_h,
                       int blk_w,
                       torch::Tensor block_partition_i32,
                       torch::Tensor edge_to_col_i32,
                       torch::Tensor edge_to_row_i32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
preprocess_gpu_srcfp32_strict_lb(torch::Tensor colind_i32,
                                 torch::Tensor rowptr_i32,
                                 int num_nodes,
                                 int blk_h,
                                 int blk_w,
                                 torch::Tensor block_partition_i32,
                                 torch::Tensor edge_to_col_i32,
                                 torch::Tensor edge_to_row_i32);

std::vector<torch::Tensor> run_spmm_srcfp32(torch::Tensor dense,
                                             torch::Tensor row_window_offset,
                                             torch::Tensor tcblocktile_id,
                                             torch::Tensor tcblock_offset,
                                             torch::Tensor sparse_a_to_x_idx,
                                             int num_nodes,
                                             int num_edges,
                                             std::string exeplan);

std::vector<torch::Tensor> run_spmm_srcfp32_strict_lb(torch::Tensor dense,
                                                       torch::Tensor row_window_offset,
                                                       torch::Tensor tcblocktile_id,
                                                       torch::Tensor tcblock_offset,
                                                       torch::Tensor sparse_a_to_x_idx,
                                                       int num_nodes,
                                                       int num_edges,
                                                       std::string exeplan);
