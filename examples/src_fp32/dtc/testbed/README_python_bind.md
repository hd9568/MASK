# src_fp32 DTC: Bind Handwritten CUDA Op To Python

This note gives a minimal path to bind your handwritten CUDA operator for end2end.

Current status in this folder:

- Implemented pybind entry: DTCSpMM.cpp
- Implemented runtime bridge: DTCSpMM_kernel.cu
- Build script: setup.py

## Why current code is not end2end-ready

- Current entry is executable-style in [sparsene/examples/src_fp32/dtc/testbed/main.cu](sparsene/examples/src_fp32/dtc/testbed/main.cu).
- End2end needs in-process Python callable functions (preprocess and run), not subprocess binaries.

## Target Python API surface

Keep names compatible with the current external router in [sparsene/end2end/gcn/external_backend_spmm.py](sparsene/end2end/gcn/external_backend_spmm.py):

- preprocess_gpu
- run_DTCSpMM
- optional: preprocess_gpu_strict_lb, preprocess_gpu_multi_binding
- optional: run_DTCSpMM_strict_lb, run_DTCSpMM_multi_binding

## Step 1: Extract callable CUDA/C++ entry points

Move core logic from main path into reusable functions.

Example declarations:

```cpp
// dtc_runtime_api.h
#pragma once
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
preprocess_gpu_srcfp32(torch::Tensor colind_i32,
                       torch::Tensor rowptr_i32,
                       int num_nodes,
                       int blk_h,
                       int blk_w,
                       torch::Tensor block_partition_i32,
                       torch::Tensor edge_to_col_i32,
                       torch::Tensor edge_to_row_i32);

std::vector<torch::Tensor>
run_spmm_srcfp32(torch::Tensor dense,
                 torch::Tensor row_window_offset,
                 torch::Tensor tcblocktile_id,
                 torch::Tensor tcblock_offset,
                 torch::Tensor sparse_a_to_x_idx,
                 int num_nodes,
                 int num_edges,
                 std::string exeplan);
```

## Step 2: Add pybind wrapper module

Create DTCSpMM.cpp in this folder and export functions using pybind.

```cpp
#include <torch/extension.h>
#include <pybind11/stl.h>
#include "dtc_runtime_api.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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
  return preprocess_gpu_srcfp32(colind, rowptr, num_nodes, blk_h, blk_w,
                                block_partition, edge_to_col, edge_to_row);
}

std::vector<torch::Tensor>
run_DTCSpMM(torch::Tensor dense,
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
  return run_spmm_srcfp32(dense, row_window_offset, tcblocktile_id,
                          tcblock_offset, sparse_a_to_x_idx,
                          num_nodes, num_edges, exeplan);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_gpu", &preprocess_gpu, "src_fp32 preprocess (CUDA)");
  m.def("run_DTCSpMM", &run_DTCSpMM, "src_fp32 spmm (CUDA)");
}
```

## Step 3: Build extension with setup.py

Template setup.py:

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

SPUTNIK_PATH = os.getenv("SPUTNIK_PATH", "/workspace/baselines/DTC-SpMM_ASPLOS24/third_party/sputnik")
GLOG_LIB_DIR = os.getenv("GLOG_LIB_DIR", "/workspace/baselines/DTC-SpMM_ASPLOS24/third_party/glog/build")
GLOG_INCLUDE_DIR = os.getenv("GLOG_INCLUDE_DIR", "/workspace/baselines/DTC-SpMM_ASPLOS24/third_party/glog/build/glog")

setup(
    name="DTCSpMM",
    ext_modules=[
        CUDAExtension(
            "DTCSpMM",
            [
                "DTCSpMM.cpp",
                "DTCSpMM_kernel.cu",
            ],
            include_dirs=[SPUTNIK_PATH, GLOG_INCLUDE_DIR],
            library_dirs=[
                os.path.join(SPUTNIK_PATH, "build/sputnik"),
                GLOG_LIB_DIR,
            ],
            libraries=["sputnik", "glog"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

Build:

```bash
cd /workspace/sparsene/examples/src_fp32/dtc/testbed
python3 setup.py build_ext --inplace
```

## Step 4: Hook to end2end

Use env variables so end2end picks your module first:

```bash
cd /workspace/sparsene/end2end
SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed \
SPARSENE_DTC_MODULE_NAME=DTCSpMM \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
bash scripts/smoke_test_dtc.sh
```

If your module depends on shared libs, preload them:

```bash
SPARSENE_DTC_PRELOAD_LIBS=/workspace/baselines/DTC-SpMM_ASPLOS24/third_party/sputnik/build/sputnik/libsputnik.so:/usr/lib/x86_64-linux-gnu/libglog.so.0
```

## Done criteria

- No "fallback to torch SpMM" in logs
- No import failure in logs
- End2end smoke exits with PASS
