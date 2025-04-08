#!/bin/bash

#! DTC-SpMM env
# source ~/miniconda3/etc/profile.d/conda.sh

# conda activate DTCSpMM
# source /workspace/DTC-SpMM_ASPLOS24/init_dtc.sh
# source /workspace/DTC-SpMM_ASPLOS24/third_party/load_sputnik.sh

# 假设输入文件为sparse_files.txt
input_file="/workspace/sparsene-compiler/example/dtc-spmm/scripts/selected_mtx.txt"

# 检查文件是否存在
if [ ! -f "$input_file" ]; then
  echo "文件 $input_file 不存在"
  exit 1
fi

# 逐行读取txt文件中的路径并调用程序
while IFS= read -r line
do
  # 确保行不为空
  if [ -n "$line" ]; then
    echo "正在测试 $line ..."
    #! mykernel half
    # ncu --clock-control none -k dtc_spmm_kernel_fp16_val_idx_bind_stage2 /workspace/sparsene-compiler/example/dtc-spmm/build/src_fp16/row_majorB_vs_bind_reorder/dtc_half_row_majorB_vs_bind_reorder -filename "$line" -N 128 | grep Duration
    #! smat half
    # ncu --clock-control none -k mmaCBTKernelSparse /workspace/smat/src/cuda_hgemm/output/bin/hgemm -N=128 -n_mult=16 -enable_wmma=true -enable_mma=true -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false -filename="$line"  | grep Duration
    #! mykernel float
    # ncu --clock-control none -k dtc_spmm_kernel_fp32_val_idx_bind_stage2  /workspace/sparsene-compiler/example/dtc-spmm/build/src_fp32/row_majorB_vs_bind_reorder/dtc_float_row_majorB_vs_bind_reorder -filename "$line" -N 128 | grep Duration
    #! DTC-SpMM float
    ncu --clock-control none -k regex:spmm_forward_cuda_kernel.* python /workspace/DTC-SpMM_ASPLOS24/scripts/DTCSpMM/run_DTC_SpMM_normal.py --dataset "$line" | grep Duration
  fi
done < "$input_file"

#! 获得ncu报告 my
# ncu --clock-control none  --import-source yes --section regex:. -o ./report/dtc-half-row-majorB-vs-bind-reorder-auto.6 /workspace/sparsene-compiler/example/dtc-spmm/build/src_fp16/row_majorB_vs_bind_reorder/dtc_half_row_majorB_vs_bind_reorder -filename ../test/dc2.mtx -N 128 > ../cumemcheck.log

#! 获得ncu报告smat
# ncu --clock-control none --import-source yes --section regex:. -o ./report/hgemm-cop20k /workspace/smat/src/cuda_hgemm/output/bin/hgemm -N=128 -enable_wmma=true -enable_mma=true -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false -filename=/workspace/sparsene/example/dtc-spmm/test/cop20k_A.mtx