#!/bin/bash

# 假设输入文件为sparse_files.txt
input_file="/workspace/sparsene/example/dtc-spmm/scripts/filtered_mtx_files.txt.small"

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
    /workspace/sparsene/example/dtc-spmm/build/dtc_half_row_majorB_vs_bind_reorder -filename "$line" -N 512
  fi
done < "$input_file"