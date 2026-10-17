#!/bin/bash

# 假设输入文件为sparse_files.txt
input_file="/Chainsaw/script/filtered_mtx_files.txt.small"
input_file="/workspace/sparsene/examples/src_fp16/dtc/testbed/scripts/selected_mtx_path.txt"
input_file="/workspace/sparsene/examples/src_fp16/dtc/testbed/scripts/selected_mtx_path.txt.2"

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
    # ./myhalf -filename "$line"
    echo "plan 0"
    ./build/dtc-plan_0-fp16 -filename "$line" -N 1024 -mtx_flag 1 -warmup 100 -repeat 1000
    echo "plan 1"
    ./build/dtc-plan_1-fp16 -filename "$line" -N 1024 -mtx_flag 1 -warmup 100 -repeat 1000
    echo "plan 2"
    ./build/dtc-plan_2-fp16 -filename "$line" -N 1024 -mtx_flag 1 -warmup 100 -repeat 1000
    echo "plan 3"
    ./build/dtc-plan_3-fp16 -filename "$line" -N 1024 -mtx_flag 1 -warmup 100 -repeat 1000
    echo "plan 4"
    ./build/dtc-plan_4-fp16 -filename "$line" -N 1024 -mtx_flag 1 -warmup 100 -repeat 1000
    echo "plan 5"
    ./build/dtc-plan_5-fp16 -filename "$line" -N 1024 -mtx_flag 1 -warmup 100 -repeat 1000
    echo "plan 6"
    ./build/dtc-plan_6-fp16 -filename "$line" -N 1024 -mtx_flag 1 -warmup 100 -repeat 1000
    echo "plan 7"
    ./build/dtc-plan_7-fp16 -filename "$line" -N 1024 -mtx_flag 1 -warmup 100 -repeat 1000
    echo "plan 8"
    ./build/dtc-plan_8-fp16 -filename "$line" -N 1024 -mtx_flag 1 -warmup 100 -repeat 1000
    # ./spmv_half_pretime "$line" | grep " ms" 
  fi
done < "$input_file"