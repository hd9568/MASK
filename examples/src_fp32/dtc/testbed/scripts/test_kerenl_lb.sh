#!/bin/bash

# 假设输入文件为sparse_files.txt
input_file="/workspace/sparsene/example/dtc-spmm/scripts/filtered_mtx_files.txt.small"
input_file="/workspace/sparsene/examples/scripts/selected_mtx.2.txt"
input_file="/workspace/sparsene/examples/src_fp32/dtc/testbed/scripts/mtx_files2.txt"

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

    echo "Normal Binding"  
    timeout 300 ncu --clock-control none --set full -k regex:dtc* /workspace/sparsene/examples/src_fp32/dtc/testbed/build/demo-dtc-tf32 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000 -ncu 1 > /workspace/sparsene/examples/scripts/tmp.log 2>&1
    cat /workspace/sparsene/examples/scripts/tmp.log | grep -E "Duration |Waves Per SM|Active Warps [pP]er SM" 
    # /workspace/sparsene/examples/src_fp32/dtc/testbed/build/demo-dtc-tf32 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000 


    echo "Multi Binding"
    timeout 300 ncu --clock-control none --set full -k regex:dtc* /workspace/sparsene/examples/src_fp32/dtc/testbed/build/demo-dtc-multi-binding-tf32 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000 -ncu 1 > /workspace/sparsene/examples/scripts/tmp.log 2>&1
    cat /workspace/sparsene/examples/scripts/tmp.log | grep -E "Duration |Waves Per SM|Active Warps [pP]er SM" 
    # /workspace/sparsene/examples/src_fp32/dtc/testbed/build/demo-dtc-multi-binding-tf32 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000


    echo "Strict LB"
    timeout 300 ncu --clock-control none --set full -k regex:dtc* /workspace/sparsene/examples/src_fp32/dtc/testbed/build/demo-dtc-strict-lb-tf32 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000 -ncu 1 > /workspace/sparsene/examples/scripts/tmp.log 2>&1
    cat /workspace/sparsene/examples/scripts/tmp.log | grep -E "Duration |Waves Per SM|Active Warps [pP]er SM" 
    # /workspace/sparsene/examples/src_fp32/dtc/testbed/build/demo-dtc-strict-lb-tf32 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000
  fi
done < "$input_file"