#!/bin/bash

# 假设输入文件为sparse_files.txt
input_file="/workspace/sparsene/example/dtc-spmm/scripts/filtered_mtx_files.txt.small"
input_file="/workspace/sparsene/examples/scripts/selected_mtx.1.txt"

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
    # echo "no SMEM"  
    # ncu --clock-control none --set full -k regex:acc* /workspace/sparsene/examples/src_fp16/acc/testbed/build/demo-acc-fp16 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000 -ncu 1 > /workspace/sparsene/examples/scripts/tmp.log 2>&1
    # cat /workspace/sparsene/examples/scripts/tmp.log | grep -E "Duration |Waves Per SM|Active Warps [pP]er SM" 
    # echo "SMEM"
    # ncu --clock-control none --set full -k regex:acc* /workspace/sparsene/examples/src_fp16/acc/testbed/build/demo-acc-mco-smem-fp16 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000 -ncu 1 > /workspace/sparsene/examples/scripts/tmp.log 2>&1
    # cat /workspace/sparsene/examples/scripts/tmp.log | grep -E "Duration |Waves Per SM|Active Warps [pP]er SM" 

    echo "no SMEM"  
    ncu --clock-control none --set full -k regex:bitbsr* /workspace/sparsene/examples/src_fp16/bitbsr/testbed/build/demo-bitbsr-fp16 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000 -ncu 1 > /workspace/sparsene/examples/scripts/tmp.log 2>&1
    cat /workspace/sparsene/examples/scripts/tmp.log | grep -E "Duration |Waves Per SM|Active Warps [pP]er SM" 
    echo "SMEM"
    ncu --clock-control none --set full -k regex:bitbsr* /workspace/sparsene/examples/src_fp16/bitbsr/testbed/build/demo-bitbsr-idx-smem-fp16 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000 -ncu 1 > /workspace/sparsene/examples/scripts/tmp.log 2>&1
    cat /workspace/sparsene/examples/scripts/tmp.log | grep -E "Duration |Waves Per SM|Active Warps [pP]er SM" 
  fi
done < "$input_file"