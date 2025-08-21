#!/bin/bash

# 固定稀疏度
SPARSITY=0.9

# 输出文件
OUT_FILE=/workspace/sparsene/examples/src_fp16/acc/testbed/build/test_result.2.log
> $OUT_FILE   # 清空旧结果

for ((m=512; m<=8192; m*=2)); do
    k=$m  # m=k
    for ((n=512; n<=2048; n*=2)); do
        echo "Running M=$m, N=$n, K=$k" | tee -a $OUT_FILE
        # for flag in 0 1; do
            echo "  Kernel without SMEM" | tee -a $OUT_FILE
            /workspace/sparsene/examples/src_fp16/acc/testbed/build/demo-acc-fp16 -mtx_flag 0 -M $m -N $n -K $k -sparsity $SPARSITY 2>&1 | tee -a $OUT_FILE
            echo "  Kernel with SMEM" | tee -a $OUT_FILE
            /workspace/sparsene/examples/src_fp16/acc/testbed/build/demo-acc-mco-smem-fp16 -mtx_flag 0 -M $m -N $n -K $k -sparsity $SPARSITY 2>&1 | tee -a $OUT_FILE
            
        # done
        echo "" | tee -a $OUT_FILE
    done
done

# ncu --clock-control none --set full  -k regex:acc*  demo-acc-mco-smem-fp16 -mtx_flag 0 -ncu 1 -M 1728 -N 1024 -K 4096 -sparsity 0.9 2>&1 | tee -a  test_result.3.log
# ncu --clock-control none --set full -k regex:acc* demo-acc-fp16 -mtx_flag 0 -ncu 1 -M 1728 -N 1024 -K 4096 -sparsity 0.9 2>&1 | tee -a test_result.3.log