#!/bin/bash

# 匹配的目录
DIR="/workspace/sparsene/examples/src_fp32/dtc/testbed/build"

# 遍历所有 dtc-plan_xxx-tf32
for exe in "$DIR"/dtc-plan_*"-tf32"; do
    if [[ -x "$exe" ]]; then
        echo "正在执行: $exe"
        "$exe" -filename /workspace/selectedMM/cop20k_A.mtx -N 512
    else
        echo "跳过: $exe (不是可执行文件)"
    fi
done
