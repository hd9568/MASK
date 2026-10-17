#!/bin/bash

PLAN_FILE="/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/less_1.4ms_plan.txt"
PLAN_FILE="/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/test_multi_shift.txt"
PLAN_FILE="/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/test_single_shift.txt"
MATRIX="/workspace/selectedMM/cop20k_A.mtx"
NVAL=512

LOGDIR="/workspace/sparsene/examples/src_fp32/acc/testbed/build/logs"
LOGDIR="/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/plan_fig/2_0ms/"
LOGDIR="/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/"
FILE="test_single_shift_motivation.log"
mkdir -p "$LOGDIR"

while read -r plan; do
    # 跳过空行
    [ -z "$plan" ] && continue

    echo "正在执行 plan_${plan} ..." >> "${LOGDIR}/${FILE}"
    # ncu --clock-control none --set full -k regex:acc* \
        # ./acc-plan_${plan}-tf32 -filename "$MATRIX" -N "$NVAL" -ncu 1 \
        # | grep -E "Duration |Waves Per SM|Active Warps [pP]er SM|Block Limit Registers|Block Limit Shared Mem|Registers Per Thread|Shared Memory Configuration Size" >> "${LOGDIR}/${FILE}"
    # ./acc-plan_${plan}-tf32 -mtx_flag 0 -M 1024 -K 4096 -N 512 >> "${LOGDIR}/${FILE}"
    ./acc-plan_${plan}-tf32 -filename "$MATRIX" -N "$NVAL" >> "${LOGDIR}/${FILE}"
done < "$PLAN_FILE"
