import re
import json

# log_file = "/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/test_single_shift.log"
log_file = "/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/test_single_shift_exchangexy.log"
output_json = "plan_results_exchange.json"

plan_results = []
current_plan = None
current_time = None

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        # 提取 plan 编号
        match_plan = re.search(r"plan_(\d+)", line)
        if match_plan:
            current_plan = int(match_plan.group(1))
            current_time = None
            continue

        # 提取 mykernel_time
        match_time = re.search(r"mykernel_time:\s*([\d.]+)\s*ms", line)
        if match_time and current_plan is not None:
            current_time = float(match_time.group(1))
            continue

        # 提取 active block num
        match_block = re.search(r"active block num:\s*(\d+)", line)
        if match_block and current_plan is not None and current_time is not None:
            block_num = int(match_block.group(1))
            plan_results.append({
                "plan_id": current_plan,
                "time_ms": current_time,
                "block_num": block_num
            })
            # 一个plan处理完，清空
            current_plan = None
            current_time = None

# 保存到 JSON 文件
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(plan_results, f, indent=2)

print(f"结果已保存到 {output_json}")
