#!/usr/bin/env python3
import json
import re

# 原始 JSON 文件
input_json = "plan_results.json"
# 处理 exchange_xy 的 log 文件
log_file = "/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/test_single_shift_exchangexy.log"
# 输出 JSON 文件
output_json = "plan_results_exchange.json"

# 1️⃣ 读取原始 JSON
with open(input_json, "r", encoding="utf-8") as f:
    plan_results = json.load(f)

# 2️⃣ 从 exchange_xy log 提取数据
exchange_data = {}
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
            # 存入字典，方便后续合并
            exchange_data[current_plan] = {
                "time_ms_exchange_xy": current_time,
                "block_num_exchange_xy": block_num
            }
            current_plan = None
            current_time = None

# 3️⃣ 合并数据到原 JSON
for item in plan_results:
    plan_id = item["plan_id"]
    if plan_id in exchange_data:
        item.update(exchange_data[plan_id])
    else:
        # 没有找到对应的 exchange_xy 数据，可以填 None
        item["time_ms_exchange_xy"] = None
        item["block_num_exchange_xy"] = None

# 4️⃣ 保存到新的 JSON 文件
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(plan_results, f, indent=2)

print(f"结果已保存到 {output_json}")
