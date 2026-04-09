import json
import matplotlib.pyplot as plt
import numpy as np

# 读取 JSON 文件
json_file = "plan_results_exchange.json"
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 按 time_ms 升序排列
data_sorted = sorted(data, key=lambda x: x["time_ms"])

# 横坐标索引
plan_indices = np.arange(1, len(data_sorted) + 1)

# 提取纵坐标
time_ms = [d["time_ms"] for d in data_sorted]
time_ms_xy = [d.get("time_ms_exchange_xy", np.nan) for d in data_sorted]

# 绘制散点图
plt.figure(figsize=(12, 6))
plt.scatter(plan_indices, time_ms, color='blue', marker='o', label='time_ms', s=2)
plt.scatter(plan_indices, time_ms_xy, color='red', marker='s', label='time_ms_exchange_xy', s=2)

plt.xlabel("Plan (sorted by time_ms)", fontsize=14)
plt.ylabel("Execution Time (ms)", fontsize=14)
plt.title("Execution Time per Plan (Scatter Plot)", fontsize=16)
plt.legend(fontsize=12, loc="upper left")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("single_shift-time_vs_exchange_xy.pdf")
plt.show()
