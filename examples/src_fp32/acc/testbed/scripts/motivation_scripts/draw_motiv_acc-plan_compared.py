import re
import matplotlib.pyplot as plt

# ================== 收集真实时间 ==================
log_file = "/workspace/sparsene/examples/src_fp32/acc/testbed/build/cumemcheck.1.log"

true_times = []
current_plan = None

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        # 提取 plan 编号
        match_plan = re.search(r"acc-plan_(\d+)-tf32", line)
        if match_plan:
            current_plan = int(match_plan.group(1))
            continue

        # 提取 mykernel_time
        match_time = re.search(r"mykernel_time:\s*([\d.]+)\s*ms", line)
        if match_time and current_plan is not None:
            time_ms = float(match_time.group(1))
            true_times.append((current_plan, time_ms))
            current_plan = None   # 一个 plan 对应一个时间，清空

# 排序（真实时间从小到大）
true_times.sort(key=lambda x: x[1])

# 散点图绘制
times = [t[1] for t in true_times if t[1] < 3.0]

x = list(range(len(times)))

plt.figure(figsize=(8,5))
plt.scatter(x, times, s=5, label="Plans")

# 特殊点 1.70772ms
special_time = 1.70772

# 找到与特殊点最接近的索引
special_idx = min(range(len(times)), key=lambda i: abs(times[i]-special_time))

plt.scatter(special_idx, times[special_idx], color="red", s=20, zorder=5, label="ACC-SpMM's Plan")

# 添加箭头注释
plt.annotate("ACC-SpMM's Plan\n(1.70772 ms)",
             xy=(special_idx, times[special_idx]), 
             xytext=(special_idx-50, times[special_idx]+0.2),
             arrowprops=dict(facecolor='black', arrowstyle="->"),
             fontsize=10)

plt.ylabel("Time (ms)")
plt.xlabel("Different Pipeline Plans")
plt.title("Plan execution times")
plt.legend()
plt.grid(True)
plt.savefig("/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/motiv-acc-plan.pdf", )
plt.show()
