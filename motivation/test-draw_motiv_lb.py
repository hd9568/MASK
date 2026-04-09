import matplotlib.pyplot as plt
import numpy as np

# 文件路径
file_path = "/workspace/sparsene/motivation/TCBlockNum/ASIC_680k_TCBlockNum.txt"

# 读取数据
with open(file_path, "r") as f:
    lines = f.readlines()

num_row_windows = int(lines[0].strip())
offsets = [int(x.strip()) for x in lines[1:]]

# 计算每个 row window 的 tc block 数量
tc_block_nums = [offsets[i+1] - offsets[i] for i in range(num_row_windows)]

# 统计每个 tc block num 出现次数
unique, counts = np.unique(tc_block_nums, return_counts=True)

# 转为百分比
percentages = counts / counts.sum() * 100

# 绘制柱状图
plt.figure(figsize=(10,6))
plt.bar(unique, percentages, width=0.8, edgecolor='black')
plt.xscale('log')  # 横坐标使用 log
plt.xlabel("TC Block Num per Row Window (log scale)")
plt.ylabel("Percentage (%)")
plt.title("Distribution of TC Block Numbers")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("/workspace/sparsene/motivation/TCBlockNum_figs/test.pdf")
plt.show()
