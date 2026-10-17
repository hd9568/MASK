import matplotlib.pyplot as plt
import numpy as np

# 文件路径
file_path = "/workspace/sparsene/motivation/TCBlockNum/dc2_TCBlockNum.txt"
file_path = "/workspace/sparsene/motivation/TCBlockNum/cop20k_A_TCBlockNum.txt"

# 读取数据
with open(file_path, "r") as f:
    lines = f.readlines()

num_row_windows = int(lines[0].strip())
offsets = [int(x.strip()) for x in lines[1:]]

# 计算每个 row window 的 tc block 数量
tc_block_nums = [offsets[i+1] - offsets[i] for i in range(num_row_windows)]

# 计算 CDF
sorted_tc = np.sort(tc_block_nums)
cdf = np.arange(1, len(sorted_tc)+1) / len(sorted_tc) * 100  # 百分比

# 绘图
plt.figure(figsize=(10,6))
plt.plot(sorted_tc, cdf, marker='', linestyle='-', color='b', label='CDF')

# 标注最大和最小 tc block
min_tc = min(tc_block_nums)
max_tc = max(tc_block_nums)
plt.axvline(min_tc, color='r', linestyle='--', label=f'Min TC Block = {min_tc}')
plt.axvline(max_tc, color='g', linestyle='--', label=f'Max TC Block = {max_tc}')

plt.xscale('log')  # 横坐标使用 log
plt.xlabel("TC Block Num per Row Window (log scale)")
plt.ylabel("CDF (%)")
plt.title("CDF of TC Block Numbers per Row Window")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("/workspace/sparsene/motivation/TCBlockNum_figs/test3.pdf")
plt.show()
