import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 14,       # 全局字体大小
    'axes.titlesize': 16,  # 坐标轴标题大小
    'axes.labelsize': 14,  # 坐标轴标签大小
    'legend.fontsize': 11, # 图例字体大小
    'xtick.labelsize': 14, # x 轴刻度字体大小
    'ytick.labelsize': 14  # y 轴刻度字体大小
})


def read_tc_block_nums(file_path):
    """读取文件，计算每个 row window 的 TC block 数量"""
    with open(file_path, "r") as f:
        lines = f.readlines()
    num_row_windows = int(lines[0].strip())
    offsets = [int(x.strip()) for x in lines[1:]]
    tc_block_nums = [offsets[i+1] - offsets[i] for i in range(num_row_windows)]
    return tc_block_nums

# 文件路径
file1 = "/workspace/sparsene/motivation/TCBlockNum/dc2_TCBlockNum.txt"
file2 = "/workspace/sparsene/motivation/TCBlockNum/cop20k_A_TCBlockNum.txt"

# 读取数据
tc1 = read_tc_block_nums(file1)
tc2 = read_tc_block_nums(file2)

# 计算 CDF
def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data) * 100  # 百分比
    return sorted_data, cdf

sorted_tc1, cdf1 = compute_cdf(tc1)
sorted_tc2, cdf2 = compute_cdf(tc2)

# 绘制上下两个子图（不共享 x 轴）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# 上图：文件1
ax1.plot(sorted_tc1, cdf1, linestyle='-', color='b')
ax1.set_ylabel("CDF (%)")
ax1.set_title("(a) CDF of TC Block Numbers: dc2")
ax1.axvline(min(tc1), color='r', linestyle='--', label=f'Min = {min(tc1)}')
ax1.axvline(max(tc1), color='g', linestyle='--', label=f'Max = {max(tc1)}')
ax1.legend(loc='upper right', bbox_to_anchor=(1, 0.9)) # legend 固定在左上角
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xscale('log')

# 下图：文件2
ax2.plot(sorted_tc2, cdf2, linestyle='-', color='b')
# ax2.set_xlabel("TC Block Num per Row Window (log scale)")
ax2.set_ylabel("CDF (%)")
ax2.set_title("(b) CDF of TC Block Numbers: cop20k_A")
ax2.axvline(min(tc2), color='r', linestyle='--', label=f'Min = {min(tc2)}')
ax2.axvline(max(tc2), color='g', linestyle='--', label=f'Max = {max(tc2)}')
ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9)) # legend 固定在左上角
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig("/workspace/sparsene/motivation/TCBlockNum_figs/motiv-lb.pdf")
plt.show()
