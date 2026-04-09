import numpy as np

# ===== 参数设置 =====
N_total = 684          # 总元素个数
sample_sizes = [10, 20, 50, 200]  # 每次抽样的数量
top_ks = [10, 20, 50, 200]        # 前 k 个元素关注
num_trials = 100000    # 模拟次数

# ===== 模拟 =====
np.random.seed(42)  # 保证可重复

results = {k: {n: 0 for n in sample_sizes} for k in top_ks}

for trial in range(num_trials):
    # 从总集合中随机抽样
    sample = np.random.choice(N_total, max(sample_sizes), replace=False)
    
    for sample_size in sample_sizes:
        sample_subset = sample[:sample_size]
        for k in top_ks:
            # 计算 sample_subset 中有多少落在前 k 个元素
            count_in_topk = np.sum(sample_subset < k)
            results[k][sample_size] += count_in_topk / k  # 比例

# 计算平均比例
print("随机抽样落入前 k 的比例（模拟平均值）：")
for k in top_ks:
    row = [f"{results[k][n]/num_trials:.3f}" for n in sample_sizes]
    print(f"top-{k}: {' '.join(row)}")
