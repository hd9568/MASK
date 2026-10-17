import re
import json
import numpy as np

# ===== 1. 读取 plan 文本并解析 =====
with open("/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/filtered_plans.txt", "r") as f:
    plans_text = f.read()

plans = [line.strip() for line in plans_text.strip().split("\n")]

def parse_plan(plan):
    parts = plan.split(',')
    row_num = int(parts[0])
    ops_str = ','.join(parts[1:])
    
    stages = re.split(r'\|\((\d+)\)>', ops_str)
    ops_list, shift_list = [], []
    for i, s in enumerate(stages):
        if i % 2 == 0:
            ops_list.append([op.strip() for op in s.split(',') if op.strip()])
        else:
            shift_list.append(int(s))
    return {'row': row_num, 'ops': ops_list, 'shifts': shift_list, 'raw': plan}

parsed_plans = [parse_plan(p) for p in plans]

# ===== 2. 策略筛选函数 =====
def select_g2s_first(plans):
    result = []
    for p in plans:
        if all(not any(op.startswith('G2s') for op in stage) or stage[0].startswith('G2s') for stage in p['ops']):
            result.append(p)
    return result

def select_shift_one(plans):
    return [p for p in plans if all(shift == 1 for shift in p['shifts'])]

# ===== 3. 读取 plan 时间 JSON 并计算排名 =====
with open("/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/plan_results.json", "r") as f:
    plan_times = json.load(f)

# 根据 time_ms 排序得到 plan_id 的排名（越小越好）
sorted_plans = sorted(plan_times, key=lambda x: x['time_ms'])
plan_rank_dict = {p['plan_id']: i for i, p in enumerate(sorted_plans)}  # 排名从0开始

# ===== 4. 模拟随机抽样落在前 k 的比例 =====
N_total = len(plan_times)
sample_sizes = [10, 20, 50, 200]  # 随机抽样数量
top_ks = [10, 20, 50, 200]        # 关注前 k
num_trials = 100000
np.random.seed(42)

def simulate_topk_probability(plan_subset_ids):
    """模拟随机从 plan_subset_ids 中抽样落入前 k 的比例"""
    subset_size = len(plan_subset_ids)
    ranks = [plan_rank_dict[pid] for pid in plan_subset_ids]
    results = {k: {n: 0 for n in sample_sizes} for k in top_ks}
    
    for _ in range(num_trials):
        np.random.shuffle(ranks)
        for sample_size in sample_sizes:
            if sample_size >= subset_size:
                # 如果抽样数量 >= 子集大小，直接用整个子集
                sample = ranks
            else:
                sample = ranks[:sample_size]
            for k in top_ks:
                count_in_topk = sum(r < k for r in sample)
                results[k][sample_size] += count_in_topk / k

    # 平均比例
    for k in top_ks:
        for n in sample_sizes:
            results[k][n] /= num_trials
    return results

# ===== 5. 对不同策略的 plan 子集模拟 =====
# 示例：策略1
g2s_first_plans = select_g2s_first(parsed_plans)
print(len(g2s_first_plans))
g2s_first_ids = [p['row'] for p in g2s_first_plans]
res_g2s = simulate_topk_probability(g2s_first_ids)

print("策略1 - G2s最先发射:")
for k in top_ks:
    row = [f"{res_g2s[k][n]:.3f}" for n in sample_sizes]
    print(f"top-{k}: {' '.join(row)}")

# 示例：策略2
shift_one_plans = select_shift_one(parsed_plans)
shift_one_ids = [p['row'] for p in shift_one_plans]
res_shift1 = simulate_topk_probability(shift_one_ids)

print("\n策略2 - shift=1:")
for k in top_ks:
    row = [f"{res_shift1[k][n]:.3f}" for n in sample_sizes]
    print(f"top-{k}: {' '.join(row)}")
