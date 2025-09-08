import re
from typing import Literal, List, Dict, Union, Tuple, Any, Optional, Sequence, Set
import random

import re
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics import ndcg_score
import json

# random.seed(111)  # 设置随机种子
import time

def find_rank_mismatches(true_times, pred_times):
    true_dict = dict(true_times)
    pred_dict = dict(pred_times)
    common_plans = sorted(set(true_dict) & set(pred_dict))

    true_sorted = sorted(common_plans, key=lambda x: true_dict[x])
    pred_sorted = sorted(common_plans, key=lambda x: pred_dict[x])

    mismatches = []
    for p in common_plans:
        true_rank = true_sorted.index(p)
        pred_rank = pred_sorted.index(p)
        rank_diff = pred_rank - true_rank
        mismatches.append((p, true_rank, pred_rank, rank_diff))

    # 按错位程度排序
    mismatches.sort(key=lambda x: abs(x[3]), reverse=True)
    return mismatches


def evaluate_ranking_new(true_times, pred_times, topk=10, sample_size=32, num_samples=20):
    """
    评估模拟器排序与真实排序的相似度
    - Kendall Tau
    - Spearman Rho
    - Pearson Correlation
    - NDCG
    - Precision@k

    参数:
        true_times: [(plan_number, time), ...] 真实时间
        pred_times: [(plan_number, time), ...] 模拟器预测时间
        topk: 用于 Precision@k
        sample_size: 每次采样的计划数量
        num_samples: 重复采样次数

    返回:
        dict: { "kendall_tau": (mean, std), "spearman_rho": (mean, std), 
                "pearson": (mean, std), "ndcg": value, "precision@k": value }
    """
    true_dict = dict(true_times)
    pred_dict = dict(pred_times)
    common_plans = sorted(set(true_dict.keys()) & set(pred_dict.keys()))
    print("common_plan size = ", len(common_plans))
    if not common_plans:
        raise ValueError("没有共同的 plan_number，无法对齐！")

    n = len(common_plans)
    if sample_size > n:
        sample_size = n   # 样本数量不能超过总数量

    tau_list, rho_list, pearson_list = [], [], []

    for _ in range(num_samples):
        sampled = random.sample(common_plans, sample_size)

        # 排名
        true_sorted = sorted(sampled, key=lambda x: true_dict[x])
        pred_sorted = sorted(sampled, key=lambda x: pred_dict[x])

        true_rank = [true_sorted.index(p) for p in sampled]
        pred_rank = [pred_sorted.index(p) for p in sampled]

        # Kendall’s Tau
        tau, _ = kendalltau(true_rank, pred_rank)
        if not np.isnan(tau):
            tau_list.append(tau)

        # Spearman’s Rho
        rho, _ = spearmanr(true_rank, pred_rank)
        if not np.isnan(rho):
            rho_list.append(rho)

        # Pearson Correlation (用时间直接算)
        true_vals = [true_dict[p] for p in sampled]
        pred_vals = [pred_dict[p] / 600 for p in sampled]
        pr, _ = pearsonr(true_vals, pred_vals)
        if not np.isnan(pr):
            pearson_list.append(pr)

    # NDCG（全量，不采样）
    true_scores = np.array([1.0 / true_dict[p] for p in common_plans])
    pred_scores = np.array([1.0 / pred_dict[p] for p in common_plans])
    ndcg = ndcg_score([true_scores], [pred_scores])

    # Precision@k（全量，不采样）
    true_sorted = sorted(common_plans, key=lambda x: true_dict[x])
    pred_sorted = sorted(common_plans, key=lambda x: pred_dict[x])
    true_topk = set(true_sorted[:topk])
    pred_topk = set(pred_sorted[:topk])
    precision_at_k = len(true_topk & pred_topk) / topk

    # print("tau_list", tau_list)
    # print("spearman list", rho_list)
    # print("person_list", pearson_list)
    return {
        "kendall_tau": (np.mean(tau_list), np.std(tau_list)),
        "spearman_rho": (np.mean(rho_list), np.std(rho_list)),
        "pearson": (np.mean(pearson_list), np.std(pearson_list)),
        "ndcg": ndcg,
        "precision@k": precision_at_k
    }


# ================== 评估函数 ==================
def evaluate_ranking(true_times, pred_times, topk=10):
    """
    评估模拟器排序与真实排序的相似度
    """
    true_dict = dict(true_times)
    pred_dict = dict(pred_times)
    common_plans = sorted(set(true_dict.keys()) & set(pred_dict.keys()))
    if not common_plans:
        raise ValueError("没有共同的 plan_number，无法对齐！")

    # 构建排名
    true_sorted = sorted(common_plans, key=lambda x: true_dict[x])   # 按真实时间升序
    pred_sorted = sorted(common_plans, key=lambda x: pred_dict[x])   # 按预测时间升序

    # 转换成排名向量
    true_rank = [true_sorted.index(p) for p in common_plans]
    pred_rank = [pred_sorted.index(p) for p in common_plans]

    # Kendall’s Tau
    tau, _ = kendalltau(true_rank, pred_rank)

    # Spearman’s Rho
    rho, _ = spearmanr(true_rank, pred_rank)

    # NDCG （用 1/time 作为 relevance）
    true_scores = np.array([1.0 / true_dict[p] for p in common_plans])
    pred_scores = np.array([1.0 / (pred_dict[p] / 600) for p in common_plans])
    ndcg = ndcg_score([true_scores], [pred_scores])

    # Precision@k
    true_topk = set(true_sorted[:topk])
    pred_topk = set(pred_sorted[:topk])
    precision_at_k = len(true_topk & pred_topk) / topk

    return {
        "kendall_tau": tau,
        "spearman_rho": rho,
        "ndcg": ndcg,
        "precision@k": precision_at_k
    }


# ----------- 解析 plan 函数 -----------
def parse_plan(plan_text: str):
    """
    解析 plan 文本生成 plan_ops 列表 (op_name, order, stage_id)
    同时解析每个 stage 之间的 shift 数量，生成 pipeline_shifts
    """
    plan_text = plan_text.strip()

    # 用正则切分，同时保留分隔符 (例如 |(1)>)
    parts = re.split(r'(\|\(\d+\)>)', plan_text)

    plan_ops = []
    pipeline_shifts = []
    order = 1
    stage_id = 0
    plan_number = None

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        # 匹配 shift 符号 |(N)>
        m = re.match(r'\|\((\d+)\)>', part)
        if m:
            pipeline_shifts.append(int(m.group(1)))
            continue

        # 普通 stage
        ops = [op.strip() for op in part.split(",") if op.strip()]

        # 第一阶段可能有 plan_number
        if stage_id == 0 and ops and ops[0][0].isdigit():
            plan_number = int(ops[0])
            ops = ops[1:]  # 去掉 plan_number

        # 生成 plan_ops
        for op in ops:
            plan_ops.append((op, order, stage_id))
            order += 1

        stage_id += 1

    return plan_number, plan_ops, pipeline_shifts


# ----------- 输入 plan 文本 -----------
plan_text = "45,G2sSparseIndexLoadOp, G2rSparseMcoOffLoadOp |(1)> G2rSparseMcoMaskLoadOp, G2sSparseMcoValLoadOp, G2sMatrixBLoadOp |(1)> S2sRestoreMatrixAOp, S2rAValLoadOp, S2rBValLoadOp, CalculateOp"
plan_text = "2052,G2rSparseMcoOffLoadOp, G2sSparseIndexLoadOp |(1)> G2rSparseMcoMaskLoadOp, G2sSparseMcoValLoadOp, G2sMatrixBLoadOp |(1)> S2sRestoreMatrixAOp, S2rAValLoadOp, S2rBValLoadOp, CalculateOp"
plan_text = "0,G2sSparseIndexLoadOp, G2rSparseMcoOffLoadOp, G2rSparseMcoMaskLoadOp, G2sSparseMcoValLoadOp |(1)> G2sMatrixBLoadOp, S2sRestoreMatrixAOp, S2rAValLoadOp |(1)> S2rBValLoadOp, CalculateOp"

# 依赖关系 (from -> to)
dependencies = [
    ("G2sSparseIndexLoadOp", "G2sMatrixBLoadOp"),
    ("G2sMatrixBLoadOp", "S2rBValLoadOp"),
    # ("G2rSparseMcoOffLoadOp", "G2rSparseMcoMaskLoadOp"),
    ("G2rSparseMcoOffLoadOp", "G2sSparseMcoValLoadOp"),
    ("G2rSparseMcoMaskLoadOp", "S2sRestoreMatrixAOp"),
    ("G2sSparseMcoValLoadOp", "S2sRestoreMatrixAOp"),
    ("S2sRestoreMatrixAOp", "S2rAValLoadOp"),
    ("S2rAValLoadOp", "CalculateOp"),
    ("S2rBValLoadOp", "CalculateOp"),
]

plan_latency={
    "G2sSparseIndexLoadOp": {"Ta": 12, "Ts": 1},
    "G2rSparseMcoOffLoadOp": {"Ta": 2, "Ts": 2},
    "G2rSparseMcoMaskLoadOp": {"Ta": 1, "Ts": 1},
    "G2sSparseMcoValLoadOp": {"Ta": 28, "Ts": 2},
    "G2sMatrixBLoadOp": {"Ta": 82, "Ts": 3},
    "S2sRestoreMatrixAOp": {"Ta": 42, "Ts": 42},
    "S2rAValLoadOp": {"Ta": 1, "Ts": 1},
    "S2rBValLoadOp": {"Ta": 2, "Ts": 2},
    "CalculateOp": {"Ta": 2, "Ts": 2}
}



class Pipeline:
    stages: List[List[str]]
    shifts: List[int]

    def __init__(
        self,
        stages: List[List[str]],
        shifts: List[int]        
    ):
        self.stages = list(stages)
        self.shifts = list(shifts)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def max_shift(self) -> int:
        return max(self.shifts)

    @property
    def nbuf(self) -> int:
        return self.max_shift + 1
    


def fill_dispatch_queue(dispatch_queue:List[Tuple[str, int]], k:int, pipeline:Pipeline):
    # fill_len = sum(pipeline_shifts) + max(pipeline_shifts)
    # nbuf = max(pipeline_shifts) + 1
    fill_len = sum(pipeline.shifts) + max(pipeline.shifts)
    nbuf = pipeline.nbuf

    #! used in fill_dispatch() & loop_step_dispatch() & remainder_dispatch() & 
    pipeline_history = {}
    for stage in pipeline.stages:
        for op in stage:
            pipeline_history[op] = 0
    
    def short_pipeline_dispatch(dispatch_queue, pipeline):
        # print("short_pipeline_dispatch")
        stages = pipeline.stages
        shifts = pipeline.shifts
        
        def dump_short_pipeline_i(i, k_total):
            op_list: List[Tuple[str, int]] = []
            stage_num = len(stages)
            nbuf_num = max(shifts) + 1
            #> 1. 确定 op list
            for stage_i in range(stage_num):
                shift_val = sum(shifts[: stage_i])
                if 0 <= i - shift_val < k_total:
                    for op in stages[stage_i]:
                        op_list.append((op, i - shift_val))
            return op_list


        def dump_short_parallel_k_total(k_total):
            op_lists: List[Tuple[str, int]] = []
            for i in range(k_total + sum(shifts)):
                op_lists.extend(dump_short_pipeline_i(i, k_total))
            return op_lists

        dispatch_queue.extend(dump_short_parallel_k_total(k))


    def fill_dispatch(dispatch_queue: List[Tuple[str, int]], pipeline: Pipeline):
        # print("fill_dispatch")
        
        def dump_fill_body():
            # All stages start from 0
            stage_counter = [0 for _ in range(len(pipeline.stages))]
            
            # Stage k has offset -(shift0 + shift1 + ... + shiftk-1)
            # The first stage has offset 0
            stage_idx_offsets = [
                -sum(pipeline.shifts[:i]) for i in range(len(pipeline.stages))
            ]

            op_lists : List[Tuple[str, int]] = []
            total_step = 0
            for i in range(len(pipeline.stages)):
                # 最后一个
                nsteps = (
                    pipeline.shifts[i]
                    if i < len(pipeline.shifts)
                    else pipeline.max_shift
                )

                for _ in range(nsteps):
                    # Stage 0; Stage 0, 1; Stage 0, 1, 2; ...; Stage 0, 1, 2, ..., nstages - 1
                    # op_calls = [] # [op0, op1, op2, ...]
                    op_list : List[Tuple[str, int]] = []
                    for stage_idx, stage in enumerate(pipeline.stages[: i + 1]):
                        for op in stage:
                            # op_calls.append(op)
                            current_id = pipeline_history[op]
                            pipeline_history[op] += 1
                            op_list.append((op, current_id))
                        stage_counter[stage_idx] += 1
                    total_step += 1
                    
                    op_lists.extend(op_list)
            return op_lists
        
        op_lists = dump_fill_body()
        dispatch_queue.extend(op_lists)
        # print(dispatch_queue) 
        pass

    def loop_step_dispatch(dispatch_queue: List[Tuple[str, int]], pipeline: Pipeline):
        # print("loop_step_dispatch")
        # Stage k has buf_offset (shiftk + shiftk+1 + ... + shiftnstages-1)
        # The last stage has buf_offset max_shift due to 如Full
        stage_buf_offsets = [
            sum(pipeline.shifts[i:]) + pipeline.max_shift
            for i in range(len(pipeline.stages))
        ]

        # Stage k has offset -(shift0 + shift1 + ... + shiftk-1)
        # The first stage has offset 0
        stage_idx_offsets = [
            -sum(pipeline.shifts[:i]) for i in range(len(pipeline.stages))
        ]

        # for i in range(len(pipeline.stages)):
        #     nsteps = (pipeline.shifts[i] if i < len(pipeline.shifts) else pipeline.max_shift)
        #     for _ in range(nsteps):
        #         # Stage 0; Stage 0, 1; Stage 0, 1, 2; ...; Stage 0, 1, 2, ..., nstages - 1
        #         for stage_idx, stage in enumerate(pipeline.stages[: i + 1]):
        #             for op in stage:
        #                 current_id = pipeline_history[op]
        #                 pipeline_history[op] += 1
        #                 dispatch_queue.append((op, current_id))
        for step in range(pipeline.nbuf):
            for stage_idx, stage in enumerate(pipeline.stages):
                for op in stage:
                    current_id = pipeline_history[op]
                    pipeline_history[op] += 1
                    dispatch_queue.append((op, current_id))
        pass

    def remainder_dispatch(dispatch_queue: List[Tuple[str, int]], pipeline: Pipeline, i: int, k: int):
        # print("remainder_dispatch")
        
        def dump_remains_r(r: int):
            # Stage k has buf_offset (shiftk + shiftk+1 + ... + shiftnstages-1)
            # The last stage has buf_offset 0
            stage_buf_offsets = [
                sum(pipeline.shifts[i:]) + pipeline.max_shift
                for i in range(len(pipeline.stages))
            ]

            # Stage k has offset -(shift0 + shift1 + ... + shiftk-1)
            # The first stage has offset 0
            stage_idx_offsets = [
                -sum(pipeline.shifts[:i]) for i in range(len(pipeline.stages))
            ]
            op_lists : List[Tuple[str, int]] = []
            for step in range(r):
                op_list: List[Tuple[str, int]] = []
                for stage_idx, stage in enumerate(pipeline.stages):
                    for op in stage:
                        current_id = pipeline_history[op]
                        pipeline_history[op] += 1
                        op_list.append((op, current_id))
                op_lists.extend(op_list)
            return op_lists
        
        
        remain = k - i
        if remain == 0:
            return
        dispatch_queue.extend(dump_remains_r(remain))
        pass

    def empty_dispatch(dispatch_queue: List[Tuple[str, int]], pipeline: Pipeline, i: int, k: int):
        # print("empty_dispatch")
        def dump_empty_after_remain_r(r: int):
            # Stage 0 is completed
            # Stage 1, 2, 3, ... starts from [idx - shift0, idx - shift0 - shift1, idx - shift0 - shift1 - shift2, ...]
            stage_counter = [
                r + sum(pipeline.shifts[i:]) + pipeline.max_shift
                for i in range(len(pipeline.stages))
            ]

            stage_idx_offsets = [
                r - sum(pipeline.shifts[:i])
                for i in range(len(pipeline.stages))
            ]

            op_lists: List[Tuple[str, int]] = []

            for i in range(1, len(pipeline.stages)):
                nsteps = pipeline.shifts[i - 1]
                for _ in range(nsteps):
                    # Stage 1, 2, ... nstages - 1; Stage 2, 3, ... nstages - 1; ...; Stage nstages - 1
                    for stage_idx, stage in enumerate(
                        pipeline.stages[i:], start=i
                    ):
                        for op in stage:
                            current_id = pipeline_history[op]
                            pipeline_history[op] += 1
                            op_lists.append((op, current_id))
            return op_lists
        remain = k - i
        if remain == 0:
            return
        dispatch_queue.extend(dump_empty_after_remain_r(remain))
        pass

    if k <= fill_len:
        short_pipeline_dispatch(dispatch_queue, pipeline)
    else:
        fill_dispatch(dispatch_queue, pipeline)
        # for (i = fill_len; i + nbuf <= k; i += nbuf)
        i = fill_len
        while i + nbuf <= k:
            loop_step_dispatch(dispatch_queue, pipeline)
            i += nbuf
        # for i in range(fill_len, k - nbuf + 1, nbuf):
        #     loop_step_dispatch(dispatch_queue, pipeline)
        remainder_dispatch(dispatch_queue, pipeline, i, k)
        empty_dispatch(dispatch_queue, pipeline, i, k)

def add_random():
    return 0
    # return random.uniform(0, 0.2)

def simulator(dispatch_queue, dependencies, plan_latency):
    def process_consumer_to_producer(dependencies):
        consumer_to_producer = {}
        for producer, consumer in dependencies:
            if consumer not in consumer_to_producer.keys():
                consumer_to_producer[consumer] = []
            if producer not in consumer_to_producer.keys():
                consumer_to_producer[producer] = []
            consumer_to_producer[consumer].append(producer)
        return consumer_to_producer
    """
    pipeline_status = {
        "op1" : {
            1: [start],
            2: [start]
        }
    }
    """
    # pipeline_status = {
    # }
    pipeline_status = {}
    consumer_to_producer = process_consumer_to_producer(dependencies)
    ts_current = 0
    t_end = 0
    def get_op_latency(op):
        ta = plan_latency[op]["Ta"]
        ts = plan_latency[op]["Ts"]
        return ta, ts

    def get_start_time(op, i, producer_list):
        nonlocal ts_current, t_end, pipeline_status
        start = ts_current
        for producer_op in producer_list:
            if producer_op not in pipeline_status.keys():
                raise ValueError(f"{op} with {i}'s producer op {producer_op} not in pipeline_status")
            if i not in pipeline_status[producer_op].keys():
                raise ValueError(f"{op} with {i}'s producer op {producer_op} with {i} not in pipeline_status[producer_op]")
            ta, ts = get_op_latency(producer_op)
            start = max(start, pipeline_status[producer_op][i] + ta + add_random())
        if op not in pipeline_status.keys():
            pipeline_status[op] = {}
        pipeline_status[op][i] = start
        # if op == "G2sSparseIndexLoadOp":
        #     print(f"Stage 1, i = {i}, start = {start}")
        # elif op == "G2sMatrixBLoadOp":
        #     print(f"Stage 2, i = {i}, start = {start}")
        # elif op == "S2rBValLoadOp":
        #     print(f"Stage 3, i = {i}, start = {start}")
        ta, ts = get_op_latency(op)
        ts_current = start + ts + add_random()
        t_end = ts_current + ta + add_random()


    for op, i in dispatch_queue:
        producer_list = consumer_to_producer[op]
        get_start_time(op, i, producer_list)
    
    # print(t_end)
    return t_end

# ================== 收集真实时间 ==================
json_file = "/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/plan_results.json"
true_times = []
true_occupancy = {}
with open(json_file, "r") as f:
    data = json.load(f)
    for plan in data:
        pid = plan["plan_id"]
        true_times.append((pid, plan["time_ms"]))
        true_occupancy[pid] = plan["block_num"]
        # true_dict[pid] = {
        #     "time_ms": plan["time_ms"],
        #     "block_num": plan["block_num"],
        # }


true_dict = dict(true_times)
true_times.sort(key=lambda x: x[1])


all_plan_filepath = "/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/filtered_plans.txt"
# all_plan_filepath = "/workspace/sparsene/examples/src_fp32/acc/testbed/plans.txt"
# ================== 收集模拟器预测时间 ==================
pred_times = []
start = time.time()
with open(all_plan_filepath, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        dispatch_queue = []

        plan_number, plan_ops, pipeline_shifts = parse_plan(line)

        stages = []
        for op_name, i, stage_id in plan_ops:
            # 确保 stages 至少有 stage_id+1 个元素
            while len(stages) <= stage_id:
                stages.append([])
            stages[stage_id].append(op_name)

        pipeline = Pipeline(stages, pipeline_shifts)

        #! 模拟pipeline中step长度
        k = 5

        fill_dispatch_queue(dispatch_queue, k, pipeline)

        t_end = simulator(dispatch_queue, dependencies, plan_latency)

        #! 考虑occupancy对t_end做修正
        # real_time_ms = true_dict[plan_number]
        # if real_time_ms > 1.9:
        #     t_end *= 1.3
        # elif real_time_ms > 1.8: 
        #     t_end = t_end * 1.3
        # elif real_time_ms > 1.7:
        #     t_end = t_end * 1.2
        # elif real_time_ms > 1.6:
        #     t_end = t_end * 1.3

        # print(f"{plan_number}, {t_end}")
        pred_times.append((plan_number, t_end))
end = time.time()
print(f"耗时: {end - start:.6f} 秒")
# 排序（预测时间从小到大）
pred_times.sort(key=lambda x: x[1])

for i in range(50):
    print(pred_times[i][0], end=',')

def pretty_print_metrics(metrics: dict):
    for k, v in metrics.items():
        if isinstance(v, tuple):  # 相关系数返回 (value, p-value)
            val, pval = v
            print(f"{k}: {float(val):.4f}, p={float(pval):.4f}")
        else:
            print(f"{k}: {float(v):.4f}")

# ================== 寻找相同切分方式下，shift!=1比shift=1更快的plan ==================
# pred_times.sort(key=lambda x: x[0])
# for plan_number, t_end in pred_times:
#     print(f"{plan_number}, {t_end}")

# pred_dict = dict(pred_times)
# for plan_number in range(0, len(pred_times)):
#     if plan_number % 9 != 0:
#         if pred_dict[(plan_number // 9) * 9] > pred_dict[plan_number]: # 相同切分方式的plan有9个，0 ~ 8, 9 ~ 17, ...
#             print(f"plan_number = {plan_number}, pred_time = {pred_dict[plan_number]}, shift=1 plan {(plan_number // 9) * 9}, predtime = {pred_dict[(plan_number // 9) * 9]}")


# ================== 评估 ==================
metrics = evaluate_ranking_new(true_times, pred_times, topk=200)
print("相似度指标 k = 200:")
pretty_print_metrics(metrics)
metrics = evaluate_ranking_new(true_times, pred_times, topk=50)
print("相似度指标 k = 50:")
pretty_print_metrics(metrics)
metrics = evaluate_ranking_new(true_times, pred_times, topk=20)
print("相似度指标 k = 20:")
pretty_print_metrics(metrics)
metrics = evaluate_ranking_new(true_times, pred_times, topk=10)
print("相似度指标：")
pretty_print_metrics(metrics)


print("=" * 50, "random", "=" * 50)
arr = [i for i in range(len(pred_times))]
random.shuffle(arr)  # 原地打乱
random_times = []
for i in arr:
    random_times.append((arr[i], i))
metrics = evaluate_ranking_new(true_times, random_times, topk=200)
print("相似度指标 k = 200:")
pretty_print_metrics(metrics)
metrics = evaluate_ranking_new(true_times, random_times, topk=50)
print("相似度指标 k = 50:")
pretty_print_metrics(metrics)
metrics = evaluate_ranking_new(true_times, random_times, topk=20)
print("相似度指标 k = 20:")
pretty_print_metrics(metrics)
metrics = evaluate_ranking_new(true_times, random_times, topk=10)
print("相似度指标：")
pretty_print_metrics(metrics)


mismatches = find_rank_mismatches(true_times, pred_times)
# print(mismatches)


# print("=" * 50 + "half eval" + "=" * 50)
# # 保留前一半
# half_len = len(true_times) // 2
# half_len = 50
# # true_times_half = true_times[:half_len]
# pred_times_half = pred_times[:half_len]
# true_times_half = [(plan_no, true_dict[plan_no]) for plan_no, plan_time in pred_times_half]
# true_times_half.sort(key=lambda x: x[1])

# metrics = evaluate_ranking_new(true_times_half, pred_times_half, topk=50)
# print("相似度指标：")
# pretty_print_metrics(metrics)
# mismatches = find_rank_mismatches(true_times_half, pred_times_half)
# print(mismatches)


exit(0)

#=========================================画图==========================================
import matplotlib.pyplot as plt
import numpy as np

# 示例输入
# pred_times = [(0, 1.5), (1, 2.0), (2, 0.8)]
# true_times = [(0, 1.6), (1, 1.9), (2, 1.0)]

def plot_speedup(pred_times, true_times):
    # 转换成字典便于查找
    pred_dict = dict(pred_times)
    true_dict = dict(true_times)

    # 找 baseline: true_times 中位数
    sorted_true = sorted(true_times, key=lambda x: x[1])
    baseline_plan, baseline_time = sorted_true[len(sorted_true)//2]

    # 计算真实加速比
    true_speedup = {pid: baseline_time / t for pid, t in true_times}
    # 计算预测加速比（用 baseline plan 的预测时间）
    baseline_pred_time = pred_dict[baseline_plan]
    pred_speedup = {pid: baseline_pred_time / pred_dict[pid] for pid in pred_dict}


    common_plans = sorted(set(true_dict.keys()) & set(pred_dict.keys()))

    # 按真实加速比排序
    # sorted_plans = sorted(true_speedup.keys(), key=lambda pid: true_speedup[pid])
    sorted_plans = sorted(common_plans, key=lambda pid: true_speedup[pid])

    # common_plans = sorted(set(true_dict.keys()) & set(pred_dict.keys()))
    x = np.arange(len(sorted_plans))

    true_y = [true_speedup[pid] for pid in sorted_plans]
    print(true_speedup[133])
    pred_y = [pred_speedup[pid] for pid in sorted_plans]

    # 绘制
    plt.figure(figsize=(10,6))
    plt.scatter(x, true_y, label="True Speedup", color="blue", marker="o")
    plt.scatter(x, pred_y, label="Pred Speedup", color="red", marker="x")

    # 画线方便对比
    plt.plot(x, true_y, color="blue", alpha=0.5, linestyle="-")
    plt.plot(x, pred_y, color="red", alpha=0.5, linestyle="--")

    plt.xlabel("Plan index (sorted by true speedup)")
    plt.ylabel("Speedup relative to baseline plan {}".format(baseline_plan))
    plt.legend()
    plt.title("True vs Predicted Speedup (baseline plan {})".format(baseline_plan))
    plt.grid(True)
    plt.savefig("/workspace/sparsene/python/sparsene/simulator/all_plan_predict.jpg")
    plt.show()

# 调用
plot_speedup(pred_times, true_times)
