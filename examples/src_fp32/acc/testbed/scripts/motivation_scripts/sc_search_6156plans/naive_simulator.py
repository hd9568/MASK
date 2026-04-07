#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Dependency graph (producer -> consumer).
DEPENDENCIES = [
    ("G2sSparseIndexLoadOp", "G2sMatrixBLoadOp"),
    ("G2sMatrixBLoadOp", "S2rBValLoadOp"),
    ("G2rSparseMcoOffLoadOp", "G2sSparseMcoValLoadOp"),
    ("G2rSparseMcoMaskLoadOp", "S2sRestoreMatrixAOp"),
    ("G2sSparseMcoValLoadOp", "S2sRestoreMatrixAOp"),
    ("S2sRestoreMatrixAOp", "S2rAValLoadOp"),
    ("S2rAValLoadOp", "CalculateOp"),
    ("S2rBValLoadOp", "CalculateOp"),
]


PLAN_LATENCY = {
    "G2sSparseIndexLoadOp": {"Ta": 12, "Ts": 1},
    "G2rSparseMcoOffLoadOp": {"Ta": 2, "Ts": 2},
    "G2rSparseMcoMaskLoadOp": {"Ta": 1, "Ts": 1},
    "G2sSparseMcoValLoadOp": {"Ta": 28, "Ts": 2},
    "G2sMatrixBLoadOp": {"Ta": 82, "Ts": 3},
    "S2sRestoreMatrixAOp": {"Ta": 42, "Ts": 42},
    "S2rAValLoadOp": {"Ta": 1, "Ts": 1},
    "S2rBValLoadOp": {"Ta": 2, "Ts": 2},
    "CalculateOp": {"Ta": 2, "Ts": 2},
}


def parse_plan(plan_text: str) -> Tuple[int, List[Tuple[str, int, int]], List[int]]:
    """Parse one plan line into (plan_id, op list, pipeline shifts)."""
    plan_text = plan_text.strip()

    parts = re.split(r"(\|\(\d+\)>)", plan_text)
    plan_ops: List[Tuple[str, int, int]] = []
    pipeline_shifts: List[int] = []

    order = 1
    stage_id = 0
    plan_number: Optional[int] = None

    for part in parts:
        part = part.strip()
        if not part:
            continue

        m = re.match(r"\|\((\d+)\)>", part)
        if m:
            pipeline_shifts.append(int(m.group(1)))
            continue

        ops = [op.strip() for op in part.split(",") if op.strip()]
        if stage_id == 0 and ops and ops[0].isdigit():
            plan_number = int(ops[0])
            ops = ops[1:]

        for op in ops:
            plan_ops.append((op, order, stage_id))
            order += 1

        stage_id += 1

    if plan_number is None:
        raise ValueError("Missing plan id")
    if not pipeline_shifts:
        raise ValueError("Missing pipeline shifts")

    return plan_number, plan_ops, pipeline_shifts


class Pipeline:
    def __init__(self, stages: List[List[str]], shifts: List[int]):
        self.stages = list(stages)
        self.shifts = list(shifts)

    @property
    def max_shift(self) -> int:
        return max(self.shifts)

    @property
    def nbuf(self) -> int:
        return self.max_shift + 1


def build_pipeline(plan_ops: List[Tuple[str, int, int]], shifts: List[int]) -> Pipeline:
    stages: List[List[str]] = []
    for op_name, _order, stage_id in plan_ops:
        while len(stages) <= stage_id:
            stages.append([])
        stages[stage_id].append(op_name)
    return Pipeline(stages, shifts)


def fill_dispatch_queue(dispatch_queue: List[Tuple[str, int]], k: int, pipeline: Pipeline) -> None:
    fill_len = sum(pipeline.shifts) + max(pipeline.shifts)
    nbuf = pipeline.nbuf

    pipeline_history: Dict[str, int] = {}
    for stage in pipeline.stages:
        for op in stage:
            pipeline_history[op] = 0

    def short_pipeline_dispatch() -> None:
        def dump_short_pipeline_i(i: int, k_total: int) -> List[Tuple[str, int]]:
            out: List[Tuple[str, int]] = []
            for stage_i in range(len(pipeline.stages)):
                shift_val = sum(pipeline.shifts[:stage_i])
                if 0 <= i - shift_val < k_total:
                    for op in pipeline.stages[stage_i]:
                        out.append((op, i - shift_val))
            return out

        for i in range(k + sum(pipeline.shifts)):
            dispatch_queue.extend(dump_short_pipeline_i(i, k))

    def fill_dispatch() -> None:
        for i in range(len(pipeline.stages)):
            nsteps = pipeline.shifts[i] if i < len(pipeline.shifts) else pipeline.max_shift
            for _ in range(nsteps):
                for stage in pipeline.stages[: i + 1]:
                    for op in stage:
                        current_id = pipeline_history[op]
                        pipeline_history[op] += 1
                        dispatch_queue.append((op, current_id))

    def loop_step_dispatch() -> None:
        for _ in range(pipeline.nbuf):
            for stage in pipeline.stages:
                for op in stage:
                    current_id = pipeline_history[op]
                    pipeline_history[op] += 1
                    dispatch_queue.append((op, current_id))

    def remainder_dispatch(i: int) -> None:
        remain = k - i
        if remain <= 0:
            return
        for _ in range(remain):
            for stage in pipeline.stages:
                for op in stage:
                    current_id = pipeline_history[op]
                    pipeline_history[op] += 1
                    dispatch_queue.append((op, current_id))

    def empty_dispatch(i: int) -> None:
        remain = k - i
        if remain <= 0:
            return
        for stage_start in range(1, len(pipeline.stages)):
            nsteps = pipeline.shifts[stage_start - 1]
            for _ in range(nsteps):
                for stage in pipeline.stages[stage_start:]:
                    for op in stage:
                        current_id = pipeline_history[op]
                        pipeline_history[op] += 1
                        dispatch_queue.append((op, current_id))

    if k <= fill_len:
        short_pipeline_dispatch()
        return

    fill_dispatch()
    i = fill_len
    while i + nbuf <= k:
        loop_step_dispatch()
        i += nbuf
    remainder_dispatch(i)
    empty_dispatch(i)


def simulator(dispatch_queue: List[Tuple[str, int]]) -> float:
    def process_consumer_to_producer() -> Dict[str, List[str]]:
        consumer_to_producer: Dict[str, List[str]] = {}
        for producer, consumer in DEPENDENCIES:
            consumer_to_producer.setdefault(consumer, []).append(producer)
            consumer_to_producer.setdefault(producer, [])
        return consumer_to_producer

    pipeline_status: Dict[str, Dict[int, float]] = {}
    consumer_to_producer = process_consumer_to_producer()
    ts_current = 0.0
    t_end = 0.0

    def get_op_latency(op: str) -> Tuple[float, float]:
        ta = PLAN_LATENCY[op]["Ta"]
        ts = PLAN_LATENCY[op]["Ts"]
        return float(ta), float(ts)

    def get_start_time(op: str, i: int, producer_list: List[str]) -> None:
        nonlocal ts_current, t_end

        start = ts_current
        for producer_op in producer_list:
            if producer_op not in pipeline_status or i not in pipeline_status[producer_op]:
                raise ValueError(f"Dependency not ready: op={op} i={i} producer={producer_op}")
            ta, _ts = get_op_latency(producer_op)
            start = max(start, pipeline_status[producer_op][i] + ta)

        pipeline_status.setdefault(op, {})[i] = start
        ta, ts = get_op_latency(op)
        ts_current = start + ts
        t_end = ts_current + ta

    for op, i in dispatch_queue:
        if op not in PLAN_LATENCY:
            raise KeyError(f"Unknown op latency config: {op}")
        producer_list = consumer_to_producer.get(op, [])
        get_start_time(op, i, producer_list)

    return t_end


def load_true_results(path: Path, metric: str) -> Dict[int, float]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    out: Dict[int, float] = {}
    for row in rows:
        if metric in row and "plan_id" in row:
            out[int(row["plan_id"])] = float(row[metric])
    return out


def run_simulation(plans_path: Path, k: int) -> Tuple[List[Dict[str, float]], int, int, int]:
    rows: List[Dict[str, float]] = []
    n_total = 0
    n_success = 0
    n_failed = 0

    with plans_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            n_total += 1
            try:
                plan_id, plan_ops, pipeline_shifts = parse_plan(line)
                pipeline = build_pipeline(plan_ops, pipeline_shifts)
                dispatch_queue: List[Tuple[str, int]] = []
                fill_dispatch_queue(dispatch_queue, k, pipeline)
                sim_cost = simulator(dispatch_queue)
                rows.append({"plan_id": plan_id, "sim_cost": float(sim_cost)})
                n_success += 1
            except Exception:
                n_failed += 1
                continue

    rows.sort(key=lambda x: x["plan_id"])
    return rows, n_total, n_success, n_failed


def compute_rank_metrics(sim_rows: List[Dict[str, float]], true_dict: Dict[int, float], topk: int) -> Dict[str, float]:
    sim_dict = {int(x["plan_id"]): float(x["sim_cost"]) for x in sim_rows}
    common_ids = sorted(set(sim_dict).intersection(true_dict))
    if not common_ids:
        return {}

    true_rank = sorted(common_ids, key=lambda x: true_dict[x])
    sim_rank = sorted(common_ids, key=lambda x: sim_dict[x])

    true_topk = set(true_rank[: min(topk, len(true_rank))])
    sim_topk = set(sim_rank[: min(topk, len(sim_rank))])
    precision_at_k = len(true_topk.intersection(sim_topk)) / max(1, min(topk, len(common_ids)))

    sim_top1 = sim_rank[0]
    true_best = true_rank[0]

    return {
        "n_common": float(len(common_ids)),
        "precision_at_k": precision_at_k,
        "sim_top1_plan_id": float(sim_top1),
        "true_best_plan_id": float(true_best),
        "sim_top1_true_time": float(true_dict[sim_top1]),
        "true_best_time": float(true_dict[true_best]),
        "sim_top1_regret_pct": (true_dict[sim_top1] / true_dict[true_best] - 1.0) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive simulator: predict sim_cost for each plan.")
    parser.add_argument("--plans", type=Path, required=True, help="Input plans text file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--k", type=int, default=5, help="Pipeline loop length")
    parser.add_argument("--format", type=str, default="list", choices=["list", "dict"], help="Output JSON format")
    parser.add_argument("--true-results", type=Path, default=None, help="Optional true results JSON for offline evaluation")
    parser.add_argument("--true-metric", type=str, default="time_ms", help="Metric field in true results")
    parser.add_argument("--topk", type=int, default=50, help="Top-k for precision@k when true results are provided")
    args = parser.parse_args()

    sim_rows, n_total, n_success, n_failed = run_simulation(args.plans, args.k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "list":
        payload = sim_rows
    else:
        payload = {str(int(x["plan_id"])): float(x["sim_cost"]) for x in sim_rows}

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("=== Naive Simulator Summary ===")
    print(f"plans_total={n_total} success={n_success} failed={n_failed}")
    print(f"k={args.k} format={args.format}")
    print(f"output: {args.output}")

    if args.true_results is not None:
        true_dict = load_true_results(args.true_results, args.true_metric)
        metrics = compute_rank_metrics(sim_rows, true_dict, args.topk)
        if metrics:
            print("=== Optional Evaluation ===")
            print(
                "common={} precision@{}={:.4f} sim_top1={} true_best={} sim_top1_regret={:.2f}%".format(
                    int(metrics["n_common"]),
                    args.topk,
                    metrics["precision_at_k"],
                    int(metrics["sim_top1_plan_id"]),
                    int(metrics["true_best_plan_id"]),
                    metrics["sim_top1_regret_pct"],
                )
            )


if __name__ == "__main__":
    main()
