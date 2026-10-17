#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence, Tuple


SHIFT_RE = re.compile(r"\|\((\d+)\)>")


@dataclass
class Plan:
    plan_id: int
    stages: List[List[str]]
    shifts: List[int]
    op_to_stage: Dict[str, int]


@dataclass
class ClusterResult:
    cluster_id: int
    member_ids: List[int]
    medoid_id: int
    rep_ids: List[int]
    rep_times: List[float]
    est_lb: float
    est_ub: float
    est_center: float
    true_best: float


def parse_plan_line(line: str) -> Plan:
    line = line.strip()
    if not line:
        raise ValueError("Empty line")

    head, body = line.split(",", 1)
    plan_id = int(head.strip())

    shifts = [int(x) for x in SHIFT_RE.findall(body)]
    segments = SHIFT_RE.split(body)
    stage_texts = [segments[0]] + [segments[i] for i in range(2, len(segments), 2)]

    stages: List[List[str]] = []
    for text in stage_texts:
        ops = [x.strip() for x in text.split(",") if x.strip()]
        stages.append(ops)

    op_to_stage: Dict[str, int] = {}
    for sidx, ops in enumerate(stages):
        for op in ops:
            op_to_stage[op] = sidx

    return Plan(
        plan_id=plan_id,
        stages=stages,
        shifts=shifts,
        op_to_stage=op_to_stage,
    )


def load_plans(path: Path) -> Dict[int, Plan]:
    plans: Dict[int, Plan] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            p = parse_plan_line(raw)
            plans[p.plan_id] = p
    return plans


def load_results(path: Path, metric: str) -> Dict[int, float]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    out: Dict[int, float] = {}
    for row in rows:
        if metric not in row:
            raise KeyError(f"Metric {metric} is missing in result row")
        out[int(row["plan_id"])] = float(row[metric])
    return out


def kendall_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> float:
    common = [x for x in seq_a if x in set(seq_b)]
    if len(common) < 2:
        return 0.0
    pos_b = {op: i for i, op in enumerate(seq_b)}

    total = 0
    discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            total += 1
            a_i, a_j = common[i], common[j]
            if pos_b[a_i] > pos_b[a_j]:
                discordant += 1
    if total == 0:
        return 0.0
    return discordant / total


def build_distance_fn(
    plans: Dict[int, Plan],
    alpha: float,
    beta: float,
    gamma: float,
):
    all_ops = sorted({op for p in plans.values() for op in p.op_to_stage})
    max_stage_count = max(len(p.stages) for p in plans.values())
    max_shift_len = max(len(p.shifts) for p in plans.values())

    def distance(pid_a: int, pid_b: int) -> float:
        pa = plans[pid_a]
        pb = plans[pid_b]

        # Stage assignment distance.
        stage_mismatch = 0
        for op in all_ops:
            sa = pa.op_to_stage.get(op, -1)
            sb = pb.op_to_stage.get(op, -1)
            if sa != sb:
                stage_mismatch += 1
        stage_dist = stage_mismatch / max(1, len(all_ops))

        # In-stage order distance.
        order_sum = 0.0
        for sid in range(max_stage_count):
            sa_ops = pa.stages[sid] if sid < len(pa.stages) else []
            sb_ops = pb.stages[sid] if sid < len(pb.stages) else []
            order_sum += kendall_distance(sa_ops, sb_ops)
        order_dist = order_sum / max(1, max_stage_count)

        # Shift vector distance.
        shift_sum = 0
        for i in range(max_shift_len):
            va = pa.shifts[i] if i < len(pa.shifts) else 0
            vb = pb.shifts[i] if i < len(pb.shifts) else 0
            shift_sum += abs(va - vb)
        shift_dist = shift_sum / max(1, max_shift_len)

        return alpha * stage_dist + beta * order_dist + gamma * shift_dist

    return distance


def nearest_neighbor_scale(plan_ids: List[int], distance_fn) -> float:
    n = len(plan_ids)
    if n <= 1:
        return 0.0
    nearest = []
    for i in range(n):
        pid_i = plan_ids[i]
        best = float("inf")
        for j in range(n):
            if i == j:
                continue
            d = distance_fn(pid_i, plan_ids[j])
            if d < best:
                best = d
        nearest.append(best)
    return median(nearest)


def compute_medoid(member_ids: List[int], distance_fn) -> int:
    if len(member_ids) == 1:
        return member_ids[0]
    best_id = member_ids[0]
    best_sum = float("inf")
    for pid in member_ids:
        total = 0.0
        for qid in member_ids:
            if pid == qid:
                continue
            total += distance_fn(pid, qid)
        if total < best_sum:
            best_sum = total
            best_id = pid
    return best_id


def cluster_by_threshold(
    plan_ids: List[int],
    distance_fn,
    threshold: float,
    max_iter: int = 6,
) -> Dict[int, List[int]]:
    # Init with greedy pass.
    clusters: List[List[int]] = []
    medoids: List[int] = []
    for pid in plan_ids:
        best_idx = -1
        best_dist = float("inf")
        for cidx, mid in enumerate(medoids):
            d = distance_fn(pid, mid)
            if d < best_dist:
                best_dist = d
                best_idx = cidx
        if best_idx >= 0 and best_dist <= threshold:
            clusters[best_idx].append(pid)
        else:
            clusters.append([pid])
            medoids.append(pid)

    # Refinement.
    for _ in range(max_iter):
        new_medoids = [compute_medoid(members, distance_fn) for members in clusters]
        new_clusters = [[] for _ in new_medoids]

        for pid in plan_ids:
            best_idx = 0
            best_dist = float("inf")
            for cidx, mid in enumerate(new_medoids):
                d = distance_fn(pid, mid)
                if d < best_dist:
                    best_dist = d
                    best_idx = cidx
            if best_dist <= threshold:
                new_clusters[best_idx].append(pid)
            else:
                new_clusters.append([pid])
                new_medoids.append(pid)

        # Remove empties.
        compact_clusters = [c for c in new_clusters if c]
        compact_medoids = [compute_medoid(c, distance_fn) for c in compact_clusters]

        old_signature = sorted((tuple(sorted(c)) for c in clusters))
        new_signature = sorted((tuple(sorted(c)) for c in compact_clusters))
        clusters, medoids = compact_clusters, compact_medoids
        if old_signature == new_signature:
            break

    return {idx: sorted(members) for idx, members in enumerate(clusters)}


def choose_representatives(
    member_ids: List[int],
    medoid_id: int,
    runtime: Dict[int, float],
    max_reps: int,
) -> List[int]:
    reps = [medoid_id]
    if max_reps >= 2:
        best_id = min(member_ids, key=lambda x: runtime.get(x, float("inf")))
        if best_id not in reps:
            reps.append(best_id)
    return reps[:max_reps]


def estimate_uncertainty_lambda(clusters: Dict[int, List[int]], medoids: Dict[int, int], distance_fn, runtime: Dict[int, float]) -> float:
    xs: List[float] = []
    ys: List[float] = []
    for cid, members in clusters.items():
        mid = medoids[cid]
        t_mid = runtime[mid]
        for pid in members:
            d = distance_fn(pid, mid)
            y = abs(runtime[pid] - t_mid)
            xs.append(d)
            ys.append(y)

    den = sum(x * x for x in xs)
    if den <= 1e-12:
        return 0.0
    num = sum(x * y for x, y in zip(xs, ys))
    return num / den


def estimate_cluster_bounds(
    member_ids: List[int],
    rep_ids: List[int],
    runtime: Dict[int, float],
    distance_fn,
    lambda_unc: float,
) -> Tuple[float, float, float]:
    rep_times = [runtime[r] for r in rep_ids]
    center = sum(rep_times) / len(rep_times)

    # Distance to nearest representative as structure uncertainty proxy.
    dist_radius = 0.0
    for pid in member_ids:
        best = min(distance_fn(pid, rid) for rid in rep_ids)
        dist_radius += best
    dist_radius /= max(1, len(member_ids))

    unc = lambda_unc * dist_radius
    lb = min(rep_times) - unc
    ub = max(rep_times) + unc
    return lb, ub, center


def predict_from_reps(pid: int, rep_ids: List[int], runtime: Dict[int, float], distance_fn, temp: float = 0.2) -> Tuple[float, float]:
    dists = [distance_fn(pid, rid) for rid in rep_ids]
    ws = [math.exp(-d / max(temp, 1e-6)) for d in dists]
    s = sum(ws)
    if s <= 1e-12:
        mean = sum(runtime[r] for r in rep_ids) / len(rep_ids)
        std = 0.0
        return mean, std

    ws = [w / s for w in ws]
    vals = [runtime[r] for r in rep_ids]
    mean = sum(w * v for w, v in zip(ws, vals))
    var = sum(w * (v - mean) ** 2 for w, v in zip(ws, vals))
    std = math.sqrt(max(var, 0.0))
    return mean, std


def fine_search_beam(
    member_ids: List[int],
    rep_ids: List[int],
    runtime: Dict[int, float],
    distance_fn,
    beam_width: int,
    budget: int,
) -> List[int]:
    cands = [pid for pid in member_ids if pid not in set(rep_ids)]
    scored = []
    for pid in cands:
        mean, _ = predict_from_reps(pid, rep_ids, runtime, distance_fn)
        scored.append((mean, pid))
    scored.sort(key=lambda x: x[0])
    return [pid for _, pid in scored[: min(beam_width, budget, len(scored))]]


def fine_search_ucb(
    member_ids: List[int],
    rep_ids: List[int],
    runtime: Dict[int, float],
    distance_fn,
    budget: int,
    beta: float,
    seed: int,
) -> List[int]:
    rng = random.Random(seed)
    remaining = [pid for pid in member_ids if pid not in set(rep_ids)]
    selected: List[int] = []

    for _ in range(min(budget, len(remaining))):
        best_pid = None
        best_score = float("inf")
        for pid in remaining:
            mean, std = predict_from_reps(pid, rep_ids + selected, runtime, distance_fn)
            jitter = rng.uniform(-1e-6, 1e-6)
            lcb = mean - beta * std + jitter
            if lcb < best_score:
                best_score = lcb
                best_pid = pid
        if best_pid is None:
            break
        selected.append(best_pid)
        remaining.remove(best_pid)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-based group pruning for ACC pipeline plans.")
    parser.add_argument("--plans", type=Path, required=True, help="Path to filtered_plans.txt")
    parser.add_argument("--results", type=Path, required=True, help="Path to plan_results_exchange.json")
    parser.add_argument("--metric", type=str, default="time_ms", choices=["time_ms", "time_ms_exchange_xy"], help="Runtime metric field")
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=-1.0, help="Distance threshold. <0 means auto")
    parser.add_argument("--max-reps", type=int, default=2, choices=[1, 2])
    parser.add_argument("--group-keep-ratio", type=float, default=0.35, help="Keep top ratio groups by estimated LB")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--fine-budget-per-group", type=int, default=3)
    parser.add_argument("--fine-mode", type=str, default="beam", choices=["beam", "ucb"])
    parser.add_argument("--ucb-beta", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("cluster_search_report.json"))
    args = parser.parse_args()

    plans = load_plans(args.plans)
    runtime_all = load_results(args.results, args.metric)

    common_ids = sorted(set(plans).intersection(runtime_all))
    if not common_ids:
        raise RuntimeError("No overlapping plan ids between plans and results")

    plans = {pid: plans[pid] for pid in common_ids}
    runtime = {pid: runtime_all[pid] for pid in common_ids}

    distance_fn = build_distance_fn(plans, args.alpha, args.beta, args.gamma)

    auto_scale = nearest_neighbor_scale(common_ids, distance_fn)
    threshold = args.threshold if args.threshold > 0 else max(0.12, auto_scale * 1.6)

    clusters = cluster_by_threshold(common_ids, distance_fn, threshold)
    medoids = {cid: compute_medoid(members, distance_fn) for cid, members in clusters.items()}

    lambda_unc = estimate_uncertainty_lambda(clusters, medoids, distance_fn, runtime)

    cluster_results: List[ClusterResult] = []
    rep_eval_ids: List[int] = []
    for cid, members in clusters.items():
        medoid = medoids[cid]
        reps = choose_representatives(members, medoid, runtime, args.max_reps)
        rep_eval_ids.extend(reps)
        lb, ub, center = estimate_cluster_bounds(members, reps, runtime, distance_fn, lambda_unc)
        true_best = min(runtime[pid] for pid in members)
        cluster_results.append(
            ClusterResult(
                cluster_id=cid,
                member_ids=members,
                medoid_id=medoid,
                rep_ids=reps,
                rep_times=[runtime[r] for r in reps],
                est_lb=lb,
                est_ub=ub,
                est_center=center,
                true_best=true_best,
            )
        )

    cluster_results.sort(key=lambda c: c.est_lb)

    n_keep = max(1, int(math.ceil(len(cluster_results) * args.group_keep_ratio)))
    kept = cluster_results[:n_keep]
    pruned = cluster_results[n_keep:]

    refined_evals: Dict[int, List[int]] = {}
    for c in kept:
        if args.fine_mode == "beam":
            new_ids = fine_search_beam(
                c.member_ids,
                c.rep_ids,
                runtime,
                distance_fn,
                args.beam_width,
                args.fine_budget_per_group,
            )
        else:
            new_ids = fine_search_ucb(
                c.member_ids,
                c.rep_ids,
                runtime,
                distance_fn,
                args.fine_budget_per_group,
                args.ucb_beta,
                args.seed,
            )
        refined_evals[c.cluster_id] = new_ids

    evaluated_ids = sorted(set(rep_eval_ids + [pid for ids in refined_evals.values() for pid in ids]))

    global_best_true_id = min(runtime, key=lambda pid: runtime[pid])
    global_best_true_time = runtime[global_best_true_id]
    best_found_id = min(evaluated_ids, key=lambda pid: runtime[pid])
    best_found_time = runtime[best_found_id]

    summary = {
        "n_plans": len(common_ids),
        "n_clusters": len(cluster_results),
        "threshold": threshold,
        "auto_scale": auto_scale,
        "lambda_uncertainty": lambda_unc,
        "n_groups_kept": len(kept),
        "n_groups_pruned": len(pruned),
        "n_representative_evals": len(set(rep_eval_ids)),
        "n_total_evals_after_refine": len(evaluated_ids),
        "eval_ratio": len(evaluated_ids) / len(common_ids),
        "global_best_true": {
            "plan_id": global_best_true_id,
            "time": global_best_true_time,
        },
        "best_found_under_budget": {
            "plan_id": best_found_id,
            "time": best_found_time,
            "regret": best_found_time - global_best_true_time,
            "regret_pct": (best_found_time / global_best_true_time - 1.0) * 100.0,
        },
    }

    details = []
    for c in cluster_results:
        details.append(
            {
                "cluster_id": c.cluster_id,
                "n_members": len(c.member_ids),
                "member_ids": c.member_ids,
                "medoid_id": c.medoid_id,
                "representatives": [
                    {"plan_id": rid, "time": runtime[rid]} for rid in c.rep_ids
                ],
                "estimated_bound": {"lb": c.est_lb, "ub": c.est_ub, "center": c.est_center},
                "true_best": c.true_best,
                "kept": c.cluster_id in {x.cluster_id for x in kept},
                "refine_eval_ids": refined_evals.get(c.cluster_id, []),
            }
        )

    report = {"summary": summary, "clusters": details}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== Cluster Search Summary ===")
    print(f"plans={summary['n_plans']} clusters={summary['n_clusters']} threshold={summary['threshold']:.4f}")
    print(
        "kept_groups={} pruned_groups={} rep_evals={} total_evals={} eval_ratio={:.2%}".format(
            summary["n_groups_kept"],
            summary["n_groups_pruned"],
            summary["n_representative_evals"],
            summary["n_total_evals_after_refine"],
            summary["eval_ratio"],
        )
    )
    print(
        "best_true: id={} time={:.6f} | best_found: id={} time={:.6f} regret={:.6f} ({:.2f}%)".format(
            summary["global_best_true"]["plan_id"],
            summary["global_best_true"]["time"],
            summary["best_found_under_budget"]["plan_id"],
            summary["best_found_under_budget"]["time"],
            summary["best_found_under_budget"]["regret"],
            summary["best_found_under_budget"]["regret_pct"],
        )
    )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
