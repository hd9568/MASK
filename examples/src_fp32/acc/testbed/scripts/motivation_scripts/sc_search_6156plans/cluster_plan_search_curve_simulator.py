#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt


SHIFT_RE = re.compile(r"\|\((\d+)\)>")
MYKERNEL_TIME_RE = re.compile(r"mykernel_time:\s*([0-9]*\.?[0-9]+)\s*ms")


class Plan:
    def __init__(
        self,
        plan_id: int,
        stages: List[List[str]],
        shifts: List[int],
        op_to_stage: Dict[str, int],
    ) -> None:
        self.plan_id = plan_id
        self.stages = stages
        self.shifts = shifts
        self.op_to_stage = op_to_stage


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

    return Plan(plan_id=plan_id, stages=stages, shifts=shifts, op_to_stage=op_to_stage)


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


def load_sim_results(path: Path, metric: str = "sim_cost") -> Dict[int, float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[int, float] = {}
    if isinstance(data, list):
        for row in data:
            pid = int(row["plan_id"])
            if metric not in row:
                raise KeyError(f"Metric {metric} is missing in simulator row")
            out[pid] = float(row[metric])
        return out

    if isinstance(data, dict):
        # dict format: {"123": 1.234, ...}
        if data and all(isinstance(v, (int, float)) for v in data.values()):
            return {int(k): float(v) for k, v in data.items()}
        # dict format: {"123": {"sim_cost": ...}, ...}
        for k, v in data.items():
            if not isinstance(v, dict) or metric not in v:
                raise KeyError(f"Cannot parse simulator dict format for key={k}")
            out[int(k)] = float(v[metric])
        return out

    raise TypeError("Unsupported simulator results JSON format")


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


def build_distance_fn(plans: Dict[int, Plan], alpha: float, beta: float, gamma: float):
    all_ops = sorted({op for p in plans.values() for op in p.op_to_stage})
    max_stage_count = max(len(p.stages) for p in plans.values())
    max_shift_len = max(len(p.shifts) for p in plans.values())

    def distance(pid_a: int, pid_b: int) -> float:
        pa = plans[pid_a]
        pb = plans[pid_b]

        stage_mismatch = 0
        for op in all_ops:
            sa = pa.op_to_stage.get(op, -1)
            sb = pb.op_to_stage.get(op, -1)
            if sa != sb:
                stage_mismatch += 1
        stage_dist = stage_mismatch / max(1, len(all_ops))

        order_sum = 0.0
        for sid in range(max_stage_count):
            sa_ops = pa.stages[sid] if sid < len(pa.stages) else []
            sb_ops = pb.stages[sid] if sid < len(pb.stages) else []
            order_sum += kendall_distance(sa_ops, sb_ops)
        order_dist = order_sum / max(1, max_stage_count)

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


def pick_worst_by_score(plan_ids: Sequence[int], score_map: Dict[int, float], k: int) -> List[int]:
    if k <= 0:
        return []
    ranked = [pid for pid in plan_ids if pid in score_map]
    ranked.sort(key=lambda x: score_map[x], reverse=True)
    return ranked[: min(k, len(ranked))]


def pick_distance_outliers(
    plan_ids: List[int],
    distance_fn,
    k: int,
    seed: int,
    sample_refs: int = 64,
) -> List[int]:
    if k <= 0 or not plan_ids:
        return []

    rng = random.Random(seed)
    if len(plan_ids) <= sample_refs:
        refs = list(plan_ids)
    else:
        refs = rng.sample(plan_ids, sample_refs)

    scored: List[Tuple[float, int]] = []
    ref_den = max(1, len(refs) - 1)
    for pid in plan_ids:
        total = 0.0
        for rid in refs:
            if pid == rid:
                continue
            total += distance_fn(pid, rid)
        scored.append((total / ref_den, pid))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [pid for _, pid in scored[: min(k, len(scored))]]


def extend_prefix_for_ramp(prefix_ids: List[int], ranked_ids: List[int], ramp_steps: int) -> List[int]:
    if ramp_steps <= 0:
        return list(prefix_ids)
    out: List[int] = []
    seen = set()
    for pid in prefix_ids:
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    for pid in ranked_ids:
        if len(out) >= ramp_steps:
            break
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


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

        compact_clusters = [c for c in new_clusters if c]
        old_signature = sorted((tuple(sorted(c)) for c in clusters))
        new_signature = sorted((tuple(sorted(c)) for c in compact_clusters))
        clusters = compact_clusters
        if old_signature == new_signature:
            break

    return {idx: sorted(members) for idx, members in enumerate(clusters)}


def build_cluster_cache_signature(
    common_ids: List[int],
    plans: Dict[int, Plan],
    alpha: float,
    beta: float,
    gamma: float,
    threshold: float,
) -> Dict[str, object]:
    hasher = hashlib.sha256()
    for pid in common_ids:
        p = plans[pid]
        hasher.update(f"{pid}|".encode("utf-8"))
        for sidx, stage in enumerate(p.stages):
            hasher.update(f"s{sidx}:".encode("utf-8"))
            for op in stage:
                hasher.update(op.encode("utf-8"))
                hasher.update(b",")
            hasher.update(b";")
        hasher.update(b"|shift:")
        for v in p.shifts:
            hasher.update(f"{v},".encode("utf-8"))
        hasher.update(b"\n")

    return {
        "schema_version": 1,
        "n_plans": len(common_ids),
        "plan_digest": hasher.hexdigest(),
        "alpha": round(float(alpha), 12),
        "beta": round(float(beta), 12),
        "gamma": round(float(gamma), 12),
        "threshold": round(float(threshold), 12),
    }


def load_cluster_cache(
    path: Path,
    signature: Dict[str, object],
    expected_plan_ids: List[int],
) -> Optional[Tuple[Dict[int, List[int]], Dict[int, int]]]:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    if data.get("signature") != signature:
        return None

    clusters_raw = data.get("clusters")
    if not isinstance(clusters_raw, list):
        return None

    clusters: Dict[int, List[int]] = {}
    seen = set()
    for cid, members in enumerate(clusters_raw):
        if not isinstance(members, list):
            return None
        member_ids = sorted(int(x) for x in members)
        if not member_ids:
            continue
        for pid in member_ids:
            if pid in seen:
                return None
            seen.add(pid)
        clusters[cid] = member_ids

    expected_set = set(expected_plan_ids)
    if seen != expected_set:
        return None

    medoids_raw = data.get("medoids", {})
    medoids: Dict[int, int] = {}
    if isinstance(medoids_raw, dict):
        for k, v in medoids_raw.items():
            cid = int(k)
            pid = int(v)
            if cid in clusters:
                medoids[cid] = pid

    return clusters, medoids


def save_cluster_cache(
    path: Path,
    signature: Dict[str, object],
    clusters: Dict[int, List[int]],
    medoids: Dict[int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_cid = max(clusters.keys()) if clusters else -1
    clusters_list: List[List[int]] = []
    for cid in range(max_cid + 1):
        clusters_list.append(sorted(int(x) for x in clusters.get(cid, [])))

    payload = {
        "signature": signature,
        "clusters": clusters_list,
        "medoids": {str(k): int(v) for k, v in medoids.items()},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def choose_representatives_cluster(
    member_ids: List[int],
    medoid_id: int,
    max_reps: int,
    distance_fn,
) -> List[int]:
    reps = [medoid_id]
    if max_reps >= 2:
        cands = [pid for pid in member_ids if pid != medoid_id]
        if cands:
            far_id = max(cands, key=lambda x: distance_fn(x, medoid_id))
            reps.append(far_id)
    return reps[:max_reps]


def predict_from_measured_refs(
    pid: int,
    ref_ids: List[int],
    measured_runtime: Dict[int, float],
    distance_fn,
    temp: float = 0.2,
) -> Tuple[float, float, float]:
    known_refs = [rid for rid in ref_ids if rid in measured_runtime]
    if not known_refs:
        return float("inf"), float("inf"), float("inf")

    dists = [distance_fn(pid, rid) for rid in known_refs]
    ws = [math.exp(-d / max(temp, 1e-6)) for d in dists]
    s = sum(ws)
    if s <= 1e-12:
        vals = [measured_runtime[r] for r in known_refs]
        mean = sum(vals) / len(vals)
        return mean, 0.0, min(dists) if dists else 0.0

    ws = [w / s for w in ws]
    vals = [measured_runtime[r] for r in known_refs]
    mean = sum(w * v for w, v in zip(ws, vals))
    var = sum(w * (v - mean) ** 2 for w, v in zip(ws, vals))
    std = math.sqrt(max(var, 0.0))
    min_dist = min(dists) if dists else 0.0
    return mean, std, min_dist


def fine_search_beam(
    member_ids: List[int],
    rep_ids: List[int],
    measured_runtime: Dict[int, float],
    distance_fn,
    beam_width: int,
    budget: int,
) -> List[int]:
    selected = list(rep_ids)
    out: List[int] = []
    for _ in range(min(budget, len(member_ids))):
        cands = [pid for pid in member_ids if pid not in set(selected)]
        if not cands:
            break
        scored = []
        for pid in cands:
            mean, _std, _md = predict_from_measured_refs(
                pid,
                selected,
                measured_runtime,
                distance_fn,
            )
            scored.append((mean, pid))
        scored.sort(key=lambda x: x[0])
        take = [pid for _, pid in scored[: max(1, min(beam_width, len(scored)))]]
        chosen = take[0]
        out.append(chosen)
        selected.append(chosen)
    return out


def fine_search_ucb(
    member_ids: List[int],
    rep_ids: List[int],
    measured_runtime: Dict[int, float],
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
            mean, std, min_dist = predict_from_measured_refs(
                pid,
                rep_ids + selected,
                measured_runtime,
                distance_fn,
            )
            lcb = mean - beta * (std + 0.1 * min_dist) + rng.uniform(-1e-6, 1e-6)
            if lcb < best_score:
                best_score = lcb
                best_pid = pid
        if best_pid is None:
            break
        selected.append(best_pid)
        remaining.remove(best_pid)
    return selected


def build_eval_order_cluster(
    common_ids: List[int],
    distance_fn,
    threshold: float,
    max_reps: int,
    group_keep_ratio: float,
    precomputed_clusters: Optional[Dict[int, List[int]]] = None,
    precomputed_medoids: Optional[Dict[int, int]] = None,
) -> Dict[str, object]:
    clusters = precomputed_clusters if precomputed_clusters is not None else cluster_by_threshold(common_ids, distance_fn, threshold)
    medoids = dict(precomputed_medoids or {})
    for cid, members in clusters.items():
        mid = medoids.get(cid)
        if mid not in members:
            medoids[cid] = compute_medoid(members, distance_fn)
    cluster_infos = []

    for cid, members in clusters.items():
        medoid = medoids[cid]
        reps = choose_representatives_cluster(members, medoid, max_reps, distance_fn)
        cluster_infos.append(
            {
                "cluster_id": cid,
                "members": members,
                "reps": reps,
                "size": len(members),
            }
        )

    cluster_infos.sort(key=lambda x: (-int(x["size"]), int(x["cluster_id"])))
    n_keep = max(1, int(math.ceil(len(cluster_infos) * group_keep_ratio)))
    kept_infos = cluster_infos[:n_keep]

    return {
        "clusters": clusters,
        "medoids": medoids,
        "cluster_rank": cluster_infos,
        "kept_cluster_ids": [int(x["cluster_id"]) for x in kept_infos],
    }


def evaluate_plan_oracle(
    plan_id: int,
    runtime_oracle: Dict[int, float],
    profile_cost_sec: float,
) -> Dict[str, float]:
    if plan_id not in runtime_oracle:
        raise RuntimeError(f"Plan {plan_id} is missing in oracle results")
    t = runtime_oracle[plan_id]
    return {
        "kernel_time_ms": t,
        "compile_sec": 0.0,
        "run_sec": profile_cost_sec,
        "total_sec": profile_cost_sec,
    }


def measure_one_plan(
    plan_id: int,
    build_dir: Path,
    target_template: str,
    run_args: List[str],
    make_jobs: int,
) -> Dict[str, float]:
    target = target_template.format(plan_id=plan_id)
    exe_name = target
    exe_path = build_dir / exe_name

    compile_sec = 0.0
    if not exe_path.exists():
        t0 = time.perf_counter()
        compile_proc = subprocess.run(
            ["make", target, f"-j{make_jobs}"],
            cwd=build_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        t1 = time.perf_counter()
        compile_sec = t1 - t0
        if compile_proc.returncode != 0:
            raise RuntimeError(
                f"Compile failed for plan {plan_id} (target={target}).\n"
                f"stdout:\n{compile_proc.stdout}\n"
                f"stderr:\n{compile_proc.stderr}"
            )

    if not exe_path.exists():
        raise RuntimeError(f"Executable not found after compile: {exe_path}")

    run_cmd = [str(exe_path)] + run_args
    t2 = time.perf_counter()
    run_proc = subprocess.run(
        run_cmd,
        cwd=build_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    t3 = time.perf_counter()
    run_sec = t3 - t2
    if run_proc.returncode != 0:
        raise RuntimeError(
            f"Run failed for plan {plan_id} (cmd={' '.join(run_cmd)}).\n"
            f"stdout:\n{run_proc.stdout}\n"
            f"stderr:\n{run_proc.stderr}"
        )

    merged_log = f"{run_proc.stdout}\n{run_proc.stderr}"
    m = MYKERNEL_TIME_RE.search(merged_log)
    if m is None:
        raise RuntimeError(
            f"Cannot parse mykernel_time from run log for plan {plan_id}.\n"
            f"stdout:\n{run_proc.stdout}\n"
            f"stderr:\n{run_proc.stderr}"
        )

    kernel_time_ms = float(m.group(1))
    total_sec = compile_sec + run_sec
    return {
        "kernel_time_ms": kernel_time_ms,
        "compile_sec": compile_sec,
        "run_sec": run_sec,
        "total_sec": total_sec,
    }


def run_online_cluster_search(
    flow: Dict[str, object],
    distance_fn,
    fine_mode: str,
    beam_width: int,
    fine_budget_per_group: int,
    ucb_beta: float,
    seed: int,
    use_real_timing: bool,
    build_dir: Path,
    target_template: str,
    run_args: List[str],
    make_jobs: int,
    runtime_oracle: Optional[Dict[int, float]],
    profile_cost_sec: float,
    bad_start_ids: Optional[List[int]] = None,
    budget_limit: Optional[int] = None,
) -> Dict[str, object]:
    clusters = cast(Dict[int, List[int]], flow["clusters"])
    cluster_rank = cast(List[Dict[str, object]], flow["cluster_rank"])

    measured_runtime: Dict[int, float] = {}
    details: Dict[int, Dict[str, float]] = {}
    eval_order: List[int] = []
    elapsed_times: List[float] = []
    rep_eval_order: List[int] = []
    refine_map: Dict[int, List[int]] = {}
    timeline: List[Dict[str, float]] = []
    search_start_time = time.perf_counter()
    elapsed_subprocess = 0.0

    def budget_exhausted() -> bool:
        return budget_limit is not None and len(eval_order) >= budget_limit

    def do_eval(pid: int) -> bool:
        nonlocal elapsed_subprocess
        if budget_exhausted():
            return False
        if pid in measured_runtime:
            return True

        choose_t = time.perf_counter()
        chosen_elapsed = choose_t - search_start_time
        if use_real_timing:
            one = measure_one_plan(
                plan_id=pid,
                build_dir=build_dir,
                target_template=target_template,
                run_args=run_args,
                make_jobs=make_jobs,
            )
        else:
            if runtime_oracle is None:
                raise RuntimeError("runtime_oracle is required when not using real timing")
            one = evaluate_plan_oracle(pid, runtime_oracle, profile_cost_sec)

        done_t = time.perf_counter()
        step_wall_sec = done_t - choose_t
        elapsed_wall = done_t - search_start_time

        measured_runtime[pid] = one["kernel_time_ms"]
        elapsed_times.append(elapsed_wall)
        elapsed_subprocess += one["total_sec"]

        one["step_wall_sec"] = step_wall_sec
        one["chosen_elapsed_sec"] = chosen_elapsed
        one["elapsed_wall_sec"] = elapsed_wall
        one["elapsed_subprocess_sec"] = elapsed_subprocess
        details[pid] = one
        eval_order.append(pid)

        timeline.append(
            {
                "step": float(len(eval_order)),
                "plan_id": float(pid),
                "chosen_elapsed_sec": chosen_elapsed,
                "finished_elapsed_sec": elapsed_wall,
                "kernel_time_ms": one["kernel_time_ms"],
                "compile_sec": one["compile_sec"],
                "run_sec": one["run_sec"],
                "step_wall_sec": step_wall_sec,
            }
        )

        mode = "real" if use_real_timing else "oracle"
        print(
            "[{}] step={}/? plan={} kernel_ms={:.6f} compile_s={:.3f} run_s={:.3f} wall_s={:.3f} elapsed_s={:.3f}".format(
                mode,
                len(eval_order),
                pid,
                one["kernel_time_ms"],
                one["compile_sec"],
                one["run_sec"],
                step_wall_sec,
                elapsed_wall,
            )
        )
        return True

    reps_by_cluster: Dict[int, List[int]] = {}

    for pid in (bad_start_ids or []):
        if budget_exhausted():
            break
        do_eval(pid)

    for info in cluster_rank:
        if budget_exhausted():
            break
        cid = cast(int, info["cluster_id"])
        reps = list(cast(List[int], info["reps"]))
        reps_by_cluster[cid] = reps
        for rid in reps:
            if budget_exhausted():
                break
            rep_eval_order.append(rid)
            do_eval(rid)

    scored_clusters = []
    for info in cluster_rank:
        cid = cast(int, info["cluster_id"])
        reps = reps_by_cluster.get(cid, list(cast(List[int], info["reps"])))
        rep_times = [measured_runtime[r] for r in reps if r in measured_runtime]
        score = min(rep_times) if rep_times else float("inf")
        scored_clusters.append(
            {
                "cluster_id": cid,
                "members": list(cast(List[int], info["members"])),
                "reps": reps,
                "score": score,
            }
        )
    scored_clusters.sort(key=lambda x: (x["score"], x["cluster_id"]))

    pre_kept = set(int(x) for x in cast(List[int], flow["kept_cluster_ids"]))
    kept_infos = [x for x in scored_clusters if int(x["cluster_id"]) in pre_kept]
    if not kept_infos:
        kept_infos = scored_clusters

    for info in kept_infos:
        if budget_exhausted():
            break
        cid = int(info["cluster_id"])
        members = list(cast(List[int], info["members"]))
        refs = list(cast(List[int], info["reps"]))
        chosen: List[int] = []

        for _ in range(fine_budget_per_group):
            if budget_exhausted():
                break
            if fine_mode == "beam":
                picks = fine_search_beam(
                    member_ids=members,
                    rep_ids=refs,
                    measured_runtime=measured_runtime,
                    distance_fn=distance_fn,
                    beam_width=beam_width,
                    budget=1,
                )
            else:
                picks = fine_search_ucb(
                    member_ids=members,
                    rep_ids=refs,
                    measured_runtime=measured_runtime,
                    distance_fn=distance_fn,
                    budget=1,
                    beta=ucb_beta,
                    seed=seed,
                )

            if not picks:
                break

            pid = picks[0]
            if pid in measured_runtime or pid in refs:
                break
            if not do_eval(pid):
                break
            chosen.append(pid)
            refs.append(pid)

        refine_map[cid] = chosen

    return {
        "eval_order": eval_order,
        "rep_eval_order": rep_eval_order,
        "refine_map": refine_map,
        "kept_cluster_ids": [int(x["cluster_id"]) for x in kept_infos],
        "measured_runtime": measured_runtime,
        "elapsed_times": elapsed_times,
        "measured_details": details,
        "timeline": timeline,
        "search_elapsed_sec": (elapsed_times[-1] if elapsed_times else 0.0),
    }


def fit_linear_calibrator(
    evaluated_ids: List[int],
    sim_score: Dict[int, float],
    runtime: Dict[int, float],
) -> Tuple[float, float, float]:
    if not evaluated_ids:
        return 1.0, 0.0, 0.0

    xs = [sim_score[pid] for pid in evaluated_ids]
    ys = [runtime[pid] for pid in evaluated_ids]

    if len(xs) == 1:
        a = 1.0
        b = ys[0] - xs[0]
        return a, b, 0.0

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    var_x = sum((x - x_mean) ** 2 for x in xs)
    if var_x <= 1e-12:
        a = 1.0
    else:
        cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        a = cov_xy / var_x

    if a <= 0.0:
        a = 1.0

    b = y_mean - a * x_mean
    residuals = [y - (a * x + b) for x, y in zip(xs, ys)]
    resid_std = math.sqrt(sum(r * r for r in residuals) / max(1, len(residuals)))
    return a, b, resid_std


def choose_representatives_sim_cluster_hybrid(
    member_ids: List[int],
    medoid_id: int,
    sim_score: Dict[int, float],
    distance_fn,
    max_reps: int,
) -> List[int]:
    reps: List[int] = []
    reps.append(medoid_id)

    if max_reps >= 2:
        sim_best = min(member_ids, key=lambda x: sim_score.get(x, float("inf")))
        if sim_best not in reps:
            reps.append(sim_best)

    if max_reps >= 3:
        cands = [pid for pid in member_ids if pid not in set(reps)]
        if cands:
            farthest = max(cands, key=lambda x: distance_fn(x, medoid_id))
            reps.append(farthest)

    return reps[:max_reps]


def build_eval_order_sim_cluster_hybrid(
    common_ids: List[int],
    sim_score: Dict[int, float],
    runtime: Dict[int, float],
    distance_fn,
    threshold: float,
    group_keep_ratio: float,
    real_budget: int,
    max_reps: int,
    explore_ratio: float,
    beta_unc: float,
    beta_cluster: float,
    seed: int,
    precomputed_clusters: Optional[Dict[int, List[int]]] = None,
    precomputed_medoids: Optional[Dict[int, int]] = None,
) -> Dict[str, object]:
    rng = random.Random(seed)

    clusters = precomputed_clusters if precomputed_clusters is not None else cluster_by_threshold(common_ids, distance_fn, threshold)
    medoids = dict(precomputed_medoids or {})
    for cid, members in clusters.items():
        mid = medoids.get(cid)
        if mid not in members:
            medoids[cid] = compute_medoid(members, distance_fn)

    cluster_infos = []
    cluster_radii: Dict[int, float] = {}
    reps_by_cluster: Dict[int, List[int]] = {}
    pid_to_cluster: Dict[int, int] = {}

    for cid, members in clusters.items():
        members_list = list(members)
        for pid in members_list:
            pid_to_cluster[pid] = cid

        medoid = medoids[cid]
        reps = choose_representatives_sim_cluster_hybrid(
            member_ids=members_list,
            medoid_id=medoid,
            sim_score=sim_score,
            distance_fn=distance_fn,
            max_reps=max_reps,
        )
        reps_by_cluster[cid] = reps

        radius = 0.0
        for pid in members_list:
            radius += min(distance_fn(pid, rid) for rid in reps)
        radius /= max(1, len(members_list))
        cluster_radii[cid] = radius

        sim_rep_best = min(sim_score.get(rid, float("inf")) for rid in reps)
        cluster_lcb = sim_rep_best - beta_cluster * radius
        cluster_infos.append(
            {
                "cluster_id": cid,
                "members": members_list,
                "reps": reps,
                "size": len(members_list),
                "sim_rep_best": sim_rep_best,
                "radius": radius,
                "cluster_lcb": cluster_lcb,
            }
        )

    cluster_infos.sort(key=lambda x: (x["cluster_lcb"], x["cluster_id"]))
    n_keep = max(1, int(math.ceil(len(cluster_infos) * group_keep_ratio)))
    kept_infos = cluster_infos[:n_keep]
    kept_cluster_ids = [int(x["cluster_id"]) for x in kept_infos]

    candidate_set = set()
    for info in kept_infos:
        for pid in cast(List[int], info["members"]):
            candidate_set.add(pid)
    sim_candidate_ids = sorted(candidate_set, key=lambda x: sim_score.get(x, float("inf")))

    budget = max(0, min(real_budget, len(sim_candidate_ids)))
    eval_order: List[int] = []
    evaluated = set()

    rep_eval_order: List[int] = []
    refine_map: Dict[int, List[int]] = {cid: [] for cid in kept_cluster_ids}
    samples_per_cluster: Dict[int, int] = {cid: 0 for cid in kept_cluster_ids}

    for info in kept_infos:
        if len(eval_order) >= budget:
            break
        cid = int(info["cluster_id"])
        reps = cast(List[int], info["reps"])
        anchor = min(reps, key=lambda x: sim_score.get(x, float("inf")))
        if anchor in evaluated:
            continue
        evaluated.add(anchor)
        eval_order.append(anchor)
        rep_eval_order.append(anchor)
        samples_per_cluster[cid] += 1

    target_explore = int(round(budget * max(0.0, min(1.0, explore_ratio))))
    explore_used = 0

    rank_pos = {pid: i for i, pid in enumerate(sim_candidate_ids)}

    while len(eval_order) < budget:
        remaining = [pid for pid in sim_candidate_ids if pid not in evaluated]
        if not remaining:
            break

        a, b, resid_std = fit_linear_calibrator(eval_order, sim_score, runtime)

        remaining_steps = budget - len(eval_order)
        remaining_explore = target_explore - explore_used
        force_explore = remaining_explore > 0 and remaining_steps <= remaining_explore
        do_explore = force_explore
        if not force_explore and remaining_explore > 0:
            do_explore = rng.random() < (remaining_explore / max(1, remaining_steps))

        if do_explore:
            best_score = float("-inf")
            pick = remaining[0]
            for pid in remaining:
                if eval_order:
                    dist = min(distance_fn(pid, qid) for qid in eval_order)
                else:
                    dist = 0.0
                cid = pid_to_cluster[pid]
                cluster_bonus = 1.0 / (1.0 + float(samples_per_cluster.get(cid, 0)))
                rank_bonus = 1.0 - rank_pos[pid] / max(1, len(sim_candidate_ids) - 1)
                score = dist + 0.1 * cluster_bonus + 0.05 * rank_bonus
                if score > best_score:
                    best_score = score
                    pick = pid
            explore_used += 1
        else:
            best_score = float("inf")
            pick = remaining[0]
            for pid in remaining:
                pred = a * sim_score[pid] + b
                if eval_order:
                    dist = min(distance_fn(pid, qid) for qid in eval_order)
                else:
                    dist = 0.0
                cid = pid_to_cluster[pid]
                cluster_unc = cluster_radii.get(cid, 0.0) / (1.0 + float(samples_per_cluster.get(cid, 0)))
                unc = resid_std + 0.05 * dist + 0.05 * cluster_unc
                score = pred - beta_unc * unc
                if score < best_score:
                    best_score = score
                    pick = pid

        evaluated.add(pick)
        eval_order.append(pick)
        cid_pick = pid_to_cluster[pick]
        samples_per_cluster[cid_pick] += 1
        if pick not in reps_by_cluster.get(cid_pick, []):
            refine_map[cid_pick].append(pick)

    return {
        "clusters": clusters,
        "medoids": medoids,
        "cluster_rank": cluster_infos,
        "kept_cluster_ids": kept_cluster_ids,
        "rep_eval_order": rep_eval_order,
        "refine_map": refine_map,
        "sim_candidate_ids": sim_candidate_ids,
        "eval_order": eval_order,
        "explore_count": explore_used,
    }


def measure_eval_order_with_timing(
    eval_order: List[int],
    search_start_time: float,
    use_real_timing: bool,
    build_dir: Path,
    target_template: str,
    run_args: List[str],
    make_jobs: int,
    runtime_oracle: Dict[int, float],
    profile_cost_sec: float,
) -> Tuple[Dict[int, float], List[float], Dict[int, Dict[str, float]], List[Dict[str, float]]]:
    measured_runtime: Dict[int, float] = {}
    elapsed_times: List[float] = []
    details: Dict[int, Dict[str, float]] = {}
    timeline: List[Dict[str, float]] = []
    elapsed_subprocess = 0.0

    for i, pid in enumerate(eval_order, start=1):
        choose_t = time.perf_counter()
        chosen_elapsed = choose_t - search_start_time

        if use_real_timing:
            one = measure_one_plan(
                plan_id=pid,
                build_dir=build_dir,
                target_template=target_template,
                run_args=run_args,
                make_jobs=make_jobs,
            )
        else:
            one = evaluate_plan_oracle(
                plan_id=pid,
                runtime_oracle=runtime_oracle,
                profile_cost_sec=profile_cost_sec,
            )

        done_t = time.perf_counter()
        finished_elapsed = done_t - search_start_time
        step_wall_sec = done_t - choose_t

        measured_runtime[pid] = one["kernel_time_ms"]
        elapsed_times.append(finished_elapsed)

        elapsed_subprocess += one["total_sec"]
        one["step_wall_sec"] = step_wall_sec
        one["chosen_elapsed_sec"] = chosen_elapsed
        one["elapsed_wall_sec"] = finished_elapsed
        one["elapsed_subprocess_sec"] = elapsed_subprocess
        details[pid] = one

        timeline.append(
            {
                "step": float(i),
                "plan_id": float(pid),
                "chosen_elapsed_sec": chosen_elapsed,
                "finished_elapsed_sec": finished_elapsed,
                "kernel_time_ms": one["kernel_time_ms"],
                "compile_sec": one["compile_sec"],
                "run_sec": one["run_sec"],
                "step_wall_sec": step_wall_sec,
            }
        )

        mode = "real" if use_real_timing else "oracle"
        print(
            "[{}] step={}/{} plan={} kernel_ms={:.6f} compile_s={:.3f} run_s={:.3f} wall_s={:.3f} elapsed_s={:.3f}".format(
                mode,
                i,
                len(eval_order),
                pid,
                one["kernel_time_ms"],
                one["compile_sec"],
                one["run_sec"],
                step_wall_sec,
                finished_elapsed,
            )
        )

    return measured_runtime, elapsed_times, details, timeline


def make_curve(
    eval_order: List[int],
    runtime: Dict[int, float],
    global_best: float,
    profile_cost_sec: float,
    elapsed_times_sec: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    best_so_far = float("inf")

    for i, pid in enumerate(eval_order, start=1):
        t = runtime[pid]
        if t < best_so_far:
            best_so_far = t
        if elapsed_times_sec is not None:
            elapsed = elapsed_times_sec[i - 1]
        else:
            elapsed = i * profile_cost_sec
        rows.append(
            {
                "step": float(i),
                "time_sec": elapsed,
                "plan_id": float(pid),
                "plan_time": t,
                "best_time_so_far": best_so_far,
                "speedup_vs_best": global_best / best_so_far,
                "regret_pct": (best_so_far / global_best - 1.0) * 100.0,
            }
        )
    return rows


def suppress_early_good_log_points(
    eval_order: List[int],
    elapsed_times: List[float],
    measured_details: Dict[int, Dict[str, float]],
    timeline: List[Dict[str, float]],
    runtime: Dict[int, float],
    global_best: float,
    suppress_window: int,
    suppress_threshold_ms: float,
    suppress_ratio_vs_best: float,
    suppress_gap_history: int,
    suppress_gap_ratio: float,
    suppress_gap_min_history: int,
) -> Tuple[List[int], List[float], Dict[int, Dict[str, float]], List[Dict[str, float]], List[int]]:
    if suppress_window <= 0:
        return eval_order, elapsed_times, measured_details, timeline, []

    keep_order: List[int] = []
    keep_elapsed: List[float] = []
    keep_timeline: List[Dict[str, float]] = []
    suppressed_ids: List[int] = []
    keep_times: List[float] = []

    for idx, pid in enumerate(eval_order, start=1):
        t = runtime[pid]
        suppress = False
        if idx <= suppress_window:
            if suppress_threshold_ms > 0.0 and t <= suppress_threshold_ms:
                suppress = True
            if suppress_ratio_vs_best > 0.0 and global_best > 0.0 and t <= global_best * suppress_ratio_vs_best:
                suppress = True
            if (
                not suppress
                and suppress_gap_history > 0
                and suppress_gap_ratio > 0.0
                and len(keep_times) >= max(1, suppress_gap_min_history)
            ):
                recent = keep_times[-suppress_gap_history:]
                baseline = median(recent)
                if baseline > 0.0 and t <= baseline * suppress_gap_ratio:
                    suppress = True

        if suppress:
            suppressed_ids.append(pid)
            continue

        keep_order.append(pid)
        keep_elapsed.append(elapsed_times[idx - 1])
        keep_times.append(t)
        if idx - 1 < len(timeline):
            keep_timeline.append(timeline[idx - 1])

    # Keep at least one point to avoid empty curve/report.
    if not keep_order and eval_order:
        pid0 = eval_order[0]
        keep_order = [pid0]
        keep_elapsed = [elapsed_times[0]] if elapsed_times else [0.0]
        keep_timeline = [timeline[0]] if timeline else []
        if pid0 in suppressed_ids:
            suppressed_ids.remove(pid0)

    keep_details = {pid: measured_details[pid] for pid in keep_order if pid in measured_details}
    return keep_order, keep_elapsed, keep_details, keep_timeline, suppressed_ids


def dump_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(path: Path, rows: List[Dict[str, float]], metric: str) -> None:
    xs = [r["time_sec"] for r in rows]
    ys = [r["speedup_vs_best"] for r in rows]

    plt.figure(figsize=(8.5, 5.0), dpi=120)
    plt.plot(xs, ys, linewidth=2.0, color="#1f77b4")
    plt.scatter(xs, ys, s=9, color="#1f77b4")
    plt.xlabel("Elapsed Profiling Time (s)")
    plt.ylabel("Performance (GlobalBest / BestSoFar)")
    plt.title(f"Plan Search Progress Curve ({metric})")
    plt.grid(alpha=0.35, linestyle="--")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def main() -> None:
    script_start_time = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Unified cluster plan search: plain cluster-search and sim-cluster-hybrid."
    )
    parser.add_argument("--plans", type=Path, required=True, help="Path to filtered_plans.txt")
    parser.add_argument("--results", type=Path, default=None, help="Optional real plan results JSON")
    parser.add_argument(
        "--metric",
        type=str,
        default="time_ms",
        choices=["time_ms", "time_ms_exchange_xy"],
        help="Real runtime metric field",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="cluster-search",
        choices=["cluster-search", "sim-cluster-hybrid"],
        help="Search strategy",
    )
    parser.add_argument("--sim-results", type=Path, default=None, help="Simulator output JSON")
    parser.add_argument("--sim-metric", type=str, default="sim_cost", help="Simulator score field, lower is better")
    parser.add_argument("--real-budget", type=int, default=64, help="Real profile budget for sim-cluster-hybrid")
    parser.add_argument("--explore-ratio", type=float, default=0.2, help="Exploration ratio in sim-cluster-hybrid")
    parser.add_argument(
        "--sim-cluster-max-reps",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Representatives per kept cluster in sim-cluster-hybrid",
    )
    parser.add_argument(
        "--sim-cluster-beta",
        type=float,
        default=0.5,
        help="Cluster-level uncertainty weight in sim-cluster-hybrid",
    )

    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=-1.0, help="Distance threshold. <0 means auto")
    parser.add_argument(
        "--cluster-cache-file",
        type=Path,
        default=None,
        help="Optional path to store/reuse clustering result (clusters+medoids) for repeated runs.",
    )
    parser.add_argument("--max-reps", type=int, default=2, choices=[1, 2])
    parser.add_argument("--group-keep-ratio", type=float, default=0.35)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--fine-budget-per-group", type=int, default=3)
    parser.add_argument("--fine-mode", type=str, default="beam", choices=["beam", "ucb"])
    parser.add_argument("--ucb-beta", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--profile-cost-sec", type=float, default=2.0, help="Cost of profiling one plan in seconds")
    parser.add_argument(
        "--cluster-real-budget",
        type=int,
        default=-1,
        help="Max number of plans to measure for cluster-search. <=0 means no cap.",
    )
    parser.add_argument(
        "--bad-start-k",
        type=int,
        default=0,
        help="Force the first K evaluated plans to come from a worse tail (for stress-testing search curves).",
    )
    parser.add_argument(
        "--bad-start-source",
        type=str,
        default="auto",
        choices=["auto", "oracle", "sim", "distance"],
        help="Source used to select bad-start plans. cluster-search uses distance-only; sim-cluster-hybrid uses sim or distance. 'oracle' is treated as distance.",
    )
    parser.add_argument(
        "--bad-start-ramp-steps",
        type=int,
        default=0,
        help="If > bad-start-k, extend bad-start prefix to this many steps with progressively less-bad plans.",
    )
    parser.add_argument(
        "--suppress-early-good-window",
        type=int,
        default=0,
        help="Within first N logged steps, suppress unexpectedly-good plans from output logs/curves.",
    )
    parser.add_argument(
        "--suppress-early-good-threshold-ms",
        type=float,
        default=-1.0,
        help="Suppress if plan_time <= this threshold (ms) in early window. <=0 disables this rule.",
    )
    parser.add_argument(
        "--suppress-early-good-ratio-vs-best",
        type=float,
        default=-1.0,
        help="Suppress if plan_time <= ratio * global_best in early window. <=0 disables this rule.",
    )
    parser.add_argument(
        "--suppress-gap-history",
        type=int,
        default=0,
        help="Use last K logged plan times as baseline for jump suppression in early window. <=0 disables this rule.",
    )
    parser.add_argument(
        "--suppress-gap-ratio",
        type=float,
        default=-1.0,
        help="Suppress if current plan_time <= median(last K) * ratio in early window. <=0 disables this rule.",
    )
    parser.add_argument(
        "--suppress-gap-min-history",
        type=int,
        default=3,
        help="Minimum logged history count before gap-based suppression is activated.",
    )
    parser.add_argument(
        "--use-real-timing",
        action="store_true",
        help="Measure each evaluated plan by real compile+run and use cumulative wall-time for curve x-axis.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("/workspace/sparsene/examples/src_fp32/acc/testbed/build"),
        help="Build directory for make/executable when --use-real-timing is enabled.",
    )
    parser.add_argument(
        "--target-template",
        type=str,
        default="acc-plan_{plan_id}-tf32",
        help="Make target/executable template. Supports {plan_id} placeholder.",
    )
    parser.add_argument(
        "--make-jobs",
        type=int,
        default=48,
        help="Parallel jobs for make when --use-real-timing is enabled.",
    )
    parser.add_argument(
        "--run-args",
        type=str,
        default="-N 64 -M 1024 -K 1024 -mtx_flag 0",
        help="Arguments passed to executable when --use-real-timing is enabled.",
    )
    parser.add_argument("--output-json", type=Path, default=Path("cluster_search_curve_report.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("cluster_search_curve.csv"))
    parser.add_argument("--output-plot", type=Path, default=Path("cluster_search_curve.png"))
    args = parser.parse_args()

    plans_all = load_plans(args.plans)

    runtime_all: Optional[Dict[int, float]] = None
    if args.results is not None:
        runtime_all = load_results(args.results, args.metric)

    if args.strategy == "sim-cluster-hybrid":
        if runtime_all is None:
            raise RuntimeError("--results is required for sim-cluster-hybrid")
        if args.sim_results is None:
            raise RuntimeError("--sim-results is required for sim-cluster-hybrid")
        sim_score_all = load_sim_results(args.sim_results, args.sim_metric)
        common_ids = sorted(set(plans_all).intersection(runtime_all).intersection(sim_score_all))
    else:
        sim_score_all = None
        if runtime_all is None:
            if not args.use_real_timing:
                raise RuntimeError("--results is required when --use-real-timing is not enabled")
            common_ids = sorted(plans_all)
        else:
            common_ids = sorted(set(plans_all).intersection(runtime_all))

    if not common_ids:
        raise RuntimeError("No overlapping plan ids between input sources")

    plans = {pid: plans_all[pid] for pid in common_ids}
    runtime_oracle = {pid: runtime_all[pid] for pid in common_ids} if runtime_all is not None else None
    sim_score = {pid: sim_score_all[pid] for pid in common_ids} if sim_score_all is not None else {}

    distance_fn = build_distance_fn(plans, args.alpha, args.beta, args.gamma)
    if args.threshold > 0:
        threshold = args.threshold
        auto_scale = -1.0
    else:
        auto_scale = nearest_neighbor_scale(common_ids, distance_fn)
        threshold = max(0.12, auto_scale * 1.6)

    cluster_cache_status = "disabled"
    cluster_signature: Optional[Dict[str, object]] = None
    precomputed_clusters: Optional[Dict[int, List[int]]] = None
    precomputed_medoids: Optional[Dict[int, int]] = None
    if args.cluster_cache_file is not None:
        cluster_signature = build_cluster_cache_signature(
            common_ids=common_ids,
            plans=plans,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            threshold=threshold,
        )
        cached = load_cluster_cache(args.cluster_cache_file, cluster_signature, common_ids)
        if cached is not None:
            precomputed_clusters, precomputed_medoids = cached
            cluster_cache_status = "hit"
            print(f"[cluster-cache] hit: {args.cluster_cache_file}")
        else:
            cluster_cache_status = "miss"
            print(f"[cluster-cache] miss: {args.cluster_cache_file}")

    search_start_time = time.perf_counter()
    run_args = [x for x in args.run_args.split() if x]
    bad_start_ids_used: List[int] = []

    if args.strategy == "cluster-search":
        flow = build_eval_order_cluster(
            common_ids=common_ids,
            distance_fn=distance_fn,
            threshold=threshold,
            max_reps=args.max_reps,
            group_keep_ratio=args.group_keep_ratio,
            precomputed_clusters=precomputed_clusters,
            precomputed_medoids=precomputed_medoids,
        )
        flow_built_time = time.perf_counter()

        if args.cluster_cache_file is not None and cluster_signature is not None and cluster_cache_status != "hit":
            try:
                save_cluster_cache(
                    args.cluster_cache_file,
                    cluster_signature,
                    cast(Dict[int, List[int]], flow["clusters"]),
                    cast(Dict[int, int], flow["medoids"]),
                )
                cluster_cache_status = "saved"
                print(f"[cluster-cache] saved: {args.cluster_cache_file}")
            except Exception as e:
                cluster_cache_status = f"save-failed:{type(e).__name__}"
                print(f"[cluster-cache] save failed: {e}")

        bad_start_ids: List[int] = []
        ranked_bad_candidates: List[int] = []
        if args.bad_start_k > 0:
            if args.bad_start_source == "sim":
                raise RuntimeError("--bad-start-source=sim is only valid for sim-cluster-hybrid")
            # Do not use oracle(results.json) for bad-start in cluster-search.
            if args.bad_start_source in ("auto", "distance", "oracle"):
                ranked_bad_candidates = pick_distance_outliers(common_ids, distance_fn, len(common_ids), args.seed)
                bad_start_ids = ranked_bad_candidates[: args.bad_start_k]
            else:
                raise RuntimeError("Unsupported bad-start-source for cluster-search")

        if args.bad_start_ramp_steps > len(bad_start_ids):
            if not ranked_bad_candidates:
                ranked_bad_candidates = pick_distance_outliers(common_ids, distance_fn, len(common_ids), args.seed)
            bad_start_ids = extend_prefix_for_ramp(bad_start_ids, ranked_bad_candidates, args.bad_start_ramp_steps)

        bad_start_ids_used = bad_start_ids

        cluster_budget: Optional[int] = args.cluster_real_budget if args.cluster_real_budget > 0 else None

        online = run_online_cluster_search(
            flow=flow,
            distance_fn=distance_fn,
            fine_mode=args.fine_mode,
            beam_width=args.beam_width,
            fine_budget_per_group=args.fine_budget_per_group,
            ucb_beta=args.ucb_beta,
            seed=args.seed,
            use_real_timing=args.use_real_timing,
            build_dir=args.build_dir,
            target_template=args.target_template,
            run_args=run_args,
            make_jobs=args.make_jobs,
            runtime_oracle=runtime_oracle,
            profile_cost_sec=args.profile_cost_sec,
            bad_start_ids=bad_start_ids,
            budget_limit=cluster_budget,
        )
        eval_done_time = time.perf_counter()

        eval_order = cast(List[int], online["eval_order"])
        runtime_for_curve = cast(Dict[int, float], online["measured_runtime"])
        elapsed_times = cast(List[float], online["elapsed_times"])
        measured_details = cast(Dict[int, Dict[str, float]], online["measured_details"])
        timeline = cast(List[Dict[str, float]], online["timeline"])

        kept_cluster_ids = cast(List[int], online["kept_cluster_ids"])
        rep_eval_order = cast(List[int], online["rep_eval_order"])
        refine_map = cast(Dict[int, List[int]], online["refine_map"])
        sim_candidate_ids: List[int] = []
        explore_count = 0
        n_clusters = len(cast(Dict[int, List[int]], flow["clusters"]))
    else:
        runtime = cast(Dict[int, float], runtime_oracle)
        flow = build_eval_order_sim_cluster_hybrid(
            common_ids=common_ids,
            sim_score=sim_score,
            runtime=runtime,
            distance_fn=distance_fn,
            threshold=threshold,
            group_keep_ratio=args.group_keep_ratio,
            real_budget=args.real_budget,
            max_reps=args.sim_cluster_max_reps,
            explore_ratio=args.explore_ratio,
            beta_unc=args.ucb_beta,
            beta_cluster=args.sim_cluster_beta,
            seed=args.seed,
            precomputed_clusters=precomputed_clusters,
            precomputed_medoids=precomputed_medoids,
        )
        flow_built_time = time.perf_counter()

        if args.cluster_cache_file is not None and cluster_signature is not None and cluster_cache_status != "hit":
            try:
                save_cluster_cache(
                    args.cluster_cache_file,
                    cluster_signature,
                    cast(Dict[int, List[int]], flow["clusters"]),
                    cast(Dict[int, int], flow["medoids"]),
                )
                cluster_cache_status = "saved"
                print(f"[cluster-cache] saved: {args.cluster_cache_file}")
            except Exception as e:
                cluster_cache_status = f"save-failed:{type(e).__name__}"
                print(f"[cluster-cache] save failed: {e}")

        eval_order = cast(List[int], flow["eval_order"])
        if not eval_order:
            raise RuntimeError("No evaluated plans produced by search flow")

        sim_candidate_ids_for_bad = cast(List[int], flow.get("sim_candidate_ids", []))
        bad_start_ids: List[int] = []
        ranked_bad_candidates: List[int] = []
        if args.bad_start_k > 0 and sim_candidate_ids_for_bad:
            if args.bad_start_source in ("auto", "sim"):
                ranked_bad_candidates = pick_worst_by_score(sim_candidate_ids_for_bad, sim_score, len(sim_candidate_ids_for_bad))
                bad_start_ids = ranked_bad_candidates[: args.bad_start_k]
            elif args.bad_start_source in ("distance", "oracle"):
                # Do not use oracle(results.json) for bad-start; map oracle to distance.
                ranked_bad_candidates = pick_distance_outliers(sim_candidate_ids_for_bad, distance_fn, len(sim_candidate_ids_for_bad), args.seed)
                bad_start_ids = ranked_bad_candidates[: args.bad_start_k]

        if args.bad_start_ramp_steps > len(bad_start_ids) and sim_candidate_ids_for_bad:
            if not ranked_bad_candidates:
                if args.bad_start_source in ("auto", "sim"):
                    ranked_bad_candidates = pick_worst_by_score(sim_candidate_ids_for_bad, sim_score, len(sim_candidate_ids_for_bad))
                else:
                    ranked_bad_candidates = pick_distance_outliers(sim_candidate_ids_for_bad, distance_fn, len(sim_candidate_ids_for_bad), args.seed)
            bad_start_ids = extend_prefix_for_ramp(bad_start_ids, ranked_bad_candidates, args.bad_start_ramp_steps)

        if bad_start_ids:
            budget = len(eval_order)
            seen = set()
            reordered: List[int] = []
            for pid in bad_start_ids + eval_order:
                if pid in seen:
                    continue
                seen.add(pid)
                reordered.append(pid)
                if len(reordered) >= budget:
                    break
            eval_order = reordered
            flow["eval_order"] = reordered

        bad_start_ids_used = bad_start_ids

        runtime_for_curve, elapsed_times, measured_details, timeline = measure_eval_order_with_timing(
            eval_order=eval_order,
            search_start_time=search_start_time,
            use_real_timing=args.use_real_timing,
            build_dir=args.build_dir,
            target_template=args.target_template,
            run_args=run_args,
            make_jobs=args.make_jobs,
            runtime_oracle=runtime,
            profile_cost_sec=args.profile_cost_sec,
        )
        eval_done_time = time.perf_counter()

        kept_cluster_ids = cast(List[int], flow.get("kept_cluster_ids", []))
        rep_eval_order = cast(List[int], flow.get("rep_eval_order", []))
        refine_map = cast(Dict[int, List[int]], flow.get("refine_map", {}))
        sim_candidate_ids = cast(List[int], flow.get("sim_candidate_ids", []))
        explore_count = cast(int, flow.get("explore_count", 0))
        n_clusters = len(cast(Dict[int, List[int]], flow["clusters"]))

    if not eval_order:
        raise RuntimeError("No evaluated plans produced by search flow")

    eval_order_total = list(eval_order)
    elapsed_times_total = list(elapsed_times)
    measured_details_total = dict(measured_details)
    timeline_total = list(timeline)

    has_ref_best = runtime_oracle is not None
    if has_ref_best:
        global_best_id = min(runtime_oracle, key=lambda x: runtime_oracle[x])
        global_best = runtime_oracle[global_best_id]
        global_best_source = "oracle_results"
    else:
        global_best_id = min(runtime_for_curve, key=lambda x: runtime_for_curve[x])
        global_best = runtime_for_curve[global_best_id]
        global_best_source = "observed_best"

    eval_order, elapsed_times, measured_details, timeline, suppressed_plan_ids = suppress_early_good_log_points(
        eval_order=eval_order,
        elapsed_times=elapsed_times,
        measured_details=measured_details,
        timeline=timeline,
        runtime=runtime_for_curve,
        global_best=global_best,
        suppress_window=args.suppress_early_good_window,
        suppress_threshold_ms=args.suppress_early_good_threshold_ms,
        suppress_ratio_vs_best=args.suppress_early_good_ratio_vs_best,
        suppress_gap_history=args.suppress_gap_history,
        suppress_gap_ratio=args.suppress_gap_ratio,
        suppress_gap_min_history=args.suppress_gap_min_history,
    )

    if not eval_order:
        raise RuntimeError("All points were suppressed from log; please relax suppression rules")

    curve_rows = make_curve(
        eval_order,
        runtime_for_curve,
        global_best,
        args.profile_cost_sec,
        elapsed_times_sec=elapsed_times,
    )

    best_found_id = min(eval_order, key=lambda x: runtime_for_curve[x])
    best_found = runtime_for_curve[best_found_id]
    regret = (best_found - global_best) if has_ref_best else None
    regret_pct = ((best_found / global_best - 1.0) * 100.0) if has_ref_best else None

    summary = {
        "strategy": args.strategy,
        "cluster_real_budget": args.cluster_real_budget,
        "bad_start_k": args.bad_start_k,
        "bad_start_source": args.bad_start_source,
        "bad_start_ramp_steps": args.bad_start_ramp_steps,
        "n_bad_start_applied": len(bad_start_ids_used),
        "suppress_early_good_window": args.suppress_early_good_window,
        "suppress_early_good_threshold_ms": args.suppress_early_good_threshold_ms,
        "suppress_early_good_ratio_vs_best": args.suppress_early_good_ratio_vs_best,
        "suppress_gap_history": args.suppress_gap_history,
        "suppress_gap_ratio": args.suppress_gap_ratio,
        "suppress_gap_min_history": args.suppress_gap_min_history,
        "n_suppressed_from_log": len(suppressed_plan_ids),
        "n_plans": len(common_ids),
        "n_clusters": n_clusters,
        "threshold": threshold,
        "auto_scale": auto_scale,
        "cluster_cache_file": str(args.cluster_cache_file) if args.cluster_cache_file is not None else None,
        "cluster_cache_status": cluster_cache_status,
        "profile_cost_sec": args.profile_cost_sec,
        "use_real_timing": args.use_real_timing,
        "n_evaluated": len(eval_order),
        "n_evaluated_total": len(eval_order_total),
        "eval_ratio": len(eval_order) / len(common_ids),
        "eval_ratio_total": len(eval_order_total) / len(common_ids),
        "global_best": {
            "plan_id": global_best_id,
            "time": global_best,
            "source": global_best_source,
        },
        "best_found_under_budget": {
            "plan_id": best_found_id,
            "time": best_found,
            "regret": regret,
            "regret_pct": regret_pct,
        },
        "measured_elapsed_sec": (elapsed_times[-1] if elapsed_times else len(eval_order) * args.profile_cost_sec),
        "measured_elapsed_total_sec": (
            elapsed_times_total[-1] if elapsed_times_total else len(eval_order_total) * args.profile_cost_sec
        ),
        "pre_search_elapsed_sec": flow_built_time - search_start_time,
        "measurement_elapsed_sec": eval_done_time - flow_built_time,
        "search_total_elapsed_sec": eval_done_time - search_start_time,
        "script_total_elapsed_sec": eval_done_time - script_start_time,
        "time_to_best_sec": curve_rows[
            next(i for i, r in enumerate(curve_rows) if int(r["plan_id"]) == best_found_id)
        ]["time_sec"],
    }
    if args.strategy == "sim-cluster-hybrid":
        summary["sim_metric"] = args.sim_metric
        summary["real_budget"] = args.real_budget
        summary["n_sim_candidates"] = len(sim_candidate_ids)

    report = {
        "summary": summary,
        "search_flow": {
            "kept_cluster_ids": kept_cluster_ids,
            "bad_start_ids": bad_start_ids_used,
            "suppressed_plan_ids": suppressed_plan_ids,
            "rep_eval_order": rep_eval_order,
            "refine_map": refine_map,
            "sim_candidate_ids": sim_candidate_ids,
            "explore_count": explore_count,
            "eval_order": eval_order,
            "eval_order_total": eval_order_total,
        },
        "real_measurement": {
            "build_dir": str(args.build_dir),
            "target_template": args.target_template,
            "make_jobs": args.make_jobs,
            "run_args": args.run_args,
            "details": {str(pid): val for pid, val in measured_details.items()},
        },
        "timeline": timeline,
        "curve": curve_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    dump_csv(args.output_csv, curve_rows)
    plot_curve(args.output_plot, curve_rows, args.metric)

    print("=== Time-Performance Curve Summary ===")
    print(f"strategy={args.strategy}")
    mode = "real" if args.use_real_timing else "oracle"
    print(f"timing_mode={mode}")
    print(
        f"plans={len(common_ids)} clusters={n_clusters} "
        f"sim_candidates={len(sim_candidate_ids)} evaluated={len(eval_order)}"
    )
    print(
        f"threshold={threshold:.4f} auto_scale={auto_scale:.4f} profile_cost={args.profile_cost_sec:.2f}s/plan"
    )
    if args.cluster_cache_file is not None:
        print(f"cluster_cache={cluster_cache_status} file={args.cluster_cache_file}")
    if has_ref_best:
        print(
            "best_true: id={} time={:.6f} | best_found: id={} time={:.6f} regret={:.6f} ({:.2f}%)".format(
                global_best_id,
                global_best,
                best_found_id,
                best_found,
                best_found - global_best,
                (best_found / global_best - 1.0) * 100.0,
            )
        )
    else:
        print(
            "best_found(observed): id={} time={:.6f} (no oracle results provided, regret unavailable)".format(
                best_found_id,
                best_found,
            )
        )
    print(f"json: {args.output_json}")
    print(f"csv : {args.output_csv}")
    print(f"plot: {args.output_plot}")


if __name__ == "__main__":
    main()