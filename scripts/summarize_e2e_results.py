#!/usr/bin/env python3
"""Summarize end2end pipeline results from a batch log directory.

Usage:
    python scripts/summarize_e2e_results.py logs/logs_all_datasets_20260604_214951

Produces a markdown table with one row per dataset, showing:
    dataset, M, K, N, mykernel_ms, cusparse_ms, speedup_vs_cusparse, status

For datasets in the failure list, scans iter_*/run.log for timing
and marks the status as "FAILED_NUMERICAL" (if timing available) or "FAILED".
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


def extract_mk(features_path: Path) -> Tuple[int, int]:
    if not features_path.exists():
        return 0, 0
    with open(features_path) as f:
        data = json.load(f)
    shape = data.get("shape", {})
    return int(shape.get("M", 0)), int(shape.get("N", 0))


def extract_n(iter_dir: Path) -> int:
    run_log = iter_dir / "run.log"
    if not run_log.exists():
        return 0
    with open(run_log) as f:
        for line in f:
            line = line.strip()
            if line.startswith("N:"):
                return int(line.split(":")[1].strip())
    return 0


def extract_best(summary_path: Path) -> Optional[Dict]:
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        data = json.load(f)
    return data.get("best")


def extract_timing_from_run_logs(subdir: Path) -> Optional[Dict]:
    """Scan iter_*/run.log for mykernel/cusparse timing when summary.json is absent."""
    best_mykernel = None
    best_cusparse = None
    for iter_dir in sorted(subdir.iterdir()):
        if not iter_dir.is_dir() or not iter_dir.name.startswith("iter_"):
            continue
        run_log = iter_dir / "run.log"
        if not run_log.exists():
            continue
        with open(run_log) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("mykernel_time:"):
                    best_mykernel = float(stripped.split(":")[1].strip().split()[0])
                if stripped.startswith("cusparse_time:"):
                    best_cusparse = float(stripped.split(":")[1].strip().split()[0])
        if best_mykernel is not None:
            return {"mykernel_ms": best_mykernel, "cusparse_ms": best_cusparse or 0}
    return None


def scan_log_root(log_root: Path) -> list:
    failed_file = log_root / "end2end_all_failed.txt"
    failed_set: set = set()
    if failed_file.exists():
        with open(failed_file) as f:
            for line in f:
                name = line.strip()
                if name.endswith(".mtx"):
                    name = name[:-4]
                failed_set.add(name)

    results = []
    for subdir in sorted(log_root.iterdir()):
        if not subdir.is_dir():
            continue
        dirname = subdir.name
        parts = dirname.split("_e2e_", 1)
        if len(parts) != 2:
            continue
        dataset = parts[1]

        features_path = subdir / "matrix_features.json"
        summary_path = subdir / "summary.json"
        iter_dirs = sorted(
            [d for d in subdir.iterdir() if d.is_dir() and d.name.startswith("iter_")],
            key=lambda d: d.name,
        )
        first_iter = iter_dirs[0] if iter_dirs else subdir

        M, K_sp = extract_mk(features_path)
        N = extract_n(first_iter)

        best = extract_best(summary_path)

        # For failed datasets with no summary.json, try to recover timing
        if best is None and dataset in failed_set:
            best = extract_timing_from_run_logs(subdir)
            if best is not None:
                # Mark as failed with timing available
                mykernel = best.get("mykernel_ms", 0)
                cusparse = best.get("cusparse_ms", 0)
                speedup = cusparse / mykernel if cusparse > 0 and mykernel > 0 else None
                results.append({
                    "dataset": dataset,
                    "M": M, "K": K_sp, "N": N,
                    "mykernel_ms": mykernel,
                    "cusparse_ms": cusparse,
                    "speedup": speedup,
                    "status": "FAILED_NUMERICAL",
                })
                continue

        if best is None:
            status = "FAILED" if dataset in failed_set else "NO SUMMARY"
            results.append({
                "dataset": dataset,
                "M": M, "K": K_sp, "N": N,
                "mykernel_ms": None, "cusparse_ms": None,
                "speedup": None, "status": status,
            })
            continue

        mykernel = best.get("mykernel_ms", 0)
        cusparse = best.get("cusparse_ms", 0)
        passed = best.get("passed", False)
        speedup = cusparse / mykernel if cusparse > 0 and mykernel > 0 else None
        status = "ok" if passed else ("FAILED_NUMERICAL" if dataset in failed_set else "FAILED")

        results.append({
            "dataset": dataset,
            "M": M, "K": K_sp, "N": N,
            "mykernel_ms": mykernel,
            "cusparse_ms": cusparse,
            "speedup": speedup,
            "status": status,
        })

    results.sort(key=lambda r: r["dataset"])
    return results


def render_table(results: list) -> str:
    header = "| dataset | M | K | N | mykernel_ms | cusparse_ms | speedup_vs_cusparse | status |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, sep]

    for r in results:
        ds = r["dataset"] + ".mtx" if not r["dataset"].endswith(".mtx") else r["dataset"]

        if r["mykernel_ms"] is None:
            lines.append(
                f"| {ds} | {r['M']} | {r['K']} | {r['N']} "
                f"| — | — | — | {r['status']} |"
            )
            continue

        mk = f"{r['mykernel_ms']:.6f}"
        cu = f"{r['cusparse_ms']:.6f}" if r["cusparse_ms"] > 0 else "N/A"
        sp = f"{r['speedup']:.4f}" if r["speedup"] is not None else "—"
        lines.append(
            f"| {ds} | {r['M']} | {r['K']} | {r['N']} "
            f"| {mk} | {cu} | {sp} | {r['status']} |"
        )

    return "\n".join(lines) + "\n"


def render_stats(results: list) -> str:
    """Render summary statistics below the table."""
    valid = [r for r in results if r["speedup"] is not None]
    beating = [r for r in valid if r["speedup"] > 1.0]
    total_valid = len(valid)
    total_beating = len(beating)

    if total_valid == 0:
        return "\n_No valid results to compute statistics._\n"

    avg_all = sum(r["speedup"] for r in valid) / total_valid
    lines = [
        f"\n> **统计**（共 {total_valid} 个有性能数据的数据集）：",
        f"> - speedup_vs_cusparse 平均值：**{avg_all:.4f}**",
    ]
    if total_beating > 0:
        avg_beat = sum(r["speedup"] for r in beating) / total_beating
        lines.append(
            f"> - 超过 cusparse 的数据集：**{total_beating}** 个，平均 speedup：**{avg_beat:.4f}**"
        )
    else:
        lines.append("> - 无数据集性能超过 cusparse。")

    failed = [r for r in results if "FAILED" in r["status"]]
    if failed:
        lines.append(f"> - 数值错误（FAILED_NUMERICAL）：**{len(failed)}** 个")

    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/summarize_e2e_results.py <log_directory>")
        sys.exit(1)

    log_root = Path(sys.argv[1])
    if not log_root.is_dir():
        print(f"Error: '{log_root}' is not a directory", file=sys.stderr)
        sys.exit(1)

    results = scan_log_root(log_root)
    table = render_table(results)
    print(table)
    print(render_stats(results))


if __name__ == "__main__":
    main()
