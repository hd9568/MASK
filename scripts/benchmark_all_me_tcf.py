#!/usr/bin/env python3
"""Run ME_TCF heuristic (no agent) on all datasets, produce a markdown comparison table.

Usage:
    python scripts/benchmark_all_me_tcf.py                        # all datasets
    python scripts/benchmark_all_me_tcf.py Yeast Protein cant      # selected
    python scripts/benchmark_all_me_tcf.py -o results.md           # custom output

Output: logs/bench_me_tcf_YYYYMMDD_HHMMSS.md
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(os.environ.get("DATASET_DIR", "/mnt/lustre-client/zhangkaige/selected_mtx"))
PIPELINE = REPO_ROOT / "python" / "test" / "end2end_pipeline.py"


def resolve_datasets(selected: List[str]) -> List[Path]:
    """Resolve dataset paths from names or glob."""
    if selected:
        paths = []
        for name in selected:
            p = DATASET_DIR / name if name.endswith(".mtx") else DATASET_DIR / f"{name}.mtx"
            if p.exists():
                paths.append(p)
            else:
                print(f"WARNING: {p} not found, skipping", file=sys.stderr)
        return paths
    return sorted(DATASET_DIR.glob("*.mtx"))


def run_one(dataset_path: Path, log_root: Path, timeout: int = 1200) -> Tuple[bool, str]:
    """Run end2end_pipeline.py on one dataset. Returns (success, stderr_tail)."""
    ds = dataset_path.stem
    label = f"bench_{ds}"
    cmd = [
        sys.executable, str(PIPELINE),
        "--mode", "heuristic",
        "-d", str(DATASET_DIR),
        "-m", dataset_path.name,
        "-N", "512",
        "-i", "10",
        "-w", "5",
        "-r", "20",
        "-t", str(timeout),
        "-l", label,
        "--log-root", str(log_root),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True, text=True,
        timeout=timeout + 60,
    )
    success = proc.returncode == 0
    tail = "\n".join(proc.stderr.strip().split("\n")[-5:]) if proc.stderr else ""
    return success, tail


def extract_row(subdir: Path) -> Optional[Dict]:
    """Extract one table row from a dataset run directory."""
    ds = subdir.name.split("_e2e_")[-1]
    ds_name = ds + ".mtx"

    # matrix features
    mf = subdir / "matrix_features.json"
    M, K = 0, 0
    if mf.exists():
        with open(mf) as f:
            shape = json.load(f).get("shape", {})
        M = int(shape.get("M", 0))
        K = int(shape.get("N", 0))

    # N from run.log
    N = 0
    for iter_dir in sorted(subdir.iterdir()):
        if not iter_dir.is_dir() or not iter_dir.name.startswith("iter_"):
            continue
        rl = iter_dir / "run.log"
        if rl.exists():
            with open(rl) as f:
                for line in f:
                    if line.strip().startswith("N:"):
                        N = int(line.strip().split(":")[1].strip())
                    if line.strip().startswith("mykernel_time:"):
                        break
            break

    # summary
    summary = subdir / "summary.json"
    if not summary.exists():
        return {"dataset": ds_name, "M": M, "K": K, "N": N,
                "sparsene_ms": None, "cusparse_ms": None,
                "speedup": None, "status": "FAILED"}

    with open(summary) as f:
        s = json.load(f)
    best = s.get("best", {})
    mykernel = best.get("mykernel_ms")
    cusparse = best.get("cusparse_ms")
    passed = best.get("passed", False)

    speedup = cusparse / mykernel if cusparse and mykernel and mykernel > 0 else None
    status = "ok" if passed else "FAILED"

    return {
        "dataset": ds_name, "M": M, "K": K, "N": N,
        "sparsene_ms": mykernel, "cusparse_ms": cusparse,
        "speedup": speedup, "status": status,
    }


def render_table(rows: List[Dict]) -> str:
    header = "| dataset | M | K | N | sparsene_ms | cusparse_ms | speedup_vs_cusparse | status |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, sep]

    for r in rows:
        if r["sparsene_ms"] is None:
            lines.append(
                f"| {r['dataset']} | {r['M']} | {r['K']} | {r['N']} "
                f"| — | — | — | {r['status']} |"
            )
        else:
            sp = f"{r['sparsene_ms']:.6f}"
            cu = f"{r['cusparse_ms']:.6f}" if r["cusparse_ms"] else "N/A"
            su = f"{r['speedup']:.4f}" if r["speedup"] is not None else "—"
            lines.append(
                f"| {r['dataset']} | {r['M']} | {r['K']} | {r['N']} "
                f"| {sp} | {cu} | {su} | {r['status']} |"
            )

    return "\n".join(lines)


def render_stats(rows: List[Dict]) -> str:
    valid = [r for r in rows if r["speedup"] is not None]
    beating = [r for r in valid if r["speedup"] > 1.0]
    if not valid:
        return "\n_No valid results._\n"

    avg_all = sum(r["speedup"] for r in valid) / len(valid)
    out = [
        f"\n> **统计**（共 {len(valid)} 个成功数据集）：",
        f"> - speedup_vs_cusparse 平均值：**{avg_all:.4f}**",
    ]
    if beating:
        avg_beat = sum(r["speedup"] for r in beating) / len(beating)
        out.append(f"> - 超过 cusparse 的数据集：**{len(beating)}** 个，平均 speedup：**{avg_beat:.4f}**")
    else:
        out.append("> - 无数据集性能超过 cusparse。")

    failed = [r for r in rows if r["status"] in ("FAILED", "FAILED_NUMERICAL")]
    if failed:
        out.append(f"> - 失败数据集：**{len(failed)}** 个")

    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Benchmark ME_TCF on all datasets")
    parser.add_argument("datasets", nargs="*", help="Specific dataset names (without .mtx)")
    parser.add_argument("-o", "--output", type=str, help="Output markdown file path")
    parser.add_argument("-t", "--timeout", type=int, default=1200, help="Timeout per dataset (seconds)")
    args = parser.parse_args()

    paths = resolve_datasets(args.datasets)
    if not paths:
        print("No datasets found", file=sys.stderr)
        sys.exit(1)

    epoch = time.strftime("%Y%m%d_%H%M%S")
    log_root = REPO_ROOT / "logs" / f"bench_me_tcf_{epoch}"
    log_root.mkdir(parents=True, exist_ok=True)

    out_path = args.output or (log_root.parent / f"bench_me_tcf_{epoch}.md")

    print(f"Datasets: {len(paths)}")
    print(f"Log dir:  {log_root}")
    print(f"Output:   {out_path}")
    print()

    rows = []
    total = len(paths)

    for i, mtx_path in enumerate(paths):
        ds = mtx_path.stem
        print(f"[{i+1}/{total}] {ds} …", end=" ", flush=True)

        success, tail = run_one(mtx_path, log_root, args.timeout)

        # Find the created subdirectory
        subdirs = sorted(
            [d for d in log_root.iterdir() if d.is_dir() and d.name.endswith(f"_bench_{ds}")],
            key=lambda d: d.stat().st_mtime, reverse=True,
        )
        if subdirs:
            row = extract_row(subdirs[0])
            rows.append(row)
            t = row.get("sparsene_ms")
            st = row.get("status")
            if t:
                print(f"{t:.3f}ms ({st})")
            else:
                print(f"{st}")
        else:
            print("no output directory")
            rows.append({
                "dataset": ds + ".mtx", "M": 0, "K": 0, "N": 512,
                "sparsene_ms": None, "cusparse_ms": None,
                "speedup": None, "status": "FAILED",
            })

    rows.sort(key=lambda r: r["dataset"])

    with open(out_path, "w") as f:
        f.write("# ME_TCF Generated Kernel vs cuSPARSE\n\n")
        f.write(render_table(rows))
        f.write("\n")
        f.write(render_stats(rows))

    print(f"\nWrote: {out_path}")

    # Also print to stdout
    print()
    print(render_table(rows))
    print(render_stats(rows))


if __name__ == "__main__":
    main()
