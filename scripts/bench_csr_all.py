#!/usr/bin/env python3
"""Benchmark CSR-generated kernel on all 26 datasets, produce latest_generated_csr.md."""
import os, re, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "examples/src_fp32/dtc/testbed/build_multi_dataset/demo-dtc-csr-tf32"
DATASET_DIR = Path(os.environ.get("DATASET_DIR", "/mnt/lustre-client/zhangkaige/selected_mtx"))
OUT = REPO / "docs/latest_generated_csr.md"
GPU_THRESHOLD = 20
N_COLS = "512"
WARMUP = "20"
REPEAT = "100"

def gpu_util():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        return int(out.stdout.strip().replace(" %", ""))
    except Exception:
        return -1

def wait_gpu():
    while True:
        u = gpu_util()
        if u < 0:
            print("  [WARN] nvidia-smi failed, sleep 10s...", flush=True)
            time.sleep(10)
            continue
        if u < GPU_THRESHOLD:
            return
        print(f"  [WAIT] GPU={u}%, sleeping 5s...", flush=True)
        time.sleep(5)

def run_one(mtx_path):
    env = os.environ.copy()
    env["SPARSENE_DISABLE_MTX_CACHE"] = "1"
    t0 = time.time()
    p = subprocess.run(
        [str(BIN), "-filename", str(mtx_path), "-mtx_flag", "1",
         "-N", N_COLS, "-warmup", WARMUP, "-repeat", REPEAT],
        capture_output=True, text=True, env=env, timeout=600,
    )
    elapsed = time.time() - t0
    text = p.stdout + p.stderr

    ds = mtx_path.name
    my = re.search(r"mykernel_time:\s*([0-9.]+)\s*ms", text)
    cu = re.search(r"cusparse_time:\s*([0-9.]+)\s*ms", text)
    errs = len(re.findall(r"Error\(", text))
    nan = re.search(r"check nan:.*cusp\s*([0-9.\-]+)\s*mykernel\s*([0-9.\-]+)", text)
    dim = re.search(r"Sparse A\[(\d+),(\d+)\]", text)

    M, K = "", ""
    if dim:
        M, K = dim.groups()

    my_v = my.group(1) if my else ""
    cu_v = cu.group(1) if cu else ""
    speedup = ""
    status = "ok"

    if errs > 0:
        status = f"correctness_errors_{errs}"
    elif not my or float(my.group(1)) == 0:
        status = "gpu_busy_or_no_timing"
    elif nan and nan.group(1) != nan.group(2):
        status = "nan_mismatch"

    if status == "ok" and my_v and cu_v and float(cu_v) > 0:
        speedup = f"{float(cu_v)/float(my_v):.4f}"

    return {
        "dataset": ds, "M": M, "K": K, "N": N_COLS,
        "mykernel_ms": my_v, "cusparse_ms": cu_v,
        "speedup_vs_cusparse": speedup, "status": status,
        "elapsed_s": f"{elapsed:.1f}"
    }

def main():
    mtx_files = sorted(DATASET_DIR.glob("*.mtx"))
    if not mtx_files:
        print(f"No .mtx files in {DATASET_DIR}")
        sys.exit(1)

    print(f"CSR Benchmark: {len(mtx_files)} datasets")
    print(f"Binary: {BIN}")
    print(f"Output: {OUT}")
    print()

    rows = []
    fast = 0
    fail = 0
    speeds = []

    for i, f in enumerate(mtx_files):
        ds = f.stem
        print(f"[{i+1}/{len(mtx_files)}] {ds} ...", end=" ", flush=True)
        wait_gpu()
        print("running...", end=" ", flush=True)
        r = run_one(f)
        rows.append(r)

        if r["status"] == "ok":
            t = float(r["mykernel_ms"])
            su = float(r["speedup_vs_cusparse"]) if r["speedup_vs_cusparse"] else 0
            speeds.append(su)
            print(f"{t:.3f}ms (cu={r['cusparse_ms']}ms, {su:.4f}x) [{r['elapsed_s']}s]")
            if su > 1.0:
                fast += 1
        else:
            fail += 1
            print(f"{r['status']} [{r['elapsed_s']}s]")

    # Sort consistently with latest_generated.md
    rows.sort(key=lambda r: r["dataset"])

    # Build markdown
    lines = [
        "| dataset | M | K | N | mykernel_ms | cusparse_ms | speedup_vs_cusparse | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        su = r["speedup_vs_cusparse"] if r["speedup_vs_cusparse"] else ""
        my = r["mykernel_ms"] if r["mykernel_ms"] else ""
        cu = r["cusparse_ms"] if r["cusparse_ms"] else ""
        lines.append(
            f"| {r['dataset']} | {r['M']} | {r['K']} | {r['N']} "
            f"| {my} | {cu} | {su} | {r['status']} |"
        )

    # Stats
    ok_rows = [r for r in rows if r["status"] == "ok" and r["speedup_vs_cusparse"]]
    if ok_rows:
        avg = sum(float(r["speedup_vs_cusparse"]) for r in ok_rows) / len(ok_rows)
        beating = [r for r in ok_rows if float(r["speedup_vs_cusparse"]) > 1.0]
        beat_avg = sum(float(r["speedup_vs_cusparse"]) for r in beating) / len(beating) if beating else 0.0
        lines.extend([
            "",
            f"> **统计**（共 {len(ok_rows)} 个有性能数据的数据集）：",
            f"> - speedup_vs_cusparse 平均值：**{avg:.4f}**",
            f"> - 超过 cusparse 的数据集：**{len(beating)}** 个，平均 speedup：**{beat_avg:.4f}**",
        ])

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nDone. Written to {OUT}")
    print(f"OK: {len(ok_rows)}, failed: {fail}")
    if speeds:
        import statistics
        print(f"Median speedup: {statistics.median(speeds):.4f}")

if __name__ == "__main__":
    main()
