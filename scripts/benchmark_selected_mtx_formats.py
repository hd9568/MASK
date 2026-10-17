#!/usr/bin/env python3
"""Benchmark generated vs hand kernels for the selected sparse formats."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_DIR = Path("/workspace/scripts/selected_mtx")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmark_results" / "selected_mtx_formats"

TIME_RE = re.compile(r"mykernel_time:\s*([0-9.]+)\s*ms")
CHECK_RE = re.compile(r"check nan:.*")


@dataclass(frozen=True)
class FormatConfig:
    name: str
    build_dir: Path
    generated_target: str
    hand_target: str
    generated_exe: str
    hand_exe: str


FORMATS: Dict[str, FormatConfig] = {
    "dtc": FormatConfig(
        name="dtc",
        build_dir=REPO_ROOT / "examples/src_fp32/dtc/testbed/build-generated",
        generated_target="demo-dtc-generated-tf32",
        hand_target="demo-dtc-tf32",
        generated_exe="demo-dtc-generated-tf32",
        hand_exe="demo-dtc-tf32",
    ),
    "acc": FormatConfig(
        name="acc",
        build_dir=REPO_ROOT / "examples/src_fp32/acc/testbed/build-generated",
        generated_target="demo-acc-generated-tf32",
        hand_target="demo-acc-tf32",
        generated_exe="demo-acc-generated-tf32",
        hand_exe="demo-acc-tf32",
    ),
    "bitbsr": FormatConfig(
        name="bitbsr",
        build_dir=REPO_ROOT / "examples/src_fp32/bitbsr/testbed/build-generated",
        generated_target="demo-bitbsr-generated-tf32",
        hand_target="demo-bitbsr-tf32",
        generated_exe="demo-bitbsr-generated-tf32",
        hand_exe="demo-bitbsr-tf32",
    ),
    "sr_bcrs": FormatConfig(
        name="sr_bcrs",
        build_dir=REPO_ROOT / "examples/src_fp32/sr_bcrs/testbed/build-generated",
        generated_target="demo-srbcrs-generated-tf32",
        hand_target="demo-srbcrs-tf32",
        generated_exe="demo-srbcrs-generated-tf32",
        hand_exe="demo-srbcrs-tf32",
    ),
}


RUN_FIELDS = [
    "timestamp",
    "format",
    "matrix",
    "variant",
    "trial",
    "status",
    "time_ms",
    "check",
    "returncode",
    "elapsed_sec",
    "log_path",
]

COMPARISON_FIELDS = [
    "timestamp",
    "format",
    "matrix",
    "generated_avg_ms",
    "hand_avg_ms",
    "ratio_generated_over_hand",
    "generated_status",
    "hand_status",
    "generated_check",
    "hand_check",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark generated and hand kernels on selected matrices."
    )
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--formats", nargs="+", choices=sorted(FORMATS), default=sorted(FORMATS))
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def shell_env() -> dict:
    env = os.environ.copy()
    cutlass_paths = [
        "/workspace/sparsene/examples/cutlass/include",
        "/workspace/sparsene/examples/cutlass/examples/common",
        "/workspace/sparsene/examples/cutlass/tools/util/include",
    ]
    old_cpath = env.get("CPATH")
    env["CPATH"] = ":".join(cutlass_paths + ([old_cpath] if old_cpath else []))
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "python"))
    return env


def run_command(
    cmd: List[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=shell_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    elapsed = time.time() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + f"\n[elapsed_sec] {elapsed:.3f}\n")
    return proc


def ensure_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        out = args.output_dir
    else:
        out = DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    return out


def matrix_files(matrix_dir: Path) -> List[Path]:
    files = sorted(matrix_dir.glob("*.mtx"))
    if not files:
        raise FileNotFoundError(f"No .mtx files found under {matrix_dir}")
    return files


def append_csv(path: Path, fields: List[str], row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def regenerate_results(out_dir: Path, timeout: int) -> None:
    log_path = out_dir / "logs" / "regenerate_sparse_formats.log"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "python/test/generate_sparse_formats.py"),
        "--results-dir",
        str(REPO_ROOT / "python/test/results"),
    ]
    proc = run_command(cmd, cwd=REPO_ROOT, log_path=log_path, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"generate_sparse_formats.py failed, see {log_path}")


def build_targets(formats: Iterable[FormatConfig], out_dir: Path, timeout: int) -> None:
    for fmt in formats:
        for target in [fmt.generated_target, fmt.hand_target]:
            log_path = out_dir / "logs" / f"build_{fmt.name}_{target}.log"
            cmd = ["cmake", "--build", str(fmt.build_dir), "--target", target, "-j8"]
            print(f"[{timestamp()}] build {fmt.name} {target}", flush=True)
            proc = run_command(cmd, cwd=REPO_ROOT, log_path=log_path, timeout=timeout)
            if proc.returncode != 0:
                raise RuntimeError(f"Build failed for {target}, see {log_path}")


def parse_run_output(stdout: str) -> tuple[Optional[float], str]:
    time_match = TIME_RE.search(stdout)
    check_match = CHECK_RE.search(stdout)
    time_ms = float(time_match.group(1)) if time_match else None
    check = check_match.group(0) if check_match else ""
    return time_ms, check


def run_one(
    fmt: FormatConfig,
    matrix: Path,
    variant: str,
    trial: int,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    exe_name = fmt.generated_exe if variant == "generated" else fmt.hand_exe
    exe = fmt.build_dir / exe_name
    log_name = f"{fmt.name}_{matrix.stem}_{variant}_trial{trial}.log"
    log_path = out_dir / "logs" / log_name
    cmd = [
        str(exe),
        "-filename",
        str(matrix),
        "-N",
        str(args.N),
        "-warmup",
        str(args.warmup),
        "-repeat",
        str(args.repeat),
    ]

    start = time.time()
    status = "ok"
    returncode = ""
    stdout = ""
    try:
        proc = subprocess.run(
            cmd,
            cwd=fmt.build_dir,
            env=shell_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
        stdout = proc.stdout
        returncode = proc.returncode
        if proc.returncode != 0:
            status = "failed"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        returncode = "timeout"
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stdout += f"\n[TIMEOUT] exceeded {args.timeout} seconds\n"

    elapsed = time.time() - start
    log_path.write_text(stdout + f"\n[elapsed_sec] {elapsed:.3f}\n")
    time_ms, check = parse_run_output(stdout)
    if time_ms is None and status == "ok":
        status = "missing_time"

    return {
        "timestamp": timestamp(),
        "format": fmt.name,
        "matrix": matrix.name,
        "variant": variant,
        "trial": trial,
        "status": status,
        "time_ms": "" if time_ms is None else f"{time_ms:.6f}",
        "check": check,
        "returncode": returncode,
        "elapsed_sec": f"{elapsed:.3f}",
        "log_path": str(log_path),
    }


def write_comparison(
    fmt_name: str,
    matrix_name: str,
    rows: List[dict],
    comparisons_csv: Path,
    comparisons_jsonl: Path,
) -> None:
    by_variant: Dict[str, List[dict]] = {"generated": [], "hand": []}
    for row in rows:
        by_variant[row["variant"]].append(row)

    def ok_times(variant: str) -> List[float]:
        vals = []
        for row in by_variant[variant]:
            if row["status"] == "ok" and row["time_ms"] != "":
                vals.append(float(row["time_ms"]))
        return vals

    gen_times = ok_times("generated")
    hand_times = ok_times("hand")
    gen_avg = mean(gen_times) if gen_times else None
    hand_avg = mean(hand_times) if hand_times else None
    ratio = gen_avg / hand_avg if gen_avg is not None and hand_avg not in (None, 0.0) else None

    comparison = {
        "timestamp": timestamp(),
        "format": fmt_name,
        "matrix": matrix_name,
        "generated_avg_ms": "" if gen_avg is None else f"{gen_avg:.6f}",
        "hand_avg_ms": "" if hand_avg is None else f"{hand_avg:.6f}",
        "ratio_generated_over_hand": "" if ratio is None else f"{ratio:.6f}",
        "generated_status": ",".join(row["status"] for row in by_variant["generated"]),
        "hand_status": ",".join(row["status"] for row in by_variant["hand"]),
        "generated_check": by_variant["generated"][-1]["check"] if by_variant["generated"] else "",
        "hand_check": by_variant["hand"][-1]["check"] if by_variant["hand"] else "",
    }
    append_csv(comparisons_csv, COMPARISON_FIELDS, comparison)
    append_jsonl(comparisons_jsonl, comparison)


def main() -> int:
    args = parse_args()
    out_dir = ensure_output_dir(args)
    selected_formats = [FORMATS[name] for name in args.formats]
    matrices = matrix_files(args.matrix_dir)

    metadata = {
        "started_at": timestamp(),
        "repo_root": str(REPO_ROOT),
        "matrix_dir": str(args.matrix_dir),
        "formats": args.formats,
        "matrix_count": len(matrices),
        "N": args.N,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "trials": args.trials,
        "timeout": args.timeout,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(f"[{timestamp()}] output_dir={out_dir}", flush=True)
    print(f"[{timestamp()}] matrices={len(matrices)} formats={','.join(args.formats)}", flush=True)

    if args.regenerate:
        print(f"[{timestamp()}] regenerating sparse format results", flush=True)
        regenerate_results(out_dir, args.timeout)

    if not args.skip_build:
        build_targets(selected_formats, out_dir, args.timeout)

    runs_csv = out_dir / "runs.csv"
    runs_jsonl = out_dir / "runs.jsonl"
    comparisons_csv = out_dir / "comparisons.csv"
    comparisons_jsonl = out_dir / "comparisons.jsonl"

    total_pairs = len(matrices) * len(selected_formats)
    done_pairs = 0
    for matrix in matrices:
        for fmt in selected_formats:
            pair_rows: List[dict] = []
            for variant in ["generated", "hand"]:
                for trial in range(1, args.trials + 1):
                    print(
                        f"[{timestamp()}] run {fmt.name} {matrix.name} {variant} trial={trial}",
                        flush=True,
                    )
                    row = run_one(fmt, matrix, variant, trial, args, out_dir)
                    append_csv(runs_csv, RUN_FIELDS, row)
                    append_jsonl(runs_jsonl, row)
                    pair_rows.append(row)
                    print(
                        f"[{timestamp()}] -> {row['status']} {row['time_ms']} ms {row['check']}",
                        flush=True,
                    )
                    if row["status"] != "ok" and not args.continue_on_error:
                        return 1
            write_comparison(fmt.name, matrix.name, pair_rows, comparisons_csv, comparisons_jsonl)
            done_pairs += 1
            print(f"[{timestamp()}] completed pair {done_pairs}/{total_pairs}", flush=True)

    print(f"[{timestamp()}] done. comparisons={comparisons_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
