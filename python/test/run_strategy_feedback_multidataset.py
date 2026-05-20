from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparsene.formats.DTC_SpMM import ME_TCF_FORMAT
from sparsene.logging import get_logger, set_logging_level_for_all
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.strategy_agent import (
    PipelineStrategy,
    StrategyConfig,
    StrategyDecision,
    StrategyFeedbackContext,
    TensorPlacementStrategy,
    run_strategy_agent,
)
from sparsene.op_gen.nvir.generate import SPARSE_FORMAT_REGISTRY, build_gmem_interfaces
from sparsene.op_gen.opir.cValFlattenPass import CValFlattenPass
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.varlenLoweringPass import VarlenLoweringPass
from sparsene.transform.rts import derive_rts


logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PY_DIR = REPO_ROOT / "python"
TESTBED_DIR = REPO_ROOT / "examples" / "src_fp32" / "dtc" / "testbed"
DEFAULT_INC_PATH = PY_DIR / "me_tcf_kernel3.inc"
DEFAULT_LOOP_LOG_DIR = REPO_ROOT / "docs" / "agent_feedback_runs"
DEFAULT_BUILD_DIR = TESTBED_DIR / "build_agent_multidataset"
DEFAULT_BUILD_BINARY = DEFAULT_BUILD_DIR / "demo-dtc-generated-multi-dataset-tf32"
DEFAULT_NCU_BIN = Path.home() / "local" / "cuda" / "bin" / "ncu"
DEFAULT_DATASET_ROOT = Path("/mnt/lustre-client/zhangkaige/selected_mtx")
DEFAULT_MTX_CACHE_DIR = Path(
    os.environ.get("SPARSENE_MTX_CACHE_DIR", "/mnt/lustre-client/huoda/mtx_cache")
)
DEFAULT_HOST_CXX = Path(os.environ.get("NVCC_CCBIN", "/opt/rh/gcc-toolset-13/root/usr/bin/g++"))
DEFAULT_CUDA_ARCH = os.environ.get("CUDA_ARCH", "sm_80")
ERROR_LINE_RE = re.compile(
    r"Error\(([-0-9]+),\s*([-0-9]+)\):\s*cusp\(([-0-9.eE]+)\)\s*mykernel\(([-0-9.eE]+)\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset-glob", type=str, default="*.mtx")
    parser.add_argument(
        "--exclude-substrings",
        type=str,
        default="/.git/,/build/,/tmp/,/checkpoint/,/mtx_cache/",
        help="Comma-separated substrings to exclude from dataset paths.",
    )
    parser.add_argument(
        "--prefer-subdirs",
        type=str,
        default="selected_mtx",
        help="Comma-separated preferred subdirs. Matching datasets are listed first.",
    )
    parser.add_argument(
        "--synthetic-shapes",
        type=str,
        default="1024x1024x1024,2048x2048x512",
        help="Comma-separated MxKxN synthetic shapes used for iterative evaluation.",
    )
    parser.add_argument(
        "--synthetic-sparsities",
        type=str,
        default="0.9,0.95",
        help="Comma-separated synthetic sparsities used for iterative evaluation.",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--n-cols", type=int, default=512)
    parser.add_argument("--eval-warmup", type=int, default=5)
    parser.add_argument("--eval-repeat", type=int, default=20)
    parser.add_argument("--final-warmup", type=int, default=20)
    parser.add_argument("--final-repeat", type=int, default=100)
    parser.add_argument("--final-benchmark-parallel", type=int, default=1, help="Number of datasets to benchmark in parallel during final evaluation. Must be 1 when only one GPU is available.")
    parser.add_argument("--final-benchmark-timeout", type=int, default=120, help="Timeout seconds per dataset during final benchmark.")
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-repeat", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--strategy-provider", type=str, default="openai_http")
    parser.add_argument("--strategy-model", type=str, default=None)
    parser.add_argument("--strategy-api-key", type=str, default=None)
    parser.add_argument("--strategy-api-base-url", type=str, default=None)
    parser.add_argument("--strategy-anthropic-base-url", type=str, default=None)
    parser.add_argument("--strategy-timeout-sec", type=int, default=180)
    parser.add_argument("--strategy-temperature", type=float, default=0.35)
    parser.add_argument("--strategy-max-tokens", type=int, default=1200)
    parser.add_argument("--run-label", type=str, default="", help="Label appended to the run directory name, e.g. model name.")
    parser.add_argument("--max-datasets", type=int, default=0, help="0 means use all discovered datasets.")
    parser.add_argument(
        "--feedback-dataset",
        type=str,
        default=None,
        help="Optional .mtx path or basename used for per-iteration timing/profile feedback instead of synthetic cases.",
    )
    parser.add_argument("--ncu-bin", type=str, default=str(DEFAULT_NCU_BIN))
    parser.add_argument("--loop-log-dir", type=str, default=str(DEFAULT_LOOP_LOG_DIR))
    parser.add_argument("--build-dir", type=str, default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--mtx-cache-dir", type=str, default=str(DEFAULT_MTX_CACHE_DIR))
    parser.add_argument("--cuda-arch", type=str, default=DEFAULT_CUDA_ARCH)
    parser.add_argument("--host-cxx", type=str, default=str(DEFAULT_HOST_CXX))
    parser.add_argument("--seed-heuristic-baseline", action="store_true", default=True)
    parser.add_argument("--no-seed-heuristic-baseline", action="store_false", dest="seed_heuristic_baseline")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env_overrides: Optional[Dict[str, str]] = None,
    timeout_sec: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    try:
        return subprocess.run(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        raise


def _run_command_killable(
    command: Sequence[str],
    *,
    cwd: Path,
    env_overrides: Optional[Dict[str, str]] = None,
    timeout_sec: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        stdout, _stderr = proc.communicate(timeout=5)
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout_sec, output=stdout)
    return subprocess.CompletedProcess(
        args=command,
        returncode=proc.returncode or 0,
        stdout=stdout or "",
    )


def _clean_num(value: str) -> Optional[float]:
    stripped = value.strip().replace(",", "")
    if stripped in {"", "nan", "NaN", "None"}:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _coerce_timeout_stdout(stdout: Any) -> str:
    if stdout is None:
        return ""
    if isinstance(stdout, bytes):
        return stdout.decode("utf-8", errors="replace")
    return str(stdout)


def _parse_perf_output(text: str) -> Dict[str, Any]:
    my_match = re.search(r"mykernel_time:\s*([0-9.]+)\s*ms", text)
    cu_match = re.search(r"cusparse_time:\s*([0-9.]+)\s*ms", text)
    nan_match = re.search(r"check nan:\s*cusp\s*([0-9.\-]+)\s*mykernel\s*([0-9.\-]+)", text)
    error_count = len(re.findall(r"Error\(", text))
    error_samples = []
    deltas: List[float] = []
    pos_delta = 0
    neg_delta = 0
    rows: List[int] = []
    cols: List[int] = []
    for match in ERROR_LINE_RE.finditer(text):
        row = int(match.group(1))
        col = int(match.group(2))
        cusp = float(match.group(3))
        mykernel = float(match.group(4))
        delta = mykernel - cusp
        deltas.append(delta)
        rows.append(row)
        cols.append(col)
        if delta > 0:
            pos_delta += 1
        elif delta < 0:
            neg_delta += 1
        if len(error_samples) < 8:
            error_samples.append(
                {
                    "row": row,
                    "col": col,
                    "cusparse": cusp,
                    "mykernel": mykernel,
                    "delta": delta,
                }
            )
    if pos_delta and not neg_delta:
        bias_direction = "mykernel_gt_cusparse"
    elif neg_delta and not pos_delta:
        bias_direction = "mykernel_lt_cusparse"
    elif pos_delta or neg_delta:
        bias_direction = "mixed"
    else:
        bias_direction = "none"
    result: Dict[str, Any] = {
        "mykernel_ms": float(my_match.group(1)) if my_match else None,
        "cusparse_ms": float(cu_match.group(1)) if cu_match else None,
        "nan_check_cusparse": float(nan_match.group(1)) if nan_match else None,
        "nan_check_mykernel": float(nan_match.group(2)) if nan_match else None,
        "error_count": error_count,
        "correctness_feedback": {
            "error_count": error_count,
            "first_error": error_samples[0] if error_samples else None,
            "error_samples": error_samples,
            "bias_direction": bias_direction,
            "mean_delta": round(sum(deltas) / len(deltas), 6) if deltas else 0.0,
            "max_abs_delta": round(max(abs(delta) for delta in deltas), 6) if deltas else 0.0,
            "row_range": [min(rows), max(rows)] if rows else None,
            "col_range": [min(cols), max(cols)] if cols else None,
        },
    }
    my_ms = result["mykernel_ms"]
    cu_ms = result["cusparse_ms"]
    result["ratio_vs_cusparse"] = my_ms / cu_ms if my_ms and cu_ms else None
    result["speedup_vs_generated"] = cu_ms / my_ms if my_ms and cu_ms else None
    result["passed"] = (
        my_ms is not None
        and cu_ms is not None
        and error_count == 0
        and result["nan_check_cusparse"] == result["nan_check_mykernel"]
    )
    return result


def _extract_ncu_metrics_csv(text: str) -> str:
    csv_lines: List[str] = []
    in_csv = False
    for line in text.splitlines():
        if not in_csv and line.startswith('"ID"'):
            in_csv = True
        if in_csv:
            if '"GPU Speed Of Light Throughput","Duration",' in line:
                continue
            csv_lines.append(line)
    return "\n".join(csv_lines) + ("\n" if csv_lines else "")


def _parse_ncu_csv(text: str) -> Dict[str, Any]:
    metric_aliases = {
        ("Launch Statistics", "Block Size"): "block_size",
        ("Launch Statistics", "Grid Size"): "grid_size",
        ("Launch Statistics", "Registers Per Thread"): "registers_per_thread",
        ("Launch Statistics", "Static Shared Memory Per Block"): "static_smem_per_block",
        ("Launch Statistics", "Waves Per SM"): "waves_per_sm",
        ("GPU Speed Of Light Throughput", "Memory Throughput"): "memory_throughput_pct",
        ("GPU Speed Of Light Throughput", "DRAM Throughput"): "dram_throughput_pct",
        ("GPU Speed Of Light Throughput", "Compute (SM) Throughput"): "compute_throughput_pct",
        ("Memory Workload Analysis", "L1/TEX Hit Rate"): "l1tex_hit_rate_pct",
        ("Memory Workload Analysis", "L2 Hit Rate"): "l2_hit_rate_pct",
        ("Memory Workload Analysis", "Mem Pipes Busy"): "mem_pipes_busy_pct",
    }
    collected: Dict[str, List[float]] = {}
    advice: List[str] = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 15:
            continue
        section = row[11].strip()
        metric = row[12].strip()
        value = row[14].strip()
        alias = metric_aliases.get((section, metric))
        if alias is not None:
            numeric_value = _clean_num(value)
            if numeric_value is not None:
                collected.setdefault(alias, []).append(numeric_value)
        if len(row) >= 18 and row[15].strip() == "SOLBottleneck":
            advice.append(row[17].strip())

    summary: Dict[str, Any] = {}
    for key, values in collected.items():
        summary[key] = round(sum(values) / len(values), 4)
    summary["advice"] = advice[:3]
    return summary


def _dataset_to_label(path: Path) -> str:
    return path.name.replace(".mtx", "")


def _parse_synthetic_shapes(spec: str) -> List[Tuple[int, int, int]]:
    shapes: List[Tuple[int, int, int]] = []
    for item in spec.split(","):
        token = item.strip().lower().replace(" ", "")
        if not token:
            continue
        dims = token.split("x")
        if len(dims) != 3:
            raise ValueError(f"Invalid synthetic shape spec: {item}")
        shapes.append((int(dims[0]), int(dims[1]), int(dims[2])))
    if not shapes:
        raise ValueError("At least one synthetic shape is required.")
    return shapes


def _parse_synthetic_sparsities(spec: str) -> List[float]:
    values = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one synthetic sparsity is required.")
    return values


def _build_synthetic_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for M, K, N in _parse_synthetic_shapes(args.synthetic_shapes):
        for sparsity in _parse_synthetic_sparsities(args.synthetic_sparsities):
            label = f"syn_M{M}_K{K}_N{N}_sp{str(sparsity).replace('.', 'p')}"
            cases.append(
                {
                    "label": label,
                    "M": M,
                    "K": K,
                    "N": N,
                    "sparsity": sparsity,
                }
            )
    return cases


def _resolve_feedback_dataset(
    args: argparse.Namespace,
    discovered_datasets: List[Path],
) -> Optional[Path]:
    if not args.feedback_dataset:
        return None
    candidate = Path(args.feedback_dataset)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        root_candidate = Path(args.dataset_root) / candidate
        if root_candidate.exists():
            resolved = root_candidate.resolve()
        else:
            matches = [path for path in discovered_datasets if path.name == candidate.name]
            if not matches:
                raise RuntimeError(f"Feedback dataset not found: {args.feedback_dataset}")
            resolved = matches[0]
    if not resolved.exists():
        raise RuntimeError(f"Feedback dataset does not exist: {resolved}")
    return resolved


def _common_env_overrides(
    args: argparse.Namespace,
    *,
    disable_cache: bool,
) -> Dict[str, str]:
    env = {
        "SPARSENE_MTX_CACHE_DIR": str(Path(args.mtx_cache_dir)),
    }
    if disable_cache:
        env["SPARSENE_DISABLE_MTX_CACHE"] = "1"
    return env


def _build_flattened_ops():
    rts = derive_rts(ME_TCF_FORMAT)
    computent = computent_from_rts("ME_TCF", rts)
    ops = generate_from_computent(computent)
    lowered_ops = VarlenLoweringPass(
        op_builder=None,
        varlen2LenArrayTable=computent.varlen2LenArrayTable,
    ).run(ops)
    flattened_ops = CValFlattenPass(op_builder=None).run(lowered_ops)
    return flattened_ops


def _build_gmem_inouts():
    flattened_ops = _build_flattened_ops()
    return build_gmem_interfaces(flattened_ops, SPARSE_FORMAT_REGISTRY["ME_TCF"])


def _discover_datasets(args: argparse.Namespace) -> List[Path]:
    root = Path(args.dataset_root)
    if not root.exists():
        return []
    exclude_substrings = [item.strip() for item in args.exclude_substrings.split(",") if item.strip()]
    prefer_subdirs = [item.strip().replace("\\", "/") for item in args.prefer_subdirs.split(",") if item.strip()]
    discovered = []
    for path in root.glob(args.dataset_glob):
        if not path.is_file():
            continue
        path_str = str(path).replace("\\", "/")
        if any(item in path_str for item in exclude_substrings):
            continue
        discovered.append(path.resolve())
    discovered = sorted(set(discovered))

    def sort_key(path: Path) -> Tuple[int, str]:
        path_str = str(path).replace("\\", "/")
        priority = 1
        for idx, token in enumerate(prefer_subdirs):
            if token in path_str:
                priority = 0
                return (priority, f"{idx:03d}:{path_str}")
        return (priority, path_str)

    discovered.sort(key=sort_key)
    if args.max_datasets > 0:
        discovered = discovered[: args.max_datasets]
    return discovered


def _strategy_signature(decision: StrategyDecision) -> str:
    tensor = decision.tensor_placement
    pipeline = decision.pipeline
    return "|".join(
        [
            str(tensor.tile_b),
            str(tensor.materialize_b_array_ref_to_smem),
            str(tensor.materialize_val_sidx_to_smem),
            str(tensor.enable_b_array_ref_cp_async),
            tensor.b_array_ref_mode,
            tensor.b_smem_swizzle,
            tensor.coo_idx_output_mem,
            tensor.coo_val_output_mem,
            tensor.coo_load_mode,
            tensor.coo_restore_swizzle,
            str(tensor.align_cp_async_smem_outputs),
            str(pipeline.enable_async_pipeline),
            str(pipeline.sync_threads_on_smem_input),
            str(pipeline.max_wait_prior),
            ",".join(str(x) for x in pipeline.stage_shifts),
        ]
    )


def _strategy_payload(decision: StrategyDecision) -> Dict[str, Any]:
    payload = decision.to_dict()
    payload["signature"] = _strategy_signature(decision)
    return payload


def _make_mutation(decision: StrategyDecision, mutation_name: str) -> StrategyDecision:
    tensor = TensorPlacementStrategy(**asdict(decision.tensor_placement))
    pipeline = PipelineStrategy(**asdict(decision.pipeline))
    rationale = decision.rationale + f" [script_diversify:{mutation_name}]"

    if mutation_name == "disable_b_cp_async":
        tensor.enable_b_array_ref_cp_async = False
        tensor.b_array_ref_mode = "sync_scalar"
    elif mutation_name == "sync_coo":
        tensor.coo_load_mode = "sync_scalar"
    elif mutation_name == "coo_rmem":
        tensor.coo_idx_output_mem = "rmem"
        tensor.coo_val_output_mem = "rmem"
        tensor.coo_load_mode = "sync_scalar"
    elif mutation_name == "disable_async_pipeline":
        pipeline.enable_async_pipeline = False
        tensor.enable_b_array_ref_cp_async = False
        tensor.b_array_ref_mode = "sync_scalar"
        if tensor.coo_load_mode == "cp_async_x4_aligned_fallback":
            tensor.coo_load_mode = "sync_scalar"
    elif mutation_name == "no_val_sidx_smem":
        tensor.materialize_val_sidx_to_smem = False
    elif mutation_name == "max_wait_prior_zero":
        pipeline.max_wait_prior = 0
    elif mutation_name == "max_wait_prior_1":
        pipeline.max_wait_prior = 1
    elif mutation_name == "max_wait_prior_2":
        pipeline.max_wait_prior = 2
    elif mutation_name == "max_wait_prior_3":
        pipeline.max_wait_prior = 3
    elif mutation_name == "stage_shift_double":
        pipeline.stage_shifts = [2]
    elif mutation_name == "stage_shift_0":
        pipeline.stage_shifts = [0]
    elif mutation_name == "stage_shift_1":
        pipeline.stage_shifts = [1]
    elif mutation_name == "no_sync_smem":
        pipeline.sync_threads_on_smem_input = False
    elif mutation_name == "no_align_16":
        tensor.align_cp_async_smem_outputs = False
    elif mutation_name == "tile_b_16":
        tensor.tile_b = 16
    elif mutation_name == "tile_b_32":
        tensor.tile_b = 32
    elif mutation_name == "tile_b_64":
        tensor.tile_b = 64
    else:
        return decision

    return StrategyDecision(
        provider=f"{decision.provider}+{mutation_name}",
        rationale=rationale,
        tensor_placement=tensor,
        pipeline=pipeline,
    )


def _maybe_diversify_decision(
    decision: StrategyDecision,
    *,
    iteration: int,
    seen_signatures: Dict[str, int],
    stagnation_count: int,
) -> StrategyDecision:
    signature = _strategy_signature(decision)
    if seen_signatures.get(signature, 0) == 0 and stagnation_count < 2 and iteration % 5 not in {1, 0}:
        return decision

    mutation_pool = [
        "disable_b_cp_async",
        "sync_coo",
        "coo_rmem",
        "disable_async_pipeline",
        "max_wait_prior_zero",
        "max_wait_prior_1",
        "max_wait_prior_2",
        "max_wait_prior_3",
        "stage_shift_double",
        "stage_shift_0",
        "stage_shift_1",
        "no_sync_smem",
        "no_align_16",
        "tile_b_16",
        "tile_b_32",
        "tile_b_64",
    ]
    mutation_name = mutation_pool[(iteration - 1) % len(mutation_pool)]
    diversified = _make_mutation(decision, mutation_name)
    if _strategy_signature(diversified) == signature:
        return decision
    return diversified


def _build_binary(args: argparse.Namespace, iter_dir: Path) -> Path:
    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    binary_path = build_dir / DEFAULT_BUILD_BINARY.name
    command = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-ftemplate-backtrace-limit=0",
        f"-arch={args.cuda_arch}",
        "-ccbin",
        args.host_cxx,
        "-Xcompiler",
        "-fopenmp",
        "-Df32",
        f"-I{TESTBED_DIR / 'cutlass' / 'include'}",
        f"-I{TESTBED_DIR / 'cutlass' / 'examples' / 'common'}",
        f"-I{TESTBED_DIR / 'cutlass' / 'tools' / 'util' / 'include'}",
        f"-I{TESTBED_DIR / 'src'}",
        f"-I{TESTBED_DIR}",
        str(TESTBED_DIR / "host_program_generated.cu"),
        str(TESTBED_DIR / "main.cu"),
        str(TESTBED_DIR / "src" / "mmio.cu"),
        str(TESTBED_DIR / "src" / "mmio_highlevel.cu"),
        str(TESTBED_DIR / "src" / "utils.cu"),
        "-lcusparse",
        "-lcublas",
        "-o",
        str(binary_path),
    ]
    result = _run_command(command, cwd=REPO_ROOT, timeout_sec=args.timeout_sec)
    (iter_dir / "build.log").write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"build failed, see {iter_dir / 'build.log'}")
    return binary_path


def _generate_inc_for_strategy(decision: StrategyDecision, iter_dir: Path) -> Path:
    strategy_json_path = iter_dir / "strategy_for_codegen.json"
    strategy_json_path.write_text(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False))
    result = _run_command(
        [
            "python",
            "test/test_generate.py",
            "--strategy-provider",
            "manual_json",
            "--strategy-manual-json",
            str(strategy_json_path),
            "--strategy-log-dir",
            str(iter_dir / "codegen_agent_logs"),
        ],
        cwd=PY_DIR,
    )
    (iter_dir / "codegen.log").write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"codegen failed, see {iter_dir / 'codegen.log'}")
    return DEFAULT_INC_PATH


def _run_dataset(
    binary_path: Path,
    dataset_path: Path,
    *,
    n_cols: int,
    warmup: int,
    repeat: int,
    timeout_sec: int,
    iter_log_dir: Path,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    dataset_name = _dataset_to_label(dataset_path)
    log_path = iter_log_dir / f"{dataset_name}.log"
    try:
        result = _run_command(
            [
                str(binary_path),
                "-filename",
                str(dataset_path),
                "-mtx_flag",
                "1",
                "-N",
                str(n_cols),
                "-warmup",
                str(warmup),
                "-repeat",
                str(repeat),
            ],
            cwd=TESTBED_DIR,
            env_overrides=env_overrides,
            timeout_sec=timeout_sec,
        )
        log_path.write_text(result.stdout)
        parsed = _parse_perf_output(result.stdout)
        status = "ok" if result.returncode == 0 and parsed["passed"] else "failed"
        reason = ""
        if result.returncode != 0:
            reason = f"exit_{result.returncode}"
        elif not parsed["passed"]:
            if parsed["error_count"] > 0:
                reason = f"correctness_errors_{parsed['error_count']}"
            else:
                reason = "missing_timing_or_nan"
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(_coerce_timeout_stdout(exc.stdout) + "\n[TIMEOUT]\n")
        parsed = {
            "mykernel_ms": None,
            "cusparse_ms": None,
            "ratio_vs_cusparse": None,
            "speedup_vs_generated": None,
            "nan_check_cusparse": None,
            "nan_check_mykernel": None,
            "error_count": 0,
            "correctness_feedback": {"error_count": 0},
            "passed": False,
        }
        status = "timeout"
        reason = "timeout"

    payload = {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "status": status,
        "reason": reason,
        "log_path": str(log_path),
        **parsed,
    }
    return payload


def _run_synthetic_case(
    binary_path: Path,
    case: Dict[str, Any],
    *,
    warmup: int,
    repeat: int,
    timeout_sec: int,
    iter_log_dir: Path,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    label = case["label"]
    log_path = iter_log_dir / f"{label}.log"
    try:
        result = _run_command(
            [
                str(binary_path),
                "-M",
                str(case["M"]),
                "-K",
                str(case["K"]),
                "-N",
                str(case["N"]),
                "-sparsity",
                str(case["sparsity"]),
                "-warmup",
                str(warmup),
                "-repeat",
                str(repeat),
            ],
            cwd=TESTBED_DIR,
            env_overrides=env_overrides,
            timeout_sec=timeout_sec,
        )
        log_path.write_text(result.stdout)
        parsed = _parse_perf_output(result.stdout)
        status = "ok" if result.returncode == 0 and parsed["passed"] else "failed"
        reason = ""
        if result.returncode != 0:
            reason = f"exit_{result.returncode}"
        elif not parsed["passed"]:
            if parsed["error_count"] > 0:
                reason = f"correctness_errors_{parsed['error_count']}"
            else:
                reason = "missing_timing_or_nan"
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(_coerce_timeout_stdout(exc.stdout) + "\n[TIMEOUT]\n")
        parsed = {
            "mykernel_ms": None,
            "cusparse_ms": None,
            "ratio_vs_cusparse": None,
            "speedup_vs_generated": None,
            "nan_check_cusparse": None,
            "nan_check_mykernel": None,
            "error_count": 0,
            "correctness_feedback": {"error_count": 0},
            "passed": False,
        }
        status = "timeout"
        reason = "timeout"

    return {
        "dataset": label,
        "dataset_path": "",
        "status": status,
        "reason": reason,
        "log_path": str(log_path),
        "M": case["M"],
        "K": case["K"],
        "N": case["N"],
        "sparsity": case["sparsity"],
        **parsed,
    }


def _aggregate_dataset_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    passed_rows = [row for row in results if row["status"] == "ok" and row["passed"]]
    failed_rows = [row for row in results if not (row["status"] == "ok" and row["passed"])]
    mean_kernel = (
        sum(row["mykernel_ms"] for row in passed_rows if row["mykernel_ms"] is not None) / len(passed_rows)
        if passed_rows
        else None
    )
    mean_cusparse = (
        sum(row["cusparse_ms"] for row in passed_rows if row["cusparse_ms"] is not None) / len(passed_rows)
        if passed_rows
        else None
    )
    ratios = [row["ratio_vs_cusparse"] for row in passed_rows if row["ratio_vs_cusparse"] is not None]
    gmean_ratio = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else None
    score = (
        (gmean_ratio or 1e9)
        + 1000.0 * len(failed_rows)
        + 0.001 * (mean_kernel or 1e6)
    )
    return {
        "dataset_count": total,
        "passed_count": len(passed_rows),
        "failed_count": len(failed_rows),
        "mean_mykernel_ms": round(mean_kernel, 6) if mean_kernel is not None else None,
        "mean_cusparse_ms": round(mean_cusparse, 6) if mean_cusparse is not None else None,
        "gmean_ratio_vs_cusparse": round(gmean_ratio, 6) if gmean_ratio is not None else None,
        "score": round(score, 6) if math.isfinite(score) else None,
        "failed_examples": [row["dataset"] for row in failed_rows[:10]],
    }


def _aggregate_correctness_feedback(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    failed_rows = [row for row in results if not (row["status"] == "ok" and row["passed"])]
    synthetic_failed = [row for row in failed_rows if row.get("dataset", "").startswith("syn_")]
    sparsities = sorted({row.get("sparsity") for row in synthetic_failed if row.get("sparsity") is not None})
    bias_directions = [
        row.get("correctness_feedback", {}).get("bias_direction")
        for row in failed_rows
        if row.get("correctness_feedback", {}).get("bias_direction") not in {None, "none"}
    ]
    dominant_bias = "mixed"
    if bias_directions:
        if len(set(bias_directions)) == 1:
            dominant_bias = bias_directions[0]
    else:
        dominant_bias = "none"
    return {
        "failed_case_count": len(failed_rows),
        "failed_cases": [row["dataset"] for row in failed_rows[:12]],
        "synthetic_failed_case_count": len(synthetic_failed),
        "synthetic_failed_sparsities": sparsities,
        "high_sparsity_only_failure": bool(sparsities) and all(sp >= 0.95 for sp in sparsities),
        "dominant_bias_direction": dominant_bias,
        "first_failure": failed_rows[0].get("correctness_feedback", {}).get("first_error") if failed_rows else None,
        "representative_failures": [
            {
                "dataset": row["dataset"],
                "reason": row["reason"],
                "error_count": row.get("error_count"),
                "bias_direction": row.get("correctness_feedback", {}).get("bias_direction"),
                "first_error": row.get("correctness_feedback", {}).get("first_error"),
            }
            for row in failed_rows[:5]
        ],
    }


def _run_ncu_for_dataset(
    binary_path: Path,
    dataset_path: Path,
    *,
    n_cols: int,
    warmup: int,
    repeat: int,
    timeout_sec: int,
    ncu_bin: Path,
    output_path: Path,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    try:
        result = _run_command(
            [
                str(ncu_bin),
                "--target-processes",
                "all",
                "--kernel-name-base",
                "demangled",
                "--kernel-name",
                "regex:generated_kernel",
                "--section",
                "LaunchStats",
                "--section",
                "MemoryWorkloadAnalysis",
                "--section",
                "SpeedOfLight",
                "--csv",
                str(binary_path),
                "-filename",
                str(dataset_path),
                "-mtx_flag",
                "1",
                "-N",
                str(n_cols),
                "-warmup",
                str(warmup),
                "-repeat",
                str(repeat),
            ],
            cwd=TESTBED_DIR,
            env_overrides=env_overrides,
            timeout_sec=timeout_sec,
        )
        metrics_csv = _extract_ncu_metrics_csv(result.stdout)
        output_path.write_text(metrics_csv)
        summary = _parse_ncu_csv(metrics_csv)
        summary["available"] = result.returncode == 0
        summary["returncode"] = result.returncode
        summary["dataset"] = _dataset_to_label(dataset_path)
        summary["raw_log"] = str(output_path)
        return summary
    except subprocess.TimeoutExpired:
        output_path.write_text("")
        return {
            "available": False,
            "returncode": 124,
            "dataset": _dataset_to_label(dataset_path),
            "raw_log": str(output_path),
        }


def _run_ncu_for_synthetic_case(
    binary_path: Path,
    case: Dict[str, Any],
    *,
    warmup: int,
    repeat: int,
    timeout_sec: int,
    ncu_bin: Path,
    output_path: Path,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    try:
        result = _run_command(
            [
                str(ncu_bin),
                "--target-processes",
                "all",
                "--kernel-name-base",
                "demangled",
                "--kernel-name",
                "regex:generated_kernel",
                "--section",
                "LaunchStats",
                "--section",
                "MemoryWorkloadAnalysis",
                "--section",
                "SpeedOfLight",
                "--csv",
                str(binary_path),
                "-M",
                str(case["M"]),
                "-K",
                str(case["K"]),
                "-N",
                str(case["N"]),
                "-sparsity",
                str(case["sparsity"]),
                "-warmup",
                str(warmup),
                "-repeat",
                str(repeat),
            ],
            cwd=TESTBED_DIR,
            env_overrides=env_overrides,
            timeout_sec=timeout_sec,
        )
        metrics_csv = _extract_ncu_metrics_csv(result.stdout)
        output_path.write_text(metrics_csv)
        summary = _parse_ncu_csv(metrics_csv)
        summary["available"] = result.returncode == 0
        summary["returncode"] = result.returncode
        summary["dataset"] = case["label"]
        summary["raw_log"] = str(output_path)
        return summary
    except subprocess.TimeoutExpired:
        output_path.write_text("")
        return {
            "available": False,
            "returncode": 124,
            "dataset": case["label"],
            "raw_log": str(output_path),
        }


def _aggregate_profile_summaries(profile_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    available_rows = [row for row in profile_rows if row.get("available")]
    if not available_rows:
        return {"available": False, "profiled_datasets": [row.get("dataset") for row in profile_rows]}

    numeric_keys = [
        "memory_throughput_pct",
        "dram_throughput_pct",
        "compute_throughput_pct",
        "l1tex_hit_rate_pct",
        "l2_hit_rate_pct",
        "mem_pipes_busy_pct",
        "block_size",
        "grid_size",
        "registers_per_thread",
        "static_smem_per_block",
        "waves_per_sm",
    ]
    summary: Dict[str, Any] = {"available": True, "profiled_datasets": [row.get("dataset") for row in profile_rows]}
    for key in numeric_keys:
        values = [row[key] for row in available_rows if row.get(key) is not None]
        if values:
            summary[key] = round(sum(values) / len(values), 4)
    advice: List[str] = []
    for row in available_rows:
        advice.extend(row.get("advice", []))
    summary["advice"] = advice[:5]
    return summary


def _sanitize_feedback_section(payload: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(payload)
    clean.pop("raw_log", None)
    return clean


def _compact_history_row(iteration_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "iteration": iteration_summary["iteration"],
        "performance": _sanitize_feedback_section(iteration_summary["aggregate"]),
        "validation": {
            "passed": iteration_summary["aggregate"]["failed_count"] == 0,
            "failed_count": iteration_summary["aggregate"]["failed_count"],
            "passed_count": iteration_summary["aggregate"]["passed_count"],
        },
        "correctness_feedback": iteration_summary.get("correctness_feedback", {}),
        "profile_summary": _sanitize_feedback_section(iteration_summary.get("profile_summary", {})),
        "strategy": iteration_summary["strategy"],
    }


def _build_exploration_notes(
    *,
    iteration: int,
    best_entry: Optional[Dict[str, Any]],
    previous_entry: Optional[Dict[str, Any]],
    stagnation_count: int,
    seen_signatures: Dict[str, int],
) -> List[str]:
    notes: List[str] = []
    if previous_entry:
        notes.append(
            "Last iteration aggregate: mean_mykernel_ms={mean_mykernel_ms}, gmean_ratio_vs_cusparse={gmean_ratio_vs_cusparse}, failed_count={failed_count}".format(
                **previous_entry["aggregate"]
            )
        )
        prev_correctness = previous_entry.get("correctness_feedback", {})
        if prev_correctness.get("failed_case_count", 0) > 0:
            notes.append(
                "Last iteration correctness summary: failed_cases={failed_cases}, dominant_bias={dominant_bias_direction}, high_sparsity_only_failure={high_sparsity_only_failure}".format(
                    **prev_correctness
                )
            )
    if best_entry:
        notes.append(
            "Best correct aggregate so far is iter={iteration} with mean_mykernel_ms={mean_mykernel_ms} and gmean_ratio_vs_cusparse={gmean_ratio_vs_cusparse}".format(
                iteration=best_entry["iteration"],
                mean_mykernel_ms=best_entry["aggregate"].get("mean_mykernel_ms"),
                gmean_ratio_vs_cusparse=best_entry["aggregate"].get("gmean_ratio_vs_cusparse"),
            )
        )
    if stagnation_count >= 2:
        notes.append(
            "No aggregate improvement for multiple iterations. Be more aggressive and change at least two knobs from the best-known strategy."
        )
    elif iteration % 5 == 1:
        notes.append(
            "Scheduled exploration round. Prefer a materially different but still plausible strategy instead of repeating the last best configuration."
        )
    else:
        notes.append("Use balanced exploration: small but meaningful changes from the best correct strategy.")
    repeated = sorted(((count, signature) for signature, count in seen_signatures.items() if count > 1), reverse=True)
    if repeated:
        notes.append(f"Avoid repeating overused signatures; current repeated count max is {repeated[0][0]}.")
    notes.append(
        "Candidate knobs for diversity: TILE_B, B-side cp.async on/off, COO smem vs rmem, COO sync vs cp.async, async pipeline on/off, max_wait_prior, stage_shifts, sync_threads_on_smem_input, align_cp_async_smem_outputs."
    )
    return notes


def _request_strategy_with_retry(
    *,
    gmem_inouts,
    strategy_config: StrategyConfig,
    feedback_context: StrategyFeedbackContext,
    iter_dir: Path,
) -> Tuple[StrategyDecision, Path]:
    try:
        return run_strategy_agent(
            format_name="ME_TCF",
            gmem_inouts=gmem_inouts,
            config=strategy_config,
            feedback_context=feedback_context,
        )
    except Exception as exc:
        fallback_context = StrategyFeedbackContext(**feedback_context.to_dict())
        fallback_context.additional_notes = list(fallback_context.additional_notes) + [f"Retry reason: {exc}"]
        (iter_dir / "agent_retry_note.txt").write_text(f"Retrying strategy request. Reason: {exc}\n")
        return run_strategy_agent(
            format_name="ME_TCF",
            gmem_inouts=gmem_inouts,
            config=strategy_config,
            feedback_context=fallback_context,
        )


def _write_dataset_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "dataset_path",
        "status",
        "reason",
        "mykernel_ms",
        "cusparse_ms",
        "ratio_vs_cusparse",
        "speedup_vs_generated",
        "error_count",
        "nan_check_cusparse",
        "nan_check_mykernel",
        "log_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _write_best_table(
    *,
    rows: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
    iteration_label: str,
    run_dir: Path,
    prefix: str,
) -> Tuple[Path, Path]:
    csv_path = run_dir / "best_iteration_all_datasets.csv"
    md_path = run_dir / "best_iteration_all_datasets.md"
    if prefix:
        csv_path = run_dir / f"{prefix}.csv"
        md_path = run_dir / f"{prefix}.md"
    _write_dataset_csv(csv_path, rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Best Iteration All-Dataset Comparison\n\n")
        f.write(
            f"Best iteration: `{iteration_label}` with mean kernel time `{aggregate.get('mean_mykernel_ms')}` ms "
            f"and geometric-mean ratio `{aggregate.get('gmean_ratio_vs_cusparse')}`.\n\n"
        )
        headers = ["dataset", "status", "mykernel_ms", "cusparse_ms", "ratio_vs_cusparse", "reason"]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            values = [row.get(key) for key in headers]
            f.write("| " + " | ".join(str(x) if x is not None else "-" for x in values) + " |\n")
    return csv_path, md_path


def _write_final_benchmark_progress(
    log_file: Path,
    idx: int,
    total: int,
    dataset_path: Path,
    status: str,
    mykernel_ms: Optional[float],
    duration_s: float,
) -> None:
    line = (
        f"[{idx}/{total}] {dataset_path.name:30s}  "
        f"status={status:10s}  "
        f"mykernel_ms={mykernel_ms}  "
        f"elapsed={duration_s:.1f}s"
    )
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    logger.info(line)


def _run_one_final_dataset(
    binary_path: Path,
    dataset_path: Path,
    *,
    n_cols: int,
    warmup: int,
    repeat: int,
    timeout_sec: float,
    log_dir: Path,
    env_overrides: Dict[str, str],
    idx: int,
    total: int,
    progress_log: Path,
) -> Dict[str, Any]:
    dataset_name = _dataset_to_label(dataset_path)
    log_path = log_dir / f"{dataset_name}.log"
    t0 = datetime.now()
    try:
        result = _run_command_killable(
            [
                str(binary_path),
                "-filename",
                str(dataset_path),
                "-mtx_flag",
                "1",
                "-N",
                str(n_cols),
                "-warmup",
                str(warmup),
                "-repeat",
                str(repeat),
            ],
            cwd=TESTBED_DIR,
            env_overrides=env_overrides,
            timeout_sec=timeout_sec,
        )
        log_path.write_text(result.stdout)
        parsed = _parse_perf_output(result.stdout)
        status = "ok" if result.returncode == 0 and parsed["passed"] else "failed"
        reason = ""
        if result.returncode != 0:
            reason = f"exit_{result.returncode}"
        elif not parsed["passed"]:
            if parsed["error_count"] > 0:
                reason = f"correctness_errors_{parsed['error_count']}"
            else:
                reason = "missing_timing_or_nan"
    except subprocess.TimeoutExpired as exc:
        stdout_text = _coerce_timeout_stdout(getattr(exc, "output", None))
        log_path.write_text(stdout_text + "\n[TIMEOUT]\n")
        parsed = {
            "mykernel_ms": None,
            "cusparse_ms": None,
            "ratio_vs_cusparse": None,
            "speedup_vs_generated": None,
            "nan_check_cusparse": None,
            "nan_check_mykernel": None,
            "error_count": 0,
            "correctness_feedback": {"error_count": 0},
            "passed": False,
        }
        status = "timeout"
        reason = "timeout"
    except Exception as exc:
        log_path.write_text(f"[CRASH] {exc}\n")
        parsed = {
            "mykernel_ms": None,
            "cusparse_ms": None,
            "ratio_vs_cusparse": None,
            "speedup_vs_generated": None,
            "nan_check_cusparse": None,
            "nan_check_mykernel": None,
            "error_count": 0,
            "correctness_feedback": {"error_count": 0},
            "passed": False,
        }
        status = "crash"
        reason = f"crash_{exc}"
    duration_s = (datetime.now() - t0).total_seconds()
    _write_final_benchmark_progress(
        progress_log, idx, total, dataset_path, status, parsed.get("mykernel_ms"), duration_s,
    )
    return {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "status": status,
        "reason": reason,
        "log_path": str(log_path),
        **parsed,
    }


def _run_final_best_benchmark(
    *,
    args: argparse.Namespace,
    best_entry: Dict[str, Any],
    datasets: List[Path],
    run_dir: Path,
) -> Tuple[Path, Path, Dict[str, Any]]:
    final_dir = run_dir / "best_iteration_final_eval"
    final_dir.mkdir(parents=True, exist_ok=True)
    best_kernel_path = Path(best_entry["kernel_path"])
    backup_path = final_dir / "pre_final_eval_me_tcf_kernel3.inc"
    restore_needed = DEFAULT_INC_PATH.exists()
    if restore_needed:
        shutil.copy2(DEFAULT_INC_PATH, backup_path)
    shutil.copy2(best_kernel_path, DEFAULT_INC_PATH)
    try:
        binary_path = _build_binary(args, final_dir)
        log_dir = final_dir / "dataset_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        env_overrides = _common_env_overrides(args, disable_cache=False)
        progress_log = final_dir / "progress.log"
        progress_log.write_text("")
        total = len(datasets)
        max_workers = min(args.final_benchmark_parallel, total)
        if max_workers <= 1:
            results = [
                _run_one_final_dataset(
                    binary_path=binary_path,
                    dataset_path=dataset_path,
                    n_cols=args.n_cols,
                    warmup=args.final_warmup,
                    repeat=args.final_repeat,
                    timeout_sec=args.final_benchmark_timeout,
                    log_dir=log_dir,
                    env_overrides=env_overrides,
                    idx=i + 1,
                    total=total,
                    progress_log=progress_log,
                )
                for i, dataset_path in enumerate(datasets)
            ]
        else:
            results_dict: Dict[int, Dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {}
                for i, dataset_path in enumerate(datasets):
                    future = executor.submit(
                        _run_one_final_dataset,
                        binary_path=binary_path,
                        dataset_path=dataset_path,
                        n_cols=args.n_cols,
                        warmup=args.final_warmup,
                        repeat=args.final_repeat,
                        timeout_sec=args.final_benchmark_timeout,
                        log_dir=log_dir,
                        env_overrides=env_overrides,
                        idx=i + 1,
                        total=total,
                        progress_log=progress_log,
                    )
                    future_map[future] = i
                for future in as_completed(future_map):
                    i = future_map[future]
                    try:
                        results_dict[i] = future.result()
                    except Exception as exc:
                        dataset_path = datasets[i]
                        dataset_name = _dataset_to_label(dataset_path)
                        logger.warning("Final benchmark crashed for %s: %s", dataset_name, exc)
                        results_dict[i] = {
                            "dataset": dataset_name,
                            "dataset_path": str(dataset_path),
                            "status": "crash",
                            "reason": f"crash_{exc}",
                            "mykernel_ms": None,
                            "cusparse_ms": None,
                            "ratio_vs_cusparse": None,
                            "speedup_vs_generated": None,
                            "nan_check_cusparse": None,
                            "nan_check_mykernel": None,
                            "error_count": 0,
                            "correctness_feedback": {"error_count": 0},
                            "passed": False,
                            "log_path": str(log_dir / f"{dataset_name}.log"),
                        }
            results = [results_dict[i] for i in range(total)]
        aggregate = _aggregate_dataset_results(results)
        _write_dataset_csv(final_dir / "dataset_results.csv", results)
        _write_json(final_dir / "aggregate.json", aggregate)
    finally:
        if restore_needed and backup_path.exists():
            shutil.copy2(backup_path, DEFAULT_INC_PATH)
        elif not restore_needed and DEFAULT_INC_PATH.exists():
            DEFAULT_INC_PATH.unlink()

    csv_path, md_path = _write_best_table(
        rows=results,
        aggregate=aggregate,
        iteration_label=str(best_entry["iteration"]),
        run_dir=run_dir,
        prefix="best_iteration_all_datasets_final",
    )
    return csv_path, md_path, aggregate


def _write_curve_csv(run_dir: Path, history: List[Dict[str, Any]]) -> Path:
    curve_path = run_dir / "performance_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iteration",
                "mean_mykernel_ms",
                "mean_cusparse_ms",
                "gmean_ratio_vs_cusparse",
                "passed_count",
                "failed_count",
                "score",
            ]
        )
        for item in history:
            agg = item["aggregate"]
            writer.writerow(
                [
                    item["iteration"],
                    agg.get("mean_mykernel_ms"),
                    agg.get("mean_cusparse_ms"),
                    agg.get("gmean_ratio_vs_cusparse"),
                    agg.get("passed_count"),
                    agg.get("failed_count"),
                    agg.get("score"),
                ]
            )
    return curve_path


def _write_curve_plot(run_dir: Path, history: List[Dict[str, Any]]) -> Path:
    plot_path = run_dir / "performance_curve.png"
    entries = [item for item in history if item["iteration"] > 0 and item["aggregate"].get("mean_mykernel_ms") is not None]
    if not entries:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        return plot_path

    kernel_points = [(item["iteration"], item["aggregate"]["mean_mykernel_ms"]) for item in entries]
    cusparse_values = [item["aggregate"]["mean_cusparse_ms"] for item in entries if item["aggregate"].get("mean_cusparse_ms") is not None]
    cusparse_ms = sum(cusparse_values) / len(cusparse_values) if cusparse_values else 0.0
    x_values = [iteration for iteration, _ in kernel_points]
    y_values = [value for _, value in kernel_points]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_values, y_values, marker="o", linewidth=2.0, color="#2563eb", label="Generated kernel mean")
    ax.axhline(cusparse_ms, linestyle="--", linewidth=2.0, color="#dc2626", label=f"cuSPARSE mean ({cusparse_ms:.3f} ms)")
    ax.set_title("ME_TCF Feedback Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Time (ms)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks(x_values)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def _select_best_entry(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = [item for item in history if item["aggregate"]["failed_count"] == 0 and item["aggregate"].get("score") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["aggregate"]["score"])


def _feedback_config_from_args(args: argparse.Namespace, run_dir: Path) -> StrategyConfig:
    defaults = StrategyConfig()
    return StrategyConfig(
        provider=args.strategy_provider,
        log_dir=str(run_dir / "agent_runs"),
        api_key=args.strategy_api_key or defaults.api_key,
        model=args.strategy_model or defaults.model,
        api_base_url=args.strategy_api_base_url or defaults.api_base_url,
        anthropic_base_url=args.strategy_anthropic_base_url or defaults.anthropic_base_url,
        timeout_sec=args.strategy_timeout_sec,
        temperature=args.strategy_temperature,
        max_tokens=args.strategy_max_tokens,
        anthropic_version=defaults.anthropic_version,
    )


def _evaluate_iteration(
    *,
    args: argparse.Namespace,
    iter_dir: Path,
    binary_path: Path,
    synthetic_cases: List[Dict[str, Any]],
    feedback_dataset: Optional[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dataset_log_dir = iter_dir / "dataset_logs"
    dataset_log_dir.mkdir(parents=True, exist_ok=True)
    if feedback_dataset is not None:
        results = [
            _run_dataset(
                binary_path,
                feedback_dataset,
                n_cols=args.n_cols,
                warmup=args.eval_warmup,
                repeat=args.eval_repeat,
                timeout_sec=args.timeout_sec,
                iter_log_dir=dataset_log_dir,
                env_overrides=_common_env_overrides(args, disable_cache=False),
            )
        ]
    else:
        results = [
            _run_synthetic_case(
                binary_path,
                case,
                warmup=args.eval_warmup,
                repeat=args.eval_repeat,
                timeout_sec=args.timeout_sec,
                iter_log_dir=dataset_log_dir,
                env_overrides=_common_env_overrides(args, disable_cache=True),
            )
            for case in synthetic_cases
        ]
    aggregate = _aggregate_dataset_results(results)
    _write_dataset_csv(iter_dir / "dataset_results.csv", results)
    _write_json(iter_dir / "aggregate.json", aggregate)
    return results, aggregate


def _profile_iteration(
    *,
    args: argparse.Namespace,
    iter_dir: Path,
    binary_path: Path,
    synthetic_cases: List[Dict[str, Any]],
    feedback_dataset: Optional[Path],
) -> Dict[str, Any]:
    ncu_bin = Path(args.ncu_bin)
    if not ncu_bin.exists():
        return {"available": False, "profiled_datasets": []}
    profile_dir = iter_dir / "ncu_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    if feedback_dataset is not None:
        rows = [
            _run_ncu_for_dataset(
                binary_path,
                feedback_dataset,
                n_cols=args.n_cols,
                warmup=args.profile_warmup,
                repeat=args.profile_repeat,
                timeout_sec=args.timeout_sec,
                ncu_bin=ncu_bin,
                output_path=profile_dir / f"{_dataset_to_label(feedback_dataset)}.csv",
                env_overrides=_common_env_overrides(args, disable_cache=False),
            )
        ]
    else:
        if not synthetic_cases:
            return {"available": False, "profiled_datasets": []}
        rows = [
            _run_ncu_for_synthetic_case(
                binary_path,
                case,
                warmup=args.profile_warmup,
                repeat=args.profile_repeat,
                timeout_sec=args.timeout_sec,
                ncu_bin=ncu_bin,
                output_path=profile_dir / f"{case['label']}.csv",
                env_overrides=_common_env_overrides(args, disable_cache=True),
            )
            for case in synthetic_cases
        ]
    summary = _aggregate_profile_summaries(rows)
    _write_json(iter_dir / "profile_summary.json", summary)
    return summary


def _generate_and_build_for_strategy(
    *,
    decision: StrategyDecision,
    iter_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Path, Path]:
    generated_inc = _generate_inc_for_strategy(decision, iter_dir)
    iteration_inc_path = iter_dir / "me_tcf_kernel3.inc"
    shutil.copy2(generated_inc, iteration_inc_path)
    binary_path = _build_binary(args, iter_dir)
    return iteration_inc_path, binary_path


def main() -> None:
    args = parse_args()
    set_logging_level_for_all("DEBUG" if args.debug else "INFO")

    run_root = Path(args.loop_log_dir)
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root
    run_label = (args.run_label or "").strip()
    if not run_label:
        run_label = args.strategy_model or args.strategy_provider
    run_label_slug = "_" + run_label.replace(" ", "_").replace("/", "_") if run_label else ""
    run_dir = run_root / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_me_tcf_{args.strategy_provider}{run_label_slug}_multidataset")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_config.json", vars(args))

    datasets = _discover_datasets(args)
    synthetic_cases = _build_synthetic_cases(args)
    feedback_dataset = None
    if args.feedback_dataset:
        feedback_dataset = _resolve_feedback_dataset(args, datasets)
        if not datasets and feedback_dataset is not None:
            datasets = [feedback_dataset]
    if not datasets:
        raise RuntimeError(f"No .mtx datasets found under {args.dataset_root}")
    gmem_inouts = _build_gmem_inouts()
    _write_json(run_dir / "datasets.json", [str(path) for path in datasets])
    _write_json(run_dir / "synthetic_cases.json", synthetic_cases)
    _write_json(
        run_dir / "feedback_target.json",
        {
            "mode": "dataset" if feedback_dataset is not None else "synthetic",
            "dataset": str(feedback_dataset) if feedback_dataset is not None else None,
        },
    )

    base_strategy_config = _feedback_config_from_args(args, run_dir)
    history: List[Dict[str, Any]] = []
    compact_history: List[Dict[str, Any]] = []
    seen_signatures: Dict[str, int] = {}
    previous_entry: Optional[Dict[str, Any]] = None
    best_entry: Optional[Dict[str, Any]] = None
    stagnation_count = 0
    latest_aggregate: Dict[str, Any] = {}
    latest_correctness: Dict[str, Any] = {}
    latest_profile: Dict[str, Any] = {}

    if args.seed_heuristic_baseline:
        baseline_dir = run_dir / "baseline_heuristic"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_config = StrategyConfig(**asdict(base_strategy_config))
        baseline_config.provider = "heuristic"
        baseline_config.log_dir = str(baseline_dir / "agent")
        baseline_decision, baseline_agent_log_dir = run_strategy_agent(
            format_name="ME_TCF",
            gmem_inouts=gmem_inouts,
            config=baseline_config,
        )
        (baseline_dir / "strategy.json").write_text(json.dumps(_strategy_payload(baseline_decision), indent=2, ensure_ascii=False))
        baseline_inc_path, baseline_binary = _generate_and_build_for_strategy(decision=baseline_decision, iter_dir=baseline_dir, args=args)
        baseline_results, baseline_aggregate = _evaluate_iteration(
            args=args,
            iter_dir=baseline_dir,
            binary_path=baseline_binary,
            synthetic_cases=synthetic_cases,
            feedback_dataset=feedback_dataset,
        )
        baseline_correctness = _aggregate_correctness_feedback(baseline_results)
        baseline_profile = _profile_iteration(
            args=args,
            iter_dir=baseline_dir,
            binary_path=baseline_binary,
            synthetic_cases=synthetic_cases,
            feedback_dataset=feedback_dataset,
        )
        baseline_summary = {
            "iteration": 0,
            "label": "baseline_heuristic",
            "agent_log_dir": str(baseline_agent_log_dir),
            "kernel_path": str(baseline_inc_path),
            "strategy": _strategy_payload(baseline_decision),
            "dataset_results": baseline_results,
            "aggregate": baseline_aggregate,
            "correctness_feedback": baseline_correctness,
            "profile_summary": baseline_profile,
        }
        _write_json(baseline_dir / "iteration_summary.json", baseline_summary)
        history.append(baseline_summary)
        compact_history.append(_compact_history_row(baseline_summary))
        signature = baseline_summary["strategy"]["signature"]
        seen_signatures[signature] = seen_signatures.get(signature, 0) + 1
        previous_entry = baseline_summary
        best_entry = baseline_summary if baseline_aggregate["failed_count"] == 0 else None
        latest_aggregate = _sanitize_feedback_section(baseline_aggregate)
        latest_correctness = _sanitize_feedback_section(baseline_correctness)
        latest_profile = _sanitize_feedback_section(baseline_profile)

    for iteration in range(1, args.iterations + 1):
        iter_dir = run_dir / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        exploration_notes = _build_exploration_notes(
            iteration=iteration,
            best_entry=best_entry,
            previous_entry=previous_entry,
            stagnation_count=stagnation_count,
            seen_signatures=seen_signatures,
        )
        feedback_context = StrategyFeedbackContext(
            iteration=iteration,
            max_iterations=args.iterations,
            history=compact_history[-8:],
            best_iteration=_compact_history_row(best_entry) if best_entry else None,
            latest_validation={
                "passed": latest_aggregate.get("failed_count", 1) == 0,
                "failed_count": latest_aggregate.get("failed_count"),
                "passed_count": latest_aggregate.get("passed_count"),
                "dataset_count": latest_aggregate.get("dataset_count"),
            },
            correctness_feedback=latest_correctness,
            latest_performance=latest_aggregate,
            latest_profile=latest_profile,
            current_kernel_path=None,
            current_kernel_code="",
            include_full_kernel_code=False,
            additional_notes=exploration_notes,
        )

        strategy_config = StrategyConfig(**asdict(base_strategy_config))
        strategy_config.log_dir = str(iter_dir / "agent")
        decision, agent_log_dir = _request_strategy_with_retry(
            gmem_inouts=gmem_inouts,
            strategy_config=strategy_config,
            feedback_context=feedback_context,
            iter_dir=iter_dir,
        )
        decision = _maybe_diversify_decision(
            decision,
            iteration=iteration,
            seen_signatures=seen_signatures,
            stagnation_count=stagnation_count,
        )
        (iter_dir / "strategy.json").write_text(json.dumps(_strategy_payload(decision), indent=2, ensure_ascii=False))

        iteration_inc_path, binary_path = _generate_and_build_for_strategy(decision=decision, iter_dir=iter_dir, args=args)
        dataset_results, aggregate = _evaluate_iteration(
            args=args,
            iter_dir=iter_dir,
            binary_path=binary_path,
            synthetic_cases=synthetic_cases,
            feedback_dataset=feedback_dataset,
        )
        correctness_feedback = _aggregate_correctness_feedback(dataset_results)
        profile_summary = _profile_iteration(
            args=args,
            iter_dir=iter_dir,
            binary_path=binary_path,
            synthetic_cases=synthetic_cases,
            feedback_dataset=feedback_dataset,
        )

        iteration_summary = {
            "iteration": iteration,
            "agent_log_dir": str(agent_log_dir),
            "kernel_path": str(iteration_inc_path),
            "strategy": _strategy_payload(decision),
            "dataset_results": dataset_results,
            "aggregate": aggregate,
            "correctness_feedback": correctness_feedback,
            "profile_summary": profile_summary,
        }
        _write_json(iter_dir / "iteration_summary.json", iteration_summary)

        history.append(iteration_summary)
        compact_history.append(_compact_history_row(iteration_summary))
        signature = iteration_summary["strategy"]["signature"]
        seen_signatures[signature] = seen_signatures.get(signature, 0) + 1
        previous_entry = iteration_summary
        latest_aggregate = _sanitize_feedback_section(aggregate)
        latest_correctness = _sanitize_feedback_section(correctness_feedback)
        latest_profile = _sanitize_feedback_section(profile_summary)

        candidate_best = _select_best_entry(history)
        if candidate_best and (best_entry is None or candidate_best["aggregate"]["score"] < best_entry["aggregate"]["score"]):
            best_entry = candidate_best
            stagnation_count = 0
        else:
            stagnation_count += 1

        logger.info(
            "Iteration %d/%d: passed=%d/%d mean_mykernel_ms=%s gmean_ratio=%s score=%s",
            iteration,
            args.iterations,
            aggregate["passed_count"],
            aggregate["dataset_count"],
            aggregate["mean_mykernel_ms"],
            aggregate["gmean_ratio_vs_cusparse"],
            aggregate["score"],
        )

    summary_path = run_dir / "summary.json"
    _write_json(summary_path, {"history": history, "best_iteration": best_entry})
    curve_csv = _write_curve_csv(run_dir, history)
    curve_plot = _write_curve_plot(run_dir, history)
    if best_entry is not None:
        best_kernel_copy = run_dir / "best_iteration_kernel.inc"
        shutil.copy2(Path(best_entry["kernel_path"]), best_kernel_copy)
        eval_csv, eval_md = _write_best_table(
            rows=best_entry["dataset_results"],
            aggregate=best_entry["aggregate"],
            iteration_label=str(best_entry["iteration"]),
            run_dir=run_dir,
            prefix="best_iteration_all_datasets_eval",
        )
        best_csv, best_md, final_aggregate = _run_final_best_benchmark(
            args=args,
            best_entry=best_entry,
            datasets=datasets,
            run_dir=run_dir,
        )
        _write_json(run_dir / "best_iteration_final_aggregate.json", final_aggregate)
    else:
        eval_csv = run_dir / "best_iteration_all_datasets_eval.csv"
        eval_md = run_dir / "best_iteration_all_datasets_eval.md"
        best_csv = run_dir / "best_iteration_all_datasets_final.csv"
        best_md = run_dir / "best_iteration_all_datasets_final.md"
        best_kernel_copy = run_dir / "best_iteration_kernel.inc"
    logger.info(
        "Multi-dataset feedback finished. Summary=%s curve=%s plot=%s eval_csv=%s eval_md=%s best_csv=%s best_md=%s best_kernel=%s",
        summary_path,
        curve_csv,
        curve_plot,
        eval_csv,
        eval_md,
        best_csv,
        best_md,
        best_kernel_copy,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
