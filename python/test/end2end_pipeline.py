from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
PY_DIR = REPO_ROOT / "python"
TESTBED_DIR = REPO_ROOT / "examples" / "src_fp32" / "dtc" / "testbed"
SRC_DIR = TESTBED_DIR / "src"
DEFAULT_INC_PATH = PY_DIR / "me_tcf_kernel3.inc"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs"
DEFAULT_DATASET_ROOT = Path("/mnt/lustre-client/zhangkaige/selected_mtx")
DEFAULT_HOST_CXX = Path(os.environ.get("NVCC_CCBIN", "/opt/rh/gcc-toolset-13/root/usr/bin/g++"))
DEFAULT_CUDA_ARCH = os.environ.get("CUDA_ARCH", "sm_80")
DEFAULT_NCU_BIN = Path.home() / "local" / "cuda" / "bin" / "ncu"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "llm_models.yaml"
FEATURIZE_BIN = TESTBED_DIR / "featurize_mtx"
ERROR_LINE_RE = re.compile(
    r"Error\(([-0-9]+),\s*([-0-9]+)\):\s*cusp\(([-0-9.eE]+)\)\s*mykernel\(([-0-9.eE]+)\)"
)

os.environ.setdefault("PYTHONPATH", str(PY_DIR))
sys.path.insert(0, str(PY_DIR))

from sparsene.agent.client import call_anthropic_compatible, call_openai_compatible
from sparsene.agent.config import resolve_model_config
from sparsene.formats.DTC_SpMM import ME_TCF_FORMAT
from sparsene.logging import get_logger, set_logging_level_for_all
from sparsene.op_gen.computent.computent import MyNameManager, computent_from_rts
from sparsene.op_gen.fix_agent import FixAgentConfig, run_fix_agent
from sparsene.op_gen.format_design_agent import FormatDesignConfig, build_format_from_response, run_format_design_agent
from sparsene.op_gen.nvir.compiler_driver import apply_software_pipeline_and_codegen
from sparsene.op_gen.nvir.generate import SPARSE_FORMAT_REGISTRY, build_gmem_interfaces, generate_nvir
from sparsene.op_gen.opir.cValFlattenPass import CValFlattenPass
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.varlenLoweringPass import VarlenLoweringPass
from sparsene.op_gen.strategy_agent import StrategyConfig, StrategyDecision, StrategyFeedbackContext, run_strategy_agent
from sparsene.transform.rts import derive_rts


logger = get_logger("end2end_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparsene end-to-end single-matrix pipeline")
    parser.add_argument("-m", "--mtx", type=str, default="shipsec1.mtx")
    parser.add_argument("-d", "--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("-N", "--n-cols", type=int, default=512)
    parser.add_argument("-i", "--iterations", type=int, default=4)
    parser.add_argument("-w", "--eval-warmup", type=int, default=5)
    parser.add_argument("-r", "--eval-repeat", type=int, default=20)
    parser.add_argument("-W", "--profile-warmup", type=int, default=1)
    parser.add_argument("-R", "--profile-repeat", type=int, default=2)
    parser.add_argument("-t", "--timeout-sec", type=int, default=600)
    parser.add_argument("-l", "--run-label", type=str, default="end2end_pipeline")
    parser.add_argument("--mode", choices=["agent", "heuristic"], default="agent")
    parser.add_argument("--model", type=str, default="deepseek-v4-pro")
    parser.add_argument("--format-model", type=str, default="")
    parser.add_argument("--strategy-model", type=str, default="")
    parser.add_argument("--fix-model", type=str, default="")
    parser.add_argument("--strategy-max-tokens", type=int, default=4096)
    parser.add_argument("--format-max-tokens", type=int, default=8192)
    parser.add_argument("--strategy-temperature", type=float, default=0.1)
    parser.add_argument("--format-temperature", type=float, default=0.1)
    parser.add_argument("--fix-agent-iterations", type=int, default=0)
    parser.add_argument("--ncu-bin", type=str, default=str(DEFAULT_NCU_BIN))
    parser.add_argument("--cuda-arch", type=str, default=DEFAULT_CUDA_ARCH)
    parser.add_argument("--host-cxx", type=str, default=str(DEFAULT_HOST_CXX))
    parser.add_argument("--log-root", type=str, default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--model-config", type=str, default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _resolve_matrix_path(raw_path: str, dataset_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    direct = (REPO_ROOT / raw_path).resolve()
    if direct.exists():
        return direct
    return (dataset_root / raw_path).resolve()


def _model_name(args: argparse.Namespace, kind: str) -> str:
    override = {
        "format": args.format_model,
        "strategy": args.strategy_model,
        "fix": args.fix_model,
    }[kind]
    return override.strip() or args.model


def _provider_name(resolved_provider: str) -> str:
    if resolved_provider == "anthropic_compatible":
        return "anthropic_http"
    return "openai_http"


def _probe_model(model_name: str, model_config_path: str) -> Dict[str, Any]:
    resolved = resolve_model_config(model_name=model_name, yaml_path=model_config_path)
    system_prompt = "Reply with a short OK."
    user_prompt = "Health check. Return OK."
    if resolved.provider == "anthropic_compatible":
        response = call_anthropic_compatible(
            model=resolved.model,
            api_key=resolved.api_key,
            anthropic_base_url=resolved.anthropic_base_url,
            anthropic_version=resolved.anthropic_version,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=16,
            timeout_sec=min(resolved.timeout_sec, 60),
        )
    else:
        response = call_openai_compatible(
            model=resolved.model,
            api_key=resolved.api_key,
            api_base_url=resolved.api_base_url,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=16,
            timeout_sec=min(resolved.timeout_sec, 60),
        )
    return {
        "model_name": model_name,
        "provider": resolved.provider,
        "model": resolved.model,
        "usage": response.usage.to_dict(),
        "elapsed_s": response.elapsed_s,
        "content_preview": response.content[:80],
    }


def _probe_required_models(args: argparse.Namespace) -> Dict[str, Any]:
    if args.mode != "agent" and args.fix_agent_iterations <= 0:
        return {"checked": False, "reason": "heuristic_mode"}
    names: List[str] = []
    if args.mode == "agent":
        names.extend([_model_name(args, "format"), _model_name(args, "strategy")])
    if args.fix_agent_iterations > 0:
        names.append(_model_name(args, "fix"))
    seen: Dict[str, Dict[str, Any]] = {}
    for name in names:
        if name in seen:
            continue
        logger.info("Checking agent API availability for model=%s", name)
        seen[name] = _probe_model(name, args.model_config)
    return {"checked": True, "models": seen}


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_sec: int,
    env_overrides: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
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


def _run_command_killable(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_sec: float,
    env_overrides: Optional[Dict[str, str]] = None,
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
        stdout, _ = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        stdout, _ = proc.communicate(timeout=5)
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout_sec, output=stdout)
    return subprocess.CompletedProcess(args=command, returncode=proc.returncode or 0, stdout=stdout or "")


def _coerce_timeout_stdout(stdout: Any) -> str:
    if stdout is None:
        return ""
    if isinstance(stdout, bytes):
        return stdout.decode("utf-8", errors="replace")
    return str(stdout)


def _extract_features_with_cuda(mtx_path: Path, run_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    if not FEATURIZE_BIN.exists():
        logger.info("Building CUDA feature extractor: %s", FEATURIZE_BIN)
        build = _run_command(
            [
                "nvcc",
                "-O3",
                "-std=c++17",
                f"-arch={args.cuda_arch}",
                "-ccbin",
                args.host_cxx,
                f"-I{TESTBED_DIR}",
                f"-I{SRC_DIR}",
                "-o",
                str(FEATURIZE_BIN),
                str(TESTBED_DIR / "featurize_mtx.cu"),
                str(SRC_DIR / "mmio.cu"),
                str(SRC_DIR / "mmio_highlevel.cu"),
            ],
            cwd=REPO_ROOT,
            timeout_sec=max(args.timeout_sec, 300),
        )
        (run_dir / "featurize_build.log").write_text(build.stdout or "")
        if build.returncode != 0:
            raise RuntimeError(f"Failed to build featurize_mtx, see {(run_dir / 'featurize_build.log')}")
    result = subprocess.run(
        [str(FEATURIZE_BIN), str(mtx_path)],
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=max(args.timeout_sec, 600),
    )
    (run_dir / "featurize.stderr.log").write_text(result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(f"featurize_mtx failed: {result.stderr[-500:]}")
    features = json.loads(result.stdout)
    _write_json(run_dir / "matrix_features.json", features)
    return features


def _adaptive_tc_block_k(features: Dict[str, Any]) -> int:
    """Select optimal TC_BLOCK_K based on matrix sparsity characteristics.

    Valid values: 8 or 16 (must match codegen BLK_K tile size).
    Currently constrained to 8 because the generated .inc kernel
    for BLK_K=16 produces correctness errors on sparse matrices.

    To enable larger TC_BLOCK_K when the kernel codegen is fixed,
    set environment variable SPARSENE_TC_BLOCK_K_AGGRESSIVE=1.
    """
    aggressive = os.environ.get("SPARSENE_TC_BLOCK_K_AGGRESSIVE", "0") == "1"
    if not aggressive:
        return 8
    row_dist = features.get("row_nnz_distribution", {})
    avg_nnz = row_dist.get("mean", 0)
    if avg_nnz <= 0:
        return 8
    if avg_nnz < 16:
        return 16
    return 8


def _parse_perf_output(text: str) -> Dict[str, Any]:
    my_match = re.search(r"mykernel_time:\s*([0-9.]+)\s*ms", text)
    cu_match = re.search(r"cusparse_time:\s*([0-9.]+)\s*ms", text)
    nan_match = re.search(r"check nan:\s*cusp\s*([0-9.\-]+)\s*mykernel\s*([0-9.\-]+)", text)
    error_count = len(re.findall(r"Error\(", text))
    error_samples = []
    deltas: List[float] = []
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
            "mean_delta": round(sum(deltas) / len(deltas), 6) if deltas else 0.0,
            "max_abs_delta": round(max(abs(delta) for delta in deltas), 6) if deltas else 0.0,
            "row_range": [min(rows), max(rows)] if rows else None,
            "col_range": [min(cols), max(cols)] if cols else None,
        },
    }
    my_ms = result["mykernel_ms"]
    cu_ms = result["cusparse_ms"]
    result["ratio_vs_cusparse"] = my_ms / cu_ms if my_ms and cu_ms else None
    result["passed"] = (
        my_ms is not None
        and cu_ms is not None
        and error_count == 0
        and result["nan_check_cusparse"] == result["nan_check_mykernel"]
    )
    result["status"] = "ok" if result["passed"] else "failed"
    result["reason"] = "" if result["passed"] else "validation_failed"
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


def _clean_num(value: str) -> Optional[float]:
    stripped = value.strip().replace(",", "")
    if stripped in {"", "nan", "NaN", "None"}:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _parse_ncu_csv(text: str) -> Dict[str, Any]:
    import csv
    import io

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
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 13:
            continue
        section = row[4].strip('"')
        metric = row[5].strip('"')
        alias = metric_aliases.get((section, metric))
        if not alias:
            continue
        value = _clean_num(row[10].strip('"'))
        if value is None:
            continue
        collected.setdefault(alias, []).append(value)
    return {
        key: round(sum(values) / len(values), 4)
        for key, values in collected.items()
        if values
    }


def _derive_format_hints(atomic_type: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "shape_overrides": {"val_len": ["Mo"], "val_sidx": ["BLK_K", "nnz_dim_K"], "C_val": ["N", "M"]},
        "dtypes": {"idx": "int", "len": "int", "off": "int", "sidx": "int"},
        "custom_layouts": {
            "B_val": {
                "dtype": "float",
                "tensor_str": (
                    "logical_divide(make_tensor(make_gmem_ptr(d{name}), "
                    "make_layout(make_shape(K, N), make_stride(N, _1{}))), "
                    "make_shape(_1{}, get<1>(BLK_MNK{})))"
                    "(_, make_coord(_, blockIdx.y))"
                ),
                "tensor_type_str": (
                    "decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), "
                    "make_layout(make_shape(VARLEN, VARLEN), make_stride(VARLEN, _1{}))), "
                    "make_shape(_1{}, get<1>(BLK_MNK{})))"
                    "(_, make_coord(_, VARLEN)))"
                ),
            },
            "C_val": {
                "dtype": "float",
                "tensor_str": (
                    "logical_divide(make_tensor(make_gmem_ptr(d{name}), make_shape(N, M)), "
                    "select<1, 0>(BLK_MNK{}))"
                    "(make_coord(_, blockIdx.y), make_coord(_, blockIdx.x))"
                ),
                "tensor_type_str": (
                    "decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), "
                    "make_shape(VARLEN, VARLEN)), select<1, 0>(BLK_MNK{}))"
                    "(make_coord(_, VARLEN), make_coord(_, VARLEN)))"
                ),
            },
        },
    }
    if atomic_type in ("coo", "mco"):
        base["shape_overrides"]["val_coo_len"] = ["nnz_dim_K_o"]
    if atomic_type == "mco":
        base["dtypes"]["mco_mask"] = "int64_t"
    return base


def _prepare_codegen_artifacts(fmt_obj, format_name: str, atomic_type: str):
    SPARSE_FORMAT_REGISTRY[format_name] = _derive_format_hints(atomic_type)
    MyNameManager.reset()
    rts = derive_rts(fmt_obj)
    computent = computent_from_rts(format_name, rts)
    ops = generate_from_computent(computent)
    lowered = VarlenLoweringPass(op_builder=None, varlen2LenArrayTable=computent.varlen2LenArrayTable).run(ops)
    flattened = CValFlattenPass(op_builder=None).run(lowered)
    gmem_inouts = build_gmem_interfaces(flattened, SPARSE_FORMAT_REGISTRY[format_name])
    return flattened, gmem_inouts


def _generate_kernel_inc(
    *,
    flattened_ops,
    format_name: str,
    decision: StrategyDecision,
    agent_log_dir: Path,
    iter_dir: Path,
) -> Path:
    nvir_program = generate_nvir(
        opir=flattened_ops,
        format_name=format_name,
        strategy_config=StrategyConfig(provider="heuristic"),
        strategy_decision_override=decision,
        strategy_log_dir_override=str(agent_log_dir),
    )
    apply_software_pipeline_and_codegen(nvir_program, str(DEFAULT_INC_PATH))
    iter_inc_path = iter_dir / "me_tcf_kernel3.inc"
    shutil.copy2(DEFAULT_INC_PATH, iter_inc_path)
    return iter_inc_path


def _build_binary(iter_dir: Path, args: argparse.Namespace) -> Tuple[bool, str, Path]:
    build_dir = iter_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    binary_path = build_dir / "demo-dtc-generated-tf32"
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
        f"-DTC_BLOCK_K={args.tc_block_k}",
        f"-I{TESTBED_DIR / 'cutlass' / 'include'}",
        f"-I{TESTBED_DIR / 'cutlass' / 'examples' / 'common'}",
        f"-I{TESTBED_DIR / 'cutlass' / 'tools' / 'util' / 'include'}",
        f"-I{SRC_DIR}",
        f"-I{TESTBED_DIR}",
        f"-I{PY_DIR}",
        str(TESTBED_DIR / "host_program_generated.cu"),
        str(TESTBED_DIR / "main.cu"),
        str(SRC_DIR / "mmio.cu"),
        str(SRC_DIR / "mmio_highlevel.cu"),
        str(SRC_DIR / "utils.cu"),
        "-lcusparse",
        "-lcublas",
        "-o",
        str(binary_path),
    ]
    try:
        result = _run_command(command, cwd=REPO_ROOT, timeout_sec=args.timeout_sec)
    except subprocess.TimeoutExpired as exc:
        return False, _coerce_timeout_stdout(exc.output) + "\n[TIMEOUT]\n", binary_path
    return result.returncode == 0, result.stdout or "", binary_path


def _run_dataset(binary_path: Path, dataset_path: Path, iter_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    log_path = iter_dir / "run.log"
    try:
        result = _run_command_killable(
            [
                str(binary_path),
                "-filename",
                str(dataset_path),
                "-mtx_flag",
                "1",
                "-N",
                str(args.n_cols),
                "-warmup",
                str(args.eval_warmup),
                "-repeat",
                str(args.eval_repeat),
            ],
            cwd=TESTBED_DIR,
            timeout_sec=float(args.timeout_sec),
        )
        log_path.write_text(result.stdout)
        parsed = _parse_perf_output(result.stdout)
        parsed["status"] = "ok" if result.returncode == 0 and parsed["passed"] else "failed"
        parsed["reason"] = "" if parsed["status"] == "ok" else f"exit_{result.returncode}"
        return parsed
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(_coerce_timeout_stdout(getattr(exc, "output", None)) + "\n[TIMEOUT]\n")
        return {
            "mykernel_ms": None,
            "cusparse_ms": None,
            "ratio_vs_cusparse": None,
            "nan_check_cusparse": None,
            "nan_check_mykernel": None,
            "error_count": 0,
            "correctness_feedback": {"error_count": 0},
            "passed": False,
            "status": "timeout",
            "reason": "timeout",
        }


def _profile_dataset(binary_path: Path, dataset_path: Path, iter_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    ncu_bin = Path(args.ncu_bin)
    if not ncu_bin.exists():
        return {"available": False, "reason": "ncu_missing"}
    output_path = iter_dir / "ncu.csv"
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
                str(args.n_cols),
                "-warmup",
                str(args.profile_warmup),
                "-repeat",
                str(args.profile_repeat),
            ],
            cwd=TESTBED_DIR,
            timeout_sec=args.timeout_sec,
        )
    except subprocess.TimeoutExpired:
        output_path.write_text("")
        return {"available": False, "reason": "ncu_timeout"}
    metrics_csv = _extract_ncu_metrics_csv(result.stdout)
    output_path.write_text(metrics_csv)
    summary = _parse_ncu_csv(metrics_csv)
    summary["available"] = result.returncode == 0 and bool(metrics_csv.strip())
    summary["returncode"] = result.returncode
    return summary


def _score_iteration(result: Dict[str, Any]) -> Optional[float]:
    ratio = result.get("ratio_vs_cusparse")
    if ratio is None:
        return None
    penalty = 1000.0 if not result.get("passed") else 0.0
    return round(ratio + penalty, 6)


def _usage_summary(usages: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for usage in usages:
        summary["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        summary["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        summary["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
    return summary


def _compact_history_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "iteration": entry["iteration"],
        "performance": {
            "mykernel_ms": entry["performance"].get("mykernel_ms"),
            "cusparse_ms": entry["performance"].get("cusparse_ms"),
            "ratio_vs_cusparse": entry["performance"].get("ratio_vs_cusparse"),
        },
        "validation": {
            "passed": entry["performance"].get("passed"),
            "failed_count": 0 if entry["performance"].get("passed") else 1,
            "passed_count": 1 if entry["performance"].get("passed") else 0,
            "dataset_count": 1,
        },
        "correctness_feedback": entry["performance"].get("correctness_feedback", {}),
        "profile_summary": entry.get("profile_summary", {}),
        "strategy": entry["strategy"],
    }


def _backup_default_inc(run_dir: Path) -> Optional[Path]:
    if not DEFAULT_INC_PATH.exists():
        return None
    backup = run_dir / "pre_run_me_tcf_kernel3.inc"
    shutil.copy2(DEFAULT_INC_PATH, backup)
    return backup


def _restore_default_inc(backup_path: Optional[Path]) -> None:
    if backup_path is not None and backup_path.exists():
        shutil.copy2(backup_path, DEFAULT_INC_PATH)


def _maybe_run_fix_agent(
    *,
    args: argparse.Namespace,
    iter_dir: Path,
    compile_log: str,
    kernel_path: Path,
) -> Tuple[bool, str]:
    if args.fix_agent_iterations <= 0:
        return False, compile_log
    fix_model_name = _model_name(args, "fix")
    model = resolve_model_config(model_name=fix_model_name, yaml_path=args.model_config)
    fix_config = FixAgentConfig(
        model_name=fix_model_name,
        model=model.model,
        api_key=model.api_key,
        api_base_url=model.api_base_url,
        timeout_sec=model.timeout_sec,
        temperature=model.temperature,
        max_tokens=model.max_tokens,
        max_iterations=args.fix_agent_iterations,
        model_config_path=args.model_config,
    )

    def compile_fn() -> Tuple[bool, str]:
        success, output, _binary_path = _build_binary(iter_dir, args)
        (iter_dir / "build.log").write_text(output)
        return success, output

    result = run_fix_agent(
        kernel_path=kernel_path,
        error_log=compile_log,
        config=fix_config,
        log_dir=iter_dir / "fix_agent",
        compile_fn=compile_fn,
    )
    return result.success, result.final_error


def _heuristic_format_decision(blk_k: int = 8) -> Dict[str, Any]:
    return {
        "request_id": "heuristic",
        "format_name": "ME_TCF",
        "levels": [
            {
                "level": 1,
                "name": "row_window",
                "axes": [
                    {"name": "X_o", "type": "df", "direction": "ROW", "length": "M"},
                    {"name": "Y_s", "type": "sv", "direction": "COL", "length": "K"},
                    {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                ],
            },
            {
                "level": 2,
                "name": "tc_block",
                "axes": [
                    {"name": "Y_so", "type": "df", "direction": "COL", "length": "K/BLK_K"},
                    {"name": "Y_si", "type": "df", "direction": "COL", "length": "BLK_K"},
                    {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                ],
            },
        ],
        "atomic": {"type": "coo", "x_axis": "X_i", "y_axis": "Y_si"},
        "params": {"BLK_M": 16, "BLK_K": blk_k},
        "rationale": "Heuristic mode: skip format agent and use built-in ME_TCF format.",
        "confidence": "high",
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "raw_usage": {}},
        "agent_metadata": {"source": "heuristic_mode"},
    }


def main() -> None:
    args = parse_args()
    args.tc_block_k = 8  # default, overwritten after featurize
    set_logging_level_for_all("DEBUG" if args.debug else "INFO")

    dataset_root = Path(args.dataset_root)
    mtx_path = _resolve_matrix_path(args.mtx, dataset_root)
    if not mtx_path.exists():
        raise FileNotFoundError(f"Matrix not found: {mtx_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.log_root)
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root
    run_dir = run_root / f"{ts}_{args.run_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "args.json", vars(args))

    logger.info("Running end-to-end pipeline mode=%s matrix=%s", args.mode, mtx_path)
    api_probe = _probe_required_models(args)
    _write_json(run_dir / "api_probe.json", api_probe)

    backup_inc = _backup_default_inc(run_dir)
    token_usages: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []

    try:
        features = _extract_features_with_cuda(mtx_path, run_dir, args)
        args.tc_block_k = _adaptive_tc_block_k(features)
        logger.info("Adaptive TC_BLOCK_K = %d (row_nnz_mean=%.1f)", args.tc_block_k,
                     features.get("row_nnz_distribution", {}).get("mean", 0))
        format_name: str
        if args.mode == "agent":
            format_model_name = _model_name(args, "format")
            format_model = resolve_model_config(model_name=format_model_name, yaml_path=args.model_config)
            format_config = FormatDesignConfig(
                model_name=format_model_name,
                model=format_model.model,
                api_key=format_model.api_key,
                api_base_url=format_model.api_base_url,
                log_dir=str(run_dir / "format_agent"),
                timeout_sec=format_model.timeout_sec,
                temperature=args.format_temperature,
                max_tokens=args.format_max_tokens,
                model_config_path=args.model_config,
            )
            format_response = run_format_design_agent(mtx_path.stem, features, format_config)
            format_summary = asdict(format_response)
            fmt_obj = build_format_from_response(format_response)
            format_name = format_response.format_name
            atomic_type = format_response.atomic.get("type", "coo")
            token_usages.append(format_response.token_usage)
        else:
            format_summary = _heuristic_format_decision(blk_k=args.tc_block_k)
            fmt_obj = ME_TCF_FORMAT
            format_name = "ME_TCF"
            atomic_type = "coo"
        _write_json(run_dir / "format_decision.json", format_summary)

        # Agent mode: TC_BLOCK_K must match the format's BLK_K exactly.
        blk_k_from_format = format_summary.get("params", {}).get("BLK_K", 8)
        if isinstance(blk_k_from_format, int) and blk_k_from_format != args.tc_block_k:
            logger.info("Format BLK_K=%d differs from adaptive TC_BLOCK_K=%d; using format value",
                         blk_k_from_format, args.tc_block_k)
            args.tc_block_k = blk_k_from_format

        flattened_ops, gmem_inouts = _prepare_codegen_artifacts(fmt_obj, format_name, atomic_type)
        _write_json(
            run_dir / "gmem_inouts.json",
            {name: {"dtype": value.dtype, "shape": str(value.shape)} for name, value in gmem_inouts.items()},
        )

        baseline_dir = run_dir / "iter_00_baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_config = StrategyConfig(provider="heuristic", log_dir=str(baseline_dir / "agent"))
        baseline_decision, baseline_agent_log = run_strategy_agent(
            format_name=format_name,
            gmem_inouts=gmem_inouts,
            config=baseline_config,
        )
        # When TC_BLOCK_K != 8, disable the coo_restore swizzle on the
        # A tile to avoid a CuTe MMA tiling incompatibility. The host-side
        # swizzle_layout in main.cu is also gated on TC_BLOCK_K == 8.
        if args.tc_block_k != 8:
            baseline_decision.tensor_placement.coo_restore_swizzle = "none"
        baseline_inc = _generate_kernel_inc(
            flattened_ops=flattened_ops,
            format_name=format_name,
            decision=baseline_decision,
            agent_log_dir=baseline_agent_log,
            iter_dir=baseline_dir,
        )
        build_ok, build_log, baseline_binary = _build_binary(baseline_dir, args)
        (baseline_dir / "build.log").write_text(build_log)
        if not build_ok:
            raise RuntimeError("Baseline build failed; aborting pipeline.")
        baseline_perf = _run_dataset(baseline_binary, mtx_path, baseline_dir, args)
        baseline_profile = _profile_dataset(baseline_binary, mtx_path, baseline_dir, args)
        baseline_entry = {
            "iteration": 0,
            "strategy": baseline_decision.to_dict(),
            "performance": baseline_perf,
            "profile_summary": baseline_profile,
            "agent_log_dir": str(baseline_agent_log),
            "kernel_path": str(baseline_inc),
            "token_usage": baseline_decision.token_usage,
            "score": _score_iteration(baseline_perf),
        }
        history.append(baseline_entry)
        _write_json(baseline_dir / "iteration_summary.json", baseline_entry)
        best_entry = baseline_entry if baseline_perf.get("passed") else None

        if args.mode == "agent":
            strategy_model_name = _model_name(args, "strategy")
            strategy_model = resolve_model_config(model_name=strategy_model_name, yaml_path=args.model_config)
            base_strategy_config = StrategyConfig(
                provider=_provider_name(strategy_model.provider),
                model_name=strategy_model_name,
                model=strategy_model.model,
                api_key=strategy_model.api_key,
                api_base_url=strategy_model.api_base_url,
                anthropic_base_url=strategy_model.anthropic_base_url,
                timeout_sec=strategy_model.timeout_sec,
                temperature=args.strategy_temperature,
                max_tokens=args.strategy_max_tokens,
                model_config_path=args.model_config,
            )
            for iteration in range(1, args.iterations + 1):
                iter_dir = run_dir / f"iter_{iteration:02d}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                compact_history = [_compact_history_row(item) for item in history[-6:]]
                latest = history[-1]
                feedback_context = StrategyFeedbackContext(
                    iteration=iteration,
                    max_iterations=args.iterations,
                    history=compact_history,
                    best_iteration=_compact_history_row(best_entry) if best_entry else None,
                    latest_validation={
                        "passed": latest["performance"].get("passed"),
                        "failed_count": 0 if latest["performance"].get("passed") else 1,
                        "passed_count": 1 if latest["performance"].get("passed") else 0,
                        "dataset_count": 1,
                    },
                    correctness_feedback=latest["performance"].get("correctness_feedback", {}),
                    latest_performance={
                        "mykernel_ms": latest["performance"].get("mykernel_ms"),
                        "cusparse_ms": latest["performance"].get("cusparse_ms"),
                        "ratio_vs_cusparse": latest["performance"].get("ratio_vs_cusparse"),
                    },
                    latest_profile=latest.get("profile_summary", {}),
                    additional_notes=[
                        f"Matrix path: {mtx_path}",
                        "Prefer correctness-preserving improvements and avoid redundant strategy variants.",
                    ],
                )
                strategy_config = StrategyConfig(**asdict(base_strategy_config))
                strategy_config.log_dir = str(iter_dir / "agent")

                decision = None
                agent_log_dir = None
                last_exc: Optional[Exception] = None
                for attempt in range(3):
                    try:
                        decision, agent_log_dir = run_strategy_agent(
                            format_name=format_name,
                            gmem_inouts=gmem_inouts,
                            config=strategy_config,
                            feedback_context=feedback_context,
                        )
                        if args.tc_block_k != 8:
                            decision.tensor_placement.coo_restore_swizzle = "none"
                        break
                    except Exception as exc:
                        last_exc = exc
                        if attempt < 2:
                            wait_s = 2 ** attempt * 5
                            logger.warning("Strategy agent attempt %d failed: %s. Retrying in %ds...", attempt + 1, exc, wait_s)
                            time.sleep(wait_s)
                        else:
                            logger.error("Strategy agent attempt %d failed: %s. Giving up.", attempt + 1, exc)
                            raise
                assert decision is not None and agent_log_dir is not None
                token_usages.append(decision.token_usage)
                _write_json(iter_dir / "strategy.json", decision.to_dict())

                iter_inc = _generate_kernel_inc(
                    flattened_ops=flattened_ops,
                    format_name=format_name,
                    decision=decision,
                    agent_log_dir=agent_log_dir,
                    iter_dir=iter_dir,
                )
                build_ok, build_log, binary_path = _build_binary(iter_dir, args)
                (iter_dir / "build.log").write_text(build_log)
                if not build_ok:
                    fixed, final_error = _maybe_run_fix_agent(
                        args=args,
                        iter_dir=iter_dir,
                        compile_log=build_log,
                        kernel_path=iter_inc,
                    )
                    if fixed:
                        build_ok, build_log, binary_path = _build_binary(iter_dir, args)
                        (iter_dir / "build.log").write_text(build_log)
                    else:
                        entry = {
                            "iteration": iteration,
                            "strategy": decision.to_dict(),
                            "performance": {
                                "passed": False,
                                "status": "build_failed",
                                "reason": final_error[:400],
                                "mykernel_ms": None,
                                "cusparse_ms": None,
                                "ratio_vs_cusparse": None,
                                "error_count": 0,
                                "correctness_feedback": {"error_count": 0},
                            },
                            "profile_summary": {"available": False, "reason": "build_failed"},
                            "agent_log_dir": str(agent_log_dir),
                            "kernel_path": str(iter_inc),
                            "token_usage": decision.token_usage,
                            "score": None,
                        }
                        history.append(entry)
                        _write_json(iter_dir / "iteration_summary.json", entry)
                        continue

                perf = _run_dataset(binary_path, mtx_path, iter_dir, args)
                profile = _profile_dataset(binary_path, mtx_path, iter_dir, args)
                entry = {
                    "iteration": iteration,
                    "strategy": decision.to_dict(),
                    "performance": perf,
                    "profile_summary": profile,
                    "agent_log_dir": str(agent_log_dir),
                    "kernel_path": str(iter_inc),
                    "token_usage": decision.token_usage,
                    "score": _score_iteration(perf),
                }
                history.append(entry)
                _write_json(iter_dir / "iteration_summary.json", entry)
                if perf.get("passed") and (
                    best_entry is None
                    or (entry["score"] is not None and best_entry.get("score") is not None and entry["score"] < best_entry["score"])
                    or (best_entry.get("score") is None)
                ):
                    best_entry = entry

        if best_entry is None:
            raise RuntimeError("No correct iteration found in pipeline run.")

        best_kernel = run_dir / "best_iteration_kernel.inc"
        shutil.copy2(best_entry["kernel_path"], best_kernel)
        summary = {
            "best": {
                "iteration": best_entry["iteration"],
                "mykernel_ms": best_entry["performance"].get("mykernel_ms"),
                "cusparse_ms": best_entry["performance"].get("cusparse_ms"),
                "ratio_vs_cusparse": best_entry["performance"].get("ratio_vs_cusparse"),
                "passed": best_entry["performance"].get("passed", False),
                "kernel_path": str(best_kernel),
            },
            "matrix": str(mtx_path),
            "mode": args.mode,
            "tc_block_k": getattr(args, "tc_block_k", 8),
            "api_probe": api_probe,
            "format": format_summary,
            "history": history,
            "best_iteration": best_entry["iteration"],
            "best_kernel_path": str(best_kernel),
            "token_usage": {
                "aggregate": _usage_summary(token_usages),
                "per_call": token_usages,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        logger.info(
            "Best iteration=%s mykernel_ms=%s ratio_vs_cusparse=%s total_tokens=%s",
            best_entry["iteration"],
            best_entry["performance"].get("mykernel_ms"),
            best_entry["performance"].get("ratio_vs_cusparse"),
            summary["token_usage"]["aggregate"]["total_tokens"],
        )
    finally:
        _restore_default_inc(backup_inc)


if __name__ == "__main__":
    main()
