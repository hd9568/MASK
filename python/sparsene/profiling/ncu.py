from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Optional


METRIC_ALIASES = {
    "Block Size": "block_size",
    "Grid Size": "grid_size",
    "Registers Per Thread": "registers_per_thread",
    "Static Shared Memory Per Block": "static_smem_per_block",
    "Dynamic Shared Memory Per Block": "dynamic_smem_per_block",
    "Waves Per SM": "waves_per_sm",
    "Achieved Occupancy": "achieved_occupancy_pct",
    "Theoretical Occupancy": "theoretical_occupancy_pct",
    "DRAM Throughput": "dram_throughput_pct",
    "Compute (SM) Throughput": "compute_throughput_pct",
    "L1/TEX Hit Rate": "l1tex_hit_rate_pct",
    "L2 Hit Rate": "l2_hit_rate_pct",
    "Mem Pipes Busy": "mem_pipes_busy_pct",
    "Mem Busy": "mem_busy_pct",
    "Max Bandwidth": "max_bandwidth_pct",
    "L1/TEX Cache Throughput": "l1tex_throughput_pct",
    "L2 Cache Throughput": "l2_throughput_pct",
    "Achieved Active Warps Per SM": "achieved_active_warps_per_sm",
    "Theoretical Active Warps per SM": "theoretical_active_warps_per_sm",
    "Block Limit Registers": "block_limit_registers",
    "Block Limit Shared Mem": "block_limit_shared_memory",
    "Block Limit Warps": "block_limit_warps",
    "# SMs": "sm_count",
    "Duration": "duration",
}


def _metric_alias(metric: str, unit: str) -> str | None:
    """Resolve metrics whose display name is reused with different units."""

    if metric == "Memory Throughput":
        if unit == "%":
            return "memory_throughput_pct"
        if unit.lower() in {"byte/s", "bytes/s"}:
            return "memory_bandwidth_bytes_per_s"
        return "memory_throughput"
    return METRIC_ALIASES.get(metric)


def _number(value: str) -> Optional[float]:
    clean = value.strip().replace(",", "").replace("%", "")
    if clean in {"", "nan", "NaN", "None", "n/a", "N/A"}:
        return None
    try:
        return float(clean)
    except ValueError:
        return None


def extract_ncu_csv(text: str) -> str:
    lines: list[str] = []
    started = False
    for line in text.splitlines():
        if not started and line.startswith('"ID"'):
            started = True
        if started:
            lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def parse_ncu_csv(text: str) -> dict[str, Any]:
    """Parse both compact and extended NCU CSV layouts by column name."""

    if not text.strip():
        return {"available": False, "reason": "empty_ncu_csv"}
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return {"available": False, "reason": "empty_ncu_csv"}
    header = [item.strip() for item in rows[0]]
    index = {name: idx for idx, name in enumerate(header)}

    def field(row: list[str], names: Iterable[str]) -> str:
        for name in names:
            idx = index.get(name)
            if idx is not None and idx < len(row):
                return row[idx].strip()
        return ""

    collected: dict[str, list[float]] = {}
    kernels: set[str] = set()
    units: dict[str, str] = {}
    advice: list[str] = []
    for row in rows[1:]:
        metric = field(row, ("Metric Name", "Metric"))
        value = field(row, ("Metric Value", "Value"))
        unit = field(row, ("Metric Unit", "Unit"))
        kernel = field(row, ("Kernel Name", "Kernel"))
        if kernel:
            kernels.add(kernel)
        alias = _metric_alias(metric, unit)
        number = _number(value)
        if alias and number is not None:
            collected.setdefault(alias, []).append(number)
            if unit:
                units[alias] = unit
        rule_type = field(row, ("Rule Type",))
        rule_message = field(row, ("Rule Message", "Rule Description"))
        if rule_type and rule_message:
            advice.append(rule_message)
    summary: dict[str, Any] = {
        key: round(sum(values) / len(values), 6)
        for key, values in collected.items()
        if values
    }
    summary.update(
        available=bool(collected),
        kernels=sorted(kernels),
        units=units,
        advice=advice[:8],
    )
    if not collected:
        summary["reason"] = "no_recognized_metrics"
    return summary


def collect_ncu_profile(
    *,
    ncu_bin: Path,
    executable: Path,
    executable_args: list[str],
    output_dir: Path,
    timeout_sec: int,
    kernel_regex: str = "regex:generated_.*kernel",
) -> dict[str, Any]:
    """Collect one generated-kernel launch and retain raw and parsed evidence."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_log = output_dir / "ncu_full.log"
    csv_path = output_dir / "ncu_metrics.csv"
    summary_path = output_dir / "ncu_summary.json"
    command_path = output_dir / "ncu_command.json"
    command = [
        str(ncu_bin),
        "--target-processes", "all",
        "--kernel-name-base", "demangled",
        "--kernel-name", kernel_regex,
        "--launch-count", "1",
        "--section", "LaunchStats",
        "--section", "Occupancy",
        "--section", "MemoryWorkloadAnalysis",
        "--section", "SpeedOfLight",
        "--csv",
        str(executable),
        *executable_args,
    ]
    command_path.write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    if not ncu_bin.is_file():
        summary = {
            "available": False,
            "reason": "ncu_missing",
            "ncu_bin": str(ncu_bin),
            "raw_log": str(raw_log),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_sec,
        )
        raw_log.write_text(result.stdout, encoding="utf-8")
        csv_text = extract_ncu_csv(result.stdout)
        csv_path.write_text(csv_text, encoding="utf-8")
        summary = parse_ncu_csv(csv_text)
        summary.update(
            returncode=result.returncode,
            raw_log=str(raw_log),
            csv_path=str(csv_path),
            command_path=str(command_path),
        )
        if result.returncode != 0:
            summary["available"] = False
            summary["reason"] = "ncu_failed"
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        raw_log.write_text(output, encoding="utf-8")
        csv_path.write_text("", encoding="utf-8")
        summary = {
            "available": False,
            "reason": "ncu_timeout",
            "returncode": 124,
            "raw_log": str(raw_log),
            "csv_path": str(csv_path),
            "command_path": str(command_path),
        }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
