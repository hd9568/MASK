#!/usr/bin/env python3
"""Build a provenance-preserving 26-dataset speedup comparison figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_BEST = REPO_ROOT / "sota_results/summary/sparsene_a100_n512.csv"
HISTORICAL_WIDE = REPO_ROOT / "sota_results/summary/a100_n512_fp32_operator_wide_ms.csv"
SPARSENE2_REPORT = REPO_ROOT / "docs/trae/01_performance_comparison_analysis.md"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sparsene2_speedups(path: Path) -> dict[str, float]:
    """Read the audited SR-BCRS S-speedup column from the historical report."""
    values: dict[str, float] = {}
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| 数据集 | nnz | 历史最佳"):
            in_table = True
            continue
        if in_table and line.startswith("> **粗体**"):
            break
        if not in_table or not line.startswith("|") or "---" in line:
            continue
        columns = [part.strip().replace("**", "") for part in line.strip().strip("|").split("|")]
        if len(columns) != 8:
            continue
        dataset = columns[0]
        try:
            values[dataset] = float(columns[6])
        except ValueError:
            continue
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    status_path = run_dir / "operator_e2e_v7_all_status.json"
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    datasets = list(run_manifest["datasets"])

    historical_best = {row["dataset"]: float(row["best_ms"]) for row in _read_csv(HISTORICAL_BEST)}
    historical_direct = {
        row["dataset"]: float(row["cuSPARSE/direct-log"])
        for row in _read_csv(HISTORICAL_WIDE)
        if row.get("cuSPARSE/direct-log")
    }
    sparsene2 = _sparsene2_speedups(SPARSENE2_REPORT)
    status = json.loads(status_path.read_text(encoding="utf-8"))["results"]
    v7 = {row["dataset"]: row for row in status}
    missing = {
        "historical": [d for d in datasets if d not in historical_best or d not in historical_direct],
        "sparsene2": [d for d in datasets if d not in sparsene2],
        "v7": [d for d in datasets if d not in v7],
    }
    if any(missing.values()):
        raise ValueError(f"incomplete comparison sources: {missing}")

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        v7_row = v7[dataset]
        v7_speedup = (
            float(v7_row["speedup_vs_cusparse"])
            if v7_row.get("status") == "validated" and v7_row.get("correctness_passed") is True
            else None
        )
        rows.append(
            {
                "dataset": dataset,
                "cusparse_speedup": 1.0,
                "historical_sparsene_speedup": historical_direct[dataset] / historical_best[dataset],
                "historical_sparsene_best_ms": historical_best[dataset],
                "historical_direct_cusparse_ms": historical_direct[dataset],
                "sparsene2_srbcrs_speedup": sparsene2[dataset],
                "format_agent_v7_speedup": v7_speedup,
                "format_agent_v7_status": v7_row.get("status"),
                "format_agent_v7_generated_ms": v7_row.get("generated_ms"),
                "format_agent_v7_cusparse_ms": v7_row.get("cusparse_ms"),
                "format_agent_v7_format": v7_row.get("format_name"),
            }
        )

    csv_path = run_dir / "v7_speedup_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    labels = [row["dataset"] for row in rows]
    x = np.arange(len(labels))
    width = 0.19
    figure, axis = plt.subplots(figsize=(17.8, 7.6), constrained_layout=True)
    styles = [
        ("cuSPARSE", "cusparse_speedup", "#9AA0A6"),
        ("Sparsene (historical)", "historical_sparsene_speedup", "#4C78A8"),
        ("Sparsene 2.0 (optimized SR-BCRS)", "sparsene2_srbcrs_speedup", "#F58518"),
        ("Redesigned format (Sparsene 2.0 + Format Agent)", "format_agent_v7_speedup", "#54A24B"),
    ]
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    max_speedup = 1.0
    for offset, (label, key, color) in zip(offsets, styles):
        values = np.array([np.nan if row[key] is None else float(row[key]) for row in rows])
        finite = values[np.isfinite(values)]
        if finite.size:
            max_speedup = max(max_speedup, float(finite.max()))
        axis.bar(x + offset, values, width, label=label, color=color,
                 edgecolor="white", linewidth=0.35, zorder=3)
    # Preserve the category position of unsupported research designs without
    # inventing an acceleration value.
    unsupported = [i for i, row in enumerate(rows) if row["format_agent_v7_speedup"] is None]
    for index in unsupported:
        xpos = x[index] + offsets[3]
        axis.bar(xpos, 0.075, width, color="white", edgecolor="#54A24B", hatch="///", linewidth=0.8, zorder=4)
        axis.text(xpos, 0.16, "N/S", ha="center", va="bottom", color="#397A35", fontsize=7, rotation=90)

    axis.axhline(1.0, color="#4D4D4D", linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    axis.text(len(labels) - 0.35, 1.0, "cuSPARSE = 1", ha="right", va="bottom", color="#4D4D4D", fontsize=8)
    axis.set_xlim(-0.7, len(labels) - 0.3)
    axis.set_ylim(0, max_speedup * 1.16)
    axis.set_ylabel("Speedup over cuSPARSE (×)")
    # Dataset identity is conveyed by the 26 explicit x tick labels; omitting
    # a redundant axis title reserves a clean footer row for provenance.
    axis.set_xlabel("")
    axis.set_xticks(x, labels, rotation=52, ha="right")
    axis.set_title("N=512 FP32/TF32 SpMM: Historical Sparsene, SR-BCRS, and Format-Agent Redesign")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.9, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.5, columnspacing=1.6)
    figure.text(
        0.5, 0.005,
        "Historical Sparsene uses A100 direct-log cuSPARSE / best historical variant; "
        "Sparsene 2.0 and Format Agent use A800 measurements. N/S: valid research DSL, no current executable backend.",
        ha="center", va="bottom", fontsize=8, color="#444444",
    )
    stem = run_dir / "v7_speedup_comparison"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")

    (run_dir / "v7_speedup_comparison_notes.md").write_text(
        "# Speedup comparison figure\n\n"
        "- **cuSPARSE**: fixed 1.0 baseline in every group.\n"
        "- **Sparsene (historical)**: `cuSPARSE/direct-log / best_ms`, using A100 historical source files.\n"
        "- **Sparsene 2.0**: optimized A800 SR-BCRS speedup column in `docs/trae/01_performance_comparison_analysis.md`.\n"
        "- **Format Agent redesign**: current v7 A800 result only when full correctness passed; `N/S` means no semantic backend exists yet.\n"
        "- All groups are N=512 FP32/TF32. A100 and A800 are shown under the project-approved SM80 single-card comparability assumption; timer source differences remain explicitly documented.\n",
        encoding="utf-8",
    )
    print(f"wrote {csv_path} and {stem}.{{pdf,png}}")


if __name__ == "__main__":
    main()
