#!/usr/bin/env python3
"""Plot the ASIC_680k strategy-agent iteration trajectory.

The figure uses the runner's three-measurement median as the primary value and
retains min/max whiskers. Failed Agent responses are shown as explicit gaps,
never imputed as performance data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sparsene-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_rows(tuning_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(tuning_dir.glob("iteration_*/iteration_summary.json")):
        entry = json.loads(summary_path.read_text(encoding="utf-8"))
        timing = entry.get("e2e_result", {}).get("timing", {})
        samples = [float(value) for value in timing.get("generated_ms_samples", [])]
        cusparse_samples = [float(value) for value in timing.get("cusparse_ms_samples", [])]
        generated = timing.get("generated_ms")
        cusparse = timing.get("cusparse_ms")
        if generated is not None and not samples:
            samples = [float(generated)]
        if cusparse is not None and not cusparse_samples:
            cusparse_samples = [float(cusparse)]
        median_generated = float(generated) if generated is not None else None
        median_cusparse = float(cusparse) if cusparse is not None else None
        rows.append(
            {
                "iteration": int(entry["iteration"]),
                "status": entry.get("status"),
                "generated_ms": median_generated,
                "cusparse_ms": median_cusparse,
                "generated_min_ms": min(samples) if samples else None,
                "generated_max_ms": max(samples) if samples else None,
                "speedup_vs_cusparse": (
                    median_cusparse / median_generated
                    if median_generated and median_cusparse
                    else None
                ),
                "measurement_runs": timing.get("measurement_runs", len(samples)),
                "relative_spread_pct": timing.get("relative_spread_pct"),
                "strategy": entry.get("strategy", {}),
                "error": entry.get("error", ""),
            }
        )
    if not rows:
        raise ValueError(f"no iteration summaries under {tuning_dir}")
    return sorted(rows, key=lambda row: int(row["iteration"]))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "iteration", "status", "generated_ms", "generated_min_ms",
        "generated_max_ms", "cusparse_ms", "speedup_vs_cusparse",
        "measurement_runs", "relative_spread_pct", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def build_figure(tuning_dir: Path) -> tuple[Path, Path, Path]:
    rows = _load_rows(tuning_dir)
    output_dir = tuning_dir
    csv_path = output_dir / "asic_strategy_iteration_performance.csv"
    _write_csv(csv_path, rows)

    x_all = np.array([int(row["iteration"]) for row in rows], dtype=float)
    valid = [row for row in rows if row["generated_ms"] is not None]
    x = np.array([float(row["iteration"]) for row in valid])
    generated = np.array([float(row["generated_ms"]) for row in valid])
    cusparse = np.array([float(row["cusparse_ms"]) for row in valid])
    lower = generated - np.array([float(row["generated_min_ms"]) for row in valid])
    upper = np.array([float(row["generated_max_ms"]) for row in valid]) - generated
    speedups = cusparse / generated
    best_index = int(np.argmin(generated))
    best_iteration = int(x[best_index])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharex=True)
    ax_time, ax_speedup = axes
    blue = "#2F5D8C"
    orange = "#C56A2D"
    gray = "#666666"
    red = "#B23A48"

    ax_time.errorbar(
        x, generated, yerr=np.vstack([lower, upper]), fmt="o-", color=blue,
        linewidth=1.8, markersize=4.8, capsize=3.0, label="Generated kernel (median ± min/max)",
        zorder=3,
    )
    ax_time.plot(x, cusparse, "^--", color=orange, linewidth=1.35, markersize=4.2,
                 label="cuSPARSE (median)", zorder=2)
    ax_time.scatter(
        [best_iteration], [generated[best_index]], marker="*", s=130,
        color="#D4A017", edgecolors="#6F5200", linewidths=0.7,
        label=f"Best: iter {best_iteration}", zorder=5,
    )
    ax_time.annotate(
        f"iter {best_iteration}: {generated[best_index]:.3f} ms",
        xy=(best_iteration, generated[best_index]), xytext=(8, 13),
        textcoords="offset points", fontsize=8.5, color="#4A3A00",
        arrowprops={"arrowstyle": "-", "color": "#8A6D1D", "lw": 0.8},
    )
    ax_time.set_title("Kernel time")
    ax_time.set_ylabel("Time (ms, lower is better)")
    ax_time.set_xlabel("Strategy iteration")
    ax_time.legend(loc="upper left", frameon=False)

    ax_speedup.plot(x, speedups, "o-", color=blue, linewidth=1.8, markersize=4.8,
                    label="Generated / cuSPARSE")
    ax_speedup.axhline(1.0, color=gray, linewidth=1.0, linestyle=(0, (4, 3)),
                       label="cuSPARSE = 1.0×")
    ax_speedup.scatter([best_iteration], [speedups[best_index]], marker="*", s=130,
                       color="#D4A017", edgecolors="#6F5200", linewidths=0.7, zorder=5)
    ax_speedup.annotate(
        f"{speedups[best_index]:.3f}×",
        xy=(best_iteration, speedups[best_index]), xytext=(8, 13),
        textcoords="offset points", fontsize=8.5, color="#4A3A00",
    )
    ax_speedup.set_title("Relative acceleration")
    ax_speedup.set_ylabel("Speedup over cuSPARSE (×)")
    ax_speedup.set_xlabel("Strategy iteration")
    ax_speedup.legend(loc="upper left", frameon=False)

    failed = [row for row in rows if row["generated_ms"] is None]
    if failed:
        ymax = max(float(np.max(generated)) * 1.12, 1.0)
        for row in failed:
            iteration = float(row["iteration"])
            for axis in axes:
                axis.axvline(iteration, color="#E2B4B9", linewidth=0.9, alpha=0.65, zorder=0)
            ax_time.scatter([iteration], [ymax * 0.98], marker="x", s=48, color=red,
                            linewidths=1.5, zorder=4)
            ax_speedup.scatter([iteration], [ax_speedup.get_ylim()[1] * 0.96], marker="x",
                               s=48, color=red, linewidths=1.5, zorder=4)
        ax_time.text(
            0.99, 0.03, "× = Agent response failed (no timing)",
            transform=ax_time.transAxes, ha="right", va="bottom", fontsize=8, color=red,
        )

    for axis in axes:
        axis.set_xticks(x_all)
        axis.set_xlim(-0.55, max(x_all) + 0.55)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.9, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    ax_time.set_ylim(0, max(float(np.max(generated)) * 1.15, 6.5))
    ax_speedup.set_ylim(min(0.75, float(np.min(speedups)) * 0.92),
                        max(1.10, float(np.max(speedups)) * 1.08))
    figure.suptitle("ASIC_680k Strategy-Agent Feedback Trajectory (N=512, FP32)",
                    y=1.02, fontsize=13)
    figure.text(
        0.5, -0.015,
        "Three independent measurements per valid iteration; points show medians and whiskers show min/max.",
        ha="center", va="top", fontsize=8.5, color="#444444",
    )
    figure.tight_layout()
    stem = output_dir / "asic_strategy_iteration_performance"
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(figure)

    notes_path = output_dir / "asic_strategy_iteration_performance_notes.md"
    notes_path.write_text(
        "# ASIC_680k strategy iteration figure\n\n"
        f"- Source: `{tuning_dir}` iteration summaries.\n"
        "- Generated-kernel line uses the three-measurement median; whiskers retain min/max.\n"
        "- cuSPARSE is plotted per-iteration using its corresponding median.\n"
        f"- Best valid iteration: `{best_iteration}`, generated median `{generated[best_index]:.5f} ms`, "
        f"speedup `{speedups[best_index]:.5f}x`.\n"
        "- Failed Agent responses are shown as gaps and red crosses; no timing value is fabricated.\n",
        encoding="utf-8",
    )
    return csv_path, pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tuning_dir", type=Path)
    args = parser.parse_args()
    paths = build_figure(args.tuning_dir.resolve())
    print("wrote " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
