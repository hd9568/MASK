#!/usr/bin/env python3
"""Derive the three-method SparseKG paper artifacts from audited results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SOURCE_CSV = "sparsene_2agent_speedup_comparison.csv"
SOURCE_AUDIT = "sparsene_2agent_speedup_comparison_audit.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _latex_value(value: float, maximum: float) -> str:
    rendered = f"{value:.2f}"
    if math.isclose(value, maximum, rel_tol=1e-12, abs_tol=1e-12):
        return rf"\textbf{{{rendered}}}"
    return rendered


def _write_latex(
    path: Path, rows: list[dict[str, Any]], geomeans: dict[str, float]
) -> None:
    keys = ("cusparse_speedup", "sparsene_speedup", "sparsene_2agent_speedup")
    lines = [
        "% Requires: \\usepackage{booktabs,graphicx}",
        "% All entries are speedups over cuSPARSE; larger is better.",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{SpMM speedup over cuSPARSE on 26 matrices with $N=512$ and FP32 input/output. "
        "SparseKG selects the fastest full-output-correct generated kernel among "
        "strategy iterations 0--5; iteration 0 is the Format-Agent design with the "
        "deterministic safe strategy. The best result in each row is bold.}",
        "  \\label{tab:sparsekg-three-method-speedup}",
        "  \\setlength{\\tabcolsep}{10pt}",
        "  \\renewcommand{\\arraystretch}{1.05}",
        "  \\resizebox{0.82\\textwidth}{!}{%",
        "  \\begin{tabular}{lrrr}",
        "    \\toprule",
        "    Matrix & cuSPARSE & Sparsene & SparseKG (best 0--5) \\\\",
        "    \\midrule",
    ]
    for row in rows:
        values = [float(row[key]) for key in keys]
        maximum = max(values)
        rendered = [_latex_value(value, maximum) for value in values]
        lines.append(
            f"    {_latex_escape(str(row['dataset']))} & "
            + " & ".join(rendered)
            + r" \\"
        )
    gm_values = [geomeans[key] for key in keys]
    gm_maximum = max(gm_values)
    lines.extend(
        [
            "    \\midrule",
            "    GeoMean & "
            + " & ".join(_latex_value(value, gm_maximum) for value in gm_values)
            + r" \\",
            "    \\bottomrule",
            "  \\end{tabular}%",
            "  }",
            "  \\vspace{2pt}",
            "  \\begin{minipage}{0.82\\textwidth}",
            "  \\footnotesize Values use the kernel-only post-conversion contract. "
            "Historical Sparsene uses A100 measurements, whereas SparseKG uses "
            "A800 measurements; both are normalized to their corresponding cuSPARSE "
            "baseline and treated as comparable SM80 single-GPU platforms.",
            "  \\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_figure(
    path_stem: Path, rows: list[dict[str, Any]], geomeans: dict[str, float]
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    labels = [str(row["dataset"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.245
    styles = [
        ("cuSPARSE", "cusparse_speedup", "#B8B8B8", ""),
        ("Sparsene", "sparsene_speedup", "#4C78A8", "///"),
        ("SparseKG", "sparsene_2agent_speedup", "#2A9D8F", ".."),
    ]
    figure, axis = plt.subplots(figsize=(18.2, 7.2), constrained_layout=True)
    maximum = 1.0
    for offset, (label, key, color, hatch) in zip((-width, 0.0, width), styles):
        values = np.array([float(row[key]) for row in rows])
        maximum = max(maximum, float(values.max()))
        axis.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor="#303030",
            linewidth=0.42,
            hatch=hatch,
            zorder=3,
        )
    axis.axhline(1.0, color="#333333", linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    axis.set_xlim(-0.72, len(labels) - 0.28)
    axis.set_ylim(0.0, maximum * 1.14)
    axis.set_ylabel("Speedup over cuSPARSE (×)")
    axis.set_xticks(x, labels, rotation=52, ha="right")
    axis.set_title("N=512 FP32 SpMM Performance Across 26 Sparse Matrices")
    axis.grid(axis="y", color="#D5D5D5", linewidth=0.62, alpha=0.9, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", ncol=3, frameon=False, columnspacing=1.8)
    figure.text(
        0.5,
        0.004,
        "Geometric mean speedup: cuSPARSE "
        f"{geomeans['cusparse_speedup']:.2f}×, Sparsene "
        f"{geomeans['sparsene_speedup']:.2f}×, SparseKG "
        f"{geomeans['sparsene_2agent_speedup']:.2f}×. "
        "Only full-output-correct Agent candidates are eligible.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#444444",
    )
    figure.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path_stem.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    source_csv = run_dir / SOURCE_CSV
    source_audit = run_dir / SOURCE_AUDIT
    rows_in = _read_csv(source_csv)
    if len(rows_in) != 26 or len({row["dataset"] for row in rows_in}) != 26:
        raise ValueError("audited source must contain exactly 26 unique datasets")

    rows: list[dict[str, Any]] = []
    for source in rows_in:
        if source.get("two_agent_correctness_passed") != "True":
            raise ValueError(f"{source['dataset']}: selected Agent result is not correct")
        if int(source["two_agent_measurement_runs"]) != 3:
            raise ValueError(f"{source['dataset']}: expected three independent measurements")
        cu = float(source["cusparse_speedup"])
        sparsene = float(source["sparsene_speedup"])
        agent = float(source["sparsene_2agent_speedup"])
        if not math.isclose(cu, 1.0) or min(sparsene, agent) <= 0.0:
            raise ValueError(f"{source['dataset']}: invalid speedup values")
        values = {"cusparse": cu, "sparsene": sparsene, "sparsene_2agent": agent}
        rows.append(
            {
                "dataset": source["dataset"],
                "cusparse_speedup": cu,
                "sparsene_speedup": sparsene,
                "sparsene_2agent_speedup": agent,
                "best_system": max(values, key=values.get),
                "two_agent_vs_sparsene": agent / sparsene,
                "two_agent_best_iteration": int(source["two_agent_best_iteration"]),
                "two_agent_campaign": source["two_agent_campaign"],
                "two_agent_generated_ms": float(source["two_agent_generated_ms"]),
                "two_agent_cusparse_ms": float(source["two_agent_cusparse_ms"]),
                "two_agent_operator_path": source["two_agent_operator_path"],
            }
        )

    keys = ("cusparse_speedup", "sparsene_speedup", "sparsene_2agent_speedup")
    geomeans = {key: _geomean([float(row[key]) for row in rows]) for key in keys}
    wins = [row for row in rows if row["sparsene_2agent_speedup"] > row["sparsene_speedup"]]
    win_ratios = [float(row["two_agent_vs_sparsene"]) for row in wins]

    def _relative_extreme(row: dict[str, object]) -> dict[str, object]:
        ratio = float(row["two_agent_vs_sparsene"])
        return {
            "dataset": row["dataset"],
            "relative_speedup": ratio,
            "percent_change": (ratio - 1.0) * 100.0,
        }

    statistics = {
        "agent_faster_than_sparsene_count": len(wins),
        "agent_faster_than_sparsene_datasets": [row["dataset"] for row in wins],
        "winning_subset_arithmetic_mean_relative_speedup": sum(win_ratios) / len(win_ratios),
        "winning_subset_geometric_mean_relative_speedup": _geomean(win_ratios),
        "winning_subset_arithmetic_mean_speedup_over_cusparse": sum(
            float(row["sparsene_2agent_speedup"]) for row in wins
        )
        / len(wins),
        "winning_subset_geometric_mean_speedup_over_cusparse": _geomean(
            [float(row["sparsene_2agent_speedup"]) for row in wins]
        ),
        "agent_faster_than_cusparse_count": sum(
            row["sparsene_2agent_speedup"] > 1.0 for row in rows
        ),
        "all_dataset_min_relative_speedup": _relative_extreme(
            min(rows, key=lambda row: float(row["two_agent_vs_sparsene"]))
        ),
        "all_dataset_max_relative_speedup": _relative_extreme(
            max(rows, key=lambda row: float(row["two_agent_vs_sparsene"]))
        ),
        "winning_subset_min_relative_speedup": _relative_extreme(
            min(wins, key=lambda row: float(row["two_agent_vs_sparsene"]))
        ),
        "winning_subset_max_relative_speedup": _relative_extreme(
            max(wins, key=lambda row: float(row["two_agent_vs_sparsene"]))
        ),
    }

    csv_path = run_dir / "three_method_speedup_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stem = run_dir / "three_method_speedup_comparison"
    _write_figure(stem, rows, geomeans)
    latex_path = run_dir / "three_method_speedup_table.tex"
    _write_latex(latex_path, rows, geomeans)
    audit = {
        "schema_version": "sparsekg.three_method_speedup_comparison.v1",
        "performance_contract": "sparsene.kernel_only_post_conversion.v1",
        "selection": "minimum generated_ms among full-output-correct iterations 0 through 5, including iteration 0",
        "protocol": {
            "N": 512,
            "dtype": "fp32",
            "warmup": 3,
            "repeat": 32,
            "measurement_runs": 3,
        },
        "dataset_count": len(rows),
        "geometric_mean_speedups": geomeans,
        "statistics": statistics,
        "sources": {
            "audited_four_method_csv": {"path": str(source_csv), "sha256": _sha256(source_csv)},
            "audited_four_method_audit": {"path": str(source_audit), "sha256": _sha256(source_audit)},
        },
        "rows": rows,
    }
    audit_path = run_dir / "three_method_speedup_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {stem}.{{png,pdf}}")
    print(f"wrote {latex_path}")
    print(f"wrote {audit_path}")
    print(json.dumps({"geomeans": geomeans, **statistics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
