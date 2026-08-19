#!/usr/bin/env python3
"""Build the four-system 26-dataset paper figure and LaTeX table."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_BEST = REPO_ROOT / "sota_results/summary/sparsene_a100_n512.csv"
HISTORICAL_WIDE = REPO_ROOT / "sota_results/summary/a100_n512_fp32_operator_wide_ms.csv"
SPARSENE2_REPORT = REPO_ROOT / "docs/trae/01_performance_comparison_analysis.md"
PERFORMANCE_CONTRACT = "sparsene.kernel_only_post_conversion.v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sparsene2_speedups(path: Path) -> dict[str, float]:
    """Read the audited SR-BCRS speedup column from the historical report."""
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
        try:
            values[columns[0]] = float(columns[6])
        except ValueError:
            continue
    return values


def _result_for_dataset(payload: dict[str, Any], dataset: str) -> dict[str, Any]:
    matches = [item for item in payload.get("results", []) if item.get("dataset") == dataset]
    if len(matches) != 1:
        raise ValueError(f"{dataset}: expected one campaign result, found {len(matches)}")
    return matches[0]


def _best_correct_iteration(
    campaign: dict[str, Any], dataset: str, *, max_iteration: int = 5
) -> dict[str, Any]:
    result = _result_for_dataset(campaign, dataset)
    valid: list[dict[str, Any]] = []
    for entry in result.get("iterations", []):
        iteration = int(entry.get("iteration", -1))
        e2e = entry.get("e2e_result", {})
        timing = e2e.get("timing", {})
        correctness = e2e.get("correctness", {})
        if not (0 <= iteration <= max_iteration):
            continue
        if entry.get("status") != "validated" or correctness.get("passed") is not True:
            continue
        if timing.get("contract") != PERFORMANCE_CONTRACT:
            raise ValueError(f"{dataset} iteration {iteration}: incompatible timing contract")
        if int(timing.get("warmup", -1)) != 3 or int(timing.get("repeat", -1)) != 32:
            raise ValueError(f"{dataset} iteration {iteration}: incompatible warmup/repeat")
        if int(timing.get("measurement_runs", -1)) != 3:
            raise ValueError(f"{dataset} iteration {iteration}: expected three measurements")
        generated_ms = float(timing.get("generated_ms", 0.0))
        cusparse_ms = float(timing.get("cusparse_ms", 0.0))
        if generated_ms <= 0.0 or cusparse_ms <= 0.0:
            raise ValueError(f"{dataset} iteration {iteration}: non-positive timing")
        valid.append(entry)
    if not valid:
        raise ValueError(f"{dataset}: no correct iteration in [0, {max_iteration}]")
    return min(valid, key=lambda item: float(item["e2e_result"]["timing"]["generated_ms"]))


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
    return rf"\textbf{{{rendered}}}" if math.isclose(value, maximum, rel_tol=1e-12) else rendered


def _write_latex(path: Path, rows: list[dict[str, Any]], geomeans: dict[str, float]) -> None:
    columns = [
        "cusparse_speedup",
        "sparsene_speedup",
        "sparsene2_speedup",
        "sparsene_2agent_speedup",
    ]
    lines = [
        "% Requires: \\usepackage{booktabs,graphicx}",
        "% Values are speedups over the corresponding cuSPARSE baseline; larger is better.",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{SpMM speedup over cuSPARSE on 26 matrices with $N=512$ and FP32 input/output. "
        "Sparsene-2Agent reports the fastest correct generated kernel among strategy iterations 0--5; "
        "iteration 0 is the Format-Agent design with the deterministic safe strategy. "
        "The best result for each matrix is shown in bold.}",
        "  \\label{tab:sparsene-2agent-speedup}",
        "  \\setlength{\\tabcolsep}{7pt}",
        "  \\renewcommand{\\arraystretch}{1.05}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{lrrrr}",
        "    \\toprule",
        "    Matrix & cuSPARSE & Sparsene & Sparsene 2.0 & Sparsene-2Agent (best 0--5) \\\\",
        "    \\midrule",
    ]
    for row in rows:
        values = [float(row[column]) for column in columns]
        maximum = max(values)
        rendered = [_latex_value(value, maximum) for value in values]
        lines.append(f"    {_latex_escape(str(row['dataset']))} & " + " & ".join(rendered) + r" \\")
    gm_values = [geomeans[column] for column in columns]
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
            "  \\begin{minipage}{0.99\\textwidth}",
            "  \\footnotesize All values use the kernel-only post-conversion timing contract. "
            "The historical Sparsene result uses A100 measurements; Sparsene 2.0 and "
            "Sparsene-2Agent use A800 measurements. A100 and A800 are treated as comparable "
            "SM80 single-GPU platforms for this normalized comparison.",
            "  \\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_figure(path_stem: Path, rows: list[dict[str, Any]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 9.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    labels = [str(row["dataset"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.19
    styles = [
        ("cuSPARSE", "cusparse_speedup", "#B5B5B5", ""),
        ("Sparsene", "sparsene_speedup", "#4C78A8", "///"),
        ("Sparsene 2.0", "sparsene2_speedup", "#F28E2B", "\\\\"),
        ("Sparsene-2Agent (best iter. 0–5)", "sparsene_2agent_speedup", "#2A9D8F", ".."),
    ]
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    figure, axis = plt.subplots(figsize=(18.2, 7.3), constrained_layout=True)
    maximum = 1.0
    for offset, (label, key, color, hatch) in zip(offsets, styles):
        values = np.array([float(row[key]) for row in rows])
        maximum = max(maximum, float(values.max()))
        axis.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor="#333333",
            linewidth=0.38,
            hatch=hatch,
            zorder=3,
        )
    axis.axhline(1.0, color="#3F3F3F", linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    axis.set_xlim(-0.7, len(labels) - 0.3)
    axis.set_ylim(0.0, maximum * 1.15)
    axis.set_ylabel("Speedup over cuSPARSE (×)")
    axis.set_xticks(x, labels, rotation=52, ha="right")
    axis.set_title("N=512 FP32 SpMM Performance Across 26 Sparse Matrices")
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.62, alpha=0.9, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", ncol=4, frameon=False, columnspacing=1.5, handlelength=1.7)
    figure.text(
        0.5,
        0.004,
        "Sparsene-2Agent selects the fastest correct generated kernel from iterations 0–5; "
        "iteration 0 includes the Format-Agent design and safe baseline strategy.",
        ha="center",
        va="bottom",
        fontsize=8.2,
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
    datasets = list(_read_json(run_dir / "run_manifest.json")["datasets"])

    campaign_paths = {
        "asic_0_5": run_dir / "strategy_tuning_asic_v2_12iter_summary.json",
        "main_0_5": run_dir / "strategy_tuning_remaining_v2_5iter_summary.json",
        "iteration0_recovery": run_dir / "strategy_tuning_iteration0_recovery_v1_summary.json",
        "new_backends_0_5": run_dir / "strategy_tuning_yeasth_eu2005_v1_5iter_summary.json",
    }
    campaigns = {name: _read_json(path) for name, path in campaign_paths.items()}
    recovery = {"DD", "Ga41As41H72", "OVCAR-8H", "Si41Ge41H72", "Yeast"}
    new_backends = {"YeastH", "eu-2005"}

    historical_best = {row["dataset"]: float(row["best_ms"]) for row in _read_csv(HISTORICAL_BEST)}
    historical_cusparse = {
        row["dataset"]: float(row["cuSPARSE/direct-log"])
        for row in _read_csv(HISTORICAL_WIDE)
        if row.get("cuSPARSE/direct-log")
    }
    sparsene2 = _sparsene2_speedups(SPARSENE2_REPORT)

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        if dataset == "ASIC_680k":
            campaign_name = "asic_0_5"
        elif dataset in recovery:
            campaign_name = "iteration0_recovery"
        elif dataset in new_backends:
            campaign_name = "new_backends_0_5"
        else:
            campaign_name = "main_0_5"
        if dataset not in historical_best or dataset not in historical_cusparse or dataset not in sparsene2:
            raise ValueError(f"{dataset}: incomplete historical baselines")
        best = _best_correct_iteration(campaigns[campaign_name], dataset)
        timing = best["e2e_result"]["timing"]
        correctness = best["e2e_result"]["correctness"]
        speedups = {
            "cusparse_speedup": 1.0,
            "sparsene_speedup": historical_cusparse[dataset] / historical_best[dataset],
            "sparsene2_speedup": sparsene2[dataset],
            "sparsene_2agent_speedup": float(timing["speedup_vs_cusparse"]),
        }
        winner_key = max(speedups, key=speedups.get)
        rows.append(
            {
                "dataset": dataset,
                **speedups,
                "best_system": winner_key.removesuffix("_speedup"),
                "two_agent_best_iteration": int(best["iteration"]),
                "two_agent_campaign": campaign_name,
                "two_agent_generated_ms": float(timing["generated_ms"]),
                "two_agent_cusparse_ms": float(timing["cusparse_ms"]),
                "two_agent_measurement_runs": int(timing["measurement_runs"]),
                "two_agent_correctness_passed": correctness["passed"],
                "two_agent_operator_path": best.get("operator_path"),
                "two_agent_vs_sparsene": speedups["sparsene_2agent_speedup"] / speedups["sparsene_speedup"],
                "two_agent_vs_sparsene2": speedups["sparsene_2agent_speedup"] / speedups["sparsene2_speedup"],
            }
        )

    if len(rows) != 26 or {row["dataset"] for row in rows} != set(datasets):
        raise ValueError("comparison must contain exactly the 26 run-manifest datasets")
    columns = [
        "cusparse_speedup",
        "sparsene_speedup",
        "sparsene2_speedup",
        "sparsene_2agent_speedup",
    ]
    geomeans = {column: _geomean([float(row[column]) for row in rows]) for column in columns}
    win_counts = {
        system: sum(row["best_system"] == system for row in rows)
        for system in ("cusparse", "sparsene", "sparsene2", "sparsene_2agent")
    }

    csv_path = run_dir / "sparsene_2agent_speedup_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure_stem = run_dir / "sparsene_2agent_speedup_comparison"
    _write_figure(figure_stem, rows)
    latex_path = run_dir / "sparsene_2agent_speedup_table.tex"
    _write_latex(latex_path, rows, geomeans)

    audit = {
        "schema_version": "sparsene.2agent_speedup_comparison.v1",
        "performance_contract": PERFORMANCE_CONTRACT,
        "selection": "minimum generated_ms among correct iterations 0 through 5",
        "protocol": {"N": 512, "dtype": "fp32", "warmup": 3, "repeat": 32, "measurement_runs": 3},
        "dataset_count": len(rows),
        "geometric_mean_speedups": geomeans,
        "per_dataset_win_counts": win_counts,
        "sources": {
            "historical_sparsene_best": {"path": str(HISTORICAL_BEST), "sha256": _sha256(HISTORICAL_BEST)},
            "historical_cusparse": {"path": str(HISTORICAL_WIDE), "sha256": _sha256(HISTORICAL_WIDE)},
            "sparsene2": {"path": str(SPARSENE2_REPORT), "sha256": _sha256(SPARSENE2_REPORT)},
            "two_agent_campaigns": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in campaign_paths.items()
            },
        },
        "rows": rows,
    }
    audit_path = run_dir / "sparsene_2agent_speedup_comparison_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes_path = run_dir / "sparsene_2agent_speedup_comparison_notes.md"
    notes_path.write_text(
        "# Sparsene-2Agent speedup comparison\n\n"
        "- All bars are speedup over cuSPARSE; higher is better and cuSPARSE is fixed at 1.0.\n"
        "- Sparsene uses the best historical A100 result for each dataset.\n"
        "- Sparsene 2.0 uses the audited optimized SR-BCRS A800 speedup.\n"
        "- Sparsene-2Agent selects the lowest generated-kernel time among correct iterations 0--5.\n"
        "- Iteration 0 contains the Format-Agent design plus the deterministic safe strategy and is a valid final winner.\n"
        "- DD, Ga41As41H72, OVCAR-8H, Si41Ge41H72, and Yeast use a fresh iteration-0 recovery campaign with three independent measurements because their original batch was interrupted by unavailable CUDA/API access.\n"
        "- The primary contract is kernel-only after sparse-format conversion and reordering; conversion is excluded.\n"
        f"- Geometric means: cuSPARSE={geomeans['cusparse_speedup']:.4f}x, "
        f"Sparsene={geomeans['sparsene_speedup']:.4f}x, "
        f"Sparsene 2.0={geomeans['sparsene2_speedup']:.4f}x, "
        f"Sparsene-2Agent={geomeans['sparsene_2agent_speedup']:.4f}x.\n",
        encoding="utf-8",
    )
    print(f"wrote {csv_path}")
    print(f"wrote {figure_stem}.{{png,pdf}}")
    print(f"wrote {latex_path}")
    print(f"wrote {audit_path}")


if __name__ == "__main__":
    main()
