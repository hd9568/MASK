#!/usr/bin/env python3
"""Plot cuSPARSE, Sparsene, and SparseKG from the incumbent summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = (
    REPO_ROOT / "logs" / "e2e_runs" / "incumbents" / "a100_a800_n512_fp32"
)
OUTPUT_STEM = "incumbent_three_method_speedup_comparison"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _speedup_cell(value: str) -> float:
    return float(value.removesuffix("×").strip())


def _read_markdown(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 9:
            raise ValueError(f"unexpected incumbent Markdown row: {line}")
        rows.append(
            {
                "dataset": cells[0].strip("`"),
                "sparsekg_speedup": _speedup_cell(cells[3]),
                "sparsene_speedup": _speedup_cell(cells[4]),
            }
        )
    if len(rows) != 26 or len({row["dataset"] for row in rows}) != 26:
        raise ValueError("summary.md must contain exactly 26 unique datasets")
    return rows


def _read_source_rows(registry_dir: Path) -> list[dict[str, Any]]:
    markdown_path = registry_dir / "summary.md"
    csv_path = registry_dir / "summary.csv"
    markdown_rows = _read_markdown(markdown_path)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != 26:
        raise ValueError("summary.csv must contain exactly 26 rows")
    csv_by_dataset = {row["dataset"]: row for row in csv_rows}
    if len(csv_by_dataset) != 26:
        raise ValueError("summary.csv contains duplicate datasets")

    rows: list[dict[str, Any]] = []
    for markdown_row in markdown_rows:
        dataset = str(markdown_row["dataset"])
        source = csv_by_dataset.get(dataset)
        if source is None:
            raise ValueError(f"{dataset}: missing from summary.csv")
        sparsekg = float(source["speedup_vs_cusparse"])
        sparsene = float(source["sparsene1_paper_speedup_vs_cusparse"])
        if min(sparsekg, sparsene) <= 0.0:
            raise ValueError(f"{dataset}: speedups must be positive")
        if not math.isclose(
            sparsekg,
            float(markdown_row["sparsekg_speedup"]),
            rel_tol=0.0,
            abs_tol=5.1e-5,
        ):
            raise ValueError(f"{dataset}: SparseKG Markdown/CSV mismatch")
        if not math.isclose(
            sparsene,
            float(markdown_row["sparsene_speedup"]),
            rel_tol=0.0,
            abs_tol=5.1e-5,
        ):
            raise ValueError(f"{dataset}: Sparsene Markdown/CSV mismatch")
        rows.append(
            {
                "dataset": dataset,
                "cusparse_speedup": 1.0,
                "sparsene_speedup": sparsene,
                "sparsekg_speedup": sparsekg,
                "sparsekg_generated_ms": float(source["generated_ms"]),
                "sparsekg_cusparse_ms": float(source["cusparse_ms"]),
                "source_run": source["source_run"],
                "format_name": source["format_name"],
                "strategy_iteration": int(source["strategy_iteration"]),
                "operator_sha256": source["operator_sha256"],
            }
        )
    return rows


def _write_data(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(text: str) -> str:
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
    return "".join(replacements.get(character, character) for character in text)


def _latex_speedup(value: float, row_maximum: float) -> str:
    rendered = f"{value:.3f}"
    if math.isclose(value, row_maximum, rel_tol=0.0, abs_tol=1.0e-12):
        return rf"\textbf{{{rendered}}}"
    return rendered


def _write_latex_table(
    path: Path,
    rows: list[dict[str, Any]],
    geomeans: dict[str, float],
) -> None:
    keys = ("sparsene_speedup", "sparsekg_speedup")
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{N=512 FP32 SpMM speedup over cuSPARSE on 26 sparse matrices. "
        r"cuSPARSE is the normalized 1.0 baseline and is omitted from the table. "
        r"SparseKG reports the fastest compatible "
        r"correctness-gated incumbent for each matrix. The best result in each row "
        r"is shown in bold.}",
        r"  \label{tab:sparsekg-incumbent-speedup}",
        r"  \small",
        r"  \setlength{\tabcolsep}{10pt}",
        r"  \begin{tabular}{lrr}",
        r"    \toprule",
        r"    Dataset & Sparsene & SparseKG \\",
        r"    \midrule",
    ]
    for row in rows:
        values = [float(row[key]) for key in keys]
        maximum = max(values)
        rendered = [
            _latex_speedup(value, maximum) for value in values
        ]
        lines.append(
            "    "
            + _latex_escape(str(row["dataset"]))
            + " & "
            + " & ".join(rendered)
            + r" \\" 
        )
    geomean_values = [float(geomeans[key]) for key in keys]
    geomean_maximum = max(geomean_values)
    lines.extend(
        [
            r"    \midrule",
            "    GeoMean & "
            + " & ".join(
                _latex_speedup(value, geomean_maximum)
                for value in geomean_values
            )
            + r" \\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(
    path_stem: Path,
    rows: list[dict[str, Any]],
    geomeans: dict[str, float],
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
        ("SparseKG", "sparsekg_speedup", "#2A9D8F", ".."),
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
    axis.axhline(
        1.0,
        color="#333333",
        linewidth=0.9,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
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
        f"{geomeans['sparsekg_speedup']:.2f}×. "
        "SparseKG uses the fastest compatible correctness-gated incumbent per matrix.",
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
    parser.add_argument(
        "registry_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"incumbent registry directory (default: {DEFAULT_REGISTRY})",
    )
    args = parser.parse_args()
    registry_dir = args.registry_dir.resolve()
    rows = _read_source_rows(registry_dir)
    keys = ("cusparse_speedup", "sparsene_speedup", "sparsekg_speedup")
    geomeans = {
        key: _geomean([float(row[key]) for row in rows]) for key in keys
    }

    data_path = registry_dir / f"{OUTPUT_STEM}_data.csv"
    _write_data(data_path, rows)
    latex_path = registry_dir / f"{OUTPUT_STEM}_table.tex"
    _write_latex_table(latex_path, rows, geomeans)
    figure_stem = registry_dir / OUTPUT_STEM
    _write_figure(figure_stem, rows, geomeans)

    audit = {
        "schema_version": "sparsekg.incumbent_three_method_speedup.v1",
        "performance_contract": "sparsene.kernel_only_post_conversion.v1",
        "protocol": {
            "N": 512,
            "dtype": "fp32",
            "mma": "tf32",
            "warmup": 3,
            "repeat": 32,
            "measurement_runs": 3,
            "hardware_scope": "A100/A800-class SM80 single GPU",
        },
        "methods": {
            "cusparse": "normalized baseline fixed to 1.0",
            "sparsene": "published A100 N=512 FP32 per-dataset baseline",
            "sparsekg": "fastest compatible correctness-gated cross-run incumbent per dataset",
        },
        "dataset_count": len(rows),
        "geometric_mean_speedups": geomeans,
        "sparsekg_faster_than_cusparse": sum(
            float(row["sparsekg_speedup"]) > 1.0 for row in rows
        ),
        "sparsekg_faster_than_sparsene": sum(
            float(row["sparsekg_speedup"]) > float(row["sparsene_speedup"])
            for row in rows
        ),
        "sources": {
            "summary_markdown": {
                "path": str((registry_dir / "summary.md").resolve()),
                "sha256": _sha256(registry_dir / "summary.md"),
            },
            "summary_csv": {
                "path": str((registry_dir / "summary.csv").resolve()),
                "sha256": _sha256(registry_dir / "summary.csv"),
            },
        },
        "outputs": {
            "latex_table": {
                "path": str(latex_path.resolve()),
                "sha256": _sha256(latex_path),
                "best_value_rule": (
                    "maximum full-precision speedup among the displayed Sparsene and "
                    "SparseKG values in each dataset row and the geometric-mean row; "
                    "ties within absolute tolerance 1e-12 are bold; normalized "
                    "cuSPARSE=1.0 is stated in the caption and omitted from the table"
                ),
                "display_precision": 3,
            }
        },
        "rows": rows,
    }
    audit_path = registry_dir / f"{OUTPUT_STEM}_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    notes_path = registry_dir / f"{OUTPUT_STEM}_notes.md"
    notes_path.write_text(
        "# cuSPARSE / Sparsene / SparseKG incumbent comparison\n\n"
        "- cuSPARSE is normalized to 1.0 for every dataset.\n"
        "- Sparsene is the published A100 N=512 FP32 per-dataset speedup.\n"
        "- SparseKG is the fastest compatible full-output-correct incumbent maintained "
        "across the fixed-format and joint Format/Strategy runs.\n"
        "- The companion LaTeX table uses full-precision values to select each row's "
        "winner and displays speedups to three decimal places.\n"
        "- A100 and A800 are treated as comparable SM80/108-SM platforms only for this "
        "normalized comparison; absolute milliseconds are not mixed across GPUs.\n"
        f"- GeoMean: cuSPARSE={geomeans['cusparse_speedup']:.4f}×, "
        f"Sparsene={geomeans['sparsene_speedup']:.4f}×, "
        f"SparseKG={geomeans['sparsekg_speedup']:.4f}×.\n"
        f"- SparseKG wins over cuSPARSE on {audit['sparsekg_faster_than_cusparse']}/26 "
        f"datasets and over Sparsene on {audit['sparsekg_faster_than_sparsene']}/26.\n",
        encoding="utf-8",
    )
    print(f"wrote {figure_stem}.{{png,pdf}}")
    print(f"wrote {data_path}")
    print(f"wrote {latex_path}")
    print(f"wrote {audit_path}")
    print(f"wrote {notes_path}")
    print(json.dumps(audit["geometric_mean_speedups"], indent=2))


if __name__ == "__main__":
    main()
