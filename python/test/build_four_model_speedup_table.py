#!/usr/bin/env python3
"""Build a correctness-gated four-version speedup comparison and LaTeX table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "logs" / "e2e_runs"
DEFAULT_OUT = RUNS / "incumbents" / "a100_a800_n512_fp32"
DATASETS = [
    "ASIC_680k", "DD", "Ga41As41H72", "OVCAR-8H", "Si41Ge41H72",
    "Yeast", "YeastH", "cant", "conf5_4-8x8-10", "consph", "cop20k_A",
    "dc2", "ddi", "eu-2005", "mac_econ_fwd500", "mc2depi", "mip1",
    "pdb1HYS", "protein", "pwtk", "reddit", "rma10", "scircuit",
    "shipsec1", "web-BerkStan", "webbase-1M",
]
METHODS = (
    ("sparsene1_paper_speedup_vs_cusparse", "Sparsene 1.0"),
    ("sparsekg_ds_v4pro_speedup", "SparseKG (DS v4 Pro)"),
    ("sparsekg_gpt6_1_sol_speedup", "SparseKG (GPT-6.1-Sol)"),
    ("sparsekg_opus5_5_speedup", "SparseKG (Opus 5.5)"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def keyed(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    result = {row["dataset"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{label}: duplicate dataset rows")
    missing = set(DATASETS) - set(result)
    extra = set(result) - set(DATASETS)
    if missing or extra:
        raise ValueError(f"{label}: missing={sorted(missing)} extra={sorted(extra)}")
    return result


def completed_best(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    """Keep records with a best correct operator; intermediate Agent errors are OK."""
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        dataset = row["dataset"]
        speedup = row.get("best_speedup_vs_cusparse", "")
        operator = row.get("best_operator_path", "")
        status = row.get("status", "")
        if status not in {"completed", "completed_with_failures"}:
            continue
        if not operator or not speedup:
            continue
        result[dataset] = row
    if not result:
        raise ValueError(f"{label}: no correctness-gated best operators found")
    return result


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive speedups")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    registry = keyed(read_csv(DEFAULT_OUT / "summary.csv"), "incumbent summary")
    # Reuse the previously audited four-method source, which selects the
    # correctness-gated DeepSeek incumbent across its baseline/strategy runs.
    ds_rows = keyed(
        read_csv(DEFAULT_OUT / "four_method_26dataset_speedup_data.csv"),
        "DeepSeek comparison",
    )

    gpt_run = RUNS / "20261013_140947_1088479_gpt-6.1-sol_format26"
    gpt_initial = completed_best(
        read_csv(gpt_run / "strategy_gpt6_1_sol_3iter_backend_v1_summary.csv"),
        "GPT initial",
    )
    gpt_retry = completed_best(
        read_csv(gpt_run / "strategy_gpt6_1_sol_3iter_backend_v1_retry_failed9_summary.csv"),
        "GPT retry",
    )
    gpt_candidates = {dataset: [row] for dataset, row in gpt_initial.items()}
    for dataset, row in gpt_retry.items():
        gpt_candidates.setdefault(dataset, []).append(row)
    gpt = {
        dataset: max(candidates, key=lambda row: float(row["best_speedup_vs_cusparse"]))
        for dataset, candidates in gpt_candidates.items()
    }

    opus_run = RUNS / "20261014_002307_claude_opus_5_5_format26"
    opus_initial = completed_best(
        read_csv(opus_run / "strategy_claude_opus_5_5_3iter_w2_r16_summary.csv"),
        "Opus initial",
    )
    opus_retry = completed_best(
        read_csv(opus_run / "strategy_claude_opus_5_5_3iter_w2_r16_backendfix_retry_summary.csv"),
        "Opus backend-fix retry",
    )
    opus_candidates = {dataset: [row] for dataset, row in opus_initial.items()}
    for dataset, row in opus_retry.items():
        opus_candidates.setdefault(dataset, []).append(row)
    opus = {
        dataset: max(candidates, key=lambda row: float(row["best_speedup_vs_cusparse"]))
        for dataset, candidates in opus_candidates.items()
    }
    gpt_retry_winners = {
        dataset for dataset, row in gpt_retry.items() if gpt.get(dataset) is row
    }
    opus_retry_winners = {
        dataset for dataset, row in opus_retry.items() if opus.get(dataset) is row
    }

    missing_gpt, missing_opus = set(DATASETS) - set(gpt), set(DATASETS) - set(opus)
    if missing_gpt or missing_opus:
        raise ValueError(f"missing GPT={sorted(missing_gpt)} Opus={sorted(missing_opus)}")

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        source = registry[dataset]
        ds_speedup = float(ds_rows[dataset]["sparsekg_ds_v4pro"])
        values = {
            "sparsene1_paper_speedup_vs_cusparse": float(
                source["sparsene1_paper_speedup_vs_cusparse"]
            ),
            "sparsekg_ds_v4pro_speedup": ds_speedup,
            "sparsekg_gpt6_1_sol_speedup": float(gpt[dataset]["best_speedup_vs_cusparse"]),
            "sparsekg_opus5_5_speedup": float(opus[dataset]["best_speedup_vs_cusparse"]),
        }
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"{dataset}: all speedups must be positive: {values}")
        best_key = max(values, key=values.get)
        rows.append(
            {
                "dataset": dataset,
                "cusparse_speedup": 1.0,
                **values,
                "fastest": dict(METHODS)[best_key],
                "opus_source": "backendfix_retry" if dataset in opus_retry_winners else "initial",
                "opus_status": opus[dataset]["status"],
                "opus_best_iteration": opus[dataset].get("best_iteration", ""),
                "opus_operator_path": opus[dataset]["best_operator_path"],
                "gpt_source": "retry_failed9" if dataset in gpt_retry_winners else "initial",
                "gpt_status": gpt[dataset]["status"],
                "gpt_best_iteration": gpt[dataset].get("best_iteration", ""),
                "gpt_operator_path": gpt[dataset]["best_operator_path"],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.out_dir / "four_model_speedup_comparison_data.csv"
    with data_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {key: geomean([float(row[key]) for row in rows]) for key, _ in METHODS}
    wins = {
        key: sum(row["fastest"] == label for row in rows)
        for key, label in METHODS
    }
    tex_path = args.out_dir / "four_model_speedup_comparison_table.tex"
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{N=512 FP32 SpMM speedup over cuSPARSE on 26 matrices. cuSPARSE is normalized to 1.0 and omitted. Sparsene 1.0 denotes the paper result; each speedup uses its source campaign's cuSPARSE timing. SparseKG entries are the fastest correctness-gated results across the named model's runs; the fastest result in each row is bold. Opus 5.5 ASIC\_680k uses canonical TCF-MCO fallback, not the proposed segmented heavy-row layout.}",
        r"  \label{tab:sparsekg-four-model-speedup}",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Dataset & Sparsene & SparseKG (DS v4 Pro) & SparseKG (GPT-6.1-Sol) & SparseKG (Opus 5.5) \\",
        r"    \midrule",
    ]
    keys = [key for key, _ in METHODS]
    for row in rows:
        vals = [float(row[key]) for key in keys]
        maximum = max(vals)
        cells = [
            rf"\textbf{{{v:.3f}}}" if math.isclose(v, maximum, rel_tol=0.0, abs_tol=1e-12) else f"{v:.3f}"
            for v in vals
        ]
        lines.append("    " + latex_escape(row["dataset"]) + " & " + " & ".join(cells) + r" \\")
    max_geo = max(summary.values())
    lines.extend(
        [
            r"    \midrule",
            "    GeoMean & "
            + " & ".join(
                rf"\textbf{{{summary[key]:.3f}}}"
                if math.isclose(summary[key], max_geo, rel_tol=0.0, abs_tol=1e-12)
                else f"{summary[key]:.3f}"
                for key in keys
            )
            + r" \\",
            "    Fastest wins & " + " & ".join(str(wins[key]) for key in keys) + r" \\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
            "",
            "% Values are normalized by each source campaign's own cuSPARSE timing.",
            "% Opus ASIC_680k uses canonical TCF-MCO fallback; the proposed heavy-row segmentation is not implemented.",
        ]
    )
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {data_path}")
    print(f"Wrote {tex_path}")
    print("Geometric mean speedup:")
    for key, label in METHODS:
        print(f"  {label}: {summary[key]:.6f}x; fastest on {wins[key]}/26")


if __name__ == "__main__":
    main()
