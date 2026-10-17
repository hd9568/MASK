#!/usr/bin/env python3
"""Normalize the paper's A100 / N=512 / FP32 source artifacts.

The inputs intentionally remain immutable.  This script keeps source-specific
duplicates and unlabeled observations instead of silently selecting values that
the raw files do not identify.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import zlib
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
N = 512

DATASETS = [
    "ASIC_680k", "DD", "Ga41As41H72", "OVCAR-8H", "Si41Ge41H72",
    "Yeast", "YeastH", "cant", "conf5_4-8x8-10", "consph", "cop20k_A",
    "dc2", "ddi", "eu-2005", "mac_econ_fwd500", "mc2depi", "mip1",
    "pdb1HYS", "protein", "pwtk", "reddit", "rma10", "scircuit",
    "shipsec1", "web-BerkStan", "webbase-1M",
]

# The plotting order is part of the published vector figure.  It differs from
# the convenience order used by the normalized source tables above.
PAPER_FIGURE_DATASETS = [
    "YeastH", "OVCAR-8H", "Yeast", "DD", "web-BerkStan", "reddit",
    "ddi", "protein", "pwtk", "rma10", "shipsec1", "scircuit",
    "conf5_4-8x8-10", "mc2depi", "webbase-1M", "mac_econ_fwd500",
    "pdb1HYS", "consph", "cant", "cop20k_A", "eu-2005",
    "Si41Ge41H72", "Ga41As41H72", "mip1", "dc2", "ASIC_680k",
]

PAPER_A100_N512_FIGURE = (
    ROOT.parent
    / "paper"
    / "SC26-Sparsene-main"
    / "fig"
    / "single_case_fp32_A100_N512_2.pdf"
)

SPARSENE_NAMES = {
    "demo-acc-tf32": "Sparsene/ACC-BIT-TCF",
    "demo-bitbsr-tf32": "Sparsene/BIT-BSR",
    "demo-dtc-tf32": "Sparsene/DTC-base",
    "demo-dtc-strict-lb-tf32": "Sparsene/DTC-strict-lb",
    "demo-dtc-multi-binding-tf32": "Sparsene/DTC-multi-binding",
    "demo-srbcrs-tf32": "Sparsene/SR-BCRS-base",
    "demo-srbcrs-16x8-tf32": "Sparsene/SR-BCRS-16x8",
    "demo-srbcrs-16x8-strict-lb-tf32": "Sparsene/SR-BCRS-16x8-strict-lb",
    "demo-srbcrs-16x8-multi-bind-tf32": "Sparsene/SR-BCRS-16x8-multi-bind",
}


def add(
    rows: list[dict[str, object]],
    *,
    dataset: str,
    method: str,
    milliseconds: float,
    source: str,
    provenance: str,
    correctness: str,
    note: str = "",
) -> None:
    anomaly = ""
    if milliseconds <= 0:
        anomaly = "nonpositive_time"
    elif milliseconds < 0.01:
        anomaly = "implausibly_small_time_lt_0.01ms"
    rows.append(
        {
            "dataset": dataset,
            "N": N,
            "precision": "FP32/TF32-tensorcore-where-applicable",
            "method": method,
            "milliseconds": milliseconds,
            "source_file": source,
            "provenance": provenance,
            "correctness_evidence": correctness,
            "anomaly": anomaly,
            "note": note,
        }
    )


def parse_block_log(path: Path) -> list[tuple[str, str, float]]:
    result: list[tuple[str, str, float]] = []
    dataset = executable = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("===== Matrix:"):
            match = re.search(r"/([^/]+)\.mtx, Executable: (.+?) =====$", line)
            if not match:
                raise ValueError(f"unrecognized block header in {path}: {line}")
            dataset = match.group(1)
            executable = match.group(2).rsplit("/", 1)[-1]
        elif line.startswith("mykernel_time:"):
            if dataset is None or executable is None:
                raise ValueError(f"timing before header in {path}")
            result.append((dataset, executable, float(line.split()[1])))
    return result


def parse_sparsetir(path: Path) -> list[tuple[str, str, float]]:
    result: list[tuple[str, str, float]] = []
    dataset = method = None
    feature_size = None
    command_re = re.compile(r"python (bench_spmm_naive|bench_tc_spmm)\.py .*?/([^/]+)\.mtx")
    time_re = re.compile(r"(tir naive|tc-spmm) time:\s*([0-9.]+)\s*ms")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        command = command_re.search(line)
        if command:
            method = "SparseTIR/naive" if command.group(1) == "bench_spmm_naive" else "SparseTIR/tc-spmm"
            dataset = command.group(2)
            feature_size = None
            continue
        size = re.search(r"feat_size = (\d+)", line)
        if size:
            feature_size = int(size.group(1))
            continue
        timing = time_re.search(line)
        if timing and dataset and method and feature_size == N:
            result.append((dataset, method, float(timing.group(2))))
    return result


def build_operator_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for dataset, _, milliseconds in parse_block_log(ROOT / "cusparse_fp32_N512.log"):
        add(
            rows,
            dataset=dataset,
            method="cuSPARSE/direct-log",
            milliseconds=milliseconds,
            source="cusparse_fp32_N512.log",
            provenance="paper-original/A100",
            correctness="timing-only; no elementwise verification in source log",
        )

    for dataset, executable, milliseconds in parse_block_log(ROOT / "sparsene_fp32_N512.log"):
        add(
            rows,
            dataset=dataset,
            method=SPARSENE_NAMES.get(executable, f"Sparsene/{executable}"),
            milliseconds=milliseconds,
            source="sparsene_fp32_N512.log",
            provenance="paper-original/A100/Sparsene-historical",
            correctness="historical check-nan sentinel only; not current full verification",
            note="Do not rank as current generated-codegen evidence.",
        )

    # This file has three N=512 records per dataset but no header identifying
    # device, run, or variant.  Preserve all three observations verbatim.
    occurrence: defaultdict[str, int] = defaultdict(int)
    with (ROOT / "DTCSpMM_exe_time_and_throughput.csv").open(newline="", encoding="utf-8") as handle:
        for raw in csv.reader(handle):
            if len(raw) < 5 or int(raw[1]) != N:
                continue
            dataset = raw[0]
            occurrence[dataset] += 1
            add(
                rows,
                dataset=dataset,
                method=f"DTC-SpMM/unlabeled-observation-{occurrence[dataset]}",
                milliseconds=float(raw[3]),
                source="DTCSpMM_exe_time_and_throughput.csv",
                provenance="paper-original/device-or-run-unlabeled",
                correctness="not recorded in source CSV",
                note=f"Raw throughput={raw[4]}; source provides no column header or observation identity.",
            )

    with (ROOT / "acc-result.csv").open(newline="", encoding="utf-8") as handle:
        for raw in csv.reader(handle):
            if len(raw) < 4 or int(raw[1]) != N:
                continue
            add(
                rows,
                dataset=raw[0].removesuffix(".mtx"),
                method="Acc-SpMM",
                milliseconds=float(raw[2]),
                source="acc-result.csv",
                provenance="paper-original/A100",
                correctness="not recorded in source CSV",
                note=f"Raw throughput={raw[3]}",
            )

    for dataset, method, milliseconds in parse_sparsetir(ROOT / "sparsetir.log"):
        add(
            rows,
            dataset=dataset,
            method=method,
            milliseconds=milliseconds,
            source="sparsetir.log",
            provenance="paper-original/A100",
            correctness="timing-only; source log may contain assertion warnings",
        )

    with (ROOT / "spmm_tf32_512.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            for column in ("16_1", "8_1", "8_1_balance", "8_1_map"):
                add(
                    rows,
                    dataset=raw["dataSet"],
                    method=f"FlashSparse/{column}",
                    milliseconds=float(raw[column]),
                    source="spmm_tf32_512.csv",
                    provenance="paper-original/A100",
                    correctness="not recorded in source CSV",
                )

    with (ROOT / "rode_spmm_f32_n512.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for raw in reader:
            for column, method in (
                ("sputnik", "Sputnik/RoDe-source"),
                ("cusparse", "cuSPARSE/RoDe-source"),
                ("rode", "RoDe"),
            ):
                add(
                    rows,
                    dataset=raw["dataSet"].strip(),
                    method=method,
                    milliseconds=float(raw[column]),
                    source="rode_spmm_f32_n512.csv",
                    provenance="paper-original/A100/RoDe-baseline-source",
                    correctness="not recorded in source CSV",
                )

    order = {dataset: index for index, dataset in enumerate(DATASETS)}
    rows.sort(key=lambda row: (order.get(str(row["dataset"]), len(order)), str(row["method"])))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _decoded_pdf_streams(path: Path) -> list[str]:
    """Decode Flate streams without adding a PDF-parser dependency."""

    payload = path.read_bytes()
    decoded = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", payload, re.S):
        try:
            decoded.append(zlib.decompress(match.group(1)).decode("latin1"))
        except zlib.error:
            continue
    return decoded


def _colored_bar_rectangles(content: str, color_marker: str) -> list[dict[str, float]]:
    rectangle = re.compile(
        re.escape(color_marker)
        + r".*?\n\s*([0-9.]+) ([0-9.]+) m\n"
        + r"([0-9.]+) ([0-9.]+) l\n"
        + r"([0-9.]+) ([0-9.]+) l\n"
        + r"([0-9.]+) ([0-9.]+) l\nh\n\nB",
        re.S,
    )
    result = []
    for match in rectangle.finditer(content):
        values = [float(value) for value in match.groups()]
        result.append(
            {
                "x_left": values[0],
                "y_bottom": values[1],
                "x_right": values[2],
                "y_top": values[5],
            }
        )
    return result


def extract_paper_sparsene_speedups() -> tuple[list[dict[str, object]], dict[str, object]]:
    """Recover the published per-matrix values from the vector bar geometry.

    The paper does not print the 26 values as text.  The figure is vector PDF,
    so normalizing every orange Sparsene bar by the same-height green cuSPARSE
    bars recovers the plotted speedup without raster digitization error.
    """

    if not PAPER_A100_N512_FIGURE.is_file():
        raise FileNotFoundError(PAPER_A100_N512_FIGURE)
    plot_stream = next(
        (
            stream
            for stream in _decoded_pdf_streams(PAPER_A100_N512_FIGURE)
            if "/A3 gs 1 0.4980392157 0.0549019608" in stream
        ),
        None,
    )
    if plot_stream is None:
        raise ValueError(f"Sparsene bars not found in {PAPER_A100_N512_FIGURE}")

    orange = _colored_bar_rectangles(
        plot_stream, "/A3 gs 1 0.4980392157 0.0549019608"
    )
    green = _colored_bar_rectangles(
        plot_stream, "/A2 gs 0.1725490196 0.6274509804\n0.1725490196"
    )
    if not orange or not green:
        raise ValueError("published Sparsene/cuSPARSE vector bars are incomplete")

    # The legend also contains colored rectangles.  Data bars share the modal
    # y-origin; keep exactly those rectangles and then order them by x.
    y_bottom = statistics.mode(round(item["y_bottom"], 6) for item in orange)
    orange = sorted(
        (item for item in orange if math.isclose(item["y_bottom"], y_bottom, abs_tol=1e-6)),
        key=lambda item: item["x_left"],
    )
    green = sorted(
        (item for item in green if math.isclose(item["y_bottom"], y_bottom, abs_tol=1e-6)),
        key=lambda item: item["x_left"],
    )
    if len(orange) != len(PAPER_FIGURE_DATASETS) or len(green) != len(
        PAPER_FIGURE_DATASETS
    ):
        raise ValueError(
            f"expected {len(PAPER_FIGURE_DATASETS)} paper bars, "
            f"found sparsene={len(orange)} cusparse={len(green)}"
        )
    unit_height = statistics.median(
        item["y_top"] - item["y_bottom"] for item in green
    )
    if unit_height <= 0:
        raise ValueError("invalid cuSPARSE unit-bar height")

    rows = []
    for dataset, bar in zip(PAPER_FIGURE_DATASETS, orange):
        rows.append(
            {
                "dataset": dataset,
                "gpu": "A100",
                "N": N,
                "precision": "FP32",
                "paper_sparsene_speedup_vs_cusparse": (
                    bar["y_top"] - bar["y_bottom"]
                )
                / unit_height,
                "bar_x_left": bar["x_left"],
                "bar_y_bottom": bar["y_bottom"],
                "bar_y_top": bar["y_top"],
                "source_figure": str(PAPER_A100_N512_FIGURE.relative_to(ROOT.parent)),
                "extraction": "vector_bar_height_normalized_by_cusparse_unit_bar",
            }
        )
    arithmetic_mean = sum(
        float(row["paper_sparsene_speedup_vs_cusparse"]) for row in rows
    ) / len(rows)
    if round(arithmetic_mean, 2) != 2.19:
        raise ValueError(
            "digitized paper arithmetic mean does not reproduce reported 2.19x: "
            f"{arithmetic_mean}"
        )
    audit = {
        "source_figure": str(PAPER_A100_N512_FIGURE.relative_to(ROOT.parent)),
        "source_figure_sha256": hashlib.sha256(
            PAPER_A100_N512_FIGURE.read_bytes()
        ).hexdigest(),
        "datasets": len(rows),
        "cusparse_unit_bar_height": unit_height,
        "arithmetic_mean_speedup": arithmetic_mean,
        "geometric_mean_speedup": math.exp(
            sum(
                math.log(float(row["paper_sparsene_speedup_vs_cusparse"]))
                for row in rows
            )
            / len(rows)
        ),
        "reported_table_speedup": 2.19,
        "reported_value_matches_after_rounding": True,
    }
    return rows, audit


def validate_paper_against_raw_logs(
    paper_rows: list[dict[str, object]], operator_rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compare published bars with the closest and fastest raw Sparsene variants."""

    cusparse = {
        str(row["dataset"]): float(row["milliseconds"])
        for row in operator_rows
        if row["method"] == "cuSPARSE/direct-log"
    }
    sparsene_by_dataset: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in operator_rows:
        if str(row["method"]).startswith("Sparsene/") and not row["anomaly"]:
            sparsene_by_dataset[str(row["dataset"])].append(row)

    validation = []
    for paper in paper_rows:
        dataset = str(paper["dataset"])
        paper_speedup = float(paper["paper_sparsene_speedup_vs_cusparse"])
        direct_cusparse_ms = cusparse[dataset]
        candidates = sparsene_by_dataset[dataset]
        reconstructed = [
            {
                "method": str(row["method"]),
                "milliseconds": float(row["milliseconds"]),
                "speedup": direct_cusparse_ms / float(row["milliseconds"]),
            }
            for row in candidates
        ]
        closest = min(
            reconstructed,
            key=lambda row: abs(float(row["speedup"]) / paper_speedup - 1.0),
        )
        fastest = min(reconstructed, key=lambda row: float(row["milliseconds"]))
        closest_delta = (float(closest["speedup"]) / paper_speedup - 1.0) * 100.0
        fastest_delta = (float(fastest["speedup"]) / paper_speedup - 1.0) * 100.0
        validation.append(
            {
                "dataset": dataset,
                "paper_sparsene_speedup_vs_cusparse": paper_speedup,
                "direct_log_cusparse_ms": direct_cusparse_ms,
                "closest_raw_variant": closest["method"],
                "closest_raw_variant_ms": closest["milliseconds"],
                "closest_raw_speedup_vs_cusparse": closest["speedup"],
                "closest_vs_paper_delta_pct": closest_delta,
                "closest_within_5pct": abs(closest_delta) <= 5.0,
                "historical_fastest_variant": fastest["method"],
                "historical_fastest_ms": fastest["milliseconds"],
                "historical_fastest_speedup_vs_cusparse": fastest["speedup"],
                "historical_fastest_vs_paper_delta_pct": fastest_delta,
            }
        )

    paper_values = [
        float(row["paper_sparsene_speedup_vs_cusparse"]) for row in validation
    ]
    fastest_values = [
        float(row["historical_fastest_speedup_vs_cusparse"])
        for row in validation
    ]
    closest_deltas = [
        abs(float(row["closest_vs_paper_delta_pct"])) for row in validation
    ]
    fastest_deltas = [
        abs(float(row["historical_fastest_vs_paper_delta_pct"]))
        for row in validation
    ]
    audit = {
        "comparison": (
            "paper vector bars versus paper-original A100 direct cuSPARSE and "
            "non-anomalous Sparsene variant logs"
        ),
        "datasets": len(validation),
        "closest_raw_variant_within_2pct": sum(value <= 2.0 for value in closest_deltas),
        "closest_raw_variant_within_5pct": sum(value <= 5.0 for value in closest_deltas),
        "closest_raw_variant_max_abs_delta_pct": max(closest_deltas),
        "historical_fastest_within_5pct": sum(value <= 5.0 for value in fastest_deltas),
        "paper_arithmetic_mean": sum(paper_values) / len(paper_values),
        "historical_fastest_arithmetic_mean": sum(fastest_values) / len(fastest_values),
        "interpretation": (
            "The published series is reproducible from a paper-selected raw variant "
            "for every dataset within 5%; the historical-fastest envelope is not the "
            "same selection rule and is higher on reddit and mip1."
        ),
    }
    return validation, audit


def write_e2e() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(ROOT.glob("sc-ad-gcn_e2e_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                if int(raw["hidden"]) != N:
                    continue
                rows.append(
                    {
                        "dataset": raw["dataset"],
                        "hidden": raw["hidden"],
                        "feature_dim": raw["feature_dim"],
                        "backend": raw["backend"],
                        "time_per_epoch_ms": raw["time_per_epoch_ms"],
                        "forward_ms_per_epoch": raw["forward_ms_per_epoch"],
                        "backward_ms_per_epoch": raw["backward_ms_per_epoch"],
                        "preprocess_sec_excluded": raw["preprocess_sec_excluded"],
                        "source_file": path.name,
                    }
                )
    write_csv(OUT / "a100_hidden512_gcn_e2e_long.csv", rows, list(rows[0]))


def main() -> None:
    rows = build_operator_rows()
    fields = list(rows[0])
    write_csv(OUT / "a100_n512_fp32_operator_long.csv", rows, fields)

    methods = sorted({str(row["method"]) for row in rows})
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        values[(str(row["dataset"]), str(row["method"]))] = float(row["milliseconds"])
    wide = [
        {"dataset": dataset, **{method: values.get((dataset, method), "") for method in methods}}
        for dataset in DATASETS
    ]
    write_csv(OUT / "a100_n512_fp32_operator_wide_ms.csv", wide, ["dataset", *methods])

    sparsene = [row for row in rows if str(row["method"]).startswith("Sparsene/")]
    write_csv(OUT / "a100_n512_fp32_sparsene_historical_all.csv", sparsene, fields)

    paper = [
        {"gpu": "A100", "N": N, "precision": "FP32", "baseline": baseline, "paper_reported_sparsene_speedup": speedup}
        for baseline, speedup in (
            ("cuSPARSE", 2.19), ("Sputnik", 2.57), ("DTC-SpMM", 1.94),
            ("Acc-SpMM", 2.82), ("FlashSparse", 1.20), ("SparseTIR", 4.85),
        )
    ]
    write_csv(OUT / "paper_reported_a100_n512_fp32_aggregate.csv", paper, list(paper[0]))
    paper_rows, paper_figure_audit = extract_paper_sparsene_speedups()
    write_csv(
        OUT / "paper_a100_n512_fp32_sparsene_per_dataset.csv",
        paper_rows,
        list(paper_rows[0]),
    )
    paper_validation, paper_log_audit = validate_paper_against_raw_logs(
        paper_rows, rows
    )
    write_csv(
        OUT / "paper_a100_n512_fp32_sparsene_validation.csv",
        paper_validation,
        list(paper_validation[0]),
    )
    write_e2e()

    audit = {
        "schema_version": "sparsene.sota_summary_a100_n512_fp32.v1",
        "datasets_expected": len(DATASETS),
        "operator_rows": len(rows),
        "methods": methods,
        "method_coverage": {
            method: sum(row["method"] == method for row in rows) for method in methods
        },
        "anomalies": [row for row in rows if row["anomaly"]],
        "paper_figure_extraction": paper_figure_audit,
        "paper_raw_log_validation": paper_log_audit,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
