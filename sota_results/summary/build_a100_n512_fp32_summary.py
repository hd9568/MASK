#!/usr/bin/env python3
"""Normalize the paper's A100 / N=512 / FP32 source artifacts.

The inputs intentionally remain immutable.  This script keeps source-specific
duplicates and unlabeled observations instead of silently selecting values that
the raw files do not identify.
"""

from __future__ import annotations

import csv
import json
import re
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
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
