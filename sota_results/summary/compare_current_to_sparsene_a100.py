#!/usr/bin/env python3
"""Select historical Sparsene bests and compare the current generated version.

The user explicitly permits an approximate single-card A100/A800 comparison.
Rows already marked anomalous by the source normalizer are excluded from the
audited historical best so that implausible sub-microsecond log values cannot
win selection silently.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SUMMARY_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUMMARY_DIR.parents[1]
HISTORICAL_INPUT = SUMMARY_DIR / "a100_n512_fp32_sparsene_historical_all.csv"
CURRENT_INPUT = (
    REPO_ROOT
    / "logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6"
    / "operator_e2e_summary.csv"
)
HISTORICAL_OUTPUT = SUMMARY_DIR / "sparsene_a100_n512.csv"
COMPARISON_OUTPUT = SUMMARY_DIR / "sparsene_current_vs_a100_n512.csv"
AUDIT_OUTPUT = SUMMARY_DIR / "sparsene_current_vs_a100_n512_audit.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_historical_best(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    by_dataset: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["N"]) != 512:
            raise RuntimeError(f"unexpected historical N: {row}")
        by_dataset[row["dataset"]].append(row)

    simplified: list[dict[str, Any]] = []
    selected: dict[str, dict[str, str]] = {}
    for dataset, candidates in by_dataset.items():
        valid = [row for row in candidates if not row["anomaly"]]
        if not valid:
            raise RuntimeError(f"{dataset}: no non-anomalous historical candidate")
        best = min(valid, key=lambda row: float(row["milliseconds"]))
        selected[dataset] = best
        simplified.append(
            {
                "dataset": dataset,
                "N": 512,
                "precision": "FP32/TF32-tensorcore-where-applicable",
                "best_format": best["method"],
                "best_ms": best["milliseconds"],
                "variants_considered": len(valid),
                "anomalies_excluded": len(candidates) - len(valid),
            }
        )
    return simplified, selected


def compare(
    current_rows: list[dict[str, str]],
    historical_best: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    if len(current_rows) != 26 or len(historical_best) != 26:
        raise RuntimeError(
            f"expected 26 datasets, got current={len(current_rows)} "
            f"historical={len(historical_best)}"
        )
    current_datasets = {row["dataset"] for row in current_rows}
    if current_datasets != set(historical_best):
        raise RuntimeError(
            f"dataset mismatch: current_only={sorted(current_datasets - set(historical_best))}, "
            f"historical_only={sorted(set(historical_best) - current_datasets)}"
        )

    result: list[dict[str, Any]] = []
    for current in current_rows:
        if int(current["N"]) != 512:
            raise RuntimeError(f"{current['dataset']}: current N is not 512")
        if current["correctness_passed"].lower() != "true":
            raise RuntimeError(
                f"{current['dataset']}: incorrect current result cannot be ranked"
            )
        historical = historical_best[current["dataset"]]
        historical_ms = float(historical["milliseconds"])
        current_ms = float(current["generated_ms"])
        speedup = historical_ms / current_ms
        result.append(
            {
                "dataset": current["dataset"],
                "historical_a100_best_format": historical["method"],
                "historical_a100_best_ms": historical["milliseconds"],
                "current_a800_execution": (
                    f"{current['packing_mode']}/"
                    f"logical_m{current['logical_blk_m']}/"
                    f"physical_{current['physical_blk_m']}x{current['blk_k']}"
                ),
                "current_a800_generated_ms": current["generated_ms"],
                "historical_over_current_speedup": f"{speedup:.9g}",
                "current_exceeds_historical_best": current_ms < historical_ms,
            }
        )
    return result


def main() -> None:
    historical_rows = read_csv(HISTORICAL_INPUT)
    current_rows = read_csv(CURRENT_INPUT)
    simplified, historical_best = select_historical_best(historical_rows)
    comparison = compare(current_rows, historical_best)
    write_csv(HISTORICAL_OUTPUT, simplified)
    write_csv(COMPARISON_OUTPUT, comparison)

    raw_best = {
        dataset: min(rows, key=lambda row: float(row["milliseconds"]))
        for dataset, rows in (
            (
                dataset,
                [row for row in historical_rows if row["dataset"] == dataset],
            )
            for dataset in historical_best
        )
    }
    current_by_dataset = {row["dataset"]: row for row in current_rows}
    audited_wins = [
        row["dataset"]
        for row in comparison
        if row["current_exceeds_historical_best"]
    ]
    raw_wins = [
        dataset
        for dataset, historical in raw_best.items()
        if float(current_by_dataset[dataset]["generated_ms"])
        < float(historical["milliseconds"])
    ]
    audited_speedups = [
        float(row["historical_over_current_speedup"]) for row in comparison
    ]
    within_ten_percent = [
        row["dataset"]
        for row in comparison
        if 0.9 <= float(row["historical_over_current_speedup"]) <= 1.1
    ]
    format_counts = Counter(row["best_format"] for row in simplified)
    audit = {
        "schema_version": "sparsene.current_vs_historical_a100_n512.v1",
        "comparison_assumption": (
            "A100 and A800 single-card timings are treated as approximately comparable "
            "at the user's request; no hardware normalization is applied."
        ),
        "ratio_direction": "historical_a100_best_ms/current_a800_generated_ms; >1 current wins",
        "historical_selection": (
            "minimum milliseconds among rows whose anomaly field is empty"
        ),
        "historical_variants": len(historical_rows),
        "historical_anomalies_excluded": sum(bool(row["anomaly"]) for row in historical_rows),
        "datasets": len(comparison),
        "current_faster_than_audited_historical_best_count": len(audited_wins),
        "current_faster_than_audited_historical_best_datasets": audited_wins,
        "geomean_historical_over_current_speedup": math.exp(
            sum(math.log(value) for value in audited_speedups) / len(audited_speedups)
        ),
        "median_historical_over_current_speedup": statistics.median(audited_speedups),
        "datasets_within_ten_percent": within_ten_percent,
        "historical_best_format_counts": dict(sorted(format_counts.items())),
        "current_faster_if_flagged_anomalies_are_included_count": len(raw_wins),
        "current_faster_if_flagged_anomalies_are_included_datasets": raw_wins,
        "historical_input": str(HISTORICAL_INPUT.relative_to(REPO_ROOT)),
        "current_input": str(CURRENT_INPUT.relative_to(REPO_ROOT)),
        "simplified_output": str(HISTORICAL_OUTPUT.relative_to(REPO_ROOT)),
        "comparison_output": str(COMPARISON_OUTPUT.relative_to(REPO_ROOT)),
    }
    AUDIT_OUTPUT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"current faster than audited historical best: {len(audited_wins)}/"
        f"{len(comparison)}"
    )


if __name__ == "__main__":
    main()
