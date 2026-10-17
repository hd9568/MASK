#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from statistics import mean


METHOD_FILES = {
    "dtc_base": "gcn_e2e_dtc_base_26_n128.csv",
    "dtc_multibind": "gcn_e2e_dtc_multibind_26_n128.csv",
    "dtc_strict_lb": "gcn_e2e_dtc_strict_lb_26_n128.csv",
    "srbcrs_base": "gcn_e2e_srbcrs_base_26_n128.csv",
    "srbcrs_16x8": "gcn_e2e_srbcrs_16x8_26_n128.csv",
    "flashsparse": "gcn_e2e_flashsparse_26_n128.csv",
}


def read_metric_by_dataset(csv_path: Path, metric: str) -> dict:
    values = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if metric not in reader.fieldnames:
            raise ValueError(f"metric '{metric}' not found in {csv_path}")
        for row in reader:
            ds = row.get("dataset", "").strip()
            if not ds:
                continue
            try:
                v = float(row[metric])
            except Exception:
                continue
            if v <= 0:
                continue
            values[ds] = v
    return values


def geometric_mean(xs):
    xs = [x for x in xs if x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    parser = argparse.ArgumentParser(
        description="Compare Sparsene methods vs FlashSparse speedup (speedup = flash_time / method_time)."
    )
    parser.add_argument(
        "--result-dir",
        default="/workspace/sparsene/end2end/result",
        help="Directory containing gcn_e2e_*_26_n128.csv files.",
    )
    parser.add_argument(
        "--metric",
        default="train_time_sec",
        help="Metric column to compare, e.g. train_time_sec or time_per_epoch_ms.",
    )
    parser.add_argument(
        "--out-detail",
        default="speedup_vs_flashsparse_per_dataset_n128.csv",
        help="Output CSV filename for per-dataset speedups.",
    )
    parser.add_argument(
        "--out-summary",
        default="speedup_vs_flashsparse_summary_n128.csv",
        help="Output CSV filename for per-method summary.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    metric = args.metric

    tables = {}
    for method, name in METHOD_FILES.items():
        path = result_dir / name
        if not path.exists():
            raise FileNotFoundError(f"missing file for {method}: {path}")
        tables[method] = read_metric_by_dataset(path, metric)

    common = set(tables["flashsparse"].keys())
    for method in METHOD_FILES:
        common &= set(tables[method].keys())
    datasets = sorted(common)
    if not datasets:
        raise RuntimeError("no common datasets found across all method csv files")

    methods = [m for m in METHOD_FILES.keys() if m != "flashsparse"]
    detail_rows = []
    speedups_by_method = {m: [] for m in methods}

    for ds in datasets:
        flash = tables["flashsparse"][ds]
        row = {
            "dataset": ds,
            f"flashsparse_{metric}": flash,
        }
        best_method = None
        best_time = None
        for m in methods:
            t = tables[m][ds]
            s = flash / t
            row[f"{m}_{metric}"] = t
            row[f"{m}_speedup_x"] = s
            speedups_by_method[m].append(s)
            if best_time is None or t < best_time:
                best_time = t
                best_method = m
        row["best_sparsene_method"] = best_method
        row[f"best_sparsene_{metric}"] = best_time
        row["best_sparsene_speedup_x"] = flash / best_time
        detail_rows.append(row)

    detail_out = result_dir / args.out_detail
    detail_fields = [
        "dataset",
        f"flashsparse_{metric}",
    ]
    for m in methods:
        detail_fields.append(f"{m}_{metric}")
        detail_fields.append(f"{m}_speedup_x")
    detail_fields.extend([
        "best_sparsene_method",
        f"best_sparsene_{metric}",
        "best_sparsene_speedup_x",
    ])

    with detail_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)

    summary_out = result_dir / args.out_summary
    summary_fields = ["method", "datasets", "geo_mean_speedup_x", "mean_speedup_x", "wins_over_flash_count"]
    with summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for m in methods:
            vals = speedups_by_method[m]
            writer.writerow(
                {
                    "method": m,
                    "datasets": len(vals),
                    "geo_mean_speedup_x": geometric_mean(vals),
                    "mean_speedup_x": mean(vals) if vals else float("nan"),
                    "wins_over_flash_count": sum(1 for x in vals if x > 1.0),
                }
            )

        best_vals = [r["best_sparsene_speedup_x"] for r in detail_rows]
        writer.writerow(
            {
                "method": "best_sparsene_of_5",
                "datasets": len(best_vals),
                "geo_mean_speedup_x": geometric_mean(best_vals),
                "mean_speedup_x": mean(best_vals) if best_vals else float("nan"),
                "wins_over_flash_count": sum(1 for x in best_vals if x > 1.0),
            }
        )

    print(f"[OK] compared on {len(datasets)} common datasets")
    print(f"[OK] detail:  {detail_out}")
    print(f"[OK] summary: {summary_out}")


if __name__ == "__main__":
    main()
