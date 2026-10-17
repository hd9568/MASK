#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_curve(path: Path) -> List[Dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "curve" not in data:
        raise RuntimeError(f"Invalid report JSON: missing 'curve' in {path}")

    rows = data["curve"]
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Invalid report JSON: empty 'curve' in {path}")

    return rows


def extract_series(rows: List[Dict[str, float]], x_key: str, y_key: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []

    for i, row in enumerate(rows, start=1):
        if x_key not in row or y_key not in row:
            raise RuntimeError(
                f"Row {i} missing key in curve: required '{x_key}' and '{y_key}', got keys={list(row.keys())}"
            )
        xs.append(float(row[x_key]))
        ys.append(float(row[y_key]))

    return xs, ys


def write_compare_csv(
    path: Path,
    sim_rows: List[Dict[str, float]],
    nosim_rows: List[Dict[str, float]],
    x_key: str,
    y_key: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "idx", x_key, y_key])
        writer.writeheader()
        for i, row in enumerate(sim_rows, start=1):
            writer.writerow({"source": "sim", "idx": i, x_key: row[x_key], y_key: row[y_key]})
        for i, row in enumerate(nosim_rows, start=1):
            writer.writerow({"source": "no-sim", "idx": i, x_key: row[x_key], y_key: row[y_key]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sim vs no-sim search curves on one chart")
    parser.add_argument("--sim-json", type=Path, required=True, help="Path to sim strategy report JSON")
    parser.add_argument("--nosim-json", type=Path, required=True, help="Path to no-sim strategy report JSON")
    parser.add_argument("--output-plot", type=Path, default=Path("sim_vs_nosim_curve.pdf"))
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional merged CSV output")
    parser.add_argument("--sim-label", type=str, default="sim")
    parser.add_argument("--nosim-label", type=str, default="no-sim")
    parser.add_argument("--x-key", type=str, default="time_sec")
    parser.add_argument("--y-key", type=str, default="speedup_vs_best")
    parser.add_argument("--title", type=str, default="Sim vs No-Sim Search Curve")
    parser.add_argument(
        "--zero-start",
        action="store_true",
        help="Shift each line's x-axis so the first point starts from 0",
    )
    args = parser.parse_args()

    sim_rows = load_curve(args.sim_json)
    nosim_rows = load_curve(args.nosim_json)

    sim_x, sim_y = extract_series(sim_rows, args.x_key, args.y_key)
    nosim_x, nosim_y = extract_series(nosim_rows, args.x_key, args.y_key)

    if args.zero_start:
        sim_base = sim_x[0]
        nosim_base = nosim_x[0]
        sim_x = [x - sim_base for x in sim_x]
        nosim_x = [x - nosim_base for x in nosim_x]

    plt.figure(figsize=(9.0, 4.2))
    plt.plot(sim_x, sim_y, color="orange", linewidth=2.2, label=args.sim_label)
    plt.plot(nosim_x, nosim_y, color="blue", linewidth=2.2, label=args.nosim_label)
    plt.scatter(sim_x, sim_y, color="orange", s=100, marker='o')
    plt.scatter(nosim_x, nosim_y, color="blue", s=100, marker='x')

    plt.xlabel("Time (sec)", fontsize=24)
    plt.ylabel("Speedup vs best", fontsize=24)
    # plt.title(args.title, fontsize=14)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.grid(alpha=0.35, linestyle="--")
    plt.legend(fontsize=20)
    plt.tight_layout()

    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_plot)
    plt.close()

    if args.output_csv is not None:
        write_compare_csv(args.output_csv, sim_rows, nosim_rows, args.x_key, args.y_key)

    print(f"sim_json: {args.sim_json}")
    print(f"nosim_json: {args.nosim_json}")
    print(f"output_plot: {args.output_plot}")
    if args.output_csv is not None:
        print(f"output_csv: {args.output_csv}")


if __name__ == "__main__":
    main()
