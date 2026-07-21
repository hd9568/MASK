#!/usr/bin/env python3
"""Generate sparse matrix heatmaps for all MTX files.

Usage:
    source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
    python python/test/gen_mtx_heatmaps.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.io import mmread

MTX_DIR = Path("/mnt/lustre-client/zhangkaige/selected_mtx")
OUT_DIR = Path.cwd() / "mtx_fig"


def fmt_human(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    if n >= 1_000:
        return f"{n/1e3:.1f}K"
    return str(n)


def plot_one(mtx_path: Path, out_path: Path) -> None:
    import scipy.sparse
    name = mtx_path.stem

    # --- read matrix ---
    M = scipy.sparse.coo_matrix(mmread(str(mtx_path)))
    rows, cols = M.row, M.col
    nnz = len(rows)
    density = 100.0 * nnz / (M.shape[0] * M.shape[1]) if M.shape[0] * M.shape[1] > 0 else 0

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7, 6.5))

    # --- scatter / spy ---
    # For very large matrices, sample a fraction of points
    max_pts = 1_000_000
    if nnz > max_pts:
        idx = np.random.choice(nnz, max_pts, replace=False)
        rr, cc = rows[idx], cols[idx]
    else:
        rr, cc = rows, cols

    # Clean blue scatter on white background
    ax.scatter(
        cc, rr,
        s=max(0.015, 40.0 / max(nnz**0.35, 1)),
        c="#2563eb", alpha=0.35, edgecolors="none", rasterized=True,
    )

    # --- styling ---
    m, k = M.shape[0], M.shape[1]
    ax.set_xlim(-m * 0.005, k * 1.005)
    ax.set_ylim(m * 1.005, -m * 0.005)

    # Ticks
    ax.tick_params(colors="#555555", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.5)

    # Axis labels
    ax.set_xlabel(f"Columns  (K = {fmt_human(k)})", color="#444444", fontsize=10, labelpad=6)
    ax.set_ylabel(f"Rows  (M = {fmt_human(m)})", color="#444444", fontsize=10, labelpad=6)

    # Title
    title = f"{name}"
    subtitle = f"M = {fmt_human(m)}    K = {fmt_human(k)}    NNZ = {fmt_human(nnz)}    density = {density:.3f}%"
    ax.set_title(title, color="#222222", fontsize=13, fontweight="bold", pad=2)
    fig.text(0.5, 0.925, subtitle, ha="center", color="#888888", fontsize=9,
             fontfamily="monospace", transform=fig.transFigure)

    # Diagonal reference line for square matrices
    if abs(m - k) / max(m, k) < 0.1:
        ax.plot([0, min(m, k)], [0, min(m, k)], color="#00000018", linewidth=0.3,
                linestyle="--")

    plt.tight_layout(pad=2)
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path.name}  ({nnz} nnz)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mtx_files = sorted(MTX_DIR.glob("*.mtx"))
    print(f"Found {len(mtx_files)} MTX files\n")

    for f in mtx_files:
        out = OUT_DIR / f"{f.stem}.png"
        try:
            plot_one(f, out)
        except Exception as e:
            print(f"  ✗  {f.name}  ERROR: {e}")

    print(f"\nDone → {OUT_DIR}/")
    print(f"Total: {len(list(OUT_DIR.glob('*.png')))} heatmaps")


if __name__ == "__main__":
    main()
