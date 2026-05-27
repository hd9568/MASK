"""
Matrix feature extraction for format-design agent.

Reads a .mtx file, computes a comprehensive set of features that
describe the sparsity structure at global, row, column, and block
levels.  The output JSON is designed to be fed directly into the
format-design agent prompt.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Lightweight Matrix-Market reader (avoids scipy dependency).
# ---------------------------------------------------------------------------

def read_mtx_coo(path: Path) -> Tuple[int, int, List[Tuple[int, int, float]]]:
    """Read a coordinate-format .mtx file.

    Returns (num_rows, num_cols, list of (row, col, val)).
    Rows and cols are 0-based.
    """
    entries: List[Tuple[int, int, float]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    nrows = ncols = 0
    nnz_field = 0
    header_done = False

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("%"):
            continue

        if not header_done:
            parts = line.split()
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                continue
            if len(nums) >= 2:
                nrows = int(nums[0])
                ncols = int(nums[1])
                nnz_field = int(nums[2]) if len(nums) >= 3 else 0
                header_done = True
            continue

        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            r = int(parts[0]) - 1
            c = int(parts[1]) - 1
            v = float(parts[2]) if len(parts) >= 3 else 1.0
        except ValueError:
            continue
        if 0 <= r < nrows and 0 <= c < ncols:
            entries.append((r, c, v))

    return nrows, ncols, entries


# ---------------------------------------------------------------------------
# Feature dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DistributionStats:
    mean: float
    std: float
    min_val: float
    max_val: float
    p25: float
    p50: float
    p75: float
    cv: float  # coefficient of variation = std / mean (0 if mean==0)


def _distribution(values: List[int]) -> DistributionStats:
    if not values:
        return DistributionStats(0, 0, 0, 0, 0, 0, 0, 0)
    n = len(values)
    s = sorted(values)
    return DistributionStats(
        mean=sum(s) / n,
        std=math.sqrt(sum((x - sum(s) / n) ** 2 for x in s) / n),
        min_val=s[0],
        max_val=s[-1],
        p25=s[int(n * 0.25)],
        p50=s[int(n * 0.50)],
        p75=s[int(n * 0.75)],
        cv=(math.sqrt(sum((x - sum(s) / n) ** 2 for x in s) / n) / (sum(s) / n))
        if sum(s) > 0
        else 0.0,
    )


@dataclass
class BlockStats:
    """Stats for one (BLK_M, BLK_K) candidate."""

    blk_m: int
    blk_k: int
    num_row_windows: int
    num_col_windows: int
    total_tc_blocks: int  # number of non-empty BLK_M x BLK_K blocks
    tc_blocks_per_window: DistributionStats
    padding_ratio: float  # (total_cells - nnz) / total_cells across all TC blocks
    avg_density: float  # mean block fill ratio
    block_density_dist: DistributionStats  # distribution of per-block fill ratios


def _block_stats(
    entries: List[Tuple[int, int, float]],
    M: int,
    K: int,
    blk_m: int,
    blk_k: int,
) -> BlockStats:
    n_windows_m = (M + blk_m - 1) // blk_m
    n_windows_k = (K + blk_k - 1) // blk_k

    # Build per-window TC block count
    window_blocks: Dict[int, int] = defaultdict(int)
    total_tc = 0
    total_cells = 0
    block_densities: List[float] = []

    # Also track per-block nnz for density distribution
    block_nnz: Dict[Tuple[int, int], int] = defaultdict(int)

    for r, c, _v in entries:
        w_m = r // blk_m
        w_k = c // blk_k
        key = (w_m, w_k)
        block_nnz[key] += 1

    tc_per_window_vals: List[int] = []
    for w_m in range(n_windows_m):
        count = 0
        for w_k in range(n_windows_k):
            key = (w_m, w_k)
            nnz_b = block_nnz.get(key, 0)
            if nnz_b > 0:
                count += 1
                total_tc += 1
                total_cells += blk_m * blk_k
                block_densities.append(nnz_b / (blk_m * blk_k))
        window_blocks[w_m] = count
        tc_per_window_vals.append(count)

    nnz_total = len(entries)
    padding = (total_cells - nnz_total) / max(total_cells, 1)

    return BlockStats(
        blk_m=blk_m,
        blk_k=blk_k,
        num_row_windows=n_windows_m,
        num_col_windows=n_windows_k,
        total_tc_blocks=total_tc,
        tc_blocks_per_window=_distribution(tc_per_window_vals),
        padding_ratio=padding,
        avg_density=nnz_total / max(total_cells, 1),
        block_density_dist=_distribution([int(d * 10000) for d in block_densities])
        if block_densities
        else DistributionStats(0, 0, 0, 0, 0, 0, 0, 0),
    )


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------

@dataclass
class MatrixFeatures:
    path: str
    M: int
    K: int
    nnz: int
    sparsity: float
    symmetry: str  # "symmetric", "general", "skew"
    row_nnz: DistributionStats
    col_nnz: DistributionStats
    row_length_vs_index_corr: float  # Pearson correlation row_idx vs row_nnz
    block_stats: Dict[str, BlockStats]  # keyed by "BLK_MxBLK_K"
    # Load-balance indicators
    load_imbalance_cv: float  # CV of TC blocks per row window (at most-imbalanced BLK)
    top5_window_share: float  # fraction of total TC blocks in top-5 busiest windows


def _pearson_corr(xs: List[int], ys: List[int]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def featurize_matrix(
    mtx_path: Path,
    candidate_blk_pairs: Optional[List[Tuple[int, int]]] = None,
    detect_symmetry: bool = True,
) -> MatrixFeatures:
    """Extract comprehensive format-design-oriented features from a .mtx file.

    Parameters
    ----------
    mtx_path:
        Path to a Matrix-Market (.mtx) file in coordinate format.
    candidate_blk_pairs:
        List of (BLK_M, BLK_K) pairs to evaluate.  Defaults to
        [(8,8), (8,16), (16,8), (16,16), (32,8), (32,16)] if None.
    detect_symmetry:
        If True and header declares "symmetric", features use the upper
        triangle (the reader keys on banner line).
    """
    if candidate_blk_pairs is None:
        candidate_blk_pairs = [
            (8, 8),
            (8, 16),
            (16, 8),
            (16, 16),
            (32, 8),
            (32, 16),
            (64, 16),
        ]

    M, K, entries = read_mtx_coo(mtx_path)

    # Symmetry detection from banner
    symmetry = "general"
    with mtx_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("%%"):
                low = line.lower()
                if "symmetric" in low:
                    symmetry = "symmetric"
                elif "skew" in low:
                    symmetry = "skew"
                break

    nnz = len(entries)

    # Row / column NNZ distributions
    row_counts: Dict[int, int] = defaultdict(int)
    col_counts: Dict[int, int] = defaultdict(int)
    for r, c, _v in entries:
        row_counts[r] += 1
        col_counts[c] += 1

    row_nnz_list = [row_counts.get(r, 0) for r in range(M)]
    col_nnz_list = [col_counts.get(c, 0) for c in range(K)]

    # Correlation row_index vs row_nnz
    row_corr = _pearson_corr(list(range(M)), row_nnz_list)

    # Block stats for each candidate pair
    block_map: Dict[str, BlockStats] = {}
    max_cv = 0.0
    top5_share = 0.0
    for blk_m, blk_k in candidate_blk_pairs:
        bs = _block_stats(entries, M, K, blk_m, blk_k)
        key = f"{blk_m}x{blk_k}"
        block_map[key] = bs
        if bs.tc_blocks_per_window.cv > max_cv:
            max_cv = bs.tc_blocks_per_window.cv
            sorted_vals = sorted(
                [
                    count
                    for count in [
                        sum(
                            1
                            for wk in range(bs.num_col_windows)
                            if any(
                                (wm, wk) in set()
                                for _ in []
                            )
                        )
                    ]
                ]
            )
            # recalc top5 properly
            vals = []
            for wm in range(bs.num_row_windows):
                count = 0
                for wk in range(bs.num_col_windows):
                    key_pair = (wm, wk)
                    rr_start = wm * blk_m
                    cc_start = wk * blk_k
                    has_nnz = False
                    for r, c, _ in entries:
                        if rr_start <= r < rr_start + blk_m and cc_start <= c < cc_start + blk_k:
                            has_nnz = True
                            break
                    if has_nnz:
                        count += 1
                vals.append(count)
            vals.sort(reverse=True)
            total = sum(vals) or 1
            top5_share = sum(vals[:5]) / total

    return MatrixFeatures(
        path=str(mtx_path),
        M=M,
        K=K,
        nnz=nnz,
        sparsity=1.0 - nnz / (M * K) if M * K > 0 else 1.0,
        symmetry=symmetry,
        row_nnz=_distribution(row_nnz_list),
        col_nnz=_distribution(col_nnz_list),
        row_length_vs_index_corr=row_corr,
        block_stats=block_map,
        load_imbalance_cv=max_cv,
        top5_window_share=top5_share,
    )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _distribution_dict(d: DistributionStats) -> Dict[str, Any]:
    return {
        "mean": round(d.mean, 2),
        "std": round(d.std, 2),
        "min": d.min_val,
        "max": d.max_val,
        "p25": d.p25,
        "p50": d.p50,
        "p75": d.p75,
        "cv": round(d.cv, 4),
    }


def features_to_dict(feats: MatrixFeatures) -> Dict[str, Any]:
    return {
        "path": feats.path,
        "shape": {"M": feats.M, "K": feats.K, "nnz": feats.nnz},
        "sparsity": round(feats.sparsity, 4),
        "symmetry": feats.symmetry,
        "row_nnz_distribution": _distribution_dict(feats.row_nnz),
        "col_nnz_distribution": _distribution_dict(feats.col_nnz),
        "row_length_vs_index_correlation": round(feats.row_length_vs_index_corr, 4),
        "candidate_block_stats": {
            key: {
                "blk_m": bs.blk_m,
                "blk_k": bs.blk_k,
                "num_row_windows": bs.num_row_windows,
                "num_col_windows": bs.num_col_windows,
                "total_tc_blocks": bs.total_tc_blocks,
                "tc_blocks_per_window": _distribution_dict(bs.tc_blocks_per_window),
                "padding_ratio": round(bs.padding_ratio, 4),
                "avg_density": round(bs.avg_density, 4),
                "block_density_cv": round(bs.block_density_dist.cv, 4),
            }
            for key, bs in feats.block_stats.items()
        },
        "load_balance": {
            "tc_blocks_cv": round(feats.load_imbalance_cv, 4),
            "top5_window_share": round(feats.top5_window_share, 4),
        },
    }


def featurize_and_dump(
    mtx_path: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convenience: extract features and optionally save as JSON."""
    feats = featurize_matrix(mtx_path)
    payload = features_to_dict(feats)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mtx", type=str, help="Path to .mtx file")
    ap.add_argument("-o", "--output", type=str, default=None, help="Output JSON file")
    args = ap.parse_args()
    payload = featurize_and_dump(Path(args.mtx), Path(args.output) if args.output else None)
    if not args.output:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
