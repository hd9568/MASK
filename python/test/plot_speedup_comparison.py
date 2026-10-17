#!/usr/bin/env python3
"""Generate a 26-subplot bar chart comparing cuSPARSE / Sparsene / Sparsene2.0 / FormatAgent speedups."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ── Data (S's cuSPARSE as common baseline for all bars) ────────────────────

datasets = [
    "ASIC_680k", "DD", "Ga41As41H72", "OVCAR-8H", "Si41Ge41H72",
    "Yeast", "YeastH", "cant", "conf5_4-8x8-10", "consph",
    "cop20k_A", "dc2", "ddi", "eu-2005", "mac_econ_fwd500",
    "mc2depi", "mip1", "pdb1HYS", "protein", "pwtk",
    "reddit", "rma10", "scircuit", "shipsec1", "web-BerkStan",
    "webbase-1M",
]

# H: historical best ms (from sparsene_a100_n512.csv)
H_ms = {
    "ASIC_680k": 4.843788, "DD": 1.266608, "Ga41As41H72": 5.762613,
    "OVCAR-8H": 6.02135, "Si41Ge41H72": 4.250852, "Yeast": 5.15845,
    "YeastH": 9.978033, "cant": 0.594184, "conf5_4-8x8-10": 0.412629,
    "consph": 0.829543, "cop20k_A": 1.020234, "dc2": 0.736334,
    "ddi": 0.470964, "eu-2005": 6.074183, "mac_econ_fwd500": 0.914652,
    "mc2depi": 2.078522, "mip1": 0.904587, "pdb1HYS": 0.394978,
    "protein": 28.654596, "pwtk": 1.321736, "reddit": 45.038296,
    "rma10": 0.327273, "scircuit": 0.789701, "shipsec1": 1.165046,
    "web-BerkStan": 3.086048, "webbase-1M": 3.42285,
}

# T: AI-TCF_MCO ms (from updated operator_e2e_summary.csv)
T_ms = {
    "ASIC_680k": 10.2432, "DD": 2.08978, "Ga41As41H72": 22.0874,
    "OVCAR-8H": 6.2039, "Si41Ge41H72": 17.9023, "Yeast": 4.65132,
    "YeastH": 12.414, "cant": 2.74217, "conf5_4-8x8-10": 0.578662,
    "consph": 0.8704, "cop20k_A": 1.14268, "dc2": 1.44968,
    "ddi": 0.551629, "eu-2005": 13.313, "mac_econ_fwd500": 1.90894,
    "mc2depi": 4.36439, "mip1": 2.69404, "pdb1HYS": 0.914227,
    "protein": 40.7493, "pwtk": 2.10227, "reddit": 91.0501,
    "rma10": 0.549171, "scircuit": 1.92451, "shipsec1": 1.7964,
    "web-BerkStan": 5.64879, "webbase-1M": 7.77646,
}

# S: Codex SR_BCRS ms (from sr_bcrs_16x8/operator_e2e_summary.csv)
S_ms = {
    "ASIC_680k": 4.90611, "DD": 1.22525, "Ga41As41H72": 5.64454,
    "OVCAR-8H": 6.1209, "Si41Ge41H72": 4.13638, "Yeast": 5.17779,
    "YeastH": 10.2175, "cant": 0.523552, "conf5_4-8x8-10": 0.368672,
    "consph": 0.732608, "cop20k_A": 0.967072, "dc2": 0.741088,
    "ddi": 0.455232, "eu-2005": 6.13302, "mac_econ_fwd500": 0.859104,
    "mc2depi": 2.03571, "mip1": 0.890176, "pdb1HYS": 0.352256,
    "protein": 28.7099, "pwtk": 1.196, "reddit": 48.8923,
    "rma10": 0.29536, "scircuit": 1.01187, "shipsec1": 1.04531,
    "web-BerkStan": 3.09168, "webbase-1M": 3.76646,
}

# S's cuSPARSE as common baseline
S_cusparse = {
    "ASIC_680k": 4.98131, "DD": 1.60854, "Ga41As41H72": 10.4969,
    "OVCAR-8H": 7.73696, "Si41Ge41H72": 8.10918, "Yeast": 7.17994,
    "YeastH": 12.8484, "cant": 1.87904, "conf5_4-8x8-10": 1.03459,
    "consph": 3.0904, "cop20k_A": 3.2601, "dc2": 0.793728,
    "ddi": 0.89392, "eu-2005": 9.65251, "mac_econ_fwd500": 1.09872,
    "mc2depi": 2.43731, "mip1": 4.41562, "pdb1HYS": 2.08387,
    "protein": 39.822, "pwtk": 5.67114, "reddit": 69.6087,
    "rma10": 1.18918, "scircuit": 0.983488, "shipsec1": 4.02758,
    "web-BerkStan": 4.28688, "webbase-1M": 4.29578,
}


def speedup(ms: float, dataset: str) -> float:
    """Compute speedup vs common cuSPARSE baseline (S's cuSPARSE)."""
    return S_cusparse[dataset] / ms


# ── Compute speedups ───────────────────────────────────────────────────────
cusparse_vals  = [1.0] * 26
sparsene_vals  = [speedup(H_ms[d], d) for d in datasets]
sparsene2_vals = [speedup(S_ms[d], d) for d in datasets]
fmtagent_vals  = [speedup(T_ms[d], d) for d in datasets]

# ── Plot setup ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 100,
})

nrows, ncols = 6, 5
fig, axes = plt.subplots(nrows, ncols, figsize=(28, 20))
fig.suptitle(
    "Sparse Format Speedup Comparison (vs cuSPARSE, N=512, FP32/TF32)",
    fontsize=16, fontweight="bold", y=0.985,
)

bar_width = 0.5
x = np.arange(4)
labels = ["cuSPARSE", "Sparsene", "Sparsene2.0", "FormatAgent"]
colors = ["#9ca3af", "#2563eb", "#059669", "#d97706"]

global_max = max(
    max(cusparse_vals),
    max(sparsene_vals),
    max(sparsene2_vals),
    max(fmtagent_vals),
)

for idx, dataset in enumerate(datasets):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]

    vals = [
        cusparse_vals[idx],
        sparsene_vals[idx],
        sparsene2_vals[idx],
        fmtagent_vals[idx],
    ]

    bars = ax.bar(x, vals, bar_width, color=colors, edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        if val > 1.05 * global_max:
            continue
        va = "bottom" if val >= 0 else "top"
        offset = max(global_max * 0.02, 0.08)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset if val >= 0 else bar.get_height() - offset,
            f"{val:.2f}x", ha="center", va=va, fontsize=6.2, fontweight="bold",
            color="#1f2937",
        )

    ax.set_title(dataset, fontsize=9.5, fontweight="bold", pad=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Speedup", fontsize=8)
    ax.axhline(y=1.0, color="#6b7280", linewidth=0.8, linestyle="--", alpha=0.6)

    # Dynamic y-limit per subplot
    local_max = max(vals) * 1.3
    ax.set_ylim(0, max(local_max, 1.5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Hide unused subplots
for idx in range(26, nrows * ncols):
    row, col = divmod(idx, ncols)
    axes[row][col].set_axis_off()

# ── Legend at bottom ────────────────────────────────────────────────────────
handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white", linewidth=0.5)
    for c in colors
]
fig.legend(
    handles, labels, loc="lower center", ncol=4, fontsize=11,
    frameon=True, fancybox=True, shadow=False,
    bbox_to_anchor=(0.5, -0.01),
)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# ── Save ────────────────────────────────────────────────────────────────────
output_path = Path("/home/huoda/sparsene/docs/trae/speedup_comparison_26datasets.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved to {output_path}")
plt.close(fig)
