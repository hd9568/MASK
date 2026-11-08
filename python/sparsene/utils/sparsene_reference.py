"""Audited, per-matrix historical Sparsene format references.

These are the fastest recorded variants in the A100 N=512 FP32 log, not
claims about which variant the paper authors selected for each figure bar.
The historical executable is a reference measurement, never a generated
SparseKG candidate or a substitute for full-output correctness validation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_CSV = REPO_ROOT / "sota_results/summary/sparsene_a100_n512.csv"
SUPPORTED_VARIANTS = {
    "SR-BCRS-16x8",
    "SR-BCRS-16x8-strict-lb",
    "SR-BCRS-16x8-multi-bind",
    "ACC-BIT-TCF",
}


@dataclass(frozen=True)
class SparseneReference:
    dataset: str
    variant: str
    milliseconds: float
    source_csv: str

    @property
    def atomic_type(self) -> str:
        return "mco" if self.variant == "ACC-BIT-TCF" else "dense"

    @property
    def execution_policy(self) -> str:
        if self.variant.endswith("strict-lb"):
            return "strict_lb"
        if self.variant.endswith("multi-bind"):
            return "multi_bind"
        # The plain SR-BCRS and ACC executables use their row-bound host
        # program; auto can silently choose strict_lb in the new host bridge.
        return "row"

    def to_dict(self) -> dict[str, str | float]:
        return {
            "dataset": self.dataset,
            "variant": self.variant,
            "historical_ms": self.milliseconds,
            "source_csv": self.source_csv,
            "provenance": "historical_hand_tuned_reference_not_generated",
            "atomic_type": self.atomic_type,
            "initial_execution_policy": self.execution_policy,
        }


def load_sparsene_reference(
    dataset: str, csv_path: Path = DEFAULT_REFERENCE_CSV
) -> SparseneReference:
    """Fail closed on missing, ambiguous, or unsupported historical rows."""
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["dataset"] == dataset]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one Sparsene reference for {dataset}: {csv_path}")
    row = rows[0]
    if row["N"] != "512" or not row["precision"].startswith("FP32"):
        raise ValueError(f"incompatible Sparsene reference protocol for {dataset}: {row}")
    if not 0 <= int(row["anomalies_excluded"]) <= 9:
        raise ValueError(f"unexpected Sparsene anomaly count for {dataset}: {row}")
    variant = row["best_format"].removeprefix("Sparsene/")
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"no exact format0 mapping for historical variant {variant!r}")
    milliseconds = float(row["best_ms"])
    if not 0 < milliseconds < float("inf"):
        raise ValueError(f"invalid Sparsene reference time for {dataset}: {milliseconds}")
    return SparseneReference(dataset, variant, milliseconds, str(csv_path.resolve()))
