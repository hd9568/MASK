#!/usr/bin/env python3
"""Update the maintained N=512 FP32 cross-run incumbent registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparsene.utils.incumbent_registry import (
    refresh_registry_summary,
    update_registry_from_runs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = (
    REPO_ROOT / "logs" / "e2e_runs" / "incumbents" / "a100_a800_n512_fp32"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"maintained registry directory (default: {DEFAULT_REGISTRY})",
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        action="append",
        help="completed e2e run to ingest; repeat in oldest-to-newest order",
    )
    parser.add_argument(
        "--refresh-summary-only",
        action="store_true",
        help="republish summary.md/CSV/registry metadata without ingesting candidates",
    )
    parser.add_argument("--datasets", nargs="+", help="optional dataset subset")
    parser.add_argument(
        "--min-improvement-pct",
        type=float,
        default=0.0,
        help="minimum strict kernel-time improvement required for replacement",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.refresh_summary_only:
        if args.source_run:
            raise SystemExit("--refresh-summary-only cannot be combined with --source-run")
        registry = refresh_registry_summary(args.registry_dir)
        print(
            json.dumps(
                {
                    "registry": str(args.registry_dir.resolve()),
                    "datasets": registry.get("counts", {}).get("datasets"),
                    "historical_sparsene1_baseline": registry.get(
                        "historical_sparsene1_baseline"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not args.source_run:
        raise SystemExit("at least one --source-run is required")
    report = update_registry_from_runs(
        args.registry_dir,
        args.source_run,
        datasets=args.datasets,
        min_improvement_pct=args.min_improvement_pct,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"registry={args.registry_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
