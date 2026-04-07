#!/usr/bin/env python3
from __future__ import annotations

import sys

from cluster_plan_search_curve_simulator import main as merged_main


def _ensure_cluster_strategy_default() -> None:
    if "--strategy" not in sys.argv:
        sys.argv.extend(["--strategy", "cluster-search"])


if __name__ == "__main__":
    _ensure_cluster_strategy_default()
    merged_main()