#!/usr/bin/env python3
"""Remove reproducible build products while preserving experiment evidence.

The generated sources, strategy/format requests and responses, compile/run
commands and logs, correctness/performance JSON, NCU summaries, and figures are
kept. CMake/NVCC build trees are disposable and can be recreated from the
retained commands and generated sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--apply", action="store_true", help="delete disposable build trees")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    dataset_root = run_dir / "datasets"
    targets = sorted(
        path for path in dataset_root.glob("*/**/build")
        if path.is_dir() and path.name == "build"
    )
    records = [
        {"path": str(path.relative_to(run_dir)), "bytes": _size(path), "kind": "cmake_nvcc_build_tree"}
        for path in targets
    ]
    total = sum(item["bytes"] for item in records)
    manifest = {
        "schema_version": "sparsene.e2e_cleanup.v1",
        "run_dir": str(run_dir),
        "applied": bool(args.apply),
        "removed": records if args.apply else [],
        "planned": records if not args.apply else [],
        "removed_bytes": total if args.apply else 0,
        "preserved": [
            "format artifacts and matrix_features.json",
            "generated operators and IR",
            "strategy decisions, agent request/response/audit/usage logs",
            "compile/configure/run command and text logs",
            "e2e_result.json, iteration_summary.json, summary CSV/JSON",
            "NCU CSV/summary/command and trajectory/comparison figures",
        ],
    }
    if args.apply:
        for path in targets:
            shutil.rmtree(path)
        manifest_path = run_dir / "cleanup_manifest.json"
    else:
        manifest_path = run_dir / "cleanup_plan.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"targets": len(targets), "bytes": total, "manifest": str(manifest_path), "applied": args.apply}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
