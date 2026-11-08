#!/usr/bin/env python3
"""Archive six-table best operators and organize deep-search curves by model/dataset.

Only reads existing benchmark artifacts. Existing global incumbents and flat
figures remain untouched; this builds reproducible convenience views.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "logs/e2e_runs"
INC = RUNS / "incumbents/a100_a800_n512_fp32"
CURVES = RUNS / "20261020_sparsekg_deep_joint6_f5_s10_five_models/analysis/deep_joint_curves_20261021_105002"
DS_RUN = RUNS / "20260722_232100_mask_deepseek_v4_pro_format26_v7_empirical_feedback"
CONTRACT = "sparsene.kernel_only_post_conversion.v1"
MODELS = {
    "ds_v4pro": ("deepseek_v4_pro", "sparsekg_ds_v4pro_speedup"),
    "gpt6_1_sol": ("gpt_6.1_sol", "sparsekg_gpt6_1_sol_speedup"),
    "opus5_5": ("claude_opus_5_5", "sparsekg_opus5_5_speedup"),
    "deepseek_flash": ("deepseek_flash", "sparsekg_deepseek_flash_speedup"),
    "gpt6_astra": ("gpt_6_astra", "sparsekg_gpt6_astra_speedup"),
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def by_dataset(path: Path) -> dict[str, dict[str, str]]:
    result = {row["dataset"]: row for row in rows(path)}
    if len(result) != len(rows(path)):
        raise ValueError(f"duplicate dataset in {path}")
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(raw: str) -> Path:
    path = Path(raw).resolve(strict=True)
    if not path.is_relative_to(RUNS.resolve()):
        raise ValueError(f"artifact outside e2e_runs: {path}")
    return path


def source_operator(model: str, dataset: str, six: dict, four: dict,
                    ds_sources: dict, ds_summaries: dict[str, dict]) -> tuple[Path, str]:
    if six[f"{model}_selection"] == "deep_joint":
        return checked_path(six[f"{model}_deep_operator_path"]), "deep_joint"
    if model == "ds_v4pro":
        source = ds_sources[dataset]["ds_source"]
        if source == "operator_e2e_summary.csv":
            manifest = DS_RUN / "datasets" / dataset / "operator_generation/generation_manifest.json"
            return checked_path(read_json(manifest)["operator_path"]), "single_format_baseline"
        return checked_path(ds_summaries[source][dataset]["best_operator_path"]), source
    if model in {"gpt6_1_sol", "opus5_5"}:
        field = "gpt_operator_path" if model == "gpt6_1_sol" else "opus_operator_path"
        return checked_path(four[dataset][field]), "single_format_audited"
    return checked_path(six[f"{model}_operator_path"]), six[f"{model}_source"]


def audit(model: str, dataset: str, speedup: float, operator: Path) -> tuple[dict, dict, Path, Path]:
    iteration_dir = operator.parent.parent
    iteration_path = iteration_dir / "iteration_summary.json"
    if iteration_path.is_file():
        iteration = read_json(iteration_path)
        result = iteration.get("e2e_result") or {}
        strategy = iteration["strategy"]
        manifest_path = checked_path(iteration["generation_manifest"])
        status_ok = iteration.get("status") == "validated"
        identity_ok = (iteration.get("dataset") == dataset and
                       checked_path(iteration["operator_path"]) == operator and
                       iteration.get("iteration") == int(iteration_dir.name.removeprefix("iteration_")))
        correctness = result.get("correctness") or {}
        correct_ok = correctness.get("passed") is True and correctness.get("all_measurements_passed") is True
        evidence_path = iteration_path
    else:
        if model != "ds_v4pro":
            raise ValueError(f"missing validated iteration: {operator}")
        manifest_path = operator.parent.parent / "operator_generation/generation_manifest.json"
        manifest = read_json(manifest_path)
        result_path = checked_path(manifest.get("e2e_result") or str(manifest_path.parent / "e2e_result.json"))
        result = read_json(result_path)
        strategy = manifest["strategy_decision"]
        # Early v7 manifests stopped at 'generated'; the separately audited
        # e2e_result and six-table source provide the numerical gate.
        status_ok = manifest.get("status") in {"validated", "generated"}
        identity_ok = manifest.get("dataset") == dataset and checked_path(manifest["operator_path"]) == operator
        correctness = result.get("correctness") or {}
        correct_ok = correctness.get("passed") is True
        evidence_path = result_path
    manifest = read_json(manifest_path)
    timing = result.get("timing") or {}
    measured = float(timing["speedup_vs_cusparse"])
    if not (status_ok and identity_ok and correct_ok and
            manifest.get("performance_contract") == CONTRACT and
            timing.get("contract") == CONTRACT and
            math.isfinite(measured) and measured > 0 and
            math.isclose(measured, speedup, rel_tol=1e-3, abs_tol=1e-3)):
        raise ValueError(f"source/table mismatch: {model}/{dataset}: {measured} vs {speedup}")
    if sha256(operator) != manifest.get("operator_sha256"):
        raise ValueError(f"operator SHA256 mismatch: {operator}")
    format_path = checked_path(manifest["format_source"]["artifact_path"])
    if sha256(format_path) != manifest["format_source"]["artifact_sha256"]:
        raise ValueError(f"format SHA256 mismatch: {format_path}")
    return manifest, strategy, format_path, evidence_path


def compact_strategy(strategy: dict) -> dict:
    keep = ("provider", "rationale", "tensor_placement", "pipeline", "compiler", "runtime_schedule")
    return {key: strategy[key] for key in keep if key in strategy}


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect_best(dry_run: bool) -> list[dict]:
    six = by_dataset(INC / "six_model_speedup_comparison_data.csv")
    four = by_dataset(INC / "four_model_speedup_comparison_data.csv")
    ds_sources = by_dataset(INC / "four_method_26dataset_speedup_data.csv")
    source_names = {row["ds_source"] for row in ds_sources.values()} - {"operator_e2e_summary.csv"}
    ds_summaries = {name: by_dataset(DS_RUN / name) for name in source_names}
    records = []
    for model, (folder, field) in MODELS.items():
        for dataset, six_row in six.items():
            speedup = float(six_row[field])
            operator, selection = source_operator(model, dataset, six_row, four, ds_sources, ds_summaries)
            manifest, strategy, format_path, evidence_path = audit(model, dataset, speedup, operator)
            source_evidence = read_json(evidence_path)
            result = source_evidence["e2e_result"] if evidence_path.name == "iteration_summary.json" else source_evidence
            manifest_path = checked_path(source_evidence["generation_manifest"]) if evidence_path.name == "iteration_summary.json" else operator.parent.parent / "operator_generation/generation_manifest.json"
            format_iterations = [part for part in operator.parts if part.startswith("format_") and part[7:].isdigit()]
            output = INC / "datasets/by_model" / folder / dataset
            record = {
                "model": folder, "dataset": dataset, "selection": selection,
                "speedup_vs_cusparse": speedup,
                "generated_ms": float(result["timing"]["generated_ms"]),
                "performance_contract": CONTRACT,
                "format_iteration": int(format_iterations[-1][7:]) if format_iterations else None,
                "strategy_iteration": int(source_evidence["iteration"]) if evidence_path.name == "iteration_summary.json" else 0,
                "strategy_provider": strategy.get("provider"),
                "strategy_signature": source_evidence.get("strategy_signature"),
                "correctness_evidence_path": str(evidence_path),
                "operator_source_path": str(operator), "operator_sha256": sha256(operator),
                "format_source_path": str(format_path), "format_sha256": sha256(format_path),
                "generation_manifest_path": str(manifest_path),
                "format_name": manifest.get("format_name"),
                "backend": (manifest.get("backend_plan") or {}).get("backend"),
                "backend_physicalization_applied": (manifest.get("backend_physicalization") or {}).get("applied", False),
            }
            records.append(record)
            if dry_run:
                continue
            output.mkdir(parents=True, exist_ok=True)
            shutil.copy2(operator, output / "best_operator.inc")
            shutil.copy2(format_path, output / "best_format.format.json")
            shutil.copy2(manifest_path, output / "generation_manifest.json")
            write_json(output / "best_strategy.json", compact_strategy(strategy))
            write_json(output / "verification.json", {
                "dataset": dataset, "performance_contract": CONTRACT,
                "timing": result["timing"], "correctness": result["correctness"],
                "source_evidence_path": str(evidence_path),
            })
            write_json(output / "backend_plan.json", manifest.get("backend_plan") or {})
            physical = manifest.get("backend_physicalization") or {}
            if physical.get("applied") and physical.get("physical_spec"):
                write_json(output / "physical_format.json", physical["physical_spec"])
            write_json(output / "selection.json", record)
    if not dry_run:
        root = INC / "datasets/by_model"
        with (root / "index.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return records


def organize_curves(dry_run: bool) -> tuple[int, int]:
    comparisons = rows(CURVES / "deep_vs_one_shot.csv")
    points = rows(CURVES / "curve_points.csv")
    for comparison in comparisons:
        model, dataset = comparison["model"], comparison["dataset"]
        if model not in {folder for folder, _ in MODELS.values()}:
            raise ValueError(f"unexpected curve model: {model}")
        own_points = [row for row in points if row["model"] == model and row["dataset"] == dataset]
        if len(own_points) != int(comparison["valid_operators"]):
            raise ValueError(f"curve point count mismatch: {model}/{dataset}")
        target = CURVES / "models" / model / dataset
        if dry_run:
            continue
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "comparison.json", comparison)
        with (target / "points.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(own_points)
        if comparison["plotted"] == "True":
            for ext in ("pdf", "svg", "png"):
                source = CURVES / f"{model}__{dataset}.{ext}"
                if not source.is_file():
                    raise ValueError(f"missing figure: {source}")
                shutil.copy2(source, target / f"curve.{ext}")
    return len(comparisons), sum(row["plotted"] == "True" for row in comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    records = collect_best(args.dry_run)
    pairs, figures = organize_curves(args.dry_run)
    print(f"Validated {len(records)} model/dataset best bundles across {len(MODELS)} models")
    print(f"Organized {pairs} curve cases, including {figures} figures")
    if args.dry_run:
        print("Dry run: no files written")
    else:
        print(f"Best bundles: {INC / 'datasets/by_model'}")
        print(f"Curve cases: {CURVES / 'models'}")


if __name__ == "__main__":
    main()
