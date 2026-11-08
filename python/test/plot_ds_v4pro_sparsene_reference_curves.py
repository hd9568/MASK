#!/usr/bin/env python3
"""Rebase the finished DS V4 Pro curves on each matrix's historical best Sparsene.

Reuse only validated old generated points.  The old webbase-1M format0 is
SR-BCRS, so it is excluded; an optional new ACC-format0 run supplies its
replacement.  No historical reference point is treated as a generated kernel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
OLD_RUN = ROOT / "logs/e2e_runs/20261107_ds_v4pro_joint_all26_f3_s5_backendgate_v1"
OLD_CURVES = OLD_RUN / "analysis/format_strategy_curves_20261108_earliest_best/curve_points.csv"
ACC_RUN = ROOT / "logs/e2e_runs/20261108_webbase_acc_format0_s5"
REFERENCE = ROOT / "sota_results/summary/paper_a100_n512_fp32_sparsene_validation.csv"
OLD_COMPARISON = ROOT / "logs/e2e_runs/incumbents/a100_a800_n512_fp32/sparsene_sparsekg_ds_v4pro_speedup_comparison_20261108.csv"
CONTRACT = "sparsene.kernel_only_post_conversion.v1"
COLORS = ("#176F8A", "#D17A31", "#6671A9")
MARKERS = ("o", "s", "D")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_old_point(row: dict[str, str]) -> dict:
    path = Path(row["iteration_summary_path"]).resolve(strict=True)
    if not path.is_relative_to(OLD_RUN.resolve()):
        raise ValueError(f"old iteration outside source run: {path}")
    item = read_json(path)
    result = item.get("e2e_result") or {}
    timing = result.get("timing") or {}
    correctness = result.get("correctness") or {}
    op = Path(row["operator_path"]).resolve(strict=True)
    if (item.get("status") != "validated" or correctness.get("passed") is not True
            or correctness.get("all_measurements_passed") is not True
            or timing.get("contract") != CONTRACT or timing.get("measurement_runs") != 3
            or timing.get("warmup") != 3 or timing.get("repeat") != 32
            or digest(op) != row["operator_sha256"]):
        raise ValueError(f"old candidate failed provenance/correctness audit: {path}")
    generated = float(timing["generated_ms"])
    cusparse = float(timing["cusparse_ms"])
    if not (math.isfinite(generated) and generated > 0
            and math.isclose(generated, float(row["generated_ms"]), rel_tol=1e-6)
            and math.isclose(cusparse / generated, float(row["speedup_vs_cusparse"]), rel_tol=1e-5)):
        raise ValueError(f"old timing mismatch: {path}")
    return {
        "dataset": row["dataset"], "format_iteration": int(row["format_iteration"]),
        "internal_strategy_iteration": int(row["strategy_iteration"]),
        "generated_ms": generated, "cusparse_ms": cusparse,
        "speedup_vs_cusparse": cusparse / generated,
        "operator_path": str(op), "operator_sha256": row["operator_sha256"],
        "iteration_summary_path": str(path), "source_run": OLD_RUN.name,
        "point_type": "generated_kernel", "status": "validated",
    }


def checked_acc_points(run_dir: Path) -> list[dict]:
    base = run_dir / "datasets/webbase-1M"
    if not base.is_dir():
        return []
    spec_path = base / "format_iterations/format_00/formats/webbase-1M.format.json"
    if not spec_path.is_file():
        return []
    spec = read_json(spec_path)
    reference = spec.get("historical_reference") or {}
    if (reference.get("variant") != "ACC-BIT-TCF"
            or (spec.get("atomic") or {}).get("type") != "mco"):
        raise ValueError("new webbase format0 is not the intended ACC/MCO reference format")
    points = []
    tuning = base / "format_iterations/format_00/datasets/webbase-1M/strategy_tuning"
    for path in sorted(tuning.glob("iteration_*/iteration_summary.json")):
        item = read_json(path)
        if item.get("status") != "validated":
            continue
        result = item.get("e2e_result") or {}
        timing = result.get("timing") or {}
        correctness = result.get("correctness") or {}
        if (item.get("dataset") != "webbase-1M" or correctness.get("passed") is not True
                or correctness.get("all_measurements_passed") is not True
                or timing.get("contract") != CONTRACT or timing.get("warmup") != 3
                or timing.get("repeat") != 32 or timing.get("measurement_runs") != 3):
            raise ValueError(f"new ACC point failed protocol audit: {path}")
        op = Path(item["operator_path"]).resolve(strict=True)
        manifest = read_json(Path(item["generation_manifest"]).resolve(strict=True))
        sha = digest(op)
        if not op.is_relative_to(run_dir.resolve()) or manifest.get("operator_sha256") != sha:
            raise ValueError(f"new ACC operator digest mismatch: {path}")
        generated, cusparse = float(timing["generated_ms"]), float(timing["cusparse_ms"])
        if not (math.isfinite(generated) and generated > 0 and math.isfinite(cusparse) and cusparse > 0):
            raise ValueError(f"invalid new ACC timing: {path}")
        points.append({
            "dataset": "webbase-1M", "format_iteration": 0,
            "internal_strategy_iteration": int(item["iteration"]),
            "generated_ms": generated, "cusparse_ms": cusparse,
            "speedup_vs_cusparse": cusparse / generated,
            "operator_path": str(op), "operator_sha256": sha,
            "iteration_summary_path": str(path.resolve()), "source_run": run_dir.name,
            "point_type": "generated_kernel", "status": "validated",
        })
    return points


def earliest_near_best(points: list[dict]) -> dict:
    fastest = min(float(point["generated_ms"]) for point in points)
    for point in sorted(points, key=lambda p: (p["format_iteration"], p["internal_strategy_iteration"])):
        if point["generated_ms"] <= fastest * 1.015:
            return point
    raise AssertionError("no earliest near-best candidate")


def draw(dataset: str, points: list[dict], reference: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    fig.subplots_adjust(left=0.10, right=0.982, top=0.74, bottom=0.20)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#707783")
    ax.grid(axis="y", color="#DDE2E8", lw=0.75, alpha=0.82)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.axhline(1.0, color="#555E6C", lw=1.1, ls=(0, (5, 3)), zorder=1)
    ax.axvspan(7.0, 14.0, color="#F2F5F8", zorder=-1)
    for boundary in (7.0, 14.0):
        ax.axvline(boundary, color="#98A1AD", lw=0.9, ls=(0, (3, 4)), zorder=2)
    ax.scatter([0], [1.0], marker="D", s=73, color="#4D5664", edgecolor="white", zorder=6)
    ticks, labels = [0], ["Ref"]
    for fmt in range(3):
        segment = [p for p in points if p["format_iteration"] == fmt]
        segment.sort(key=lambda p: p["internal_strategy_iteration"])
        xs = [(1 if fmt == 0 else 8 if fmt == 1 else 15) + i for i in range(len(segment))]
        ys = [p["speedup_vs_sparsene"] for p in segment]
        if segment:
            ax.plot(xs, ys, color=COLORS[fmt], lw=1.9, marker=MARKERS[fmt],
                    markersize=5.0, markeredgecolor="white", markeredgewidth=0.7,
                    solid_capstyle="round", zorder=3)
            if fmt == 0:
                ax.plot([0, xs[0]], [1.0, ys[0]], color="#85909D", lw=1.15,
                        ls=(0, (2, 3)), zorder=2)
            for x, point in zip(xs, segment):
                point["plot_x"] = x
                ticks.append(x)
                labels.append(f"G{point['internal_strategy_iteration']}")
                if point["internal_strategy_iteration"] == 0:
                    ax.scatter([x], [point["speedup_vs_sparsene"]], s=65, marker="o",
                               facecolors="white", edgecolors=COLORS[fmt], linewidths=1.7, zorder=5)
        else:
            ax.text((1 if fmt == 0 else 8 if fmt == 1 else 15) + 2.5, 0.48,
                    "ACC rerun pending" if fmt == 0 else "No validated operator",
                    transform=ax.get_xaxis_transform(), ha="center", va="center",
                    color="#8C95A2", fontsize=9)
        ax.text((0 if fmt == 0 else 8 if fmt == 1 else 15) + 2.7, 1.05,
                f"Format {fmt}  ·  {len(segment)} generated valid",
                transform=ax.get_xaxis_transform(), ha="center", color=COLORS[fmt],
                fontsize=9.2, weight="semibold", clip_on=False)
    if points:
        marker = earliest_near_best(points)
        ax.scatter([marker["plot_x"]], [marker["speedup_vs_sparsene"]], marker="*",
                   s=170, facecolor="#E4B94E", edgecolor="#524625", linewidths=0.8, zorder=7)
        ax.annotate(f"{marker['speedup_vs_sparsene']:.2f}×",
                    (marker["plot_x"], marker["speedup_vs_sparsene"]), xytext=(7, 8),
                    textcoords="offset points", fontsize=9, weight="semibold", color="#303943")
    values = [1.0] + [p["speedup_vs_sparsene"] for p in points]
    low, high = min(values), max(values)
    span = max(high - low, 0.16)
    ax.set_ylim(max(0, low - 0.14 * span), high + 0.25 * span)
    ax.set_xlim(-0.7, 20.8)
    ax.set_xticks(ticks, labels)
    ax.tick_params(axis="x", length=3, color="#8C95A2")
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("Speedup over historical-best Sparsene (×)")
    ax.set_xlabel("Ref = historical format0/iteration0; G0 = generated heuristic, G1–G5 = Agent")
    ax.set_title(f"DeepSeek V4 Pro  /  {dataset}", loc="left", pad=44,
                 weight="semibold", color="#202B37")
    fig.text(0.10, 0.045,
             f"Reference: {reference['historical_fastest_variant']}  ·  A100 {float(reference['historical_fastest_ms']):.3f} ms"
             "  ·  A800 generated points  ·  star = earliest within 1.5% of fastest",
             fontsize=8.1, color="#5E6874")
    return fig


def write_csv(path: Path, data: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def write_latex(path: Path, data: list[dict]) -> None:
    lines = [
        "% Historical fastest-of-nine Sparsene reference; not the paper-published bar values.",
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{SpMM speedup over cuSPARSE at $N=512$, FP32. Sparsene uses the per-matrix historical fastest A100 variant; SparseKG uses the best validated DeepSeek V4 Pro A800 kernel. The cross-GPU ratios are approximate.}",
        "\\label{tab:ds-v4pro-historical-sparsene-reference}",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "Matrix & Sparsene & SparseKG (DS V4 Pro) & SparseKG / Sparsene " + "\\" * 2,
        "\\midrule",
    ]
    for row in data:
        ref = float(row["historical_sparsene_speedup_vs_cusparse"])
        ds = float(row["ds_v4pro_speedup_vs_cusparse"])
        ratio = ds / ref
        ref_text = f"{ref:.3f}"
        ds_text = f"{ds:.3f}"
        if ds > ref:
            ds_text = f"\\textbf{{{ds_text}}}"
        else:
            ref_text = f"\\textbf{{{ref_text}}}"
        escaped_dataset = row["dataset"].replace("_", "\\_")
        lines.append(f"{escaped_dataset} & {ref_text} & {ds_text} & {ratio:.3f} " + "\\" * 2)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acc-run-dir", type=Path, default=ACC_RUN)
    parser.add_argument("--out-dir", type=Path,
                        default=OLD_RUN / "analysis/format_strategy_curves_20261108_historical_best_reference")
    parser.add_argument("--table-stem", type=Path,
                        default=OLD_COMPARISON.with_name("sparsene_sparsekg_ds_v4pro_historical_best_reference_20261108"))
    parser.add_argument("--refresh", action="store_true",
                        help="Refresh only this script's existing analysis outputs after the ACC rerun.")
    args = parser.parse_args()
    output = args.out_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.refresh:
        raise SystemExit(f"output already exists; use a new --out-dir: {output}")
    if (args.table_stem.with_suffix(".csv").exists() or args.table_stem.with_suffix(".tex").exists()) and not args.refresh:
        raise SystemExit("table output exists; use a new --table-stem")
    references = {row["dataset"]: row for row in rows(REFERENCE)}
    comparison = {row["dataset"]: row for row in rows(OLD_COMPARISON)}
    old = [checked_old_point(row) for row in rows(OLD_CURVES)]
    if len(old) != 370 or len(references) != 26 or set(references) != set(comparison):
        raise ValueError("expected audited 370 points and the same 26 reference/comparison datasets")
    reused = [p for p in old if not (p["dataset"] == "webbase-1M" and p["format_iteration"] == 0)]
    acc = checked_acc_points(args.acc_run_dir.resolve())
    points = reused + acc
    by_dataset: dict[str, list[dict]] = {dataset: [] for dataset in references}
    for point in points:
        ref = references[point["dataset"]]
        baseline = float(ref["historical_fastest_speedup_vs_cusparse"])
        if not baseline > 0:
            raise ValueError(f"invalid historical baseline for {point['dataset']}")
        point["historical_sparsene_variant"] = ref["historical_fastest_variant"]
        point["historical_sparsene_speedup_vs_cusparse"] = baseline
        point["speedup_vs_sparsene"] = point["speedup_vs_cusparse"] / baseline
        by_dataset[point["dataset"]].append(point)
    if not all(by_dataset.values()):
        raise ValueError("at least one dataset has no validated generated operator")
    for dataset, ref in references.items():
        if not math.isclose(float(ref["historical_fastest_ms"]),
                            float(ref["direct_log_cusparse_ms"]) / float(ref["historical_fastest_speedup_vs_cusparse"]),
                            rel_tol=1e-6):
            raise ValueError(f"historical reference inconsistency: {dataset}")
    output.mkdir(parents=True, exist_ok=True)
    args.table_stem.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 14, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.linewidth": 0.8})
    coverage = []
    with PdfPages(output / "all26_format_strategy_trajectories.pdf") as book:
        for dataset in comparison:
            segment = by_dataset[dataset]
            figure = draw(dataset, segment, references[dataset])
            book.savefig(figure, facecolor="white")
            stem = output / f"deepseek_v4_pro__{dataset}"
            for suffix in ("pdf", "svg", "png"):
                figure.savefig(f"{stem}.{suffix}", dpi=220 if suffix == "png" else None,
                               bbox_inches="tight", pad_inches=0.16)
            plt.close(figure)
            best = min(segment, key=lambda p: p["generated_ms"])
            coverage.append({
                "dataset": dataset,
                "historical_sparsene_variant": references[dataset]["historical_fastest_variant"],
                "historical_sparsene_ms": float(references[dataset]["historical_fastest_ms"]),
                "historical_sparsene_speedup_vs_cusparse": float(references[dataset]["historical_fastest_speedup_vs_cusparse"]),
                "reused_generated_points": len([p for p in segment if p["source_run"] == OLD_RUN.name]),
                "new_acc_generated_points": len([p for p in segment if p["source_run"] != OLD_RUN.name]),
                "format0_status": "ACC_rerun_pending" if dataset == "webbase-1M" and not acc else "available",
                "best_this_campaign_format_iteration": best["format_iteration"],
                "best_this_campaign_strategy_iteration": best["internal_strategy_iteration"],
                "best_this_campaign_ms": best["generated_ms"],
                "best_this_campaign_vs_sparsene": best["speedup_vs_sparsene"],
                "earliest_near_best_format_iteration": earliest_near_best(segment)["format_iteration"],
                "earliest_near_best_strategy_iteration": earliest_near_best(segment)["internal_strategy_iteration"],
            })
    write_csv(output / "curve_points.csv", points)
    write_csv(output / "coverage.csv", coverage)
    table = []
    for dataset, old_row in comparison.items():
        ref = references[dataset]
        old_ms = float(old_row["ds_v4pro_generated_ms"])
        new = min(acc, key=lambda p: p["generated_ms"]) if dataset == "webbase-1M" and acc else None
        use_new = new is not None and new["generated_ms"] < old_ms
        ds_speedup = new["speedup_vs_cusparse"] if use_new else float(old_row["ds_v4pro_speedup_vs_cusparse"])
        ds_ms = new["generated_ms"] if use_new else old_ms
        baseline = float(ref["historical_fastest_speedup_vs_cusparse"])
        table.append({
            "dataset": dataset,
            "historical_sparsene_variant": ref["historical_fastest_variant"],
            "historical_sparsene_ms": ref["historical_fastest_ms"],
            "historical_sparsene_speedup_vs_cusparse": baseline,
            "ds_v4pro_generated_ms": ds_ms,
            "ds_v4pro_speedup_vs_cusparse": ds_speedup,
            "ds_v4pro_vs_historical_sparsene": ds_speedup / baseline,
            "ds_source": args.acc_run_dir.name if use_new else old_row["selection_source"],
            "operator_source_path": new["operator_path"] if use_new else old_row["operator_source_path"],
            "operator_sha256": new["operator_sha256"] if use_new else old_row["operator_sha256"],
            "correctness_evidence_path": new["iteration_summary_path"] if use_new else old_row["correctness_evidence_path"],
            "old_paper_sparsene_speedup_vs_cusparse": old_row["sparsene_speedup_vs_cusparse"],
            "reference_provenance": "historical_fastest_of_nine_A100_not_paper_bar",
        })
    write_csv(args.table_stem.with_suffix(".csv"), table)
    write_latex(args.table_stem.with_suffix(".tex"), table)
    wins = sum(float(row["ds_v4pro_vs_historical_sparsene"]) > 1 for row in table)
    (output / "README.md").write_text(
        "# DeepSeek V4 Pro：逐数据集历史最快 Sparsene 参考轨迹\n\n"
        f"复用旧运行 {len(reused)} 个已验证生成算子点；webbase-1M 的旧 SR-BCRS format0 六点排除，"
        f"新 ACC format0 已验证点 {len(acc)} 个。其余旧 format1/2 点保留。\n\n"
        "图中 Ref 是历史 A100 Sparsene 最快变体，恒为 1.0×，并非本轮生成或完成当前全量正确性门的算子。"
        "G0 是当前 A800 生成的启发式策略，G1–G5 是策略智能体；实测性能不强制为 1.0×。"
        "纵轴采用各自 cuSPARSE 归一化后的速度比；跨 A100/A800 的小幅差异应谨慎解读。"
        "金星为最早出现且落在本轮最快生成算子 1.5% 计时等价带内的点。"
        "所有逐点来源、算子 SHA256、正确性和计时协议均重新审计。\n\n"
        f"基于历史最优 Sparsene 分母，现有 best-of-history DeepSeek 在 {wins}/26 数据集数值领先；"
        f"统计表：`{args.table_stem.with_suffix('.csv')}` 和 `.tex`。"
        "原 20261108 论文柱状图口径的统计文件和图保持不变，不能混用两种分母。\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(points)} validated generated points, 26 figures, {wins}/26 DS wins; ACC format0 points={len(acc)}")
    print(f"Plots: {output}")
    print(f"Table: {args.table_stem.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
