#!/usr/bin/env python3
"""Render or verify the canonical Strategy Agent field reference.

The executable catalog lives next to StrategyDecision so code, prompts, tests,
and documentation share one source of truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparsene.op_gen.strategy_agent import strategy_field_catalog, strategy_search_space


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "docs/codex/strategy_agent_search_space.md"
GROUP_TITLES = {
    "tensor_placement": "张量放置与数据搬运",
    "pipeline": "软件流水与同步",
    "compiler": "编译器资源控制",
    "runtime_schedule": "运行时任务映射",
}


def _choice_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_document() -> str:
    catalog = strategy_field_catalog()
    space = strategy_search_space()
    group_sizes: dict[str, int] = {}
    for group, fields in catalog.items():
        size = 1
        for metadata in fields.values():
            size *= len(metadata["choices"])
        group_sizes[group] = size

    lines = [
        "# Strategy Agent 搜索空间字段手册",
        "",
        "> 状态：持续维护；权威字段定义来自 `python/sparsene/op_gen/strategy_agent.py` 的 `_STRATEGY_FIELD_CATALOG`。每次增删字段或取值后，应运行 `PYTHONPATH=python python python/test/export_strategy_space_document.py --write`，并用 `--check` 做一致性检查。",
        "",
        "本文记录 Strategy Agent 能提交的全部决策字段。这里的“原始空间”是各字段离散取值数的笛卡尔积；它包含被规范化为同一配置、对某种 backend 不适用、以及可能无法通过编译或正确性门禁的组合。因此原始空间用于描述 Agent 面对的上界，不等于某个数据集的有效独立算子数量。",
        "",
        "## 空间规模",
        "",
        "| 分组 | 字段数 | 原始组合数 |",
        "|---|---:|---:|",
    ]
    for group, fields in catalog.items():
        lines.append(
            f"| `{group}`（{GROUP_TITLES[group]}） | {len(fields)} | {group_sizes[group]:,} |"
        )
    formula = " × ".join(f"{group_sizes[group]:,}" for group in catalog)
    lines.extend(
        [
            f"| **总计** | **{sum(len(fields) for fields in catalog.values())}** | **{space['raw_cartesian_size']:,}** |",
            "",
            f"原始空间大小为 `{formula} = {space['raw_cartesian_size']:,}`。",
            "",
        ]
    )
    for group, fields in catalog.items():
        lines.extend(
            [
                f"## {GROUP_TITLES[group]}（`{group}`）",
                "",
                "| 字段 | 取值范围 | 数量 | 含义 | 生效条件与约束 |",
                "|---|---|---:|---|---|",
            ]
        )
        for name, metadata in fields.items():
            choices = "<br>".join(f"`{_choice_text(value)}`" for value in metadata["choices"])
            lines.append(
                f"| `{group}.{name}` | {choices} | {len(metadata['choices'])} | "
                f"{metadata['meaning']} | {metadata['applicability']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 规范化、门禁与非搜索字段",
            "",
            "`StrategyDecision.from_dict()` 先做跨字段规范化，例如关闭 B 的 SMEM 物化会同时关闭 B cp.async；COO 任一 staging 进入 RMEM 会使 index/value 同时进入 RMEM并强制同步加载；关闭异步流水会把异步 producer 降级为同步路径；默认 `[1,1]` stage shift 规范化为空列表。`normalize_strategy_for_backend()` 进一步按 backend 约束调度：三种 dense SR-BCRS ABI packing mode（`column_condensed_dense`、`active_fixed_blocks_dense`、`sparse_row_column_condensed_dense`）可使用 task mapping；MCO backend 将 `execution_policy` 固定为 `row`，并把 `task_group_tiles` 固定为 64。所有候选仍必须依次通过生成、编译、全输出正确性和性能门禁。",
            "",
            "下列字段属于记录或解释信息，不进入上述笛卡尔积：顶层 `provider`、顶层 `rationale`、`pipeline.rationale`、`token_usage` 和 `agent_metadata`。Agent 的请求还包含硬件、矩阵特征、固定格式、历史 scoreboard、策略覆盖率、实际执行映射和 NCU 指标，这些是选择配置的上下文，不是决策轴。",
            "",
            "反馈迭代的模型输出使用紧凑 `strategy_patch`，只列出相对 best-correct 策略真正修改的字段；本地 driver 合并 patch 后再执行完整规范化和正确性门禁。完整策略仍保存在每轮 `strategy_decision.json`，因此 patch 只是减少模型输出长度，不改变最终策略字段集合或搜索空间定义。",
            "",
            "性能比较固定为 N=512、FP32 输入输出、warmup=3、repeat=32；同一编译配置默认独立运行 3 次并按 `generated_ms` 中位数排名。小于 1.5% 的差异视为噪声区间，除非重复样本和 NCU 证据共同支持。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render_document()
    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(expected, encoding="utf-8")
        print(args.output)
        return 0
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != expected:
            print(f"out of date: {args.output}")
            return 1
        print(f"up to date: {args.output}")
        return 0
    print(expected, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
