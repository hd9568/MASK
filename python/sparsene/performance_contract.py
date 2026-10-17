"""Shared experimental timing contract for format and strategy agents."""

from __future__ import annotations

from typing import Any, Dict


EVALUATION_CONTRACT_VERSION = "sparsene.kernel_only_post_conversion.v1"

# This is a project-level experimental contract, not a model-generated claim.
# Consumers should persist it with requests, decisions, and measured results.
KERNEL_ONLY_EVALUATION_CONTRACT: Dict[str, Any] = {
    "version": EVALUATION_CONTRACT_VERSION,
    "primary_metric": "warmed_gpu_spmm_compute_time_after_format_conversion_and_reordering",
    "timed_scope_cn": (
        "格式转换、元数据构造和重排全部完成后，CUDA event 内重复执行的 GPU SpMM "
        "计算 kernel 平均时间；kernel 内实际执行的 metadata 读取、gather、restore、"
        "MMA/累加和结果写回均计时。"
    ),
    "excluded_from_primary_metric": [
        "MatrixMarket 读取、特征提取、格式设计、代码生成与编译",
        "稀疏格式转换、padding/metadata 构造与 checkpoint I/O",
        "行列置换的构造与应用，以及为正确性执行的逆置换/恢复",
        "设备内存申请、H2D/D2H 传输和结果正确性验证",
        "cuSPARSE SpMM preprocess",
    ],
    "ranking_rule_cn": (
        "格式与策略的主排名只使用上述 kernel-only 时间；不得把格式转换或重排耗时"
        "加到该时间中，也不得因这些离线步骤更耗时而降低候选的 kernel 性能排名。"
    ),
    "engineering_reporting_cn": (
        "转换/重排仍必须语义正确、可实现、可缓存并记录额外存储、置换/逆映射和"
        "元数据要求；这些属于独立工程信息，不属于当前主性能指标。"
    ),
}
