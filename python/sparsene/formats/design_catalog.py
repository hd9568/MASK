"""Source-derived format knowledge for the research format-design agent.

The objects in this module explain the declarations under ``sparsene.formats``.
They are design references, not a claim that every path has an integrated host
bridge or fresh numerical/performance validation.  Keeping this catalog next to
the declarations makes prompt knowledge auditable against the source.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


def _axis(name: str, axis_type: str, direction: str, length: str) -> Dict[str, Any]:
    return {
        "name": name,
        "type": axis_type,
        "direction": direction,
        "length": length,
    }


SOURCE_HIERARCHIES: Dict[str, Dict[str, Any]] = {
    "TCF_COLUMN_CONDENSE_2L": {
        "levels": [
            {
                "level": 1,
                "name": "row_window_condensation",
                "axes": [
                    _axis("X_o", "df", "ROW", "M"),
                    _axis("Y_s", "sv", "COL", "K"),
                    _axis("X_i", "df", "ROW", "BLK_M"),
                ],
            },
            {
                "level": 2,
                "name": "tensorcore_block",
                "axes": [
                    _axis("Y_so", "df", "COL", "K/BLK_K"),
                    _axis("Y_si", "df", "COL", "BLK_K"),
                    _axis("X_i", "df", "ROW", "BLK_M"),
                ],
            },
        ],
        "semantics_cn": [
            "外层以 BLK_M 行为窗口；sv(Y_s,K) 的 K 是逻辑列域上界，实际每个窗口保存的凝聚列数由变长元数据给出。",
            "内层把凝聚后的列序列按 BLK_K 切分；Y_so 表示块序，Y_si 与 X_i 构成 atomic tile。",
        ],
        "variation_levers_cn": [
            "根据窗口内列复用选择是否保留列凝聚。",
            "根据 tile 占用、行内规则性和对角性替换 atomic leaf。",
            "可在证据支持时增加子 tile 层或显式稳定重排。",
        ],
    },
    "BIT_BSR_ACTIVE_BLOCK_1L": {
        "levels": [
            {
                "level": 1,
                "name": "active_fixed_blocks",
                "axes": [
                    _axis("X_o", "df", "ROW", "M"),
                    _axis("Y_o_s", "sv", "COL", "K"),
                    _axis("X_i", "df", "ROW", "BLK_M"),
                    _axis("Y_i", "df", "COL", "BLK_K"),
                ],
            }
        ],
        "semantics_cn": [
            "直接枚举每个块行中的活跃固定列块，不先构造窗口内逐列凝聚序列。",
            "X_i×Y_i 是固定逻辑块，MCO 叶子用 bitmask 与紧凑 values 表示块内非零。",
        ],
        "variation_levers_cn": [
            "适合固定块坐标比逐列凝聚更自然的局部块结构。",
            "可增加块行分层、块坐标压缩或把 MCO 换成 COO/dense/custom。",
        ],
    },
    "HME_MULTI_LEVEL_5L": {
        "levels": [
            {
                "level": 1,
                "name": "outer_row_window_condensation",
                "axes": [
                    _axis("X_o", "df", "ROW", "M"),
                    _axis("Y_s", "sv", "COL", "K"),
                    _axis("X_i", "df", "ROW", "BLK_M"),
                ],
            },
            {
                "level": 2,
                "name": "outer_tc_block",
                "axes": [
                    _axis("Y_so", "df", "COL", "K/BLK_K"),
                    _axis("Y_si", "df", "COL", "BLK_K"),
                    _axis("X_i", "df", "ROW", "BLK_M"),
                ],
            },
            {
                "level": 3,
                "name": "inner_column_sparsification",
                "axes": [
                    _axis("Y_sis", "sv", "COL", "BLK_K"),
                    _axis("X_i", "df", "ROW", "BLK_M"),
                ],
            },
            {
                "level": 4,
                "name": "inner_row_split",
                "axes": [
                    _axis("X_io", "df", "ROW", "BLK_M/BLK_M_I"),
                    _axis("Y_sis", "df", "COL", "BLK_K"),
                    _axis("X_ii", "df", "ROW", "BLK_M_I"),
                ],
            },
            {
                "level": 5,
                "name": "inner_tc_block",
                "axes": [
                    _axis("Y_siso", "df", "COL", "BLK_K/BLK_K_I"),
                    _axis("Y_sisi", "df", "COL", "BLK_K_I"),
                    _axis("X_ii", "df", "ROW", "BLK_M_I"),
                ],
            },
        ],
        "semantics_cn": [
            "先执行外层 TCF 凝聚与分块，再在外层 tile 内对列进行第二次稀疏化。",
            "BLK_M_I×BLK_K_I 子 tile 提供第二个结构尺度，最后由 COO leaf 恢复。",
        ],
        "variation_levers_cn": [
            "仅在单层 tile 内仍有可利用的多尺度聚集或空洞时增加层次。",
            "内层 atomic、行列子分块与第二次稀疏化方向都可以独立设计。",
        ],
    },
    "SPARSE_ROW_SR_BCRS_3L": {
        "levels": [
            {
                "level": 1,
                "name": "sparse_row_domain",
                "axes": [
                    _axis("X_s", "sv", "ROW", "M"),
                    _axis("Y", "df", "COL", "K"),
                ],
            },
            {
                "level": 2,
                "name": "sparse_row_blocks",
                "axes": [
                    _axis("X_so", "df", "ROW", "M/BLK_M"),
                    _axis("Y_s", "sv", "COL", "K"),
                    _axis("X_si", "df", "ROW", "BLK_M"),
                ],
            },
            {
                "level": 3,
                "name": "dense_active_block",
                "axes": [
                    _axis("Y_so", "df", "COL", "K/BLK_K"),
                    _axis("Y_si", "df", "COL", "BLK_K"),
                    _axis("X_si", "df", "ROW", "BLK_M"),
                ],
            },
        ],
        "semantics_cn": [
            "外层 sv(X_s,M) 表示稀疏/变长的活跃行域，再把活跃行组织成 BLK_M 行块并凝聚列。",
            "最内层 active block 采用 dense leaf，以零计算换取无坐标 restore。",
        ],
        "variation_levers_cn": [
            "稀疏行选择可以和显式 row reorder 组合，但两者不是同一语义。",
            "当 active block 不够稠密时可换成 MCO/CSR/ELL 或增加混合分区。",
        ],
    },
    "DENSE_OPERAND_ROW_MAJOR": {
        "levels": [
            {
                "level": 1,
                "name": "dense_row_major_operand",
                "axes": [
                    _axis("X", "df", "ROW", "M"),
                    _axis("Y", "df", "COL", "K"),
                ],
            }
        ],
        "semantics_cn": ["ROW 轴在前、COL 轴在后的稠密操作数布局。"],
        "variation_levers_cn": ["可以通过交换轴顺序得到转置遍历布局。"],
    },
    "DENSE_OPERAND_COL_MAJOR": {
        "levels": [
            {
                "level": 1,
                "name": "dense_col_major_operand",
                "axes": [
                    _axis("Y", "df", "COL", "K"),
                    _axis("X", "df", "ROW", "M"),
                ],
            }
        ],
        "semantics_cn": ["COL 轴在前、ROW 轴在后的转置稠密操作数布局。"],
        "variation_levers_cn": ["它说明轴顺序是布局决策，但本身不是完整稀疏设计。"],
    },
}


SOURCE_FORMAT_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "ME_TCF",
        "source": "DTC_SpMM.py::ME_TCF_FORMAT",
        "declared_name": "ME-TCF",
        "hierarchy": "TCF_COLUMN_CONDENSE_2L",
        "atomic": {"type": "coo", "x_axis": "X_i", "y_axis": "Y_si", "layout": "row_major"},
        "params": {"BLK_M": 16, "BLK_K": 8},
        "capability_context": "integrated_e2e only for the current consistent 16x8/16x16 DTC route",
        "atomic_semantics_cn": "每个 tile 保存紧凑 values 与局部行列坐标；低占用和不规则块鲁棒，但坐标及 scatter/restore 开销较高。",
        "caveats_cn": ["窗口内列凝聚/排序不是数学矩阵的全局行列重排。"],
    },
    {
        "id": "BIT_TCF_ACC",
        "source": "Acc_SpMM.py::BIT_TCF_FORMAT",
        "declared_name": "DTC-SpMM",
        "hierarchy": "TCF_COLUMN_CONDENSE_2L",
        "atomic": {"type": "mco", "x_axis": "X_i", "y_axis": "Y_si", "layout": "row_major"},
        "params": {"BLK_M": 16, "BLK_K": 8},
        "capability_context": "standalone_validated registry profile; not the unified e2e route",
        "atomic_semantics_cn": "以固定 tile bitmask 加紧凑 values 代替逐非零坐标；适合 mask 元数据小于 COO 且占用率不极低的块。",
        "caveats_cn": ["常量名是 BIT_TCF_FORMAT，但当前 Format.name 仍写作 DTC-SpMM；不要由该命名推断它是 COO。"],
    },
    {
        "id": "CSR_TCF",
        "source": "CSR_TCF.py::CSR_TCF_FORMAT",
        "declared_name": "CSR-TCF",
        "hierarchy": "TCF_COLUMN_CONDENSE_2L",
        "atomic": {"type": "csr", "x_axis": "X_i", "y_axis": "Y_si", "layout": "row_major"},
        "params": {"BLK_M": 16, "BLK_K": 8},
        "capability_context": "codegen_declared; host and numerical integration still require verification",
        "atomic_semantics_cn": "tile 内保存 row_ptr、col_idx、values；行向遍历和局部行长度偏斜时可减少 COO 行坐标。",
        "caveats_cn": ["需要专用 CSR payload/restore，不能只把 COO 数组改名。"],
    },
    {
        "id": "ELL_TCF",
        "source": "ELL_TCF.py::ELL_TCF_FORMAT",
        "declared_name": "ELL-TCF",
        "hierarchy": "TCF_COLUMN_CONDENSE_2L",
        "atomic": {"type": "ell", "x_axis": "X_i", "y_axis": "Y_si", "layout": "row_major"},
        "params": {"BLK_M": 16, "BLK_K": 8},
        "capability_context": "codegen_declared; host and numerical integration still require verification",
        "atomic_semantics_cn": "tile 内每行填充到局部最大长度并保存二维 col_idx/values；行长度均匀时访存规则，长尾时 padding 昂贵。",
        "caveats_cn": ["必须说明 ell_len、padding sentinel 与边界恢复语义。"],
    },
    {
        "id": "DIA_TCF",
        "source": "DIA_TCF.py::DIA_TCF_FORMAT",
        "declared_name": "DIA-TCF",
        "hierarchy": "TCF_COLUMN_CONDENSE_2L",
        "atomic": {"type": "dia", "x_axis": "X_i", "y_axis": "Y_si", "layout": "row_major"},
        "params": {"BLK_M": 16, "BLK_K": 8},
        "capability_context": "codegen_declared; host and numerical integration still require verification",
        "atomic_semantics_cn": "tile 内保存对角 offset 与按对角线填充的 values；仅在局部对角数少且对角占用高时划算。",
        "caveats_cn": ["全局带状性不自动保证凝聚后的 tile 仍保持少对角结构。"],
    },
    {
        "id": "SR_BCRS",
        "source": "SR_BCRS.py::SR_BCRS_FORMAT",
        "declared_name": "SR-BCRS",
        "hierarchy": "TCF_COLUMN_CONDENSE_2L",
        "atomic": {"type": "dense", "x_axis": "X_i", "y_axis": "Y_si", "layout": "row_major"},
        "params": {"BLK_M": 16, "BLK_K": 8},
        "capability_context": "standalone_validated registry profile; not the unified e2e route",
        "atomic_semantics_cn": "仅枚举 active blocks，块内按 dense tile 存储/计算；高结构密度时省去 restore 元数据，低密度时零填充和零计算占主导。",
        "caveats_cn": ["active-block 选择仍需稀疏元数据，dense 只描述叶子内部。"],
    },
    {
        "id": "BIT_BSR_SPADEN",
        "source": "Spaden.py::BIT_BSR_FORMAT",
        "declared_name": "BIT-BSR",
        "hierarchy": "BIT_BSR_ACTIVE_BLOCK_1L",
        "atomic": {"type": "mco", "x_axis": "X_i", "y_axis": "Y_i", "layout": "row_major"},
        "params": {"BLK_M": 16, "BLK_K": 8},
        "capability_context": "standalone_validated registry profile; not the unified e2e route",
        "atomic_semantics_cn": "块坐标层稀疏、块内 mask-compact；避免逐列凝聚但必须保存/遍历活跃块索引。",
        "caveats_cn": ["它不是全矩阵稠密块网格；Y_o_s 表示每个父域中的稀疏活跃块选择。"],
    },
    {
        "id": "HME_TCF",
        "source": "HME_TCF.py::HME_TCF_FORMAT",
        "declared_name": "HME-TCF",
        "hierarchy": "HME_MULTI_LEVEL_5L",
        "atomic": {"type": "coo", "x_axis": "X_ii", "y_axis": "Y_sisi", "layout": "row_major"},
        "params": {"BLK_M": 16, "BLK_K": 16, "BLK_M_I": 8, "BLK_K_I": 8},
        "capability_context": "experimental frontend declaration",
        "atomic_semantics_cn": "外层凝聚块内部再次稀疏化和子分块，以更多元数据/调度换取多尺度空洞消除。",
        "caveats_cn": ["源码明确记录两种 HME 结构的 RTS 不稳定；不能宣称已经端到端可执行。"],
    },
    {
        "id": "ROW_REORDER_SR_BCRS",
        "source": "ROW_REORDER_SR_BCRS.py::ROW_REORDER_SR_BCRS_FORMAT",
        "declared_name": "ROW-REORDER-SR-BCRS",
        "hierarchy": "SPARSE_ROW_SR_BCRS_3L",
        "atomic": {"type": "dense", "x_axis": "X_si", "y_axis": "Y_si", "layout": "row_major"},
        "params": {"BLK_M": 16, "BLK_K": 8},
        "capability_context": "experimental declaration; explicit reorder contract is incomplete",
        "atomic_semantics_cn": "先使用稀疏变长行域，再进行行块/列凝聚和 dense active-block restore。",
        "caveats_cn": [
            "尽管文件和格式名含 ROW-REORDER，当前所有 Axis.reorder 都是 None。sv(X_s) 是稀疏行域，不等价于稳定行置换。",
            "真正 row reorder 设计必须指定 reorder 标识、置换构造、payload/metadata 置换、输出逆映射、缓存身份和额外存储；这些转换步骤在当前实验中不进入 kernel-only 主计时，但必须保证正确且可实现。",
        ],
    },
    {
        "id": "DENSE_AND_DENSE_T",
        "source": "dense.py::DENSE_FORMAT,DENSE_T_FORMAT",
        "declared_name": "dense,dense_T",
        "hierarchy": "DENSE_OPERAND_ROW_MAJOR / DENSE_OPERAND_COL_MAJOR",
        "atomic": None,
        "params": {},
        "capability_context": "operand layout building blocks, not complete sparse-format-agent outputs",
        "atomic_semantics_cn": "两个声明只通过 ROW/COL 轴顺序表达普通与转置稠密布局，没有 sparse axis，也没有 atomic child。",
        "caveats_cn": ["格式智能体的完整稀疏输出仍必须有一个 atomic leaf；不能原样返回 dense.py 的无 child Format。"],
    },
]


def complete_source_designs() -> List[Dict[str, Any]]:
    """Materialize validator-ready source patterns with an atomic leaf."""
    designs: List[Dict[str, Any]] = []
    for item in SOURCE_FORMAT_CATALOG:
        atomic = item.get("atomic")
        hierarchy_name = item.get("hierarchy")
        hierarchy = SOURCE_HIERARCHIES.get(str(hierarchy_name))
        if not isinstance(atomic, dict) or hierarchy is None:
            continue
        designs.append(
            {
                "format_name": item["id"],
                "levels": deepcopy(hierarchy["levels"]),
                "atomic": deepcopy(atomic),
                "params": deepcopy(item.get("params", {})),
            }
        )
    return designs


def prompt_source_catalog() -> Dict[str, Any]:
    """Return compact, JSON-serializable knowledge for prompt injection."""
    used_hierarchies = {
        item["hierarchy"]
        for item in SOURCE_FORMAT_CATALOG
        if item.get("hierarchy") in SOURCE_HIERARCHIES
    }
    return {
        "hierarchy_patterns": {
            name: SOURCE_HIERARCHIES[name] for name in sorted(used_hierarchies)
        },
        "source_formats": deepcopy(SOURCE_FORMAT_CATALOG),
        "composition_rule_cn": (
            "把分组/凝聚/子分块/reorder 层次与 atomic 编码视为可组合设计维度；"
            "可以复用、删减、增加或混合源码范式，但最终 levels 必须独立合法，且每个变化都要有特征和成本依据。"
        ),
    }
