"""Feature-grounded research sparse-format design for Sparsene.

The format agent designs against the general hierarchical Format DSL rather
than the capabilities of one host/backend route.  Deterministic code validates
the proposed DSL without silently replacing it with a known format, then
classifies how much of the proposal the current repository can execute.

This separation is intentional:

    matrix features -> research DSL proposal -> DSL validation
                    -> capability classification -> later implementation

An unsupported but valid proposal is useful design output.  It must never be
misrepresented as an executable kernel, and it must never be normalized to
ME-TCF merely to make the current DTC testbed accept it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

from sympy import Expr, Integer, Symbol, sympify

from sparsene.agent.runtime import AgentModelConfig, invoke_json_agent, make_zero_usage
from sparsene.format.format import (
    Axis,
    AtomicFormat,
    Direction,
    Format,
    atomic_format,
    df_axis,
    dv_axis,
    sf_axis,
    sv_axis,
)
from sparsene.formats.design_catalog import prompt_source_catalog
from sparsene.logging import get_logger
from sparsene.op_gen.strategy_agent import discover_hardware_profile
from sparsene.op_gen.backend_registry import select_backend
from sparsene.performance_contract import (
    EVALUATION_CONTRACT_VERSION,
    KERNEL_ONLY_EVALUATION_CONTRACT,
)
from sparsene.transform.rts import derive_rts

logger = get_logger(__name__)

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_MAX_TOKENS = 32768

DESIGN_SCHEMA_VERSION = "sparsene.format_design.v6"
DESIGN_PROFILE = "hierarchical_sparse_format_research_v2"
EXECUTABLE_PROFILE = "me_tcf_coo_sm80_v1"  # compatibility name for the one integrated route

AXIS_TYPES = ("df", "dv", "sf", "sv")
ATOMIC_TYPES = ("coo", "mco", "csr", "ell", "dia", "dense", "custom")
REFERENCE_TILE_KEYS = (
    "8x8",
    "8x16",
    "8x32",
    "16x8",
    "16x16",
    "16x32",
    "32x8",
    "32x16",
    "32x32",
)
INTEGRATED_TILE_KEYS = ("16x8", "16x16")
SUPPORT_LEVELS = (
    "invalid",
    "frontend_research",
    "codegen_declared",
    "standalone_validated",
    "integrated_e2e",
)

ROW = Direction.ROW
COL = Direction.COL

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORMAT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_LENGTH_EXPR_RE = re.compile(r"^[A-Za-z0-9_+\-*/() ]+$")
_REORDER_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _canonical_tcf_levels() -> List[Dict[str, Any]]:
    """Reference two-level column-condensation hierarchy used by TCF families."""
    return [
        {
            "level": 1,
            "name": "row_window",
            "axes": [
                {"name": "X_o", "type": "df", "direction": "ROW", "length": "M"},
                {"name": "Y_s", "type": "sv", "direction": "COL", "length": "K"},
                {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
            ],
        },
        {
            "level": 2,
            "name": "tc_block",
            "axes": [
                {"name": "Y_so", "type": "df", "direction": "COL", "length": "K/BLK_K"},
                {"name": "Y_si", "type": "df", "direction": "COL", "length": "BLK_K"},
                {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
            ],
        },
    ]


def _canonical_levels() -> List[Dict[str, Any]]:
    """Backward-compatible alias used by older callers and historical tests."""
    return _canonical_tcf_levels()


def _canonical_atomic(atomic_type: str = "coo") -> Dict[str, Any]:
    return {
        "type": atomic_type,
        "x_axis": "X_i",
        "y_axis": "Y_si",
        "layout": "row_major",
    }


def _bit_bsr_levels() -> List[Dict[str, Any]]:
    return [
        {
            "level": 1,
            "name": "block_row",
            "axes": [
                {"name": "X_o", "type": "df", "direction": "ROW", "length": "M"},
                {"name": "Y_o_s", "type": "sv", "direction": "COL", "length": "K"},
                {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                {"name": "Y_i", "type": "df", "direction": "COL", "length": "BLK_K"},
            ],
        }
    ]


REFERENCE_PROFILES: Dict[str, Dict[str, Any]] = {
    "ME_TCF": {
        "levels": _canonical_tcf_levels(),
        "atomic": _canonical_atomic("coo"),
        "support": "integrated_e2e",
        "description": "row-window column condensation with COO TC-block leaves",
    },
    "BIT_TCF": {
        "levels": _canonical_tcf_levels(),
        "atomic": _canonical_atomic("mco"),
        "support": "standalone_validated",
        "description": "TCF hierarchy with mask/MCO leaves (ACC-style)",
    },
    "SR_BCRS": {
        "levels": _canonical_tcf_levels(),
        "atomic": _canonical_atomic("dense"),
        "support": "standalone_validated",
        "description": "row-window hierarchy with dense block leaves",
    },
    "BIT_BSR": {
        "levels": _bit_bsr_levels(),
        "atomic": {
            "type": "mco",
            "x_axis": "X_i",
            "y_axis": "Y_i",
            "layout": "row_major",
        },
        "support": "standalone_validated",
        "description": "block sparse hierarchy with mask/MCO leaves",
    },
    "CSR_TCF": {
        "levels": _canonical_tcf_levels(),
        "atomic": _canonical_atomic("csr"),
        "support": "codegen_declared",
        "description": "TCF hierarchy with per-tile CSR leaves",
    },
    "ELL_TCF": {
        "levels": _canonical_tcf_levels(),
        "atomic": _canonical_atomic("ell"),
        "support": "codegen_declared",
        "description": "TCF hierarchy with per-tile ELL leaves",
    },
    "DIA_TCF": {
        "levels": _canonical_tcf_levels(),
        "atomic": _canonical_atomic("dia"),
        "support": "codegen_declared",
        "description": "TCF hierarchy with per-tile DIA leaves",
    },
}


# ==============================================================================
# Data structures
# ==============================================================================


@dataclass
class FormatDesignRequest:
    request_id: str
    matrix_name: str
    matrix_features: Dict[str, Any]
    hardware: Dict[str, Any]
    output_schema: Dict[str, Any]
    constraints: List[str]
    rules: List[str]
    # Historical kernel measurements are calibration evidence only.  They do
    # not replace the feature-grounded research design task or constrain the
    # DSL to an already integrated backend.
    empirical_feedback: Dict[str, Any] = field(default_factory=dict)
    prompt: str = ""


@dataclass
class FormatDesignResponse:
    request_id: str
    format_name: str
    levels: List[Dict[str, Any]]
    atomic: Dict[str, Any]
    params: Dict[str, Any]
    rationale: str
    confidence: str
    schema_version: str = DESIGN_SCHEMA_VERSION
    design_profile: str = DESIGN_PROFILE
    format_family: str = "custom"
    design_summary: str = ""
    transformation_plan: List[Dict[str, Any]] = field(default_factory=list)
    selected_candidate: str = "not_applicable"
    rationale_cn: str = ""
    feature_evidence: List[Dict[str, Any]] = field(default_factory=list)
    tradeoffs: List[str] = field(default_factory=list)
    alternatives_considered: List[Dict[str, Any]] = field(default_factory=list)
    decision_basis_cn: List[str] = field(default_factory=list)
    format_semantics_cn: List[str] = field(default_factory=list)
    selection_reason_cn: str = ""
    implementation_requirements: List[str] = field(default_factory=list)
    risks_cn: List[str] = field(default_factory=list)
    dsl_validation: Dict[str, Any] = field(default_factory=dict)
    capability: Dict[str, Any] = field(default_factory=dict)
    executable_profile: str = "none"
    evaluation_contract: Dict[str, Any] = field(
        default_factory=lambda: deepcopy(KERNEL_ONLY_EVALUATION_CONTRACT)
    )
    normalization_actions: List[str] = field(default_factory=list)
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    token_usage: Dict[str, Any] = field(default_factory=dict)
    agent_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormatDesignConfig:
    model_name: Optional[str] = DEFAULT_DEEPSEEK_MODEL
    api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    api_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    model: str = DEFAULT_DEEPSEEK_MODEL
    log_dir: str = "docs/agent_runs"
    timeout_sec: int = 180
    temperature: float = 0.35
    max_tokens: int = DEFAULT_MAX_TOKENS
    model_config_path: Optional[str] = None
    skill_names: List[str] = field(
        default_factory=lambda: ["format-design"]
    )


# ==============================================================================
# Prompt content
# ==============================================================================


DSL_EDUCATION = r"""
## 1. WHAT A SPARSENE FORMAT DESIGN MEANS

A design is a recursive hierarchy of Format levels terminated by exactly one
AtomicFormat leaf.  It is a physical sparse layout, not merely a kernel tile
choice and not merely a name copied from an existing paper.

Each level contains ordered axes.  Axis order controls traversal/data layout;
axis type controls whether the coordinate space is dense/sparse and fixed/varlen:

- df: dense fixed-length. Regular blocking or a hardware-aligned tile extent.
- dv: dense variable-length. Dense coordinates whose realized extent varies.
- sf: sparse fixed-length. A sparse selection with a fixed logical bound.
- sv: sparse variable-length. Compressed/condensed coordinates with per-parent
  variable cardinality.

Directions are ROW or COL.  Reusing a logical axis name across adjacent levels
is legal when one level refines that coordinate.  Every level must still cover
both ROW and COL directions and make its split/condensation semantics clear.

`length` is the logical domain bound or split factor used by Format/RTS, not a
natural-language description of the number of stored coordinates.  For a
variable-length axis, per-parent realized cardinality comes from metadata while
`length` remains a legal bound such as `K` or `BLK_K`.  Follow these exact rules:

- `level` is an unquoted JSON integer: `1`, `2`, ...; never `"1"`.
- A length is an integer or a string containing only M, K, N, a declared
  parameter, parentheses, and `+ - * /`.
- Do not use `ceil`, `floor`, `%`, `//`, dotted references, prose such as
  `varies per row_window`, or an undeclared symbol.
- Follow the source convention for splitting: `X_o` with length `M` plus
  `X_i` with length `BLK_M` expresses row blocking; do not replace it with
  `ceil(M/BLK_M)`.  `K/BLK_K` is a legal inner block-count expression.

The atomic leaf defines how one innermost logical tile is physically stored and
restored before MMA:

- coo: coordinate pairs + values; robust for very low/irregular occupancy, but
  coordinate metadata and restore/scatter can be expensive.
- mco: bitmask + compact values; attractive when a fixed tile mask is cheaper
  than coordinates and occupancy is not vanishingly small.
- csr: row pointer + column/value streams; useful when row-oriented traversal
  and uneven per-row occupancy matter.
- ell: fixed slots per row; useful only when per-row lengths inside a tile are
  sufficiently uniform, otherwise padding dominates.
- dia: diagonal identifiers + values; useful only with strong diagonal/local
  band structure.
- dense: no sparse coordinates inside an active block; useful when active
  blocks are dense enough to amortize zero work.
- custom: a new user-defined leaf.  It is legal only when `custom_spec`
  explicitly defines stored value arrays, metadata arrays, the mapping from
  storage coordinates to logical (row,col), and the restore algorithm.  It will
  require a new backend implementation.

`params` gives concrete positive integer design parameters referenced by level
length expressions.  `selected_candidate` is an
optional reference tile from the supplied feature table; it is evidence for a
design, not the design itself.

`transformation_plan` explains the implied Split, Swap, Sparsify, Spawn,
Atomicize, and optional reorder operations.  The hierarchical `levels` plus
`atomic` object remain the authoritative DSL.

## 2. BENCHMARK OBJECTIVE AND TIMING BOUNDARY

The primary optimization target is warmed GPU SpMM compute-kernel time after
all sparse-format conversion, padding/metadata construction, row/column
reordering, and any correctness-only inverse permutation have completed.
Those transformation steps are outside the measured time.  Do not add their
latency to the kernel metric and do not penalize or reject a format merely
because its offline conversion or reorder is expensive.

Everything actually executed inside the timed SpMM kernel remains visible to
the objective: metadata reads, indirect B gathers, atomic restore/scatter,
padding-induced work, load imbalance, MMA/accumulation, synchronization, and C
writes all count.  Therefore optimize kernel-visible metadata traffic, useful
work, locality, occupancy, and load balance.

Conversion and reorder are not exempt from correctness or implementability.
Specify their metadata, permutation/inverse-map semantics, storage, cache
identity, and required host/backend work, but label them as excluded from the
current primary timing metric.  They may be reported separately and must never
be mixed into the kernel-only ranking.

## 3. EXISTING FORMATS ARE REFERENCES, NOT A CLOSED MENU

- ME-TCF groups rows, condenses the union of touched columns in each row
  window, tiles that condensed axis, and stores each tile as COO.
- BIT-TCF/ACC uses a similar hierarchy with mask/MCO leaves.
- BIT-BSR keeps regular block coordinates and uses a mask leaf.
- SR-BCRS represents active inner blocks densely.
- CSR/ELL/DIA leaves express different local regularities.
- Additional levels or row/column reordering can represent multi-scale or
  skewed structure.

You may reproduce one of these when the evidence supports it, hybridize their
ideas, or propose a new hierarchy.  Novelty and diversity are not goals by
themselves: every level, axis, reorder, atomic encoding, and parameter must
solve a feature-backed cost problem.

The current repository only executes some known profiles.  That implementation
state MUST NOT restrict this research design.  Do not claim that a proposal is
already executable.  Deterministic code will classify backend support after
the proposal is returned.
"""


FEATURE_SEMANTICS = r"""
## 4. HOW TO USE THE MATRIX FEATURES

Start from the matrix pathology, then choose a representation:

- row/column mean, percentiles, CV, Gini, zero fraction, and top-1% share expose
  skew, long tails, empty regions, and likely load imbalance;
- adjacent-row Jaccard and row-window unique-column reduction estimate whether
  row grouping/column condensation can reuse gathered B rows;
- locality bandwidth, row spans, column gaps, and diagonal-band fraction expose
  banded or scattered structure and help evaluate DIA-like/local formats;
- reorder proxies estimate possible post-reorder kernel benefit, but they do
  not predict exact post-reorder kernel performance; reorder preprocessing
  latency is outside the primary metric and must not reduce its ranking;
- hierarchical candidate statistics show row-window/tile density, padding,
  block-count imbalance, reuse, and estimated COO/mask/dense storage for nine
  BLK_M x BLK_K shapes;
- atomic encoding should be chosen from occupancy distribution and storage/
  restore trade-offs, not from global density alone;
- high block density can justify dense leaves; low density favors coordinate
  forms; intermediate density and smaller estimated mask bytes can favor MCO;
- strong per-row regularity can motivate ELL, strong diagonal locality DIA, and
  row-skewed tiles CSR.  If the provided features cannot measure a claimed
  property, state it as an uncertainty rather than inventing it.

The hierarchical candidate table is measured under the current ME-style row
window column-condensation analysis.  It is exact evidence for that hierarchy,
but only a proxy when your design changes grouping, reordering, or hierarchy.
Make that distinction explicit.
"""


EXECUTABLE_BACKEND_COST_MODEL = r"""
## EXECUTABLE SM80 COST MODEL FOR TCF ATOMIC COMPARISONS

This is an implementation fact for the currently executable two-level
column-condensed SM80 backends.  It does not restrict a novel research DSL,
but it must calibrate any comparison that cites the existing TCF MCO or
SR-BCRS dense path:

- MCO and dense siblings enumerate the same active condensed column tiles,
  gather the same B rows, and execute the same full physical TF32 Tensor Core
  MMA tile for every active tile.  MCO **does not skip zero-position MMA work
  or B gathers** in the current backend.
- MCO saves A-value payload bytes, but every tile also loads masks and offsets,
  restores a complete physical dense A tile in shared memory, and performs
  mask rank/popcount work.  Dense directly stages its complete A tile and has
  no mask restore.
- A logical BLK_M=8 is physically padded to M=16 before restore/MMA/store;
  this extra GPU work is timed.  BLK_M=16/32 avoids that particular padding.
- `storage_estimate_bytes.mask < storage_estimate_bytes.dense` is a storage
  observation, not a kernel-time prediction.  Do not infer MCO is faster from
  storage alone, and never claim MCO values are FP16: this evaluation uses
  FP32 inputs/outputs with TF32 MMA.

For an executable two-level TCF design, compare the same hierarchy with both
MCO and dense leaves whenever their predicted costs are close.  The offline
portfolio runner preserves the Agent proposal and records this as a separate,
deterministic measurement sibling; it never silently changes the selected DSL.
"""


FORMAT_DESIGN_RULES = [
    "Design against the general hierarchical Format DSL; do not preserve ME-TCF unless the matrix evidence independently supports it.",
    "Return one to six non-empty Format levels. level must be an unquoted 1-based JSON integer; each level has two to eight uniquely named axes and both ROW and COL directions.",
    "Use only axis types df/dv/sf/sv. Atomic type may be coo/mco/csr/ell/dia/dense or a fully specified custom leaf.",
    "Every axis length must obey the exact arithmetic grammar. For sv/dv use a legal logical bound and describe realized per-parent varlen metadata in purpose/semantics, never in length.",
    "The atomic x_axis and y_axis must reference ROW and COL axes in the deepest level; never invent dangling axis references.",
    "Every nontrivial hierarchy, axis ordering, sparse/varlen tag, reorder, atomic encoding, and parameter must have a feature-backed purpose.",
    "Compare at least three materially different format alternatives using kernel-visible metadata traffic, padding/zero work, restore, locality, occupancy, and load-balance costs. Describe conversion/reorder engineering separately, but never include its latency in the kernel-only ranking.",
    "Cite six to eight exact values from the supplied feature path-value map using verbatim dotted paths; never invent, combine, or silently extrapolate measurements.",
    "Treat hierarchical candidate statistics as exact only for the analyzed ME-style condensation; label them proxies for structurally different proposals.",
    "Do not optimize for current backend support and do not claim execution readiness; list the implementation work your design would require.",
    "Write the main rationale, semantic explanation, final selection reason, and risks in Chinese so the design record is directly auditable.",
    "Prefer the simplest hierarchy that explains the evidence; novelty without a measurable benefit is a design defect.",
    "For current executable TCF MCO/dense comparisons, never claim MCO skips Tensor Core zero work or B gathers; distinguish compact A storage from timed restore/MMA cost.",
    "If data quality is invalid or decisive features are missing, lower confidence and explain which additional measurements are needed.",
]


def _default_output_schema() -> Dict[str, Any]:
    """Return a type-correct JSON example, not a prescribed format template."""
    return {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "format_name": "ExampleOnly_DoNotCopy",
        "format_family": "hybrid",
        "design_summary": "用一句中文概括层次、atomic 和预期收益",
        "levels": [
            {
                "level": 1,
                "name": "active_blocks",
                "axes": [
                    {
                        "name": "X_o",
                        "type": "df",
                        "direction": "ROW",
                        "length": "M",
                        "reorder": None,
                        "purpose": "按 BLK_M 对逻辑行域切分",
                    },
                    {
                        "name": "Y_o_s",
                        "type": "sv",
                        "direction": "COL",
                        "length": "K",
                        "reorder": None,
                        "purpose": "每个块行的实际活跃列块数由变长元数据给出",
                    },
                    {
                        "name": "X_i",
                        "type": "df",
                        "direction": "ROW",
                        "length": "BLK_M",
                        "reorder": None,
                        "purpose": "固定块内行坐标",
                    },
                    {
                        "name": "Y_i",
                        "type": "df",
                        "direction": "COL",
                        "length": "BLK_K",
                        "reorder": None,
                        "purpose": "固定块内列坐标",
                    },
                ],
            }
        ],
        "atomic": {
            "type": "mco",
            "x_axis": "X_i",
            "y_axis": "Y_i",
            "layout": "row_major",
            "rationale": "用一句中文说明 occupancy、metadata 与 restore 权衡",
            "custom_spec": None,
        },
        "params": {"BLK_M": 16, "BLK_K": 8},
        "selected_candidate": "16x8",
        "transformation_plan": [
            {
                "op": "split",
                "target": "ROW",
                "result": "产生外层块行与 X_i",
                "reason": "引用一个精确特征并说明原因",
            },
            {
                "op": "atomicize",
                "target": "X_i,Y_i",
                "result": "MCO mask 与紧凑 values",
                "reason": "引用 tile occupancy/storage 证据",
            }
        ],
        "feature_evidence": [
            {
                "feature": "row_nnz.cv",
                "value": 0.0,
                "implication": "示例；实际输出必须从 FEATURE PATH-VALUE MAP 原样复制6至8条真实 path/value",
            }
        ],
        "alternatives_considered": [
            {
                "name": "另一完整格式",
                "structure": "简短层次与 atomic 摘要",
                "pros": ["一个特征支持的优点"],
                "cons": ["metadata/padding/restore/load-balance 中的关键成本"],
                "why_rejected": "决定性原因或保留为 fallback 的条件",
            }
        ],
        "tradeoffs": ["一条简洁的跨格式权衡"],
        "rationale": "one concise English technical sentence",
        "rationale_cn": "一段简洁中文总体依据",
        "decision_basis_cn": ["一条中文分点依据"],
        "format_semantics_cn": ["至少两条：逐层解释物理布局、坐标元数据和 atomic restore"],
        "selection_reason_cn": "用一段中文说明为何完整格式优于三个备选",
        "implementation_requirements": [
            "列出 atomic、host preprocessing/reorder、metadata ABI、lowering/fusion 与 correctness 所需工作；明确这些转换步骤不计入当前 kernel-only 主指标"
        ],
        "risks_cn": ["一条静态特征无法确认、需要闭环验证的风险"],
        "confidence": "medium",
    }


# ==============================================================================
# Feature digest and prompt rendering
# ==============================================================================


def _candidate_stats(features: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    candidates = (
        features.get("hierarchical_candidates")
        or features.get("candidate_block_stats")
        or {}
    )
    return {
        key: value
        for key, value in candidates.items()
        if isinstance(key, str) and isinstance(value, dict)
    }


def _deterministic_candidate(features: Dict[str, Any]) -> str:
    """Compatibility helper; research mode does not use a forced fallback tile."""
    candidates = _candidate_stats(features)
    return "16x8" if "16x8" in candidates else (next(iter(candidates), "not_applicable"))


def _distribution_digest(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keys = (
        "count",
        "mean",
        "std",
        "min",
        "max",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "cv",
        "gini",
        "zero_fraction",
        "top1pct_share",
    )
    return {key: value.get(key) for key in keys if key in value}


def _feature_digest(features: Dict[str, Any]) -> Dict[str, Any]:
    row = features.get("row_nnz_distribution", {})
    col = features.get("col_nnz_distribution", {})
    locality = features.get("locality", {})
    reorder = features.get("reorder_proxy", {})
    digest: Dict[str, Any] = {
        "schema_version": features.get("schema_version"),
        "shape": features.get("shape", {}),
        "density": features.get("density"),
        "sparsity": features.get("sparsity"),
        "symmetry": features.get("symmetry"),
        "data_quality": features.get("data_quality", {}),
        "row_nnz": _distribution_digest(row),
        "col_nnz": _distribution_digest(col),
        "row_length_vs_index_correlation": features.get("row_length_vs_index_correlation"),
        "locality": {
            "normalized_bandwidth": locality.get("normalized_bandwidth"),
            "diagonal_band_fraction_1pct": locality.get("diagonal_band_fraction_1pct"),
            "row_span": _distribution_digest(locality.get("row_span_distribution", {})),
            "column_gap": _distribution_digest(locality.get("column_gap_distribution", {})),
        },
        "reorder_proxy": {
            key: reorder.get(key)
            for key in (
                "adjacent_row_jaccard",
                "first_col_sorted_jaccard",
                "jaccard_gain",
                "bm16_unique_col_reduction",
            )
        },
        "hierarchical_candidates": {},
    }

    candidate_fields = (
        "blk_m",
        "blk_k",
        "num_row_windows",
        "total_tc_blocks",
        "unique_columns_total",
        "structural_density",
        "padding_ratio",
        "tensorcore_zero_fraction",
        "coo_padded_nnz",
        "coo_padding_ratio",
        "b_index_padding_ratio",
        "entries_per_unique_column",
        "top5_window_share",
        "coo_bytes_per_entry",
    )
    for key in REFERENCE_TILE_KEYS:
        value = _candidate_stats(features).get(key)
        if not value:
            continue
        compact = {field: value.get(field) for field in candidate_fields}
        compact["unique_cols_per_window"] = _distribution_digest(
            value.get("unique_cols_per_window", {})
        )
        compact["tc_blocks_per_window"] = _distribution_digest(
            value.get("tc_blocks_per_window", {})
        )
        compact["nnz_per_tc_block"] = _distribution_digest(
            value.get("structural_nnz_per_tc_block", {})
        )
        compact["storage_estimate_bytes"] = value.get("storage_estimate_bytes", {})
        digest["hierarchical_candidates"][key] = compact

    encoding_best: Dict[str, Dict[str, Any]] = {}
    for encoding in ("coo", "mask", "dense"):
        choices: List[Tuple[int, str]] = []
        for tile, stats in digest["hierarchical_candidates"].items():
            storage = stats.get("storage_estimate_bytes", {})
            try:
                choices.append((int(storage[encoding]), tile))
            except (KeyError, TypeError, ValueError):
                continue
        if choices:
            byte_count, tile = min(choices)
            encoding_best[encoding] = {"tile": tile, "bytes": byte_count}
    digest["derived_format_signals"] = {
        "minimum_estimated_storage": encoding_best,
        "candidate_metrics_scope": "exact_for_me_style_row_window_column_condensation",
    }
    return digest


def _flatten_digest_values(value: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten scalar digest leaves into exact evidence paths."""
    flattened: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_digest_values(child, child_prefix))
    elif value is not None and prefix:
        flattened[prefix] = value
    return flattened


def render_format_design_prompt(request: FormatDesignRequest) -> str:
    rules_text = "\n".join(f"{i + 1}. {rule}" for i, rule in enumerate(request.rules))
    digest = _feature_digest(request.matrix_features)
    evidence_map = _flatten_digest_values(digest)
    source_catalog = prompt_source_catalog()
    return f"""You are a sparse-format research architect for Sparsene on an SM80-class GPU.

Your task is to design the physical hierarchical sparse format that best fits
this matrix.  Do not reduce the task to choosing a tile for a fixed ME-TCF
template.  Backend implementation will be assessed separately after design.

{DSL_EDUCATION}

{FEATURE_SEMANTICS}

{EXECUTABLE_BACKEND_COST_MODEL}

=== DESIGN RULES ===
{rules_text}

=== FIXED EVALUATION CONTRACT (NOT MODEL-SELECTABLE) ===
This contract controls how alternatives must be ranked and will be persisted
verbatim with the decision artifact:
{json.dumps(KERNEL_ONLY_EVALUATION_CONTRACT, indent=2, ensure_ascii=False)}

=== SOURCE-DERIVED FORMAT KNOWLEDGE (REFERENCES, NOT A CLOSED MENU) ===
This catalog is derived from python/sparsene/formats.  Learn its legal hierarchy
grammar, physical semantics, atomic trade-offs, and caveats.  You may reuse,
remove, extend, or combine the patterns when feature evidence justifies it.
Never copy a capability label into a claim of execution readiness.
{json.dumps(source_catalog, sort_keys=True, ensure_ascii=False, separators=(",", ":"))}

=== MATRIX FEATURE PATH-VALUE MAP ===
Matrix label: {request.matrix_name}
Every `feature_evidence.feature` must be copied verbatim from a key below and
its `value` must equal the mapped value.  Do not combine two paths into one.
{json.dumps(evidence_map, indent=2, sort_keys=True, ensure_ascii=False)}

=== HARDWARE CONTEXT ===
{json.dumps(request.hardware, indent=2, sort_keys=True, ensure_ascii=False)}

=== PRIOR EMPIRICAL FEEDBACK (CALIBRATION, NOT A FORMAT TEMPLATE) ===
The following is a compact record of prior kernel-only measurements for this
same matrix at N=512, FP32 on an SM80-class GPU.  It is evidence about an
earlier implementation, not a constraint to copy its hierarchy or its format.
Use it to reject invalid causal claims (especially storage-estimate => runtime)
and to make a falsifiable design choice.  If a previous proposal was slower
than the dense sibling, identify a concrete changed mechanism in the new
hierarchy, metadata, atomic leaf, physical tile, or scheduling assumptions.
Do not claim a new DSL is faster merely because this feedback exists; describe
which later measurement would validate the hypothesis.
{json.dumps(request.empirical_feedback, indent=2, sort_keys=True, ensure_ascii=False)}

=== REQUIRED REASONING PROCEDURE ===
1. Diagnose the matrix: skew/load balance, row/column regularity, locality,
   row-window reuse, candidate tile occupancy, padding, and encoding storage.
2. Compare at least three complete format families or hierarchy/atomic
   combinations by expected timed-kernel behavior.  Do not compare only 16x8
   versus 16x16 and do not fold offline conversion/reorder latency into rank.
3. Select the grouping hierarchy first, then axis order/types, then optional
   reorder, then atomic encoding, then concrete parameters.
4. Check that every level contains ROW and COL axes and that the atomic axes
   exist in the deepest level with the correct directions.
5. Explain the implied transformation sequence and physical metadata/restore
   behavior.  State when candidate statistics are merely proxies.  Record
   conversion/reorder implementation requirements separately as untimed work.
6. List implementation requirements without allowing current backend support
   to bias the design.
7. Copy six to eight exact feature path/value pairs from the supplied map.
8. Use prior empirical feedback as a calibration record when it is present:
   name the observed bottleneck and one specific design mechanism intended to
   avoid it.  It is acceptable to retain a known format only when both feature
   evidence and this measured cost model support it.
9. Return one raw JSON object only.  Do not wrap it in markdown.

Before returning, perform this exact syntax audit:
- level values are JSON integers 1..L, not quoted strings;
- every length uses only M/K/N, positive integers, declared integer params,
  parentheses, and + - * /; no functions, prose, //, %, dots, or undeclared names;
- every level contains ROW and COL, and atomic axes exist in the deepest level;
- sv/dv realized cardinality is explained in purpose/semantics, while length is
  a legal logical bound;
- params are concrete positive JSON integers;
- selected_candidate is a supplied tile key or `not_applicable`;
- six to eight evidence records copy exact path/value pairs from the map;
- compare exactly three complete alternatives and keep every explanation concise.

The following is a TYPE-CORRECT JSON EXAMPLE, not a prescribed hierarchy and
not matrix evidence.  Its `level` value is deliberately an integer.  Add/remove
levels and choose different axes/atomics/params from the source catalog when
justified.  Replace every example explanation and evidence value.

=== OUTPUT JSON EXAMPLE ===
{json.dumps(request.output_schema, indent=2, ensure_ascii=False)}
"""


# ==============================================================================
# Deterministic DSL validation and building
# ==============================================================================


def _canonical_length(value: Any) -> str:
    return re.sub(r"\s+", "", str(value))


def _parse_length_expr(length: Any, allowed_symbols: Sequence[str]) -> Expr:
    text = str(length).strip()
    if not text:
        raise ValueError("length expression is empty")
    if not _LENGTH_EXPR_RE.fullmatch(text) or "**" in text or "//" in text:
        raise ValueError(f"unsafe or unsupported length expression: {text!r}")
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    unknown = sorted(names - set(allowed_symbols))
    if unknown:
        raise ValueError(f"undeclared symbols in length expression {text!r}: {unknown}")
    locals_map = {name: Symbol(name) for name in allowed_symbols}
    expr = sympify(text, locals=locals_map, evaluate=False)
    if not isinstance(expr, Expr):
        if isinstance(expr, int):
            return Integer(expr)
        raise ValueError(f"length expression is not symbolic/integer: {text!r}")
    return expr


def _axis_signature(axis: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        axis.get("name"),
        axis.get("type"),
        axis.get("direction"),
        _canonical_length(axis.get("length", "")),
        axis.get("reorder") or None,
    )


def _structure_signature(
    levels: Sequence[Dict[str, Any]], atomic: Dict[str, Any]
) -> Tuple[Any, ...]:
    level_signature = tuple(
        tuple(_axis_signature(axis) for axis in level.get("axes", []))
        for level in levels
    )
    atomic_signature = (
        atomic.get("type"),
        atomic.get("x_axis"),
        atomic.get("y_axis"),
        atomic.get("layout", "row_major"),
    )
    return level_signature, atomic_signature


def validate_format_dsl(
    *,
    format_name: Any,
    levels: Any,
    atomic: Any,
    params: Any,
) -> Dict[str, Any]:
    """Validate general Format DSL syntax/semantics without rewriting it."""
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(format_name, str) or not _FORMAT_NAME_RE.fullmatch(format_name):
        errors.append("format_name must be a valid identifier-like name")
    if not isinstance(params, dict):
        errors.append("params must be an object")
        params = {}
    param_names: List[str] = []
    for name, value in params.items():
        if not isinstance(name, str) or not _IDENTIFIER_RE.fullmatch(name):
            errors.append(f"invalid parameter name: {name!r}")
            continue
        param_names.append(name)
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"parameter {name} must be a concrete positive JSON integer")
        elif value <= 0:
            errors.append(f"parameter {name} must be positive")

    parameter_symbols = ["M", "K", "N", *param_names]

    if not isinstance(levels, list) or not 1 <= len(levels) <= 6:
        errors.append("levels must contain between one and six Format levels")
        levels = []

    allowed_symbols = parameter_symbols
    deepest_axes: Dict[str, Dict[str, Any]] = {}
    for index, level in enumerate(levels, start=1):
        if not isinstance(level, dict):
            errors.append(f"level {index} must be an object")
            continue
        if level.get("level") != index:
            errors.append(f"level numbering must be sequential; expected {index}")
        name = level.get("name")
        if not isinstance(name, str) or not _IDENTIFIER_RE.fullmatch(name):
            errors.append(f"level {index} has an invalid name")
        axes = level.get("axes")
        if not isinstance(axes, list) or not 2 <= len(axes) <= 8:
            errors.append(f"level {index} must contain between two and eight axes")
            continue
        names_seen: set[str] = set()
        directions: set[str] = set()
        level_axes: Dict[str, Dict[str, Any]] = {}
        for axis_index, axis in enumerate(axes, start=1):
            path = f"level {index} axis {axis_index}"
            if not isinstance(axis, dict):
                errors.append(f"{path} must be an object")
                continue
            axis_name = axis.get("name")
            if not isinstance(axis_name, str) or not _IDENTIFIER_RE.fullmatch(axis_name):
                errors.append(f"{path} has an invalid name")
                continue
            if axis_name in names_seen:
                errors.append(f"level {index} repeats axis name {axis_name}")
            names_seen.add(axis_name)
            axis_type = axis.get("type")
            if axis_type not in AXIS_TYPES:
                errors.append(f"{path} has unsupported type {axis_type!r}")
            direction = axis.get("direction")
            if direction not in ("ROW", "COL"):
                errors.append(f"{path} has unsupported direction {direction!r}")
            else:
                directions.add(direction)
            try:
                _parse_length_expr(axis.get("length"), allowed_symbols)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
            reorder = axis.get("reorder")
            if reorder is not None and (
                not isinstance(reorder, str) or not _REORDER_RE.fullmatch(reorder)
            ):
                errors.append(f"{path} has an invalid reorder identifier")
            level_axes[axis_name] = axis
        if directions != {"ROW", "COL"}:
            errors.append(f"level {index} must represent both ROW and COL directions")
        if index == len(levels):
            deepest_axes = level_axes

    if not isinstance(atomic, dict):
        errors.append("atomic must be an object")
        atomic = {}
    atomic_type = atomic.get("type")
    if atomic_type not in ATOMIC_TYPES:
        errors.append(f"unsupported atomic type: {atomic_type!r}")
    layout = atomic.get("layout", "row_major")
    if layout not in ("row_major", "col_major"):
        errors.append("atomic.layout must be row_major or col_major")
    if atomic_type == "custom":
        custom_spec = atomic.get("custom_spec")
        if not isinstance(custom_spec, dict):
            errors.append("custom atomic requires a custom_spec object")
        else:
            for field_name in ("value_arrays", "metadata_arrays"):
                value = custom_spec.get(field_name)
                if not isinstance(value, list) or not value:
                    errors.append(f"custom atomic requires non-empty custom_spec.{field_name}")
            for field_name in ("logical_mapping", "restore_algorithm"):
                if not str(custom_spec.get(field_name, "")).strip():
                    errors.append(f"custom atomic requires custom_spec.{field_name}")
    x_axis = atomic.get("x_axis")
    y_axis = atomic.get("y_axis")
    if x_axis not in deepest_axes:
        errors.append("atomic.x_axis must reference an axis in the deepest level")
    elif deepest_axes[x_axis].get("direction") != "ROW":
        errors.append("atomic.x_axis must reference a ROW axis")
    if y_axis not in deepest_axes:
        errors.append("atomic.y_axis must reference an axis in the deepest level")
    elif deepest_axes[y_axis].get("direction") != "COL":
        errors.append("atomic.y_axis must reference a COL axis")
    if x_axis == y_axis and x_axis is not None:
        errors.append("atomic.x_axis and atomic.y_axis must be different")

    return {
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "warnings": warnings,
        "level_count": len(levels),
        "axis_count": sum(
            len(level.get("axes", []))
            for level in levels
            if isinstance(level, dict) and isinstance(level.get("axes"), list)
        ),
        "atomic_type": atomic_type,
    }


def _build_axis(axis: Dict[str, Any], allowed_symbols: Sequence[str]) -> Axis:
    direction = ROW if axis["direction"] == "ROW" else COL
    length = _parse_length_expr(axis["length"], allowed_symbols)
    reorder = axis.get("reorder")
    constructors = {"df": df_axis, "dv": dv_axis, "sf": sf_axis, "sv": sv_axis}
    return constructors[axis["type"]](axis["name"], direction, length, reorder)


def _build_atomic(atomic: Dict[str, Any]) -> AtomicFormat:
    if atomic["type"] == "custom":
        col_major = atomic.get("layout", "row_major") == "col_major"
        return AtomicFormat(
            name=str(atomic.get("name", "custom")),
            atomic_type="custom",
            axes=[
                Axis(
                    name=atomic["x_axis"],
                    direction=COL if col_major else ROW,
                    is_sparse=False,
                    is_varlen=False,
                ),
                Axis(
                    name=atomic["y_axis"],
                    direction=ROW if col_major else COL,
                    is_sparse=False,
                    is_varlen=False,
                ),
            ],
        )
    return atomic_format(
        atomic["type"],
        atomic["x_axis"],
        atomic["y_axis"],
        col_major=atomic.get("layout", "row_major") == "col_major",
    )


def build_format_from_response(
    response: FormatDesignResponse, *, require_integrated: bool = False
) -> Format:
    """Build a frontend Format from any deterministically valid research DSL.

    `require_integrated=True` is reserved for the current DTC end-to-end route.
    It checks capability rather than mutating a research proposal.
    """
    validation = validate_format_dsl(
        format_name=response.format_name,
        levels=response.levels,
        atomic=response.atomic,
        params=response.params,
    )
    if validation["status"] != "valid":
        raise ValueError("Invalid Format DSL: " + "; ".join(validation["errors"]))
    if require_integrated and response.capability.get("support_level") != "integrated_e2e":
        raise ValueError(
            "Format is a valid research design but is not supported by the integrated e2e route: "
            f"{response.capability.get('support_level', 'unknown')}"
        )

    allowed_symbols = ["M", "K", "N", *response.params.keys()]
    current: Format | AtomicFormat = _build_atomic(response.atomic)
    for level in reversed(response.levels):
        axes = [_build_axis(axis, allowed_symbols) for axis in level["axes"]]
        current = Format(name=level["name"], axes=axes, child=current)
    if not isinstance(current, Format):  # pragma: no cover - validated levels are non-empty
        raise ValueError("Top-level design must be a Format")
    return current


def _match_reference_profile(
    levels: Sequence[Dict[str, Any]], atomic: Dict[str, Any]
) -> Optional[str]:
    signature = _structure_signature(levels, atomic)
    for name, spec in REFERENCE_PROFILES.items():
        if signature == _structure_signature(spec["levels"], spec["atomic"]):
            return name
    return None


def assess_backend_capability(
    *,
    format_name: str,
    levels: List[Dict[str, Any]],
    atomic: Dict[str, Any],
    params: Dict[str, Any],
    selected_candidate: str,
    validation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    validation = validation or validate_format_dsl(
        format_name=format_name, levels=levels, atomic=atomic, params=params
    )
    if validation["status"] != "valid":
        return {
            "support_level": "invalid",
            "matched_profile": None,
            "backend_format_name": None,
            "format_object_buildable": False,
            "rts_derivable": False,
            "execution_ready": False,
            "reasons": list(validation["errors"]),
            "required_extensions": ["repair the Format DSL before backend assessment"],
        }

    temporary = FormatDesignResponse(
        request_id="capability-check",
        format_name=format_name,
        levels=deepcopy(levels),
        atomic=deepcopy(atomic),
        params=deepcopy(params),
        rationale="",
        confidence="low",
    )
    try:
        fmt = build_format_from_response(temporary)
    except Exception as exc:
        return {
            "support_level": "frontend_research",
            "matched_profile": None,
            "backend_format_name": None,
            "format_object_buildable": False,
            "rts_derivable": False,
            "execution_ready": False,
            "reasons": [f"frontend build/RTS derivation failed: {exc}"],
            "required_extensions": ["extend or repair Format-to-RTS lowering for this hierarchy"],
        }

    try:
        rts = derive_rts(fmt)
        rts_derivable = bool(rts.sequence)
    except Exception as exc:
        return {
            "support_level": "frontend_research",
            "matched_profile": None,
            "backend_format_name": None,
            "format_object_buildable": True,
            "rts_derivable": False,
            "execution_ready": False,
            "reasons": [f"Format object is valid but RTS derivation needs extension: {exc}"],
            "required_extensions": [
                "implement atomicization/restore semantics in Format-to-RTS and downstream IR",
                "define physical metadata arrays and host preprocessing",
                "add NVIR implementation, host ABI, cache identity, and numerical tests",
            ],
        }

    # Match executable backends by physical semantics, not by the descriptive
    # format name or exact level-name spelling.  The TCF-MCO v1 route includes
    # offline preprocessing, optional row_rcm, SM80 codegen, host ABI,
    # correctness, and kernel-only cuSPARSE comparison.
    try:
        backend_plan = select_backend(
            {
                "format_name": format_name,
                "levels": levels,
                "atomic": atomic,
                "params": params,
            },
            target="sm80",
        )
    except ValueError:
        backend_plan = None
    if backend_plan is not None:
        return {
            "support_level": "integrated_e2e",
            "matched_profile": "TCF_MCO_TENSORCORE_V1",
            "backend_format_name": backend_plan.codegen_format_name,
            "format_object_buildable": True,
            "rts_derivable": rts_derivable,
            "execution_ready": True,
            "reasons": [
                "semantic hierarchy matches the generalized two-level TCF-MCO SM80 backend",
                "host preprocessing implements optional row_rcm and logical-to-physical padding",
            ],
            "required_extensions": [],
            "backend_plan": backend_plan.to_dict(),
        }

    matched = _match_reference_profile(levels, atomic)
    if matched is None:
        support = "frontend_research"
        reasons = [
            "DSL is valid and reaches Format-to-RTS, but no exact backend profile matches its hierarchy/atomic contract."
        ]
        extensions = [
            "define physical metadata arrays and preprocessing",
            "validate RTS-to-Computent and OPIR operations",
            "add NVIR layouts/implementations and semantic fusion",
            "add host ABI, dispatch, cache identity, and numerical tests",
        ]
    else:
        support = REFERENCE_PROFILES[matched]["support"]
        reasons = [f"structure matches repository profile {matched}"]
        extensions = []
        candidate_matches_params = True
        if selected_candidate in REFERENCE_TILE_KEYS:
            tile_m, tile_k = (int(part) for part in selected_candidate.split("x", maxsplit=1))
            for param_name, tile_value in (("BLK_M", tile_m), ("BLK_K", tile_k)):
                value = params.get(param_name)
                if isinstance(value, int) and value != tile_value:
                    candidate_matches_params = False
        if support == "integrated_e2e" and (
            selected_candidate not in INTEGRATED_TILE_KEYS or not candidate_matches_params
        ):
            support = "standalone_validated"
            reasons.append(
                f"the integrated DTC route requires a consistent tile in {list(INTEGRATED_TILE_KEYS)}"
            )
            extensions.append("validate and wire the selected tile through the integrated DTC host/codegen ABI")
        elif support == "standalone_validated":
            extensions.extend(
                [
                    "add unified e2e host/preprocessing dispatch for this profile",
                    "validate the selected parameters on target matrices",
                ]
            )
        elif support == "codegen_declared":
            extensions.extend(
                [
                    "complete/verify semantic fusion and generated-kernel lowering",
                    "add matching host preprocessing and numerical regression",
                ]
            )

    return {
        "support_level": support,
        "matched_profile": matched,
        "backend_format_name": matched,
        "format_object_buildable": True,
        "rts_derivable": rts_derivable,
        "execution_ready": support == "integrated_e2e",
        "reasons": reasons,
        "required_extensions": extensions,
    }


# ==============================================================================
# Response normalization (non-structural) and evidence grounding
# ==============================================================================


def _resolve_digest_path(digest: Dict[str, Any], path: str) -> Any:
    current: Any = digest
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return current


def _values_match(expected: Any, observed: Any) -> bool:
    try:
        return abs(float(expected) - float(observed)) <= 1e-9 * max(
            1.0, abs(float(expected))
        )
    except (TypeError, ValueError):
        return expected is not None and str(expected) == str(observed)


def _clean_string_list(value: Any, field_name: str, actions: List[str], limit: int = 12) -> List[str]:
    if not isinstance(value, list):
        actions.append(f"discarded malformed {field_name}")
        return []
    return [str(item).strip() for item in value[:limit] if str(item).strip()]


def _clean_dict_list(value: Any, field_name: str, actions: List[str], limit: int = 12) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        actions.append(f"discarded malformed {field_name}")
        return []
    cleaned = [deepcopy(item) for item in value[:limit] if isinstance(item, dict)]
    if len(cleaned) != min(len(value), limit):
        actions.append(f"discarded malformed entries from {field_name}")
    return cleaned


def _canonicalize_json_scalar_syntax(
    levels: List[Dict[str, Any]], params: Dict[str, Any], actions: List[str]
) -> None:
    """Repair JSON scalar typing without changing hierarchy or expressions."""
    for level in levels:
        if not isinstance(level, dict):
            continue
        raw_level = level.get("level")
        if isinstance(raw_level, str) and raw_level.strip().isdigit():
            level["level"] = int(raw_level.strip())
            actions.append(
                f"canonicalized numeric string level {raw_level!r} to JSON integer {level['level']}"
            )
    for name, value in list(params.items()):
        if isinstance(value, str) and value.strip().isdigit():
            params[name] = int(value.strip())
            actions.append(
                f"canonicalized numeric string parameter {name}={value!r} to JSON integer"
            )


def normalize_format_payload(
    payload: Dict[str, Any], matrix_features: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate an untrusted response without replacing its format structure.

    Only malformed explanatory/evidence fields are filtered.  `levels`,
    `atomic`, and `params` are preserved as returned when they have the expected
    container type; structural problems are reported in `dsl_validation`.
    """
    actions: List[str] = []
    format_name = payload.get("format_name", "")
    levels = deepcopy(payload.get("levels")) if isinstance(payload.get("levels"), list) else []
    atomic = deepcopy(payload.get("atomic")) if isinstance(payload.get("atomic"), dict) else {}
    params = deepcopy(payload.get("params")) if isinstance(payload.get("params"), dict) else {}
    if not isinstance(payload.get("levels"), list):
        actions.append("replaced malformed levels container with an empty invalid design")
    if not isinstance(payload.get("atomic"), dict):
        actions.append("replaced malformed atomic container with an empty invalid descriptor")
    if not isinstance(payload.get("params"), dict):
        actions.append("replaced malformed params container with an empty object")
    _canonicalize_json_scalar_syntax(levels, params, actions)

    selected = str(payload.get("selected_candidate", "not_applicable"))
    available_candidates = _candidate_stats(matrix_features)
    if selected != "not_applicable" and selected not in available_candidates:
        actions.append("replaced unavailable selected_candidate with not_applicable")
        selected = "not_applicable"

    digest = _feature_digest(matrix_features)
    raw_evidence = payload.get("feature_evidence", [])
    evidence: List[Dict[str, Any]] = []
    if not isinstance(raw_evidence, list):
        actions.append("discarded malformed feature_evidence")
    else:
        for item in raw_evidence[:12]:
            if not isinstance(item, dict) or not all(
                key in item for key in ("feature", "value", "implication")
            ):
                actions.append("discarded malformed feature evidence entry")
                continue
            expected = _resolve_digest_path(digest, str(item["feature"]))
            if not _values_match(expected, item["value"]):
                actions.append(
                    f"discarded ungrounded feature evidence: {item.get('feature')}"
                )
                continue
            evidence.append(deepcopy(item))

    transformation_plan = _clean_dict_list(
        payload.get("transformation_plan", []), "transformation_plan", actions
    )
    alternatives = _clean_dict_list(
        payload.get("alternatives_considered", []), "alternatives_considered", actions
    )
    tradeoffs = _clean_string_list(payload.get("tradeoffs", []), "tradeoffs", actions)
    basis_cn = _clean_string_list(
        payload.get("decision_basis_cn", []), "decision_basis_cn", actions
    )
    semantics_cn = _clean_string_list(
        payload.get("format_semantics_cn", []), "format_semantics_cn", actions
    )
    implementation_requirements = _clean_string_list(
        payload.get("implementation_requirements", []),
        "implementation_requirements",
        actions,
    )
    risks_cn = _clean_string_list(payload.get("risks_cn", []), "risks_cn", actions)

    validation = validate_format_dsl(
        format_name=format_name,
        levels=levels,
        atomic=atomic,
        params=params,
    )
    try:
        invalid_entries = int(
            matrix_features.get("data_quality", {}).get("invalid_entries", 0)
        )
    except (TypeError, ValueError):
        invalid_entries = -1
    if invalid_entries != 0:
        validation["warnings"].append(
            "matrix data_quality.invalid_entries is non-zero or malformed; design evidence may be unreliable"
        )
    capability = assess_backend_capability(
        format_name=str(format_name),
        levels=levels,
        atomic=atomic,
        params=params,
        selected_candidate=selected,
        validation=validation,
    )

    deterministic_requirements = capability.get("required_extensions", [])
    for requirement in deterministic_requirements:
        if requirement not in implementation_requirements:
            implementation_requirements.append(requirement)

    confidence = str(payload.get("confidence", "low")).lower()
    if confidence not in {"high", "medium", "low"}:
        actions.append("normalized confidence to low")
        confidence = "low"
    if validation["status"] != "valid" or invalid_entries != 0:
        confidence = "low"
    elif len(evidence) < 6 and confidence == "high":
        actions.append("lowered confidence because fewer than six feature values are grounded")
        confidence = "medium"

    executable_profile = EXECUTABLE_PROFILE if capability.get("support_level") == "integrated_e2e" else "none"

    return {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "design_profile": DESIGN_PROFILE,
        "format_name": str(format_name),
        "format_family": str(payload.get("format_family", "custom")),
        "design_summary": str(payload.get("design_summary", "")).strip(),
        "levels": levels,
        "atomic": atomic,
        "params": params,
        "selected_candidate": selected,
        "transformation_plan": transformation_plan,
        "feature_evidence": evidence,
        "alternatives_considered": alternatives,
        "tradeoffs": tradeoffs,
        "rationale": str(payload.get("rationale", "")).strip(),
        "rationale_cn": str(payload.get("rationale_cn", "")).strip(),
        "decision_basis_cn": basis_cn,
        "format_semantics_cn": semantics_cn,
        "selection_reason_cn": str(payload.get("selection_reason_cn", "")).strip(),
        "implementation_requirements": implementation_requirements,
        "risks_cn": risks_cn,
        "confidence": confidence,
        "dsl_validation": validation,
        "capability": capability,
        "executable_profile": executable_profile,
        "evaluation_contract": deepcopy(KERNEL_ONLY_EVALUATION_CONTRACT),
        "normalization_actions": actions,
    }


# ==============================================================================
# JSON parsing, fallback, and runner
# ==============================================================================


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def _fallback_format_response(
    *,
    request_id: str,
    rationale: str,
    raw_payload: Optional[Dict[str, Any]] = None,
    token_usage: Optional[Dict[str, Any]] = None,
    agent_metadata: Optional[Dict[str, Any]] = None,
) -> FormatDesignResponse:
    """Represent model failure explicitly; never disguise it as ME-TCF."""
    validation = validate_format_dsl(
        format_name="FORMAT_DESIGN_FAILED", levels=[], atomic={}, params={}
    )
    capability = assess_backend_capability(
        format_name="FORMAT_DESIGN_FAILED",
        levels=[],
        atomic={},
        params={},
        selected_candidate="not_applicable",
        validation=validation,
    )
    return FormatDesignResponse(
        request_id=request_id,
        format_name="FORMAT_DESIGN_FAILED",
        format_family="failure",
        design_summary="No valid design was produced.",
        levels=[],
        atomic={},
        params={},
        transformation_plan=[],
        selected_candidate="not_applicable",
        rationale=rationale,
        rationale_cn=f"格式设计智能体未产生合法结果：{rationale}",
        confidence="low",
        selection_reason_cn="没有可接受的格式设计。",
        implementation_requirements=["obtain a valid model response before implementation"],
        risks_cn=["当前记录是失败状态，不能进入格式构建或算子生成。"],
        dsl_validation=validation,
        capability=capability,
        normalization_actions=["recorded explicit format-design failure; no ME-TCF fallback applied"],
        raw_payload=raw_payload or {},
        token_usage=token_usage or make_zero_usage().to_dict(),
        agent_metadata={"source": "fallback", **(agent_metadata or {})},
    )


def run_format_design_agent(
    matrix_name: str,
    matrix_features: Dict[str, Any],
    config: Optional[FormatDesignConfig] = None,
    empirical_feedback: Optional[Dict[str, Any]] = None,
) -> FormatDesignResponse:
    if config is None:
        config = FormatDesignConfig()

    hardware = discover_hardware_profile()
    hardware_dict = {
        "gpu_name": hardware.gpu_name,
        "cuda_arch": hardware.cuda_arch,
        "driver_version": hardware.driver_version,
        "total_memory_mb": hardware.total_memory_mb,
        "warp_size": hardware.warp_size,
        "design_note": "Hardware guides alignment/cost reasoning but does not restrict the research DSL to current backend profiles.",
    }

    request_id = str(uuid.uuid4())
    request = FormatDesignRequest(
        request_id=request_id,
        matrix_name=matrix_name,
        matrix_features=matrix_features,
        hardware=hardware_dict,
        output_schema=_default_output_schema(),
        constraints=[
            "Return a structurally valid general Format DSL proposal.",
            "Separate design quality from current backend capability.",
        ],
        rules=FORMAT_DESIGN_RULES,
        empirical_feedback=deepcopy(empirical_feedback or {}),
    )
    request.prompt = render_format_design_prompt(request)

    try:
        result = invoke_json_agent(
            agent_name="format_design",
            subject=matrix_name,
            log_dir=config.log_dir,
            model_config=AgentModelConfig(
                provider="openai_compatible",
                model_name=config.model_name or config.model,
                model=config.model,
                api_key=config.api_key,
                api_base_url=config.api_base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout_sec=config.timeout_sec,
                model_config_path=config.model_config_path,
            ),
            system_prompt=(
                "You are a sparse-format research architect. Design a feature-grounded "
                "hierarchical Format DSL, not a parameter choice inside a fixed format. "
                "Rank designs by warmed GPU SpMM compute-kernel time after all format "
                "conversion and reordering; those offline transformations are excluded "
                "from the primary metric and must not be used as a ranking penalty. "
                "Existing formats are references rather than a closed menu. Preserve exact "
                "measurements, return one JSON object, and never claim unsupported backend "
                "capabilities; implementation readiness is classified by deterministic code."
            ),
            user_prompt=request.prompt,
            request_payload=asdict(request),
            response_parser=_try_parse_json,
            skill_names=config.skill_names,
            json_output=True,
        )
    except Exception as exc:
        logger.error("Format design agent HTTP error: %s", exc)
        return _fallback_format_response(
            request_id=request_id,
            rationale=f"Agent call failed: {exc}",
        )

    if result.finish_reason == "length":
        logger.warning("Format design agent response was truncated at max_tokens")
        return _fallback_format_response(
            request_id=request_id,
            rationale="Model response was truncated (finish_reason=length)",
            raw_payload={"raw_text": result.response_text[:2000]},
            token_usage=result.usage.to_dict(),
            agent_metadata={
                "run_dir": str(result.run_dir),
                "resolved_model": result.resolved_model.to_dict(),
                "finish_reason": result.finish_reason,
            },
        )

    parsed = result.parsed_payload or {}
    if not parsed:
        logger.warning("Format design agent JSON parse failed")
        return _fallback_format_response(
            request_id=request_id,
            rationale="JSON parse failed",
            raw_payload={"raw_text": result.response_text[:2000]},
            token_usage=result.usage.to_dict(),
            agent_metadata={
                "run_dir": str(result.run_dir),
                "resolved_model": result.resolved_model.to_dict(),
                "finish_reason": result.finish_reason,
            },
        )

    normalized = normalize_format_payload(parsed, matrix_features)
    if normalized["normalization_actions"]:
        logger.warning(
            "Format design response required non-structural cleanup: %s",
            "; ".join(normalized["normalization_actions"]),
        )
    if normalized["dsl_validation"]["status"] != "valid":
        logger.warning(
            "Format design DSL is invalid: %s",
            "; ".join(normalized["dsl_validation"]["errors"]),
        )

    return FormatDesignResponse(
        request_id=request_id,
        raw_payload=parsed,
        token_usage=result.usage.to_dict(),
        agent_metadata={
            "run_dir": str(result.run_dir),
            "resolved_model": result.resolved_model.to_dict(),
            "design_profile": DESIGN_PROFILE,
            "support_level": normalized["capability"]["support_level"],
            "finish_reason": result.finish_reason,
        },
        **normalized,
    )
