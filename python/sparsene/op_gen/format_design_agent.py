"""
Format-design agent for sparsene — V3: Full DSL-based format design.

The agent is taught the complete sparsene hierarchical sparse format DSL and
can design new formats from scratch by composing:
  - Axis types (df/dv/sf/sv)
  - Format nesting levels
  - Atomic format (coo/mco/dense)
  - Split sizes (BLK_M, BLK_K, etc.)

Strategy decisions (tensor placement / pipeline) are NOT affected;
they remain heuristic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request
import uuid

from sparsene.format.format import (
    Axis,
    AtomicFormat,
    CooAtomicFormat,
    DenseAtomicFormat,
    Direction,
    Format,
    McoAtomicFormat,
    atomic_format,
    coo_atomic_format,
    dense_atomic_format,
    df_axis,
    dv_axis,
    mco_atomic_format,
    sf_axis,
    sv_axis,
    Expr,
    Symbol,
)
from sparsene.logging import get_logger
from sparsene.op_gen.strategy_agent import discover_hardware_profile

logger = get_logger(__name__)

DEFAULT_DEEPSEEK_API_KEY = "sk-eb7f7960366d41a5a4310277168e93da"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_MAX_TOKENS = 8192

ROW = Direction.ROW
COL = Direction.COL

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
    raw_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormatDesignConfig:
    api_key: str = DEFAULT_DEEPSEEK_API_KEY
    api_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    model: str = DEFAULT_DEEPSEEK_MODEL
    log_dir: str = "docs/agent_runs"
    timeout_sec: int = 180
    temperature: float = 0.1
    max_tokens: int = DEFAULT_MAX_TOKENS


# ==============================================================================
# DSL EDUCATION — teaches the LLM the full sparsene format DSL
# ==============================================================================

DSL_EDUCATION = r"""
## THE SPARSENE HIERARCHICAL SPARSE FORMAT DSL

You are designing a sparse format using sparsene's hierarchical DSL.
This DSL lets you compose arbitrary multi-level sparse formats.

### PRIMITIVE 1: Axis (轴)

An axis represents a dimension in the format.  4 types:

  | Type | Dense? | Fixed-length? | Factory      | When to use |
  |------|--------|---------------|--------------|-------------|
  | df   | yes    | yes           | df_axis()    | Block/padding dimensions, split counts |
  | dv   | yes    | no (varlen)   | dv_axis()    | Rare: dense but varying per instance |
  | sf   | sparse | yes           | sf_axis()    | Fixed-nnz-per-segment sparse |
  | sv   | sparse | no (varlen)   | sv_axis()    | **Variable-length sparse** — absorbs row-length variance |

Each axis has:
  - name: str         — unique identifier (e.g. "X_o", "Y_s", "X_i")
  - direction: ROW|COL
  - length: expression — can be a symbol like "M", "K", "BLK_M",
                          or a sympy expression like Symbol("K")/Symbol("BLK_K")

**CRITICAL**: The varlen K axis (sv_axis on COL direction) is the key mechanism
for handling rows with different NNZ counts.  If row_nnz CV > 0.3, you MUST
include an sv_axis on the COL direction named "Y_s" with length="K".

### PRIMITIVE 2: Format (层次)

A Format is a named grouping of axes, optionally with a child Format:

  Format(name="MyLevel", axes=[axis1, axis2, ...], child=Format(...))

You can nest an arbitrary number of Format levels.  Each level adds a
dimension of blocking/decomposition.  Typical patterns:

  Pattern A: 2-level (like ME_TCF / BIT_TCF)
    Level 1 (outer):  [df_axis("X_o",ROW,"M"), sv_axis("Y_s",COL,"K"), df_axis("X_i",ROW,"BLK_M")]
    Level 2 (inner):  [df_axis("Y_so",COL,"K/BLK_K"), df_axis("Y_si",COL,"BLK_K"), df_axis("X_i",ROW,"BLK_M")]
    Level 3 (atomic): coo_atomic_format("X_i","Y_si")

  Pattern B: 1-level + atomic (simpler, when matrix is small/regular)
    Level 1 (only): [df_axis("X_o",ROW,"M"), sv_axis("Y_s",COL,"K")]
    Atomic:          coo_atomic_format("X_o","Y_s")

### PRIMITIVE 3: Atomic Format (原子格式)

The leaf node. 3 types:

  coo_atomic_format(x_axis_name, y_axis_name):
    - COOrdinate format.  Stores (row, col, val) per non-zero.
    - Best when per-block density is 5-50%.
    - Overhead: 2 ints + 1 float per NZ.

  mco_atomic_format(x_axis_name, y_axis_name):
    - Masked-COO.  Uses bitmask + compact val array.
    - Best when per-block density < 5% (extremely sparse).
    - Overhead: bitmask bytes + val array per block.

  dense_atomic_format(x_axis_name, y_axis_name):
    - Full dense tile.  No sparsity within a TC block.
    - Best when padding < 30% (high fill rate).
    - Implicit zero-padding.

### REFERENCE: Existing formats (you CAN deviate from these)

  ME_TCF:
    Outer: df(X_o,ROW,M) + sv(Y_s,COL,K) + df(X_i,ROW,BLK_M)
    Inner: df(Y_so,COL,K/BLK_K) + df(Y_si,COL,BLK_K) + df(X_i,ROW,BLK_M)
    Atom:  coo(X_i, Y_si)

  BIT_TCF: same structure, atomic=mco(X_i, Y_si)
  SR_BCRS: same structure, atomic=dense(X_i, Y_si)

You can innovate by:
  - Adding a 3rd Format level (e.g. sub-window splitting)
  - Changing axis types (e.g. using sf_axis for fixed-sparsity segments)
  - Using different split sizes at different levels
  - Choosing BLK_M/BLK_K based on matrix statistics
"""


# ==============================================================================
# Design rules (injected into prompt)
# ==============================================================================

FORMAT_DESIGN_RULES = [
    "CURRENT CODGEN CONSTRAINT: The codegen pipeline only supports the standard "
    "2-level ME_TCF structure. You MUST use exactly these axes and nesting: "
    "  Level 1 (outer):  df(X_o,ROW,M) + sv(Y_s,COL,K) + df(X_i,ROW,BLK_M) "
    "  Level 2 (inner):  df(Y_so,COL,K/BLK_K) + df(Y_si,COL,BLK_K) + df(X_i,ROW,BLK_M) "
    "  Atomic:            coo(X_i,Y_si) or mco(X_i,Y_si) or dense(X_i,Y_si) "
    "You CAN and SHOULD optimize: BLK_M ∈ {8,16,32}, BLK_K ∈ {8,16}, and atomic type. "
    "Do NOT change axis names, directions, level count, or axis types.",

    "BLK_M: row tile size.  Use smaller (8) for short rows (<16 avg NNZ/row), "
    "medium (16) for 16-32 avg NNZ/row, larger (32) for >32 avg NNZ/row. "
    "Smaller BLK_M improves load balance for irregular matrices.",

    "BLK_K: col tile size.  Use 16 for better TC utilization, 8 to reduce "
    "padding.  If padding > 60% at 16, prefer 8.  If density_p50 at 16 < 3%, "
    "8 may offer slightly higher density.",

    "Atomic format selection by density percentile: "
    "  density_p50 < 5%     → mco (bitmask more compact than int indices) "
    "  density_p50 5-30%    → coo (balanced) "
    "  density_p50 > 30%    → dense (padding acceptable, no indirection) "
    "  density_p90 > 50%    → dense may be viable even if median is low "
    "  Wide gap p50–p90     → prefer coo for flexibility",

    "Format depth (number of levels): "
    "  Use 2-level (outer row-window + inner TC-block) for most cases. "
    "  Consider 3-level if M or K is huge (>500K) — adds a sub-window layer. "
    "  Use 1-level only for very small matrices (<1000 rows).",

    "Varlen axis rule: "
    "  If row_nnz CV > 0.3, you MUST include sv_axis(\"Y_s\", COL, \"K\") "
    "  in the outermost Format level.  This is non-negotiable for correctness.",

    "Load balance: "
    "  TC blocks CV < 0.5  → balanced, BLK_M can be larger (32) "
    "  TC blocks CV 0.5-1.0 → moderate, BLK_M = 16 "
    "  TC blocks CV > 1.0  → severe imbalance, BLK_M = 8 "
    "  top5_window_share > 40% → extreme skew, consider 3-level nesting",

    "Row correlation: "
    "  |row_corr| > 0.7 → strong correlation, consider row reorder or smaller BLK_M "
    "  Positive → later rows denser; Negative → earlier rows denser. "
    "  If reorder is beneficial, set reorder: true in the response.",

    "Symmetry: "
    "  If matrix is symmetric, column stats mirror row stats. "
    "  The sv_axis still goes on the COL direction (K axis).",

    "Safety: "
    "  All BLK_* values must be powers of 2 compatible with MMA tiling (8, 16, 32). "
    "  Atomic format's x_name and y_name must match axes from parent formats. "
    "  The sv_axis \"Y_s\" must appear exactly once in the outermost level.",
]


# ==============================================================================
# Output schema — the LLM describes the full format structure
# ==============================================================================

def _default_output_schema() -> Dict[str, Any]:
    return {
        "format_name": "string (a descriptive name for this new format)",
        "levels": [
            {
                "level": 1,
                "name": "string (e.g. 'row_window')",
                "axes": [
                    {
                        "name": "string (e.g. 'X_o')",
                        "type": "df | dv | sf | sv",
                        "direction": "ROW | COL",
                        "length": "string expression (e.g. 'M', 'K', 'BLK_M', 'K/BLK_K')",
                    }
                ],
            },
            {
                "level": 2,
                "name": "string (e.g. 'tc_block')",
                "axes": [
                    {
                        "name": "string",
                        "type": "df | dv | sf | sv",
                        "direction": "ROW | COL",
                        "length": "string expression",
                    }
                ],
            },
        ],
        "atomic": {
            "type": "coo | mco | dense",
            "x_axis": "string (axis name for rows, must exist in parent)",
            "y_axis": "string (axis name for cols, must exist in parent)",
        },
        "params": {
            "BLK_M": "int (8|16|32)",
            "BLK_K": "int (8|16)",
        },
        "reorder": "bool (whether to enable row reordering)",
        "rationale": "string (3-5 sentences explaining the design)",
        "confidence": "high | medium | low",
    }


# ==============================================================================
# Prompt rendering
# ==============================================================================


def render_format_design_prompt(request: FormatDesignRequest) -> str:
    feats = request.matrix_features
    shape = feats.get("shape", {})
    row_dist = feats.get("row_nnz_distribution", {})
    col_dist = feats.get("col_nnz_distribution", {})
    blk_stats = feats.get("candidate_block_stats", {})
    lb = feats.get("load_balance", {})

    rules_text = "\n".join(f"  {i+1}. {rule}" for i, rule in enumerate(request.rules))

    blk_lines = []
    for key, bs in blk_stats.items():
        blk_lines.append(
            f"  {key}: tc_blocks={bs['total_tc_blocks']}, "
            f"padding={bs['padding_ratio']}, "
            f"avg_density={bs['avg_density']}, "
            f"tc_cv={bs.get('tc_blocks_per_window',{}).get('cv','?')}, "
            f"density_p50={bs.get('density_p50_pct','?')}%, "
            f"density_p75={bs.get('density_p75_pct','?')}%, "
            f"density_p90={bs.get('density_p90_pct','?')}%"
        )

    return f"""You are a hierarchical sparse format design expert for GPU (SM80/A100).

{DSL_EDUCATION}

=== DESIGN RULES ===
{rules_text}

=== MATRIX FEATURES ===
Matrix: {request.matrix_name}
Shape: M={shape.get('M')}, K={shape.get('K')}, NNZ={shape.get('nnz')}
Sparsity: {feats.get('sparsity')}
Symmetry: {feats.get('symmetry')}

Row NNZ distribution:
  mean={row_dist.get('mean')}, std={row_dist.get('std')},
  min={row_dist.get('min')}, max={row_dist.get('max')},
  p25={row_dist.get('p25')}, p50={row_dist.get('p50')}, p75={row_dist.get('p75')},
  CV={row_dist.get('cv')}

Column NNZ distribution:
  mean={col_dist.get('mean')}, std={col_dist.get('std')},
  min={col_dist.get('min')}, max={col_dist.get('max')},
  p25={col_dist.get('p25')}, p50={col_dist.get('p50')}, p75={col_dist.get('p75')},
  CV={col_dist.get('cv')}

Row length vs row index correlation: {feats.get('row_length_vs_index_correlation')}

Candidate block statistics (for reference when picking BLK sizes):
{chr(10).join(blk_lines)}

Load balance:
  TC blocks CV (coefficient of variation): {lb.get('tc_blocks_cv')}
  Top-5 window share: {lb.get('top5_window_share')}

=== HARDWARE ===
{json.dumps(request.hardware, indent=2)}

=== TASK ===
Design a NEW hierarchical sparse format for this matrix.
Use the sparsene DSL to compose axes, Format levels, and atomic format.
You are NOT limited to existing format names — you design the full structure.

Steps:
1. Decide format depth (1, 2, or 3 levels) based on matrix size and irregularity.
2. For each level, choose axis types (df/dv/sf/sv) and their names/directions/lengths.
   The outermost level MUST contain sv_axis("Y_s",COL,"K") if row_nnz CV > 0.3.
3. Choose atomic format (coo/mco/dense) based on density percentiles.
4. Set BLK_M, BLK_K based on average NNZ/row and load balance.
5. Decide whether row reordering helps (reorder: true/false).
6. Provide a rationale.

Output ONLY the structured JSON matching the schema below.
Do NOT wrap in markdown code fences.  Output raw JSON.

=== OUTPUT SCHEMA ===
{json.dumps(request.output_schema, indent=2)}
"""


# ==============================================================================
# FormatBuilder — constructs Format objects from LLM JSON descriptions
# ==============================================================================

def _parse_length_expr(length_str: str) -> Expr:
    """Parse a length expression string to a sympy expression.

    Supports:
      - plain int or string symbol: "M", "K", "BLK_M", "8", "16", "32"
      - division: "K/BLK_K", "M/BLK_M"
    """
    s = length_str.strip()
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            left = _parse_length_expr(parts[0])
            right = _parse_length_expr(parts[1])
            return left / right  # type: ignore[operator]
    try:
        return Symbol(s)
    except Exception:
        return Symbol(s)


def _build_axis(ax: Dict[str, Any]) -> Axis:
    axis_type = ax.get("type", "df")
    name = ax.get("name", "?")
    direction = ROW if ax.get("direction", "ROW") == "ROW" else COL
    length = _parse_length_expr(ax.get("length", "_"))
    reorder = ax.get("reorder", None)

    if axis_type == "df":
        return df_axis(name, direction, length, reorder)
    elif axis_type == "dv":
        return dv_axis(name, direction, length, reorder)
    elif axis_type == "sf":
        return sf_axis(name, direction, length, reorder)
    elif axis_type == "sv":
        return sv_axis(name, direction, length, reorder)
    else:
        return df_axis(name, direction, length, reorder)


def _build_atomic(atomic: Dict[str, Any]) -> AtomicFormat:
    atype = atomic.get("type", "coo")
    x_name = atomic.get("x_axis", "X_i")
    y_name = atomic.get("y_axis", "Y_si")
    return atomic_format(atype, x_name, y_name)


def build_format_from_response(response: FormatDesignResponse) -> Format:
    """Construct a sparsene Format object from the LLM's description."""
    levels = response.levels
    if not levels:
        raise ValueError("No format levels in response")

    leaf = _build_atomic(response.atomic)

    current: Format | AtomicFormat = leaf
    for level_desc in reversed(levels):
        axes = [_build_axis(ax) for ax in level_desc.get("axes", [])]
        level_name = level_desc.get("name", "L")
        current = Format(name=level_name, axes=axes, child=current)

    if not isinstance(current, Format):
        raise ValueError("Top-level must be a Format, not atomic")

    return current


# ==============================================================================
# HTTP + JSON helpers
# ==============================================================================


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def _ensure_log_dir(log_dir: str) -> Path:
    p = Path(log_dir)
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        p = repo_root / log_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


# ==============================================================================
# Main agent runner
# ==============================================================================


def run_format_design_agent(
    matrix_name: str,
    matrix_features: Dict[str, Any],
    config: Optional[FormatDesignConfig] = None,
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
    }

    request_id = str(uuid.uuid4())
    req = FormatDesignRequest(
        request_id=request_id,
        matrix_name=matrix_name,
        matrix_features=matrix_features,
        hardware=hardware_dict,
        output_schema=_default_output_schema(),
        constraints=[],
        rules=FORMAT_DESIGN_RULES,
    )
    req.prompt = render_format_design_prompt(req)

    log_dir = _ensure_log_dir(config.log_dir)
    run_dir = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_format_design_{matrix_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "request.json").write_text(
        json.dumps(asdict(req), indent=2, ensure_ascii=False)
    )
    (run_dir / "prompt.txt").write_text(req.prompt)

    api_payload = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "messages": [
            {"role": "system", "content": "You are a sparse-format design expert using the sparsene hierarchical DSL. Output only valid JSON."},
            {"role": "user", "content": req.prompt},
        ],
    }

    api_url = f"{config.api_base_url.rstrip('/')}/chat/completions"
    raw_response_text = ""

    try:
        data = json.dumps(api_payload).encode("utf-8")
        http_req = urllib_request.Request(
            api_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
            method="POST",
        )
        with urllib_request.urlopen(http_req, timeout=config.timeout_sec) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            msg = raw["choices"][0]["message"]
            raw_response_text = msg.get("content", "") or ""
            if not raw_response_text and msg.get("reasoning_content"):
                raw_response_text = msg["reasoning_content"]

        (run_dir / "response.json").write_text(
            json.dumps(raw, indent=2, ensure_ascii=False)
        )
    except (urllib_error.URLError, urllib_error.HTTPError) as exc:
        logger.error("Format design agent HTTP error: %s", exc)
        (run_dir / "response_error.txt").write_text(str(exc))
        return FormatDesignResponse(
            request_id=request_id,
            format_name="ME_TCF_fallback",
            levels=[
                {
                    "level": 1, "name": "row_window",
                    "axes": [
                        {"name": "X_o", "type": "df", "direction": "ROW", "length": "M"},
                        {"name": "Y_s", "type": "sv", "direction": "COL", "length": "K"},
                        {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                    ],
                },
                {
                    "level": 2, "name": "tc_block",
                    "axes": [
                        {"name": "Y_so", "type": "df", "direction": "COL", "length": "K/BLK_K"},
                        {"name": "Y_si", "type": "df", "direction": "COL", "length": "BLK_K"},
                        {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                    ],
                },
            ],
            atomic={"type": "coo", "x_axis": "X_i", "y_axis": "Y_si"},
            params={"BLK_M": 32, "BLK_K": 8},
            rationale=f"Agent call failed: {exc}. Falling back to default ME_TCF.",
            confidence="low",
        )

    parsed = _try_parse_json(raw_response_text)

    if not parsed:
        logger.warning("Format design agent: JSON parse failed. Falling back to ME_TCF.")
        return FormatDesignResponse(
            request_id=request_id,
            format_name="ME_TCF_fallback",
            levels=[
                {
                    "level": 1, "name": "row_window",
                    "axes": [
                        {"name": "X_o", "type": "df", "direction": "ROW", "length": "M"},
                        {"name": "Y_s", "type": "sv", "direction": "COL", "length": "K"},
                        {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                    ],
                },
                {
                    "level": 2, "name": "tc_block",
                    "axes": [
                        {"name": "Y_so", "type": "df", "direction": "COL", "length": "K/BLK_K"},
                        {"name": "Y_si", "type": "df", "direction": "COL", "length": "BLK_K"},
                        {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                    ],
                },
            ],
            atomic={"type": "coo", "x_axis": "X_i", "y_axis": "Y_si"},
            params={"BLK_M": 32, "BLK_K": 8},
            rationale="JSON parse failed. Falling back to default ME_TCF.",
            confidence="low",
            raw_payload={"raw_text": raw_response_text[:500]},
        )

    # Parse levels from LLM response
    levels = parsed.get("levels", [])
    if not levels:
        logger.warning("No levels in format design response. Using fallback.")
        levels = [
            {
                "level": 1, "name": "row_window",
                "axes": [
                    {"name": "X_o", "type": "df", "direction": "ROW", "length": "M"},
                    {"name": "Y_s", "type": "sv", "direction": "COL", "length": "K"},
                    {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                ],
            },
            {
                "level": 2, "name": "tc_block",
                "axes": [
                    {"name": "Y_so", "type": "df", "direction": "COL", "length": "K/BLK_K"},
                    {"name": "Y_si", "type": "df", "direction": "COL", "length": "BLK_K"},
                    {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                ],
            },
        ]

    atomic = parsed.get("atomic", {"type": "coo", "x_axis": "X_i", "y_axis": "Y_si"})
    params = parsed.get("params", {"BLK_M": 32, "BLK_K": 8})

    blk_m = int(params.get("BLK_M", 32))
    blk_k = int(params.get("BLK_K", 8))
    if blk_m not in {8, 16, 32}:
        blk_m = 32
    if blk_k not in {8, 16}:
        blk_k = 8

    return FormatDesignResponse(
        request_id=request_id,
        format_name=parsed.get("format_name", "CustomFormat"),
        levels=levels,
        atomic=atomic,
        params={"BLK_M": blk_m, "BLK_K": blk_k},
        rationale=parsed.get("rationale", ""),
        confidence=parsed.get("confidence", "medium"),
        raw_payload=parsed,
    )
