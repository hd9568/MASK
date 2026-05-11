from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List, cast
import keyword
import re

import sparsene.op_gen.opir.op_ir as op_ir
from sparsene.op_gen.nvir.nvop import (
    NvOp,
    NvOpInput,
    NvOpOutput,
    NvOpTensor,
    SwizzleLayout,
    IoMemType,
    OpMemType,
    NvOpType,
    NvOpImpl,
    GmemInout,
    NvOpProgram,
    ForLoopNvOp,
    ConstantNvOp,
    NvOpSequence,
    Shape,
    IntShape,
    MnkShape,
    VarlenShape,
    ParamShape,
)
from sparsene.logging import get_logger


logger = get_logger(__name__)

SourceType = Union[NvOpOutput, str]
SourceBindingMap = Dict[op_ir.Value, SourceType]


SPARSE_FORMAT_REGISTRY = {
    "ME_TCF": {
        "shape_overrides": {
            "val_len": ["Mo"],               # 拦截 val_len_offset 等
            "val_coo_len": ["nnz_dim_K_o"],  # 拦截 val_coo_len_offset 等
            "val_sidx": ["BLK_K", "nnz_dim_K"],
            "C_val": ["N", "M"],
        },
        # 2. 强制指定特殊变量的数据类型
        "dtypes": {
            "mco_mask": "int64_t",
            "idx": "int", 
            "len": "int", 
            "off": "int", 
            "sidx": "int"
        },
        # 3. 核心 B_val 和 C_val 的 Grid 切分映射
        "custom_layouts": {
            "B_val": {
                "dtype": "float",
                "tensor_str": "logical_divide(make_tensor(make_gmem_ptr(d{name}), make_layout(make_shape(K, N), make_stride(N, _1{}))), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y))",
                "tensor_type_str": "decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_layout(make_shape(VARLEN, VARLEN), make_stride(VARLEN, _1{}))), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, VARLEN)))"
            },
            "C_val": {
                "dtype": "float",
                "tensor_str": "logical_divide(make_tensor(make_gmem_ptr(d{name}), make_shape(N, M)), select<1, 0>(BLK_MNK{}))(make_coord(_, blockIdx.y), make_coord(_, blockIdx.x))",
                "tensor_type_str": "decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN, VARLEN)), select<1, 0>(BLK_MNK{}))(make_coord(_, VARLEN), make_coord(_, VARLEN)))"
            }
        }
    }
}

class HardwareMapper:
    def __init__(self):
        self.loop_depth = 0  # 追踪循环嵌套深度

    def map_loop(self) -> str:
        """外层循环映射到 blockIdx.x，内层保持 sequential"""
        if self.loop_depth == 0:
            return "blockIdx.x"
        return "sequential"

@dataclass
class LoweringState:
    program: NvOpProgram
    hardware_mapper: HardwareMapper = field(default_factory=HardwareMapper) # 新增 Mapper
    value_sources: Dict[op_ir.Value, SourceType] = field(default_factory=dict)
    name_counter: Dict[str, int] = field(default_factory=dict)
    use_fixed_c_tile_shape: bool = False

    def unique_name(self, raw_name: str) -> str:
        base = _sanitize_identifier(raw_name or "op")
        idx = self.name_counter.get(base, 0)
        self.name_counter[base] = idx + 1
        return base if idx == 0 else f"{base}_{idx}"


def _sanitize_identifier(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = "v"
    if sanitized[0].isdigit():
        sanitized = f"v_{sanitized}"
    if keyword.iskeyword(sanitized) or sanitized in {"add", "sub", "mul", "div", "pow", "const"}:
        sanitized = f"gen_{sanitized}"
    return sanitized


def _try_parse_static_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value

    if hasattr(value, "is_Integer") and bool(getattr(value, "is_Integer")):
        return int(value)

    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None

    if str(value_int) == str(value):
        return value_int
    return None


def _io_mem_to_tag(mem: IoMemType) -> str:
    if mem == "gmem":
        return "g"
    if mem == "smem":
        return "s"
    if mem == "rmem":
        return "r"
    return "x"


def _blk_mnk_expr(dim_name: str) -> Optional[str]:
    blk_to_idx = {
        "BLK_M": 0,
        "BLK_N": 1,
        "BLK_K": 2,
    }
    if dim_name in blk_to_idx:
        return f"get<{blk_to_idx[dim_name]}>(BLK_MNK{{}})"
    return None


def _default_dim_entries(dims: List[Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for dim in dims:
        dim_str = _sanitize_identifier(str(dim))
        blk_expr = _blk_mnk_expr(dim_str)

        if blk_expr is not None:
            entries.append(
                {
                    "raw": dim_str,
                    "shape": ParamShape(dim_str),
                    "shape_arg": blk_expr,
                    "type_arg": blk_expr,
                }
            )
        elif dim_str.startswith("BLK_") or dim_str.startswith("TILE_"):
            entries.append(
                {
                    "raw": dim_str,
                    "shape": ParamShape(dim_str),
                    "shape_arg": f"{dim_str}{{}}",
                    "type_arg": f"{dim_str}{{}}",
                }
            )
        elif dim_str.isdigit():
            entries.append(
                {
                    "raw": dim_str,
                    "shape": IntShape(int(dim_str)),
                    "shape_arg": f"Int<{dim_str}>{{}}",
                    "type_arg": f"Int<{dim_str}>{{}}",
                }
            )
        else:
            entries.append(
                {
                    "raw": dim_str,
                    "shape": VarlenShape(dim_str),
                    "shape_arg": dim_str,
                    "type_arg": "VARLEN",
                }
            )
    return entries


def _maybe_swap_blk_k_last_dim(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(entries) >= 2 and entries[-1]["raw"] == "BLK_K":
        swapped = list(entries)
        swapped[-2], swapped[-1] = swapped[-1], swapped[-2]
        return swapped
    return entries


def build_gmem_interfaces(opir_root, format_hints: Dict[str, Any]) -> Dict[str, GmemInout]:
    gmem_inouts: Dict[str, GmemInout] = {}
    
    custom_layouts = format_hints.get("custom_layouts", {})
    shape_overrides = format_hints.get("shape_overrides", {})
    dtypes_hints = format_hints.get("dtypes", {})

    def _clean_symbol(name: str) -> str:
        # 去除编译器 Pass 生成的后缀，以便精准查表
        return re.sub(r'(_new|_offset|_1)+$', '', name)

    for op in opir_root.block.ops:
        if not (isinstance(op, op_ir.ExternalSymbolOp) and isinstance(op.result.type, op_ir.ArrayType)):
            continue
            
        symbol = str(op.attributes.get("symbol", op.result.name_hint))
        base_symbol = _clean_symbol(symbol)
        
        # --- A. 决定 Dtype ---
        dtype = "float" # 默认 fallback
        if base_symbol in custom_layouts:
            dtype = custom_layouts[base_symbol].get("dtype", dtype)
        else:
            for key_pattern, configured_dtype in dtypes_hints.items():
                if key_pattern in base_symbol:
                    dtype = configured_dtype
                    break
                    
        # --- B. 决定 Shape ---
        nvir_shapes = []
        cute_shape_args, cute_type_args = [], []
        
        if base_symbol in shape_overrides:
            dim_entries = _default_dim_entries(list(shape_overrides[base_symbol]))
        else:
            dim_entries = _default_dim_entries(list(op.result.type.dims))

        if base_symbol not in custom_layouts:
            dim_entries = _maybe_swap_blk_k_last_dim(dim_entries)

        nvir_shapes = [entry["shape"] for entry in dim_entries]
        cute_shape_args = [entry["shape_arg"] for entry in dim_entries]
        cute_type_args = [entry["type_arg"] for entry in dim_entries]

        # --- C. 决定 CuTe Layout ---
        if base_symbol in custom_layouts:
            tensor_str = custom_layouts[base_symbol]["tensor_str"].replace("{name}", symbol)
            tensor_type_str = custom_layouts[base_symbol]["tensor_type_str"]
        else:
            shape_str = ", ".join(cute_shape_args)
            type_str = ", ".join(cute_type_args)
            tensor_str = f"make_tensor(make_gmem_ptr(d{symbol}), make_shape({shape_str}))"
            tensor_type_str = f"decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape({type_str})))"

        gmem_inouts[symbol] = GmemInout(
            shape=Shape(*nvir_shapes),
            name=symbol,
            dtype=dtype,
            tensor_str=tensor_str,
            tensor_type_str=tensor_type_str
        )
        
        print("(debug) Registered gmem interface:", symbol, "->", gmem_inouts[symbol], "tensor_str:", tensor_str, "tensor_type_str:", tensor_type_str)
    return gmem_inouts


def _insert_outer_blk_y_loop(
    *,
    program: NvOpProgram,
    state: LoweringState,
) -> ForLoopNvOp:
    blk_y_loop_zero_op = ConstantNvOp(
        name=state.unique_name("BlkYLoopZeroOp"),
        shape=Shape(),
        dtype="int",
        value=0,
    )
    program.append_op(blk_y_loop_zero_op)

    block_dim_y_op = ConstantNvOp(
        name=state.unique_name("BlockDimYOp"),
        shape=Shape(),
        dtype="int",
        value="blockDim.y",
    )
    program.append_op(block_dim_y_op)

    blk_y_loop_op = ForLoopNvOp(
        name=state.unique_name(f"{program.name}_blk_y_loop"),
        blk_idx_mapping="y",
        loop_l=NvOpInput(
            idx=0,
            name="l",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=blk_y_loop_zero_op.outputs[0],
            ),
        ),
        loop_r=NvOpInput(
            idx=1,
            name="r",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=block_dim_y_op.outputs[0],
            ),
        ),
        loop_result=[],
        body=NvOpSequence(),
        iter_args={},
    )
    program.append_op(blk_y_loop_op)
    return blk_y_loop_op


def _fixed_c_tile_shape() -> Shape:
    return Shape(
        IntShape(4),
        Shape(
            MnkShape("BLK_MMA_MNK", "m"),
            MnkShape("BLK_MMA_MNK", "n"),
        ),
    )


def _fixed_reg_a_shape() -> Shape:
    return Shape(
        IntShape(4),
        MnkShape("BLK_MMA_MNK", "m"),
    )


def _fixed_reg_b_shape() -> Shape:
    return Shape(
        IntShape(2),
        MnkShape("BLK_MMA_MNK", "n"),
    )


def _fixed_coo_tile_shape() -> Shape:
    return Shape(ParamShape("get<0>(BLK_MNK{}) * get<2>(BLK_MNK{})"))


def _insert_outer_blk_x_loop(
    *,
    parent_loop: ForLoopNvOp,
    state: LoweringState,
) -> ForLoopNvOp:
    blk_x_loop_zero_op = ConstantNvOp(
        name=state.unique_name("BlkXLoopZeroOp"),
        shape=Shape(),
        dtype="int",
        value=0,
    )
    parent_loop.append(blk_x_loop_zero_op)

    block_dim_x_op = ConstantNvOp(
        name=state.unique_name("BlockDimXOp"),
        shape=Shape(),
        dtype="int",
        value="blockDim.x",
    )
    parent_loop.append(block_dim_x_op)

    blk_x_loop_op = ForLoopNvOp(
        name=state.unique_name(f"{parent_loop.name}_blk_x_loop"),
        blk_idx_mapping="x",
        loop_l=NvOpInput(
            idx=0,
            name="l",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=blk_x_loop_zero_op.outputs[0],
            ),
        ),
        loop_r=NvOpInput(
            idx=1,
            name="r",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=block_dim_x_op.outputs[0],
            ),
        ),
        loop_result=[],
        body=NvOpSequence(),
        iter_args={},
    )
    parent_loop.append(blk_x_loop_op)
    return blk_x_loop_op


def _lower_first_for_loop_to_blk_x(
    *,
    for_op: op_ir.ForLoopOp,
    blk_x_loop_op: ForLoopNvOp,
    state: LoweringState,
) -> None:
    block_arg_sources: SourceBindingMap = {}

    block_arg_sources[for_op.get_induction_var()] = "blockIdx.x"
    for i in range(for_op.num_iter_args):
        block_arg_sources[for_op.get_iter_arg(i)] = _resolve_source(
            for_op.operands[i + 2].source,
            block_arg_sources={},
            state=state,
        )

    fixed_zero_source: Optional[NvOpOutput] = None
    c_val_load_op: Optional[op_ir.Op] = None
    c_val_load_skip_ops: set[int] = set()

    for body_op in for_op.body.ops:
        if getattr(body_op, "name", "") == "c_val_load":
            c_val_load_op = body_op
            break

    def _collect_c_val_load_producers(value: op_ir.Value) -> None:
        if not isinstance(value, op_ir.OpResult):
            return
        producer = value.defining_op
        if producer is None:
            return
        if producer is c_val_load_op:
            return
        if isinstance(producer, op_ir.ExternalSymbolOp):
            return
        if producer not in for_op.body.ops:
            return
        producer_id = id(producer)
        if producer_id in c_val_load_skip_ops:
            return
        c_val_load_skip_ops.add(producer_id)
        for producer_operand in producer.operands:
            _collect_c_val_load_producers(producer_operand.source)

    if c_val_load_op is not None and len(c_val_load_op.operands) >= 2:
        _collect_c_val_load_producers(c_val_load_op.operands[1].source)

    state.hardware_mapper.loop_depth += 1
    prev_fixed_c_shape = state.use_fixed_c_tile_shape
    state.use_fixed_c_tile_shape = True

    loop_result_sources: List[SourceType] = []
    for body_op in for_op.body.ops:
        if id(body_op) in c_val_load_skip_ops:
            continue

        if body_op is c_val_load_op:
            if fixed_zero_source is None:
                zeros_op = ConstantNvOp(
                    name=state.unique_name("ZerosOp"),
                    shape=_fixed_c_tile_shape(),
                    dtype="float",
                    value=0,
                )
                blk_x_loop_op.append(zeros_op)
                fixed_zero_source = zeros_op.outputs[0]

            for result in body_op.results:
                state.value_sources[result] = fixed_zero_source
            continue

        if isinstance(body_op, op_ir.LoopResultOp):
            if len(body_op.operands) == 0:
                continue
            source = _resolve_source(
                body_op.operands[0].source,
                block_arg_sources=block_arg_sources,
                state=state,
            )
            loop_result_sources.append(source)
            continue

        _lower_op(
            body_op,
            target_append=blk_x_loop_op.append,
            block_arg_sources=block_arg_sources,
            state=state,
        )

    state.use_fixed_c_tile_shape = prev_fixed_c_shape
    state.hardware_mapper.loop_depth -= 1

    for i, result in enumerate(for_op.results):
        if i < len(loop_result_sources):
            state.value_sources[result] = loop_result_sources[i]

def generate_nvir(opir: op_ir.MetaOp, format_name: str = "ME_TCF") -> NvOpProgram:
    if format_name not in SPARSE_FORMAT_REGISTRY:
        raise ValueError(f"Unsupported format: {format_name}")
        
    format_hints = SPARSE_FORMAT_REGISTRY[format_name]
    
    # 1. 查表生成 GmemInouts
    gmem_inouts = build_gmem_interfaces(opir, format_hints)

    

    program_name = _sanitize_identifier(getattr(opir, "name", "generated_kernel"))
    if program_name in ["", "meta"]: program_name = "generated_kernel"

    program = NvOpProgram(name=program_name, gmem_inouts=gmem_inouts)
    state = LoweringState(program=program) # 自动带上 HardwareMapper

    # 在解析nvir之前，这里首先帮我插入一层 for op 给 y 维度
    blk_y_loop_op = _insert_outer_blk_y_loop(program=program, state=state)

    # 固定将遇到的第一个 ForLoopOp 映射给 x 维度：
    # 1) 忽略该 for 的原始左右边界计算
    # 2) 直接使用 [0, blockDim.x) 作为循环范围
    # 3) 将该 for 的 body 下放到 blk_x_loop_op 中继续 lowering
    first_for_op: Optional[op_ir.ForLoopOp] = None
    for top_op in opir.block.ops:
        if isinstance(top_op, op_ir.ForLoopOp):
            first_for_op = top_op
            break

    if first_for_op is not None:
        blk_x_loop_op = _insert_outer_blk_x_loop(
            parent_loop=blk_y_loop_op,
            state=state,
        )
        _lower_first_for_loop_to_blk_x(
            for_op=first_for_op,
            blk_x_loop_op=blk_x_loop_op,
            state=state,
        )
        return program


    # 2. 启动 Lowering
    _lower_block(
        block=opir.block,
        target_append=blk_y_loop_op.append,
        block_arg_sources={},
        state=state,
    )
    return program


#! ===========================================================================

def _classify_logical_kind(op: op_ir.Op) -> str:
    if isinstance(op, op_ir.LoadOffsetOp):
        return "load_offset"
    if isinstance(op, op_ir.ArrayRefOp):
        return "array_ref"
    if isinstance(op, op_ir.LoadOp):
        return "load"
    if isinstance(op, op_ir.ArangeOp):
        return "arange"
    if isinstance(op, op_ir.AddOp):
        return "add"
    if isinstance(op, op_ir.SubOp):
        return "sub"
    if isinstance(op, op_ir.MulOp):
        return "mul"
    if isinstance(op, op_ir.DivOp):
        return "div"
    if isinstance(op, op_ir.PowOp):
        return "pow"
    if isinstance(op, op_ir.DeviceOp):
        return op.name
    return op.name


def _resolve_source(
    value: op_ir.Value,
    *,
    block_arg_sources: SourceBindingMap,
    state: LoweringState,
) -> SourceType:
    if value in block_arg_sources:
        return block_arg_sources[value]

    if isinstance(value, op_ir.BlockArgument):
        name_hint = value.name_hint or "_idx"
        blk_expr = _blk_mnk_expr(name_hint)
        return blk_expr if blk_expr is not None else name_hint

    if getattr(value, "name_hint", None) == "_":
        return "_"

    if value in state.value_sources:
        return state.value_sources[value]

    if isinstance(value, op_ir.OpResult) and isinstance(value.defining_op, op_ir.ExternalSymbolOp):
        symbol = str(value.defining_op.attributes["symbol"])
        if isinstance(value.type, op_ir.ArrayType):
            return state.program.gmem_tensor_ops[symbol].outputs[0]
        blk_expr = _blk_mnk_expr(symbol)
        if blk_expr is not None:
            return blk_expr
        return symbol

    fallback_name = getattr(value, "name_hint", None) or "0"
    blk_expr = _blk_mnk_expr(fallback_name)
    return blk_expr if blk_expr is not None else fallback_name


def _infer_input_mem(value: op_ir.Value, source: SourceType) -> IoMemType:
    if isinstance(source, NvOpOutput):
        return source.tensor.mem
    if isinstance(value.type, op_ir.ArrayType):
        return "gmem"
    return "rmem"


def _is_placeholder_source(source: SourceType) -> bool:
    return isinstance(source, str) and source == "_"


def _device_mem_hint(op: op_ir.Op) -> Optional[IoMemType]:
    mem = op.attributes.get("mem", None)
    if not isinstance(mem, str):
        return None
    mem = mem.upper()
    if "2" not in mem:
        return None
    dst = mem.split("2")[-1]
    if dst == "G":
        return "gmem"
    if dst == "S":
        return "smem"
    if dst == "R":
        return "rmem"
    return None


def _choose_output_mem(
    logical_kind: str,
    result_type: op_ir.Type,
    mem_hint: Optional[IoMemType],
) -> IoMemType:
    if not isinstance(result_type, op_ir.ArrayType):
        return "rmem"

    if logical_kind == "mma":
        return "rmem"
    if logical_kind in ["coo_atomic_val_restore", "mco_atomic_val_restore"]:
        return "smem"
    if logical_kind in ["coo_atomic_format_load_val", "mco_atomic_format_load_val"]:
        return "smem"
    if logical_kind in ["mco_atomic_format_load_mask", "load_offset"]:
        return "rmem"
    if logical_kind in ["coo_atomic_format_load_idx"] and isinstance(result_type, op_ir.ArrayType):
        return "smem"
    if logical_kind in ["c_val_load"]:
        return "rmem"
    if logical_kind in ["ldmatrix"]:
        return "rmem"

    if mem_hint is not None:
        return mem_hint
    return "rmem"


def _choose_op_type(logical_kind: str, outputs: List[NvOpOutput]) -> NvOpType:
    if logical_kind == "load_offset":
        return NvOpType.LOAD_LR_OFF

    if logical_kind in [
        "c_val_load",
        "c_val_store",
        "coo_atomic_format_load_idx",
        "coo_atomic_format_load_val",
        "mco_atomic_format_load_mask",
        "mco_atomic_format_load_val",
        "load",
    ]:
        if outputs and len(outputs[0].tensor.shape) <= 1:
            return NvOpType.COPY_1D
        if "coo" in logical_kind or "mco" in logical_kind:
            return NvOpType.COPY_2D_SP
        return NvOpType.COPY_2D

    if logical_kind in ["mma", "coo_atomic_val_restore", "mco_atomic_val_restore", "ldmatrix"]:
        return NvOpType.OTHERS

    return NvOpType.UNKNOWN


def _infer_mem_type(inputs: List[NvOpInput], outputs: List[NvOpOutput]) -> OpMemType:
    if not inputs and not outputs:
        return ("x", "x")

    src_tag = _io_mem_to_tag(inputs[0].tensor.mem) if inputs else "x"
    dst_tag = _io_mem_to_tag(outputs[0].tensor.mem) if outputs else "x"
    return cast(OpMemType, (src_tag, dst_tag))


def _make_input(
    *,
    idx: int,
    name: str,
    value: op_ir.Value,
    source: SourceType,
) -> NvOpInput:
    value_type = value.type if value.type is not None else op_ir.IntType()
    inferred_shape = opir_type_to_nvir_shape(value_type)
    inferred_dtype = opir_type_to_nvir_dtype(value_type)
    inferred_mem = _infer_input_mem(value, source)

    # Prefer the already-lowered producer tensor metadata when available.
    # This preserves shape/layout normalizations such as `val_sidx_new`
    # being materialized as `[BLK_K, nnz_dim_K]` instead of the original
    # OPIR-facing `[nnz_dim_K, BLK_K]`.
    if isinstance(source, NvOpOutput):
        inferred_shape = source.tensor.shape
        inferred_dtype = source.tensor.dtype
        inferred_mem = source.tensor.mem

    return NvOpInput(
        idx=idx,
        name=name,
        tensor=NvOpTensor(
            shape=inferred_shape,
            mem=inferred_mem,
            dtype=inferred_dtype,
            source=source,
        ),
    )


def _insert_ldmatrix_if_needed(
    *,
    source: SourceType,
    value: op_ir.Value,
    target_append,
    state: LoweringState,
    operand_index: int,
) -> SourceType:
    if not isinstance(source, NvOpOutput):
        return source
    if source.tensor.mem != "smem":
        return source

    if operand_index == 0:
        op_name = state.unique_name("S2rAValLoadOp")
        output_shape = _fixed_reg_a_shape()
        output_name = "REGA"
        input_name = "tileA"
        impl = NvOpImpl(
            f"""__syncthreads();
int row = lid % 16;
int col = lid / 16;
ldmatrix_m8n8k8_x4(
    (uint32_t*)({output_name}(_, _0{{}}, buf_idx).data()),
    (void*)(&{input_name}(row, col * 4))
);"""
        )
    elif operand_index == 1:
        op_name = state.unique_name("S2rBValLoadOp")
        output_shape = _fixed_reg_b_shape()
        output_name = "REGB"
        input_name = "tileB"
        impl = NvOpImpl(
            f"""__syncthreads();
for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{{}}); n_iter++) {{
    int row_b = lid / 2;
    int col_b = lid % 2;
    ldmatrix_m8n8k8_x2(
        (uint32_t*)({output_name}(_, n_iter, buf_idx).data()),
        (void*)(&{input_name}(row_b, col_b * 4 + n_iter * 8))
    );
    {output_name}(_0{{}}, n_iter, buf_idx) = __shfl_sync(0xffffffff, {output_name}(_0{{}}, n_iter, buf_idx), lid / 4 + lid % 4 * 8);
    {output_name}(_1{{}}, n_iter, buf_idx) = __shfl_sync(0xffffffff, {output_name}(_1{{}}, n_iter, buf_idx), lid / 4 + lid % 4 * 8);
}}"""
        )
    else:
        return source
    input_tensor = NvOpInput(
        idx=0,
        name=input_name,
        tensor=NvOpTensor(
            shape=opir_type_to_nvir_shape(value.type),
            mem="smem",
            dtype=opir_type_to_nvir_dtype(value.type),
            source=source,
        ),
    )
    output_tensor = NvOpOutput(
        idx=0,
        name=output_name,
        tensor=NvOpTensor(
            shape=output_shape,
            mem="rmem",
            dtype=opir_type_to_nvir_dtype(value.type),
        ),
    )

    ldmatrix_op = NvOp(
        name=op_name,
        inputs=[input_tensor],
        outputs=[output_tensor],
        impl=impl,
        mem_type=("s", "r"),
        op_type=NvOpType.OTHERS,
        logical_kind="ldmatrix",
        matrix_operand="A" if operand_index == 0 else "B",
    )
    target_append(ldmatrix_op)
    return ldmatrix_op.outputs[0]


def _make_impl(logical_kind: str, inputs: List[NvOpInput], outputs: List[NvOpOutput]) -> NvOpImpl:
    if logical_kind == "add" and len(inputs) == 2 and len(outputs) == 1:
        return NvOpImpl(f"{outputs[0].name}(0) = {inputs[0].name} + {inputs[1].name};")
    if logical_kind == "sub" and len(inputs) == 2 and len(outputs) == 1:
        return NvOpImpl(f"{outputs[0].name}(0) = {inputs[0].name} - {inputs[1].name};")
    if logical_kind == "mul" and len(inputs) == 2 and len(outputs) == 1:
        return NvOpImpl(f"{outputs[0].name}(0) = {inputs[0].name} * {inputs[1].name};")
    if logical_kind == "div" and len(inputs) == 2 and len(outputs) == 1:
        return NvOpImpl(f"{outputs[0].name}(0) = {inputs[0].name} / {inputs[1].name};")
    if logical_kind == "pow" and len(inputs) == 2 and len(outputs) == 1:
        return NvOpImpl(f"{outputs[0].name}(0) = pow({inputs[0].name}, {inputs[1].name});")

    if logical_kind == "load_offset" and len(inputs) >= 2 and len(outputs) >= 2:
        return NvOpImpl(
            f"{outputs[0].name}(0) = {inputs[0].name}({inputs[1].name});\n"
            f"{outputs[1].name}(0) = {inputs[0].name}({inputs[1].name} + 1);"
        )

    if logical_kind == "load" and len(inputs) >= 1 and len(outputs) == 1:
        if len(outputs[0].tensor.shape) == 0:
            indices = ", ".join(inp.name for inp in inputs[1:])
            if not indices:
                indices = "0"
            return NvOpImpl(f"{outputs[0].name}(0) = {inputs[0].name}({indices});")
        return NvOpImpl("//! TODO: tensor load lowering")

    if logical_kind == "coo_atomic_format_load_idx" and len(inputs) >= 3 and len(outputs) == 2:
        return NvOpImpl(
            f"""{outputs[1].name}(0) = {inputs[2].name} - {inputs[1].name};
if (({inputs[1].name} & 3) == 0) {{
    auto thr_tiler = make_shape(_4{{}});
    auto input = make_tensor({inputs[0].name}.data() + {inputs[1].name}, size<0>(shape({outputs[0].name})));
    for (int i_load = lid; i_load * 4 + {inputs[1].name} < {inputs[2].name}; i_load += 32) {{
        auto thr_coord = make_coord(i_load);
        auto src = local_tile(input, thr_tiler, thr_coord);
        auto dst = local_tile({outputs[0].name}(_, buf_idx), thr_tiler, thr_coord);
        copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, int>{{}}, src, dst);
    }}
}} else {{
    int nnz_tile = get<0>(BLK_MNK{{}}) * get<2>(BLK_MNK{{}});
    for (int i_load = lid; i_load < nnz_tile; i_load += 32) {{
        int src_idx = {inputs[1].name} + i_load;
        {outputs[0].name}(i_load, buf_idx) = src_idx < {inputs[2].name} ? {inputs[0].name}(src_idx) : 0;
    }}
}}"""
        )
    if logical_kind == "coo_atomic_format_load_idx" and len(inputs) >= 2 and len(outputs) == 1:
        return NvOpImpl(
            f"""int nnz_tile = get<0>(BLK_MNK{{}}) * get<2>(BLK_MNK{{}});
for (int i_load = lid; i_load < nnz_tile; i_load += 32) {{
    int src_idx = {inputs[1].name} + i_load;
    {outputs[0].name}(i_load, buf_idx) = src_idx < get<0>(shape_i0) ? {inputs[0].name}(src_idx) : 0;
}}"""
        )

    if logical_kind == "coo_atomic_format_load_val" and len(inputs) >= 3 and len(outputs) == 1:
        return NvOpImpl(
            f"""if (({inputs[1].name} & 3) == 0) {{
    auto thr_tiler = make_shape(_4{{}});
    auto input = make_tensor({inputs[0].name}.data() + {inputs[1].name}, size<0>(shape({outputs[0].name})));
    for (int i_load = lid; i_load * 4 + {inputs[1].name} < {inputs[2].name}; i_load += 32) {{
        auto thr_coord = make_coord(i_load);
        auto src = local_tile(input, thr_tiler, thr_coord);
        auto dst = local_tile({outputs[0].name}(_, buf_idx), thr_tiler, thr_coord);
        copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{{}}, src, dst);
    }}
}} else {{
    int nnz_tile = get<0>(BLK_MNK{{}}) * get<2>(BLK_MNK{{}});
    for (int i_load = lid; i_load < nnz_tile; i_load += 32) {{
        int src_idx = {inputs[1].name} + i_load;
        {outputs[0].name}(i_load, buf_idx) = src_idx < {inputs[2].name} ? {inputs[0].name}(src_idx) : float(0);
    }}
}}"""
        )
    if logical_kind == "coo_atomic_format_load_val" and len(inputs) >= 2 and len(outputs) == 1:
        return NvOpImpl(
            f"""int nnz_tile = get<0>(BLK_MNK{{}}) * get<2>(BLK_MNK{{}});
for (int i_load = lid; i_load < nnz_tile; i_load += 32) {{
    int src_idx = {inputs[1].name} + i_load;
    {outputs[0].name}(i_load, buf_idx) = src_idx < get<0>(shape_i0) ? {inputs[0].name}(src_idx) : float(0);
}}"""
        )

    if logical_kind == "mco_atomic_format_load_mask" and len(inputs) >= 2 and len(outputs) == 1:
        return NvOpImpl(f"{outputs[0].name}(0) = {inputs[0].name}({inputs[1].name});")

    if logical_kind == "mco_atomic_format_load_val" and len(inputs) >= 2 and len(outputs) == 1:
        return NvOpImpl(
            f"""int nnz_tile = size<0>(shape({outputs[0].name}));
for (int i_load = lid; i_load < nnz_tile; i_load += 32) {{
    {outputs[0].name}(i_load, buf_idx) = {inputs[0].name}({inputs[1].name} + i_load);
}}"""
        )

    if logical_kind == "coo_atomic_val_restore" and len(inputs) >= 3 and len(outputs) == 1:
        out_name = outputs[0].name
        out_dtype = outputs[0].tensor.dtype
        return NvOpImpl(
            f"""int tile_elems = get<0>(shape({out_name})) * get<1>(shape({out_name}));
for (int i_clear = lid; i_clear < tile_elems; i_clear += 32) {{
    *(({out_dtype}*)({out_name}(_, _, buf_idx).data().get() + i_clear)) = {out_dtype}(0);
}}
__syncthreads();
for (int i_restore = lid; i_restore < {inputs[2].name}; i_restore += 32) {{
    auto value = {inputs[0].name}(i_restore);
    int idx = {inputs[1].name}(i_restore);
    if (idx < tile_elems) {{
        *(({out_dtype}*)({out_name}(_, _, buf_idx).data().get() + idx)) = value;
    }}
}}"""
        )

    if logical_kind == "mco_atomic_val_restore" and len(inputs) >= 3 and len(outputs) == 1:
        out_name = outputs[0].name
        out_dtype = outputs[0].tensor.dtype
        return NvOpImpl(
            f"""int tile_elems = get<0>(shape({out_name})) * get<1>(shape({out_name}));
for (int i_clear = lid; i_clear < tile_elems; i_clear += 32) {{
    *(({out_dtype}*)({out_name}(_, _, buf_idx).data().get() + i_clear)) = {out_dtype}(0);
}}
__syncthreads();
if (lid == 0) {{
    int value_ptr = 0;
    unsigned long long mask = static_cast<unsigned long long>({inputs[1].name});
    for (int idx = 0; idx < tile_elems && value_ptr < {inputs[2].name}; ++idx) {{
        if ((mask >> idx) & 1ULL) {{
            *(({out_dtype}*)({out_name}(_, _, buf_idx).data().get() + idx)) = {inputs[0].name}(value_ptr);
            ++value_ptr;
        }}
    }}
}}
__syncthreads();"""
        )

    if logical_kind == "ldmatrix" and len(inputs) >= 1 and len(outputs) == 1:
        return NvOpImpl(
            f"""int rows = get<0>(shape({inputs[0].name}));
int cols = get<1>(shape({inputs[0].name}));
for (int linear = lid; linear < rows * cols; linear += 32) {{
    int row = linear / cols;
    int col = linear % cols;
    {outputs[0].name}(row, col, buf_idx) = {inputs[0].name}(row, col);
}}"""
        )

    if logical_kind == "mma":
        return NvOpImpl(
            f"""for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{{}}); m_iter++) {{
    for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{{}}); n_iter++) {{
        uint32_t frag_A[4];
        uint32_t frag_B[2];
        asm("cvt.rna.tf32.f32  %0, %1;\\n" : "=r"(frag_A[0]) : "f"({inputs[0].name}(0, m_iter)));
        asm("cvt.rna.tf32.f32  %0, %1;\\n" : "=r"(frag_A[1]) : "f"({inputs[0].name}(1, m_iter)));
        asm("cvt.rna.tf32.f32  %0, %1;\\n" : "=r"(frag_A[2]) : "f"({inputs[0].name}(2, m_iter)));
        asm("cvt.rna.tf32.f32  %0, %1;\\n" : "=r"(frag_A[3]) : "f"({inputs[0].name}(3, m_iter)));
        asm("cvt.rna.tf32.f32  %0, %1;\\n" : "=r"(frag_B[0]) : "f"({inputs[1].name}(0, n_iter)));
        asm("cvt.rna.tf32.f32  %0, %1;\\n" : "=r"(frag_B[1]) : "f"({inputs[1].name}(1, n_iter)));
        mma_m16n8k8_fp32_tf32_tf32_fp32({outputs[0].name}(_, make_coord(m_iter, n_iter)).data(), frag_A, frag_B);
    }}
}}"""
        )

    # if logical_kind in ["c_val_load", "c_val_store"]:
    #     return NvOpImpl("//! TODO: placement-aware c_val load/store lowering")
    if logical_kind == "c_val_load" and len(inputs) >= 1 and len(outputs) >= 1:
        # inputs[0] 就是局部切好的 C_val，我们直接把它拷贝给寄存器 outputs[0]
        # 注意: CuTe 推荐使用 local_tile 获取局部视图，再 copy
        # (这里假设 inputs[1] 是原来的全局偏移，我们直接忽略它)
        return NvOpImpl(
            f"auto local_C = local_tile({inputs[0].name}, make_shape(get<0>(BLK_MNK{{}}), get<1>(BLK_MNK{{}})), make_coord(_0{{}}, _0{{}}));\n"
            f"copy(local_C, {outputs[0].name}(0));"
        )
        
    if logical_kind == "c_val_store" and len(inputs) >= 2:
        return NvOpImpl(
            f"""for (int i_tileN = 0; i_tileN < get<1>(BLK_MNK{{}}) / 8; i_tileN++) {{
    int row = lid / 4;
    int col = i_tileN * 8 + lid % 4 * 2;
    {inputs[1].name}(col, row) = {inputs[0].name}(0, i_tileN);
    {inputs[1].name}(col + 1, row) = {inputs[0].name}(1, i_tileN);
    {inputs[1].name}(col, row + 8) = {inputs[0].name}(2, i_tileN);
    {inputs[1].name}(col + 1, row + 8) = {inputs[0].name}(3, i_tileN);
}}"""
        )
    if logical_kind == "array_ref":
        if len(outputs) == 1 and outputs[0].tensor.mem == "smem" and len(inputs) >= 2:
            out_name = outputs[0].name
            src_name = inputs[0].name
            idx_name = inputs[1].name

            if len(outputs[0].tensor.shape) == 1:
                return NvOpImpl(
                    f"""int tile_len = size<0>(shape({out_name}));
for (int i_load = lid; i_load < tile_len; i_load += 32) {{
    {out_name}(i_load, buf_idx) = {src_name}(i_load, {idx_name});
}}"""
                )


            if len(outputs[0].tensor.shape) >= 2:
                if outputs[0].tensor.row_major:
                    return NvOpImpl(
                        f"""auto load_tile_n = _4{{}};
for (int iter_i = lid; iter_i < get<0>(shape({out_name})) * get<1>(shape({out_name})) / load_tile_n; iter_i += 32) {{
    int row = iter_i / (get<1>(shape({out_name})) / load_tile_n);
    int col = iter_i % (get<1>(shape({out_name})) / load_tile_n);
    int src_row = {idx_name}(row);
    auto thr_tiler_gmem = make_shape(_1{{}}, _4{{}});
    auto thr_coord_gmem = make_coord(src_row, col);
    auto src = local_tile({src_name}, thr_tiler_gmem, thr_coord_gmem);
    auto thr_tiler_smem = make_shape(_1{{}}, _4{{}});
    auto thr_coord_smem = make_coord(row, col);
    auto dst = local_tile({out_name}(_, _, buf_idx), thr_tiler_smem, thr_coord_smem);
    copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{{}}, src(0, _), dst(0, _));
}}"""
                    )
                return NvOpImpl(
                    f"""int rows = size<0>(shape({out_name}));
int cols = size<1>(shape({out_name}));
for (int linear = lid; linear < rows * cols; linear += 32) {{
    int row = linear / cols;
    int col = linear % cols;
    int src_row = {idx_name}(row);
    {out_name}(row, col, buf_idx) = {src_name}(src_row, col);
}}"""
                )

        return NvOpImpl("//! array_ref lowered as logical view/alias")

    return NvOpImpl("//! TODO: generic lower")


def _lower_constant(
    op: op_ir.ConstantOp,
    *,
    target_append,
    state: LoweringState,
) -> None:
    value = op.attributes["value"]
    if hasattr(value, "is_Integer") and bool(getattr(value, "is_Integer")):
        lowered_value: Union[int, float, str] = int(value)
    elif hasattr(value, "is_Float") and bool(getattr(value, "is_Float")):
        lowered_value = float(value)
    else:
        lowered_value = str(value)

    const_name = state.unique_name(op.result.name_hint or "const")
    nv_const = ConstantNvOp(
        name=const_name,
        shape=Shape(),
        dtype=opir_type_to_nvir_dtype(op.result.type),
        value=lowered_value,
    )
    target_append(nv_const)
    state.value_sources[op.result] = nv_const.outputs[0]


def _lower_for_loop(
    op: op_ir.ForLoopOp,
    *,
    target_append,
    block_arg_sources: SourceBindingMap,
    state: LoweringState,
) -> None:
    mapping = state.hardware_mapper.map_loop()
    if mapping == "blockIdx.x":
        logger.info("Mapping outermost loop to blockIdx.x")
        
        inner_block_arg_sources = dict(block_arg_sources)
        # 强制将 %i1 绑定为 blockIdx.x
        inner_block_arg_sources[op.get_induction_var()] = "blockIdx.x"
        
        # 透传 iter_args (如 C_tile_io -> C_val)
        for i in range(op.num_iter_args):
            inner_block_arg_sources[op.get_iter_arg(i)] = _resolve_source(
                op.operands[i + 2].source, block_arg_sources=block_arg_sources, state=state
            )
            
        state.hardware_mapper.loop_depth += 1
        
        # 展平执行 (直接 target_append，不生成 ForLoopNvOp)
        loop_result_sources = _lower_block(
            block=op.body,
            target_append=target_append,
            block_arg_sources=inner_block_arg_sources,
            state=state,
        )
        
        state.hardware_mapper.loop_depth -= 1
        
        # 将展平结果映射回 results
        for i, result in enumerate(op.results):
            if i < len(loop_result_sources):
                state.value_sources[result] = loop_result_sources[i]
    else:

        loop_name = state.unique_name(op.name)

        loop_l_source = _resolve_source(op.range[0], block_arg_sources=block_arg_sources, state=state)
        loop_r_source = _resolve_source(op.range[1], block_arg_sources=block_arg_sources, state=state)

        loop_l = _make_input(idx=0, name="l", value=op.range[0], source=loop_l_source)
        loop_r = _make_input(idx=1, name="r", value=op.range[1], source=loop_r_source)

        iter_arg_bindings: Dict[op_ir.BlockArgument, SourceType] = {}
        loop_iter_args: Dict[str, NvOpInput] = {}
        off_l_bound = False

        off_l_alias_values: List[op_ir.Value] = []
        for body_op in op.body.ops:
            for operand in body_op.operands:
                candidate = operand.source
                if not (
                    isinstance(candidate, op_ir.OpResult)
                    and isinstance(candidate.defining_op, op_ir.LoadOffsetOp)
                    and candidate.result_idx_in_owner == 0
                ):
                    continue

                candidate_source = _resolve_source(
                    candidate,
                    block_arg_sources=block_arg_sources,
                    state=state,
                )
                if (
                    isinstance(candidate_source, NvOpOutput)
                    and candidate_source.op is not None
                    and candidate_source.op.op_type
                    in {
                        NvOpType.LOAD_LR_OFF,
                        NvOpType.LOAD_LR_PAIR,
                    }
                    and candidate_source.idx == 0
                ):
                    if not off_l_bound:
                        loop_iter_args["off_l"] = _make_input(
                            idx=2 + len(loop_iter_args),
                            name="off_l",
                            value=candidate,
                            source=candidate_source,
                        )
                        off_l_bound = True
                    if candidate not in off_l_alias_values:
                        off_l_alias_values.append(candidate)

        for i in range(op.num_iter_args):
            iter_arg = op.get_iter_arg(i)
            iter_source = _resolve_source(
                op.operands[i + 2].source,
                block_arg_sources=block_arg_sources,
                state=state,
            )

            binding: SourceType = iter_source
            if (
                not off_l_bound
                and isinstance(iter_source, NvOpOutput)
                and iter_source.op is not None
                and iter_source.op.op_type
                in {
                    NvOpType.LOAD_LR_OFF,
                    NvOpType.LOAD_LR_PAIR,
                }
                and iter_source.idx == 0
            ):
                loop_iter_args["off_l"] = _make_input(
                    idx=2 + len(loop_iter_args),
                    name="off_l",
                    value=iter_arg,
                    source=iter_source,
                )
                binding = "off_l"
                off_l_bound = True

            iter_arg_bindings[iter_arg] = binding

        loop_outputs: List[NvOpOutput] = []
        use_fixed_c_tile = (
            state.use_fixed_c_tile_shape
            and op.num_iter_args > 0
            and any(
                "c" in (op.get_iter_arg(i).name_hint or "").lower()
                for i in range(op.num_iter_args)
            )
        )
        for idx, result in enumerate(op.results):
            init_source: Optional[SourceType] = None
            if idx < op.num_iter_args:
                init_source = _resolve_source(
                    op.operands[idx + 2].source,
                    block_arg_sources=block_arg_sources,
                    state=state,
                )

            if isinstance(init_source, NvOpOutput):
                output_mem: IoMemType = init_source.tensor.mem
            else:
                output_mem = "rmem"

            loop_outputs.append(
                NvOpOutput(
                    idx=idx,
                    name=f"{loop_name}_result_{idx}",
                    tensor=NvOpTensor(
                        shape=_fixed_c_tile_shape() if use_fixed_c_tile and idx == 0 else opir_type_to_nvir_shape(result.type),
                        mem=output_mem,
                        dtype=opir_type_to_nvir_dtype(result.type),
                    ),
                )
            )

        loop_nvop = ForLoopNvOp(
            name=loop_name,
            loop_l=loop_l,
            loop_r=loop_r,
            loop_result=loop_outputs,
            iter_args=loop_iter_args,
        )
        target_append(loop_nvop)

        inner_block_arg_sources = dict(block_arg_sources)
        # The loop induction variable is consumed inside the generated
        # pipeline body through `shift_constant`. Binding it to `l + {c}`
        # preserves the concrete step index for fill / loop_step /
        # remainder / empty, instead of reading the uninitialized `_idx`
        # member of `for_loop`.
        inner_block_arg_sources[op.get_induction_var()] = "l + {c}"
        inner_block_arg_sources[op.range[0]] = "l"
        inner_block_arg_sources[op.range[1]] = "r"
        for i in range(op.num_iter_args):
            iter_arg = op.get_iter_arg(i)
            inner_block_arg_sources[iter_arg] = iter_arg_bindings[iter_arg]
        if off_l_bound:
            for alias_value in off_l_alias_values:
                inner_block_arg_sources[alias_value] = "off_l"

        state.hardware_mapper.loop_depth += 1
        loop_result_sources = _lower_block(
            block=op.body,
            target_append=loop_nvop.append,
            block_arg_sources=inner_block_arg_sources,
            state=state,
        )
        state.hardware_mapper.loop_depth -= 1

        for idx, source in enumerate(loop_result_sources):
            if idx >= len(loop_nvop.outputs):
                break
            if isinstance(source, NvOpOutput):
                loop_nvop.set_loop_result(idx, source)

        for i, result in enumerate(op.results):
            if i < len(loop_nvop.outputs):
                state.value_sources[result] = loop_nvop.outputs[i]


def _lower_array_ref(
    op: op_ir.ArrayRefOp,
    *,
    target_append,
    block_arg_sources: SourceBindingMap,
    state: LoweringState,
) -> None:
    op_name = state.unique_name(op.name)

    inputs: List[NvOpInput] = []
    for operand in op.operands:
        value = operand.source
        source = _resolve_source(value, block_arg_sources=block_arg_sources, state=state)
        if _is_placeholder_source(source):
            continue
        inputs.append(
            _make_input(
                idx=len(inputs),
                name=op.operand_name(len(inputs)),
                value=value,
                source=source,
            )
        )

    array_source = inputs[0].tensor.source if inputs else None
    output_origin = array_source if isinstance(array_source, NvOpOutput) else None

    base_tensor_name = ""
    if output_origin is not None and output_origin.op is not None:
        base_tensor_name = str(output_origin.op.name)
        if base_tensor_name.endswith("Op"):
            base_tensor_name = base_tensor_name[:-2]

    is_val_sidx_ref = base_tensor_name.startswith("val_sidx")
    is_b_val_ref = base_tensor_name == "B_val"

    materialize_to_smem = (
        output_origin is not None
        and output_origin.tensor.mem == "gmem"
        and isinstance(op.result.type, op_ir.ArrayType)
        and (
            is_val_sidx_ref
            or is_b_val_ref
            or (
                any(len(inp.tensor.shape) > 0 for inp in inputs[1:])
                and len(op.result.type.dims) >= 2
            )
        )
    )

    if materialize_to_smem:
        output_mem: IoMemType = "smem"
        output_origin = None
    else:
        output_mem = output_origin.tensor.mem if output_origin is not None else "rmem"

    output_shape = opir_type_to_nvir_shape(op.result.type)
    if is_b_val_ref:
        normalized_shape: List[Any] = []
        for dim in output_shape:
            if isinstance(dim, VarlenShape) and dim.name == "N":
                normalized_shape.append(MnkShape("BLK_MNK", "n"))
            else:
                normalized_shape.append(dim)
        output_shape = Shape(*normalized_shape)

    outputs = [
        NvOpOutput(
            idx=0,
            name=op.result_name(0),
            tensor=NvOpTensor(
                shape=output_shape,
                mem=output_mem,
                dtype=opir_type_to_nvir_dtype(op.result.type),
                row_major=is_b_val_ref,
                swizzle=SwizzleLayout(3, 2, 3) if is_b_val_ref else None,
            ),
            origin=output_origin,
        )
    ]

    cp_async_enabled = is_b_val_ref
    nvop = NvOp(
        name=op_name,
        inputs=inputs,
        outputs=outputs,
        impl=_make_impl("array_ref", inputs, outputs),
        mem_type=_infer_mem_type(inputs, outputs),
        op_type=(NvOpType.COPY_2D_SP if materialize_to_smem else NvOpType.OTHERS),
        logical_kind="array_ref",
        array_ref_role="b_val" if is_b_val_ref else "val_sidx" if is_val_sidx_ref else "generic",
        cp_async=cp_async_enabled,
    )
    if outputs and outputs[0].tensor.mem == "smem":
        outputs[0].attrs["cp_async"] = cp_async_enabled
    target_append(nvop)
    state.value_sources[op.result] = nvop.outputs[0]


def _lower_generic_op(
    op: op_ir.Op,
    *,
    target_append,
    block_arg_sources: SourceBindingMap,
    state: LoweringState,
) -> None:
    logical_kind = _classify_logical_kind(op)
    op_name = state.unique_name(logical_kind)

    inputs: List[NvOpInput] = []
    for operand in op.operands:
        value = operand.source
        source = _resolve_source(value, block_arg_sources=block_arg_sources, state=state)
        if _is_placeholder_source(source):
            continue

        input_idx = len(inputs)

        if logical_kind == "mma" and input_idx < 2:
            source = _insert_ldmatrix_if_needed(
                source=source,
                value=value,
                target_append=target_append,
                state=state,
                operand_index=input_idx,
            )

        lowered_input = _make_input(
            idx=input_idx,
            name=op.operand_name(input_idx),
            value=value,
            source=source,
        )

        if logical_kind == "mma" and isinstance(source, NvOpOutput):
            if source.op is not None and source.op.attrs.get("logical_kind") == "ldmatrix":
                lowered_input.tensor.shape = source.tensor.shape
                lowered_input.tensor.dtype = source.tensor.dtype

        inputs.append(lowered_input)

    if state.use_fixed_c_tile_shape and logical_kind == "c_val_store" and len(inputs) >= 1:
        inputs[0].tensor.shape = _fixed_c_tile_shape()

    outputs: List[NvOpOutput] = []
    mem_hint = _device_mem_hint(op)
    for idx, result in enumerate(op.results):
        output_origin: Optional[NvOpOutput] = None
        if logical_kind == "mma" and idx == 0 and len(inputs) >= 3:
            if isinstance(inputs[2].tensor.source, NvOpOutput):
                output_origin = inputs[2].tensor.source

        output_shape = opir_type_to_nvir_shape(result.type)
        if logical_kind == "mma" and idx == 0:
            output_shape = _fixed_c_tile_shape()
        elif logical_kind in {"coo_atomic_format_load_idx", "coo_atomic_format_load_val"} and idx == 0:
            output_shape = _fixed_coo_tile_shape()

        outputs.append(
            NvOpOutput(
                idx=idx,
                name=result.name_hint or op.result_name(idx),
                tensor=NvOpTensor(
                    shape=output_shape,
                    mem=_choose_output_mem(logical_kind, result.type, mem_hint),
                    dtype=opir_type_to_nvir_dtype(result.type),
                    row_major=(logical_kind == "coo_atomic_val_restore"),
                    swizzle=(
                        SwizzleLayout(1, 2, 3)
                        if logical_kind == "coo_atomic_val_restore"
                        else None
                    ),
                ),
                origin=output_origin,
            )
        )

    nvop = NvOp(
        name=op_name,
        inputs=inputs,
        outputs=outputs,
        impl=_make_impl(logical_kind, inputs, outputs),
        mem_type=_infer_mem_type(inputs, outputs),
        op_type=_choose_op_type(logical_kind, outputs),
        logical_kind=logical_kind,
    )
    if logical_kind in {"coo_atomic_format_load_idx", "coo_atomic_format_load_val"}:
        nvop.attrs["cp_async"] = True
        for out in outputs:
            if out.tensor.mem == "smem":
                out.attrs["cp_async"] = True
    # Mark non-cp.async ops explicitly to avoid inserting useless pipeline primitives.
    # This is particularly important for ME_TCF, where many helpers write to SMEM
    # but are implemented via normal stores (not cp.async).
    if logical_kind in {
        "coo_atomic_val_restore",
        "mco_atomic_format_load_mask",
        "mco_atomic_format_load_val",
        "mco_atomic_val_restore",
    }:
        nvop.attrs["cp_async"] = False
        for out in outputs:
            if out.tensor.mem == "smem":
                out.attrs["cp_async"] = False
    target_append(nvop)

    for idx, result in enumerate(op.results):
        state.value_sources[result] = nvop.outputs[idx]


def _lower_op(
    op: op_ir.Op,
    *,
    target_append,
    block_arg_sources: SourceBindingMap,
    state: LoweringState,
) -> None:
    if isinstance(op, op_ir.MetaOp):
        _lower_block(
            block=op.block,
            target_append=target_append,
            block_arg_sources=block_arg_sources,
            state=state,
        )
        return

    if isinstance(op, op_ir.ExternalSymbolOp):
        symbol = str(op.attributes["symbol"])
        if isinstance(op.result.type, op_ir.ArrayType):
            state.value_sources[op.result] = state.program.gmem_tensor_ops[symbol].outputs[0]
        else:
            state.value_sources[op.result] = symbol
        return

    if isinstance(op, op_ir.ConstantOp):
        _lower_constant(op, target_append=target_append, state=state)
        return

    if isinstance(op, op_ir.ForLoopOp):
        _lower_for_loop(
            op,
            target_append=target_append,
            block_arg_sources=block_arg_sources,
            state=state,
        )
        return

    if isinstance(op, op_ir.ArrayRefOp):
        _lower_array_ref(
            op,
            target_append=target_append,
            block_arg_sources=block_arg_sources,
            state=state,
        )
        return

    if isinstance(op, op_ir.NoOp):
        return

    _lower_generic_op(
        op,
        target_append=target_append,
        block_arg_sources=block_arg_sources,
        state=state,
    )


def _lower_block(
    *,
    block: op_ir.Block,
    target_append,
    block_arg_sources: SourceBindingMap,
    state: LoweringState,
) -> List[SourceType]:
    loop_result_sources: List[SourceType] = []
    for op in block.ops:
        print(f"(debug) Lowering op: {op}, block_arg_sources: {block_arg_sources}, current value_sources keys: {[str(k) for k in state.value_sources.keys()]}")
        if isinstance(op, op_ir.LoopResultOp):
            if len(op.operands) == 0:
                continue
            source = _resolve_source(
                op.operands[0].source,
                block_arg_sources=block_arg_sources,
                state=state,
            )
            loop_result_sources.append(source)
            continue

        _lower_op(
            op,
            target_append=target_append,
            block_arg_sources=block_arg_sources,
            state=state,
        )
    return loop_result_sources


def visit_and_extract_gmem_inouts(opir: op_ir.MetaOp) -> Dict[str, GmemInout]:
    gmem_inouts: Dict[str, GmemInout] = {}

    def _visit(op_node: op_ir.Op) -> None:
        if isinstance(op_node, op_ir.BlockOp):
            for inner_op in op_node.block.ops:
                _visit(inner_op)
            return

        if isinstance(op_node, op_ir.ExternalSymbolOp):
            if not isinstance(op_node.result.type, op_ir.ArrayType):
                return

            symbol = str(op_node.attributes["symbol"])
            if symbol in gmem_inouts:
                return

            dim_entries = _default_dim_entries(list(op_node.result.type.dims))
            dim_entries = _maybe_swap_blk_k_last_dim(dim_entries)

            shape = Shape(*[entry["shape"] for entry in dim_entries])
            shape_expr = ", ".join(entry["shape_arg"] for entry in dim_entries) if dim_entries else "_1{}"
            type_expr = ", ".join(entry["type_arg"] for entry in dim_entries) if dim_entries else "_1{}"
            tensor_str = (
                f"make_tensor(make_gmem_ptr(d{symbol}), make_shape({shape_expr}))"
            )
            tensor_type_str = (
                f"decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape({type_expr})))"
            )

            gmem_inouts[symbol] = GmemInout(
                shape=shape,
                name=symbol,
                dtype=opir_type_to_nvir_dtype(op_node.result.type),
                tensor_str=tensor_str,
                tensor_type_str=tensor_type_str,
            )

    _visit(opir)
    print("(debug) Extracted gmem inouts:")
    for symbol, gmem_inout in gmem_inouts.items():
        print(f"  {symbol}: {gmem_inout}, tensor_str={gmem_inout.tensor_str}, tensor_type_str={gmem_inout.tensor_type_str}")
    return gmem_inouts


def opir_type_to_nvir_shape(type: op_ir.Type) -> Shape:
    if not isinstance(type, op_ir.ArrayType):
        return Shape()

    nvir_shape = []
    for dim in type.dims:
        int_dim = _try_parse_static_int(dim)
        if int_dim is not None:
            nvir_shape.append(IntShape(int_dim))
            continue

        dim_str = str(dim)
        if dim_str == "BLK_M":
            nvir_shape.append(MnkShape("BLK_MNK", "m"))
        elif dim_str == "BLK_N":
            nvir_shape.append(MnkShape("BLK_MNK", "n"))
        elif dim_str == "BLK_K":
            nvir_shape.append(MnkShape("BLK_MNK", "k"))
        elif re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", dim_str) and dim_str.startswith("BLK_"):
            nvir_shape.append(ParamShape(dim_str))
        else:
            nvir_shape.append(VarlenShape(_sanitize_identifier(dim_str)))
    return Shape(*nvir_shape)


def opir_type_to_nvir_dtype(type: op_ir.Type) -> str:
    def opir_datatype_to_nvir_dtype(datatype: op_ir.Type) -> str:
        if isinstance(datatype, op_ir.IntType):
            return "int"
        if isinstance(datatype, op_ir.FloatType):
            return "float"
        raise ValueError(f"Unsupported datatype: {datatype}")

    if isinstance(type, op_ir.ArrayType):
        return opir_type_to_nvir_dtype(type.datatype)
    return opir_datatype_to_nvir_dtype(type)
