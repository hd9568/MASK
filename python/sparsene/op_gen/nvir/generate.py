from sparsene.op_gen.nvir.nvop import (
    NvOp,
    NvOpInput,
    NvOpOutput,
    LayoutHint,
    IoMemType,
    OpMemType,
    NvOpType,
    NvOpImpl,
    GmemInout,
    NvOpSequence,
)
import sparsene.op_gen.opir.op_ir as op_ir
from typing import Dict, Sequence
from sparsene.op_gen.nvir.nvop import Shape, IntShape, VarlenShape


def generate_nvir(opir: op_ir.MetaOp) -> NvOpSequence:
    gmem_inouts = visit_and_extract_gmem_inouts(opir)
    nvop_sequence = NvOpSequence(
        name=opir.name,
        stages=[],
        shifts=[],
        prologue=[],
        epilogue=[],
        gmem_inouts=gmem_inouts,
    )
    return nvop_sequence

def visit_and_construct_nvops(opir: op_ir.MetaOp) -> NvOpSequence:


def visit_and_extract_gmem_inouts(opir: op_ir.MetaOp) -> Dict[str, GmemInout]:
    gmem_inouts = {}

    def _visit(opir: op_ir.Op) -> None:
        if isinstance(opir, op_ir.BlockOp):
            for op in opir.block.ops:
                _visit(op)
        elif isinstance(opir, op_ir.ExternalSymbolOp):
            assert isinstance(opir.result.type, op_ir.ArrayType)
            gmem_inouts[opir.attributes["symbol"]] = GmemInout(
                shape=opir_dims_to_nvir_shape(opir.result.type.dims),
                name=opir.attributes["symbol"],
                dtype=opir_datatype_to_nvir_dtype(opir.result.type.datatype),
            )

    _visit(opir)
    return gmem_inouts


def opir_dims_to_nvir_shape(dims: Sequence[op_ir.Expr]) -> Shape:
    nvir_shape = []
    for dim in dims:
        nvir_shape.append(str(dim))
    return Shape(*nvir_shape)


def opir_datatype_to_nvir_dtype(datatype: op_ir.Type) -> str:
    match datatype:
        case op_ir.IntType():
            return "int"
        case op_ir.FloatType():
            return "float"
        case _:
            raise ValueError(f"Unsupported datatype: {datatype}")
