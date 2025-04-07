from sparsene.op_gen.nvir.nvop import (
    NvOp,
    NvOpInput,
    NvOpOutput,
    NvOpTensor,
    LayoutHint,
    IoMemType,
    OpMemType,
    NvOpType,
    NvOpImpl,
    GmemInout,
    NvOpSequence,
    NvOpProgram,
    MetaNvOp,
    ForLoopNvOp,
    GmemInout,
    GmemTensorNvOp,
)
import sparsene.op_gen.opir.op_ir as op_ir
from typing import Dict
from sparsene.op_gen.nvir.nvop import Shape, IntShape, VarlenShape


def generate_nvir(opir: op_ir.MetaOp) -> NvOpProgram:
    gmem_inouts = visit_and_extract_gmem_inouts(opir)
    meta_op = visit_and_construct_nvops(opir, gmem_inouts)
    nvop_program = NvOpProgram(
        name=opir.name,
        meta_op=meta_op,
        gmem_inouts=gmem_inouts,
    )
    return nvop_program


def visit_and_construct_nvops(
    opir: op_ir.MetaOp, gmem_inouts: Dict[str, GmemInout]
) -> MetaNvOp:
    op_to_nvop: Dict[op_ir.Op, NvOp] = {}
    block_to_nvop: Dict[op_ir.Block, NvOp] = {}

    def opir_value_to_nvir_input(idx: int, name: str, value: op_ir.Value) -> NvOpInput:
        if isinstance(value, op_ir.BlockArgument):
            return NvOpInput(
                idx=idx,
                name=name,
                tensor=NvOpTensor(
                    shape=opir_type_to_nvir_shape(value.type),
                    mem="xmem",
                    dtype=opir_type_to_nvir_dtype(value.type),
                    source=(block_to_nvop[value.owner], value.arg_idx),
                ),
            )
        elif isinstance(value, op_ir.OpResult):
            return NvOpInput(
                idx=idx,
                name=name,
                tensor=NvOpTensor(
                    shape=opir_type_to_nvir_shape(value.type),
                    mem="xmem",
                    dtype=opir_type_to_nvir_dtype(value.type),
                    source=(op_to_nvop[value.defining_op], value.result_idx_in_owner),
                ),
            )
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")

    def _visit(op: op_ir.Op) -> NvOp:
        if isinstance(op, op_ir.MetaOp):
            seq = NvOpSequence(name=op.name, ops=[])
            for op_inside in op.block.ops:
                seq.ops.append(_visit(op_inside))
            result_op = MetaNvOp(op.name, seq)
            block_to_nvop[op.block] = result_op
        elif isinstance(op, op_ir.ForLoopOp):
            loop_l = opir_value_to_nvir_input(0, "l", op.range[0])
            loop_r = opir_value_to_nvir_input(1, "r", op.range[1])
            iter_args = {}
            loop_result = []
            seq = NvOpSequence(name=op.name, ops=[])
            for op_inside in op.body.ops:
                seq.ops.append(_visit(op_inside))
            result_op = ForLoopNvOp(
                name=op.name,
                loop_l=loop_l,
                loop_r=loop_r,
                body=seq,
                iter_args=iter_args,
                loop_result=loop_result,
            )

            for i in range(op.num_iter_args):
                loop_result.append(
                    NvOpOutput(
                        idx=i,
                        name=op.get_iter_arg(i).name_hint or "",
                        tensor=NvOpTensor(
                            shape=opir_type_to_nvir_shape(op.results[i].type),
                            dtype=opir_type_to_nvir_dtype(op.results[i].type),
                            mem="xmem",
                            source=result_op.outputs[i],
                        ),
                    )
                )
            block_to_nvop[op.body] = result_op
        elif isinstance(op, op_ir.ExternalSymbolOp):
            result_op = GmemTensorNvOp(
                gmem_inouts[op.attributes["symbol"]],
            )
        else:
            inputs = []
            outputs = []
            result_op = NvOp(
                name=op.name,
                inputs=inputs,
                outputs=outputs,
                impl=NvOpImpl(code_template=""),
                mem_type=("x", "x"),
                parameters={},
            )
            for i in range(op.num_operands):
                inputs.append(
                    opir_value_to_nvir_input(
                        i, op.operand_name(i), op.operands[i].source
                    )
                )
            for i in range(op.num_results):
                outputs.append(
                    NvOpOutput(
                        i,
                        op.result_name(i),
                        NvOpTensor(
                            shape=opir_type_to_nvir_shape(op.results[i].type),
                            dtype=opir_type_to_nvir_dtype(op.results[i].type),
                            mem="xmem",
                            source=result_op.outputs[i],
                        ),
                    )
                )
        op_to_nvop[op] = result_op
        return result_op

    meta_op = _visit(opir)
    assert isinstance(meta_op, MetaNvOp)
    return meta_op


def visit_and_extract_gmem_inouts(opir: op_ir.MetaOp) -> Dict[str, GmemInout]:
    gmem_inouts = {}

    def _visit(opir: op_ir.Op) -> None:
        if isinstance(opir, op_ir.BlockOp):
            for op in opir.block.ops:
                _visit(op)
        elif isinstance(opir, op_ir.ExternalSymbolOp):
            assert isinstance(opir.result.type, op_ir.ArrayType)
            gmem_inouts[opir.attributes["symbol"]] = GmemInout(
                shape=opir_type_to_nvir_shape(opir.result.type),
                name=opir.attributes["symbol"],
                dtype=opir_type_to_nvir_dtype(opir.result.type),
            )

    _visit(opir)
    return gmem_inouts


def opir_type_to_nvir_shape(type: op_ir.Type) -> Shape:
    if isinstance(type, op_ir.ArrayType):
        nvir_shape = []
        for dim in type.dims:
            try:
                int_dim = int(dim)
                nvir_shape.append(IntShape(int_dim))
            except TypeError:
                nvir_shape.append(VarlenShape(dim))
        return Shape(*nvir_shape)
    else:
        return Shape()


def opir_type_to_nvir_dtype(type: op_ir.Type) -> str:
    def opir_datatype_to_nvir_dtype(datatype: op_ir.Type) -> str:
        assert not isinstance(datatype, op_ir.ArrayType)
        match datatype:
            case op_ir.IntType():
                return "int"
            case op_ir.FloatType():
                return "float"
            case _:
                raise ValueError(f"Unsupported datatype: {datatype}")

    if isinstance(type, op_ir.ArrayType):
        return opir_type_to_nvir_dtype(type.datatype)
    else:
        return opir_datatype_to_nvir_dtype(type)
