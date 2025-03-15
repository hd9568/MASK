from __future__ import annotations

from sparsene.op_gen.computent.computent import computent_from_rts, Computent, ArrayDef
from sparsene.op_gen.nvir.stage import Stage
from sparsene.format.format import Format, Expr, Symbol, Number
from sparsene.op_gen.computent.computent import ArrayDefCollection
from typing import Any, List, Optional, Dict, Literal, Tuple, Sequence
from sparsene.op_gen.computent.computent import (
    Schedule,
    DenseAxisIterator,
    SparseAxisIterator,
    SparseAxisSlicer,
    AtomicFormatOp,
    AtomicFormatType,
    ArrayRef,
    DataType,
)
from dataclasses import dataclass
from sparsene.op_gen.opir.op_ir import (
    Op,
    DeviceOp,
    OpBuilder,
    SymbolTable,
    ConstantOp,
    ForLoopOp,
    ArrayRefOp,
    Value,
    IntType,
    MetaOp,
    ExternalSymbolOperation,
    op_builder,
    context,
    Type,
    ArrayType,
    OpOperand,
    FloatType,
    OpResult,
)
from sparsene.op_gen.opir.ops import (
    SparseOffsetLoadOp,
    CooAtomicFormatLoadOffOp,
    CooAtomicFormatLoadIdxOp,
    CooAtomicFormatLoadValOp,
    ValSidxLoadOp,
    BValLoadOp,
    CooAtomicValRestoreOp,
    MmaOp,
    CValStoreOp,
)


def datatype_to_type(datatype: DataType) -> Type:
    match datatype:
        case DataType.FLOAT:
            return context.get_type(FloatType)
        case DataType.INT:
            return context.get_type(IntType)
        case _:
            raise ValueError(f"Unsupported datatype: {datatype}")


def get_array_type(array_def: ArrayDef) -> Type:
    dims = list(array_def.dims)
    return ArrayType(dims, datatype_to_type(array_def.datatype))


def generate_from_computent(computent: Computent) -> List[Op]:

    computent_op = op_builder.build(MetaOp)
    assert isinstance(computent_op, Op)
    symbol_table = SymbolTable(computent_op)
    for array_def in computent.array_defs:
        array_def_op = op_builder.build(
            ExternalSymbolOperation,
            array_def.name,
            get_array_type(array_def),
        )
        symbol_table.add_symbol(array_def.name, array_def_op)

    schedule = computent.schedule
    array_defs = computent.array_defs
    format = computent.format
    if isinstance(schedule, DenseAxisIterator):
        ops = generate_from_dense_axis_iterator(schedule, array_defs, format, None)
    elif isinstance(schedule, SparseAxisIterator):
        raise NotImplementedError(
            "Sparse axis iterator as root schedule is not implemented"
        )
    elif isinstance(schedule, SparseAxisSlicer):
        raise NotImplementedError(
            "Sparse axis slicer as root schedule is not implemented"
        )
    elif isinstance(schedule, AtomicFormatOp):
        raise NotImplementedError(
            "Atomic format op as root schedule is not implemented"
        )
    else:
        raise ValueError(f"Unsupported schedule type: {type(schedule)}")

    return ops


def generate_from_dense_axis_iterator(
    schedule: DenseAxisIterator,
    array_defs: ArrayDefCollection,
    format: Format,
    parent_schedule: Optional[Schedule] = None,
) -> List[Op]:
    l = op_builder.build(ConstantOp, 0, context.get_type(IntType))
    r = op_builder.build(
        ConstantOp, format.get_axis(schedule.axis).length, context.get_type(IntType)
    )

    for_loop_op = op_builder.build(
        ForLoopOp,
        induction_var=schedule.name,
        range=(l.get_result(), r.get_result()),
    )

    if isinstance(schedule.body, DenseAxisIterator):
        ops = generate_from_dense_axis_iterator(schedule.body, array_defs, format)
    elif isinstance(schedule.body, SparseAxisIterator):
        ops = generate_from_sparse_axis_iterator(
            schedule.body, array_defs, format, for_loop_op.get_induction_var(), schedule
        )
    elif isinstance(schedule.body, SparseAxisSlicer):
        ops = generate_from_sparse_axis_slicer(
            schedule.body, array_defs, format, for_loop_op.get_induction_var(), schedule
        )
    elif isinstance(schedule.body, AtomicFormatOp):
        ops = generate_from_atomic_format_op(
            schedule.body, array_defs, format, for_loop_op.get_iter_arg(0)
        )
    else:
        raise ValueError(f"Unsupported schedule type: {type(schedule.body)}")

    for op in ops:
        for_loop_op.body.add_operation(op)

    return [l, r, for_loop_op]


def generate_from_sparse_axis_iterator(
    schedule: SparseAxisIterator,
    array_defs: ArrayDefCollection,
    format: Format,
    offset_index: Value,
    parent_schedule: Optional[Schedule] = None,
) -> List[Op]:
    ops = []

    soff_array_name = schedule.offset_array
    soff = op_builder.build(
        ExternalSymbolOperation,
        soff_array_name,
        get_array_type(array_defs[soff_array_name]),
    )
    ops.append(soff)

    lr = SparseOffsetLoadOp(
        mem="G2R",
        operands=[
            soff.result,
            offset_index,
        ],
    )
    ops.append(lr)
    l = lr.results[0]
    r = lr.results[1]

    C_val_tile = op_builder.build(
        ConstantOp,
        0,
        ArrayType(
            [format.get_axis(schedule.axis).length],
            context.get_type(FloatType),
        ),
    )
    ops.append(C_val_tile)

    for_loop_op = op_builder.build(
        ForLoopOp,
        induction_var=schedule.name,
        range=(l, r),
        iter_args={
            "C_val_tile": C_val_tile.get_result(),
        },
    )
    ops.append(for_loop_op)

    if isinstance(schedule.body, AtomicFormatOp):
        body_ops = generate_from_atomic_format_op(
            schedule.body, array_defs, format, for_loop_op.get_iter_arg(0)
        )
    elif isinstance(schedule.body, DenseAxisIterator):
        raise NotImplementedError(
            "Dense axis iterator as body of sparse axis iterator is not implemented"
        )
    elif isinstance(schedule.body, SparseAxisIterator):
        raise NotImplementedError(
            "Sparse axis iterator as body of sparse axis iterator is not implemented"
        )
    elif isinstance(schedule.body, SparseAxisSlicer):
        raise NotImplementedError(
            "Sparse axis slicer as body of sparse axis iterator is not implemented"
        )
    else:
        raise ValueError(f"Unsupported schedule type: {type(schedule.body)}")

    for body_op in body_ops:
        for_loop_op.body.add_operation(body_op)

    C_val_store_op = op_builder.build(
        CValStoreOp,
        mem="R2S",
        operands=[
            for_loop_op.get_iter_arg(0),
        ],
    )
    ops.append(C_val_store_op)

    return ops


def generate_from_sparse_axis_slicer(
    schedule: SparseAxisSlicer,
    array_defs: ArrayDefCollection,
    format: Format,
    offset_index: Value,
    parent_schedule: Optional[Schedule] = None,
) -> List[Op]:
    raise NotImplementedError("Sparse axis slicer is not implemented")


def generate_from_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> List[Op]:
    match schedule.type:
        case AtomicFormatType.DENSE:
            return generate_from_dense_atomic_format_op(
                schedule, array_defs, format, C_val_tile
            )
        case AtomicFormatType.COO:
            return generate_from_coo_atomic_format_op(
                schedule, array_defs, format, C_val_tile
            )
        case AtomicFormatType.MCO:
            return generate_from_mco_atomic_format_op(
                schedule, array_defs, format, C_val_tile
            )


def build_array_def(
    array_def: ArrayDef,
    format: Format,
) -> OpResult:
    return op_builder.build(
        ExternalSymbolOperation,
        array_def.name,
        get_array_type(array_def),
    ).result


def build_array_ref_indices(
    array_ref: ArrayRef,
    array_defs: ArrayDefCollection,
    format: Format,
    op_list: List[Op],
) -> Sequence[OpResult]:
    indices: List[OpResult] = []
    for index in array_ref.indices:
        if isinstance(index, ArrayRef):
            index_result = build_array_ref(index, array_defs, format, op_list)
        else:
            index_op = op_builder.build(
                ExternalSymbolOperation,
                index,
                context.get_type(IntType),
            )
            op_list.append(index_op)
            index_result = index_op.result
        indices.append(index_result)
    return indices


def build_array_ref(
    array_ref: ArrayRef,
    array_defs: ArrayDefCollection,
    format: Format,
    op_list: List[Op],
) -> OpResult:
    array = build_array_def(array_defs[array_ref.array], format)
    indices = build_array_ref_indices(array_ref, array_defs, format, op_list)

    array_ref_op = op_builder.build(
        ArrayRefOp,
        array,
        indices,
    )
    op_list.append(array_ref_op)
    return array_ref_op.result


def build_mma_op(
    A_val_tile: Value,
    B_val_tile: Value,
    C_val_tile: Value,
) -> OpResult:
    return op_builder.build(MmaOp, "R2R", [A_val_tile, B_val_tile, C_val_tile]).result


def build_coo_atomic_format_op(
    coo_idx: Value,
    coo_val: Value,
    B_val_tile: Value,
    C_val_tile: Value,
) -> MetaOp:
    meta_op = op_builder.build(MetaOp)
    meta_op.add_operand(OpOperand(coo_idx))
    meta_op.add_operand(OpOperand(coo_val))
    meta_op.add_operand(OpOperand(B_val_tile))
    meta_op.add_operand(OpOperand(C_val_tile))

    A_val_tile = op_builder.build(
        CooAtomicValRestoreOp,
        mem="S2R",
        operands=[
            coo_val,
            coo_idx,
        ],
    ).result
    meta_op.block.add_operation(A_val_tile.defining_op)

    C_val_tile = build_mma_op(A_val_tile, B_val_tile, C_val_tile)
    meta_op.block.add_operation(C_val_tile.defining_op)

    return meta_op


def generate_from_coo_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> List[Op]:
    ops: List[Op] = []

    assert len(schedule.args) == 4
    assert isinstance(schedule.args[0], ArrayRef)
    assert isinstance(schedule.args[1], ArrayRef)
    assert isinstance(schedule.args[2], ArrayRef)
    assert isinstance(schedule.args[3], ArrayRef)
    off_array_name = schedule.args[0].array
    idx_array_name = schedule.args[1].array
    val_array_name = schedule.args[2].array
    B_val_array_name = schedule.args[3].array

    off = build_array_def(array_defs[off_array_name], format)
    coo_idx = build_array_def(array_defs[idx_array_name], format)
    coo_val = build_array_def(array_defs[val_array_name], format)
    B_val = build_array_def(array_defs[B_val_array_name], format)
    ops.append(off.defining_op)
    ops.append(coo_idx.defining_op)
    ops.append(coo_val.defining_op)
    ops.append(B_val.defining_op)

    lr = op_builder.build(
        CooAtomicFormatLoadOffOp,
        mem="G2R",
        operands=[
            off,
            *build_array_ref_indices(schedule.args[0], array_defs, format, ops),
        ],
    )
    ops.append(lr)

    coo_idx = op_builder.build(
        CooAtomicFormatLoadIdxOp,
        mem="G2R",
        operands=[
            coo_idx,
            *build_array_ref_indices(schedule.args[1], array_defs, format, ops),
            *lr.results,
        ],
    ).result
    ops.append(coo_idx.defining_op)

    coo_val = op_builder.build(
        CooAtomicFormatLoadValOp,
        mem="G2S",
        operands=[
            coo_val,
            *build_array_ref_indices(schedule.args[2], array_defs, format, ops),
            *lr.results,
        ],
    ).result
    ops.append(coo_val.defining_op)

    B_val_tile = op_builder.build(
        BValLoadOp,
        mem="G2R",
        operands=[
            B_val,
            *build_array_ref_indices(schedule.args[3], array_defs, format, ops),
        ],
    ).result
    ops.append(B_val_tile.defining_op)

    coo_atomic_format_op = build_coo_atomic_format_op(
        coo_idx,
        coo_val,
        B_val_tile,
        C_val_tile,
    )
    ops.append(coo_atomic_format_op)

    return ops


def generate_from_mco_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> List[Op]:
    raise NotImplementedError("MCO atomic format op is not implemented")


def generate_from_dense_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> List[Op]:
    raise NotImplementedError("Dense atomic format op is not implemented")
