from __future__ import annotations

from sparsene.op_gen.rts_parser import computent_from_rts, Computent, ArrayDef
from sparsene.model.stage import Stage
from sparsene.model.format import Format, Expr, Symbol, Number
from sparsene.op_gen.rts_parser import ArrayDefCollection
from typing import Any, List, Optional, Dict, Literal, Tuple
from sparsene.op_gen.rts_parser import (
    Schedule,
    DenseAxisIterator,
    SparseAxisIterator,
    SparseAxisSlicer,
    AtomicFormatOp,
    AtomicFormatType,
)
from dataclasses import dataclass
from sparsene.op_gen.op_ir import (
    Op,
    DeviceOp,
    OpBuilder,
    SymbolTable,
    ConstantOp,
    ForLoopOp,
    Value,
    IndexType,
    MetaOp,
    ExternalSymbolOperation,
    op_builder,
    context,
    Type,
    ArrayType,
)
from sparsene.op_gen.ops import (
    SparseOffsetLoadOp,
    GetFromRangeOp,
    CooAtomicFormatLoadOffOp,
    CooAtomicFormatLoadIdxOp,
    CooAtomicFormatLoadValOp,
    ValSidxLoadOp,
    BValLoadOp,
    CooAtomicValRestoreOp,
    MmaOp,
)


def get_array_type(array_def: ArrayDef, format: Format) -> Type:
    dims = []
    for dim in array_def.dims:
        length = format.get_axis(dim).length
        dims.append(length)
    return ArrayType(dims)


def get_array_name(axis_name: str, format: Format) -> str:
    return format.get_axis(axis_name).name


def generate_from_computent(computent: Computent) -> List[Op]:

    computent_op = op_builder.build(MetaOp)
    assert isinstance(computent_op, Op)
    symbol_table = SymbolTable(computent_op)
    for array_def in computent.array_defs:
        array_def_op = op_builder.build(
            ExternalSymbolOperation,
            array_def.name,
            get_array_type(array_def, computent.format),
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
    l = op_builder.build(ConstantOp, 0)
    r = op_builder.build(ConstantOp, format.get_axis(schedule.axis).length)

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
        ops = generate_from_atomic_format_op(schedule.body, array_defs, format)
    else:
        raise ValueError(f"Unsupported schedule type: {type(schedule.body)}")

    for op in ops:
        for_loop_op.body.add_operation(op)

    return [for_loop_op]


def generate_from_sparse_axis_iterator(
    schedule: SparseAxisIterator,
    array_defs: ArrayDefCollection,
    format: Format,
    offset_index: Value,
    parent_schedule: Optional[Schedule] = None,
) -> List[Op]:
    ops = []

    soff_array_name = get_array_name(schedule.axis, format)
    soff = op_builder.build(
        ExternalSymbolOperation,
        soff_array_name,
        get_array_type(array_defs[soff_array_name], format),
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
        ArrayType([format.get_axis(schedule.axis).length]),
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
) -> List[Op]:
    match schedule.type:
        case AtomicFormatType.DENSE:
            return generate_from_dense_atomic_format_op(schedule, array_defs, format)
        case AtomicFormatType.COO:
            return generate_from_coo_atomic_format_op(schedule, array_defs, format)
        case AtomicFormatType.MCO:
            return generate_from_mco_atomic_format_op(schedule, array_defs, format)


def generate_from_dense_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    parent_schedule: Optional[Schedule] = None,
) -> List[Op]:
    raise NotImplementedError("Dense atomic format op is not implemented")


def generate_from_coo_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    parent_schedule: Optional[Schedule] = None,
) -> List[Op]:
    ops = []

    lr = op_builder.build(
        CooAtomicFormatLoadOffOp,
        mem="G2R",
        operands=[
            soff.result,
            offset_index,
        ],
    )
    ops.append(lr)
    raise NotImplementedError("COO atomic format op is not implemented")


def generate_from_mco_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    parent_schedule: Optional[Schedule] = None,
) -> List[Op]:
    raise NotImplementedError("MCO atomic format op is not implemented")
