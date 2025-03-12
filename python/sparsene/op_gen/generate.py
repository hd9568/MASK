from __future__ import annotations

from sparsene.op_gen.rts_parser import computent_from_rts, Computent
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
    ForLoopOp,
    Value,
    Index,
)


def generate_from_computent(computent: Computent) -> Op:
    return generate_from_schedule(
        computent.schedule, computent.array_defs, computent.format
    )


def generate_from_schedule(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    parent_schedule: Optional[Schedule] = None,
) -> Op:
    if isinstance(schedule, DenseAxisIterator):
        return generate_from_dense_axis_iterator(
            schedule, array_defs, format, parent_schedule
        )
    elif isinstance(schedule, SparseAxisIterator):
        return generate_from_sparse_axis_iterator(
            schedule, array_defs, format, parent_schedule
        )
    elif isinstance(schedule, SparseAxisSlicer):
        return generate_from_sparse_axis_slicer(
            schedule, array_defs, format, parent_schedule
        )
    elif isinstance(schedule, AtomicFormatOp):
        return generate_from_atomic_format_op(
            schedule, array_defs, format, parent_schedule
        )
    else:
        raise ValueError(f"Unsupported schedule type: {type(schedule)}")


def generate_from_dense_axis_iterator(
    schedule: DenseAxisIterator,
    array_defs: ArrayDefCollection,
    format: Format,
    parent_schedule: Optional[Schedule] = None,
) -> Op:
    l = op_builder.build("l", ConstantOp(value=0))
    r = op_builder.build("r", ConstantOp(value=format.get_axis(schedule.axis).length))

    for_loop_op = symbol_table.add_symbol("_", ForLoopOp(
        range=(l, r),
        args=[Value(name=schedule.name, type=Index())],
        body=NoOp(),
    ))

    loop = (
        Value(name=schedule.name, type=Index()),
        Value(name=0, type=Index()),
        Value(name=format.get_axis(schedule.axis).length, type=Index()),
    )
    for_loop_op = ForLoopOp(loop, {}, NoOp())
    symbol_table[schedule.name] = for_loop_op
    for_loop_op.body = generate_from_schedule(
        schedule.body, array_defs, format, symbol_table, schedule
    )
    return for_loop_op


def generate_from_sparse_axis_iterator(
    schedule: SparseAxisIterator,
    array_defs: ArrayDefCollection,
    format: Format,
    parent_schedule: Optional[Schedule] = None,
) -> Op:
    lr = SparseOffsetLoadOp(
        mem="G2R",
        args=[
            ArrayDef("val_soff"),
            symbol_table["v5"].,
        ],
    )
    loop = (
        VarIndex(var_name=schedule.name),
        ConstantIndex(0),
        ConstantIndex(format.get_axis(schedule.axis).length),
    )
    args = {}
    body = generate_from_schedule(
        schedule.body, array_defs, format, parent_schedule
    )
    return ForLoopOp(loop, args, body)


def generate_from_sparse_axis_slicer(
    schedule: SparseAxisSlicer,
    array_defs: ArrayDefCollection,
    format: Format,
    parent_schedule: Optional[Schedule] = None,
) -> Op:
    pass


def generate_from_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    parent_schedule: Optional[Schedule] = None,
) -> Op:
    match schedule.type:
        case AtomicFormatType.DENSE:
            return generate_from_dense_atomic_format_op(
                schedule, array_defs, format, parent_schedule
            )
        case AtomicFormatType.COO:
            return generate_from_coo_atomic_format_op(
                schedule, array_defs, format, parent_schedule
            )
        case AtomicFormatType.MCO:
            return generate_from_mco_atomic_format_op(
                schedule, array_defs, format, parent_schedule
            )
    pass


def generate_from_dense_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    parent_schedule: Optional[Schedule] = None,
) -> Op:
    pass


def generate_from_coo_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    parent_schedule: Optional[Schedule] = None,
) -> Op:
    pass


def generate_from_mco_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    parent_schedule: Optional[Schedule] = None,
) -> Op:
    pass
