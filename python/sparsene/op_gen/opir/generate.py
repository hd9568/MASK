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
    Type,
    ArrayType,
    OpOperand,
    FloatType,
    OpResult,
    LoopResultOp,
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
            return FloatType()
        case DataType.INT:
            return IntType()
        case _:
            raise ValueError(f"Unsupported datatype: {datatype}")


def get_array_type(array_def: ArrayDef) -> Type:
    dims = list(array_def.dims)
    return ArrayType(dims, datatype_to_type(array_def.datatype))


def generate_from_computent(computent: Computent) -> MetaOp:

    computent_op = op_builder.build(MetaOp)
    with op_builder.op_scope(computent_op):
        schedule = computent.schedule
        array_defs = computent.array_defs
        format = computent.format
        if isinstance(schedule, DenseAxisIterator):
            generate_from_dense_axis_iterator(schedule, array_defs, format)
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
    return computent_op


def generate_from_dense_axis_iterator(
    schedule: DenseAxisIterator,
    array_defs: ArrayDefCollection,
    format: Format,
    parent_schedule: Optional[Schedule] = None,
) -> None:
    l = op_builder.build(ConstantOp, 0, IntType())
    r = op_builder.build(ConstantOp, format.get_axis(schedule.axis).length, IntType())

    for_loop_op = op_builder.build(
        ForLoopOp,
        induction_var=schedule.induction_var,
        range=(l.get_result(), r.get_result()),
    )

    with op_builder.op_scope(for_loop_op):
        op_builder.add_symbol(schedule.induction_var, for_loop_op.get_induction_var())
        if isinstance(schedule.body, DenseAxisIterator):
            raise NotImplementedError(
                "Dense axis iterator as body of dense axis iterator is not implemented"
            )
        elif isinstance(schedule.body, SparseAxisIterator):
            generate_from_sparse_axis_iterator(
                schedule.body,
                array_defs,
                format,
                for_loop_op.get_induction_var(),
                schedule,
            )
        elif isinstance(schedule.body, SparseAxisSlicer):
            raise NotImplementedError(
                "Sparse axis slicer as body of dense axis iterator is not implemented"
            )
        elif isinstance(schedule.body, AtomicFormatOp):
            raise NotImplementedError(
                "Atomic format op as body of dense axis iterator is not implemented"
            )
        else:
            raise ValueError(f"Unsupported schedule type: {type(schedule.body)}")


def generate_from_sparse_axis_iterator(
    schedule: SparseAxisIterator,
    array_defs: ArrayDefCollection,
    format: Format,
    offset_index: Value,
    parent_schedule: Optional[Schedule] = None,
) -> None:
    soff = op_builder.lookup_symbol(
        schedule.offset_array, get_array_type(array_defs[schedule.offset_array])
    )

    lr = op_builder.build(SparseOffsetLoadOp, "G2R", [soff, offset_index])
    l = lr.results[0]
    r = lr.results[1]

    C_val_tile = op_builder.build(
        ConstantOp,
        0,
        ArrayType(
            [format.get_axis(schedule.axis).length],
            FloatType(),
        ),
    )

    for_loop_op = op_builder.build(
        ForLoopOp,
        induction_var=schedule.induction_var,
        range=(l, r),
        iter_args={
            "C_val_tile": C_val_tile.get_result(),
        },
    )

    with op_builder.op_scope(for_loop_op):
        op_builder.add_symbol(schedule.induction_var, for_loop_op.get_induction_var())
        if isinstance(schedule.body, AtomicFormatOp):
            op_result = generate_from_atomic_format_op(
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
        op_builder.build(LoopResultOp, op_result.defining_op)

    op_builder.build(
        CValStoreOp,
        mem="R2S",
        operands=[for_loop_op.result],
    )


def generate_from_sparse_axis_slicer(
    schedule: SparseAxisSlicer,
    array_defs: ArrayDefCollection,
    format: Format,
    offset_index: Value,
    parent_schedule: Optional[Schedule] = None,
) -> None:
    raise NotImplementedError("Sparse axis slicer is not implemented")


def generate_from_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:
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
) -> Sequence[Value]:
    indices: List[Value] = []
    for index in array_ref.indices:
        if isinstance(index, ArrayRef):
            index_result = build_array_ref(index, array_defs, format)
        else:
            index_result = op_builder.lookup_symbol(
                index,
                IntType(),
            )
        indices.append(index_result)
    return indices


def build_array_ref(
    array_ref: ArrayRef,
    array_defs: ArrayDefCollection,
    format: Format,
) -> OpResult:
    array = build_array_def(array_defs[array_ref.array], format)
    indices = build_array_ref_indices(array_ref, array_defs, format)

    array_ref_op = op_builder.build(
        ArrayRefOp,
        array,
        indices,
    )
    return array_ref_op.result


def build_mma_op(
    A_val_tile: Value,
    B_val_tile: Value,
    C_val_tile: Value,
) -> OpResult:
    mma_op = op_builder.build(MmaOp, "R2R", [A_val_tile, B_val_tile, C_val_tile])
    return mma_op.result


def build_coo_atomic_format_op(
    coo_idx: Value,
    coo_val: Value,
    B_val_tile: Value,
    C_val_tile: Value,
) -> OpResult:
    A_val_tile = op_builder.build(
        CooAtomicValRestoreOp,
        mem="S2R",
        operands=[
            coo_val,
            coo_idx,
        ],
    ).result
    C_val_tile = build_mma_op(A_val_tile, B_val_tile, C_val_tile)

    return C_val_tile


def generate_from_coo_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:

    assert len(schedule.args) == 4
    assert isinstance(schedule.args[0], ArrayRef)
    assert isinstance(schedule.args[1], ArrayRef)
    assert isinstance(schedule.args[2], ArrayRef)
    assert isinstance(schedule.args[3], ArrayRef)

    off_array = op_builder.lookup_symbol(
        schedule.args[0].array, get_array_type(array_defs[schedule.args[0].array])
    )
    idx_array = op_builder.lookup_symbol(
        schedule.args[1].array, get_array_type(array_defs[schedule.args[1].array])
    )
    val_array = op_builder.lookup_symbol(
        schedule.args[2].array, get_array_type(array_defs[schedule.args[2].array])
    )
    B_val_array = op_builder.lookup_symbol(
        schedule.args[3].array, get_array_type(array_defs[schedule.args[3].array])
    )

    lr = op_builder.build(
        CooAtomicFormatLoadOffOp,
        mem="G2R",
        operands=[
            off_array,
            *build_array_ref_indices(schedule.args[0], array_defs, format),
        ],
    )

    coo_idx = op_builder.build(
        CooAtomicFormatLoadIdxOp,
        mem="G2R",
        operands=[
            idx_array,
            *build_array_ref_indices(schedule.args[1], array_defs, format),
            *lr.results,
        ],
    ).result

    coo_val = op_builder.build(
        CooAtomicFormatLoadValOp,
        mem="G2S",
        operands=[
            val_array,
            *build_array_ref_indices(schedule.args[2], array_defs, format),
            *lr.results,
        ],
    ).result

    B_val_tile = op_builder.build(
        BValLoadOp,
        mem="G2R",
        operands=[
            B_val_array,
            *build_array_ref_indices(schedule.args[3], array_defs, format),
        ],
    ).result

    coo_atomic_format_op_result = build_coo_atomic_format_op(
        coo_idx,
        coo_val,
        B_val_tile,
        C_val_tile,
    )

    return coo_atomic_format_op_result


def generate_from_mco_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:
    raise NotImplementedError("MCO atomic format op is not implemented")


def generate_from_dense_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:
    raise NotImplementedError("Dense atomic format op is not implemented")
