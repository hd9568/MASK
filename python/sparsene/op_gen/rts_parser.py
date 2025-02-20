from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, List, Optional, Dict, Tuple
from sparsene.model.transformation import (
    Transformation,
    TransformationSequence,
    SplitTransformation,
    SwapTransformation,
    SparsifyTransformation,
    SpawnTransformation,
    CooizeTransformation,
    MaskizeTransformation,
)
from sparsene.model.format import (
    Direction,
    Format,
    Axis,
    df_axis,
    sv_axis,
    coo_atomic_format,
    mask_atomic_format,
)
from sparsene.formats.dense import get_dense_format
from sympy import Expr, Number, Symbol, sympify


class AtomicFormatType(Enum):
    DENSE = 0
    COO = 1
    MASK = 2


@dataclass(kw_only=True)
class Slice:
    start: Variable
    end: Variable


@dataclass(kw_only=True)
class ArrayRef:
    array: str
    indices: List[Variable | Slice]


Variable = ArrayRef | Expr


@dataclass(kw_only=True)
class AtomicFormatOp:
    axes: Tuple[str, str]
    type: AtomicFormatType
    args: List[ArrayRef]


@dataclass(kw_only=True)
class DimIterator:
    axis: str
    body: Schedule
    is_sparse: bool


class ArrayType(Enum):
    VAL = 0
    INDEX = 1
    MASK = 2
    OFFSET = 3


@dataclass(kw_only=True)
class ArrayDef:
    name: str
    dims: List[str]
    type: ArrayType

    def indexed_at(self, axis: str) -> Optional[int]:
        for i, dim in enumerate(self.dims):
            if dim == axis:
                return i
        return None


Schedule = DimIterator | AtomicFormatOp


@dataclass(kw_only=True)
class ArrayDefCollection:
    array_defs: Dict[str, ArrayDef]

    def __init__(self, **kwargs: ArrayDef):
        self.array_defs = deepcopy(kwargs)

    def __iter__(self) -> Iterator[ArrayDef]:
        return iter(self.array_defs.values())

    def __getitem__(self, key: str) -> ArrayDef:
        return self.array_defs[key]

    def items(self) -> Iterator[Tuple[str, ArrayDef]]:
        return iter(self.array_defs.items())


@dataclass(kw_only=True)
class Computent:
    name: str
    schedule: Schedule
    array_defs: ArrayDefCollection


DENSE_SCHEDULE: Schedule = AtomicFormatOp(
    axes=("X", "Y"),
    type=AtomicFormatType.DENSE,
    args=[
        ArrayRef(array="val", indices=[]),
    ],
)


DENSE_ARRAY_DEFS = ArrayDefCollection(
    val=ArrayDef(
        name="val",
        dims=["X", "Y"],
        type=ArrayType.VAL,
    )
)


def apply_split_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    transform: SplitTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    schedule, array_defs, format = (
        deepcopy(schedule),
        deepcopy(array_defs),
        deepcopy(format),
    )

    axis_idx = transform.axes[0]

    length_before = format.axes[axis_idx].length
    new_length_outer = length_before / transform.split_size  # type: ignore
    new_length_inner = transform.split_size

    # 1. Handle format
    axis_to_split = format.axes[axis_idx]
    new_axis_outer = Axis(
        name=axis_to_split.name + "_o",
        direction=axis_to_split.direction,
        length=new_length_outer,
        is_sparse=axis_to_split.is_sparse,
        is_varlen=axis_to_split.is_varlen,
    )
    new_axis_inner = Axis(
        name=axis_to_split.name + "_i",
        direction=axis_to_split.direction,
        length=new_length_inner,
        is_sparse=axis_to_split.is_sparse,
        is_varlen=axis_to_split.is_varlen,
    )
    format.axes[axis_idx] = new_axis_outer
    format.axes.insert(axis_idx + 1, new_axis_inner)

    # 2. Handle array_defs
    for array_def in array_defs:
        idx_dim = array_def.indexed_at(axis_to_split.name)
        if idx_dim is None:
            continue

        array_def.dims[idx_dim] = new_axis_outer.name
        array_def.dims.insert(idx_dim + 1, new_axis_inner.name)

    # 3. Handle schedule
    # TODO: Implement

    return schedule, array_defs, format


def apply_swap_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    transform: SwapTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    schedule, array_defs, format = (
        deepcopy(schedule),
        deepcopy(array_defs),
        deepcopy(format),
    )

    # 1. Handle format
    format.axes[transform.axes[0]], format.axes[transform.axes[1]] = (
        format.axes[transform.axes[1]],
        format.axes[transform.axes[0]],
    )

    # 2. Handle array_defs
    for array_def in array_defs:
        idx_dim_0 = array_def.indexed_at(format.axes[transform.axes[0]].name)
        idx_dim_1 = array_def.indexed_at(format.axes[transform.axes[1]].name)
        if idx_dim_0 is None or idx_dim_1 is None:
            continue

        array_def.dims[idx_dim_0], array_def.dims[idx_dim_1] = (
            array_def.dims[idx_dim_1],
            array_def.dims[idx_dim_0],
        )

    # 3. Handle schedule
    # TODO: Implement

    return schedule, array_defs, format


def apply_sparsify_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    transform: SparsifyTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    schedule, array_defs, format = (
        deepcopy(schedule),
        deepcopy(array_defs),
        deepcopy(format),
    )
    # TODO: Implement
    return schedule, array_defs, format


def apply_spawn_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    transform: SpawnTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    schedule, array_defs, format = (
        deepcopy(schedule),
        deepcopy(array_defs),
        deepcopy(format),
    )

    # 1. Handle format
    parent = format
    while isinstance(parent.child, Format):
        parent = parent.child
    assert parent.child is None

    parent.child = Format(
        axes=[
            df_axis(
                name=parent.axes[-2].name,
                direction=parent.axes[-2].direction,
                length=parent.axes[-2].length,
            ),
            df_axis(
                name=parent.axes[-1].name,
                direction=parent.axes[-1].direction,
                length=parent.axes[-1].length,
            ),
        ]
    )

    # 2. Handle array_defs
    # Nothing to do

    # 3. Handle schedule
    # Nothing to do

    return schedule, array_defs, format


def apply_cooize_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    transform: CooizeTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    schedule, array_defs, format = (
        deepcopy(schedule),
        deepcopy(array_defs),
        deepcopy(format),
    )
    # 1. Handle format
    parent = format
    while isinstance(parent.child, Format):
        parent = parent.child
    assert parent.child is None

    parent.child = coo_atomic_format(
        x_name=parent.axes[-2].name,
        y_name=parent.axes[-1].name,
        col_major=parent.axes[-2].direction == Direction.COL,
    )

    # 2. Handle array_defs
    # TODO: Implement

    # 3. Handle schedule
    # TODO: Implement

    return schedule, array_defs, format


def apply_maskize_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    transform: MaskizeTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    schedule, array_defs, format = (
        deepcopy(schedule),
        deepcopy(array_defs),
        deepcopy(format),
    )
    # 1. Handle format
    parent = format
    while isinstance(parent.child, Format):
        parent = parent.child
    assert parent.child is None

    parent.child = coo_atomic_format(
        x_name=parent.axes[-2].name,
        y_name=parent.axes[-1].name,
        col_major=parent.axes[-2].direction == Direction.COL,
    )

    # 2. Handle array_defs
    # TODO: Implement

    # 3. Handle schedule
    # TODO: Implement

    return schedule, array_defs, format


def generate_from_rts(name: str, rts: TransformationSequence) -> Computent:
    format = get_dense_format()
    schedule = deepcopy(DENSE_SCHEDULE)
    array_defs = deepcopy(DENSE_ARRAY_DEFS)

    for transform in rts.sequence:
        if isinstance(transform, SplitTransformation):
            schedule, array_defs, format = apply_split_transformation(
                schedule, array_defs, format, transform
            )
        elif isinstance(transform, SwapTransformation):
            schedule, array_defs, format = apply_swap_transformation(
                schedule, array_defs, format, transform
            )
        elif isinstance(transform, SparsifyTransformation):
            schedule, array_defs, format = apply_sparsify_transformation(
                schedule, array_defs, format, transform
            )
        elif isinstance(transform, SpawnTransformation):
            schedule, array_defs, format = apply_spawn_transformation(
                schedule, array_defs, format, transform
            )
        elif isinstance(transform, CooizeTransformation):
            schedule, array_defs, format = apply_cooize_transformation(
                schedule, array_defs, format, transform
            )
        elif isinstance(transform, MaskizeTransformation):
            schedule, array_defs, format = apply_maskize_transformation(
                schedule, array_defs, format, transform
            )

    return Computent(
        name=name,
        schedule=schedule,
        array_defs=array_defs,
    )
